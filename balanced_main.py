from datasets import load_dataset
import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torchvision import transforms as T
from tqdm import tqdm
from typing import Union
import os
import numpy as np
import math
import pandas as pd
from PIL import Image
from copy import deepcopy
from geoformer import Geoformer

class BaseMultiTaskSampler(metaclass=abc.ABCMeta):
    def __init__(self, task_dict: dict, rng: Union[int, np.random.RandomState, None]):
        self.task_dict = task_dict
        if isinstance(rng, int) or rng is None:
            rng = np.random.RandomState(rng)
        self.rng = rng
        self.task_names = list(task_dict.keys())

    def pop(self):
        raise NotImplementedError()

    def iter(self):
        yield self.pop()


class SpecifiedProbMultiTaskSampler(BaseMultiTaskSampler):
    def __init__(
            self,
            task_dict: dict,
            rng: Union[int, np.random.RandomState],
            task_to_unweighted_probs: dict,
    ):
        super().__init__(task_dict=task_dict, rng=rng)
        assert task_dict.keys() == task_to_unweighted_probs.keys()
        self.task_to_unweighted_probs = task_to_unweighted_probs
        self.task_names = list(task_to_unweighted_probs.keys())
        self.unweighted_probs_arr = np.array([task_to_unweighted_probs[k] for k in self.task_names])
        self.task_p = self.unweighted_probs_arr / self.unweighted_probs_arr.sum()

    def pop(self):
        task_name = self.rng.choice(self.task_names, p=self.task_p)
        return task_name, self.task_dict[task_name]
    
class Trainer:
    def __init__(self, model, optimizer, epochs, task_probs):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.epochs = epochs
        self.task_probs = task_probs
        
    def train(self, train_loader, val_loader, epoch_start, epoch_end, steps_per_epoch):
        sampler = SpecifiedProbMultiTaskSampler(train_dataloader, 0, self.task_probs)
        for epoch in range(epoch_start, epoch_end):
            self.train_one_epoch(sampler, steps_per_epoch)
            
    def train_one_step(self, sampler, steps_per_epoch):
        self.model.train()
        for i in range(steps_per_epoch):
            
            
        
        
    
class Trainer:
    def __init__(self, model, train_loader, val_loader, device, epochs=100, log_dir='logs', checkpoint_dir='checkpoints'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.loss_fn1 = nn.MSELoss()
        self.loss_fn2 = nn.CrossEntropyLoss()
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4, fused=True)
        self.gradient_accumulation_steps = 4
        
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        with torch.no_grad():
            samp = torch.randn(1, 3, 224, 224)
            output = self.model(samp)
        self.task_names = output.keys()
        self.targets = ['latitude', 'longitude']
        self.auxiliary = list(set(self.task_names) - set(self.targets))
    
    def calculate_losses(self, outputs, targets, weights=None):
        losses = []
        for idx, (key, value) in enumerate(outputs.items()):
            if key in ['longitude', 'latitude', 'dist_sea']:
                task_loss = F.mse_loss(value, targets[key])
            else:
                task_loss = F.cross_entropy(value, targets[key])
            losses.append(task_loss * (weights[idx] if weights is not None else 1.0))
        return torch.stack(losses)
    
    def calculate_haverstine_distance(self, outputs, targets, avg=False):
        predicted_longitude = outputs["longitude"]
        predicted_latitude = outputs["latitude"]
        target_longitude = targets["longitude"]
        target_latitude = targets["latitude"]
        
        lon1, lat1, lon2, lat2 = map(torch.deg2rad, [predicted_longitude, predicted_latitude, target_longitude, target_latitude])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
        c = 2 * torch.asin(torch.sqrt(a))
        
        d = 6371 * c
        return torch.mean(d) if avg else d
        
        
    def save_checkpoint(self, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, f'{self.checkpoint_dir}/checkpoint_epoch_{epoch}.pth')
        
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']
    
    def compute_inner_prod(self, grad1, grad2, metric_incr=True):
        if len(grad1) <= len(grad2):
            numel = len(grad1)
        else:
            numel = len(grad2)
            
        prod = torch.dot(grad1[0], grad2[0])
        for i in range(1, numel):
            prod += torch.dot(grad1[i], grad2[i])
            
        if metric_incr:
            prod *= -1
            
        return prod
    
    def compute_cosine_similarity(self, grad1, grad2):
        prod_sum = 0
        norm1 = 0
        norm2 = 0
        for g1, g2 in zip(grad1, grad2):
            prod_sum += torch.sum(g1 * g2)
            norm1 += torch.sum(g1 ** 2)
            norm2 += torch.sum(g2 ** 2)
        
        norm1 = torch.sqrt(norm1)
        norm2 = torch.sqrt(norm2)
        
        # Avoid division by zero
        denominator = norm1 * norm2
        if denominator < 1e-6:
            return 0.0
        
        return prod_sum / denominator
    
    def train(self):
        init_model_params = deepcopy(self.model.state_dict())
        
        performance_dict = {}
        task_weights = {name: 0 for name in self.task_names}
        for name in self.targets:
            task_weights[name] = 1 / len(self.targets)
        
        for auxiliary in self.auxiliary:
            self.model.load_state_dict(init_model_params)
            new_task_weights = deepcopy(task_weights)
            new_task_weights[auxiliary] = 1
            
        
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = []
            total_distance = []
            zipped = zip(self.train_loader, self.val_loader)
            train_progress_bar = tqdm(zipped, total=min(len(self.train_loader), len(self.val_loader)), desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for i, (train_batch, val_batch) in enumerate(train_progress_bar):
                train_x, train_y = train_batch
                train_x = train_x.to(self.device)
                train_y = {key: value.to(self.device) for key, value in train_y.items()}
                
                outputs = self.model(train_x)
                
                task_gradients = []
                task_losses = self.calculate_losses(outputs, train_y)
                for task_id in range(num_tasks):
                    self.optimizer.zero_grad()
                    if task_id < num_tasks - 1:
                        task_losses[task_id].backward(retain_graph=True)
                    else:
                        task_losses[task_id].backward()
                    task_gradients.append([torch.reshape(p.grad.clone(), (-1, )) for p in list(self.model.backbone.parameters()) + list(self.model.fpn.parameters()) if p.requires_grad])
                       
                val_x, val_y = val_batch
                val_x = val_x.to(self.device)
                val_y = {key: value.to(self.device) for key, value in val_y.items()}
                self.optimizer.zero_grad()
                val_outputs = self.model(val_x)
                main_metric = self.calculate_haverstine_distance(val_outputs, val_y, avg=True)
                main_metric.backward()
                main_gradient = [torch.reshape(p.grad.clone(), (-1, )) for p in list(self.model.backbone.parameters()) + list(self.model.fpn.parameters()) if p.requires_grad]
                
                cosine_similarities = torch.zeros(num_tasks, device=self.device)
                for j in range(num_tasks):
                    cosine_similarities[j] = self.compute_cosine_similarity(task_gradients[j], main_gradient)
                        
                weights = F.softmax(cosine_similarities, dim=0)
                
                self.optimizer.zero_grad()
                outputs = self.model(train_x)
                avg_distance = self.calculate_haverstine_distance(outputs, train_y, avg=True)
                loss = self.calculate_losses(outputs, train_y, weights).sum()
                loss.backward()
                self.optimizer.step()
                
                total_loss.append(loss.item())
                total_distance.append(avg_distance.item())
                train_progress_bar.set_postfix({'dist': avg_distance.item(), 'cs': [f'{w:.2f}' for w in weights.tolist()]})
            
            avg_train_loss = np.median(total_loss)
            avg_train_dist = np.median(total_distance)
                
            print(f"Epoch [{epoch+1}/{self.epochs}] | Train Loss: {avg_train_loss:.4f} | Train Haversine Distance: {avg_train_dist:.4f}")
            
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1, avg_train_loss)
    
        
class ImageDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None, id_column='id'):
        self.img_dir = img_dir
        self.csv_data = pd.read_csv(csv_file)
        self.transform = transform or transforms.ToTensor()
        self.id_column = id_column
        self.ids = self.csv_data[self.id_column].tolist()

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        row = self.csv_data.iloc[idx]
        
        img_id = row[self.id_column]
        
        img_name = os.path.join(self.img_dir, f"{img_id}.jpg")
        
        try:
            image = Image.open(img_name).convert('RGB')
        except IOError:
            print(f"Error loading image {img_name}")
            # You might want to handle this error more gracefully
            return None
        
        if self.transform:
            image = self.transform(image)
        
        # Convert metadata to tensors
        metadata_dict = {
            'longitude': torch.tensor(row['longitude'], dtype=torch.float32),
            'latitude': torch.tensor(row['latitude'], dtype=torch.float32),
            'lat_bin': F.one_hot(torch.tensor(row['lat_bin']), num_classes=100).float(),
            'lon_bin': F.one_hot(torch.tensor(row['lon_bin']), num_classes=100).float(),
            'climate': F.one_hot(torch.tensor(row['climate']), num_classes=31).float(),
            'soil': F.one_hot(torch.tensor(row['soil']), num_classes=15).float(),
            'dist_sea': torch.tensor(row['dist_sea'], dtype=torch.float32),
            'drive_side': F.one_hot(torch.tensor(row['drive_side']), num_classes=2).float(),
            'land_cover': F.one_hot(torch.tensor(row['land_cover']), num_classes=12).float(),
            'quadtree_10_1000': F.one_hot(torch.tensor(row['quadtree_10_1000']), num_classes=11399).float()
        }
        
        return {'image': image, **metadata_dict}


def custom_collate(batch):
    images = torch.stack([item['image'] for item in batch])
    other_data = {key: torch.stack([item[key] for item in batch]) for key in batch[0] if key != 'image'}
    return images, other_data

from PIL import ImageFilter, ImageOps
import random
from torchvision import transforms
class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class gray_scale(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.Grayscale(3)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img
        
def train_transform():
    return T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.RandomChoice([gray_scale(p=1.0), Solarization(p=1.0), GaussianBlur(p=1.0)]),
        T.ToTensor()
    ])
    
def val_transform():
    return T.Compose([
        T.RandomResizedCrop(224),
        T.ToTensor()
    ])
    
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
            
if __name__ == "__main__":
    dataset_root = 'datasets/OpenWorld'
    train_dataset = ImageDataset(f'{dataset_root}/images/train', f'{dataset_root}/train_extras_removed.csv', train_transform())
    test_dataset = ImageDataset(f'{dataset_root}/images/test', f'{dataset_root}/test.csv', train_transform())
    
    train_size = int(0.5 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    
    model = Geoformer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print_trainable_parameters(model)
    
    trainer = Trainer(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        device=device,
        epochs=1,
        log_dir='logs/openworld_training',
        checkpoint_dir='checkpoints/openworld_model'
    )
    
    trainer.train()