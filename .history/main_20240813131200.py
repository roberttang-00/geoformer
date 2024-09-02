
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms as T
import timm.optim as optim
import timm
import os
import numpy as np
import time
from tqdm import tqdm
import pandas as pd
from PIL import Image
import math
import lmdb
import pickle

# torch.use_deterministic_algorithms(True)
torch.set_float32_matmul_precision('high')
# torch.autograd.set_detect_anomaly(True)
    
class Geoformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('crossvit_small_240', num_classes=0, drop_rate=0.3)

        self.feature_dims = self._get_dim()
        print(f"dim: {self.feature_dims}")

        self.quadtree = nn.Linear(self.feature_dims, 11399)

        self.lat_lon = nn.Linear(self.feature_dims, 2)
        
    def _get_dim(self):
        samp = torch.rand(1, 3, 224, 224)
        with torch.no_grad():
            out = self.backbone(samp)
        return out.shape[1]
        
    def forward(self, x):
        features = self.backbone(x)
        
        lat_lon = self.lat_lon(features)
        
        outputs = {
            "latitude": lat_lon[:, 0],
            "longitude": lat_lon[:, 1],
            "quadtree_10_1000": self.quadtree(features),
        }
        return outputs
    
class FAMO:
    def __init__(
        self,
        n_tasks,
        device,
        gamma=0.001,
        w_lr=0.025,
        max_norm=1.0
    ):
        self.min_losses = torch.zeros(n_tasks).to(device)
        self.w = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm
        self.n_tasks = n_tasks
        self.device = device
    
    def set_min_losses(self, losses):
        self.min_losses = losses
    
    def get_weighted_loss(self, losses):
        self.prev_loss = losses
        z = F.softmax(self.w, -1)
        D = losses - self.min_losses + 1
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        return loss
    
    def update(self, curr_loss):
        delta = (self.prev_loss - self.min_losses + 1e-8).log() - \
                (curr_loss      - self.min_losses + 1e-8).log()
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.w, -1),
                                    self.w,
                                    grad_outputs=delta.detach())[0]
        self.w_opt.zero_grad()
        self.w.grad = d
        self.w_opt.step()
    
class Trainer:
    def __init__(self, model, dataloader, val_dataloader, device, test_dataloader=None, famo=None, epochs=100, log_dir='logs', checkpoint_dir='checkpoints'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model = torch.compile(self.model)
        
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.epochs = epochs

        self.gradient_accumulation_steps = 8

        self.max_lr = 3e-4
        self.min_lr = self.max_lr * 0.1
        self.max_steps = self.epochs * len(self.dataloader) // self.gradient_accumulation_steps
        self.warmup_steps = int(self.max_steps * 0.10)
        
        print(f'steps per epoch: {len(self.dataloader) // self.gradient_accumulation_steps}')
        print(f'total steps: {self.max_steps}')

        self.optimizer = optim.Adan(self.model.parameters(), lr=self.max_lr)
        # self.lookahead = optim.Lookahead(self.optimizer)
        
        def inspect_optimizer_groups(optimizer):
            total_params = 0
            for i, group in enumerate(optimizer.param_groups):
                params_in_group = sum(p.numel() for p in group['params'] if p.requires_grad)
                total_params += params_in_group
                print(f"Group {i}: {params_in_group} parameters, LR: {group['lr']}")
            print(f"Total parameters in optimizer: {total_params}")
            
        inspect_optimizer_groups(self.optimizer)
        
        self.famo = famo
        
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def get_lr(self, it):
        if it < self.warmup_steps:
            return self.max_lr * (it + 1) / self.warmup_steps
        if it > self.max_steps:
            return self.min_lr
        decay_ratio = (it - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)
    
    def calculate_losses(self, outputs, targets):
        losses = []
        for key, value in outputs.items():
            if key in ['longitude', 'latitude', 'dist_sea']:
                task_loss = F.mse_loss(value, targets[key])
            else:
                task_loss = F.cross_entropy(value, targets[key])
            losses.append(task_loss)
        return torch.stack(losses)
    
    def calculate_haverstine_distance(self, outputs, targets):
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
        return torch.mean(d)
        
    def save_checkpoint(self, step):
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, f'{self.checkpoint_dir}/checkpoint_{step}.pth')
        
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['step']
    
    def validate(self):
        quadtree_accs = []
        distances = []

        self.model.eval()
        val_progress = tqdm(self.val_dataloader, total=len(self.val_dataloader))
        with torch.no_grad():
            for batch in val_progress:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = {key: value.to(self.device) for key, value in targets.items()}
                outputs = self.model(inputs)
                
                quadtree_accuracy = outputs['quadtree_10_1000'].argmax(dim=1).eq(targets['quadtree_10_1000'].argmax(dim=1)).float().mean()

                val_dist = self.calculate_haverstine_distance(outputs, targets)
                quadtree_accs.append(quadtree_accuracy.item())
                distances.append(val_dist.item())

        print(f"Validation | Distances: {np.median(distances):.2f} | Quadtree Accuracy: {np.median(quadtree_accs):.3f}")
    
    def test(self):
        quadtree_accs = []

        self.model.eval()
        val_progress = tqdm(self.test_dataloader, total=len(self.test_dataloader))
        with torch.no_grad():
            for batch in val_progress:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = {key: value.to(self.device) for key, value in targets.items()}
                outputs = self.model(inputs)
                
                quadtree_accuracy = outputs['quadtree_10_1000'].argmax(dim=1).eq(targets['quadtree_10_1000'].argmax(dim=1)).float().mean()

                quadtree_accs.append(quadtree_accuracy.item())

        print(f"Test | Quadtree Accuracy: {np.median(quadtree_accs):.3f}")
    
    def train(self, starting_step=0, validate_first=False):
        current_step = starting_step
        def cycling_dataloader(dataloader):
            while True:
                for data in dataloader:
                    yield data
        data_iter = cycling_dataloader(self.dataloader)
        print("Starting training...")
        self.model.train()
        while current_step < self.max_steps:
            t0 = time.time()
            
            self.optimizer.zero_grad()
            accum_loss = 0.0
            quadtree_accs = 0.0
            for _ in range(self.gradient_accumulation_steps):
                inputs, targets = next(data_iter)
                inputs = inputs.to(self.device)
                targets = {key: value.to(self.device) for key, value in targets.items()}
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = self.model(inputs)
                    task_losses = self.calculate_losses(outputs, targets)
                    quadtree_accuracy = outputs['quadtree_10_1000'].argmax(dim=1).eq(targets['quadtree_10_1000'].argmax(dim=1)).float().mean()
                    quadtree_accs += quadtree_accuracy
                    loss = task_losses.sum()
                    loss /= self.gradient_accumulation_steps
                loss.backward()
                accum_loss += loss.detach()
                
            # scaler.unscale_(self.optimizer)
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
            lr = self.get_lr(current_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            # scaler.step(self.optimizer)
            # scaler.update()
            self.optimizer.step()
            
            torch.cuda.synchronize()
            t1 = time.time()
            dt = t1 - t0
            images_processed = self.gradient_accumulation_steps * inputs.size(0)
            images_per_sec = images_processed / dt
        
            current_step += 1

            quadtree_accs /= self.gradient_accumulation_steps
            
            print(f"Step: {current_step:05} | L: {accum_loss.item():.4f} | QA: {quadtree_accs.item():.4f} | norm: {norm:.4f} | dt: {dt * 1000:.2f}ms | img/sec: {images_per_sec:.2f}")
        
            if (current_step + 1) % 1000 == 0:
                self.save_checkpoint(current_step + 1)
                torch.cuda.empty_cache()
        
class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, quadtree_data):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)
        self.length = 500000
        self.quadtree_data = quadtree_data
        self.cluster_metadata = quadtree_data.set_index('cluster_id')[['mean_lat', 'mean_lon', 'min_lat', 'min_lon', 'max_lat', 'max_lon']].to_dict('index')

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.txn.get(f'{index}'.encode())
        data = pickle.loads(data)

        img = data['image']
        latitude = data['latitude']
        longitude = data['longitude']
        quadtree = data['quadtree_10_1000']
        
        cluster_id = int(quadtree)
        qt_metadata = self.cluster_metadata[cluster_id]
        
        lat = float(latitude)
        lon = float(longitude)
        
        lat_mean = qt_metadata['mean_lat']
        lon_mean = qt_metadata['mean_lon']

        lat_min = qt_metadata['min_lat']
        lon_min = qt_metadata['min_lon']

        lat_max = qt_metadata['max_lat']
        lon_max = qt_metadata['max_lon']

        lat_range = lat_max - lat_min
        lat_relative = (lat - lat_mean) / (lat_range / 2)

        lon_range = lon_max - lon_min
        lon_relative = (lon - lon_mean) / (lon_range / 2)

        lat_relative = max(-1, min(1, lat_relative))
        lon_relative = max(-1, min(1, lon_relative))

        
        metadata_dict = {
            'longitude': torch.tensor(lon_relative, dtype=torch.float32),
            'latitude': torch.tensor(lat_relative, dtype=torch.float32),
            'quadtree_10_1000': F.one_hot(torch.tensor(cluster_id), num_classes=11399).float()
        }
        
        return {'image': torch.tensor(img, dtype=torch.float32), **metadata_dict}

class ImageDataset(Dataset):
    def __init__(self, img_dir, csv_file, quadtree_data, transform=None, id_column='id'):
        self.img_dir = img_dir
        self.csv_data = pd.read_csv(csv_file)
        self.transform = transform or T.ToTensor()
        self.id_column = id_column
        self.ids = self.csv_data[self.id_column].tolist()
        self.quadtree_data = quadtree_data
        self.cluster_metadata = quadtree_data.set_index('cluster_id')[['mean_lat', 'mean_lon', 'min_lat', 'min_lon', 'max_lat', 'max_lon']].to_dict('index')

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
            return None
        
        if self.transform:
            image = self.transform(image)

        cluster_id = row['quadtree_10_1000']
        qt_row = self.cluster_metadata[cluster_id]

        lat = row['latitude']
        lon = row['longitude']

        lat_min = qt_row['min_lat']
        lon_min = qt_row['min_lon']

        lat_max = qt_row['max_lat']
        lon_max = qt_row['max_lon']

        lat_range = lat_max - lat_min
        lat_relative = (lat - lat_min) / lat_range

        lon_range = lon_max - lon_min
        lon_relative = (lon - lon_min) / lon_range
        
        if lat_range <= 0:
            print(cluster_id)
            print(lat)
            print(lat_min)
            print(lat_max)
            assert False
        if lon_range <= 0:
            print(cluster_id)
            print(lon)
            print(lon_min)
            print(lon_max)
        
        # Convert metadata to tensors
        metadata_dict = {
            'longitude': torch.tensor(lon_relative, dtype=torch.float32),
            'latitude': torch.tensor(lat_relative, dtype=torch.float32),
            'quadtree_10_1000': F.one_hot(torch.tensor(cluster_id), num_classes=11399).float()
        }
        
        return {'image': image, **metadata_dict}


def custom_collate(batch):
    images = torch.stack([item['image'] for item in batch])
    other_data = {key: torch.stack([item[key] for item in batch]) for key in batch[0] if key != 'image'}
    return images, other_data

from PIL import ImageFilter, ImageOps
import random
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
        self.transf = T.Grayscale(3)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img

def train_transform():
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.4, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomChoice([gray_scale(p=1.0), Solarization(p=1.0), GaussianBlur(p=1.0)]),
        T.ToTensor()
    ])
    
def val_transform():
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
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
    dataset_root = 'datasets/osv5m'
    quadtree_data = pd.read_csv(f'{dataset_root}/quadtree_10_1000.csv')

    # train_dataset = LMDBDataset(f'dataset_lmdb', quadtree_data)
    # test_dataset = LMDBDataset(f'{dataset_root}/images/test/test_db', quadtree_data, train_transform())
    
    train_dataset = ImageDataset(f'{dataset_root}/images/train', f'{dataset_root}/train.csv', quadtree_data, train_transform())
    test_dataset = ImageDataset(f'{dataset_root}/images/test', f'{dataset_root}/test.csv', quadtree_data, val_transform())
    
    train_size = int(0.99 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    batch_size = 224
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=custom_collate, pin_memory=True, drop_last=True, prefetch_factor=2)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=custom_collate, pin_memory=True, drop_last=True, prefetch_factor=2)
    
    model = Geoformer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print_trainable_parameters(model)
    
    trainer = Trainer(
        model=model,
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        epochs=100,
        log_dir='logs/openworld_training',
        checkpoint_dir='checkpoints/openworld_model'
    )
    
    step = trainer.load_checkpoint('checkpoints/openworld_model/checkpoint_101000.pth')
    # trainer.test()
    
    trainer.train(step)