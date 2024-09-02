from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms as T
import timm
import os
import numpy as np
import time
from tqdm import tqdm
import pandas as pd
from PIL import Image

# torch.use_deterministic_algorithms(True)
torch.set_float32_matmul_precision('high')
torch.autograd.set_detect_anomaly(True)

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, 1))
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, features):
        laterals = [conv(feature) for feature, conv in zip(features, self.lateral_convs)]
        
        for i in range(len(laterals) - 1, 0, -1):
            target_size = laterals[i-1].shape[-2:]
            laterals[i - 1] += F.interpolate(laterals[i], size=target_size, mode='nearest')
            
        fpn_features = [conv(lateral) for lateral, conv in zip(laterals, self.fpn_convs)]
        pooled_features = [self.avgpool(feature) for feature in fpn_features]
        
        return pooled_features

class ExpertNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, expert_size=0.25):
        super().__init__()
        hidden_dim = int(input_dim * expert_size)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Geoformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('convnextv2_tiny', features_only=True)

        feature_channels = self.backbone.feature_info.channels()
        
        self.fpn = FeaturePyramidNetwork(feature_channels, 256)

        self.dim = 256 * len(feature_channels)
        
        self.num_shared_experts = 1
        self.num_experts = 32
        self.num_activated_experts = 4
        self.expert_size = 1 / self.num_activated_experts
        
        self.shared_experts = nn.ModuleList([
            ExpertNetwork(self.dim, 64, self.expert_size)
            for _ in range(self.num_shared_experts)
        ])
        
        self.routed_experts = nn.ModuleList([
            ExpertNetwork(self.dim, 64, self.expert_size)
            for _ in range(self.num_experts)
        ])
        
        self.router = nn.Linear(self.dim, self.num_experts)
        
        self.expert_balance_factor = 0.001
        
        self.lat_lon = nn.Linear(64, 2)
        self.quadtree = nn.Linear(64, 11399)

        
    def _get_dim(self):
        samp = torch.rand(1, 3, 224, 224)
        with torch.no_grad():
            out = self.backbone(samp)
        return out.shape[1]
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.fpn(x)
        
        x = torch.cat([feature.flatten(1) for feature in x], dim=1)
        routing_logits = self.router(x)
        routing_probs = F.softmax(routing_logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.num_activated_experts - self.num_shared_experts, dim=-1)
        normalized_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        shared_output = sum(expert(x) for expert in self.shared_experts)
        batch_size = x.size(0)

        output_size = 64
        
        routed_output = torch.zeros((batch_size, output_size), device=x.device)
        
        for expert_idx in range(self.num_experts):
            batch_mask = top_k_indices == expert_idx
            if not batch_mask.any():
                continue 
            
            relevant_x = x[batch_mask.any(dim=1).nonzero(as_tuple=True)[0]]
            relevant_probs = normalized_probs[batch_mask]
            
            expert_output = self.routed_experts[expert_idx](relevant_x)
            
            routed_output.index_add_(0, 
                                    batch_mask.nonzero()[:, 0], 
                                    expert_output * relevant_probs.unsqueeze(-1))
            
        # routed_output = torch.zeros_like(shared_output)
        # for i in range(self.num_activated_experts - self.num_shared_experts):
        #     expert_idx = top_k_indices[:, i]
        #     expert_output = torch.stack([
        #         self.routed_experts[idx.item()](x[b:b+1])
        #         for b, idx in enumerate(expert_idx)
        #     ])
        #     routed_output += expert_output.squeeze(1) * normalized_probs[:, i].unsqueeze(-1)
            
        combined_output = shared_output + routed_output
        lat_lon = self.lat_lon(combined_output)
        
        outputs = {
            "latitude": lat_lon[:, 0],
            "longitude": lat_lon[:, 1],
            "quadtree_10_1000": self.quadtree(combined_output),
        }
        expert_balance_loss = self._compute_expert_balance_loss(routing_probs, top_k_indices) * self.expert_balance_factor
        return outputs, expert_balance_loss
    
    def _compute_expert_balance_loss(self, routing_probs, top_k_indices):
        # routing_probs: [batch_size, num_routed_experts]
        # top_k_indices: [batch_size, K], where K = num_activated_experts - num_shared_experts
        batch_size, num_routed_experts = routing_probs.shape
        K = top_k_indices.size(1)

        # Calculate f_i (Equation 13)
        expert_mask = torch.zeros_like(routing_probs)
        expert_mask.scatter_(1, top_k_indices, 1.0)
        f_i = (num_routed_experts / (K * batch_size)) * expert_mask.sum(dim=0)

        # Calculate P_i (Equation 14)
        P_i = routing_probs.mean(dim=0)

        # Calculate the loss (Equation 12)
        loss = torch.sum(f_i * P_i)

        return loss
    
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
        self.w = torch.tensor([1.0, 1.0] + [0.05] * (n_tasks - 2), device=device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm
        self.n_tasks = n_tasks
        self.device = device
    
    def set_min_losses(self, losses):
        self.min_losses = losses
    
    def get_weighted_loss(self, losses):
        self.prev_loss = losses
        z = F.softmax(self.w, -1)
        D = losses - self.min_losses + 1e-8
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
        
def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm 
    
class Trainer:
    def __init__(self, model, dataloader, val_dataloader, test_dataloader, device, famo, epochs=100, log_dir='logs', checkpoint_dir='checkpoints'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model = torch.compile(self.model)
        
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.epochs = epochs
    
        # self.optimizer = torch.optim.AdamW([
        #     {'params': self.model.backbone.parameters(), 'lr': 1e-5, 'weight_decay': 0.0},
        #     {'params': [p for n, p in self.model.named_parameters() if not n.startswith('backbone')], 'lr': 1e-4, 'weight_decay': 1e-4},
        # ], fused=True)
        self.optimizer = torch.optim.RAdam(self.model.parameters(), lr=1e-3)
        
        def inspect_optimizer_groups(optimizer):
            total_params = 0
            for i, group in enumerate(optimizer.param_groups):
                params_in_group = sum(p.numel() for p in group['params'] if p.requires_grad)
                total_params += params_in_group
                print(f"Group {i}: {params_in_group} parameters, LR: {group['lr']}")
            print(f"Total parameters in optimizer: {total_params}")
            
        inspect_optimizer_groups(self.optimizer)
        self.gradient_accumulation_steps = 4
        self.famo = famo
        
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
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
            'famo_state': self.famo.w,
        }
        torch.save(checkpoint, f'{self.checkpoint_dir}/checkpoint_{step}.pth')
        
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.famo.w = checkpoint['famo_state']
        return checkpoint['step']
    
    def test(self):
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

        print(f"Test | Distances: {np.median(distances):.2f} | Quadtree Accuracy: {np.median(quadtree_accs):.3f}")
    
    def train(self, starting_step=0, validate_first=False):
        max_steps = self.epochs * len(self.dataloader)
        current_step = starting_step
        def cycling_dataloader(dataloader):
            while True:
                for data in dataloader:
                    yield data
        data_iter = cycling_dataloader(self.dataloader)
        print("Starting training...")
        while current_step < max_steps:
            if validate_first or current_step != starting_step and current_step % 1000 == 0:
                quadtree_accs = []
                distances = []
        
                self.model.eval()
                val_progress = tqdm(self.val_dataloader, total=len(self.val_dataloader))
                with torch.no_grad():
                    for batch in val_progress:
                        inputs, targets = batch
                        inputs = inputs.to(self.device)
                        targets = {key: value.to(self.device) for key, value in targets.items()}
                        outputs, _ = self.model(inputs)
                        
                        quadtree_accuracy = outputs['quadtree_10_1000'].argmax(dim=1).eq(targets['quadtree_10_1000'].argmax(dim=1)).float().mean()

                        val_dist = self.calculate_haverstine_distance(outputs, targets)
                        quadtree_accs.append(quadtree_accuracy.item())
                        distances.append(val_dist.item())
                validate_first = False

                print(f"Step: {current_step:05} | Distances: {np.median(distances):.2f} | Quadtree Accuracy: {np.median(quadtree_accs):.3f}")
            self.model.train()
            
            t0 = time.time()
            
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _ in range(self.gradient_accumulation_steps):
                inputs, targets = next(data_iter)
                inputs = inputs.to(self.device)
                targets = {key: value.to(self.device) for key, value in targets.items()}
                outputs, balance_loss = self.model(inputs)
                task_losses = self.calculate_losses(outputs, targets)
                losses = torch.cat([task_losses, balance_loss.unsqueeze(0)])
                loss = self.famo.get_weighted_loss(losses)
                loss /= self.gradient_accumulation_steps
                accum_loss += loss.detach()
                
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
            self.optimizer.step()
            
            with torch.no_grad():
                new_outputs, new_balance_loss = self.model(inputs)
                new_task_losses = torch.cat([self.calculate_losses(new_outputs, targets), new_balance_loss.unsqueeze(0)])
                self.famo.update(new_task_losses)
            
            torch.cuda.synchronize()
            t1 = time.time()
            dt = t1 - t0
            images_processed = self.gradient_accumulation_steps * inputs.size(0)
            images_per_sec = images_processed / dt
        
            current_step += 1
            
            print(f"Step: {current_step:05} | Loss: {accum_loss.item():.4f} | Norm: {norm:.4f} | dt: {dt * 1000:.2f}ms | img/sec: {images_per_sec:.2f} | w: {[f'{w:.2f}' for w in F.softmax(self.famo.w, dim=0).tolist()]}")
        
            if (current_step + 1) % 500 == 0:
                self.save_checkpoint(current_step + 1)
                torch.cuda.empty_cache()
    
    # def train(self, starting_step=0):
    #     print("Starting training...")
    #     for epoch in range(self.epochs):
    #         progress_bar = tqdm(self.dataloader, total=len(self.dataloader), desc=f"Epoch {epoch+1}/{self.epochs}")
    #         for batch in progress_bar:
    #             inputs, targets = batch
    #             inputs = inputs.to(self.device)
    #             targets = {key: value.to(self.device) for key, value in targets.items()}
    #             outputs = self.model(inputs)
    #             losses = self.calculate_losses(outputs, targets)
    #             loss = self.famo.get_weighted_loss(losses)
    #             self.optimizer.zero_grad(set_to_none=True)
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    #             self.optimizer.step()
                
    #             with torch.no_grad():
    #                 new_outputs = self.model(inputs)
    #                 new_task_losses = self.calculate_losses(new_outputs, targets)
    #                 self.famo.update(new_task_losses)
                
    #             break
    #         break
        
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
            # 'climate': F.one_hot(torch.tensor(row['climate']), num_classes=31).float(),
            # 'soil': F.one_hot(torch.tensor(row['soil']), num_classes=15).float(),
            # 'dist_sea': torch.tensor(row['dist_sea'], dtype=torch.float32),
            # 'drive_side': F.one_hot(torch.tensor(row['drive_side']), num_classes=2).float(),
            # 'land_cover': F.one_hot(torch.tensor(row['land_cover']), num_classes=12).float(),
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
    
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    subset_size = 256 # int(0.5 * len(train_dataset))
    val_size = 256 # int(0.5 * len(val_dataset))
    train_subset = Subset(train_dataset, list(range(subset_size)))
    val_subset = Subset(val_dataset, list(range(val_size)))
    
    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    
    model = Geoformer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print_trainable_parameters(model)
    
    famo = FAMO(4, device)
    
    trainer = Trainer(
        model=model,
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        famo=famo,
        epochs=100,
        log_dir='logs/openworld_training',
        checkpoint_dir='checkpoints/openworld_model'
    )
    
    # step = trainer.load_checkpoint('checkpoints/openworld_model/checkpoint_500.pth')
    # trainer.test()
    
    trainer.train()