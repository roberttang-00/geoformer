
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.optim as optim
import timm
import os
import numpy as np
import time
from tqdm import tqdm
import pandas as pd
import math
import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator


# torch.use_deterministic_algorithms(True)
torch.set_float32_matmul_precision('high')
# torch.autograd.set_detect_anomaly(True)

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
        self.backbone = timm.create_model('fastvit_s12', num_classes=0)

        dim = self._get_dim()
        print(f"dim: {dim}")

        self.shared_layer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.lat_lon = nn.Linear(dim, 2)

        self.quadtree = nn.Linear(dim, 11399)

        
    def _get_dim(self):
        samp = torch.rand(1, 3, 224, 224)
        with torch.no_grad():
            out = self.backbone(samp)
        return out.shape[1]
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.shared_layer(x)
        lat_lon = self.lat_lon(x)
        
        outputs = {
            "latitude": lat_lon[:, 0],
            "longitude": lat_lon[:, 1],
            "quadtree_10_1000": self.quadtree(x),
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
    
    
class ExternalInputIterator(object):
    def __init__(self, csv_file, batch_size, quadtree_data):
        self.csv_data = pd.read_csv(csv_file)
        self.batch_size = batch_size
        self.quadtree_data = quadtree_data
        self.cluster_metadata = quadtree_data.set_index('cluster_id')[['mean_lat', 'mean_lon', 'min_lat', 'min_lon', 'max_lat', 'max_lon']].to_dict('index')
        self.data_size = len(self.csv_data)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.data_size:
            self.i = 0
            raise StopIteration
        
        batch = []
        for _ in range(self.batch_size):
            if self.i >= self.data_size:
                break

            row = self.csv_data.iloc[self.i]
            cluster_id = row['quadtree_10_1000']
            qt_metadata = self.cluster_metadata[cluster_id]

            lat = row['latitude']
            lon = row['longitude']

            lat_mean, lon_mean = qt_metadata['mean_lat'], qt_metadata['mean_lon']
            lat_min, lon_min = qt_metadata['min_lat'], qt_metadata['min_lon']
            lat_max, lon_max = qt_metadata['max_lat'], qt_metadata['max_lon']

            lat_range = lat_max - lat_min
            lon_range = lon_max - lon_min

            lat_relative = (lat - lat_mean) / (lat_range / 2)
            lon_relative = (lon - lon_mean) / (lon_range / 2)

            lat_relative = max(-1, min(1, lat_relative))
            lon_relative = max(-1, min(1, lon_relative))

            batch.append({
                'file_name': f"{row['id']}.jpg",
                'latitude': lat_relative,
                'longitude': lon_relative,
                'quadtree_10_1000': cluster_id
            })
            self.i += 1
        return batch

    @property
    def size(self):
        return self.data_size

    next = __next__
    
class DALIPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, csv_file, quadtree_data, is_train=True):
        super().__init__(batch_size, num_threads, device_id, seed=12)
        self.input = fn.external_source(source=ExternalInputIterator(csv_file, batch_size, quadtree_data), num_outputs=4)
        self.data_dir = data_dir
        self.is_train = is_train

    def define_graph(self):
        file_name, latitude, longitude, quadtree = self.input()
        jpegs = fn.file_reader(file_root=self.data_dir, file_name=file_name)
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        
        if self.is_train:
            images = fn.random_resized_crop(images, device="gpu", size=224, random_area=(0.7, 1.0))
            images = fn.flip(images, device="gpu", horizontal=fn.random.coin_flip(probability=0.5))
            
            condition = fn.random.uniform(range=(0, 1))
            gaussian_applied = fn.cast(condition < 0.33, dtype=types.FLOAT)
            solarize_applied = fn.cast((condition >= 0.33) & (condition < 0.66), dtype=types.FLOAT)
            grayscale_applied = fn.cast(condition >= 0.66, dtype=types.FLOAT)
            
            blurred = fn.gaussian_blur(images, device="gpu", window_size=7)
            solarized = fn.color_twist(images, device="gpu", brightness=fn.random.uniform(range=(0.5, 2)))
            gray = fn.color_space_conversion(images, device="gpu", image_type=types.RGB, output_type=types.GRAY)
            gray = fn.cat(gray, gray, gray, axis=2)
            
            images = gaussian_applied * blurred + solarize_applied * solarized + grayscale_applied * gray
        else:
            images = fn.resize(images, device="gpu", size=224)

        # Convert to float in range [0, 1]
        images = fn.cast(images, dtype=types.FLOAT) / 255.0

        # NHWC to NCHW
        images = fn.transpose(images, perm=[2, 0, 1])
        
        return images, latitude, longitude, quadtree
    
class Trainer:
    def __init__(self, model, train_pipeline, device, val_pipeline=None, test_pipeline=None, famo=None, epochs=100, log_dir='logs', checkpoint_dir='checkpoints'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model = torch.compile(self.model)
        
        self.train_pipeline = train_pipeline
        self.val_pipeline = val_pipeline
        self.test_pipeline = test_pipeline
        self.epochs = epochs

        self.gradient_accumulation_steps = 8

        self.max_lr = 1e-4
        self.min_lr = self.max_lr * 0.1
        self.warmup_steps = 1000
        self.max_steps = self.epochs * len(self.dataloader) // self.gradient_accumulation_steps

        self.optimizer = optim.Adan(self.model.parameters(), lr=self.max_lr, weight_decay=1e-2)
        
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
        validate_first = False

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
    
    def train(self, starting_step=0):
        current_step = starting_step
        train_loader = DALIGenericIterator(self.train_pipeline, ['image', 'latitude', 'longitude', 'quadtree_10_1000'], size=self.train_pipeline.epoch_size("Reader"))
        
        print("Starting training...")
        self.model.train()
        while current_step < self.max_steps:
            t0 = time.time()
            
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            quadtree_accs = 0.0
            for _ in range(self.gradient_accumulation_steps):
                try:
                    batch = next(train_loader)
                except StopIteration:
                    train_loader.reset()
                    batch = next(train_loader)
                inputs = batch[0]['image']
                targets = {
                    'quadtree_10_1000': F.one_hot(batch[0]['quadtree_10_1000'].squeeze().long(), num_classes=11399).float(),
                    'latitude': batch[0]['latitude'],
                    'longitude': batch[0]['longitude']
                }
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = self.model(inputs)
                    task_losses = self.calculate_losses(outputs, targets)
                    quadtree_accuracy = outputs['quadtree_10_1000'].argmax(dim=1).eq(targets['quadtree_10_1000'].argmax(dim=1)).float().mean()
                    quadtree_accs += quadtree_accuracy
                    loss = task_losses.sum()
                    loss /= self.gradient_accumulation_steps
                loss.backward()
                accum_loss += loss.detach()
                
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # lr = self.get_lr(current_step)
            # for param_group in self.optimizer.param_groups:
            #     param_group['lr'] = lr
            self.optimizer.step()
            
            torch.cuda.synchronize()
            t1 = time.time()
            dt = t1 - t0
            images_processed = self.gradient_accumulation_steps * inputs.size(0)
            images_per_sec = images_processed / dt
        
            current_step += 1

            quadtree_accs /= self.gradient_accumulation_steps
            
            print(f"Step: {current_step:05} | L: {accum_loss.item():.4f} | QA: {quadtree_accs.item():.4f} | Norm: {norm:.4f} | dt: {dt * 1000:.2f}ms | img/sec: {images_per_sec:.2f}")
        
            if (current_step + 1) % 1000 == 0:
                self.save_checkpoint(current_step + 1)
                torch.cuda.empty_cache()
    
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

    batch_size = 224
    num_threads = 4
    device_id = 0

    train_pipeline = DALIPipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, 
                                  data_dir=f'{dataset_root}/images/train', 
                                  csv_file=f'{dataset_root}/train.csv', 
                                  quadtree_data=quadtree_data, is_train=True)
    # val_pipeline = DALIPipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, 
    #                             data_dir=f'{dataset_root}/images/train', 
    #                             csv_file=f'{dataset_root}/val.csv', 
    #                             quadtree_data=quadtree_data, is_train=False)
    test_pipeline = DALIPipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, 
                                 data_dir=f'{dataset_root}/images/test', 
                                 csv_file=f'{dataset_root}/test.csv', 
                                 quadtree_data=quadtree_data, is_train=False)

    train_pipeline.build()
    # val_pipeline.build()
    test_pipeline.build()
    
    model = Geoformer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print_trainable_parameters(model)
    
    trainer = Trainer(
        model=model,
        train_pipeline=train_pipeline,
        test_pipeline=test_pipeline,
        device=device,
        epochs=10,
        log_dir='logs/openworld_training',
        checkpoint_dir='checkpoints/openworld_model'
    )
    
    # step = trainer.load_checkpoint('checkpoints/openworld_model/checkpoint_13000.pth')
    # trainer.test()
    
    trainer.train()