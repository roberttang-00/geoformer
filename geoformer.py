import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

torch.set_float32_matmul_precision('high')

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, 1))
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        
    def forward(self, features):
        results = []
        last_feature = self.lateral_convs[-1](features[-1])
        results.append(self.fpn_convs[-1](last_feature))
        
        for feature, lateral_conv, fpn_conv in zip(
            reversed(features[:-1]), reversed(self.lateral_convs[:-1]), reversed(self.fpn_convs[:-1])
        ):
            feature_size = feature.shape[-2:]
            top_down_feature = F.interpolate(last_feature, size=feature_size, mode='nearest')
            lateral_feature = lateral_conv(feature)
            last_feature = lateral_feature + top_down_feature
            results.append(fpn_conv(last_feature))
        
        return tuple(reversed(results))

class Geoformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224.dino', pretrained=True, features_only=True)
        feature_channels = self.backbone.feature_info.channels()
        
        self.fpn = FeaturePyramidNetwork(feature_channels, 256)
        
        for name, param in self.backbone.named_parameters():
            if 'model.blocks.11' not in name:
                param.requires_grad = False
        
        # self.backbone.head.fc = nn.Identity()

        self.dim = 256 * len(feature_channels)
        print(f"dim: {self.dim}")
        
        self.lat_lon = nn.Linear(self.dim, 2)
        self.lat_bin = nn.Linear(self.dim, 100)
        self.lon_bin = nn.Linear(self.dim, 100)
        self.quadtree = nn.Linear(self.dim, 11399)
        self.drive_side = nn.Linear(self.dim, 2)                        
        self.land_cover = nn.Linear(self.dim, 12)
        self.climate = nn.Linear(self.dim, 31)
        self.soil = nn.Linear(self.dim, 15)
        self.dist_sea = nn.Linear(self.dim, 1)
        
    def _get_dim(self):
        samp = torch.rand(1, 3, 224, 224)
        with torch.no_grad():
            out = self.backbone(samp)
        return out.shape[1]
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.fpn(x)
        x = torch.cat([f.mean(dim=(2, 3)) for f in x], dim=1)
        lat_lon = self.lat_lon(x)
        
        outputs = {
            "latitude": lat_lon[:, 0],
            "longitude": lat_lon[:, 1],
            "lat_bin": self.lat_bin(x),
            "lon_bin": self.lon_bin(x),
            "quadtree_10_1000": self.quadtree(x),
            "drive_side": self.drive_side(x),
            "land_cover": self.land_cover(x),
            "climate": self.climate(x),
            "soil": self.soil(x),
            "dist_sea": self.dist_sea(x).squeeze()
        }
        return outputs