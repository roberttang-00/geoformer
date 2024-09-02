import torch
import timm


class Geoformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('convnextv2_base', pretrained=True, num_classes=0)

        # feature_channels = self.backbone.feature_info.channels()
        #
        # self.fpn = FeaturePyramidNetwork(feature_channels, 256)

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
        # expert_balance_loss = self._compute_expert_balance_loss(routing_probs, top_k_indices)
        return outputs