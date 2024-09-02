import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_m

class EfficientNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = efficientnet_v2_m(weights='IMAGENET1K_V1')
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier[1].in_features, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)