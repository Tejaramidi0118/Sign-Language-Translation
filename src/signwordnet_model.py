import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ChannelGate(nn.Module):
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(dim // reduction, 8)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.fc2(F.relu(self.fc1(x)))
        gate = torch.sigmoid(g)
        return x * gate

class GatedResAdapter(nn.Module):
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.gate = ChannelGate(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(F.relu(self.fc1(x)))
        h = self.gate(h)
        return x + h

class SignWordNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.adapter = GatedResAdapter(dim=in_features, hidden=max(128, in_features // 2))

        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = self.adapter(feat)
        logits = self.head(feat)
        return logits

