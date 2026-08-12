"""Siamese consumption-percentage regressor.

Shared-weight ResNet50 backbone (modified to accept a 4th mask channel on
top of RGB), feeding into a concatenation + MLP regression head (the
"Parametric" option from the roadmap slide — simplest to get working
end-to-end first; swap in the Hybrid ML or Non-Parametric head later
without touching the backbone).

Backbone is frozen EXCEPT for conv1: conv1's weights for the 4th (mask)
channel start from a naive initialization (not ImageNet-pretrained), so
freezing it too would prevent the network from ever learning to use the
mask information at all. Everything from layer1 onward stays frozen,
which is what protects against overfitting on a small dataset.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class SiameseConsumptionNet(nn.Module):
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Expand conv1 from 3 (RGB) to 4 (RGB + mask) input channels.
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight  # keep pretrained RGB weights
            new_conv.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)  # init mask channel
        backbone.conv1 = new_conv

        backbone.fc = nn.Identity()  # drop ImageNet classifier, keep pooled 2048-d features
        self.backbone = backbone
        self.feature_dim = 2048

        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = name.startswith("conv1")

        # Standard Siamese combination trick: [a, b, |a-b|, a*b]
        combined_dim = self.feature_dim * 4
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        feat_before = self.encode(before)
        feat_after = self.encode(after)
        combined = torch.cat(
            [feat_before, feat_after, torch.abs(feat_before - feat_after), feat_before * feat_after],
            dim=1,
        )
        out = self.head(combined).squeeze(1)
        return torch.sigmoid(out)  # 0-1, matches normalized target (pct / 100)

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)
