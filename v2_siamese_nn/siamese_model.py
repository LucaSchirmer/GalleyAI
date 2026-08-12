"""Siamese consumption model with two separate output heads.

Shared-weight ResNet50 backbone (modified to accept a 4th mask channel on
top of RGB) feeds a shared MLP trunk, which then splits into TWO separate
output heads:
  - regression_head: raw linear output, trained with MSE against the
    continuous 0-1 pct_* targets.
  - classification_head: raw logit, trained with BCEWithLogitsLoss against
    the binary 0/1 consumed/not-consumed targets.

Previously both task types shared one output neuron and one loss. That
forced a genuinely binary target (0 or 100) through the same continuous
regression neuron as real percentages — minimizing squared error across
that mix mathematically pulls ambiguous-looking classification predictions
toward the middle, which is exactly the "clustering near 50%" symptom seen
in evaluation. Splitting into two heads (hard parameter sharing on the
trunk, separate task-appropriate loss per head) is the standard fix for
heterogeneous regression+classification multi-task setups.

Backbone is frozen EXCEPT for conv1 (see prior reasoning: its mask-channel
weights aren't pretrained, so freezing it too would block the network from
ever learning to use the mask at all).
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

        # Shared trunk (hard parameter sharing) -> two separate small heads.
        self.trunk = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.regression_head = nn.Linear(64, 1)      # raw linear output, no activation
        self.classification_head = nn.Linear(64, 1)  # raw logit, use with BCEWithLogitsLoss

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, before: torch.Tensor, after: torch.Tensor):
        """Returns (regression_output, classification_logit), each shape (B,).

        Both heads run on every sample in the batch — cheap, since they're
        just two Linear(64, 1) layers — the caller picks which one applies
        per-sample based on that sample's task type.
        """
        feat_before = self.encode(before)
        feat_after = self.encode(after)
        combined = torch.cat(
            [feat_before, feat_after, torch.abs(feat_before - feat_after), feat_before * feat_after],
            dim=1,
        )
        trunk_out = self.trunk(combined)
        reg_out = self.regression_head(trunk_out).squeeze(1)
        clf_logit = self.classification_head(trunk_out).squeeze(1)
        return reg_out, clf_logit

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)


def combined_prediction(reg_out: torch.Tensor, clf_logit: torch.Tensor, is_regression: torch.Tensor) -> torch.Tensor:
    """
    Merges the two heads' outputs into one 0-1 "predicted fraction" per
    sample, picking whichever head applies for that sample (regression ->
    clamped raw output, classification -> sigmoid of the logit). Used by
    evaluation/reporting scripts that want one combined prediction array,
    matching the pre-two-head reporting format.
    """
    reg_pred = torch.clamp(reg_out, 0.0, 1.0)
    clf_pred = torch.sigmoid(clf_logit)
    return torch.where(is_regression, reg_pred, clf_pred)