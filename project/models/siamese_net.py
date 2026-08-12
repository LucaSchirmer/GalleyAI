"""Assembles a full Siamese consumption model from a chosen backbone and a
chosen regression head (see models/backbones.py and models/heads/ for the
available options). The classification head (drinks/extras/cookie) is
always attached, independent of which regression head is picked -- see
heads/classification_head.py's docstring for why.

Note: the Hybrid ML head (embeddings -> Gradient Boosting Regressor) is
NOT one of the REGRESSION_HEAD_BUILDERS below -- it has no gradient-trained
weights inside this network at all, so it isn't part of this module. See
training/train_hybrid_gbr.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.backbones import build_backbone
from models.heads.classification_head import ClassificationHead
from models.heads.distance_head import DistanceRegressionHead
from models.heads.mlp_head import ConcatMLPRegressionHead

REGRESSION_HEAD_BUILDERS = {
    "mlp": lambda feature_dim: ConcatMLPRegressionHead(feature_dim),
    "cosine": lambda feature_dim: DistanceRegressionHead(metric="cosine"),
    "euclidean": lambda feature_dim: DistanceRegressionHead(metric="euclidean"),
}


def list_regression_heads():
    return list(REGRESSION_HEAD_BUILDERS.keys())


class SiameseConsumptionNet(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        head_name: str = "mlp",
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        if head_name not in REGRESSION_HEAD_BUILDERS:
            raise ValueError(f"Unknown head '{head_name}'. Options: {list_regression_heads()}")

        self.backbone = build_backbone(backbone_name, pretrained=pretrained, freeze=freeze_backbone)
        self.regression_head = REGRESSION_HEAD_BUILDERS[head_name](self.backbone.feature_dim)
        self.classification_head = ClassificationHead(self.backbone.feature_dim)

        self.backbone_name = backbone_name
        self.head_name = head_name

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, before: torch.Tensor, after: torch.Tensor):
        """Returns (regression_output, classification_logit), each shape (B,).

        Both heads run on every sample in the batch -- cheap relative to
        the backbone forward passes -- and the caller (training/engine.py)
        picks which one applies per-sample based on that sample's task.
        """
        feat_before = self.encode(before)
        feat_after = self.encode(after)
        reg_out = self.regression_head(feat_before, feat_after)
        clf_logit = self.classification_head(feat_before, feat_after)
        return reg_out, clf_logit

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)


def combined_prediction(reg_out: torch.Tensor, clf_logit: torch.Tensor, is_regression: torch.Tensor) -> torch.Tensor:
    """Merges the two heads' outputs into one 0-1 "predicted fraction" per
    sample, picking whichever head applies for that sample (regression ->
    clamped raw output, classification -> sigmoid of the logit)."""
    reg_pred = torch.clamp(reg_out, 0.0, 1.0)
    clf_pred = torch.sigmoid(clf_logit)
    return torch.where(is_regression, reg_pred, clf_pred)
