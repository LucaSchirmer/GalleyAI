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
from data.metric_vocabulary import METRIC_NAMES
from data.siamese_dataset import AUX_FEATURE_DIM

REGRESSION_HEAD_BUILDERS = {
    "mlp": lambda feature_dim, **kwargs: ConcatMLPRegressionHead(feature_dim, **kwargs),
    "cosine": lambda feature_dim, **kwargs: DistanceRegressionHead(metric="cosine"),
    "euclidean": lambda feature_dim, **kwargs: DistanceRegressionHead(metric="euclidean"),
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
        unfreeze_last_blocks: int = 0,
        metric_embedding_dim: int = 16,
        use_metric_embedding: bool = True,
        use_aux_features: bool = True,
        regression_head_kwargs: dict | None = None,
    ):
        super().__init__()
        if head_name not in REGRESSION_HEAD_BUILDERS:
            raise ValueError(f"Unknown head '{head_name}'. Options: {list_regression_heads()}")

        self.backbone = build_backbone(backbone_name, pretrained=pretrained, freeze=freeze_backbone)
        if unfreeze_last_blocks:
            self.backbone.unfreeze_last_blocks(unfreeze_last_blocks)
        self.metric_embedding = (
            nn.Embedding(len(METRIC_NAMES), metric_embedding_dim)
            if use_metric_embedding else None
        )
        self.use_aux_features = use_aux_features
        context_dim = (metric_embedding_dim if use_metric_embedding else 0) + (
            AUX_FEATURE_DIM if use_aux_features else 0
        )
        regression_head_kwargs = dict(regression_head_kwargs or {})
        regression_head_kwargs["context_dim"] = context_dim
        self.regression_head = REGRESSION_HEAD_BUILDERS[head_name](self.backbone.feature_dim, **regression_head_kwargs)
        self.classification_head = ClassificationHead(self.backbone.feature_dim, context_dim=context_dim)

        self.backbone_name = backbone_name
        self.head_name = head_name

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        metric_id: torch.Tensor,
        aux_features: torch.Tensor,
    ):
        """Returns (regression_output, classification_logit), each shape (B,).

        Both heads run on every sample in the batch -- cheap relative to
        the backbone forward passes -- and the caller (training/engine.py)
        picks which one applies per-sample based on that sample's task.
        """
        feat_before = self.encode(before)
        feat_after = self.encode(after)
        context_parts = []
        if self.metric_embedding is not None:
            context_parts.append(self.metric_embedding(metric_id))
        if self.use_aux_features:
            context_parts.append(aux_features)
        context = torch.cat(context_parts, dim=1) if context_parts else None
        reg_out = self.regression_head(feat_before, feat_after, context)
        clf_logit = self.classification_head(feat_before, feat_after, context)
        return reg_out, clf_logit

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def optimizer_param_groups(self, head_lr: float, backbone_lr: float):
        """Return disjoint AdamW groups for pretrained and new parameters."""
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        backbone_ids = {id(p) for p in backbone_params}
        head_params = [p for p in self.parameters() if p.requires_grad and id(p) not in backbone_ids]
        return [
            {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
            {"params": head_params, "lr": head_lr, "name": "heads_and_context"},
        ]


def combined_prediction(reg_out: torch.Tensor, clf_logit: torch.Tensor, is_regression: torch.Tensor) -> torch.Tensor:
    """Merges the two heads' outputs into one 0-1 "predicted fraction" per
    sample, picking whichever head applies for that sample (regression ->
    clamped raw output, classification -> sigmoid of the logit)."""
    reg_pred = torch.clamp(reg_out, 0.0, 1.0)
    clf_pred = torch.sigmoid(clf_logit)
    return torch.where(is_regression, reg_pred, clf_pred)
