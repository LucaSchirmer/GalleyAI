"""Parametric regression head: Concatenation + Multi-Layer Perceptron.

Combines the two embeddings using the standard Siamese trick
[a, b, |a-b|, a*b] before an MLP trunk. This is the richest of the three
gradient-trained regression heads -- it can in principle learn any
function of the two embeddings, at the cost of having the most parameters
and thus the highest overfitting risk on a small dataset.

This is a straight refactor of the original SiameseConsumptionNet's
regression path (previously inlined directly in the model) into its own
module so it can be swapped for the other heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConcatMLPRegressionHead(nn.Module):
    def __init__(self, feature_dim: int, context_dim: int = 0, trunk_hidden=(256, 64), dropout: float = 0.3):
        super().__init__()
        combined_dim = feature_dim * 4 + context_dim
        h1, h2 = trunk_hidden
        self.trunk = nn.Sequential(
            nn.Linear(combined_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        self.out = nn.Linear(h2, 1)  # raw linear output, no activation

    def forward(
        self,
        feat_before: torch.Tensor,
        feat_after: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        combined = torch.cat(
            [feat_before, feat_after, torch.abs(feat_before - feat_after), feat_before * feat_after],
            dim=1,
        )
        if context is not None:
            combined = torch.cat([combined, context], dim=1)
        trunk_out = self.trunk(combined)
        return self.out(trunk_out).squeeze(1)
