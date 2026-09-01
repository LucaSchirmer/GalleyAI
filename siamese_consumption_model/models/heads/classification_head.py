"""Classification head for binary consumed/not-consumed targets (drinks,
extras, cookie).

Always uses concatenation + a small MLP, independent of whichever
regression head is selected for the current run. It needs its own trunk
because the distance-based and Hybrid-GBR regression heads don't produce
an intermediate representation that could be shared -- the classification
path has to stand on its own regardless of which regression head is being
compared in a given run.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, feature_dim: int, context_dim: int = 0, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        combined_dim = feature_dim * 4 + context_dim
        self.net = nn.Sequential(
            nn.Linear(combined_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),  # raw logit, use with BCEWithLogitsLoss
        )

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
        return self.net(combined).squeeze(1)
