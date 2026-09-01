"""Non-parametric regression head: Cosine Similarity / Normalized Euclidean
Distance.

"Non-parametric" here means the before/after COMPARISON itself has no
learned weights -- it's fixed math (cosine similarity or normalized
Euclidean distance) applied directly to the backbone embeddings, unlike
the MLP head which learns an arbitrary function of them.

The only two learnable numbers in this head are a scale and bias used to
calibrate that raw similarity/distance value onto the 0-100% consumption
scale. This calibration is necessary, not optional: raw cosine similarity
lives in [-1, 1] and raw Euclidean distance is unbounded, and there's no
guarantee the sign even points the right way (e.g. "very similar
before/after embeddings" could mean "almost nothing was eaten" -- the
model needs at least an affine transform to align that with "0% consumed"
vs "100% consumed" during training). Everything else about the before/
after comparison stays fixed, closed-form math.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistanceRegressionHead(nn.Module):
    def __init__(self, metric: str = "cosine"):
        super().__init__()
        if metric not in ("cosine", "euclidean"):
            raise ValueError("metric must be 'cosine' or 'euclidean'")
        self.metric = metric
        # Calibration only -- the comparison itself is fixed, no other weights.
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def raw_similarity(self, feat_before: torch.Tensor, feat_after: torch.Tensor) -> torch.Tensor:
        if self.metric == "cosine":
            return F.cosine_similarity(feat_before, feat_after, dim=1, eps=1e-8)  # in [-1, 1]
        a = F.normalize(feat_before, dim=1, eps=1e-8)
        b = F.normalize(feat_after, dim=1, eps=1e-8)
        dist = torch.norm(a - b, dim=1)  # in [0, 2]
        return 1.0 - dist  # similarity-like: higher = more similar

    def forward(
        self,
        feat_before: torch.Tensor,
        feat_after: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sim = self.raw_similarity(feat_before, feat_after)
        return self.scale * sim + self.bias  # raw regression output
