from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIAMESE_ROOT = PROJECT_ROOT / "siamese_consumption_model"
sys.path.insert(0, str(SIAMESE_ROOT))

from data.siamese_dataset import expected_polygon_labels
from training.engine import run_epoch


def test_wrap_metric_uses_segmentation_dataset_class_names() -> None:
    assert expected_polygon_labels("pct_wrap_merged") == ["wrap_half_1", "wrap_half_2"]


def test_fish_metric_uses_segmentation_dataset_class_name() -> None:
    assert expected_polygon_labels("pct_fish_rice_veg") == [
        "fish_salmon",
        "rice",
        "carrots",
        "broccoli",
    ]


def test_run_epoch_reports_balanced_classification_accuracy() -> None:
    class FixedModel(torch.nn.Module):
        def forward(self, before, after, metric_id, aux_features):
            # Labels below are [0, 0, 1, 1]; predictions are [0, 0, 0, 1].
            return torch.zeros(4), torch.tensor([-2.0, -2.0, -2.0, 2.0])

    batch = (
        torch.zeros(4, 4, 1, 1),
        torch.zeros(4, 4, 1, 1),
        torch.tensor([0.0, 0.0, 1.0, 1.0]),
        ["classification"] * 4,
        torch.zeros(4, dtype=torch.long),
        torch.zeros(4, 5),
    )
    metrics = run_epoch(FixedModel(), [batch], torch.device("cpu"))

    assert metrics["clf_acc"] == 0.75
    assert metrics["clf_recall_not_consumed"] == 1.0
    assert metrics["clf_recall_consumed"] == 0.5
    assert metrics["clf_balanced_acc"] == 0.75
