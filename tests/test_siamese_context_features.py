from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SIAMESE_ROOT = Path(__file__).resolve().parents[1] / "siamese_consumption_model"
sys.path.insert(0, str(SIAMESE_ROOT))

from data.metric_vocabulary import METRIC_NAMES, metric_index
from data.siamese_dataset import mask_statistics
from data.siamese_dataset import ConsumptionPairDataset
from models.siamese_net import SiameseConsumptionNet
from models.backbones import Backbone
from training.generate_report import infer_checkpoint_context


class SiameseContextFeatureTests(unittest.TestCase):
    def test_report_detects_legacy_checkpoint_without_context(self):
        state_dict = {
            "classification_head.net.0.weight": torch.empty(128, 3072),
        }

        self.assertEqual(infer_checkpoint_context(state_dict, feature_dim=768), (0, False))

    def test_report_detects_metric_embedding_and_aux_features(self):
        state_dict = {
            "classification_head.net.0.weight": torch.empty(128, 3093),
            "metric_embedding.weight": torch.empty(9, 16),
        }

        self.assertEqual(infer_checkpoint_context(state_dict, feature_dim=768), (16, True))

    def test_report_rejects_unknown_context_width(self):
        state_dict = {
            "classification_head.net.0.weight": torch.empty(128, 3074),
        }

        with self.assertRaisesRegex(RuntimeError, "Could not infer checkpoint context architecture"):
            infer_checkpoint_context(state_dict, feature_dim=768)

    def test_metric_vocabulary_is_stable_and_rejects_unknown_fields(self):
        self.assertEqual(len(METRIC_NAMES), len(set(METRIC_NAMES)))
        self.assertEqual(metric_index("pct_wrap_merged"), METRIC_NAMES.index("pct_wrap_merged"))
        with self.assertRaisesRegex(ValueError, "Unknown metric"):
            metric_index("pct_unknown")

    def test_mask_statistics_expose_area_ratio_reduction_and_missing_after(self):
        before = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        after = torch.tensor([[1.0, 0.0], [0.0, 0.0]])

        stats = mask_statistics(before, after)

        torch.testing.assert_close(
            stats,
            torch.tensor([0.5, 0.25, 0.5, 0.5, 0.0]),
        )

    def test_mask_statistics_marks_a_disappeared_after_mask(self):
        before = torch.ones(2, 2)
        after = torch.zeros(2, 2)

        stats = mask_statistics(before, after)

        self.assertEqual(stats[-1].item(), 1.0)
        self.assertEqual(stats[2].item(), 0.0)
        self.assertEqual(stats[3].item(), 1.0)

    def test_optimizer_groups_are_disjoint_and_use_separate_learning_rates(self):
        model = SiameseConsumptionNet.__new__(SiameseConsumptionNet)
        torch.nn.Module.__init__(model)
        model.backbone = torch.nn.Linear(2, 2)
        model.regression_head = torch.nn.Linear(2, 1)
        model.classification_head = torch.nn.Linear(2, 1)

        groups = model.optimizer_param_groups(head_lr=1e-3, backbone_lr=1e-5)

        self.assertEqual([group["lr"] for group in groups], [1e-5, 1e-3])
        backbone_ids = {id(parameter) for parameter in groups[0]["params"]}
        head_ids = {id(parameter) for parameter in groups[1]["params"]}
        self.assertTrue(backbone_ids)
        self.assertTrue(head_ids)
        self.assertTrue(backbone_ids.isdisjoint(head_ids))

    def test_balanced_sampling_upweights_rare_metric_target_strata(self):
        dataset = ConsumptionPairDataset.__new__(ConsumptionPairDataset)
        common = {
            "task": "regression", "metric_name": "pct_wrap_merged", "target_pct": 10.0,
        }
        rare = {
            "task": "regression", "metric_name": "pct_wrap_merged", "target_pct": 100.0,
        }
        dataset.samples = [common, common.copy(), rare]

        weights = dataset.balanced_sample_weights()

        self.assertEqual(weights.tolist(), [0.5, 0.5, 1.0])

    def test_partial_vit_unfreezing_only_opens_final_blocks_and_norm(self):
        backbone = Backbone.__new__(Backbone)
        torch.nn.Module.__init__(backbone)
        backbone.name = "vit_b16"
        backbone.model = torch.nn.Module()
        backbone.model.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(4)])
        backbone.model.norm = torch.nn.LayerNorm(2)
        for parameter in backbone.model.parameters():
            parameter.requires_grad = False

        backbone.unfreeze_last_blocks(2)

        self.assertFalse(any(p.requires_grad for p in backbone.model.blocks[0].parameters()))
        self.assertFalse(any(p.requires_grad for p in backbone.model.blocks[1].parameters()))
        self.assertTrue(all(p.requires_grad for p in backbone.model.blocks[2].parameters()))
        self.assertTrue(all(p.requires_grad for p in backbone.model.blocks[3].parameters()))
        self.assertTrue(all(p.requires_grad for p in backbone.model.norm.parameters()))


if __name__ == "__main__":
    unittest.main()
