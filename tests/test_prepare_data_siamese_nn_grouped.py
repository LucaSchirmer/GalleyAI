import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.prepare_data_siamese_nn_grouped import (
    assert_before_images_are_disjoint,
    group_pairs_by_before,
    split_stratum_counts,
    split_groups,
)


def pair(before: str, after: str):
    return {"before": before, "after": after}


class GroupedSplitTests(unittest.TestCase):
    def test_direct_execution_ignores_unrelated_scripts_package(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "prepare_data_siamese_nn_grouped.py"

        with tempfile.TemporaryDirectory() as temp_dir:
            foreign_package = Path(temp_dir) / "scripts"
            foreign_package.mkdir()
            (foreign_package / "__init__.py").write_text("", encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONPATH"] = temp_dir
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import runpy, sys; "
                        f"sys.path.insert(0, {str(script.parent)!r}); "
                        f"runpy.run_path({str(script)!r}, run_name='import_probe')"
                    ),
                ],
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_keeps_every_before_image_in_exactly_one_split(self):
        groups = [
            [pair(f"before-{index}", f"after-{index}-a"),
             pair(f"before-{index}", f"after-{index}-b")]
            for index in range(20)
        ]

        splits, group_counts = split_groups(groups, seed=42)

        self.assertEqual(group_counts, {"train": 14, "val": 3, "test": 3})
        self.assertEqual(sum(map(len, splits.values())), 40)
        assert_before_images_are_disjoint(splits)

    def test_coalesces_duplicate_before_groups_before_splitting(self):
        groups = [
            [pair("same-before", "after-a")],
            [pair("same-before", "after-b")],
            [pair("other-before", "after-c")],
        ]

        grouped = group_pairs_by_before(groups)
        splits, group_counts = split_groups(groups, seed=1, train_ratio=0.5, val_ratio=0)

        self.assertEqual(len(grouped), 2)
        self.assertEqual(group_counts, {"train": 1, "val": 0, "test": 1})
        owners = {
            split_name
            for split_name, pairs in splits.items()
            if any(item["before"] == "same-before" for item in pairs)
        }
        self.assertEqual(len(owners), 1)
        self.assertEqual(
            sum(item["before"] == "same-before" for pairs in splits.values() for item in pairs),
            2,
        )

    def test_same_seed_produces_same_split(self):
        groups = [[pair(f"before-{index}", f"after-{index}")] for index in range(10)]

        first, _ = split_groups(groups, seed=7)
        second, _ = split_groups(groups, seed=7)

        self.assertEqual(first, second)

    def test_targets_pair_ratios_when_group_sizes_differ(self):
        groups = [
            [pair(f"before-{group}", f"after-{group}-{item}") for item in range(size)]
            for group, size in enumerate([10, 8, 6, 5, 4, 3, 2, 2])
        ]

        splits, _ = split_groups(groups, seed=42)

        pair_counts = {name: len(items) for name, items in splits.items()}
        self.assertLessEqual(abs(pair_counts["train"] - 28), 1)
        self.assertLessEqual(abs(pair_counts["val"] - 6), 1)
        self.assertLessEqual(abs(pair_counts["test"] - 6), 1)
        assert_before_images_are_disjoint(splits)

    def test_rejects_invalid_ratios(self):
        with self.assertRaises(ValueError):
            split_groups([], train_ratio=0.9, val_ratio=0.2)

    def test_rejects_pair_without_before_path(self):
        with self.assertRaisesRegex(ValueError, "before"):
            split_groups([[{"after": "after-a"}]])

    def test_stratifies_regression_target_bins_across_splits(self):
        groups = []
        for index in range(30):
            target = 0.0 if index < 15 else 100.0
            groups.append([{
                "before": f"before-{index}",
                "after": f"after-{index}",
                "category": "wrap",
                "numbers": {"pct_wrap_merged": [target]},
            }])

        splits, _ = split_groups(groups, seed=42)
        counts = split_stratum_counts(splits)

        for split_name in ("train", "val", "test"):
            self.assertGreater(counts[split_name]["reg|pct_wrap_merged|0-20"], 0)
            self.assertGreater(counts[split_name]["reg|pct_wrap_merged|81-100"], 0)


if __name__ == "__main__":
    unittest.main()
