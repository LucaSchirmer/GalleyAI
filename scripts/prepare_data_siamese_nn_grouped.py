"""Build leakage-resistant Siamese train/val/test manifests.

Unlike ``prepare_data_siamese_nn.py``, this variant assigns complete groups
of pairs to a split.  A group is identified by its ``before`` image, so every
``after`` image mapped to the same ``before`` image is kept in exactly one of
train, validation, or test.

Run from the repository root:

    python scripts/prepare_data_siamese_nn_grouped.py

The generated files use the same ``data_pairs_with_splits`` location expected
by the training scripts.  Running this script therefore replaces the current
manifests, but does not copy or modify any images.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

if __package__:
    # Package/module usage, including imports from the test suite.
    from . import prepare_data_siamese_nn as source
else:
    # Direct execution: python scripts/<this file>.py. Import the sibling
    # directly so an unrelated installed package named ``scripts`` cannot
    # shadow this repository's scripts directory.
    import prepare_data_siamese_nn as source


Pair = Dict[str, Any]
Splits = Dict[str, List[Pair]]
SPLIT_NAMES = ("train", "val", "test")


def regression_target_bin(value: float) -> str:
    value = max(0.0, min(100.0, float(value)))
    if value <= 20:
        return "0-20"
    if value <= 40:
        return "21-40"
    if value <= 60:
        return "41-60"
    if value <= 80:
        return "61-80"
    return "81-100"


def pair_strata(pair: Pair) -> Counter:
    """Describe a pair using category, regression-bin, and class strata.

    Category is balanced independently. Combining it with every target bin
    makes strata too sparse for this dataset's roughly fifty tray groups.
    """
    category = pair.get("category") or "unknown"
    strata = Counter({f"category|{category}": 1})
    for metric_name, values in pair.get("numbers", {}).items():
        if values:
            target_bin = regression_target_bin(values[0])
            strata[f"reg|{metric_name}|{target_bin}"] += 1
    for field, nested_values in pair.get("choices", {}).items():
        if not field.startswith(("drink_", "extra_", "status_")):
            continue
        values = nested_values[0] if nested_values else []
        if values and values[0] in {"Consumed", "Not consumed"}:
            strata[f"clf|{field}|{values[0]}"] += 1
    return strata


def split_stratum_counts(splits: Splits) -> Dict[str, Counter]:
    return {
        split_name: sum((pair_strata(pair) for pair in pairs), Counter())
        for split_name, pairs in splits.items()
    }


def group_pairs_by_before(groups: Iterable[Iterable[Pair]]) -> Dict[str, List[Pair]]:
    """Coalesce input groups by their before-image path.

    Coalescing makes the leakage guarantee hold even if the same before image
    accidentally occurs in more than one manifest entry.
    """
    by_before: Dict[str, List[Pair]] = {}
    for group in groups:
        for pair in group:
            before = pair.get("before")
            if not isinstance(before, str) or not before:
                raise ValueError("Every pair must contain a non-empty 'before' path")
            by_before.setdefault(before, []).append(pair)
    return by_before


def split_groups(
    groups: Iterable[Iterable[Pair]],
    seed: int = source.SEED,
    train_ratio: float = source.TRAIN_RATIO,
    val_ratio: float = source.VAL_RATIO,
) -> Tuple[Splits, Dict[str, int]]:
    """Split by unique before image while targeting the requested pair ratios.

    Complete groups are assigned greedily, largest first, to the split with
    the greatest remaining pair budget. Random shuffling before the stable
    size sort makes equal-sized groups seed-dependent. Exact percentages are
    not always possible because a group is never divided.
    """
    if not 0 <= train_ratio <= 1 or not 0 <= val_ratio <= 1:
        raise ValueError("train_ratio and val_ratio must be between 0 and 1")
    if train_ratio + val_ratio > 1:
        raise ValueError("train_ratio + val_ratio must not exceed 1")

    by_before = group_pairs_by_before(groups)
    group_strata = {
        before: sum((pair_strata(pair) for pair in pairs), Counter())
        for before, pairs in by_before.items()
    }
    total_strata = sum(group_strata.values(), Counter())

    before_paths = list(by_before)
    random.Random(seed).shuffle(before_paths)
    before_paths.sort(
        key=lambda before: (
            sum(count / total_strata[stratum] for stratum, count in group_strata[before].items()),
            len(by_before[before]),
        ),
        reverse=True,
    )

    total_pairs = sum(len(pairs) for pairs in by_before.values())
    targets = {
        "train": total_pairs * train_ratio,
        "val": total_pairs * val_ratio,
        "test": total_pairs * (1 - train_ratio - val_ratio),
    }
    ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": 1 - train_ratio - val_ratio,
    }
    stratum_targets = {
        name: {stratum: total * ratios[name] for stratum, total in total_strata.items()}
        for name in SPLIT_NAMES
    }
    grouped_paths = {name: [] for name in SPLIT_NAMES}
    assigned_pairs = {name: 0 for name in SPLIT_NAMES}
    assigned_strata = {name: Counter() for name in SPLIT_NAMES}

    def assignment_delta(split_name: str, before: str) -> float:
        pair_count = len(by_before[before])
        pair_target = targets[split_name]
        current_pairs = assigned_pairs[split_name]
        delta = 4.0 * (
            ((current_pairs + pair_count - pair_target) ** 2 - (current_pairs - pair_target) ** 2)
            / (pair_target + 1.0)
        )
        for stratum, count in group_strata[before].items():
            target = stratum_targets[split_name][stratum]
            current = assigned_strata[split_name][stratum]
            delta += (
                (current + count - target) ** 2 - (current - target) ** 2
            ) / (target + 1.0)
        return delta

    for before in before_paths:
        split_name = min(SPLIT_NAMES, key=lambda name: assignment_delta(name, before))
        grouped_paths[split_name].append(before)
        assigned_pairs[split_name] += len(by_before[before])
        assigned_strata[split_name].update(group_strata[before])
    splits = {
        split_name: [pair for before in paths for pair in by_before[before]]
        for split_name, paths in grouped_paths.items()
    }
    group_counts = {name: len(paths) for name, paths in grouped_paths.items()}
    return splits, group_counts


def assert_before_images_are_disjoint(splits: Splits) -> None:
    """Raise if a before image occurs in more than one generated split."""
    owners: Dict[str, str] = {}
    for split_name, pairs in splits.items():
        for pair in pairs:
            before = pair["before"]
            previous = owners.setdefault(before, split_name)
            if previous != split_name:
                raise RuntimeError(
                    f"Before image '{before}' occurs in both '{previous}' and "
                    f"'{split_name}'"
                )


def main() -> None:
    print(f"Loading consumption labels from {source.CONSUMPTION_INDEX_PATH}...")
    consumption_lookup = source.load_consumption_lookup(source.CONSUMPTION_INDEX_PATH)
    print(f"  {len(consumption_lookup)} labeled images available\n")

    print("Indexing image files on disk...")
    unconsumed_index = source.index_images_by_stem(source.UNCONSUMED_IMAGES_DIR)
    consumed_index = source.index_images_by_stem(source.CONSUMED_IMAGES_DIR)
    print(
        f"  Found {len(unconsumed_index)} unconsumed images and "
        f"{len(consumed_index)} consumed images\n"
    )

    print(f"Loading manifests from {source.MANIFEST_DIR}...")
    groups, stats, _ = source.load_manifest_groups(
        source.MANIFEST_DIR,
        consumption_lookup,
        unconsumed_index,
        consumed_index,
    )
    if not groups:
        raise RuntimeError("No usable (before, after) pairs found - nothing to split.")

    splits, group_counts = split_groups(groups)
    assert_before_images_are_disjoint(splits)
    total_pairs = sum(len(pairs) for pairs in splits.values())
    total_groups = sum(group_counts.values())

    print(
        f"\nSplitting {total_pairs} pairs in {total_groups} unique before-image "
        "groups (targeting 70/15/15 pairs)..."
    )
    source.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, pairs in splits.items():
        out_path = source.OUTPUT_DIR / f"{split_name}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(pairs, fh, indent=2, ensure_ascii=False)
        group_pct = 100 * group_counts[split_name] / total_groups
        pair_pct = 100 * len(pairs) / total_pairs
        print(
            f"  {split_name:5s}: {group_counts[split_name]:3d} groups "
            f"({group_pct:5.1f}%), {len(pairs):4d} pairs ({pair_pct:5.1f}%) "
            f"-> {out_path}"
        )

    print(f"\nAll done! Manifests written to {source.OUTPUT_DIR.resolve()}")
    print("Verified: no before image occurs in more than one split.")
    print(f"Balanced {len(sum(split_stratum_counts(splits).values(), Counter()))} target strata best-effort across groups.")


if __name__ == "__main__":
    main()
