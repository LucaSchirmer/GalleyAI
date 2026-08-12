"""Builds a train/val/test split of (before, after) consumption image pairs.

Reads one manifest .json per main-course category from MANIFEST_DIR (each
entry pairs one "unconsumed" reference photo with zero or more "consumed"
photos of the same physical meal, taken at different points while it was
being eaten). Each (before, one after) pair becomes one training sample,
labeled with the pct_*/choices values already annotated for that "after"
photo in consumption_index.json (matched by filename stem).

Splitting is done at the MEAL-INSTANCE level (i.e. per manifest entry / per
"before" photo), never at the individual-pair level: all "after" photos of
the same meal are near-duplicates of each other (same tray, progressively
eaten) and must land in the same split, or the split would leak.

Output is three manifest files (train.json / val.json / test.json) that
reference the original image paths — no image files are copied.

Run:
    python scripts/prepare_data_siamese_nn.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

MANIFEST_DIR = Path("manifests")  # one .json per category, e.g. wrap.json, salad.json
CONSUMPTION_INDEX_PATH = Path("data_consumed/consumption_index.json")

# The manifest's "unconsumed"/"consumed" paths are the ORIGINAL raw capture
# layout (images/unconsumed/<cat>/...) and don't exist literally on disk
# anymore, nor do their filenames match (real files carry a Label-Studio hash
# prefix like "a5e6c40__..." that the manifest names don't have). So we never
# trust the manifest path directly — we only use its filename's STEM to look
# the real file up in these two folders.
UNCONSUMED_IMAGES_DIR = Path("data/images")
CONSUMED_IMAGES_DIR = Path("data_consumed/images")

OUTPUT_DIR = Path("data_pairs_with_splits")

SEED        = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test ratio is whatever remains (0.15)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CORE LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def suffix_stem(name: str) -> str:
    """Strips a Label-Studio-style hash prefix like 'a5e6c40__' from a filename stem."""
    stem = Path(name).stem
    if "__" in stem:
        return stem.split("__", 1)[1]
    return stem


def load_consumption_lookup(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Loads consumption_index.json and returns {stem: record} for every
    annotated "consumed" image, where record has "numbers" and "choices".
    """
    if not path.exists():
        raise FileNotFoundError(f"consumption index not found: {path.resolve()}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    lookup: Dict[str, Dict[str, Any]] = {}
    for stem, records in data.get("indexed_by_image_stem", {}).items():
        if records:
            lookup[stem] = records[0]
    return lookup


def index_images_by_stem(images_dir: Path) -> Dict[str, Path]:
    """
    Indexes every file in images_dir by its suffix_stem, so a manifest
    filename (without the hash prefix real files carry) can be resolved to
    its actual on-disk path.
    """
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir.resolve()}")
    indexed: Dict[str, Path] = {}
    for path in images_dir.iterdir():
        if path.is_file():
            indexed[suffix_stem(path.name)] = path
    return indexed


def load_manifest_groups(
    manifest_dir: Path,
    consumption_lookup: Dict[str, Dict[str, Any]],
    unconsumed_index: Dict[str, Path],
    consumed_index: Dict[str, Path],
):
    """
    Walks every *.json in manifest_dir (one file per category) and returns a
    list of "groups" — one group per meal instance (per manifest entry),
    where each group is a list of pair-dicts sharing the same "before" photo.
    Both "before" and "after" manifest paths are resolved to their real
    on-disk file via unconsumed_index/consumed_index (matched by stem, since
    the manifest's own paths/filenames don't exist as-is on disk). Groups or
    pairs that can't be resolved (missing file or missing label) are dropped.
    Returns (groups, global_stats, per_category_stats).
    """
    manifest_paths = sorted(manifest_dir.glob("*.json"))
    if not manifest_paths:
        raise FileNotFoundError(f"No .json manifests found in {manifest_dir.resolve()}")

    groups: List[List[Dict[str, Any]]] = []
    stats = {
        "categories": 0, "entries": 0, "no_consumed": 0,
        "missing_before_file": 0, "missing_after_file": 0,
        "missing_label": 0, "pairs": 0,
    }
    per_category_stats: Dict[str, Dict[str, int]] = {}
    seen_stem_category: Dict[str, str] = {}  # defensive collision check

    def blank_cat_stats() -> Dict[str, int]:
        return {
            "entries": 0, "no_consumed": 0, "missing_before_file": 0,
            "missing_after_file": 0, "missing_label": 0, "pairs": 0, "groups": 0,
        }

    for manifest_path in manifest_paths:
        category = manifest_path.stem
        stats["categories"] += 1
        cat_stats = per_category_stats.setdefault(category, blank_cat_stats())

        with manifest_path.open("r", encoding="utf-8") as fh:
            entries = json.load(fh)

        print(f"  {category}: {len(entries)} meal instance(s)")

        for entry in entries:
            stats["entries"] += 1
            cat_stats["entries"] += 1

            raw_unconsumed = entry.get("unconsumed")
            if not raw_unconsumed or raw_unconsumed == "invalid":
                stats["missing_before_file"] += 1
                cat_stats["missing_before_file"] += 1
                print(f"    ⚠ invalid or missing 'unconsumed' path in manifest — "
                      f"skipping whole meal instance")
                continue

            before_stem = suffix_stem(Path(raw_unconsumed).name)
            consumed_paths = entry.get("consumed", [])
            possible_elements = entry.get("possibleElements", [])

            if not consumed_paths:
                stats["no_consumed"] += 1
                cat_stats["no_consumed"] += 1
                print(f"    ⚠ no 'consumed' (after) photos listed for meal instance "
                      f"'{before_stem}' — skipping (nothing to pair/label yet)")
                continue

            before_actual = unconsumed_index.get(before_stem)
            if before_actual is None:
                stats["missing_before_file"] += 1
                cat_stats["missing_before_file"] += 1
                print(f"    ⚠ no real file found under {UNCONSUMED_IMAGES_DIR} for "
                      f"'before' stem '{before_stem}' — skipping whole meal instance")
                continue

            group: List[Dict[str, Any]] = []
            for after_path in consumed_paths:
                after_stem = suffix_stem(Path(after_path).name)

                # Defensive check: the same stem should never belong to two
                # different categories (would mean a real timestamp collision).
                prior_category = seen_stem_category.get(after_stem)
                if prior_category is not None and prior_category != category:
                    raise ValueError(
                        f"Stem '{after_stem}' appears in both category '{prior_category}' "
                        f"and '{category}'. This should not happen if timestamps are "
                        "date+time based — check for a genuine duplicate capture."
                    )
                seen_stem_category[after_stem] = category

                after_actual = consumed_index.get(after_stem)
                if after_actual is None:
                    stats["missing_after_file"] += 1
                    cat_stats["missing_after_file"] += 1
                    print(f"    ⚠ no real file found under {CONSUMED_IMAGES_DIR} for "
                          f"'after' stem '{after_stem}' — skipping this pair")
                    continue

                record = consumption_lookup.get(after_stem)
                if record is None:
                    stats["missing_label"] += 1
                    cat_stats["missing_label"] += 1
                    print(f"    ⚠ no consumption_index label found for stem "
                          f"'{after_stem}' — skipping this pair")
                    continue

                group.append(
                    {
                        "before": str(before_actual),
                        "after": str(after_actual),
                        "category": category,
                        "possible_elements": possible_elements,
                        "numbers": record.get("numbers", {}),
                        "choices": record.get("choices", {}),
                    }
                )

            if group:
                groups.append(group)
                stats["pairs"] += len(group)
                cat_stats["pairs"] += len(group)
                cat_stats["groups"] += 1

    return groups, stats, per_category_stats


def split_pairs(groups: List[List[Dict[str, Any]]]):
    """
    Flattens all groups into individual (before, after) pairs and splits
    at the PAIR level (not group level) — each pair is shuffled and
    assigned to train/val/test independently, since each "after" photo
    carries its own independent consumption label.

    Note: pairs from the same meal instance can end up in different
    splits, so val/test may see the same tray/background as train, just
    at a different consumption stage — keep that in mind when reading
    val/test metrics.
    """
    all_pairs = [pair for group in groups for pair in group]

    random.seed(SEED)
    shuffled = all_pairs[:]
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * TRAIN_RATIO)
    n_val = round(n * VAL_RATIO)

    splits = {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train : n_train + n_val],
        "test":  shuffled[n_train + n_val :],
    }
    return splits, n


def main():
    print(f"Loading consumption labels from {CONSUMPTION_INDEX_PATH}...")
    consumption_lookup = load_consumption_lookup(CONSUMPTION_INDEX_PATH)
    print(f"  {len(consumption_lookup)} labeled images available\n")

    print(f"Indexing image files on disk...")
    unconsumed_index = index_images_by_stem(UNCONSUMED_IMAGES_DIR)
    consumed_index = index_images_by_stem(CONSUMED_IMAGES_DIR)
    print(f"  Found {len(unconsumed_index)} unconsumed images and {len(consumed_index)} consumed images\n")

    print(f"Loading manifests from {MANIFEST_DIR}...")
    groups, stats, per_category_stats = load_manifest_groups(
        MANIFEST_DIR,
        consumption_lookup,
        unconsumed_index,
        consumed_index,
    )

    print(f"\nPer-category breakdown:")
    header = f"  {'category':22s} {'entries':>8s} {'no_after':>9s} {'no_before_file':>15s} {'no_after_file':>14s} {'no_label':>9s} {'pairs':>6s} {'groups':>7s}"
    print(header)
    for category, cat_stats in per_category_stats.items():
        print(
            f"  {category:22s} {cat_stats['entries']:>8d} {cat_stats['no_consumed']:>9d} "
            f"{cat_stats['missing_before_file']:>15d} {cat_stats['missing_after_file']:>14d} "
            f"{cat_stats['missing_label']:>9d} {cat_stats['pairs']:>6d} {cat_stats['groups']:>7d}"
        )

    print(f"\nManifest summary (all categories):")
    print(f"  categories        : {stats['categories']}")
    print(f"  meal instances    : {stats['entries']}")
    print(f"  skipped (no after): {stats['no_consumed']}")
    print(f"  skipped (no before file): {stats['missing_before_file']}")
    print(f"  skipped (no after file) : {stats['missing_after_file']}")
    print(f"  skipped (no label): {stats['missing_label']}")
    print(f"  usable pairs      : {stats['pairs']}")
    print(f"  usable groups     : {len(groups)}")

    if not groups:
        raise RuntimeError("No usable (before, after) pairs found — nothing to split.")

    print(f"\nSplitting {stats['pairs']} pairs (pair-level)...")
    splits, total_pairs = split_pairs(groups)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, flat_pairs in splits.items():
        out_path = OUTPUT_DIR / f"{split_name}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(flat_pairs, fh, indent=2, ensure_ascii=False)
        pct = 100 * len(flat_pairs) / total_pairs if total_pairs else 0
        print(f"  {split_name:5s}: {len(flat_pairs):4d} pairs ({pct:.1f}%) -> {out_path}")

    print(f"\nAll done! Manifests written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()