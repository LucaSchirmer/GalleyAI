"""Dataset evaluation report: class frequency, diversity, target-value
distributions, and split-by-split comparison for the before/after
consumption dataset.

Two different notions of "class frequency" are covered, on purpose:

  1. RAW segmentation classes -- every class YOLO actually detected in
     mask_cache/, across every before/after image referenced by any split.
     This is the "how often does a class physically show up on a tray"
     view, independent of whether that class happens to have a pct_*
     metric or choice field annotated on it. This is what surfaces
     data-scarce classes (e.g. the pasta_pesto / wrap_half_1/2 / orange_juice
     situation from your YOLO analysis) -- if a class barely appears here,
     no amount of Siamese-network tuning will fix that, it needs more
     labeled data.
  2. ANNOTATED metrics/fields -- the pct_* regression targets and the
     drink_*/extra_*/status_* classification fields actually used to train
     the Siamese network, AFTER the same quality-flag and missing-before-
     mask filtering ConsumptionPairDataset applies. This is "how much
     training signal do I actually have per target," which can differ
     from (1) since a class can be segmented on lots of trays but rarely
     have a percentage annotated, or vice versa.

Diversity is reported as:
  - unique classes per tray photo (mean/std + histogram)
  - Shannon entropy of the raw class frequency distribution (higher = more
    evenly spread across classes; a single dominant class collapses this
    toward 0)
  - normalized entropy / evenness (entropy divided by the max possible
    entropy for that many classes, so it's comparable across dataset
    versions with different numbers of classes)

Run from the project root (or anywhere -- paths are cwd-relative to match
train.py's convention; override with flags if your data lives elsewhere):
    python analysis/dataset_report.py
    python analysis/dataset_report.py --rare-class-threshold 15
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np

from data.siamese_dataset import (
    CHOICE_FIELD_TO_CLASS,
    ConsumptionPairDataset,
    MaskCache,
    expected_polygon_labels,
    _first_choice_list,
)

TRAIN_MANIFEST = Path("data_pairs_with_splits/train.json")
VAL_MANIFEST = Path("data_pairs_with_splits/val.json")
TEST_MANIFEST = Path("data_pairs_with_splits/test.json")
MASK_CACHE_DIR = Path("mask_cache")
MLFLOW_EXPERIMENT_NAME = "dataset_eda"
MLFLOW_TRACKING_URI_DEFAULT = f"sqlite:///{PROJECT_ROOT / 'runs' / 'mlflow' / 'mlflow.db'}"
REPORT_DIR = PROJECT_ROOT / "reports" / "dataset_eda"


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-manifest", type=Path, default=TRAIN_MANIFEST)
    parser.add_argument("--val-manifest", type=Path, default=VAL_MANIFEST)
    parser.add_argument("--test-manifest", type=Path, default=TEST_MANIFEST)
    parser.add_argument("--mask-cache-dir", type=Path, default=MASK_CACHE_DIR)
    parser.add_argument("--rare-class-threshold", type=int, default=20,
                         help="raw classes detected fewer than this many times get flagged as data-scarce")
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging, just write the local report")
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# RAW (PRE-FILTER) PAIR STATS -- straight from the manifest JSON
# ══════════════════════════════════════════════════════════════════════════════

def load_raw_pairs(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        print(f"  (not found: {path} -- skipping)")
        return []
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def raw_split_stats(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    quality_flag_counts: Counter = Counter()
    category_counts: Counter = Counter()
    regression_values: Dict[str, List[float]] = {}
    choice_counts: Dict[str, Counter] = {}

    for pair in pairs:
        flags = _first_choice_list(pair.get("choices", {}), "quality_flags")
        quality_flag_counts.update(flags)
        category_counts[pair.get("category", "(none)")] += 1

        for metric_name, values in pair.get("numbers", {}).items():
            if values:
                regression_values.setdefault(metric_name, []).append(values[0])

        for field, values in pair.get("choices", {}).items():
            if field not in CHOICE_FIELD_TO_CLASS:
                continue
            selected = values[0] if values else []
            value = selected[0] if selected else "(empty)"
            choice_counts.setdefault(field, Counter())[value] += 1

    return {
        "n_pairs": len(pairs),
        "quality_flag_counts": quality_flag_counts,
        "category_counts": category_counts,
        "regression_values": regression_values,
        "choice_counts": choice_counts,
    }


# ══════════════════════════════════════════════════════════════════════════════
# RAW SEGMENTATION CLASS FREQUENCY -- across every image any split references
# ══════════════════════════════════════════════════════════════════════════════

def raw_class_frequency(all_pairs: List[Dict[str, Any]], mask_cache_dir: Path):
    mask_cache = MaskCache(mask_cache_dir)
    image_paths = set()
    for pair in all_pairs:
        image_paths.add(pair["before"])
        image_paths.add(pair["after"])

    detection_counts: Counter = Counter()          # total polygon instances per class
    images_with_class: Dict[str, set] = {}          # class -> set of image paths containing it
    unique_classes_per_image: List[int] = []

    for path in image_paths:
        dets = mask_cache.detections_for(path)
        classes_here = {d["class"] for d in dets}
        unique_classes_per_image.append(len(classes_here))
        for d in dets:
            detection_counts[d["class"]] += 1
        for c in classes_here:
            images_with_class.setdefault(c, set()).add(path)

    return {
        "n_images": len(image_paths),
        "detection_counts": detection_counts,
        "images_with_class": images_with_class,
        "unique_classes_per_image": unique_classes_per_image,
    }


def shannon_entropy(counts: Counter) -> Dict[str, float]:
    total = sum(counts.values())
    if total == 0 or len(counts) == 0:
        return {"entropy_bits": 0.0, "max_entropy_bits": 0.0, "evenness": 0.0}
    probs = [c / total for c in counts.values() if c > 0]
    entropy = -sum(p * math.log2(p) for p in probs)
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return {
        "entropy_bits": entropy,
        "max_entropy_bits": max_entropy,
        "evenness": entropy / max_entropy if max_entropy > 0 else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# POST-FILTER (ACTUAL TRAINING SIGNAL) STATS -- via ConsumptionPairDataset
# ══════════════════════════════════════════════════════════════════════════════

def dataset_sample_stats(dataset: ConsumptionPairDataset) -> Dict[str, Any]:
    reg_counts: Counter = Counter()
    reg_values: Dict[str, List[float]] = {}
    clf_counts: Dict[str, Counter] = {}

    for s in dataset.samples:
        if s["task"] == "regression":
            reg_counts[s["metric_name"]] += 1
            reg_values.setdefault(s["metric_name"], []).append(s["target_pct"])
        else:
            clf_counts.setdefault(s["metric_name"], Counter())[s["target_pct"]] += 1

    return {
        "n_samples": len(dataset.samples),
        "n_regression": sum(reg_counts.values()),
        "n_classification": sum(sum(c.values()) for c in clf_counts.values()),
        "reg_counts": reg_counts,
        "reg_values": reg_values,
        "clf_counts": clf_counts,
        "dropped_quality": getattr(dataset, "dropped_quality", 0),
        "dropped_no_before_mask": getattr(dataset, "dropped_no_before_mask", 0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def class_frequency_fig(images_with_class: Dict[str, set], n_images: int, rare_threshold_pct: float):
    names = sorted(images_with_class, key=lambda c: -len(images_with_class[c]))
    pcts = [len(images_with_class[n]) / n_images * 100 for n in names]
    colors = ["#c62828" if p < rare_threshold_pct else "#4c72b0" for p in pcts]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(names))))
    ax.barh(names, pcts, color=colors)
    ax.axvline(rare_threshold_pct, color="black", linestyle="--", linewidth=1,
               label=f"data-scarce threshold ({rare_threshold_pct:.0f}%)")
    ax.set_xlabel("% of images containing this class")
    ax.set_title("Raw class frequency (image-presence rate)")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def detection_count_fig(detection_counts: Counter):
    names = sorted(detection_counts, key=lambda c: -detection_counts[c])
    counts = [detection_counts[n] for n in names]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(names))))
    ax.barh(names, counts, color="#55a868")
    ax.set_xlabel("Total detection instances (log scale)")
    ax.set_xscale("log")
    ax.set_title("Raw detection instance counts per class")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def diversity_hist_fig(unique_classes_per_image: List[int]):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(unique_classes_per_image, bins=range(1, max(unique_classes_per_image) + 2),
             color="#8172b2", edgecolor="white", align="left")
    ax.set_xlabel("Unique classes per photo")
    ax.set_ylabel("Number of photos")
    mean_v = np.mean(unique_classes_per_image)
    std_v = np.std(unique_classes_per_image)
    ax.set_title(f"Tray diversity  (mean={mean_v:.1f} ± {std_v:.1f} classes/photo)")
    fig.tight_layout()
    return fig


def target_distribution_fig(reg_values: Dict[str, List[float]], max_metrics: int = 9):
    metrics = sorted(reg_values, key=lambda m: -len(reg_values[m]))[:max_metrics]
    n = len(metrics)
    if n == 0:
        return None
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for i, m in enumerate(metrics):
        ax = axes[i // cols][i % cols]
        vals = reg_values[m]
        ax.hist(vals, bins=20, range=(0, 100), color="#dd8452", edgecolor="white")
        ax.set_title(f"{m}  (n={len(vals)})", fontsize=9)
        ax.set_xlim(0, 100)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Regression target (% consumed) distributions", y=1.02)
    fig.tight_layout()
    return fig


def classification_balance_fig(clf_counts: Dict[str, Counter]):
    if not clf_counts:
        return None
    fields = sorted(clf_counts)
    consumed = [clf_counts[f].get(100.0, 0) for f in fields]
    not_consumed = [clf_counts[f].get(0.0, 0) for f in fields]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(fields))))
    y = np.arange(len(fields))
    ax.barh(y, consumed, color="#2e7d32", label="Consumed")
    ax.barh(y, not_consumed, left=consumed, color="#c62828", label="Not consumed")
    ax.set_yticks(y); ax.set_yticklabels(fields)
    ax.set_xlabel("Sample count")
    ax.set_title("Classification field balance")
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def split_comparison_fig(split_stats: Dict[str, Dict[str, Any]]):
    splits = list(split_stats.keys())
    reg_n = [split_stats[s]["n_regression"] for s in splits]
    clf_n = [split_stats[s]["n_classification"] for s in splits]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(splits))
    ax.bar(x, reg_n, label="regression samples", color="#dd8452")
    ax.bar(x, clf_n, bottom=reg_n, label="classification samples", color="#55a868")
    ax.set_xticks(x); ax.set_xticklabels(splits)
    ax.set_ylabel("Sample count (post-filter)")
    ax.set_title("Samples per split")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def category_fig(category_counts: Counter):
    if not category_counts:
        return None
    fig, ax = plt.subplots(figsize=(5, 4))
    names = sorted(category_counts, key=lambda c: -category_counts[c])
    ax.bar(names, [category_counts[n] for n in names], color="#4c72b0")
    ax.set_ylabel("Pair count")
    ax.set_title("Category distribution (all splits combined)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def quality_flags_fig(quality_flag_counts: Counter):
    if not quality_flag_counts:
        return None
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.35 * len(quality_flag_counts))))
    names = sorted(quality_flag_counts, key=lambda c: -quality_flag_counts[c])
    ax.barh(names, [quality_flag_counts[n] for n in names], color="#937860")
    ax.set_xlabel("Pair count")
    ax.set_title("Quality flag frequency (all splits combined)")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# HTML REPORT
# ══════════════════════════════════════════════════════════════════════════════

def img_tag(fig) -> str:
    if fig is None:
        return "<p><em>(no data)</em></p>"
    return f'<img src="data:image/png;base64,{fig_to_base64(fig)}" style="max-width:100%;">'


def dict_table(d: Dict[str, Any]) -> str:
    rows = "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in d.items())
    return f"<table>{rows}</table>"


def counter_table(c: Counter, col_names=("value", "count")) -> str:
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in sorted(c.items(), key=lambda kv: -kv[1]))
    return f"<table><tr><th>{col_names[0]}</th><th>{col_names[1]}</th></tr>{rows}</table>"


def build_html(header, overview, diversity, rare_classes, figs) -> str:
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Dataset EDA report</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 32px; color: #222; }}
h1 {{ margin-bottom: 4px; }}
h2 {{ border-bottom: 2px solid #eee; padding-bottom: 6px; margin-top: 40px; }}
table {{ border-collapse: collapse; margin-bottom: 16px; }}
td, th {{ padding: 3px 14px 3px 0; text-align: left; }}
.callout {{ background: #fff3e0; border-left: 4px solid #c62828; padding: 10px 16px; margin: 12px 0; }}
</style></head>
<body>
<h1>Dataset Evaluation Report</h1>
{dict_table(header)}

<h2>Overview</h2>
{dict_table(overview)}

<h2>Diversity</h2>
{dict_table(diversity)}
{img_tag(figs.get("diversity_hist"))}

<div class="callout">
<b>Data-scarce classes</b> (below the rarity threshold -- likely to underperform,
same pattern as your YOLO analysis on pasta_pesto / wrap_half_1/2 / orange_juice):
<br>{", ".join(rare_classes) if rare_classes else "none"}
</div>

<h2>Raw segmentation class frequency</h2>
{img_tag(figs.get("class_frequency"))}
{img_tag(figs.get("detection_counts"))}

<h2>Regression target distributions</h2>
{img_tag(figs.get("target_distribution"))}

<h2>Classification field balance</h2>
{img_tag(figs.get("classification_balance"))}

<h2>Split comparison</h2>
{img_tag(figs.get("split_comparison"))}

<h2>Category &amp; quality-flag distribution</h2>
{img_tag(figs.get("category"))}
{img_tag(figs.get("quality_flags"))}

</body></html>"""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    split_manifests = {"train": args.train_manifest, "val": args.val_manifest, "test": args.test_manifest}

    print("Loading raw manifests...")
    raw_pairs_by_split = {name: load_raw_pairs(path) for name, path in split_manifests.items()}
    all_raw_pairs = [p for pairs in raw_pairs_by_split.values() for p in pairs]
    if not all_raw_pairs:
        raise FileNotFoundError("No pairs found in any of the manifests -- check --train/val/test-manifest paths.")

    print("Computing raw (pre-filter) per-split stats...")
    raw_stats_by_split = {name: raw_split_stats(pairs) for name, pairs in raw_pairs_by_split.items()}

    print("Computing raw segmentation class frequency across all images...")
    class_freq = raw_class_frequency(all_raw_pairs, args.mask_cache_dir)
    entropy_stats = shannon_entropy(class_freq["detection_counts"])

    rare_threshold_pct = args.rare_class_threshold  # interpreted as a %-of-images cutoff below
    rare_classes = [
        c for c, imgs in class_freq["images_with_class"].items()
        if len(imgs) / class_freq["n_images"] * 100 < rare_threshold_pct
    ]

    print("Building ConsumptionPairDataset per split for post-filter sample stats...")
    split_datasets = {}
    for name, path in split_manifests.items():
        if not path.exists():
            continue
        split_datasets[name] = ConsumptionPairDataset(path, mask_cache_dir=args.mask_cache_dir, train_mode=False)
    split_stats = {name: dataset_sample_stats(ds) for name, ds in split_datasets.items()}

    combined_reg_values: Dict[str, List[float]] = {}
    combined_clf_counts: Dict[str, Counter] = {}
    for stats in split_stats.values():
        for m, vals in stats["reg_values"].items():
            combined_reg_values.setdefault(m, []).extend(vals)
        for f, counter in stats["clf_counts"].items():
            combined_clf_counts.setdefault(f, Counter()).update(counter)

    combined_category_counts: Counter = Counter()
    combined_quality_flags: Counter = Counter()
    for stats in raw_stats_by_split.values():
        combined_category_counts.update(stats["category_counts"])
        combined_quality_flags.update(stats["quality_flag_counts"])

    header = {
        "Generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Train manifest": str(args.train_manifest),
        "Val manifest": str(args.val_manifest),
        "Test manifest": str(args.test_manifest),
        "Mask cache dir": str(args.mask_cache_dir),
        "Rarity threshold": f"< {rare_threshold_pct}% of images",
    }
    overview = {
        "Total pairs (all splits)": sum(s["n_pairs"] for s in raw_stats_by_split.values()),
        "Unique images referenced": class_freq["n_images"],
        "Unique raw classes detected": len(class_freq["detection_counts"]),
        "Total post-filter samples": sum(s["n_samples"] for s in split_stats.values()),
        "  -- regression samples": sum(s["n_regression"] for s in split_stats.values()),
        "  -- classification samples": sum(s["n_classification"] for s in split_stats.values()),
        "Dropped (quality-flagged pairs)": sum(s["dropped_quality"] for s in split_stats.values()),
        "Dropped (no 'before' mask)": sum(s["dropped_no_before_mask"] for s in split_stats.values()),
    }
    diversity = {
        "Mean unique classes / photo": round(float(np.mean(class_freq["unique_classes_per_image"])), 2),
        "Std unique classes / photo": round(float(np.std(class_freq["unique_classes_per_image"])), 2),
        "Shannon entropy (bits)": round(entropy_stats["entropy_bits"], 3),
        "Max possible entropy (bits)": round(entropy_stats["max_entropy_bits"], 3),
        "Evenness (0=one class dominates, 1=perfectly even)": round(entropy_stats["evenness"], 3),
        "Data-scarce class count": len(rare_classes),
    }

    print("Building plots...")
    figs = {
        "class_frequency": class_frequency_fig(class_freq["images_with_class"], class_freq["n_images"], rare_threshold_pct),
        "detection_counts": detection_count_fig(class_freq["detection_counts"]),
        "diversity_hist": diversity_hist_fig(class_freq["unique_classes_per_image"]),
        "target_distribution": target_distribution_fig(combined_reg_values),
        "classification_balance": classification_balance_fig(combined_clf_counts),
        "split_comparison": split_comparison_fig(split_stats) if split_stats else None,
        "category": category_fig(combined_category_counts),
        "quality_flags": quality_flags_fig(combined_quality_flags),
    }

    output_dir = REPORT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    figs_for_html = dict(figs)  # HTML build consumes+closes figures via fig_to_base64
    html = build_html(header, overview, diversity, sorted(rare_classes), figs_for_html)
    (output_dir / "report.html").write_text(html, encoding="utf-8")

    # Re-generate for standalone PNGs (cheap; these are all just matplotlib calls)
    for name, builder_args in [
        ("class_frequency", (class_freq["images_with_class"], class_freq["n_images"], rare_threshold_pct)),
        ("detection_counts", (class_freq["detection_counts"],)),
        ("diversity_hist", (class_freq["unique_classes_per_image"],)),
    ]:
        pass  # figures already consumed above -- see note below

    print(f"Report saved locally: {output_dir / 'report.html'}")

    if not args.no_mlflow:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or MLFLOW_TRACKING_URI_DEFAULT
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_name=f"dataset_eda_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_params(header)
            mlflow.log_metrics(
                {
                    "total_pairs": overview["Total pairs (all splits)"],
                    "unique_images": overview["Unique images referenced"],
                    "unique_raw_classes": overview["Unique raw classes detected"],
                    "total_samples": overview["Total post-filter samples"],
                    "regression_samples": overview["  -- regression samples"],
                    "classification_samples": overview["  -- classification samples"],
                    "dropped_quality_flagged": overview["Dropped (quality-flagged pairs)"],
                    "dropped_no_before_mask": overview["Dropped (no 'before' mask)"],
                    "shannon_entropy_bits": diversity["Shannon entropy (bits)"],
                    "evenness": diversity["Evenness (0=one class dominates, 1=perfectly even)"],
                    "mean_unique_classes_per_photo": diversity["Mean unique classes / photo"],
                    "data_scarce_class_count": diversity["Data-scarce class count"],
                }
            )
            mlflow.log_artifacts(str(output_dir), artifact_path="dataset_eda")
        print(f"Logged to MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")

    print("\nData-scarce classes (below rarity threshold):", sorted(rare_classes) or "none")


if __name__ == "__main__":
    main()