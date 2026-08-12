"""Evaluates a trained Siamese consumption-regressor checkpoint.

Produces the regression-model equivalent of what Ultralytics gives you for
free on the YOLO side: a predicted-vs-true scatter (the confusion-matrix
analogue), residual and error-distribution plots, per-category and
per-metric breakdowns, a naive baseline comparison, and worst/best
prediction galleries with the actual before/after photos. Everything is
also logged to MLflow as a new run under the same experiment as training,
so it's browsable next to the training curves.

Run:
    python scripts/evaluate_siamese.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from siamese_dataset import ConsumptionPairDataset, MaskCache, build_mask_array
from siamese_model import SiameseConsumptionNet


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CHECKPOINT_PATH = Path("runs/siamese/resnet50_frozen_v1/best.pt")
TRAIN_MANIFEST = Path("data_pairs_with_splits/train.json")  # only used for the baseline mean
EVAL_MANIFEST = Path("data_pairs_with_splits/val.json")      # switch to test.json for a final check

OUTPUT_DIR = Path("runs/siamese/resnet50_frozen_v1/eval")
MLFLOW_EXPERIMENT_NAME = "siamese_consumption"
N_GALLERY = 8  # how many worst/best examples to show in the prediction galleries

BATCH_SIZE = 16
NUM_WORKERS = 2


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(model, dataset, device):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    preds, targets = [], []

    model.eval()
    with torch.no_grad():
        for before, after, target in loader:
            before, after = before.to(device), after.to(device)
            pred = model(before, after).cpu().numpy()
            preds.append(pred)
            targets.append(target.numpy())

    preds = np.concatenate(preds) * 100
    targets = np.concatenate(targets) * 100

    categories = [s["metric_name"] for s in dataset.samples]  # per-sample metric name, e.g. pct_rice
    before_paths = [s["before"] for s in dataset.samples]
    after_paths = [s["after"] for s in dataset.samples]

    return preds, targets, categories, before_paths, after_paths


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def regression_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    errors = preds - targets
    abs_errors = np.abs(errors)
    mae = abs_errors.mean()
    rmse = np.sqrt((errors ** 2).mean())
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # corrcoef is undefined (and warns) when either array has zero variance —
    # common in small per-category/per-metric subgroups. Guard explicitly.
    if len(preds) > 1 and np.std(preds) > 0 and np.std(targets) > 0:
        corr = np.corrcoef(preds, targets)[0, 1]
    else:
        corr = float("nan")

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "pearson_r": float(corr),
        "n": int(len(preds)),
    }


def breakdown_by(preds: np.ndarray, targets: np.ndarray, keys: list) -> dict:
    result = {}
    for key in sorted(set(keys)):
        mask = np.array([k == key for k in keys])
        result[key] = regression_metrics(preds[mask], targets[mask])
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_scatter(preds, targets, out_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(targets, preds, alpha=0.5, s=20)
    lims = [0, 100]
    plt.plot(lims, lims, "r--", label="perfect prediction")
    plt.xlabel("True consumption (%)")
    plt.ylabel("Predicted consumption (%)")
    plt.title("Predicted vs True (the regression 'confusion matrix')")
    plt.xlim(lims); plt.ylim(lims)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_residuals(preds, targets, out_path):
    residuals = preds - targets
    plt.figure(figsize=(6, 4))
    plt.scatter(targets, residuals, alpha=0.5, s=20)
    plt.axhline(0, color="r", linestyle="--")
    plt.xlabel("True consumption (%)")
    plt.ylabel("Prediction error (pred - true)")
    plt.title("Residuals vs True — look for systematic bias by consumption level")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_error_histogram(preds, targets, out_path):
    abs_errors = np.abs(preds - targets)
    plt.figure(figsize=(6, 4))
    plt.hist(abs_errors, bins=20)
    plt.xlabel("Absolute error (percentage points)")
    plt.ylabel("Count")
    plt.title("Absolute Error Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_mae_bar(breakdown: dict, title: str, out_path):
    labels = list(breakdown.keys())
    maes = [breakdown[k]["mae"] for k in labels]
    ns = [breakdown[k]["n"] for k in labels]
    plt.figure(figsize=(max(6, len(labels) * 0.8), 4))
    bars = plt.bar(labels, maes)
    for bar, n in zip(bars, ns):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={n}",
                  ha="center", va="bottom", fontsize=8)
    plt.ylabel("MAE (percentage points)")
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


MASK_OVERLAY_COLOR = (255, 60, 0)  # RGB, drawn semi-transparent over the model's actual mask region
MASK_OVERLAY_ALPHA = 0.45


def overlay_mask(ax, image: Image.Image, mask_arr: np.ndarray):
    ax.imshow(image)
    rgba = np.zeros((*mask_arr.shape, 4))
    rgba[..., 0] = MASK_OVERLAY_COLOR[0] / 255
    rgba[..., 1] = MASK_OVERLAY_COLOR[1] / 255
    rgba[..., 2] = MASK_OVERLAY_COLOR[2] / 255
    rgba[..., 3] = mask_arr * MASK_OVERLAY_ALPHA
    ax.imshow(rgba)


def plot_prediction_gallery(indices, preds, targets, dataset, mask_cache: MaskCache, title, out_path):
    """
    For each sample index, shows the before/after photo with the ACTUAL mask
    region the model was given highlighted (the region driving this specific
    metric's prediction), plus the metric name in the caption — without
    this, pred/true numbers are meaningless since you can't tell which food
    item, or which region, they refer to.
    """
    n = len(indices)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for row, idx in enumerate(indices):
        sample = dataset.samples[idx]
        classes = sample["classes"]
        img_size = dataset.img_size

        before_img = Image.open(sample["before"]).convert("RGB").resize((img_size, img_size))
        after_img = Image.open(sample["after"]).convert("RGB").resize((img_size, img_size))

        before_mask = build_mask_array(mask_cache.detections_for(sample["before"]), classes, img_size)
        after_mask = build_mask_array(mask_cache.detections_for(sample["after"]), classes, img_size)

        overlay_mask(axes[row, 0], before_img, before_mask)
        axes[row, 0].set_title(f"before — {sample['metric_name']}", fontsize=9)
        axes[row, 0].axis("off")

        overlay_mask(axes[row, 1], after_img, after_mask)
        axes[row, 1].set_title(
            f"after — pred={preds[idx]:.1f}%  true={targets[idx]:.1f}%", fontsize=9
        )
        axes[row, 1].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
    model = SiameseConsumptionNet(freeze_backbone=True).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    print(f"Loading eval set from {EVAL_MANIFEST}...")
    eval_ds = ConsumptionPairDataset(EVAL_MANIFEST, train_mode=False)
    preds, targets, metric_names, before_paths, after_paths = run_inference(model, eval_ds, device)

    print(f"Loading train set from {TRAIN_MANIFEST} (for baseline comparison only)...")
    train_ds = ConsumptionPairDataset(TRAIN_MANIFEST, train_mode=False)
    train_targets = np.array([s["target_pct"] for s in train_ds.samples], dtype=np.float32)
    baseline_pred = np.full_like(targets, train_targets.mean())

    overall = regression_metrics(preds, targets)
    baseline = regression_metrics(baseline_pred, targets)
    by_category = breakdown_by(preds, targets, [s["category"] for s in eval_ds.samples])
    by_metric = breakdown_by(preds, targets, metric_names)

    print("\n=== Overall ===")
    print(f"  n            : {overall['n']}")
    print(f"  MAE          : {overall['mae']:.2f} pct points")
    print(f"  RMSE         : {overall['rmse']:.2f} pct points")
    print(f"  R^2          : {overall['r2']:.3f}")
    print(f"  Pearson r    : {overall['pearson_r']:.3f}")
    print(f"  Baseline MAE : {baseline['mae']:.2f} pct points "
          f"(always predicting the train-set mean, {train_targets.mean():.1f}%)")
    print(f"  -> model beats baseline by {baseline['mae'] - overall['mae']:.2f} pct points"
          if overall['mae'] < baseline['mae'] else
          "  -> WARNING: model does not beat the naive mean-prediction baseline")

    print("\n=== By category ===")
    for cat, m in by_category.items():
        print(f"  {cat:20s} n={m['n']:3d}  MAE={m['mae']:.2f}  R^2={m['r2']:.3f}")

    print("\n=== By metric (food class) ===")
    for metric, m in by_metric.items():
        print(f"  {metric:24s} n={m['n']:3d}  MAE={m['mae']:.2f}  R^2={m['r2']:.3f}")

    # --- plots ---
    plot_scatter(preds, targets, OUTPUT_DIR / "scatter_pred_vs_true.png")
    plot_residuals(preds, targets, OUTPUT_DIR / "residuals_vs_true.png")
    plot_error_histogram(preds, targets, OUTPUT_DIR / "error_histogram.png")
    plot_mae_bar(by_category, "MAE by Category", OUTPUT_DIR / "mae_by_category.png")
    plot_mae_bar(by_metric, "MAE by Food Class (metric)", OUTPUT_DIR / "mae_by_metric.png")

    abs_errors = np.abs(preds - targets)
    worst_idx = np.argsort(-abs_errors)[:N_GALLERY]
    best_idx = np.argsort(abs_errors)[:N_GALLERY]
    mask_cache = MaskCache()
    plot_prediction_gallery(worst_idx, preds, targets, eval_ds, mask_cache,
                             "Worst Predictions", OUTPUT_DIR / "worst_predictions.png")
    plot_prediction_gallery(best_idx, preds, targets, eval_ds, mask_cache,
                             "Best Predictions", OUTPUT_DIR / "best_predictions.png")

    summary = {
        "overall": overall,
        "baseline_mean_predictor": baseline,
        "by_category": by_category,
        "by_metric": by_metric,
    }
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # --- MLflow logging ---
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or "runs/mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"eval_{EVAL_MANIFEST.stem}_{CHECKPOINT_PATH.parent.name}"):
        mlflow.log_metrics(
            {
                "eval_mae": overall["mae"],
                "eval_rmse": overall["rmse"],
                "eval_r2": overall["r2"],
                "eval_pearson_r": overall["pearson_r"],
                "eval_baseline_mae": baseline["mae"],
            }
        )
        for png in OUTPUT_DIR.glob("*.png"):
            mlflow.log_artifact(str(png))
        mlflow.log_artifact(str(OUTPUT_DIR / "summary.json"))

    print(f"\nAll plots + summary.json written to {OUTPUT_DIR.resolve()}")
    print("Also logged to MLflow under experiment "
          f"'{MLFLOW_EXPERIMENT_NAME}' as a new run.")


if __name__ == "__main__":
    main()
