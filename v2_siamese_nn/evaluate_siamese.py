"""Evaluates a trained Siamese consumption-regressor checkpoint.

Two genuinely different task types get produced by the dataset:
  - "regression" samples (pct_* fields): a continuous 0-100% target.
  - "classification" samples (drinks, extras, cookie status): a binary
    consumed/not-consumed target, stored as 0 or 100 to reuse the same
    model output, but NOT a real percentage — averaging its error in with
    genuine regression samples produces a meaningless combined number
    (a classification miss on a 0/100 target inflates "MAE" hugely even
    though it might just be one wrong yes/no call).

So every metric and plot below is computed SEPARATELY per task type and
never pooled together. Regression gets MAE/RMSE/R²/Pearson r + the usual
plots. Classification gets accuracy/precision/recall/F1 + a confusion
matrix, thresholding at 50%.

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
from siamese_model import SiameseConsumptionNet, combined_prediction


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CHECKPOINT_PATH = Path("runs/siamese/resnet50_frozen_v3_twohead/best.pt")
TRAIN_MANIFEST = Path("data_pairs_with_splits/train.json")  # only used for the baseline
EVAL_MANIFEST = Path("data_pairs_with_splits/val.json")      # switch to test.json for a final check

OUTPUT_DIR = Path("runs/siamese/resnet50_frozen_v1/eval")
MLFLOW_EXPERIMENT_NAME = "siamese_consumption"
N_GALLERY = 8  # how many worst/best examples to show per prediction gallery

CLASSIFICATION_THRESHOLD = 50.0  # pct >= this -> predicted "Consumed"

BATCH_SIZE = 16
NUM_WORKERS = 2


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(model, dataset, device):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    preds = []

    model.eval()
    with torch.no_grad():
        for before, after, _, task in loader:
            before, after = before.to(device), after.to(device)
            is_regression = torch.tensor([t == "regression" for t in task], device=device)
            reg_out, clf_logit = model(before, after)
            pred = combined_prediction(reg_out, clf_logit, is_regression)
            preds.append(pred.cpu().numpy())

    preds = np.concatenate(preds) * 100
    targets = np.array([s["target_pct"] for s in dataset.samples], dtype=np.float32)
    return preds, targets


def split_by_task(dataset) -> dict:
    """Returns {"regression": bool_mask, "classification": bool_mask}."""
    tasks = np.array([s["task"] for s in dataset.samples])
    return {
        "regression": tasks == "regression",
        "classification": tasks == "classification",
    }


# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION METRICS
# ══════════════════════════════════════════════════════════════════════════════

def regression_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    if len(preds) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"),
                "pearson_r": float("nan"), "n": 0}

    errors = preds - targets
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors ** 2).mean())
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    if len(preds) > 1 and np.std(preds) > 0 and np.std(targets) > 0:
        corr = np.corrcoef(preds, targets)[0, 1]
    else:
        corr = float("nan")

    return {
        "mae": float(mae), "rmse": float(rmse), "r2": float(r2),
        "pearson_r": float(corr), "n": int(len(preds)),
    }


def regression_breakdown_by(preds, targets, keys) -> dict:
    result = {}
    keys = np.array(keys)
    for key in sorted(set(keys)):
        mask = keys == key
        result[key] = regression_metrics(preds[mask], targets[mask])
    return result


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION METRICS
# ══════════════════════════════════════════════════════════════════════════════

def classification_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    if len(preds) == 0:
        return {"accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "tp": 0, "fp": 0, "tn": 0, "fn": 0, "n": 0}

    pred_label = preds >= CLASSIFICATION_THRESHOLD
    true_label = targets >= CLASSIFICATION_THRESHOLD

    tp = int(np.sum(pred_label & true_label))
    fp = int(np.sum(pred_label & ~true_label))
    tn = int(np.sum(~pred_label & ~true_label))
    fn = int(np.sum(~pred_label & true_label))

    accuracy = (tp + tn) / len(preds)
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision == precision and recall == recall and (precision + recall) > 0
          else float("nan"))

    return {
        "accuracy": float(accuracy), "precision": float(precision), "recall": float(recall),
        "f1": float(f1), "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": int(len(preds)),
    }


def classification_breakdown_by(preds, targets, keys) -> dict:
    result = {}
    keys = np.array(keys)
    for key in sorted(set(keys)):
        mask = keys == key
        result[key] = classification_metrics(preds[mask], targets[mask])
    return result


# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_scatter(preds, targets, out_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(targets, preds, alpha=0.5, s=20)
    lims = [0, 100]
    plt.plot(lims, lims, "r--", label="perfect prediction")
    plt.xlabel("True consumption (%)")
    plt.ylabel("Predicted consumption (%)")
    plt.title("Predicted vs True — regression fields only")
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
    plt.title("Residuals vs True (regression only) — look for bias by consumption level")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_error_histogram(preds, targets, out_path):
    abs_errors = np.abs(preds - targets)
    plt.figure(figsize=(6, 4))
    plt.hist(abs_errors, bins=20)
    plt.xlabel("Absolute error (percentage points)")
    plt.ylabel("Count")
    plt.title("Absolute Error Distribution (regression only)")
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


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_accuracy_bar(breakdown: dict, title: str, out_path):
    labels = list(breakdown.keys())
    accs = [breakdown[k]["accuracy"] * 100 for k in labels]
    ns = [breakdown[k]["n"] for k in labels]
    plt.figure(figsize=(max(6, len(labels) * 0.8), 4))
    bars = plt.bar(labels, accs, color="seagreen")
    for bar, n in zip(bars, ns):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={n}",
                  ha="center", va="bottom", fontsize=8)
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(metrics: dict, out_path):
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    plt.figure(figsize=(4, 4))
    plt.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=14)
    plt.xticks([0, 1], ["Not consumed", "Consumed"])
    plt.yticks([0, 1], ["Not consumed", "Consumed"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Classification Confusion Matrix (n={metrics['n']})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION GALLERIES (mask-overlay, shared by both task types)
# ══════════════════════════════════════════════════════════════════════════════

MASK_OVERLAY_COLOR = (255, 60, 0)
MASK_OVERLAY_ALPHA = 0.45


def overlay_mask(ax, image: Image.Image, mask_arr: np.ndarray):
    ax.imshow(image)
    rgba = np.zeros((*mask_arr.shape, 4))
    rgba[..., 0] = MASK_OVERLAY_COLOR[0] / 255
    rgba[..., 1] = MASK_OVERLAY_COLOR[1] / 255
    rgba[..., 2] = MASK_OVERLAY_COLOR[2] / 255
    rgba[..., 3] = mask_arr * MASK_OVERLAY_ALPHA
    ax.imshow(rgba)


def plot_prediction_gallery(indices, preds, targets, dataset, mask_cache: MaskCache, title, out_path, is_classification=False):
    n = len(indices)
    if n == 0:
        return
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

        if is_classification:
            pred_label = "Consumed" if preds[idx] >= CLASSIFICATION_THRESHOLD else "Not consumed"
            true_label = "Consumed" if targets[idx] >= CLASSIFICATION_THRESHOLD else "Not consumed"
            caption = f"after — pred={pred_label} ({preds[idx]:.0f}%)  true={true_label}"
        else:
            caption = f"after — pred={preds[idx]:.1f}%  true={targets[idx]:.1f}%"

        overlay_mask(axes[row, 1], after_img, after_mask)
        axes[row, 1].set_title(caption, fontsize=9)
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
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))

    print(f"Loading eval set from {EVAL_MANIFEST}...")
    eval_ds = ConsumptionPairDataset(EVAL_MANIFEST, train_mode=False)
    preds, targets = run_inference(model, eval_ds, device)
    metric_names = [s["metric_name"] for s in eval_ds.samples]
    task_masks = split_by_task(eval_ds)

    print(f"Loading train set from {TRAIN_MANIFEST} (for baseline comparison only)...")
    train_ds = ConsumptionPairDataset(TRAIN_MANIFEST, train_mode=False)
    train_task_masks = split_by_task(train_ds)
    train_targets_all = np.array([s["target_pct"] for s in train_ds.samples], dtype=np.float32)

    # ---------- REGRESSION ----------
    reg_mask = task_masks["regression"]
    reg_preds, reg_targets = preds[reg_mask], targets[reg_mask]
    reg_metric_names = [m for m, keep in zip(metric_names, reg_mask) if keep]

    train_reg_targets = train_targets_all[train_task_masks["regression"]]
    reg_baseline_pred = np.full_like(reg_targets, train_reg_targets.mean()) if len(reg_targets) else reg_targets

    reg_overall = regression_metrics(reg_preds, reg_targets)
    reg_baseline = regression_metrics(reg_baseline_pred, reg_targets)
    reg_by_metric = regression_breakdown_by(reg_preds, reg_targets, reg_metric_names)

    print("\n=== REGRESSION (pct_* fields only) ===")
    print(f"  n            : {reg_overall['n']}")
    print(f"  MAE          : {reg_overall['mae']:.2f} pct points")
    print(f"  RMSE         : {reg_overall['rmse']:.2f} pct points")
    print(f"  R^2          : {reg_overall['r2']:.3f}")
    print(f"  Pearson r    : {reg_overall['pearson_r']:.3f}")
    print(f"  Baseline MAE : {reg_baseline['mae']:.2f} pct points "
          f"(train mean {train_reg_targets.mean():.1f}%)" if len(train_reg_targets) else "  Baseline MAE : n/a")
    print("\n  by metric:")
    for metric, m in reg_by_metric.items():
        print(f"    {metric:24s} n={m['n']:3d}  MAE={m['mae']:.2f}  R^2={m['r2']:.3f}")

    # ---------- CLASSIFICATION ----------
    clf_mask = task_masks["classification"]
    clf_preds, clf_targets = preds[clf_mask], targets[clf_mask]
    clf_metric_names = [m for m, keep in zip(metric_names, clf_mask) if keep]

    clf_overall = classification_metrics(clf_preds, clf_targets)
    clf_by_metric = classification_breakdown_by(clf_preds, clf_targets, clf_metric_names)

    print("\n=== CLASSIFICATION (drinks/extras/cookie, consumed vs not) ===")
    print(f"  n            : {clf_overall['n']}")
    print(f"  Accuracy     : {clf_overall['accuracy'] * 100:.1f}%")
    print(f"  Precision    : {clf_overall['precision']:.3f}")
    print(f"  Recall       : {clf_overall['recall']:.3f}")
    print(f"  F1           : {clf_overall['f1']:.3f}")
    print(f"  Confusion    : TP={clf_overall['tp']} FP={clf_overall['fp']} "
          f"TN={clf_overall['tn']} FN={clf_overall['fn']}")
    print("\n  by field:")
    for metric, m in clf_by_metric.items():
        print(f"    {metric:20s} n={m['n']:3d}  accuracy={m['accuracy'] * 100:.1f}%  f1={m['f1']:.3f}")

    # --- regression plots ---
    if reg_overall["n"] > 0:
        plot_scatter(reg_preds, reg_targets, OUTPUT_DIR / "scatter_pred_vs_true.png")
        plot_residuals(reg_preds, reg_targets, OUTPUT_DIR / "residuals_vs_true.png")
        plot_error_histogram(reg_preds, reg_targets, OUTPUT_DIR / "error_histogram.png")
        plot_mae_bar(reg_by_metric, "MAE by Food Class (regression fields only)",
                     OUTPUT_DIR / "mae_by_metric.png")

    # --- classification plots ---
    if clf_overall["n"] > 0:
        plot_accuracy_bar(clf_by_metric, "Accuracy by Field (classification)",
                           OUTPUT_DIR / "accuracy_by_field.png")
        plot_confusion_matrix(clf_overall, OUTPUT_DIR / "confusion_matrix.png")

    # --- prediction galleries, kept separate per task type ---
    mask_cache = MaskCache()

    if reg_overall["n"] > 0:
        reg_indices_in_full = np.where(reg_mask)[0]
        reg_abs_errors = np.abs(reg_preds - reg_targets)
        worst = reg_indices_in_full[np.argsort(-reg_abs_errors)[:N_GALLERY]]
        best = reg_indices_in_full[np.argsort(reg_abs_errors)[:N_GALLERY]]
        plot_prediction_gallery(worst, preds, targets, eval_ds, mask_cache,
                                 "Worst Regression Predictions", OUTPUT_DIR / "worst_predictions_regression.png")
        plot_prediction_gallery(best, preds, targets, eval_ds, mask_cache,
                                 "Best Regression Predictions", OUTPUT_DIR / "best_predictions_regression.png")

    if clf_overall["n"] > 0:
        clf_indices_in_full = np.where(clf_mask)[0]
        # "worst" for classification = misclassified, ranked by confidence (how wrong/confident it was)
        clf_pred_label = preds[clf_indices_in_full] >= CLASSIFICATION_THRESHOLD
        clf_true_label = targets[clf_indices_in_full] >= CLASSIFICATION_THRESHOLD
        misclassified = clf_indices_in_full[clf_pred_label != clf_true_label]
        correct = clf_indices_in_full[clf_pred_label == clf_true_label]
        worst_clf = misclassified[:N_GALLERY]
        best_clf = correct[:N_GALLERY]
        plot_prediction_gallery(worst_clf, preds, targets, eval_ds, mask_cache,
                                 "Misclassified Examples", OUTPUT_DIR / "worst_predictions_classification.png",
                                 is_classification=True)
        plot_prediction_gallery(best_clf, preds, targets, eval_ds, mask_cache,
                                 "Correctly Classified Examples", OUTPUT_DIR / "best_predictions_classification.png",
                                 is_classification=True)

    summary = {
        "regression": {
            "overall": reg_overall,
            "baseline_mean_predictor": reg_baseline,
            "by_metric": reg_by_metric,
        },
        "classification": {
            "overall": clf_overall,
            "by_metric": clf_by_metric,
        },
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
                "eval_regression_mae": reg_overall["mae"],
                "eval_regression_rmse": reg_overall["rmse"],
                "eval_regression_r2": reg_overall["r2"],
                "eval_regression_pearson_r": reg_overall["pearson_r"],
                "eval_regression_baseline_mae": reg_baseline["mae"],
                "eval_classification_accuracy": clf_overall["accuracy"],
                "eval_classification_precision": clf_overall["precision"],
                "eval_classification_recall": clf_overall["recall"],
                "eval_classification_f1": clf_overall["f1"],
            }
        )
        for png in OUTPUT_DIR.glob("*.png"):
            mlflow.log_artifact(str(png))
        mlflow.log_artifact(str(OUTPUT_DIR / "summary.json"))

    print(f"\nAll plots + summary.json written to {OUTPUT_DIR.resolve()}")
    print(f"Also logged to MLflow under experiment '{MLFLOW_EXPERIMENT_NAME}' as a new run.")


if __name__ == "__main__":
    main()