"""Runs a trained (backbone, head) checkpoint on a manifest split -- normally
the test set -- and produces a full visual report: regression scatter/
residual/per-metric-MAE plots, a classification confusion matrix and
per-field accuracy breakdown, and a before/after sample gallery with
multi-class segmentation-mask overlays (one color per class) annotated with
ground truth vs. predicted values.

Everything is saved locally under reports/<run_name>_<split>/ AND logged to
MLflow as a new run in the same experiment as training, tagged with a link
back to the original training run, so the report metrics/plots sit right
next to that run's training curves for comparison.

Run from the project root (or anywhere -- see train.py's PROJECT_ROOT note):
    python training/generate_report.py --backbone convnext_tiny --head cosine
    python training/generate_report.py --backbone resnet50 --head mlp --split val
    python training/generate_report.py --backbone resnet50 --head mlp \
        --checkpoint runs/siamese/resnet50_mlp/best.pt --gallery-samples 20
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")  # headless -- no display available when this runs on a server/WSL
import matplotlib as mpl
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import confusion_matrix

from data.siamese_dataset import ConsumptionPairDataset, build_mask_array
from models.siamese_net import SiameseConsumptionNet, list_regression_heads
from models.backbones import list_backbones

TEST_MANIFEST = Path("data_pairs_with_splits/test.json")
MASK_CACHE_DIR = Path("mask_cache")
MLFLOW_EXPERIMENT_NAME = "siamese_consumption"
MLFLOW_TRACKING_URI_DEFAULT = f"sqlite:///{PROJECT_ROOT / 'runs' / 'mlflow' / 'mlflow.db'}"
REPORT_ROOT = PROJECT_ROOT / "reports"

CORRECT_COLOR = "#2e7d32"   # green
WRONG_COLOR = "#c62828"     # red
MASK_ALPHA = 110             # 0-255, overlay opacity


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", required=True, choices=list_backbones())
    parser.add_argument("--head", required=True, choices=list_regression_heads())
    parser.add_argument("--checkpoint", type=Path, default=None, help="defaults to runs/siamese/<backbone>_<head>/best.pt")
    parser.add_argument("--split", default="test", help="label used in the report header, e.g. 'test' or 'val'")
    parser.add_argument("--manifest", type=Path, default=TEST_MANIFEST, help="manifest json for --split")
    parser.add_argument("--mask-cache-dir", type=Path, default=MASK_CACHE_DIR)
    parser.add_argument("--run-name", default=None, help="defaults to '<backbone>_<head>' -- must match the training run to link back to it")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gallery-samples", type=int, default=12, help="number of before/after tray pairs to render in the sample gallery")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def load_model(backbone: str, head: str, checkpoint_path: Path, device) -> SiameseConsumptionNet:
    model = SiameseConsumptionNet(backbone_name=backbone, head_name=head, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def collect_predictions(model, dataset: ConsumptionPairDataset, device, batch_size: int) -> List[Dict[str, Any]]:
    """Runs inference over every sample and returns one record per sample
    with everything the report needs -- predictions plus the metadata
    (image paths, classes, task) that dataset.__getitem__ doesn't expose."""
    records: List[Dict[str, Any]] = []
    samples = dataset.samples

    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            before = torch.stack([dataset._load_tensor(s["before"], s["classes"]) for s in batch]).to(device)
            after = torch.stack([dataset._load_tensor(s["after"], s["classes"]) for s in batch]).to(device)

            reg_out, clf_logit = model(before, after)
            reg_pred = torch.clamp(reg_out, 0.0, 1.0).cpu().numpy()
            clf_pred = torch.sigmoid(clf_logit).cpu().numpy()

            for i, s in enumerate(batch):
                pred = reg_pred[i] if s["task"] == "regression" else clf_pred[i]
                records.append(
                    {
                        "before": s["before"],
                        "after": s["after"],
                        "category": s.get("category"),
                        "classes": s["classes"],
                        "metric_name": s["metric_name"],
                        "task": s["task"],
                        "target_pct": s["target_pct"],  # already 0-100
                        "pred_pct": float(pred) * 100.0,
                    }
                )
    return records


# ══════════════════════════════════════════════════════════════════════════════
# MASK OVERLAY (multi-class, one color per class)
# ══════════════════════════════════════════════════════════════════════════════

def build_class_color_map(class_names: List[str]) -> Dict[str, Tuple[int, int, int]]:
    palette = mpl.colormaps["tab20"].resampled(max(len(class_names), 1))
    colors = {}
    for i, name in enumerate(sorted(class_names)):
        r, g, b, _ = palette(i)
        colors[name] = (int(r * 255), int(g * 255), int(b * 255))
    return colors


def render_overlay(image_path: str, detections: List[Dict[str, Any]], color_map: Dict[str, Tuple[int, int, int]], img_size: int = 384) -> Image.Image:
    """Draws every detection for this image, colored by class, blended over
    the photo -- not just the class(es) relevant to one sample's target, so
    the whole tray's segmentation is visible at once."""
    base = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for det in detections:
        color = color_map.get(det["class"], (255, 255, 255))
        points = [(x * img_size, y * img_size) for x, y in det["polygon"]]
        if len(points) >= 3:
            draw.polygon(points, fill=(*color, MASK_ALPHA), outline=(*color, 255))
    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def regression_scatter_fig(records: List[Dict[str, Any]]):
    reg = [r for r in records if r["task"] == "regression"]
    if not reg:
        return None
    metrics = sorted({r["metric_name"] for r in reg})
    cmap = mpl.colormaps["tab10"].resampled(max(len(metrics), 1))

    fig, ax = plt.subplots(figsize=(6, 6))
    for i, m in enumerate(metrics):
        pts = [r for r in reg if r["metric_name"] == m]
        ax.scatter([p["target_pct"] for p in pts], [p["pred_pct"] for p in pts],
                   label=m, color=cmap(i), alpha=0.7, s=28, edgecolors="none")
    ax.plot([0, 100], [0, 100], "k--", linewidth=1, label="perfect prediction")
    ax.set_xlabel("Ground truth % consumed")
    ax.set_ylabel("Predicted % consumed")
    ax.set_title("Regression: predicted vs. ground truth")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    return fig


def residual_hist_fig(records: List[Dict[str, Any]]):
    reg = [r for r in records if r["task"] == "regression"]
    if not reg:
        return None
    errors = np.array([r["pred_pct"] - r["target_pct"] for r in reg])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(errors, bins=25, color="#4c72b0", edgecolor="white")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Prediction error (predicted - ground truth), pct points")
    ax.set_ylabel("Count")
    ax.set_title(f"Residuals  (MAE={np.abs(errors).mean():.1f}%,  RMSE={np.sqrt((errors ** 2).mean()):.1f}%)")
    fig.tight_layout()
    return fig


def mae_per_metric_fig(records: List[Dict[str, Any]]):
    reg = [r for r in records if r["task"] == "regression"]
    if not reg:
        return None
    by_metric: Dict[str, List[float]] = {}
    for r in reg:
        by_metric.setdefault(r["metric_name"], []).append(abs(r["pred_pct"] - r["target_pct"]))
    names = sorted(by_metric, key=lambda k: -np.mean(by_metric[k]))
    maes = [np.mean(by_metric[n]) for n in names]
    counts = [len(by_metric[n]) for n in names]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(names))))
    bars = ax.barh(names, maes, color="#dd8452")
    for bar, n in zip(bars, counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"n={n}", va="center", fontsize=8)
    ax.set_xlabel("MAE (percentage points)")
    ax.set_title("Regression MAE per metric")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def confusion_matrix_fig(records: List[Dict[str, Any]]):
    clf = [r for r in records if r["task"] == "classification"]
    if not clf:
        return None
    y_true = [1 if r["target_pct"] >= 50 else 0 for r in clf]
    y_pred = [1 if r["pred_pct"] >= 50 else 0 for r in clf]
    cm_arr = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm_arr, cmap="Blues")
    labels = ["Not consumed", "Consumed"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                     color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black", fontsize=13)
    acc = (np.array(y_true) == np.array(y_pred)).mean() * 100
    ax.set_title(f"Classification confusion matrix  (acc={acc:.1f}%)")
    fig.tight_layout()
    return fig


def clf_acc_per_field_fig(records: List[Dict[str, Any]]):
    clf = [r for r in records if r["task"] == "classification"]
    if not clf:
        return None
    by_field: Dict[str, List[bool]] = {}
    for r in clf:
        correct = (r["pred_pct"] >= 50) == (r["target_pct"] >= 50)
        by_field.setdefault(r["metric_name"], []).append(correct)
    names = sorted(by_field, key=lambda k: np.mean(by_field[k]))
    accs = [np.mean(by_field[n]) * 100 for n in names]
    counts = [len(by_field[n]) for n in names]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(names))))
    bars = ax.barh(names, accs, color="#55a868")
    for bar, n in zip(bars, counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"n={n}", va="center", fontsize=8)
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 105)
    ax.set_title("Classification accuracy per field")
    fig.tight_layout()
    return fig


def sample_gallery_figs(records: List[Dict[str, Any]], dataset: ConsumptionPairDataset,
                          n_samples: int, seed: int):
    """One figure per unique before/after tray pair: mask overlay for both
    photos (every detected class, one color each) plus a text panel listing
    ground truth vs. predicted for every metric measured on that pair."""
    by_pair: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in records:
        by_pair.setdefault((r["before"], r["after"]), []).append(r)

    rng = random.Random(seed)
    pairs = list(by_pair.keys())
    rng.shuffle(pairs)
    pairs = pairs[:n_samples]

    all_classes = sorted({det["class"] for s in dataset.samples
                           for det in dataset.mask_cache.detections_for(s["before"])}
                          | {det["class"] for s in dataset.samples
                             for det in dataset.mask_cache.detections_for(s["after"])})
    color_map = build_class_color_map(all_classes)

    figs = []
    for before_path, after_path in pairs:
        pair_records = by_pair[(before_path, after_path)]
        before_dets = dataset.mask_cache.detections_for(before_path)
        after_dets = dataset.mask_cache.detections_for(after_path)
        before_img = render_overlay(before_path, before_dets, color_map, dataset.img_size)
        after_img = render_overlay(after_path, after_dets, color_map, dataset.img_size)

        n_rows = len(pair_records)
        fig = plt.figure(figsize=(9, max(3.2, 2.6 + 0.28 * n_rows)))
        gs = fig.add_gridspec(2, 2, height_ratios=[3.2, max(0.9, 0.28 * n_rows)])

        ax_before = fig.add_subplot(gs[0, 0])
        ax_before.imshow(before_img); ax_before.set_title("Before", fontsize=10); ax_before.axis("off")
        ax_after = fig.add_subplot(gs[0, 1])
        ax_after.imshow(after_img); ax_after.set_title("After", fontsize=10); ax_after.axis("off")

        ax_text = fig.add_subplot(gs[1, :])
        ax_text.axis("off")
        used_classes = sorted({c for r in pair_records for c in r["classes"]})
        legend_line = "  ".join(f"■ {c}" for c in used_classes)
        lines = [f"Classes involved: {legend_line}", ""]
        for r in pair_records:
            if r["task"] == "regression":
                err = abs(r["pred_pct"] - r["target_pct"])
                status = "OK" if err < 10 else ("~" if err < 20 else "X")
                lines.append(f"[{status}] {r['metric_name']:<28s} GT={r['target_pct']:5.1f}%   Pred={r['pred_pct']:5.1f}%   |err|={err:4.1f}pt")
            else:
                gt_label = "Consumed" if r["target_pct"] >= 50 else "Not consumed"
                pred_label = "Consumed" if r["pred_pct"] >= 50 else "Not consumed"
                status = "OK" if gt_label == pred_label else "X"
                lines.append(f"[{status}] {r['metric_name']:<28s} GT={gt_label:<13s} Pred={pred_label:<13s} (p={r['pred_pct']:.0f}%)")
        ax_text.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=8.5, family="monospace")

        fig.suptitle(f"{Path(before_path).stem}  →  {Path(after_path).stem}", fontsize=9, y=1.0)
        fig.tight_layout()
        figs.append(fig)

    color_legend_fig = _class_color_legend_fig(color_map)
    return figs, color_legend_fig


def _class_color_legend_fig(color_map: Dict[str, Tuple[int, int, int]]):
    fig, ax = plt.subplots(figsize=(6, max(1.5, 0.28 * len(color_map))))
    ax.axis("off")
    for i, (name, color) in enumerate(sorted(color_map.items())):
        ax.add_patch(plt.Rectangle((0, len(color_map) - i - 1), 0.6, 0.8,
                                    color=tuple(c / 255 for c in color)))
        ax.text(0.8, len(color_map) - i - 0.6, name, va="center", fontsize=9)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, len(color_map))
    ax.set_title("Segmentation class colors", fontsize=10)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# HTML REPORT
# ══════════════════════════════════════════════════════════════════════════════

def img_tag(fig) -> str:
    if fig is None:
        return "<p><em>(not applicable -- no samples of this task type in this split)</em></p>"
    return f'<img src="data:image/png;base64,{fig_to_base64(fig)}" style="max-width:100%;">'


def build_html_report(header: Dict[str, Any], overall: Dict[str, float],
                       agg_figs: Dict[str, Any], gallery_figs: List[Any], legend_fig) -> str:
    header_rows = "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in header.items())
    overall_rows = "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in overall.items())
    gallery_html = "".join(f'<div style="margin-bottom:24px;">{img_tag(f)}</div>' for f in gallery_figs)

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Evaluation report</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 32px; color: #222; }}
h1 {{ margin-bottom: 4px; }}
h2 {{ border-bottom: 2px solid #eee; padding-bottom: 6px; margin-top: 40px; }}
table {{ border-collapse: collapse; margin-bottom: 16px; }}
td {{ padding: 3px 14px 3px 0; }}
</style></head>
<body>
<h1>Siamese Consumption Model -- Evaluation Report</h1>
<table>{header_rows}</table>

<h2>Overall metrics</h2>
<table>{overall_rows}</table>

<h2>Regression</h2>
{img_tag(agg_figs.get("scatter"))}
{img_tag(agg_figs.get("residual"))}
{img_tag(agg_figs.get("mae_per_metric"))}

<h2>Classification</h2>
{img_tag(agg_figs.get("confusion"))}
{img_tag(agg_figs.get("acc_per_field"))}

<h2>Sample gallery ({len(gallery_figs)} tray pairs)</h2>
{img_tag(legend_fig)}
{gallery_html}

</body></html>"""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = args.run_name or f"{args.backbone}_{args.head}"
    checkpoint_path = args.checkpoint or (PROJECT_ROOT / "runs" / "siamese" / run_name / "best.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Train it first with: python training/train.py --backbone {args.backbone} --head {args.head}"
        )

    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_model(args.backbone, args.head, checkpoint_path, device)

    print(f"Loading {args.split} split: {args.manifest}")
    dataset = ConsumptionPairDataset(args.manifest, mask_cache_dir=args.mask_cache_dir, train_mode=False)

    print("Running inference...")
    records = collect_predictions(model, dataset, device, args.batch_size)

    reg = [r for r in records if r["task"] == "regression"]
    clf = [r for r in records if r["task"] == "classification"]
    overall: Dict[str, float] = {}
    if reg:
        errors = np.array([r["pred_pct"] - r["target_pct"] for r in reg])
        overall["regression_mae_pct"] = round(float(np.abs(errors).mean()), 2)
        overall["regression_rmse_pct"] = round(float(np.sqrt((errors ** 2).mean())), 2)
        overall["regression_n_samples"] = len(reg)
    if clf:
        correct = [( (r["pred_pct"] >= 50) == (r["target_pct"] >= 50) ) for r in clf]
        overall["classification_accuracy"] = round(float(np.mean(correct)) * 100, 2)
        overall["classification_n_samples"] = len(clf)

    print("Overall:", overall)

    print("Building plots...")
    agg_figs = {
        "scatter": regression_scatter_fig(records),
        "residual": residual_hist_fig(records),
        "mae_per_metric": mae_per_metric_fig(records),
        "confusion": confusion_matrix_fig(records),
        "acc_per_field": clf_acc_per_field_fig(records),
    }

    print(f"Building sample gallery ({args.gallery_samples} pairs)...")
    gallery_figs, legend_fig = sample_gallery_figs(records, dataset, args.gallery_samples, args.seed)

    header = {
        "Run": run_name,
        "Split evaluated": args.split,
        "Manifest": str(args.manifest),
        "Backbone": args.backbone,
        "Regression head": args.head,
        "Checkpoint": str(checkpoint_path),
        "Generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Total samples": len(records),
    }

    output_dir = REPORT_ROOT / f"{run_name}_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the HTML *before* re-saving individual PNGs, since fig_to_base64
    # closes each figure as it consumes it.
    figs_for_html = {k: v for k, v in agg_figs.items()}
    gallery_copy = list(gallery_figs)

    # Save individual PNGs too, so they show up as standalone artifacts in
    # the MLflow UI (not just embedded inside the HTML).
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    for name, fig in agg_figs.items():
        if fig is not None:
            fig.savefig(figures_dir / f"{name}.png", bbox_inches="tight", dpi=130)
    if legend_fig is not None:
        legend_fig.savefig(figures_dir / "class_color_legend.png", bbox_inches="tight", dpi=130)
    gallery_dir = output_dir / "gallery"
    gallery_dir.mkdir(exist_ok=True)
    for i, fig in enumerate(gallery_figs):
        fig.savefig(gallery_dir / f"sample_{i:03d}.png", bbox_inches="tight", dpi=130)

    html = build_html_report(header, overall, figs_for_html, gallery_copy, legend_fig)
    report_path = output_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"Report saved locally: {report_path}")

    # ── MLflow logging ──
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or MLFLOW_TRACKING_URI_DEFAULT
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    parent_run_id = None
    existing = mlflow.search_runs(filter_string=f"tags.mlflow.runName = '{run_name}'", max_results=1)
    if len(existing):
        parent_run_id = existing.iloc[0]["run_id"]

    with mlflow.start_run(run_name=f"{run_name}_{args.split}_eval"):
        mlflow.log_params(
            {
                "backbone": args.backbone,
                "head": args.head,
                "split": args.split,
                "checkpoint": str(checkpoint_path),
                "parent_run_id": parent_run_id,
                "parent_run_name": run_name,
                "gallery_samples": len(gallery_figs),
            }
        )
        mlflow.log_metrics(overall)
        mlflow.log_artifacts(str(output_dir), artifact_path="report")

    print("Logged to MLflow experiment:", MLFLOW_EXPERIMENT_NAME)
    print(f"Open report.html directly, or browse the 'report' artifact folder in the MLflow UI.")


if __name__ == "__main__":
    main()
