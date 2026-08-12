"""Generates a browsable HTML audit report for the Siamese consumption model.

For every (before, after) pair in the given manifest, produces:
  - a side-by-side image of the before/after photos with the YOLO
    segmentation masks overlaid (color-coded per class, with a legend) —
    so you can directly eyeball whether the segmentation is catching the
    right regions, independent of the consumption prediction
  - a table of every pct_* metric predicted for that pair: predicted %,
    true %, absolute error

Pairs are sorted worst-max-error-first in the report, so the meal
instances most worth investigating are at the top.

Run:
    python scripts/generate_prediction_report.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from siamese_dataset import ConsumptionPairDataset, MaskCache
from siamese_model import SiameseConsumptionNet, combined_prediction


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CHECKPOINT_PATH = Path("runs/siamese/resnet50_frozen_v3_twohead/best.pt")
EVAL_MANIFEST = Path("data_pairs_with_splits/val.json")  # switch to train.json/test.json as needed

OUTPUT_DIR = Path("runs/siamese/resnet50_frozen_v1/report")
IMAGES_SUBDIR = "pairs"

THUMB_SIZE = 384
OVERLAY_FILL_ALPHA = 90    # 0-255, semi-transparent fill for the mask region
OVERLAY_OUTLINE_ALPHA = 255
BATCH_SIZE = 16
NUM_WORKERS = 2

CLASSIFICATION_THRESHOLD = 50.0  # pct >= this -> predicted "Consumed"

# Fixed color per class so the same class always gets the same color across
# every report you generate. Extend this list if you add more classes.
CLASS_NAMES = [
    "bread_roll", "broccoli", "butter", "carrots", "cherry_jam", "chicken",
    "chocolate_cake", "coffee", "cola", "cookie", "fish_salmon", "fruit_salad",
    "honey", "orange_juice", "pasta_pesto", "plum_jam", "rice", "main_salad",
    "side_salad", "tea", "vanilla_pudding_with_fruits", "water",
    "wrap_half_1", "wrap_half_2",
]
PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
    (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
    (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
    (188, 143, 143), (60, 60, 220), (0, 100, 0), (139, 69, 19),
]
CLASS_COLOR = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(CLASS_NAMES)}


# ══════════════════════════════════════════════════════════════════════════════
# MASK OVERLAY DRAWING
# ══════════════════════════════════════════════════════════════════════════════

def draw_overlay(image_path: str, detections: List[Dict[str, Any]], size: int) -> Tuple[Image.Image, List[str]]:
    """Returns (image with mask overlays baked in, list of class names present)."""
    base = Image.open(image_path).convert("RGB").resize((size, size))
    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    classes_present: List[str] = []
    for det in detections:
        cls = det["class"]
        color = CLASS_COLOR.get(cls, (255, 255, 255))
        points = [(x * size, y * size) for x, y in det["polygon"]]
        if len(points) < 3:
            continue
        classes_present.append(cls)
        try:
            draw.polygon(points, fill=(*color, OVERLAY_FILL_ALPHA), outline=(*color, OVERLAY_OUTLINE_ALPHA), width=2)
        except TypeError:
            # older Pillow without polygon(width=...)
            draw.polygon(points, fill=(*color, OVERLAY_FILL_ALPHA), outline=(*color, OVERLAY_OUTLINE_ALPHA))

    composited = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    return composited, sorted(set(classes_present))


def add_legend(image: Image.Image, classes_present: List[str], label: str) -> Image.Image:
    """Adds a title label + small color-swatch legend strip below the image."""
    swatch_h = 18
    legend_rows = max(1, -(-len(classes_present) // 4))  # ceil division, 4 per row
    legend_h = legend_rows * swatch_h + 6
    title_h = 22

    canvas = Image.new("RGB", (image.width, image.height + title_h + legend_h), (255, 255, 255))
    canvas.paste(image, (0, title_h))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((4, 2), label, fill=(0, 0, 0), font=font)

    x, y = 4, image.height + title_h + 3
    for i, cls in enumerate(classes_present):
        if i > 0 and i % 4 == 0:
            x = 4
            y += swatch_h
        color = CLASS_COLOR.get(cls, (128, 128, 128))
        draw.rectangle([x, y, x + 10, y + 10], fill=color)
        draw.text((x + 14, y - 1), cls, fill=(0, 0, 0), font=font)
        x += 100

    return canvas


def build_pair_image(before_path: str, after_path: str, mask_cache: MaskCache, size: int) -> Image.Image:
    before_overlay, before_classes = draw_overlay(before_path, mask_cache.detections_for(before_path), size)
    after_overlay, after_classes = draw_overlay(after_path, mask_cache.detections_for(after_path), size)

    before_final = add_legend(before_overlay, before_classes, "BEFORE (unconsumed)")
    after_final = add_legend(after_overlay, after_classes, "AFTER (consumed)")

    gap = 10
    combined = Image.new(
        "RGB",
        (before_final.width + after_final.width + gap, max(before_final.height, after_final.height)),
        (255, 255, 255),
    )
    combined.paste(before_final, (0, 0))
    combined.paste(after_final, (before_final.width + gap, 0))
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE + GROUPING
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
            preds.append(combined_prediction(reg_out, clf_logit, is_regression).cpu())
    return torch.cat(preds).numpy() * 100


def group_by_pair(dataset, preds):
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for sample, pred in zip(dataset.samples, preds):
        key = (sample["before"], sample["after"])
        group = groups.setdefault(
            key,
            {"before": sample["before"], "after": sample["after"],
             "category": sample.get("category"), "rows": []},
        )
        true_pct = sample["target_pct"]
        pred = float(pred)

        if sample["task"] == "classification":
            pred_label = "Consumed" if pred >= CLASSIFICATION_THRESHOLD else "Not consumed"
            true_label = "Consumed" if true_pct >= CLASSIFICATION_THRESHOLD else "Not consumed"
            row = {
                "metric": sample["metric_name"],
                "task": "classification",
                "pred": pred,
                "pred_label": pred_label,
                "true_label": true_label,
                "correct": pred_label == true_label,
            }
        else:
            row = {
                "metric": sample["metric_name"],
                "task": "regression",
                "pred": pred,
                "true": float(true_pct),
                "abs_error": abs(pred - float(true_pct)),
            }

        group["rows"].append(row)
    return list(groups.values())


# ══════════════════════════════════════════════════════════════════════════════
# HTML REPORT
# ══════════════════════════════════════════════════════════════════════════════

def build_html(groups: List[Dict[str, Any]], images_subdir: str) -> str:
    rows_html = []
    for i, group in enumerate(groups):
        reg_rows = [r for r in group["rows"] if r["task"] == "regression"]
        clf_rows = [r for r in group["rows"] if r["task"] == "classification"]
        max_err = max((r["abs_error"] for r in reg_rows), default=0.0)
        n_wrong = sum(1 for r in clf_rows if not r["correct"])

        reg_table = ""
        if reg_rows:
            reg_body = "".join(
                f"<tr><td>{r['metric']}</td><td>{r['pred']:.1f}%</td>"
                f"<td>{r['true']:.1f}%</td><td>{r['abs_error']:.1f}</td></tr>"
                for r in sorted(reg_rows, key=lambda r: -r["abs_error"])
            )
            reg_table = f"""
              <h4>Regression (pct_*)</h4>
              <table>
                <tr><th>metric</th><th>predicted</th><th>true</th><th>abs error</th></tr>
                {reg_body}
              </table>"""

        clf_table = ""
        if clf_rows:
            clf_body = "".join(
                f"<tr><td>{r['metric']}</td><td>{r['pred_label']} ({r['pred']:.0f}%)</td>"
                f"<td>{r['true_label']}</td><td>{'✓' if r['correct'] else '✗ WRONG'}</td></tr>"
                for r in sorted(clf_rows, key=lambda r: r["correct"])  # wrong ones first
            )
            clf_table = f"""
              <h4>Classification (consumed / not consumed)</h4>
              <table>
                <tr><th>field</th><th>predicted</th><th>true</th><th>correct?</th></tr>
                {clf_body}
              </table>"""

        header_bits = [f"max regression error {max_err:.1f} pct pts"]
        if n_wrong:
            header_bits.append(f"{n_wrong} misclassified")
        rows_html.append(
            f"""
            <div class="card">
              <h3>#{i:04d} — {group['category']} — {', '.join(header_bits)}</h3>
              <img src="{images_subdir}/{i:04d}.png">
              {reg_table}
              {clf_table}
            </div>
            """
        )

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Consumption Model — Prediction Audit Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; background: #fafafa; }}
    .card {{ background: white; border: 1px solid #ddd; border-radius: 6px;
             padding: 12px; margin-bottom: 20px; }}
    img {{ max-width: 100%; }}
    h4 {{ margin-bottom: 4px; }}
    table {{ border-collapse: collapse; margin-top: 4px; margin-bottom: 12px; }}
    td, th {{ border: 1px solid #ccc; padding: 4px 10px; text-align: left; font-size: 13px; }}
    th {{ background: #f0f0f0; }}
  </style>
</head>
<body>
  <h1>Prediction Audit Report</h1>
  <p>{len(groups)} meal instance(s), sorted worst first (misclassifications, then regression error).</p>
  {''.join(rows_html)}
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    images_dir = OUTPUT_DIR / IMAGES_SUBDIR
    images_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
    model = SiameseConsumptionNet(freeze_backbone=True).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))

    print(f"Loading eval set from {EVAL_MANIFEST}...")
    dataset = ConsumptionPairDataset(EVAL_MANIFEST, train_mode=False)

    print("Running inference...")
    preds = run_inference(model, dataset, device)

    print("Grouping predictions by meal instance...")
    groups = group_by_pair(dataset, preds)

    def severity(group):
        reg_rows = [r for r in group["rows"] if r["task"] == "regression"]
        clf_rows = [r for r in group["rows"] if r["task"] == "classification"]
        max_err = max((r["abs_error"] for r in reg_rows), default=0.0)
        n_wrong = sum(1 for r in clf_rows if not r["correct"])
        return (n_wrong, max_err)

    groups.sort(key=severity, reverse=True)

    print(f"Rendering {len(groups)} mask-overlay images...")
    mask_cache = MaskCache()
    for i, group in enumerate(groups):
        combined = build_pair_image(group["before"], group["after"], mask_cache, THUMB_SIZE)
        combined.save(images_dir / f"{i:04d}.png")
        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1}/{len(groups)}")

    print("Writing HTML report...")
    html = build_html(groups, IMAGES_SUBDIR)
    report_path = OUTPUT_DIR / "index.html"
    report_path.write_text(html, encoding="utf-8")

    print(f"\nDone. Open {report_path.resolve()} in a browser.")


if __name__ == "__main__":
    main()