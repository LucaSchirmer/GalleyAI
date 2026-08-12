"""Evaluate a segmentation-mask pixel-counting consumption baseline.

The baseline predicts consumption from the reduction in the segmented food
area between a before photo and an after photo:

    consumption (%) = 100 * (before foreground pixels - after foreground pixels)
                            / before foreground pixels

It reads the cached YOLO polygons in ``mask_cache/``; it does not run YOLO or
load the Siamese model.  The script writes a CSV with every usable prediction,
a JSON summary, and a standalone HTML report.

Run from the repository root:

    python pixel_counting_baseline/run_pixel_counting_baseline.py --split val

Use ``--split test`` for final results, or pass ``--manifest`` for an
arbitrary pair manifest.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFESTS = {
    "train": PROJECT_ROOT / "data_pairs_with_splits" / "train.json",
    "val": PROJECT_ROOT / "data_pairs_with_splits" / "val.json",
    "test": PROJECT_ROOT / "data_pairs_with_splits" / "test.json",
}
QUALITY_FLAG_EXCLUDES = {"Food rearranged significantly"}
METRIC_LABEL_ALIASES = {
    "pct_vanilla_pudding": ["vanilla_pudding_with_fruits"],
    "pct_salad_dish_main": ["main_salad"],
    "pct_chicken_rice_veg": ["chicken", "rice", "carrots", "broccoli"],
}


def suffix_stem(path: str) -> str:
    """Return the cache key, dropping Label Studio's optional hash prefix."""
    stem = Path(path).stem
    return stem.split("__", 1)[1] if "__" in stem else stem


def classes_for_metric(metric_name: str) -> list[str]:
    if metric_name in METRIC_LABEL_ALIASES:
        return METRIC_LABEL_ALIASES[metric_name]
    return [metric_name[4:] if metric_name.startswith("pct_") else metric_name]


def mean(values: Iterable[float]) -> float:
    """Small compatibility helper for Python versions without statistics.fmean."""
    values = list(values)
    return sum(values) / len(values)


def first_choice_list(choices: dict[str, Any], field: str) -> list[str]:
    values = choices.get(field, [])
    return values[0] if values else []


class MaskCache:
    def __init__(self, directory: Path):
        self.directory = directory
        self.loaded: dict[str, list[dict[str, Any]]] = {}

    def detections_for(self, image_path: str) -> list[dict[str, Any]]:
        stem = suffix_stem(image_path)
        if stem not in self.loaded:
            path = self.directory / f"{stem}.json"
            if path.exists():
                with path.open(encoding="utf-8") as file:
                    self.loaded[stem] = json.load(file).get("detections", [])
            else:
                self.loaded[stem] = []
        return self.loaded[stem]


def foreground_pixel_count(
    detections: Iterable[dict[str, Any]], classes: list[str], raster_size: int
) -> int:
    """Rasterize the union of target-class polygons and count its pixels."""
    mask = Image.new("1", (raster_size, raster_size), 0)
    draw = ImageDraw.Draw(mask)
    class_set = set(classes)
    for detection in detections:
        if detection.get("class") not in class_set:
            continue
        points = detection.get("polygon", [])
        if len(points) >= 3:
            draw.polygon(
                [(x * raster_size, y * raster_size) for x, y in points], fill=1
            )
    return sum(mask.getdata())


def predict_consumption(before_pixels: int, after_pixels: int) -> float:
    """Return the area-reduction prediction, bounded to the valid range."""
    if before_pixels <= 0:
        raise ValueError("before mask contains no foreground pixels")
    return max(0.0, min(100.0, 100.0 * (before_pixels - after_pixels) / before_pixels))


def regression_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not rows:
        return {"n": 0, "mae": None, "rmse": None, "bias": None, "r2": None}
    errors = [row["prediction_pct"] - row["target_pct"] for row in rows]
    targets = [row["target_pct"] for row in rows]
    mae = mean(abs(error) for error in errors)
    rmse = math.sqrt(mean(error * error for error in errors))
    target_mean = mean(targets)
    ss_total = sum((target - target_mean) ** 2 for target in targets)
    ss_residual = sum(error * error for error in errors)
    return {
        "n": len(rows),
        "mae": mae,
        "rmse": rmse,
        "bias": mean(errors),
        "r2": 1 - ss_residual / ss_total if ss_total else None,
    }


def breakdown(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float | int | None]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {name: regression_metrics(group) for name, group in sorted(groups.items())}


def fmt(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def metrics_table(title: str, metrics: dict[str, dict[str, float | int | None]]) -> str:
    body = "".join(
        "<tr>"
        f"<td>{html.escape(name)}</td><td>{fmt(result['n'])}</td>"
        f"<td>{fmt(result['mae'])}</td><td>{fmt(result['rmse'])}</td>"
        f"<td>{fmt(result['bias'])}</td><td>{fmt(result['r2'], 3)}</td>"
        "</tr>"
        for name, result in metrics.items()
    )
    return f"""<section><h2>{html.escape(title)}</h2><table>
<thead><tr><th>Group</th><th>n</th><th>MAE (pp)</th><th>RMSE (pp)</th><th>Bias (pp)</th><th>R²</th></tr></thead>
<tbody>{body}</tbody></table></section>"""


def write_report(
    output_dir: Path,
    manifest: Path,
    raster_size: int,
    rows: list[dict[str, Any]],
    skipped: dict[str, int],
) -> None:
    overall = regression_metrics(rows)
    by_metric = breakdown(rows, "metric_name")
    by_category = breakdown(rows, "category")
    summary = {
        "method": "100 * (before_mask_pixels - after_mask_pixels) / before_mask_pixels, clamped to [0, 100]",
        "manifest": str(manifest),
        "raster_size": raster_size,
        "overall": overall,
        "by_metric": by_metric,
        "by_category": by_category,
        "skipped": skipped,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    columns = [
        "before", "after", "category", "metric_name", "classes", "target_pct",
        "prediction_pct", "absolute_error_pct_points", "before_pixels", "after_pixels",
    ]
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "classes": ",".join(row["classes"])})

    worst = sorted(rows, key=lambda row: row["absolute_error_pct_points"], reverse=True)[:50]
    worst_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['metric_name'])}</td><td>{html.escape(str(row['category']))}</td>"
        f"<td>{fmt(row['target_pct'])}</td><td>{fmt(row['prediction_pct'])}</td>"
        f"<td>{fmt(row['absolute_error_pct_points'])}</td>"
        f"<td>{row['before_pixels']}</td><td>{row['after_pixels']}</td>"
        f"<td>{html.escape(Path(row['after']).name)}</td>"
        "</tr>"
        for row in worst
    )
    skipped_items = "".join(
        f"<li>{html.escape(reason.replace('_', ' '))}: {count}</li>"
        for reason, count in sorted(skipped.items())
    ) or "<li>None</li>"
    report = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Pixel Counting Baseline Report</title><style>
body {{ font: 15px/1.45 system-ui, sans-serif; max-width: 1200px; margin: 2rem auto; padding: 0 1rem; color: #18212b; }}
h1 {{ margin-bottom: .15rem; }} .muted {{ color: #52606d; }} .cards {{ display:flex; gap:1rem; flex-wrap:wrap; margin:1.5rem 0; }}
.card {{ border:1px solid #d7dee7; border-radius:8px; padding:1rem; min-width:135px; }} .value {{ font-size:1.7rem; font-weight:700; }}
table {{ border-collapse:collapse; width:100%; margin:.5rem 0 2rem; font-size:.9rem; }} th, td {{ border:1px solid #d7dee7; padding:.45rem .55rem; text-align:right; }} th:first-child, td:first-child {{ text-align:left; }} thead {{ background:#f3f6f9; }}
</style></head><body>
<h1>Pixel Counting Baseline</h1>
<p class="muted">Manifest: {html.escape(str(manifest))} · mask rasterization: {raster_size}×{raster_size}</p>
<p>Prediction: <code>100 × (before mask pixels − predicted-after mask pixels) / before mask pixels</code>, clamped to 0–100%. The masks are the cached YOLO segmentation predictions.</p>
<div class="cards"><div class="card"><div class="muted">Usable samples</div><div class="value">{overall['n']}</div></div>
<div class="card"><div class="muted">MAE</div><div class="value">{fmt(overall['mae'])} pp</div></div>
<div class="card"><div class="muted">RMSE</div><div class="value">{fmt(overall['rmse'])} pp</div></div>
<div class="card"><div class="muted">Bias</div><div class="value">{fmt(overall['bias'])} pp</div></div>
<div class="card"><div class="muted">R²</div><div class="value">{fmt(overall['r2'], 3)}</div></div></div>
<h2>Excluded samples</h2><ul>{skipped_items}</ul>
{metrics_table('Performance by food metric', by_metric)}
{metrics_table('Performance by source category', by_category)}
<section><h2>50 largest errors</h2><table><thead><tr><th>Metric</th><th>Category</th><th>True %</th><th>Predicted %</th><th>Absolute error (pp)</th><th>Before pixels</th><th>After pixels</th><th>After image</th></tr></thead><tbody>{worst_rows}</tbody></table></section>
</body></html>"""
    (output_dir / "report.html").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=DEFAULT_MANIFESTS, default="val")
    parser.add_argument("--manifest", type=Path, help="Overrides --split with a specific JSON manifest.")
    parser.add_argument("--mask-cache", type=Path, default=PROJECT_ROOT / "mask_cache")
    parser.add_argument("--output-dir", type=Path, help="Defaults to pixel_counting_baseline/reports/<split>.")
    parser.add_argument("--raster-size", type=int, default=640, help="Square resolution used to count polygon pixels.")
    parser.add_argument("--include-quality-flagged", action="store_true")
    args = parser.parse_args()

    if args.raster_size <= 0:
        parser.error("--raster-size must be positive")
    manifest = args.manifest or DEFAULT_MANIFESTS[args.split]
    output_dir = args.output_dir or PROJECT_ROOT / "pixel_counting_baseline" / "reports" / manifest.stem
    if not manifest.exists():
        parser.error(f"Manifest not found: {manifest}")
    if not args.mask_cache.exists():
        parser.error(f"Mask cache not found: {args.mask_cache}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with manifest.open(encoding="utf-8") as file:
        pairs = json.load(file)
    cache = MaskCache(args.mask_cache)
    rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = defaultdict(int)

    for pair in pairs:
        flags = set(first_choice_list(pair.get("choices", {}), "quality_flags"))
        if not args.include_quality_flagged and flags & QUALITY_FLAG_EXCLUDES:
            skipped["quality_flagged_pair"] += 1
            continue
        for metric_name, values in pair.get("numbers", {}).items():
            if not values:
                skipped["missing_target"] += 1
                continue
            classes = classes_for_metric(metric_name)
            before_pixels = foreground_pixel_count(
                cache.detections_for(pair["before"]), classes, args.raster_size
            )
            if before_pixels == 0:
                skipped["missing_before_mask"] += 1
                continue
            after_pixels = foreground_pixel_count(
                cache.detections_for(pair["after"]), classes, args.raster_size
            )
            prediction = predict_consumption(before_pixels, after_pixels)
            target = float(values[0])
            rows.append({
                "before": pair["before"], "after": pair["after"],
                "category": pair.get("category", "unknown"), "metric_name": metric_name,
                "classes": classes, "target_pct": target, "prediction_pct": prediction,
                "absolute_error_pct_points": abs(prediction - target),
                "before_pixels": before_pixels, "after_pixels": after_pixels,
            })

    write_report(output_dir, manifest, args.raster_size, rows, dict(skipped))
    overall = regression_metrics(rows)
    print(f"Evaluated {overall['n']} samples from {manifest}")
    print(f"MAE: {fmt(overall['mae'])} percentage points | RMSE: {fmt(overall['rmse'])}")
    print(f"Report: {output_dir / 'report.html'}")
    print(f"Predictions: {output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
