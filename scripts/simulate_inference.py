"""Run one manifest-linked tray pair through segmentation and Siamese inference.

The input image may be either the consumed or unconsumed image.  A consumed
image normally selects one pair; an unconsumed image can select several pairs,
in which case use ``--pair-index`` after inspecting ``--list-matches``.

Example (run from the repository root):
    python scripts/simulate_inference.py \
        --image data_consumed/images/4b7e1c26__all_markers_shot_20260528_154910.jpg \
        --yolo-checkpoint path/to/yolo_best.pt \
        --backbone vit_b16 --head mlp \
        --siamese-checkpoint siamese_consumption_model/runs/siamese/vit_b16_mlp/best.pt
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = REPO_ROOT / "siamese_consumption_model"
sys.path.insert(0, str(MODEL_ROOT))

from data.siamese_dataset import ConsumptionPairDataset, mask_statistics, suffix_stem
from data.metric_vocabulary import metric_index
from models.backbones import list_backbones
from models.siamese_net import SiameseConsumptionNet, list_regression_heads


DEFAULT_MANIFESTS = [
    REPO_ROOT / "data_pairs_with_splits" / f"{split}.json"
    for split in ("train", "val", "test")
]
DEFAULT_YOLO_CHECKPOINT = (
    REPO_ROOT
    / "runs/segment/baseline_v3_current-2/weights/best.pt"
)
DEFAULT_MASK_CACHE_DIR = REPO_ROOT / "mask_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, required=True, help="consumed or unconsumed image path")
    parser.add_argument("--manifest", type=Path, action="append", dest="manifests",
                        help="pair manifest; repeat to search several (default: train, val, test)")
    parser.add_argument("--pair-index", type=int, default=0, help="zero-based match to use")
    parser.add_argument("--list-matches", action="store_true", help="show matches and exit")
    parser.add_argument("--yolo-checkpoint", type=Path, default=DEFAULT_YOLO_CHECKPOINT)
    parser.add_argument("--mask-cache-dir", type=Path, default=DEFAULT_MASK_CACHE_DIR,
                        help="shared YOLO mask cache (default: repository mask_cache/)")
    parser.add_argument("--refresh-masks", action="store_true",
                        help="ignore shared cached masks and run YOLO for this simulation")
    parser.add_argument("--siamese-checkpoint", type=Path, required=True)
    parser.add_argument("--backbone", required=True, choices=list_backbones())
    parser.add_argument("--head", required=True, choices=list_regression_heads())
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--device", default=None, help="e.g. cpu, cuda, cuda:0 (default: auto)")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _same_image(candidate: str, supplied: Path) -> bool:
    candidate_path = Path(candidate)
    supplied_abs = supplied.resolve()
    possible_paths = [candidate_path]
    if not candidate_path.is_absolute():
        possible_paths.append(REPO_ROOT / candidate_path)
    return any(path.resolve() == supplied_abs for path in possible_paths) or candidate_path.name == supplied.name


def find_matches(image_path: Path, manifests: Iterable[Path]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for manifest in manifests:
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest}")
        pairs = json.loads(manifest.read_text(encoding="utf-8"))
        for pair in pairs:
            if _same_image(pair["before"], image_path) or _same_image(pair["after"], image_path):
                matches.append({"manifest": str(manifest), "pair": pair})
    return matches


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def mask_cache_path(cache_dir: Path, image_path: Path) -> Path:
    return cache_dir / f"{suffix_stem(image_path.name)}.json"


def write_detections(cache_dir: Path, image_path: Path, detections: list[dict[str, Any]]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    mask_cache_path(cache_dir, image_path).write_text(
        json.dumps({"detections": detections}, indent=2), encoding="utf-8"
    )


def resolve_detections(
    image_path: Path,
    shared_cache_dir: Path,
    run_cache_dir: Path,
    segment,
    refresh: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    """Use a shared cached mask when possible and retain a run-local copy."""
    shared_path = mask_cache_path(shared_cache_dir, image_path)
    if shared_path.exists() and not refresh:
        detections = json.loads(shared_path.read_text(encoding="utf-8"))["detections"]
        source = "cache"
    else:
        detections = segment(image_path)
        source = "YOLO"
    write_detections(run_cache_dir, image_path, detections)
    return detections, source


def validate_mask_cache(cache_dir: Path, model_path: Path, conf: float, imgsz: int) -> None:
    metadata_path = cache_dir / "_cache_metadata.json"
    if not metadata_path.exists():
        raise RuntimeError(
            f"Mask-cache metadata not found: {metadata_path}. "
            "Use --refresh-masks or select a cache generated by cache_segmentation_masks.py."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cached_model = Path(metadata["model_path"]).resolve()
    mismatches = []
    if cached_model != model_path.resolve():
        mismatches.append(f"checkpoint {cached_model}")
    if float(metadata["confidence_threshold"]) != conf:
        mismatches.append(f"confidence {metadata['confidence_threshold']}")
    if int(metadata["image_size"]) != imgsz:
        mismatches.append(f"image size {metadata['image_size']}")
    if mismatches:
        raise RuntimeError(
            "Mask cache was generated with different settings (" + ", ".join(mismatches) + "). "
            "Use matching arguments or pass --refresh-masks."
        )


def segment_image(model: Any, image_path: Path, conf: float, imgsz: int) -> list[dict[str, Any]]:
    result = model.predict(source=str(image_path), conf=conf, imgsz=imgsz, verbose=False)[0]
    detections: list[dict[str, Any]] = []
    if result.masks is not None:
        for cls_idx, polygon, bbox in zip(
            result.boxes.cls.tolist(), result.masks.xyn, result.boxes.xyxyn.tolist()
        ):
            detections.append({
                "class": result.names[int(cls_idx)],
                "polygon": polygon.tolist(),
                "bbox": bbox,
            })
    return detections


def save_overlay(image_path: Path, detections: list[dict[str, Any]], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors: dict[str, tuple[int, int, int]] = {}
    rng = np.random.default_rng(42)
    for detection in detections:
        class_name = detection["class"]
        colors.setdefault(class_name, tuple(int(v) for v in rng.integers(40, 240, size=3)))
        color = colors[class_name]
        points = [(x * image.width, y * image.height) for x, y in detection["polygon"]]
        if len(points) >= 3:
            draw.polygon(points, fill=(*color, 90), outline=(*color, 255), width=3)
    Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB").save(output_path)


def load_model(args: argparse.Namespace, device: torch.device) -> SiameseConsumptionNet:
    model = SiameseConsumptionNet(
        backbone_name=args.backbone,
        head_name=args.head,
        pretrained=False,
    )
    state_dict = torch.load(args.siamese_checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def display_name(metric_name: str) -> str:
    name = metric_name
    for prefix in ("pct_", "status_", "drink_", "extra_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.replace("_", " ").title()


def write_ui_report(output_dir: Path, predictions: list[dict[str, Any]], backbone: str, head: str) -> Path:
    cards = []
    for prediction in predictions:
        consumed = float(prediction["predicted_consumed_pct"])
        ground_truth = float(prediction["manifest_ground_truth_pct"])
        remaining = max(0.0, 100.0 - consumed)
        error = abs(consumed - ground_truth)
        item = html.escape(display_name(prediction["item"]))
        classes = html.escape(", ".join(prediction["classes"]))
        cards.append(f"""<article class="result-card">
          <div class="result-heading"><h3>{item}</h3><span class="error">Error · {error:.1f} pp</span></div>
          <div class="measure"><div><strong>Prediction</strong><b>{consumed:.1f}%</b></div>
            <div class="meter prediction"><span style="width:{consumed:.2f}%"></span></div></div>
          <div class="measure"><div><strong>Ground truth</strong><b>{ground_truth:.1f}%</b></div>
            <div class="meter truth"><span style="width:{ground_truth:.2f}%"></span></div></div>
          <div class="result-meta"><span>Predicted remaining · {remaining:.1f}%</span><span>Mask: {classes}</span></div>
        </article>""")

    report = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>GalleyAI simulation report</title>
<style>
:root{{color-scheme:dark;--bg:#0a0f18;--panel:#121a27;--line:#263247;--muted:#91a0b7;--text:#f4f7fb;--accent:#62d6a5;--accent2:#5ca9ff}}
*{{box-sizing:border-box}}body{{margin:0;min-height:100vh;background:radial-gradient(circle at 50% 0,#17243a 0,var(--bg) 45%);color:var(--text);font:16px/1.5 Inter,Segoe UI,Arial,sans-serif}}
main{{width:min(1180px,calc(100% - 32px));margin:auto;padding:38px 0 50px}}header{{display:flex;justify-content:space-between;align-items:end;gap:24px;margin-bottom:22px}}
.eyebrow{{margin:0;color:var(--accent);font-size:.76rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase}}h1{{margin:4px 0 0;font-size:clamp(1.8rem,4vw,3.1rem);letter-spacing:-.04em}}.model{{color:var(--muted);text-align:right}}
.progress{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:22px}}.progress div{{height:5px;border-radius:99px;background:var(--line);transition:.25s}}.progress div.active{{background:linear-gradient(90deg,var(--accent),var(--accent2))}}
.step{{display:none}}.step.active{{display:block;animation:fade .25s ease}}@keyframes fade{{from{{opacity:0;transform:translateY(8px)}}}}.step-title{{display:flex;align-items:baseline;justify-content:space-between;gap:20px;margin-bottom:14px}}.step-title h2{{margin:0;font-size:1.25rem}}.step-title p{{color:var(--muted);margin:0}}
.images{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}figure{{margin:0;padding:10px;border:1px solid var(--line);background:var(--panel);border-radius:16px}}figure img{{display:block;width:100%;max-height:66vh;object-fit:contain;border-radius:10px;background:#05080d}}figcaption{{padding:10px 4px 2px;font-weight:750}}
.results{{display:grid;gap:12px}}.result-card{{padding:18px;border:1px solid var(--line);background:var(--panel);border-radius:14px}}.result-heading,.result-meta,.measure>div:first-child{{display:flex;justify-content:space-between;gap:20px}}.result-heading h3{{margin:0}}.error{{color:var(--muted);font-size:.9rem}}.measure{{margin-top:14px}}.measure strong{{font-size:.86rem;color:var(--muted)}}.measure b{{font-size:.92rem}}.meter{{height:9px;margin:6px 0 0;overflow:hidden;border-radius:99px;background:#263247}}.meter span{{display:block;height:100%;border-radius:inherit}}.meter.prediction span{{background:linear-gradient(90deg,var(--accent2),var(--accent))}}.meter.truth span{{background:#f0b35a}}.result-meta{{margin-top:16px;padding-top:12px;border-top:1px solid var(--line);color:var(--muted);font-size:.85rem}}
nav{{display:flex;justify-content:space-between;margin-top:22px}}button{{border:1px solid var(--line);border-radius:10px;padding:11px 22px;background:var(--panel);color:var(--text);font:inherit;font-weight:750;cursor:pointer}}button.primary{{margin-left:auto;border:0;color:#07120e;background:var(--accent)}}button:disabled{{opacity:.35;cursor:default}}
@media(max-width:720px){{.images{{grid-template-columns:1fr}}header,.step-title{{align-items:start;flex-direction:column}}.model{{text-align:left}}.result-heading,.result-meta{{flex-direction:column;gap:4px}}}}
</style></head><body><main>
<header><div><p class="eyebrow">GalleyAI · inference simulation</p><h1>Consumption report</h1></div><div class="model">Model<br><strong>{html.escape(backbone)} · {html.escape(head)}</strong></div></header>
<div class="progress"><div class="active"></div><div></div><div></div></div>
<section class="step active"><div class="step-title"><h2>1 · Input pair</h2><p>Original images without segmentation</p></div><div class="images"><figure><img src="before.jpg" alt="Meal before consumption"><figcaption>Before · unconsumed</figcaption></figure><figure><img src="after.jpg" alt="Meal after consumption"><figcaption>After · consumed</figcaption></figure></div></section>
<section class="step"><div class="step-title"><h2>2 · Segmentation</h2><p>Detected food regions used by the model</p></div><div class="images"><figure><img src="before_segmentation.jpg" alt="Before segmentation"><figcaption>Before · segmentation masks</figcaption></figure><figure><img src="after_segmentation.jpg" alt="After segmentation"><figcaption>After · segmentation masks</figcaption></figure></div></section>
<section class="step"><div class="step-title"><h2>3 · Consumption estimate</h2><p>Predicted amount consumed for each part</p></div><div class="results">{''.join(cards)}</div></section>
<nav><button id="back" disabled>Back</button><button id="next" class="primary">Next · show segmentation</button></nav>
</main><script>
const steps=[...document.querySelectorAll('.step')],bars=[...document.querySelectorAll('.progress div')],back=document.querySelector('#back'),next=document.querySelector('#next');let current=0;
function render(){{steps.forEach((x,i)=>x.classList.toggle('active',i===current));bars.forEach((x,i)=>x.classList.toggle('active',i<=current));back.disabled=current===0;next.style.visibility=current===2?'hidden':'visible';next.textContent=current===0?'Next · show segmentation':'Next · show predictions';window.scrollTo({{top:0,behavior:'smooth'}})}}
back.addEventListener('click',()=>{{if(current>0){{current--;render()}}}});next.addEventListener('click',()=>{{if(current<2){{current++;render()}}}});
</script></body></html>"""
    report_path = output_dir / "report.html"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    args.image = args.image.resolve()
    manifests = args.manifests or DEFAULT_MANIFESTS
    matches = find_matches(args.image, manifests)
    if not matches:
        raise ValueError(f"No pair containing '{args.image.name}' was found in the selected manifest(s).")

    print(f"Found {len(matches)} matching pair(s):")
    for index, match in enumerate(matches):
        pair = match["pair"]
        print(f"  [{index}] after={Path(pair['after']).name}  before={Path(pair['before']).name}  "
              f"manifest={Path(match['manifest']).name}")
    if args.list_matches:
        return
    if args.pair_index < 0 or args.pair_index >= len(matches):
        raise IndexError(f"--pair-index must be between 0 and {len(matches) - 1}")

    selected = matches[args.pair_index]
    pair = dict(selected["pair"])
    before_path = resolve_repo_path(pair["before"]).resolve()
    after_path = resolve_repo_path(pair["after"]).resolve()
    for path in (before_path, after_path, args.yolo_checkpoint, args.siamese_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir = args.output_dir or (
        REPO_ROOT / "runs" / "simulation" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    cache_dir = output_dir / "masks"
    cache_dir.mkdir()
    shutil.copy2(before_path, output_dir / "before.jpg")
    shutil.copy2(after_path, output_dir / "after.jpg")

    if not args.refresh_masks:
        validate_mask_cache(args.mask_cache_dir, args.yolo_checkpoint, args.conf, args.imgsz)

    yolo = None

    def run_yolo(image_path: Path) -> list[dict[str, Any]]:
        nonlocal yolo
        if yolo is None:
            from ultralytics import YOLO

            print(f"\nLoading segmentation model: {args.yolo_checkpoint}")
            yolo = YOLO(str(args.yolo_checkpoint))
        return segment_image(yolo, image_path, args.conf, args.imgsz)

    before_detections, before_source = resolve_detections(
        before_path, args.mask_cache_dir, cache_dir, run_yolo, args.refresh_masks
    )
    after_detections, after_source = resolve_detections(
        after_path, args.mask_cache_dir, cache_dir, run_yolo, args.refresh_masks
    )
    print(f"\nSegmentation masks: before={before_source}, after={after_source}")
    save_overlay(before_path, before_detections, output_dir / "before_segmentation.jpg")
    save_overlay(after_path, after_detections, output_dir / "after_segmentation.jpg")

    pair["before"] = str(before_path)
    pair["after"] = str(after_path)
    one_pair_manifest = output_dir / "selected_pair.json"
    one_pair_manifest.write_text(json.dumps([pair], indent=2), encoding="utf-8")
    dataset = ConsumptionPairDataset(one_pair_manifest, mask_cache_dir=cache_dir, train_mode=False)
    if not dataset.samples:
        raise RuntimeError("No inferable items remain. Check the YOLO detections in the output mask JSON files.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Running Siamese model on {device}: {args.siamese_checkpoint}")
    model = load_model(args, device)
    predictions: list[dict[str, Any]] = []
    with torch.inference_mode():
        for sample in dataset.samples:
            before = dataset._load_tensor(sample["before"], sample["classes"]).unsqueeze(0).to(device)
            after = dataset._load_tensor(sample["after"], sample["classes"]).unsqueeze(0).to(device)
            metric_id = torch.tensor([metric_index(sample["metric_name"])], device=device)
            aux_features = mask_statistics(before[0, 3], after[0, 3]).unsqueeze(0)
            reg_out, clf_logit = model(before, after, metric_id, aux_features)
            value = torch.clamp(reg_out, 0, 1) if sample["task"] == "regression" else torch.sigmoid(clf_logit)
            predictions.append({
                "item": sample["metric_name"],
                "task": sample["task"],
                "classes": sample["classes"],
                "predicted_consumed_pct": round(float(value.item()) * 100, 2),
                "manifest_ground_truth_pct": float(sample["target_pct"]),
            })

    result = {
        "input_image": str(args.image),
        "before_image": str(before_path),
        "after_image": str(after_path),
        "source_manifest": selected["manifest"],
        "backbone": args.backbone,
        "head": args.head,
        "predictions": predictions,
    }
    result_path = output_dir / "predictions.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path = write_ui_report(output_dir, predictions, args.backbone, args.head)

    print("\nPredicted consumption:")
    for prediction in predictions:
        print(f"  {prediction['item']:<28} {prediction['predicted_consumed_pct']:>6.2f}% "
              f"(manifest GT {prediction['manifest_ground_truth_pct']:.2f}%)")
    print(f"\nSimulation artifacts: {output_dir.resolve()}")
    print(f"Predictions JSON:      {result_path.resolve()}")
    print(f"Interactive report:    {report_path.resolve()}")


if __name__ == "__main__":
    main()
