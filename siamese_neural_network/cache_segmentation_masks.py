"""Runs the trained YOLO segmentation model once over every image (both
"before"/unconsumed and "after"/consumed) and caches the detections to disk.

This is a deliberate separate offline step: the Siamese training loop will
run for many epochs over the same images, and re-running YOLO inference
every epoch would be enormously wasteful. Cache once here, then the
training Dataset just reads small JSON files.

Output: one JSON per image stem in MASK_CACHE_DIR, e.g.
  mask_cache/all_markers_shot_20260528_150411.json
  {
    "detections": [
      {"class": "rice", "polygon": [[x, y], ...], "bbox": [x1, y1, x2, y2]}
      ...
    ]
  }
polygon points and bbox are normalized to [0, 1] (fraction of image width/height).

Run:
    python scripts/cache_segmentation_masks.py
"""

from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

MODEL_PATH = Path("runs/mlflow/593474382547047869/693ece4bf847448cb7ffbb1019c88244/artifacts/weights/best.pt")  # your trained YOLO checkpoint

IMAGE_DIRS = [
    Path("data/images"),          # "before" / unconsumed
    Path("data_consumed/images"), # "after" / consumed
]

MASK_CACHE_DIR = Path("mask_cache")

CONF_THRESHOLD = 0.25
IMG_SIZE = 640  # match training imgsz
FORCE_RECOMPUTE = False  # set True to overwrite existing cache entries


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CORE LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def suffix_stem(name: str) -> str:
    """Strips a Label-Studio-style hash prefix like 'a5e6c40__' from a filename stem."""
    stem = Path(name).stem
    if "__" in stem:
        return stem.split("__", 1)[1]
    return stem


def cache_one_image(model: YOLO, image_path: Path, out_path: Path) -> None:
    result = model.predict(
        source=str(image_path),
        conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE,
        verbose=False,
    )[0]

    detections = []
    if result.masks is not None:
        names = result.names
        classes = result.boxes.cls.tolist()
        polygons = result.masks.xyn  # list of Nx2 arrays, already normalized [0,1]
        bboxes = result.boxes.xyxyn.tolist()  # normalized [x1,y1,x2,y2]

        for cls_idx, polygon, bbox in zip(classes, polygons, bboxes):
            detections.append(
                {
                    "class": names[int(cls_idx)],
                    "polygon": polygon.tolist(),
                    "bbox": bbox,
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump({"detections": detections}, fh)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"YOLO checkpoint not found: {MODEL_PATH.resolve()}")

    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(str(MODEL_PATH))

    total = 0
    skipped_cached = 0
    processed = 0

    for images_dir in IMAGE_DIRS:
        if not images_dir.exists():
            raise FileNotFoundError(f"Images dir not found: {images_dir.resolve()}")

        image_paths = sorted(
            p for p in images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        print(f"\n{images_dir}: {len(image_paths)} image(s)")

        for image_path in image_paths:
            total += 1
            stem = suffix_stem(image_path.name)
            out_path = MASK_CACHE_DIR / f"{stem}.json"

            if out_path.exists() and not FORCE_RECOMPUTE:
                skipped_cached += 1
                continue

            cache_one_image(model, image_path, out_path)
            processed += 1
            if processed % 25 == 0:
                print(f"  ...{processed} newly cached")

    print(f"\nDone. total images seen: {total}, newly cached: {processed}, "
          f"already cached (skipped): {skipped_cached}")
    print(f"Cache written to: {MASK_CACHE_DIR.resolve()}")


if __name__ == "__main__":
    main()
