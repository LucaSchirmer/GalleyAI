"""Evaluate a trained segmentation checkpoint on the held-out test split."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data_with_splits" / "dataset.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model))

    metrics = model.val(
        data=str(args.data.resolve()),
        split="test",
        imgsz=args.imgsz,
        device=args.device,
        plots=True,
        save_json=True,
    )

    print("\n── Test Results ──")
    print(f"  mAP50      : {metrics.seg.map50:.3f}")
    print(f"  mAP50-95   : {metrics.seg.map:.3f}")
    print(f"  Precision  : {metrics.seg.mp:.3f}")
    print(f"  Recall     : {metrics.seg.mr:.3f}")
    print("\n── Per-class mask mAP50-95 ──")
    for class_id, class_name in model.names.items():
        print(f"  {class_name:32s}: {metrics.seg.maps[class_id]:.3f}")

if __name__ == "__main__":
    main()
