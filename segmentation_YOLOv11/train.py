"""Train YOLO segmentation on the current prepared dataset.

Run from anywhere in the repository. By default this fine-tunes the previous
best checkpoint; pass ``--model yolo11m-seg.pt`` for a fresh pretrained run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "data_with_splits" / "dataset.yaml"
DEFAULT_MODEL = REPO_ROOT / "runs" / "segment" / "runs" / "segment" / "baseline_v2" / "weights" / "best.pt"
DEFAULT_PROJECT = REPO_ROOT / "runs" / "segment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL),
                        help="checkpoint to fine-tune, or e.g. yolo11m-seg.pt")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="baseline_v3_current")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data.resolve()}")
    if args.model != "yolo11m-seg.pt" and not Path(args.model).exists():
        raise FileNotFoundError(f"Starting checkpoint not found: {Path(args.model).resolve()}")

    print(f"Training data: {args.data.resolve()}")
    print(f"Starting checkpoint: {args.model}")
    print(f"Output: {(args.project / args.name).resolve()}")

    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        amp=True,
        workers=args.workers,
        mask_ratio=2,
        cache=True,
        cos_lr=True,
        seed=42,
        project=str(args.project.resolve()),
        name=args.name,
    )


if __name__ == "__main__":
    main()
