"""Trains the Siamese before/after consumption model: pick any backbone +
any gradient-trained regression head via CLI flags. Everything else about
the training loop stays identical, and every run logs to the same MLflow
experiment for direct comparison (RQ2: "which architecture performs best").

Backbones:  resnet50, convnext_tiny, mobilenetv3_large, vit_b16, swin_t
Heads:      mlp, cosine, euclidean

Note: the Hybrid ML head (frozen embeddings -> Gradient Boosting Regressor)
is NOT selectable here -- it has no gradient-trained weights at all, so it
lives in its own script, train_hybrid_gbr.py (see that file's docstring
for why). Every other head is trained here.

Run from the project root:
    python training/train.py --backbone resnet50 --head mlp
    python training/train.py --backbone convnext_tiny --head cosine
    python training/train.py --backbone vit_b16 --head euclidean --epochs 50
    python training/train.py --backbone swin_t --head mlp --batch-size 8

List all options:
    python training/train.py --help
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as `python training/train.py` from the project root without
# installing the project as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import torch
from torch.utils.data import DataLoader

from data.siamese_dataset import ConsumptionPairDataset
from models.backbones import list_backbones
from models.siamese_net import SiameseConsumptionNet, list_regression_heads
from training.engine import run_epoch

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION (defaults -- override any of these via CLI flags)
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_MANIFEST = Path("data_pairs_with_splits/train.json")
VAL_MANIFEST = Path("data_pairs_with_splits/val.json")
MLFLOW_EXPERIMENT_NAME = "siamese_consumption"
CHECKPOINT_ROOT = Path("runs/siamese")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", default="resnet50", choices=list_backbones())
    parser.add_argument("--head", default="mlp", choices=list_regression_heads())
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30, help="early-stop patience (epochs w/o val improvement)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-freeze-backbone", action="store_true", help="fine-tune the whole backbone, not just the patched first conv")
    parser.add_argument("--no-pretrained", action="store_true", help="train backbone from scratch (no ImageNet weights)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None, help="MLflow run name; defaults to '<backbone>_<head>'")
    parser.add_argument("--train-manifest", type=Path, default=TRAIN_MANIFEST)
    parser.add_argument("--val-manifest", type=Path, default=VAL_MANIFEST)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"backbone={args.backbone}  head={args.head}")

    train_ds = ConsumptionPairDataset(args.train_manifest, train_mode=True)
    val_ds = ConsumptionPairDataset(args.val_manifest, train_mode=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SiameseConsumptionNet(
        backbone_name=args.backbone,
        head_name=args.head,
        pretrained=not args.no_pretrained,
        freeze_backbone=not args.no_freeze_backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_name = args.run_name or f"{args.backbone}_{args.head}"
    checkpoint_dir = CHECKPOINT_ROOT / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or "sqlite:///runs/mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "backbone": args.backbone,
                "head": args.head,
                "epochs": args.epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "freeze_backbone": not args.no_freeze_backbone,
                "pretrained": not args.no_pretrained,
                "img_size": train_ds.img_size,
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "seed": args.seed,
            }
        )

        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, device, optimizer)
            val_metrics = run_epoch(model, val_loader, device, optimizer=None)

            mlflow.log_metrics(
                {
                    "train_loss": train_metrics["loss"],
                    "train_reg_mae_pct": train_metrics["reg_mae"],
                    "train_clf_acc": train_metrics["clf_acc"],
                    "val_loss": val_metrics["loss"],
                    "val_reg_mae_pct": val_metrics["reg_mae"],
                    "val_clf_acc": val_metrics["clf_acc"],
                },
                step=epoch,
            )
            print(
                f"epoch {epoch:3d}/{args.epochs}  "
                f"train_loss={train_metrics['loss']:.4f} reg_mae={train_metrics['reg_mae']:.2f}% "
                f"clf_acc={train_metrics['clf_acc']*100:.1f}%  |  "
                f"val_loss={val_metrics['loss']:.4f} reg_mae={val_metrics['reg_mae']:.2f}% "
                f"clf_acc={val_metrics['clf_acc']*100:.1f}%"
            )

            val_loss = val_metrics["loss"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_path = checkpoint_dir / "best.pt"
                torch.save(model.state_dict(), best_path)
                mlflow.log_artifact(str(best_path))
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    print(f"Early stopping at epoch {epoch} (no val improvement in {args.patience} epochs)")
                    break

        mlflow.log_metric("best_val_loss", best_val_loss)
        print(f"\nBest val_loss: {best_val_loss:.4f}")
        print(f"Best checkpoint: {checkpoint_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
