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
    python training/train.py --skip-fish-rice-veg

List all options:
    python training/train.py --help
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

# PROJECT_ROOT anchors things that belong INSIDE this package regardless of
# cwd (checkpoints, the MLflow db). Your actual dataset files (train.json /
# val.json / mask_cache/) live at your repo root instead -- one level ABOVE
# this package -- and you run this script from there, so those default to
# plain cwd-relative paths, same as your original script. Override any of
# them with --train-manifest / --val-manifest / --mask-cache-dir if your
# data ever moves.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.siamese_dataset import ConsumptionPairDataset
from models.backbones import list_backbones
from models.siamese_net import SiameseConsumptionNet, list_regression_heads
from training.engine import run_epoch

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION (defaults -- override any of these via CLI flags)
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_MANIFEST = Path("data_pairs_with_splits/train.json")
VAL_MANIFEST = Path("data_pairs_with_splits/val.json")
MASK_CACHE_DIR = Path("mask_cache")
MLFLOW_EXPERIMENT_NAME = "siamese_consumption"
MLFLOW_TRACKING_URI_DEFAULT = f"sqlite:///{PROJECT_ROOT / 'runs' / 'mlflow' / 'mlflow.db'}"
CHECKPOINT_ROOT = PROJECT_ROOT / "runs" / "siamese"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", default="resnet50", choices=list_backbones())
    parser.add_argument("--head", default="mlp", choices=list_regression_heads())
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30, help="early-stop patience (epochs w/o val improvement)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-5,
                        help="learning rate for trainable pretrained backbone parameters")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-hidden-dims", type=int, nargs=2, metavar=("H1", "H2"), default=(256, 64),
                        help="hidden widths for the MLP regression head")
    parser.add_argument("--mlp-dropout", type=float, default=0.3)
    parser.add_argument("--regression-loss", choices=("mse", "huber"), default="mse")
    parser.add_argument("--huber-beta", type=float, default=0.1,
                        help="SmoothL1 transition in normalized 0-1 target units")
    parser.add_argument("--no-freeze-backbone", action="store_true", help="fine-tune the whole backbone, not just the patched first conv")
    parser.add_argument("--unfreeze-last-blocks", type=int, default=0,
                        help="unfreeze only the last N ViT/Swin blocks (recommended: 1 or 2)")
    parser.add_argument("--metric-embedding-dim", type=int, default=16)
    parser.add_argument("--no-metric-embedding", action="store_true",
                        help="ablation: omit explicit metric identity")
    parser.add_argument("--no-aux-features", action="store_true",
                        help="ablation: omit explicit mask geometry")
    parser.add_argument("--balanced-sampling", action="store_true",
                        help="sample task/metric/target strata with inverse frequency")
    parser.add_argument("--skip-fish-rice-veg", action="store_true",
                        help="exclude pct_fish_rice_veg regression samples from train and validation")
    parser.add_argument("--no-pretrained", action="store_true", help="train backbone from scratch (no ImageNet weights)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None, help="MLflow run name; defaults to '<backbone>_<head>'")
    parser.add_argument("--train-manifest", type=Path, default=TRAIN_MANIFEST)
    parser.add_argument("--val-manifest", type=Path, default=VAL_MANIFEST)
    parser.add_argument("--mask-cache-dir", type=Path, default=MASK_CACHE_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.no_freeze_backbone and args.unfreeze_last_blocks:
        raise ValueError("Choose either --no-freeze-backbone or --unfreeze-last-blocks, not both")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"backbone={args.backbone}  head={args.head}")

    excluded_metrics = ("pct_fish_rice_veg",) if args.skip_fish_rice_veg else ()
    train_ds = ConsumptionPairDataset(
        args.train_manifest, mask_cache_dir=args.mask_cache_dir, train_mode=True,
        excluded_metric_names=excluded_metrics,
    )
    val_ds = ConsumptionPairDataset(
        args.val_manifest, mask_cache_dir=args.mask_cache_dir, train_mode=False,
        excluded_metric_names=excluded_metrics,
    )
    train_sampler = None
    if args.balanced_sampling:
        generator = torch.Generator().manual_seed(args.seed)
        train_sampler = WeightedRandomSampler(
            train_ds.balanced_sample_weights(),
            num_samples=len(train_ds),
            replacement=True,
            generator=generator,
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    regression_head_kwargs = {}
    if args.head == "mlp":
        regression_head_kwargs = {
            "trunk_hidden": tuple(args.mlp_hidden_dims),
            "dropout": args.mlp_dropout,
        }

    model = SiameseConsumptionNet(
        backbone_name=args.backbone,
        head_name=args.head,
        pretrained=not args.no_pretrained,
        freeze_backbone=not args.no_freeze_backbone,
        unfreeze_last_blocks=args.unfreeze_last_blocks,
        metric_embedding_dim=args.metric_embedding_dim,
        use_metric_embedding=not args.no_metric_embedding,
        use_aux_features=not args.no_aux_features,
        regression_head_kwargs=regression_head_kwargs,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.optimizer_param_groups(head_lr=args.lr, backbone_lr=args.backbone_lr),
        weight_decay=args.weight_decay,
    )

    filter_tag = "_no_fish_rice_veg" if args.skip_fish_rice_veg else ""
    run_name = args.run_name or f"{args.backbone}_{args.head}{filter_tag}"
    checkpoint_dir = CHECKPOINT_ROOT / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or MLFLOW_TRACKING_URI_DEFAULT
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    best_val_loss = float("inf")
    best_val_clf_balanced_acc = float("-inf")
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
                "backbone_learning_rate": args.backbone_lr,
                "weight_decay": args.weight_decay,
                "mlp_hidden_dims": ",".join(map(str, args.mlp_hidden_dims)) if args.head == "mlp" else "n/a",
                "mlp_dropout": args.mlp_dropout if args.head == "mlp" else "n/a",
                "regression_loss": args.regression_loss,
                "huber_beta": args.huber_beta if args.regression_loss == "huber" else "n/a",
                "freeze_backbone": not args.no_freeze_backbone,
                "unfreeze_last_blocks": args.unfreeze_last_blocks,
                "metric_embedding_dim": args.metric_embedding_dim,
                "metric_embedding": not args.no_metric_embedding,
                "auxiliary_mask_features": not args.no_aux_features,
                "balanced_sampling": args.balanced_sampling,
                "skip_fish_rice_veg": args.skip_fish_rice_veg,
                "pretrained": not args.no_pretrained,
                "img_size": train_ds.img_size,
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "seed": args.seed,
            }
        )

        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, device, optimizer, args.regression_loss, args.huber_beta)
            val_metrics = run_epoch(model, val_loader, device, optimizer=None,
                                    regression_loss=args.regression_loss, huber_beta=args.huber_beta)

            mlflow.log_metrics(
                {
                    "train_loss": train_metrics["loss"],
                    "train_reg_mae_pct": train_metrics["reg_mae"],
                    "train_clf_acc": train_metrics["clf_acc"],
                    "train_clf_balanced_acc": train_metrics["clf_balanced_acc"],
                    "val_loss": val_metrics["loss"],
                    "val_reg_mae_pct": val_metrics["reg_mae"],
                    "val_clf_acc": val_metrics["clf_acc"],
                    "val_clf_balanced_acc": val_metrics["clf_balanced_acc"],
                    "val_clf_recall_consumed": val_metrics["clf_recall_consumed"],
                    "val_clf_recall_not_consumed": val_metrics["clf_recall_not_consumed"],
                },
                step=epoch,
            )
            print(
                f"epoch {epoch:3d}/{args.epochs}  "
                f"train_loss={train_metrics['loss']:.4f} reg_mae={train_metrics['reg_mae']:.2f}% "
                f"clf_acc={train_metrics['clf_acc']*100:.1f}% "
                f"clf_bal_acc={train_metrics['clf_balanced_acc']*100:.1f}%  |  "
                f"val_loss={val_metrics['loss']:.4f} reg_mae={val_metrics['reg_mae']:.2f}% "
                f"clf_acc={val_metrics['clf_acc']*100:.1f}% "
                f"clf_bal_acc={val_metrics['clf_balanced_acc']*100:.1f}%"
            )

            val_clf_balanced_acc = val_metrics["clf_balanced_acc"]
            if not math.isnan(val_clf_balanced_acc) and val_clf_balanced_acc > best_val_clf_balanced_acc:
                best_val_clf_balanced_acc = val_clf_balanced_acc
                best_clf_path = checkpoint_dir / "best_classification.pt"
                torch.save(model.state_dict(), best_clf_path)

            # Regression MAE is the primary outcome and the reported model
            # comparison metric, so do not let classification BCE decide
            # which regression checkpoint is retained.
            val_loss = val_metrics["reg_mae"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_path = checkpoint_dir / "best.pt"
                torch.save(model.state_dict(), best_path)
                torch.save(model.state_dict(), checkpoint_dir / "best_regression.pt")
                mlflow.log_artifact(str(best_path))
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    print(f"Early stopping at epoch {epoch} (no val improvement in {args.patience} epochs)")
                    break

        mlflow.log_metric("best_val_reg_mae_pct", best_val_loss)
        if best_val_clf_balanced_acc != float("-inf"):
            mlflow.log_metric("best_val_clf_balanced_acc", best_val_clf_balanced_acc)
            mlflow.log_artifact(str(checkpoint_dir / "best_classification.pt"))
        mlflow.log_artifact(str(checkpoint_dir / "best_regression.pt"))
        print(f"\nBest val_reg_mae: {best_val_loss:.2f}%")
        print(f"Best regression checkpoint: {checkpoint_dir / 'best_regression.pt'}")
        print(f"Best classification checkpoint: {checkpoint_dir / 'best_classification.pt'}")
        print(f"Compatibility alias for regression reports: {checkpoint_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
