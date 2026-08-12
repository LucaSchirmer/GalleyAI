"""Trains the Hybrid ML regression head: frozen-backbone embeddings ->
scikit-learn GradientBoostingRegressor.

Unlike the other regression heads (mlp/cosine/euclidean), this one has no
gradient-trained weights inside the Siamese network at all -- the backbone
runs once, frozen, purely as a feature extractor, and a classical ML model
does the regression on top of engineered distance features. That means:

  - No epochs/optimizer/backprop loop -- just one embedding-extraction pass
    over train + val, then a single sklearn .fit() call.
  - Only applies to REGRESSION task samples (pct_* targets). The
    classification (drinks/extras/cookie) samples aren't handled here --
    they don't have a natural place in "which regression head is best",
    and a GBR classifier would be a separate comparison entirely.
  - Still logged into the SAME MLflow experiment / run-naming scheme as
    train.py, so every regression head -- gradient-trained or not -- shows
    up side by side for RQ2.

Engineered features fed to the GBR, per sample:
    [feat_before, feat_after, |feat_before - feat_after], feat_before * feat_after,
     cosine_similarity, euclidean_distance]
i.e. the same "vector distance" building blocks used by the other heads,
just handed to a classical regressor instead of a learned trunk.

Run from the project root:
    python training/train_hybrid_gbr.py --backbone resnet50
    python training/train_hybrid_gbr.py --backbone convnext_tiny --n-estimators 500
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

from data.siamese_dataset import ConsumptionPairDataset
from models.backbones import build_backbone, list_backbones

TRAIN_MANIFEST = Path("data_pairs_with_splits/train.json")
VAL_MANIFEST = Path("data_pairs_with_splits/val.json")
MLFLOW_EXPERIMENT_NAME = "siamese_consumption"
CHECKPOINT_ROOT = Path("runs/siamese")


def extract_features(backbone, loader, device):
    """Returns (X, y): engineered distance features and 0-1 targets,
    restricted to regression-task samples only (see module docstring)."""
    feats, targets = [], []
    backbone.eval()
    with torch.no_grad():
        for before, after, target, task in loader:
            keep = [i for i, t in enumerate(task) if t == "regression"]
            if not keep:
                continue
            before, after = before[keep].to(device), after[keep].to(device)
            target = target[keep]

            feat_before = backbone(before)
            feat_after = backbone(after)
            diff = torch.abs(feat_before - feat_after)
            prod = feat_before * feat_after
            cosine = F.cosine_similarity(feat_before, feat_after, dim=1, eps=1e-8).unsqueeze(1)
            euclid = torch.norm(feat_before - feat_after, dim=1, keepdim=True)

            batch_feat = torch.cat([feat_before, feat_after, diff, prod, cosine, euclid], dim=1)
            feats.append(batch_feat.cpu().numpy())
            targets.append(target.numpy())

    return np.concatenate(feats, axis=0), np.concatenate(targets, axis=0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", default="resnet50", choices=list_backbones())
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None, help="MLflow run name; defaults to '<backbone>_hybrid_gbr'")
    parser.add_argument("--train-manifest", type=Path, default=TRAIN_MANIFEST)
    parser.add_argument("--val-manifest", type=Path, default=VAL_MANIFEST)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"backbone={args.backbone}  head=hybrid_gbr")

    # train_mode=False for BOTH splits: the GBR sees fixed, non-augmented
    # embeddings -- there's no epoch loop to average random flips over.
    train_ds = ConsumptionPairDataset(args.train_manifest, train_mode=False)
    val_ds = ConsumptionPairDataset(args.val_manifest, train_mode=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # freeze=True is somewhat moot here (no backward pass happens at all),
    # but keeps the backbone construction identical to the gradient-trained path.
    backbone = build_backbone(args.backbone, pretrained=True, freeze=True).to(device)

    print("Extracting train embeddings...")
    X_train, y_train = extract_features(backbone, train_loader, device)
    print("Extracting val embeddings...")
    X_val, y_val = extract_features(backbone, val_loader, device)
    print(f"X_train shape: {X_train.shape}  X_val shape: {X_val.shape}")

    gbr = GradientBoostingRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        random_state=args.seed,
    )

    run_name = args.run_name or f"{args.backbone}_hybrid_gbr"
    checkpoint_dir = CHECKPOINT_ROOT / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or "sqlite:///runs/mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "backbone": args.backbone,
                "head": "hybrid_gbr",
                "n_estimators": args.n_estimators,
                "learning_rate": args.learning_rate,
                "max_depth": args.max_depth,
                "train_samples": int(X_train.shape[0]),
                "val_samples": int(X_val.shape[0]),
                "feature_dim": int(X_train.shape[1]),
                "seed": args.seed,
            }
        )

        gbr.fit(X_train, y_train)

        train_pred = np.clip(gbr.predict(X_train), 0.0, 1.0)
        val_pred = np.clip(gbr.predict(X_val), 0.0, 1.0)

        train_mae = mean_absolute_error(y_train, train_pred) * 100
        val_mae = mean_absolute_error(y_val, val_pred) * 100
        val_rmse = math.sqrt(mean_squared_error(y_val, val_pred)) * 100

        mlflow.log_metrics(
            {
                "train_reg_mae_pct": train_mae,
                "val_reg_mae_pct": val_mae,
                "val_reg_rmse_pct": val_rmse,
            }
        )

        model_path = checkpoint_dir / "gbr.joblib"
        joblib.dump(gbr, model_path)
        mlflow.log_artifact(str(model_path))

        print(f"train MAE: {train_mae:.2f}%  val MAE: {val_mae:.2f}%  val RMSE: {val_rmse:.2f}%")
        print(f"Saved: {model_path}")


if __name__ == "__main__":
    main()
