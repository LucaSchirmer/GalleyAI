"""Trains the Siamese before/after consumption-percentage regressor.

Uses the same MLflow store as the YOLO training runs (runs/mlflow by
default, or MLFLOW_TRACKING_URI if set) so both show up in the same
`mlflow server --backend-store-uri runs/mlflow` UI, just under a
different experiment name.

Run:
    python scripts/train_siamese.py
"""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader

from siamese_dataset import ConsumptionPairDataset
from siamese_model import SiameseConsumptionNet


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_MANIFEST = Path("data_pairs_with_splits/train.json")
VAL_MANIFEST = Path("data_pairs_with_splits/val.json")

EPOCHS = 100
PATIENCE = 15          # early stop if val loss hasn't improved in this many epochs
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
FREEZE_BACKBONE = True
NUM_WORKERS = 2
SEED = 42

MLFLOW_EXPERIMENT_NAME = "siamese_consumption"
MLFLOW_RUN_NAME = "resnet50_frozen_v1"

CHECKPOINT_DIR = Path("runs/siamese") / MLFLOW_RUN_NAME
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL LOOPS
# ══════════════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    loss_fn = torch.nn.MSELoss()
    total_loss = 0.0
    total_mae = 0.0
    n = 0

    with torch.set_grad_enabled(is_train):
        for before, after, target in loader:
            before, after, target = before.to(device), after.to(device), target.to(device)

            pred = model(before, after)
            loss = loss_fn(pred, target)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = target.size(0)
            total_loss += loss.item() * batch_size
            total_mae += torch.abs(pred - target).sum().item() * 100  # back to 0-100 scale
            n += batch_size

    return total_loss / n, total_mae / n


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = ConsumptionPairDataset(TRAIN_MANIFEST, train_mode=True)
    val_ds = ConsumptionPairDataset(VAL_MANIFEST, train_mode=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = SiameseConsumptionNet(freeze_backbone=FREEZE_BACKBONE).to(device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or "runs/mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    with mlflow.start_run(run_name=MLFLOW_RUN_NAME):
        mlflow.log_params(
            {
                "epochs": EPOCHS,
                "patience": PATIENCE,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "freeze_backbone": FREEZE_BACKBONE,
                "backbone": "resnet50",
                "head": "concat_mlp",
                "img_size": train_ds.img_size,
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "seed": SEED,
            }
        )

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_mae = run_epoch(model, train_loader, device, optimizer)
            val_loss, val_mae = run_epoch(model, val_loader, device, optimizer=None)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_mae_pct": train_mae,
                    "val_loss": val_loss,
                    "val_mae_pct": val_mae,
                },
                step=epoch,
            )
            print(f"epoch {epoch:3d}/{EPOCHS}  train_loss={train_loss:.4f} "
                  f"train_mae={train_mae:.2f}%  val_loss={val_loss:.4f} val_mae={val_mae:.2f}%")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_path = CHECKPOINT_DIR / "best.pt"
                torch.save(model.state_dict(), best_path)
                mlflow.log_artifact(str(best_path))
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE:
                    print(f"Early stopping at epoch {epoch} (no val improvement in {PATIENCE} epochs)")
                    break

        mlflow.log_metric("best_val_loss", best_val_loss)
        print(f"\nBest val_loss: {best_val_loss:.4f}")
        print(f"Best checkpoint: {CHECKPOINT_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
