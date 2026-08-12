"""Trains the Siamese before/after consumption model (two-head version).

Each batch mixes regression samples (pct_* fields) and classification
samples (drinks/extras/cookie) — every sample runs through BOTH heads
(cheap), but only the matching head's output contributes to that sample's
loss: MSE against the regression head for regression samples, BCE-with-
logits against the classification head for classification samples. The
two task losses have configurable weights for backprop and are logged
separately so you can see if one task is dominating or stalling.

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
PATIENCE = 30
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
FREEZE_BACKBONE = True
NUM_WORKERS = 2
SEED = 42

# Each task loss is averaged over the samples of that task in a batch, then
# weighted here.  Keeping both at 1.0 gives the two tasks equal importance
# independent of how many examples of each happen to occur in a batch.
# Tune these only against the task-specific validation metrics below.
REGRESSION_LOSS_WEIGHT = 1.0
CLASSIFICATION_LOSS_WEIGHT = 1.0

MLFLOW_EXPERIMENT_NAME = "siamese_consumption"
MLFLOW_RUN_NAME = "resnet50_frozen_v3_twohead"

CHECKPOINT_DIR = Path("runs/siamese") / MLFLOW_RUN_NAME
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL LOOPS
# ══════════════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    mse_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()

    reg_mse_sum, reg_loss_n = 0.0, 0
    clf_bce_sum, clf_loss_n = 0.0, 0
    reg_mae_sum, reg_n = 0.0, 0
    clf_correct_sum, clf_n = 0.0, 0

    with torch.set_grad_enabled(is_train):
        for before, after, target, task in loader:
            before, after, target = before.to(device), after.to(device), target.to(device)
            is_regression = torch.tensor([t == "regression" for t in task], device=device)
            is_classification = ~is_regression

            reg_out, clf_logit = model(before, after)

            # Mean each loss over its own task samples, not the whole batch.
            # This avoids a task being implicitly down-weighted merely because
            # it has fewer examples in this particular batch.
            reg_loss = None
            clf_loss = None
            loss = torch.zeros((), device=device)
            if is_regression.any():
                reg_loss = mse_loss(reg_out[is_regression], target[is_regression])
                loss = loss + REGRESSION_LOSS_WEIGHT * reg_loss
            if is_classification.any():
                clf_loss = bce_loss(clf_logit[is_classification], target[is_classification])
                loss = loss + CLASSIFICATION_LOSS_WEIGHT * clf_loss

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if reg_loss is not None:
                reg_count = int(is_regression.sum().item())
                reg_mse_sum += reg_loss.item() * reg_count
                reg_loss_n += reg_count
            if clf_loss is not None:
                clf_count = int(is_classification.sum().item())
                clf_bce_sum += clf_loss.item() * clf_count
                clf_loss_n += clf_count

            if is_regression.any():
                reg_clamped = torch.clamp(reg_out[is_regression], 0.0, 1.0)
                reg_mae_sum += torch.abs(reg_clamped - target[is_regression]).sum().item() * 100
                reg_n += int(is_regression.sum().item())

            if is_classification.any():
                clf_pred_label = torch.sigmoid(clf_logit[is_classification]) >= 0.5
                clf_true_label = target[is_classification] >= 0.5
                clf_correct_sum += (clf_pred_label == clf_true_label).sum().item()
                clf_n += int(is_classification.sum().item())

    reg_mse = reg_mse_sum / reg_loss_n if reg_loss_n else float("nan")
    clf_bce = clf_bce_sum / clf_loss_n if clf_loss_n else float("nan")
    objective_terms = []
    if reg_loss_n:
        objective_terms.append(REGRESSION_LOSS_WEIGHT * reg_mse)
    if clf_loss_n:
        objective_terms.append(CLASSIFICATION_LOSS_WEIGHT * clf_bce)

    metrics = {
        # Compute this from epoch-wide task means, rather than averaging
        # batch losses whose regression/classification proportions differ.
        "loss": sum(objective_terms) if objective_terms else float("nan"),
        "reg_mse": reg_mse,
        "clf_bce": clf_bce,
        "reg_mae": reg_mae_sum / reg_n if reg_n else float("nan"),
        "clf_acc": clf_correct_sum / clf_n if clf_n else float("nan"),
    }
    return metrics


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
                "regression_loss_weight": REGRESSION_LOSS_WEIGHT,
                "classification_loss_weight": CLASSIFICATION_LOSS_WEIGHT,
                "freeze_backbone": FREEZE_BACKBONE,
                "backbone": "resnet50",
                "head": "two_head_shared_trunk",  # regression (MSE) + classification (BCE)
                "img_size": train_ds.img_size,
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "seed": SEED,
            }
        )

        for epoch in range(1, EPOCHS + 1):
            train_metrics = run_epoch(model, train_loader, device, optimizer)
            val_metrics = run_epoch(model, val_loader, device, optimizer=None)

            mlflow.log_metrics(
                {
                    "train_loss": train_metrics["loss"],
                    "train_reg_mse": train_metrics["reg_mse"],
                    "train_clf_bce": train_metrics["clf_bce"],
                    "train_reg_mae_pct": train_metrics["reg_mae"],
                    "train_clf_acc": train_metrics["clf_acc"],
                    "val_loss": val_metrics["loss"],
                    "val_reg_mse": val_metrics["reg_mse"],
                    "val_clf_bce": val_metrics["clf_bce"],
                    "val_reg_mae_pct": val_metrics["reg_mae"],
                    "val_clf_acc": val_metrics["clf_acc"],
                },
                step=epoch,
            )
            print(
                f"epoch {epoch:3d}/{EPOCHS}  "
                f"train_loss={train_metrics['loss']:.4f} reg_mae={train_metrics['reg_mae']:.2f}% "
                f"clf_acc={train_metrics['clf_acc']*100:.1f}%  |  "
                f"val_loss={val_metrics['loss']:.4f} reg_mae={val_metrics['reg_mae']:.2f}% "
                f"clf_acc={val_metrics['clf_acc']*100:.1f}%"
            )

            val_loss = val_metrics["loss"]
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
