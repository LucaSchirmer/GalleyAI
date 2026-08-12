"""Shared train/eval loop for any (backbone, head) combination produced by
models.siamese_net.SiameseConsumptionNet.

Every batch mixes regression samples (pct_* fields) and classification
samples (drinks/extras/cookie) -- every sample runs through BOTH heads
(cheap), but only the matching head's output contributes to that sample's
loss: MSE against the regression head for regression samples, BCE-with-
logits against the classification head for classification samples. The
two per-task losses are summed (unweighted) for backprop; both are also
logged separately so you can see if one task is dominating or stalling.

This is backbone/head-agnostic on purpose: model(before, after) always
returns (reg_out, clf_logit) regardless of which backbone or regression
head was chosen, so this loop doesn't need to know or care which one is
active.
"""

from __future__ import annotations

import torch


def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    mse_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()

    total_loss = 0.0
    reg_mae_sum, reg_n = 0.0, 0
    clf_correct_sum, clf_n = 0.0, 0
    n = 0

    with torch.set_grad_enabled(is_train):
        for before, after, target, task in loader:
            before, after, target = before.to(device), after.to(device), target.to(device)
            is_regression = torch.tensor([t == "regression" for t in task], device=device)
            is_classification = ~is_regression

            reg_out, clf_logit = model(before, after)

            loss = torch.zeros((), device=device)
            if is_regression.any():
                loss = loss + mse_loss(reg_out[is_regression], target[is_regression])
            if is_classification.any():
                loss = loss + bce_loss(clf_logit[is_classification], target[is_classification])

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = target.size(0)
            total_loss += loss.item() * batch_size
            n += batch_size

            if is_regression.any():
                reg_clamped = torch.clamp(reg_out[is_regression], 0.0, 1.0)
                reg_mae_sum += torch.abs(reg_clamped - target[is_regression]).sum().item() * 100
                reg_n += int(is_regression.sum().item())

            if is_classification.any():
                clf_pred_label = torch.sigmoid(clf_logit[is_classification]) >= 0.5
                clf_true_label = target[is_classification] >= 0.5
                clf_correct_sum += (clf_pred_label == clf_true_label).sum().item()
                clf_n += int(is_classification.sum().item())

    metrics = {
        "loss": total_loss / n if n else float("nan"),
        "reg_mae": reg_mae_sum / reg_n if reg_n else float("nan"),
        "clf_acc": clf_correct_sum / clf_n if clf_n else float("nan"),
    }
    return metrics
