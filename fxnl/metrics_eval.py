"""Classification metrics for in-sample vs out-of-sample rows."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)


def _safe_log_loss(y_true: np.ndarray, proba: np.ndarray, *, labels: list[int]) -> float:
    try:
        return float(log_loss(y_true, proba, labels=labels))
    except Exception:
        return float("nan")


def eval_split(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None,
    *,
    n_classes: int,
) -> dict[str, float]:
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    labels = list(range(n_classes))
    if n_classes == 2 and proba is not None and proba.shape[1] >= 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
        except Exception:
            out["roc_auc"] = float("nan")
        out["log_loss"] = _safe_log_loss(y_true, proba, labels=labels)
        out["f1_macro"] = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
    elif n_classes > 2 and proba is not None:
        out["roc_auc"] = float("nan")
        out["log_loss"] = _safe_log_loss(y_true, proba, labels=labels)
        out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    else:
        out["roc_auc"] = float("nan")
        out["log_loss"] = float("nan")
        out["f1_macro"] = float(
            f1_score(y_true, y_pred, average="macro" if n_classes > 2 else "binary", zero_division=0),
        )
    return out


def flatten_result_row(
    model_name: str,
    *,
    pair: str,
    lead: int,
    target_mode: str,
    n_train: int,
    n_test: int,
    n_features: int,
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model_name,
        "pair": pair,
        "lead": lead,
        "target": target_mode,
        "n_train": n_train,
        "n_test": n_test,
        "n_features": n_features,
        "train_accuracy": train_metrics.get("accuracy"),
        "test_accuracy": test_metrics.get("accuracy"),
        "train_balanced_acc": train_metrics.get("balanced_accuracy"),
        "test_balanced_acc": test_metrics.get("balanced_accuracy"),
        "train_log_loss": train_metrics.get("log_loss"),
        "test_log_loss": test_metrics.get("log_loss"),
        "train_f1_macro": train_metrics.get("f1_macro"),
        "test_f1_macro": test_metrics.get("f1_macro"),
        "train_roc_auc": train_metrics.get("roc_auc"),
        "test_roc_auc": test_metrics.get("roc_auc"),
    }
    if extra:
        for k, v in extra.items():
            row[f"extra_{k}"] = v
    return row
