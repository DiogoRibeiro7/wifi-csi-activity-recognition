"""Evaluation metrics for activity recognition models."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], num_classes: int | None = None
) -> Dict[str, Any]:
    """Return classification metrics including per-class accuracy and confusion matrix.

    Parameters
    ----------
    y_true:
        Ground-truth integer class labels.
    y_pred:
        Predicted integer class labels.
    num_classes:
        Optional total number of classes. When ``None`` it is inferred from
        ``y_true`` and ``y_pred``.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing overall accuracy, precision, recall, F1 score,
        per-class accuracy list, and confusion matrix.
    """
    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "per_class_accuracy": [],
            "confusion_matrix": [],
        }

    if num_classes is None:
        num_classes = int(max(max(y_true), max(y_pred)) + 1)
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.divide(
            np.diag(cm),
            cm.sum(axis=1),
            out=np.zeros(num_classes),
            where=cm.sum(axis=1) != 0,
        )

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "per_class_accuracy": per_class_acc.tolist(),
        "confusion_matrix": cm.tolist(),
    }


__all__ = ["classification_metrics"]
