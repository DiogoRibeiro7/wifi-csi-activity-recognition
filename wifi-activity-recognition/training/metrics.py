"""Evaluation metrics for activity recognition models."""

from __future__ import annotations

from typing import Dict, Sequence

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> Dict[str, float]:
    """Return accuracy, precision, recall, and F1 scores."""
    if len(y_true) == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


__all__ = ["classification_metrics"]
