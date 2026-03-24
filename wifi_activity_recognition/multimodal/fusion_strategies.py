"""Fusion strategies for combining multi-modal sensor data."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def early_fusion(features: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate feature vectors along the last dimension.

    Args:
        features: Sequence of feature arrays with matching leading dimensions.

    Returns:
        Concatenated feature array.
    """
    if not features:
        raise ValueError("No features provided for fusion")
    return np.concatenate(features, axis=-1)


def late_fusion(
    predictions: Sequence[np.ndarray], weights: Sequence[float] | None = None
) -> np.ndarray:
    """Fuse prediction vectors via weighted averaging.

    Args:
        predictions: Sequence of model prediction arrays.
        weights: Optional non-negative weights. If ``None`` each model is
            weighted equally.

    Returns:
        Weighted average of the predictions.
    """
    if not predictions:
        raise ValueError("No predictions provided for fusion")
    stacked = np.stack(predictions)
    if weights is None:
        w = np.ones(len(predictions), dtype=float)
    else:
        if len(weights) != len(predictions):
            raise ValueError("Weights length must match predictions")
        w = np.asarray(weights, dtype=float)
        if np.any(w < 0):
            raise ValueError("Weights must be non-negative")
    w /= w.sum()
    return (w[:, None] * stacked).sum(axis=0)


def hybrid_attention_fusion(
    features: Sequence[np.ndarray], predictions: Sequence[np.ndarray]
) -> np.ndarray:
    """Fuse predictions using attention derived from feature magnitudes.

    The L2 norm of each feature vector is transformed into a softmax weight
    which determines the contribution of each prediction vector.
    """
    if len(features) != len(predictions):
        raise ValueError("Features and predictions must have the same length")
    energies = np.array([np.linalg.norm(f) for f in features])
    if np.all(energies == 0):
        attn = np.ones(len(features)) / len(features)
    else:
        attn = np.exp(energies) / np.exp(energies).sum()
    stacked = np.stack(predictions)
    return (attn[:, None] * stacked).sum(axis=0)


def uncertainty_aware_fusion(
    predictions: Sequence[np.ndarray], uncertainties: Sequence[float]
) -> np.ndarray:
    """Fuse predictions using inverse-variance weighting.

    Args:
        predictions: Sequence of prediction vectors.
        uncertainties: Sequence of standard deviations representing the
            uncertainty of each prediction vector. Lower values indicate higher
            confidence.

    Returns:
        Weighted average of the predictions where weights are proportional to
        ``1 / uncertainty^2``.
    """
    if len(predictions) != len(uncertainties):
        raise ValueError("Predictions and uncertainties must have the same length")
    stacked = np.stack(predictions)
    u = np.asarray(uncertainties, dtype=float)
    if np.any(u <= 0):
        raise ValueError("Uncertainties must be positive")
    precisions = 1.0 / (u**2)
    weights = precisions / precisions.sum()
    return (weights[:, None] * stacked).sum(axis=0)
