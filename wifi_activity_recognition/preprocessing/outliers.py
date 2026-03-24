"""Outlier detection utilities for CSI preprocessing."""

from __future__ import annotations

import numpy as np


def detect_outliers_zscore(
    data: np.ndarray, threshold: float = 3.0, axis: int | None = 0
) -> np.ndarray:
    """Identify outliers using a z-score threshold.

    Parameters
    ----------
    data:
        Input array to evaluate.
    threshold:
        Z-score above which a sample is considered an outlier.
    axis:
        Axis along which to compute statistics. ``None`` flattens the array.

    Returns
    -------
    np.ndarray
        Boolean mask with ``True`` marking outliers.
    """
    median = np.median(data, axis=axis, keepdims=True)
    mad = np.median(np.abs(data - median), axis=axis, keepdims=True)
    mad = np.where(mad == 0, 1, mad)
    modified_z = 0.6745 * np.abs(data - median) / mad
    return modified_z > threshold


def remove_outliers_zscore(
    data: np.ndarray, threshold: float = 3.0, axis: int | None = 0
) -> np.ndarray:
    """Replace outliers with ``NaN`` based on z-score detection."""
    mask = detect_outliers_zscore(data, threshold=threshold, axis=axis)
    cleaned = data.astype(float).copy()
    cleaned[mask] = np.nan
    return cleaned


__all__ = ["detect_outliers_zscore", "remove_outliers_zscore"]
