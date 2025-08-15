"""CSI normalization utilities."""

from __future__ import annotations

import numpy as np


def min_max_normalize(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """Scale data to the [0, 1] range along a specified axis.

    Parameters
    ----------
    data:
        Input array.
    axis:
        Axis along which to compute the minimum and maximum. Defaults to 0.

    Returns
    -------
    np.ndarray
        Normalized array where each element is scaled between 0 and 1. If the
        input has constant values, zeros are returned to avoid division by
        zero.
    """
    data_min = np.min(data, axis=axis, keepdims=True)
    data_max = np.max(data, axis=axis, keepdims=True)
    denom = data_max - data_min
    denom[denom == 0] = 1
    return (data - data_min) / denom


def z_score_normalize(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """Standardize data using the z-score along a specified axis.

    Parameters
    ----------
    data:
        Input array.
    axis:
        Axis along which to compute the mean and standard deviation.

    Returns
    -------
    np.ndarray
        Array with zero mean and unit variance along the specified axis.
    """
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    std[std == 0] = 1
    return (data - mean) / std


def log_normalize(data: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply logarithmic scaling to amplitudes.

    Parameters
    ----------
    data:
        Input array. Values should be non-negative.
    eps:
        Small constant to avoid log of zero.

    Returns
    -------
    np.ndarray
        Log-scaled array.
    """
    return np.log(np.maximum(data, eps))
