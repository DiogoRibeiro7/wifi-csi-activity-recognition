"""Advanced feature extraction from :class:`CSIData`."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import entropy, kurtosis, skew

from ..hardware.base import CSIData


def statistical_moments(
    csi: CSIData,
    axis: int = -1,
    field: str = "amplitude",
) -> Dict[str, np.ndarray]:
    """Compute mean, variance, skewness and kurtosis along ``axis``."""
    data = getattr(csi, field)
    moments = {
        "mean": np.mean(data, axis=axis),
        "variance": np.var(data, axis=axis),
        "skewness": skew(data, axis=axis),
        "kurtosis": kurtosis(data, axis=axis),
    }
    return moments


def spectral_entropy(
    csi: CSIData,
    axis: int = -1,
    field: str = "amplitude",
) -> np.ndarray:
    """Compute normalized spectral entropy of the selected field."""
    data = getattr(csi, field)
    psd = np.abs(np.fft.fft(data, axis=axis)) ** 2
    psd /= np.sum(psd, axis=axis, keepdims=True) + 1e-12
    h = entropy(psd, axis=axis)
    return h / np.log(psd.shape[axis])


def sample_entropy(
    csi: CSIData,
    m: int = 2,
    r: float = 0.2,
    axis: int = -1,
    field: str = "amplitude",
) -> np.ndarray:
    """Estimate sample entropy for each sequence along ``axis``."""
    if m <= 0:
        raise ValueError("m must be positive")
    data = getattr(csi, field)
    data_moved = np.moveaxis(data, axis, 0)
    n = data_moved.shape[0]
    flat = data_moved.reshape(n, -1)
    r_vals = r * np.std(flat, axis=0)
    result = np.zeros(flat.shape[1])
    for idx in range(flat.shape[1]):
        series = flat[:, idx]
        r_i = r_vals[idx]

        def _count(m_val: int) -> int:
            cnt = 0
            for i in range(n - m_val):
                template = series[i : i + m_val]
                for j in range(i + 1, n - m_val + 1):
                    if np.max(np.abs(template - series[j : j + m_val])) < r_i:
                        cnt += 1
            return cnt

        B = _count(m)
        A = _count(m + 1)
        result[idx] = -np.log((A + 1e-12) / (B + 1e-12))
    return result.reshape(data_moved.shape[1:])


__all__ = ["statistical_moments", "spectral_entropy", "sample_entropy"]
