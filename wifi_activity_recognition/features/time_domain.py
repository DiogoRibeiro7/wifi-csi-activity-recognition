"""Time-domain feature extraction for CSI sequences."""

from __future__ import annotations

import numpy as np


def compute_rms(signal: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute root mean square along a given axis."""
    return np.sqrt(np.mean(np.square(signal), axis=axis))


def zero_crossing_rate(signal: np.ndarray, axis: int = -1) -> np.ndarray:
    """Calculate the zero crossing rate of the signal."""
    s = np.sign(signal)
    crossings = np.diff(s, axis=axis)
    count = np.sum(crossings != 0, axis=axis)
    n_samples = signal.shape[axis]
    return count / n_samples


__all__ = ["compute_rms", "zero_crossing_rate"]
