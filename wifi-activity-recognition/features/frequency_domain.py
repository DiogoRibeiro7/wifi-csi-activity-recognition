"""Frequency-domain features for CSI sequences."""

from __future__ import annotations

import numpy as np


def compute_fft(signal: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute the one-dimensional FFT along the specified axis."""
    return np.fft.rfft(signal, axis=axis)


def power_spectrum(signal: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute the power spectrum of the signal."""
    fft_vals = compute_fft(signal, axis=axis)
    return np.abs(fft_vals) ** 2


__all__ = ["compute_fft", "power_spectrum"]
