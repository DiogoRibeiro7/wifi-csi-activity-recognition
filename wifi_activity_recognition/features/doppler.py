"""Doppler feature computation for CSI sequences."""

from __future__ import annotations

import numpy as np


def doppler_spectrum(signal: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute Doppler spectrum via FFT along time axis."""
    fft_vals = np.fft.fftshift(np.fft.fft(signal, axis=axis), axes=axis)
    return np.abs(fft_vals)


__all__ = ["doppler_spectrum"]
