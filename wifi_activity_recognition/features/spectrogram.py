"""Spectrogram feature utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import stft


def compute_spectrogram(
    signal: np.ndarray,
    fs: float = 1.0,
    nperseg: int = 64,
    noverlap: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute magnitude spectrogram of a 1-D signal.

    Parameters
    ----------
    signal:
        Input time-domain signal.
    fs:
        Sampling frequency of the signal.
    nperseg:
        Length of each segment for the STFT.
    noverlap:
        Number of points to overlap between segments. Defaults to ``nperseg // 2``.

    Returns
    -------
    f, t, spec : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Frequencies, times, and magnitude of the STFT.
    """
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, t, np.abs(Zxx)


__all__ = ["compute_spectrogram"]
