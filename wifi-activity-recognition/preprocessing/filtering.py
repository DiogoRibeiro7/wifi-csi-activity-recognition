"""Noise filtering utilities."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter


def butterworth_filter(
    data: np.ndarray, cutoff: float, fs: float, order: int = 4, axis: int = 0
) -> np.ndarray:
    """Apply a low-pass Butterworth filter along a specified axis.

    Parameters
    ----------
    data:
        Input array where the chosen axis represents time.
    cutoff:
        Cutoff frequency in Hz.
    fs:
        Sampling frequency in Hz.
    order:
        Filter order.
    axis:
        Axis along which filtering is performed. Defaults to 0.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return lfilter(b, a, data, axis=axis)


def moving_average_filter(
    data: np.ndarray, window_size: int, axis: int = 0
) -> np.ndarray:
    """Smooth data using a moving average."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    kernel = np.ones(window_size) / window_size
    return np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode="same"), axis, data
    )


def kalman_filter(
    data: np.ndarray,
    process_variance: float = 1e-2,
    measurement_variance: float = 1e-2,
    axis: int = 0,
) -> np.ndarray:
    """Filter 1D signals with a simple Kalman filter along a specified axis."""
    data_moved = np.moveaxis(data, axis, 0)
    n_timesteps, *rest = data_moved.shape
    filtered = np.zeros_like(data_moved)
    for idx in np.ndindex(*rest):
        x_est = data_moved[(0,) + idx]
        p_est = 1.0
        filtered[(0,) + idx] = x_est
        for t in range(1, n_timesteps):
            p_est += process_variance
            kalman_gain = p_est / (p_est + measurement_variance)
            x_est += kalman_gain * (data_moved[(t,) + idx] - x_est)
            p_est = (1 - kalman_gain) * p_est
            filtered[(t,) + idx] = x_est
    return np.moveaxis(filtered, 0, axis)
