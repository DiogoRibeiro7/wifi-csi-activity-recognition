"""Noise filtering utilities for :class:`CSIData`."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np
from scipy.signal import butter, lfilter

from ..hardware.base import CSIData


def butterworth_filter(
    csi: CSIData,
    cutoff: float,
    fs: float,
    order: int = 4,
    axis: int = -1,
    fields: Iterable[str] = ("amplitude",),
) -> CSIData:
    """Apply a low-pass Butterworth filter to selected fields.

    Parameters
    ----------
    csi:
        Input CSI sample.
    cutoff:
        Cutoff frequency in Hz.
    fs:
        Sampling frequency in Hz.
    order:
        Filter order.
    axis:
        Axis along which filtering is performed. Defaults to the last axis.
    fields:
        Iterable of fields to filter. Defaults to ``("amplitude",)``.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    updates = {}
    for field in fields:
        data = getattr(csi, field)
        updates[field] = lfilter(b, a, data, axis=axis)
    return replace(csi, **updates)


def moving_average_filter(
    csi: CSIData,
    window_size: int,
    axis: int = -1,
    fields: Iterable[str] = ("amplitude",),
) -> CSIData:
    """Smooth fields using a moving average."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    kernel = np.ones(window_size) / window_size
    updates = {}
    for field in fields:
        data = getattr(csi, field)
        updates[field] = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="same"), axis, data
        )
    return replace(csi, **updates)


def kalman_filter(
    csi: CSIData,
    process_variance: float = 1e-2,
    measurement_variance: float = 1e-2,
    axis: int = -1,
    fields: Iterable[str] = ("amplitude",),
) -> CSIData:
    """Filter 1D signals with a simple Kalman filter."""
    updates = {}
    for field in fields:
        data = getattr(csi, field)
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
        updates[field] = np.moveaxis(filtered, 0, axis)
    return replace(csi, **updates)
