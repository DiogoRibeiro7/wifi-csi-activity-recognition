"""Fractal dimension features for CSI sequences.

Functions in this module estimate fractal dimensions from
:class:`~wifi_activity_recognition.hardware.base.CSIData` without modifying
inputs. Both Higuchi and Katz estimators are provided.

Examples
--------
>>> from wifi_activity_recognition.hardware.base import CSIData
>>> import numpy as np
>>> amp = np.random.randn(1, 1, 64)
>>> phase = np.zeros_like(amp)
>>> csi = CSIData(0.0, amp, phase, 5.0, 20.0, 1, 1, 64)
>>> higuchi_fd(csi).shape
(1, 1)
"""

from __future__ import annotations

import numpy as np

from ..hardware.base import CSIData


def higuchi_fd(
    csi: CSIData,
    kmax: int = 10,
    field: str = "amplitude",
    axis: int = -1,
) -> np.ndarray:
    """Estimate Higuchi's fractal dimension along ``axis``.

    Parameters
    ----------
    csi: CSIData
        Input data.
    kmax: int
        Maximum interval size.
    field: str
        CSI attribute to analyse.
    axis: int
        Axis representing the sequence dimension.
    """
    if kmax < 2:
        raise ValueError("kmax must be at least 2")
    data = getattr(csi, field)
    data_moved = np.moveaxis(data, axis, 0)
    n = data_moved.shape[0]
    flat = data_moved.reshape(n, -1)
    result = np.zeros(flat.shape[1])
    for idx, series in enumerate(flat.T):
        Lk = []
        for k in range(1, kmax + 1):
            Lm = []
            for m in range(k):
                idxs = np.arange(1, int(np.floor((n - m) / k)), dtype=int)
                if len(idxs) == 0:
                    continue
                diffs = np.abs(series[m + idxs * k] - series[m + k * (idxs - 1)])
                norm = (n - 1) / (len(idxs) * k)
                Lm.append(norm * np.sum(diffs))
            if Lm:
                Lk.append(np.mean(Lm) / k)
        if len(Lk) < 2:
            result[idx] = 0.0
        else:
            x = np.log(np.arange(1, len(Lk) + 1))
            y = np.log(Lk)
            result[idx] = np.polyfit(x, y, 1)[0]
    result = result.reshape(data_moved.shape[1:])
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result


def katz_fd(
    csi: CSIData,
    field: str = "amplitude",
    axis: int = -1,
) -> np.ndarray:
    """Compute Katz fractal dimension along ``axis``."""
    data = getattr(csi, field)
    data_moved = np.moveaxis(data, axis, 0)
    n = data_moved.shape[0]
    flat = data_moved.reshape(n, -1)
    result = np.zeros(flat.shape[1])
    for idx, series in enumerate(flat.T):
        diff = np.abs(np.diff(series))
        L = np.sum(diff)
        d = np.max(np.abs(series - series[0]))
        if L == 0 or d == 0:
            result[idx] = 0.0
        else:
            result[idx] = np.log(n) / (np.log(d / L) + np.log(n))
    return result.reshape(data_moved.shape[1:])


__all__ = ["higuchi_fd", "katz_fd"]
