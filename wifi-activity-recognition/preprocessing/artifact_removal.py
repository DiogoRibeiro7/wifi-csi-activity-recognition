"""Motion artifact detection and environmental interference mitigation."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np

from ..hardware.base import CSIData


def detect_motion_artifacts(
    csi: CSIData,
    threshold: float = 3.0,
    axis: int = -1,
    field: str = "amplitude",
) -> np.ndarray:
    """Return a boolean mask where motion artifacts are detected.

    Artifacts are identified where the absolute derivative exceeds ``threshold``
    times the standard deviation along ``axis``.

    Parameters
    ----------
    csi:
        Input CSI sample.
    threshold:
        Sensitivity multiplier for detection. Typical range is 2--5.
    axis:
        Axis along which to compute the derivative.
    field:
        CSI field to analyse, e.g. ``"amplitude"`` or ``"phase"``.

    Returns
    -------
    np.ndarray
        Boolean mask indicating detected artifacts.

    Raises
    ------
    ValueError
        If ``threshold`` is not positive or ``axis`` is out of range.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    data = getattr(csi, field)
    if axis >= data.ndim or axis < -data.ndim:
        raise ValueError("axis out of range")
    diff = np.diff(data, axis=axis, prepend=np.take(data, [0], axis=axis))
    std = np.std(data, axis=axis, keepdims=True)
    std_safe = np.where(std == 0, np.inf, std)
    return np.abs(diff) > threshold * std_safe


def remove_motion_artifacts(
    csi: CSIData,
    threshold: float = 3.0,
    axis: int = -1,
    fields: Iterable[str] = ("amplitude",),
) -> CSIData:
    """Detect and interpolate motion artifacts.

    For each specified field, artifacts detected by
    :func:`detect_motion_artifacts` are replaced with linear interpolation along
    ``axis``.

    Parameters
    ----------
    csi:
        Input CSI sample.
    threshold:
        Sensitivity multiplier used during detection.
    axis:
        Axis along which interpolation is performed.
    fields:
        Iterable of fields to process. Defaults to ``("amplitude",)``.

    Returns
    -------
    :class:`CSIData`
        CSI sample with artifacts interpolated.
    """
    updates = {}
    for field in fields:
        mask = detect_motion_artifacts(csi, threshold=threshold, axis=axis, field=field)
        data = getattr(csi, field)
        data_moved = np.moveaxis(data, axis, 0)
        mask_moved = np.moveaxis(mask, axis, 0)
        repaired = data_moved.copy()
        for idx in np.ndindex(*mask_moved.shape[1:]):
            series = data_moved[(slice(None),) + idx]
            mask_series = mask_moved[(slice(None),) + idx]
            if mask_series.any():
                good = np.where(~mask_series)[0]
                if len(good) < 2:
                    continue
                repaired[(slice(None),) + idx] = np.interp(
                    np.arange(len(series)), good, series[good]
                )
        repaired = np.moveaxis(repaired, 0, axis)
        updates[field] = repaired
    return replace(csi, **updates)


def mitigate_interference(
    csi: CSIData,
    subcarriers: Iterable[int],
    fields: Iterable[str] = ("amplitude", "phase"),
) -> CSIData:
    """Zero out specified subcarriers to mitigate narrowband interference."""
    n_sc = csi.n_subcarriers
    mask = np.ones(n_sc, dtype=bool)
    for idx in subcarriers:
        if idx < 0 or idx >= n_sc:
            raise ValueError("subcarrier index out of range")
        mask[idx] = False
    updates = {}
    for field in fields:
        data = getattr(csi, field).copy()
        data[..., ~mask] = 0
        updates[field] = data
    return replace(csi, **updates)


__all__ = [
    "detect_motion_artifacts",
    "remove_motion_artifacts",
    "mitigate_interference",
]
