"""Advanced filtering utilities for :class:`CSIData`."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Tuple

import numpy as np
from scipy import ndimage, signal

from ..hardware.base import CSIData


def adaptive_wiener_filter(
    csi: CSIData,
    mysize: int | Tuple[int, ...] = 3,
    noise: float | None = None,
    axis: int = -1,
    fields: Iterable[str] = ("amplitude",),
) -> CSIData:
    """Apply Wiener filtering to selected fields.

    Parameters
    ----------
    csi:
        Input CSI sample.
    mysize:
        Size of the Wiener filter window. Typical values range from 3 to 7.
    noise:
        Estimated noise power. ``None`` lets :func:`scipy.signal.wiener` estimate it.
    axis:
        Axis along which filtering is performed.
    fields:
        Iterable of fields to filter. Defaults to ``("amplitude",)``.
    """
    updates = {}
    for field in fields:
        data = getattr(csi, field)
        filtered = np.apply_along_axis(
            lambda m: signal.wiener(m, mysize, noise), axis, data
        )
        updates[field] = filtered
    return replace(csi, **updates)


def median_filter(
    csi: CSIData,
    kernel_size: int = 3,
    axis: int = -1,
    fields: Iterable[str] = ("amplitude",),
) -> CSIData:
    """Remove impulse noise using median filtering.

    ``kernel_size`` must be an odd positive integer (e.g., 3, 5).
    """
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    updates = {}
    for field in fields:
        data = getattr(csi, field)
        filtered = np.apply_along_axis(
            lambda m: signal.medfilt(m, kernel_size), axis, data
        )
        updates[field] = filtered
    return replace(csi, **updates)


def morphological_filter(
    csi: CSIData,
    size: int = 3,
    operation: str = "opening",
    axis: int = -1,
    fields: Iterable[str] = ("amplitude",),
) -> CSIData:
    """Apply morphological filtering using grey-scale operations.

    Parameters
    ----------
    size:
        Window size for the morphological operation. Typical values are 3 or 5.
    operation:
        Either ``"opening"`` or ``"closing"``.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if operation not in {"opening", "closing"}:
        raise ValueError("operation must be 'opening' or 'closing'")
    updates = {}
    func = ndimage.grey_opening if operation == "opening" else ndimage.grey_closing
    footprint = np.ones(size)
    for field in fields:
        data = getattr(csi, field)
        filtered = np.apply_along_axis(
            lambda m: func(m, footprint=footprint), axis, data
        )
        updates[field] = filtered
    return replace(csi, **updates)


def multirate_resample(
    csi: CSIData,
    up: int = 1,
    down: int = 1,
    axis: int = -1,
    fields: Iterable[str] = ("amplitude", "phase"),
) -> CSIData:
    """Resample fields by the rational factor ``up / down``.

    Typical values for ``up`` and ``down`` range from 1 to 4.  Both must be
    positive integers.
    """
    if up <= 0 or down <= 0:
        raise ValueError("up and down must be positive integers")
    updates = {}
    for field in fields:
        data = getattr(csi, field)
        resampled = np.apply_along_axis(
            lambda m: signal.resample_poly(m, up, down), axis, data
        )
        updates[field] = resampled
    n_sc_new = getattr(csi, fields[0]).shape[axis] * up // down
    return replace(csi, n_subcarriers=n_sc_new, **updates)


__all__ = [
    "adaptive_wiener_filter",
    "median_filter",
    "morphological_filter",
    "multirate_resample",
]
