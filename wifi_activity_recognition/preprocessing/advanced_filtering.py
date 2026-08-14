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
    """Apply adaptive Wiener filtering to selected fields.

    Parameters
    ----------
    csi:
        Input CSI sample.
    mysize:
        Size of the Wiener filter window. Typical values range from 3 to 7.
        Must be a positive integer or tuple of integers.
    noise:
        Estimated noise power. ``None`` lets :func:`scipy.signal.wiener`
        estimate it automatically. Values must be non-negative.
    axis:
        Axis along which filtering is performed. ``0`` corresponds to receive
        antennas and ``-1`` to subcarriers.
    fields:
        Iterable of fields to filter. Supports ``"amplitude"`` and
        ``"phase"``.

    Returns
    -------
    :class:`CSIData`
        New instance with filtered fields.

    Raises
    ------
    ValueError
        If parameters are out of range or invalid.
    """
    if isinstance(mysize, int) and mysize <= 0:
        raise ValueError("mysize must be positive")
    if noise is not None and noise < 0:
        raise ValueError("noise must be non-negative")

    updates = {}
    for field in fields:
        data = getattr(csi, field)
        if axis >= data.ndim or axis < -data.ndim:
            raise ValueError("axis out of range")

        def _wiener(m: np.ndarray) -> np.ndarray:
            # ``signal.wiener`` fails when the variance is zero; in that case
            # simply return the input unchanged.
            if np.var(m) == 0:
                return m
            return signal.wiener(m, mysize, noise)

        filtered = np.apply_along_axis(_wiener, axis, data)
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
    csi:
        Input CSI sample.
    size:
        Window size for the morphological operation. Typical values are 3 or 5.
    operation:
        Either ``"opening"`` or ``"closing"``.
    axis:
        Axis along which filtering is performed.
    fields:
        Iterable of fields to filter.
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

    Raises
    ------
    ValueError
        If ``up`` or ``down`` is not positive, if ``fields`` is empty, or if
        resampling the subcarrier axis would leave ``amplitude`` and ``phase``
        with different lengths.
    """
    if up <= 0 or down <= 0:
        raise ValueError("up and down must be positive integers")

    # ``fields`` is typed as an Iterable, so materialise it before iterating:
    # a generator would otherwise be exhausted by the first pass, and the old
    # code indexed it directly, which fails for any non-sequence iterable.
    fields = tuple(fields)
    if not fields:
        raise ValueError("fields must name at least one field to resample")

    # Resampling the subcarrier axis changes n_subcarriers, which both
    # amplitude and phase are validated against. Leaving one behind produces a
    # CSIData that cannot be constructed.
    resamples_subcarriers = axis in (-1, csi.amplitude.ndim - 1)
    if resamples_subcarriers and set(fields) != {"amplitude", "phase"}:
        raise ValueError(
            "resampling the subcarrier axis requires fields to include both "
            f"'amplitude' and 'phase', got {fields}"
        )

    updates = {}
    for field in fields:
        data = getattr(csi, field)
        resampled = np.apply_along_axis(
            lambda m: signal.resample_poly(m, up, down), axis, data
        )
        updates[field] = resampled

    # Take the new length from the resampled array itself. resample_poly
    # returns ceil(n * up / down) samples, which floor division gets wrong
    # whenever n * up is not divisible by down.
    n_sc_new = updates[fields[0]].shape[axis] if resamples_subcarriers else None
    if n_sc_new is None:
        return replace(csi, **updates)
    return replace(csi, n_subcarriers=int(n_sc_new), **updates)


__all__ = [
    "adaptive_wiener_filter",
    "median_filter",
    "morphological_filter",
    "multirate_resample",
]
