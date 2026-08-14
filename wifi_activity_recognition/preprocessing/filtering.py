"""Noise filtering utilities for :class:`CSIData`.

A single :class:`CSIData` packet has axes ``(n_rx, n_tx, n_subcarriers)`` --
none of which is time. Filters parameterised in Hz are therefore only
meaningful across a *sequence* of packets, where time is the packet index and
``fs`` is the capture rate.

Every filter here accepts either form:

* a sequence of packets -- filtering runs along time and returns a new list
* a single packet -- filtering runs along ``axis`` within that packet, which
  is a frequency-domain operation across subcarriers, not a temporal one

``AXIS_RX``, ``AXIS_TX`` and ``AXIS_SUBCARRIER`` name the per-packet axes so
single-packet calls say which one they mean instead of relying on ``-1``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Sequence, Union

import numpy as np
from scipy.signal import butter, lfilter

from ..hardware.base import CSIData

# Axis labels for a single packet, plus the sequence axis.
AXIS_RX = 0
AXIS_TX = 1
AXIS_SUBCARRIER = 2

CSIInput = Union[CSIData, Sequence[CSIData]]


def _is_sequence(data: CSIInput) -> bool:
    """Return True when ``data`` is a sequence of packets rather than one."""
    return not isinstance(data, CSIData)


def _stack(packets: Sequence[CSIData], field: str) -> np.ndarray:
    """Stack one field across packets into a ``(time, rx, tx, subcarrier)`` array."""
    if len(packets) == 0:
        raise ValueError("cannot filter an empty sequence of packets")

    arrays = [getattr(packet, field) for packet in packets]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(
            f"all packets must share a shape to filter along time, got {shapes}"
        )
    return np.stack(arrays, axis=0)


def _unstack(
    packets: Sequence[CSIData], updates: dict[str, np.ndarray]
) -> list[CSIData]:
    """Rebuild per-packet objects from time-stacked arrays."""
    return [
        replace(packet, **{field: values[index] for field, values in updates.items()})
        for index, packet in enumerate(packets)
    ]


def _apply(
    data: CSIInput,
    fields: Iterable[str],
    axis: int,
    transform: Any,
) -> Union[CSIData, list[CSIData]]:
    """Run ``transform`` over ``fields``, along time or within one packet."""
    fields = tuple(fields)

    if _is_sequence(data):
        packets = list(data)
        # Time is axis 0 once stacked.
        updates = {field: transform(_stack(packets, field), 0) for field in fields}
        return _unstack(packets, updates)

    updates = {field: transform(getattr(data, field), axis) for field in fields}
    return replace(data, **updates)


def butterworth_filter(
    data: CSIInput,
    cutoff: float,
    fs: float,
    order: int = 4,
    axis: int = AXIS_SUBCARRIER,
    fields: Iterable[str] = ("amplitude",),
) -> Union[CSIData, list[CSIData]]:
    """Apply a low-pass Butterworth filter.

    Parameters
    ----------
    data:
        A sequence of packets -- filtered along time, which is what ``cutoff``
        and ``fs`` describe -- or a single packet, filtered along ``axis``.
    cutoff:
        Cutoff frequency in Hz. Must be below the Nyquist frequency.
    fs:
        Sampling frequency in Hz. For a packet sequence this is the capture
        rate, i.e. packets per second.
    order:
        Filter order.
    axis:
        Axis used only for the single-packet form. Defaults to the subcarrier
        axis. Ignored when ``data`` is a sequence, where time is always used.
    fields:
        Iterable of fields to filter. Defaults to ``("amplitude",)``.

    Raises
    ------
    ValueError
        If ``fs`` or ``cutoff`` are non-positive, or ``cutoff`` is at or above
        the Nyquist frequency, which would make the normalised cutoff invalid.
    """
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")

    nyq = 0.5 * fs
    if cutoff >= nyq:
        raise ValueError(
            f"cutoff {cutoff} Hz must be below the Nyquist frequency {nyq} Hz "
            f"for fs={fs} Hz"
        )

    b, a = butter(order, cutoff / nyq, btype="low", analog=False)
    return _apply(data, fields, axis, lambda arr, ax: lfilter(b, a, arr, axis=ax))


def moving_average_filter(
    data: CSIInput,
    window_size: int,
    axis: int = AXIS_SUBCARRIER,
    fields: Iterable[str] = ("amplitude",),
) -> Union[CSIData, list[CSIData]]:
    """Smooth fields using a moving average, along time or within a packet."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    kernel = np.ones(window_size) / window_size

    def smooth(arr: np.ndarray, ax: int) -> np.ndarray:
        return np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="same"), ax, arr
        )

    return _apply(data, fields, axis, smooth)


def kalman_filter(
    data: CSIInput,
    process_variance: float = 1e-2,
    measurement_variance: float = 1e-2,
    axis: int = AXIS_SUBCARRIER,
    fields: Iterable[str] = ("amplitude",),
) -> Union[CSIData, list[CSIData]]:
    """Filter with a scalar Kalman filter, along time or within a packet."""
    if process_variance < 0:
        raise ValueError(f"process_variance must be non-negative, got {process_variance}")
    if measurement_variance <= 0:
        raise ValueError(
            f"measurement_variance must be positive, got {measurement_variance}"
        )

    def run(arr: np.ndarray, ax: int) -> np.ndarray:
        moved = np.moveaxis(arr, ax, 0)
        n_steps, *rest = moved.shape
        filtered = np.zeros_like(moved)
        for idx in np.ndindex(*rest):
            x_est = moved[(0,) + idx]
            p_est = 1.0
            filtered[(0,) + idx] = x_est
            for step in range(1, n_steps):
                p_est += process_variance
                gain = p_est / (p_est + measurement_variance)
                x_est += gain * (moved[(step,) + idx] - x_est)
                p_est = (1 - gain) * p_est
                filtered[(step,) + idx] = x_est
        return np.moveaxis(filtered, 0, ax)

    return _apply(data, fields, axis, run)


__all__ = [
    "AXIS_RX",
    "AXIS_SUBCARRIER",
    "AXIS_TX",
    "butterworth_filter",
    "kalman_filter",
    "moving_average_filter",
]
