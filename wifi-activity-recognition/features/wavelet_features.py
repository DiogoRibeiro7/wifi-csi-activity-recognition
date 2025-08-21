"""Wavelet-based features for CSI sequences.

This module implements continuous and discrete wavelet transform features
for :class:`~wifi_activity_recognition.hardware.base.CSIData` objects.
The functions operate on the last dimension of the CSI arrays (typically
subcarriers or time samples) and return NumPy arrays with wavelet
coefficients or energy measures while leaving the original ``CSIData``
immutable.

Examples
--------
>>> from wifi_activity_recognition.hardware.base import CSIData
>>> import numpy as np
>>> amp = np.random.randn(1, 1, 32)
>>> phase = np.zeros_like(amp)
>>> csi = CSIData(0.0, amp, phase, 5.0, 20.0, 1, 1, 32)
>>> coeffs = cwt_coefficients(csi, scales=[1, 2, 3])
>>> coeffs.shape
(1, 1, 3, 32)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pywt

from ..hardware.base import CSIData


def cwt_coefficients(
    csi: CSIData,
    scales: Sequence[float],
    wavelet: str = "morl",
    field: str = "amplitude",
) -> np.ndarray:
    """Compute continuous wavelet transform coefficients.

    Parameters
    ----------
    csi:
        Input CSI data.
    scales:
        Sequence of scales for the CWT.
    wavelet:
        Name of the mother wavelet (``"morl"`` or ``"ricker"``).
    field:
        Attribute of :class:`CSIData` to analyse (``"amplitude"`` or ``"phase"``).

    Returns
    -------
    np.ndarray
        Array of CWT coefficients with shape
        ``(n_rx, n_tx, len(scales), n_samples)``.
    """
    data = getattr(csi, field)
    flat = data.reshape(-1, data.shape[-1])
    coeffs = np.array([pywt.cwt(row, scales, wavelet)[0] for row in flat])
    return coeffs.reshape(data.shape[0], data.shape[1], len(scales), data.shape[2])


def dwt_energy(
    csi: CSIData,
    wavelet: str = "db1",
    level: int = 3,
    field: str = "amplitude",
) -> np.ndarray:
    """Compute energy of discrete wavelet transform coefficients.

    Returns an array of shape ``(n_rx, n_tx, level + 1)`` containing the
    energy of approximation and detail coefficients at each level.
    """
    data = getattr(csi, field)
    flat = data.reshape(-1, data.shape[-1])
    energies = []
    for row in flat:
        coeffs = pywt.wavedec(row, wavelet, level=level)
        energies.append([float(np.sum(c**2)) for c in coeffs])
    return np.array(energies).reshape(data.shape[0], data.shape[1], -1)


def wavelet_packet_energy(
    csi: CSIData,
    wavelet: str = "db1",
    maxlevel: int = 3,
    field: str = "amplitude",
) -> np.ndarray:
    """Energy of nodes from wavelet packet decomposition.

    The output shape is ``(n_rx, n_tx, 2 ** maxlevel)`` corresponding to the
    energy of each node at the deepest level of the packet tree.
    """
    data = getattr(csi, field)
    flat = data.reshape(-1, data.shape[-1])
    features = []
    for row in flat:
        wp = pywt.WaveletPacket(row, wavelet, maxlevel=maxlevel)
        nodes = wp.get_level(maxlevel, order="freq")
        features.append([float(np.sum(n.data**2)) for n in nodes])
    return np.array(features).reshape(data.shape[0], data.shape[1], -1)


def scale_energy(
    csi: CSIData,
    scales: Sequence[float],
    wavelet: str = "morl",
    field: str = "amplitude",
) -> np.ndarray:
    """Compute CWT energy at each scale.

    This is derived by summing the squared magnitude of CWT coefficients
    across the sample axis. The resulting array has shape
    ``(n_rx, n_tx, len(scales))``.
    """
    coeffs = cwt_coefficients(csi, scales, wavelet=wavelet, field=field)
    return np.sum(np.abs(coeffs) ** 2, axis=-1)


__all__ = [
    "cwt_coefficients",
    "dwt_energy",
    "wavelet_packet_energy",
    "scale_energy",
]
