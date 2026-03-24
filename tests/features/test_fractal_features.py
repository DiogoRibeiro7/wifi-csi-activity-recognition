"""Tests for fractal feature extraction."""

from pathlib import Path

import numpy as np


from wifi_activity_recognition.features import (  # type: ignore  # noqa: E402
    higuchi_fd,
    katz_fd,
)
from wifi_activity_recognition.hardware.base import (  # type: ignore  # noqa: E402
    CSIData,
)


def _make_csi(seq: np.ndarray) -> CSIData:
    phase = np.zeros_like(seq)
    n_rx, n_tx, n_sc = seq.shape
    return CSIData(0.0, seq.copy(), phase, 5.0, 20.0, n_tx, n_rx, n_sc)


def test_higuchi_constant() -> None:
    """Constant sequence has low Higuchi FD."""
    amp = np.ones((1, 1, 32))
    csi = _make_csi(amp)
    fd = higuchi_fd(csi)
    assert fd.shape == (1, 1)
    assert np.all(fd < 1.0)


def test_katz_variation() -> None:
    """Katz FD is positive for varying signals."""
    amp = np.stack([np.arange(32), np.arange(32) ** 2]).reshape(2, 1, 32)
    csi = _make_csi(amp)
    fd = katz_fd(csi)
    assert fd.shape == (2, 1)
    assert np.all(fd > 0)

