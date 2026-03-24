"""Tests for information theoretic features."""

import sys
import types
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi_activity_recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.features import (  # type: ignore  # noqa: E402
    mutual_information,
    shannon_entropy,
)
from wifi_activity_recognition.hardware.base import (  # type: ignore  # noqa: E402
    CSIData,
)


def _make_csi(amp: np.ndarray) -> CSIData:
    phase = np.zeros_like(amp)
    n_rx, n_tx, n_sc = amp.shape
    return CSIData(0.0, amp.copy(), phase, 5.0, 20.0, n_tx, n_rx, n_sc)


def test_shannon_entropy_constant() -> None:
    """Constant signal has near-zero entropy."""
    amp = np.ones((1, 1, 16))
    csi = _make_csi(amp)
    ent = shannon_entropy(csi)
    assert ent.shape == (1, 1)
    assert ent[0, 0] < 1e-6


def test_mutual_information_identical() -> None:
    """Identical signals yield high mutual information."""
    amp = np.random.randn(1, 1, 32)
    csi1 = _make_csi(amp)
    csi2 = _make_csi(amp)
    mi = mutual_information(csi1, csi2)
    assert mi > 0.5


def test_mutual_information_independent() -> None:
    """Independent signals have lower MI than identical ones."""
    amp1 = np.random.randn(1, 1, 32)
    csi1 = _make_csi(amp1)
    mi_same = mutual_information(csi1, _make_csi(amp1))
    mi_ind = mutual_information(csi1, _make_csi(np.random.randn(1, 1, 32)))
    assert mi_ind < mi_same

