"""Tests for artifact removal utilities."""

from pathlib import Path

import numpy as np


from wifi_activity_recognition.hardware.base import (  # type: ignore  # noqa: E402
    CSIData,
)
from wifi_activity_recognition.preprocessing import (  # type: ignore  # noqa: E402
    mitigate_interference,
    remove_motion_artifacts,
)


def _make_csi(amplitude: np.ndarray, phase: np.ndarray | None = None) -> CSIData:
    phase = np.zeros_like(amplitude) if phase is None else phase
    n_rx, n_tx, n_sc = amplitude.shape
    return CSIData(
        timestamp=0.0,
        amplitude=amplitude,
        phase=phase,
        frequency=5.0,
        bandwidth=20.0,
        n_tx=n_tx,
        n_rx=n_rx,
        n_subcarriers=n_sc,
    )


def test_remove_motion_artifacts() -> None:
    """Artifacts are interpolated."""
    amp = np.ones((1, 1, 10))
    phase = np.zeros_like(amp)
    amp[..., 5] = 10.0
    phase[..., 6] = 5.0
    csi = _make_csi(amp, phase)
    cleaned = remove_motion_artifacts(csi, threshold=2.0, fields=("amplitude", "phase"))
    assert cleaned.amplitude[..., 5] != 10.0
    assert cleaned.phase[..., 6] != 5.0
    assert csi.amplitude[..., 5] == 10.0


def test_mitigate_interference() -> None:
    """Specified subcarriers are zeroed."""
    amp = np.ones((1, 1, 8))
    csi = _make_csi(amp)
    mitigated = mitigate_interference(csi, [1, 3])
    assert np.allclose(mitigated.amplitude[..., [1, 3]], 0)
    assert np.allclose(csi.amplitude[..., [1, 3]], 1)


def test_remove_motion_artifacts_constant_series() -> None:
    """Constant input remains unchanged."""
    amp = np.ones((1, 1, 8))
    csi = _make_csi(amp)
    cleaned = remove_motion_artifacts(csi, threshold=2.0)
    assert np.allclose(cleaned.amplitude, amp)

