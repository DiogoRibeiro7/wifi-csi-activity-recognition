"""Tests for multipath analysis utilities."""

import numpy as np

from wifi_activity_recognition.hardware.base import (  # type: ignore  # noqa: E402
    CSIData,
)
from wifi_activity_recognition.preprocessing import (  # type: ignore  # noqa: E402
    separate_multipath_components,
)


def _make_csi(amplitude: np.ndarray, phase: np.ndarray) -> CSIData:
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


def test_separate_multipath_components() -> None:
    """Components sum to the original signal."""
    rng = np.random.default_rng(0)
    amp = rng.standard_normal((1, 1, 16))
    phase = rng.standard_normal((1, 1, 16))
    csi = _make_csi(amp, phase)
    comps = separate_multipath_components(csi, n_components=1)
    assert len(comps) == 1
    recon = sum(comp.complex_csi for comp in comps)
    assert np.allclose(recon, csi.complex_csi, atol=1e-6)


def test_separate_multipath_components_validation() -> None:
    """Invalid component counts raise an error."""
    amp = np.ones((1, 1, 8))
    phase = np.zeros_like(amp)
    csi = _make_csi(amp, phase)
    try:
        separate_multipath_components(csi, n_components=9)
        assert False, "Expected ValueError"
    except ValueError:
        pass
