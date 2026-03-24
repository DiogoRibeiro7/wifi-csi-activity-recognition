"""Tests for segmentation utilities."""

import sys
import types
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi_activity_recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.hardware.base import (  # type: ignore  # noqa: E402
    CSIData,
)
from wifi_activity_recognition.preprocessing import (  # type: ignore  # noqa: E402
    segment_windows,
)


def _make_csi(value: float) -> CSIData:
    amp = np.array([[[value]]])
    phase = np.zeros_like(amp)
    return CSIData(
        timestamp=value,
        amplitude=amp,
        phase=phase,
        frequency=5.0,
        bandwidth=20.0,
        n_tx=1,
        n_rx=1,
        n_subcarriers=1,
    )


def test_segment_windows() -> None:
    """Segment sequence into overlapping windows."""
    data = [_make_csi(float(i)) for i in range(10)]
    segments = segment_windows(data, window_size=4, overlap=0.5)
    assert len(segments) == 4
    assert all(len(seg) == 4 for seg in segments)
    assert segments[0][0].amplitude[0, 0, 0] == 0.0


def test_segment_windows_invalid() -> None:
    """Error when window size exceeds data length."""
    data = [_make_csi(float(i)) for i in range(3)]
    try:
        segment_windows(data, window_size=5)
    except ValueError:
        return
    raise AssertionError("expected ValueError")

