"""Tests for synthetic CSI data generation."""

import sys
import types
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi-activity-recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.datasets import (  # type: ignore  # noqa: E402
    generate_synthetic_csi,
)


def test_generate_synthetic_csi_shapes():
    data, labels = generate_synthetic_csi(
        num_samples=8,
        num_subcarriers=4,
        num_antennas=2,
        num_classes=3,
        random_state=0,
    )
    assert data.shape == (8, 2, 4)
    assert labels.shape == (8,)
    assert set(labels) <= {0, 1, 2}
