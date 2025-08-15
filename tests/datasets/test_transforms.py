"""Tests for dataset augmentation transforms."""

import sys
import types
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi-activity-recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.datasets import (  # type: ignore  # noqa: E402
    add_noise,
    time_shift,
)


def test_add_noise_deterministic():
    data = np.zeros((5, 3))
    noisy = add_noise(data, noise_std=0.1, random_state=42)
    noisy_again = add_noise(data, noise_std=0.1, random_state=42)
    assert np.allclose(noisy, noisy_again)
    assert not np.allclose(noisy, data)


def test_time_shift():
    data = np.arange(10).reshape(10, 1)
    shifted = time_shift(data, 2)
    assert shifted[0, 0] == data[-2, 0]
