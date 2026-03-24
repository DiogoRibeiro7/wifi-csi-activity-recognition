"""Tests for dataset loading utilities."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi_activity_recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.datasets import (  # type: ignore  # noqa: E402
    load_dataset,
    split_dataset,
)


def test_split_dataset_shapes():
    """Ensure split_dataset returns correct subset sizes."""
    data = np.arange(100).reshape(50, 2)
    labels = np.arange(50)
    (train_data, _), (val_data, _), (test_data, _) = split_dataset(
        data, labels, val_ratio=0.2, test_ratio=0.1, shuffle=False
    )
    assert len(train_data) == 35
    assert len(val_data) == 10
    assert len(test_data) == 5


def test_load_dataset(tmp_path: Path):
    """Verify load_dataset reads files and splits data."""
    data = np.random.randn(20, 3)
    labels = np.arange(20)
    np.save(tmp_path / "data.npy", data)
    np.save(tmp_path / "labels.npy", labels)
    (train, _), (val, _), (test, _) = load_dataset(
        tmp_path, val_ratio=0.25, test_ratio=0.25
    )
    assert len(train) == 10
    assert len(val) == 5
    assert len(test) == 5


def test_invalid_split_ratios():
    """split_dataset raises when ratios exceed total."""
    data = np.zeros((10, 1))
    labels = np.zeros(10)
    with pytest.raises(ValueError):
        split_dataset(data, labels, val_ratio=0.6, test_ratio=0.5)

