"""Tests for Dataset convenience helpers."""

import sys
import types
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi_activity_recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.datasets import Dataset  # type: ignore  # noqa: E402


def test_dataset_from_files(tmp_path: Path):
    data = np.random.rand(20, 1, 8, 8).astype(np.float32)
    labels = np.random.randint(0, 2, 20)
    data_path = tmp_path / "data.npy"
    labels_path = tmp_path / "labels.npy"
    np.save(data_path, data)
    np.save(labels_path, labels)

    ds = Dataset.from_files(data_path, labels_path, val_ratio=0.2, test_ratio=0.2)

    assert len(ds) == 12  # 20 * (1 - 0.2 - 0.2)
    assert ds.input_shape == (1, 8, 8)
    assert set(ds.classes) == {0, 1}

