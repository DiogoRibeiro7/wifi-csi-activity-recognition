"""Tests for public dataset loaders."""

from pathlib import Path

import numpy as np


from wifi_activity_recognition.datasets import (  # type: ignore  # noqa: E402
    load_signfi,
    load_widar3,
)


def _create_dataset(path: Path, samples: int):
    data = np.random.randn(samples, 3)
    labels = np.arange(samples)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "data.npy", data)
    np.save(path / "labels.npy", labels)


def test_load_widar3(tmp_path: Path):
    dataset_dir = tmp_path / "widar3"
    _create_dataset(dataset_dir, 12)
    (train, _), (val, _), (test, _) = load_widar3(
        tmp_path, val_ratio=0.25, test_ratio=0.25
    )
    assert len(train) == 6
    assert len(val) == 3
    assert len(test) == 3


def test_load_signfi(tmp_path: Path):
    dataset_dir = tmp_path / "signfi"
    _create_dataset(dataset_dir, 15)
    (train, _), (val, _), (test, _) = load_signfi(
        tmp_path, val_ratio=0.2, test_ratio=0.2
    )
    assert len(train) == 9
    assert len(val) == 3
    assert len(test) == 3

