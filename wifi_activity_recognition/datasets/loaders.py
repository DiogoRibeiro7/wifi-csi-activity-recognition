"""Dataset loading and splitting utilities."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def load_dataset(
    directory: Path | str,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Load ``data.npy`` and ``labels.npy`` from ``directory`` and split them.

    The function expects two ``.npy`` files named ``data.npy`` and
    ``labels.npy`` inside ``directory``.  ``val_ratio`` and ``test_ratio``
    control how many samples are allocated to the respective splits.  ``shuffle``
    controls whether data are shuffled before splitting and ``random_state``
    makes the operation deterministic.
    """
    directory = Path(directory)
    data_file = directory / "data.npy"
    labels_file = directory / "labels.npy"
    if not data_file.exists() or not labels_file.exists():  # pragma: no cover -
        raise FileNotFoundError("data.npy and labels.npy must exist in directory")
    data = np.load(data_file)
    labels = np.load(labels_file)
    return split_dataset(data, labels, val_ratio, test_ratio, shuffle, random_state)


def split_dataset(
    data: np.ndarray,
    labels: np.ndarray,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Split data and labels into train, validation, and test sets.

    ``val_ratio`` and ``test_ratio`` must sum to less than ``1.0``; the
    remaining samples are assigned to the training split.  If ``shuffle`` is
    ``True`` the data are shuffled deterministically using ``random_state``.
    """
    if len(data) != len(labels):  # pragma: no cover - safety check
        raise ValueError("Data and labels must have the same length")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be < 1")

    num_samples = len(data)
    indices = np.arange(num_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    test_size = int(num_samples * test_ratio)
    val_size = int(num_samples * val_ratio)

    test_indices = indices[:test_size]
    val_indices = indices[test_size : test_size + val_size]
    train_indices = indices[test_size + val_size :]

    train = (data[train_indices], labels[train_indices])
    val = (data[val_indices], labels[val_indices])
    test = (data[test_indices], labels[test_indices])
    return train, val, test


__all__ = ["load_dataset", "split_dataset"]
