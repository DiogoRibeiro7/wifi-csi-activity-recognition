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
    """Load ``data.npy`` and ``labels.npy`` from ``directory`` and split them."""
    directory = Path(directory)
    data = np.load(directory / "data.npy")
    labels = np.load(directory / "labels.npy")
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
    """Split data and labels into train, validation, and test sets."""
    if len(data) != len(labels):  # pragma: no cover - safety check
        raise ValueError("Data and labels must have the same length")

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
