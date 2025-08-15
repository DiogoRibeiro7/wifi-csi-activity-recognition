"""Helpers for loading public CSI datasets."""

from pathlib import Path
from typing import Tuple

import numpy as np

from .loaders import load_dataset


def load_widar3(
    root: Path | str,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    **kwargs,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Load the Widar3.0 dataset from ``root`` using ``load_dataset``."""
    directory = Path(root) / "widar3"
    return load_dataset(directory, val_ratio=val_ratio, test_ratio=test_ratio, **kwargs)


def load_signfi(
    root: Path | str,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    **kwargs,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Load the SignFi dataset from ``root`` using ``load_dataset``."""
    directory = Path(root) / "signfi"
    return load_dataset(directory, val_ratio=val_ratio, test_ratio=test_ratio, **kwargs)


__all__ = ["load_widar3", "load_signfi"]
