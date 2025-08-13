"""Dataset utilities and public dataset loaders."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np

from .loaders import load_dataset, split_dataset
from .public_datasets import load_signfi, load_widar3
from .synthetic import generate_synthetic_csi
from .transforms import add_noise, time_shift


@dataclass
class Dataset:
    """Container for training, validation, and test splits."""

    train: Tuple[np.ndarray, np.ndarray]
    val: Tuple[np.ndarray, np.ndarray]
    test: Tuple[np.ndarray, np.ndarray]


_PUBLIC_DATASETS: Dict[
    str,
    Callable[
        [Path | str],
        Tuple[
            Tuple[np.ndarray, np.ndarray],
            Tuple[np.ndarray, np.ndarray],
            Tuple[np.ndarray, np.ndarray],
        ],
    ],
] = {
    "widar3": load_widar3,
    "signfi": load_signfi,
}


def load_public_dataset(name: str, root: Path | str, **split_kwargs) -> Dataset:
    """Load a public dataset by name.

    Args:
        name: Name of the dataset (``"widar3"`` or ``"signfi"``).
        root: Directory containing the dataset files.
        **split_kwargs: Arguments forwarded to :func:`load_dataset`.

    Returns:
        Dataset: Loaded dataset with train/val/test splits.

    Raises:
        ValueError: If the dataset name is unknown.
    """
    try:
        loader = _PUBLIC_DATASETS[name.lower()]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Unknown public dataset '{name}'") from exc
    train, val, test = loader(root, **split_kwargs)
    return Dataset(train=train, val=val, test=test)


__all__ = [
    "Dataset",
    "add_noise",
    "time_shift",
    "generate_synthetic_csi",
    "load_dataset",
    "split_dataset",
    "load_public_dataset",
    "load_widar3",
    "load_signfi",
]
