"""Dataset utilities and public dataset loaders."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from .loaders import load_dataset, split_dataset
from .public_datasets import load_signfi, load_widar3
from .synthetic import generate_synthetic_csi
from .transforms import add_noise, time_shift


@dataclass
class Dataset:
    """Container for training, validation, and test splits.

    The class computes metadata such as available classes and input shape on
    initialization to streamline integration with the training pipeline and CLI
    utilities.
    """

    train: Tuple[np.ndarray, np.ndarray]
    val: Tuple[np.ndarray, np.ndarray]
    test: Tuple[np.ndarray, np.ndarray]
    classes: List[int] = field(init=False)
    input_shape: Tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        """Compute class labels and input shape."""
        train_labels = self.train[1]
        self.classes = sorted(set(int(l) for l in train_labels.tolist()))
        self.input_shape = tuple(self.train[0][0].shape)

    def __len__(self) -> int:  # pragma: no cover - trivial
        """Return the number of training samples."""
        return len(self.train[0])

    @classmethod
    def from_files(
        cls,
        data_path: Path | str,
        labels_path: Path | str,
        hardware_type: str | None = None,
        **split_kwargs: float,
    ) -> "Dataset":
        """Load arrays from disk and create a :class:`Dataset` instance."""
        data = np.load(data_path)
        labels = np.load(labels_path)
        train, val, test = split_dataset(data, labels, **split_kwargs)
        return cls(train=train, val=val, test=test)


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
