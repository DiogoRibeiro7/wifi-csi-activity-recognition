"""CSI data input/output helpers."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


def save_csi_to_hdf5(
    data: np.ndarray, path: str | Path, dataset_name: str = "csi"
) -> None:
    """Save CSI data to an HDF5 file."""
    with h5py.File(path, "w") as f:
        f.create_dataset(dataset_name, data=data)


def load_csi_from_hdf5(path: str | Path, dataset_name: str = "csi") -> np.ndarray:
    """Load CSI data from an HDF5 file."""
    with h5py.File(path, "r") as f:
        return np.array(f[dataset_name])


def save_csi_to_json(data: np.ndarray, path: str | Path) -> None:
    """Save CSI data to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data.tolist(), f)


def load_csi_from_json(path: str | Path) -> np.ndarray:
    """Load CSI data from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return np.array(json.load(f))
