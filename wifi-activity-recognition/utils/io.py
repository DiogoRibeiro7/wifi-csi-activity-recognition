"""CSI data input/output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence, Union

import h5py
import numpy as np


def save_csi_to_hdf5(
    data: Union[np.ndarray, Sequence[Any]],
    path: str | Path,
    dataset_name: str = "csi",
) -> None:
    """Save CSI data to an HDF5 file."""
    with h5py.File(path, "w") as f:
        if isinstance(data, np.ndarray):
            f.create_dataset(dataset_name, data=data)
        else:
            amps = np.stack([d.amplitude for d in data])
            phases = np.stack([d.phase for d in data])
            f.create_dataset("amplitude", data=amps)
            f.create_dataset("phase", data=phases)


def load_csi_from_hdf5(path: str | Path, dataset_name: str = "csi") -> np.ndarray:
    """Load CSI data from an HDF5 file."""
    with h5py.File(path, "r") as f:
        return np.array(f[dataset_name])


def save_csi_to_json(data: Union[np.ndarray, Sequence[Any]], path: str | Path) -> None:
    """Save CSI data to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(data, np.ndarray):
            json.dump(data.tolist(), f)
        else:
            json.dump([d.to_dict() for d in data], f)


def load_csi_from_json(path: str | Path) -> np.ndarray:
    """Load CSI data from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return np.array(json.load(f))


def save_csi_data(data: Union[np.ndarray, Sequence[Any]], path: str | Path) -> None:
    """Save CSI data inferring format from file extension."""
    path = Path(path)
    if path.suffix in {".h5", ".hdf5"}:
        save_csi_to_hdf5(data, path)
    elif path.suffix == ".json":
        save_csi_to_json(data, path)
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported file extension: {path.suffix}")


def load_csi_data(path: str | Path) -> np.ndarray:
    """Load CSI data inferring format from file extension."""
    path = Path(path)
    if path.suffix in {".h5", ".hdf5"}:
        return load_csi_from_hdf5(path)
    if path.suffix == ".json":
        return load_csi_from_json(path)
    raise ValueError(f"Unsupported file extension: {path.suffix}")


def save_predictions(
    predictions: Sequence[str | int],
    confidences: Sequence[float],
    path: str | Path,
) -> None:
    """Save prediction results to JSON."""
    records = [
        {"prediction": p, "confidence": c} for p, c in zip(predictions, confidences)
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f)
