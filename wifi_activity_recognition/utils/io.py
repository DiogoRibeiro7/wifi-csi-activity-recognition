"""CSI data input/output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Union

import h5py
import numpy as np

from ..hardware.base import CSIData


def save_csi_to_hdf5(
    data: Union[np.ndarray, Sequence[CSIData]],
    path: str | Path,
    dataset_name: str = "csi",
) -> None:
    """Save CSI data or ``CSIData`` objects to an HDF5 file."""
    with h5py.File(path, "w") as f:
        if isinstance(data, np.ndarray):
            f.create_dataset(dataset_name, data=data)
            return

        amps = np.stack([d.amplitude for d in data])
        phases = np.stack([d.phase for d in data])
        f.create_dataset("amplitude", data=amps)
        f.create_dataset("phase", data=phases)
        f.create_dataset("timestamp", data=[d.timestamp for d in data])
        f.create_dataset("frequency", data=[d.frequency for d in data])
        f.create_dataset("bandwidth", data=[d.bandwidth for d in data])


def load_csi_from_hdf5(
    path: str | Path, dataset_name: str = "csi"
) -> Union[np.ndarray, list[CSIData]]:
    """Load CSI data or ``CSIData`` objects from an HDF5 file."""
    with h5py.File(path, "r") as f:
        if dataset_name in f:
            return np.array(f[dataset_name])

        amps = np.array(f["amplitude"])
        phases = np.array(f["phase"])
        timestamps = np.array(f["timestamp"])
        freqs = np.array(f["frequency"])
        bws = np.array(f["bandwidth"])

        result: list[CSIData] = []
        for amp, ph, ts, fr, bw in zip(amps, phases, timestamps, freqs, bws):
            n_rx, n_tx, n_sc = amp.shape
            result.append(
                CSIData(
                    timestamp=float(ts),
                    amplitude=amp,
                    phase=ph,
                    frequency=float(fr),
                    bandwidth=float(bw),
                    n_tx=int(n_tx),
                    n_rx=int(n_rx),
                    n_subcarriers=int(n_sc),
                )
            )
        return result


def save_csi_to_json(
    data: Union[np.ndarray, Sequence[CSIData]], path: str | Path
) -> None:
    """Save CSI data or ``CSIData`` objects to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(data, np.ndarray):
            json.dump(data.tolist(), f)
        else:
            json.dump([d.to_dict() for d in data], f)


def load_csi_from_json(path: str | Path) -> Union[np.ndarray, list[CSIData]]:
    """Load CSI data or ``CSIData`` objects from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if (
        isinstance(obj, list)
        and obj
        and isinstance(obj[0], dict)
        and "amplitude" in obj[0]
    ):
        result: list[CSIData] = []
        for item in obj:
            amp = np.asarray(item["amplitude"])
            ph = np.asarray(item["phase"])
            n_rx, n_tx, n_sc = amp.shape
            result.append(
                CSIData(
                    timestamp=float(item["timestamp"]),
                    amplitude=amp,
                    phase=ph,
                    frequency=float(item["frequency"]),
                    bandwidth=float(item["bandwidth"]),
                    n_tx=int(n_tx),
                    n_rx=int(n_rx),
                    n_subcarriers=int(n_sc),
                )
            )
        return result

    return np.array(obj)


def save_csi_data(data: Union[np.ndarray, Sequence[CSIData]], path: str | Path) -> None:
    """Save CSI data inferring format from file extension."""
    path = Path(path)
    if path.suffix in {".h5", ".hdf5"}:
        save_csi_to_hdf5(data, path)
    elif path.suffix == ".json":
        save_csi_to_json(data, path)
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported file extension: {path.suffix}")


def load_csi_data(path: str | Path) -> Union[np.ndarray, list[CSIData]]:
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


def save_evaluation_results(results: dict, path: str | Path) -> None:
    """Save evaluation metrics to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
