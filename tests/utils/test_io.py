"""Tests for I/O utilities."""

from pathlib import Path

import numpy as np


from wifi_activity_recognition.hardware.base import (  # type: ignore  # noqa: E402
    CSIData,
)
from wifi_activity_recognition.utils.io import (  # type: ignore  # noqa: E402
    load_csi_data,
    load_csi_from_hdf5,
    load_csi_from_json,
    save_csi_data,
    save_csi_to_hdf5,
    save_csi_to_json,
    save_predictions,
)


def test_hdf5_roundtrip(tmp_path):
    """Saving and loading arrays via HDF5 preserves data."""
    data = np.random.rand(3, 3)
    path = tmp_path / "csi.h5"
    save_csi_to_hdf5(data, path)
    loaded = load_csi_from_hdf5(path)
    assert np.allclose(loaded, data)


def test_json_roundtrip(tmp_path):
    """Saving and loading arrays via JSON preserves data."""
    data = np.random.rand(2, 2)
    path = tmp_path / "csi.json"
    save_csi_to_json(data, path)
    loaded = load_csi_from_json(path)
    assert np.allclose(loaded, data)


def test_generic_roundtrip(tmp_path):
    """Generic save/load infers format from extension."""
    data = np.random.rand(2, 2)
    h5_path = tmp_path / "csi.h5"
    json_path = tmp_path / "csi.json"
    save_csi_data(data, h5_path)
    save_csi_data(data, json_path)
    assert np.allclose(load_csi_data(h5_path), data)
    assert np.allclose(load_csi_data(json_path), data)


def test_save_predictions(tmp_path):
    """Prediction helper writes JSON records."""
    preds = ["walk", "sit"]
    confs = [0.9, 0.8]
    path = tmp_path / "preds.json"
    save_predictions(preds, confs, path)
    assert path.exists()


class _DummyCSI:
    def __init__(self) -> None:
        self.amplitude = np.ones((1, 1, 2))
        self.phase = np.zeros((1, 1, 2))

    def to_dict(self):
        return {
            "timestamp": 0.0,
            "amplitude": self.amplitude.tolist(),
            "phase": self.phase.tolist(),
            "frequency": 5.0,
            "bandwidth": 20.0,
            "n_tx": 1,
            "n_rx": 1,
            "n_subcarriers": 2,
        }


def test_save_csidata_list(tmp_path):
    """Saving list of CSIData objects to JSON succeeds."""
    path = tmp_path / "list.json"
    save_csi_data([_DummyCSI()], path)
    assert path.exists()


def _make_csi() -> CSIData:
    amp = np.ones((1, 1, 2))
    phase = np.zeros_like(amp)
    return CSIData(
        timestamp=0.0,
        amplitude=amp,
        phase=phase,
        frequency=5.0,
        bandwidth=20.0,
        n_tx=1,
        n_rx=1,
        n_subcarriers=2,
    )


def test_csidata_json_roundtrip(tmp_path):
    """Roundtrip JSON serialization for CSIData objects."""
    csi = _make_csi()
    path = tmp_path / "csi.json"
    save_csi_to_json([csi], path)
    loaded = load_csi_from_json(path)
    assert isinstance(loaded, list)
    assert np.allclose(loaded[0].amplitude, csi.amplitude)


def test_csidata_hdf5_roundtrip(tmp_path):
    """Roundtrip HDF5 serialization for CSIData objects."""
    csi = _make_csi()
    path = tmp_path / "csi.h5"
    save_csi_to_hdf5([csi], path)
    loaded = load_csi_from_hdf5(path)
    assert isinstance(loaded, list)
    assert np.allclose(loaded[0].phase, csi.phase)

