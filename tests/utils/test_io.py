import sys
import types
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi-activity-recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

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
    data = np.random.rand(3, 3)
    path = tmp_path / "csi.h5"
    save_csi_to_hdf5(data, path)
    loaded = load_csi_from_hdf5(path)
    assert np.allclose(loaded, data)


def test_json_roundtrip(tmp_path):
    data = np.random.rand(2, 2)
    path = tmp_path / "csi.json"
    save_csi_to_json(data, path)
    loaded = load_csi_from_json(path)
    assert np.allclose(loaded, data)


def test_generic_roundtrip(tmp_path):
    data = np.random.rand(2, 2)
    h5_path = tmp_path / "csi.h5"
    json_path = tmp_path / "csi.json"
    save_csi_data(data, h5_path)
    save_csi_data(data, json_path)
    assert np.allclose(load_csi_data(h5_path), data)
    assert np.allclose(load_csi_data(json_path), data)


def test_save_predictions(tmp_path):
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
    path = tmp_path / "list.json"
    save_csi_data([_DummyCSI()], path)
    assert path.exists()
