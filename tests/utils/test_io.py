import sys
import types
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi-activity-recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.utils.io import (  # type: ignore  # noqa: E402
    load_csi_from_hdf5,
    load_csi_from_json,
    save_csi_to_hdf5,
    save_csi_to_json,
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
