import sys
import types
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi_activity_recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.preprocessing import (  # type: ignore  # noqa: E402
    detect_outliers_zscore,
    remove_outliers_zscore,
)


def test_detect_outliers_zscore():
    data = np.array([1, 1, 1, 100])
    mask = detect_outliers_zscore(data, threshold=3)
    assert mask[-1] and not mask[0]


def test_remove_outliers_zscore():
    data = np.array([1.0, 1.0, 1.0, 100.0])
    cleaned = remove_outliers_zscore(data, threshold=3)
    assert np.isnan(cleaned[-1])
    assert cleaned[0] == 1.0

