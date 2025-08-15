import sys
import types
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi-activity-recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.utils.visualization import (  # type: ignore  # noqa: E402
    plot_activity_timeline,
    plot_csi_heatmap,
)


def test_plot_csi_heatmap_returns_axes():
    data = np.random.rand(10, 5)
    ax = plot_csi_heatmap(data)
    assert ax.images


def test_plot_activity_timeline_returns_axes():
    times = [0, 1, 2, 3]
    labels = ["a", "b", "b", "a"]
    ax = plot_activity_timeline(times, labels)
    assert ax.lines
