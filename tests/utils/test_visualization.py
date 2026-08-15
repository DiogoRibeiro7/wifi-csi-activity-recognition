"""Tests for visualization utilities."""

import matplotlib
import numpy as np

matplotlib.use("Agg")


from wifi_activity_recognition.utils import (  # type: ignore  # noqa: E402
    visualization as viz,
)


def test_plot_csi_heatmap_returns_axes():
    """Heatmap helper returns axes with colorbar."""
    data = np.random.rand(10, 5)
    ax = viz.plot_csi_heatmap(data, colorbar=True)
    assert ax.images
    assert len(ax.figure.axes) == 2


def test_plot_activity_timeline_returns_axes():
    """Timeline helper returns axes with line artists."""
    times = [0, 1, 2, 3]
    labels = ["a", "b", "b", "a"]
    ax = viz.plot_activity_timeline(times, labels)
    assert ax.lines
