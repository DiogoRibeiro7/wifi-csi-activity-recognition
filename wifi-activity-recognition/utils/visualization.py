"""Visualization helpers for CSI data and activities."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_csi_heatmap(csi: np.ndarray, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot a heatmap of CSI amplitudes or phases."""
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(csi, aspect="auto", origin="lower")
    ax.set_xlabel("Subcarrier")
    ax.set_ylabel("Packet")
    return ax


def plot_activity_timeline(
    timestamps: Sequence[float], labels: Sequence[str | int], ax: plt.Axes | None = None
) -> plt.Axes:
    """Plot an activity timeline as a step chart."""
    if ax is None:
        _, ax = plt.subplots()
    unique = {label: idx for idx, label in enumerate(dict.fromkeys(labels))}
    numeric = [unique[label] for label in labels]
    ax.step(timestamps, numeric, where="post")
    ax.set_yticks(list(unique.values()))
    ax.set_yticklabels(list(unique.keys()))
    ax.set_xlabel("Time")
    ax.set_ylabel("Activity")
    return ax
