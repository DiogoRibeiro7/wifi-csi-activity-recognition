"""Tests for graph-based features."""

import numpy as np

from wifi_activity_recognition.features import (  # type: ignore  # noqa: E402
    build_correlation_graph,
    centrality_measures,
    community_structure,
    spectral_graph_features,
)
from wifi_activity_recognition.hardware.base import (  # type: ignore  # noqa: E402
    CSIData,
)


def _make_csi() -> CSIData:
    amp = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0]],
            [[1.0, 2.0, 3.0, 4.0]],
        ]
    )
    phase = np.zeros_like(amp)
    return CSIData(0.0, amp, phase, 5.0, 20.0, 1, 2, 4)


def test_correlation_graph_edges() -> None:
    """Highly correlated streams create an edge."""
    csi = _make_csi()
    G = build_correlation_graph(csi, threshold=0.9)
    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 1


def test_centrality_and_spectral() -> None:
    """Centrality dict contains expected keys and eigenvalues have shape."""
    csi = _make_csi()
    G = build_correlation_graph(csi, threshold=0.0)
    cent = centrality_measures(G)
    assert set(cent.keys()) == {"degree", "betweenness", "closeness"}
    eig = spectral_graph_features(G, k=2)
    assert eig.shape == (2,)


def test_community_structure() -> None:
    """All nodes belong to one community in this setup."""
    csi = _make_csi()
    G = build_correlation_graph(csi, threshold=0.0)
    comms = community_structure(G)
    assert len(comms) == 1
