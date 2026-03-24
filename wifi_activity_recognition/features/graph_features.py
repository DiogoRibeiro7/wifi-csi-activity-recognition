"""Graph-based features derived from CSI antenna correlations.

The functions build graphs where nodes represent antenna pairs and edges
are weighted by correlation. Standard graph metrics and spectral features
can then be computed for downstream analysis.
"""

from __future__ import annotations

from typing import Dict, List, Set

import networkx as nx
import numpy as np

from ..hardware.base import CSIData


def build_correlation_graph(
    csi: CSIData,
    field: str = "amplitude",
    threshold: float = 0.0,
) -> nx.Graph:
    """Create a correlation graph between antenna streams.

    Nodes correspond to ``n_rx * n_tx`` antenna pairs. An undirected edge is
    added when Pearson correlation between two streams exceeds ``threshold``.
    The edge weight equals the correlation value.
    """
    data = getattr(csi, field)
    n_nodes = data.shape[0] * data.shape[1]
    flat = data.reshape(n_nodes, data.shape[2])
    corr = np.corrcoef(flat)
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            w = corr[i, j]
            if w > threshold:
                G.add_edge(i, j, weight=float(w))
    return G


def centrality_measures(graph: nx.Graph) -> Dict[str, Dict[int, float]]:
    """Compute basic centrality measures for a graph."""
    return {
        "degree": nx.degree_centrality(graph),
        "betweenness": nx.betweenness_centrality(graph, weight="weight"),
        "closeness": nx.closeness_centrality(graph),
    }


def community_structure(graph: nx.Graph) -> List[Set[int]]:
    """Detect communities using greedy modularity maximisation."""
    from networkx.algorithms.community import greedy_modularity_communities

    return list(greedy_modularity_communities(graph))


def spectral_graph_features(graph: nx.Graph, k: int = 3) -> np.ndarray:
    """Return the smallest ``k`` eigenvalues of the normalised Laplacian."""
    if graph.number_of_nodes() == 0:
        return np.zeros(k)
    lap = nx.normalized_laplacian_matrix(graph).toarray()
    eigvals = np.linalg.eigvalsh(lap)
    k = min(k, len(eigvals))
    return eigvals[:k]


__all__ = [
    "build_correlation_graph",
    "centrality_measures",
    "community_structure",
    "spectral_graph_features",
]
