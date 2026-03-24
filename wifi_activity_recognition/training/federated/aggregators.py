"""Federated aggregation strategies."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch

StateDict = Dict[str, torch.Tensor]
WeightedUpdate = Tuple[StateDict, int]


def fed_avg(updates: Iterable[WeightedUpdate]) -> StateDict:
    """Average client updates weighted by the number of samples.

    Parameters
    ----------
    updates:
        Iterable of ``(state_dict, num_samples)`` tuples produced by clients.

    Returns
    -------
    Dict[str, torch.Tensor]
        Averaged model parameters.
    """
    updates = list(updates)
    if not updates:
        raise ValueError("No updates provided to federated aggregator")
    total_samples = sum(n for _, n in updates)
    avg_state: StateDict = {}
    for key in updates[0][0].keys():
        avg_state[key] = sum(state[key] * n for state, n in updates) / total_samples
    return avg_state


def fed_prox(updates: Iterable[WeightedUpdate], mu: float = 0.0) -> StateDict:
    """Fedprox aggregation.

    The server-side aggregation is identical to FedAvg; the proximal term is
    handled during local training on the client. The ``mu`` parameter is
    included for API completeness.
    """
    _ = mu  # parameter kept for API compatibility
    return fed_avg(updates)


__all__ = ["fed_avg", "fed_prox"]
