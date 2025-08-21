"""Federated learning simulation utilities."""

from __future__ import annotations

from typing import Callable, Dict, List

from torch import nn

from .server import FederatedServer


def run_simulation(
    server: FederatedServer,
    rounds: int,
    epochs: int,
    eval_fn: Callable[[nn.Module, Dict[str, object]], Dict[str, float]] | None = None,
) -> List[Dict[str, float]]:
    """Run a federated learning simulation.

    Parameters
    ----------
    server:
        Initialized :class:`FederatedServer` coordinating the clients.
    rounds:
        Number of federated rounds to execute.
    epochs:
        Number of local epochs per client in each round.
    eval_fn:
        Optional callable returning metrics given the global model and a mapping
        of environment names to datasets.
    """
    history: List[Dict[str, float]] = []
    env_map = {c.environment: c.dataset for c in server.clients}
    for _ in range(rounds):
        server.train_round(epochs)
        if eval_fn is not None:
            history.append(eval_fn(server.model, env_map))
    return history


__all__ = ["run_simulation"]
