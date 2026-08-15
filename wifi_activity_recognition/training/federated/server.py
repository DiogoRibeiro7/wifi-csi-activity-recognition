"""Federated server coordinating client training."""

from __future__ import annotations

import copy
import random
import typing as t
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Sequence

import torch
from torch import nn

from .aggregators import fed_avg
from .privacy import secure_aggregate

if t.TYPE_CHECKING:  # pragma: no cover
    from .client import FederatedClient

StateDict = Dict[str, torch.Tensor]

ClientSelector = Callable[
    [Sequence["FederatedClient"], float, random.Random], List["FederatedClient"]
]


def random_selector(
    clients: Sequence["FederatedClient"], fraction: float, rng: random.Random
) -> List["FederatedClient"]:
    """Select a random subset of clients."""
    num_clients = max(1, int(len(clients) * fraction))
    return rng.sample(list(clients), num_clients)


def hardware_balanced_selector(
    clients: Sequence["FederatedClient"], fraction: float, rng: random.Random
) -> List["FederatedClient"]:
    """Select clients ensuring balanced hardware representation."""
    num_clients = max(1, int(len(clients) * fraction))
    groups: Dict[str, List["FederatedClient"]] = defaultdict(list)
    for c in clients:
        groups[c.hardware].append(c)
    selected: List["FederatedClient"] = []
    # Round-robin pick from each hardware group
    while len(selected) < num_clients and any(groups.values()):
        for group in list(groups.values()):
            if group:
                selected.append(group.pop(0))
                if len(selected) == num_clients:
                    break
    # If still short, fill randomly from remaining
    remaining = [c for g in groups.values() for c in g]
    if len(selected) < num_clients and remaining:
        selected.extend(rng.sample(remaining, num_clients - len(selected)))
    return selected


@dataclass
class FederatedServer:
    """Server orchestrating federated learning rounds."""

    model: nn.Module
    clients: Sequence["FederatedClient"]
    aggregator: Callable[[Iterable[tuple[StateDict, int]]], StateDict] = fed_avg
    fraction: float = 1.0
    rng: random.Random = field(default_factory=random.Random)
    selector: ClientSelector = random_selector
    secure_aggregation: bool = False
    balance_environments: bool = False

    def _select_clients(self) -> List["FederatedClient"]:
        """Select a subset of clients according to ``fraction`` and ``selector``."""
        return self.selector(self.clients, self.fraction, self.rng)

    def _aggregate(self, updates: List[tuple[StateDict, int]]) -> StateDict:
        """Aggregate updates with optional secure aggregation."""
        if self.secure_aggregation:
            states = {str(i): s for i, (s, _) in enumerate(updates)}
            total = sum(n for _, n in updates)
            summed = secure_aggregate(states)
            return {k: v / total for k, v in summed.items()}
        return self.aggregator(updates)

    def train_round(self, epochs: int) -> None:
        """Run a single federated learning round."""
        selected = self._select_clients()
        global_state = self.model.state_dict()
        updates: List[tuple[StateDict, int]] = []
        for client in selected:
            client.update_model(copy.deepcopy(global_state))
            state, n_samples = client.train(epochs)
            updates.append((state, n_samples))
        if self.balance_environments:
            env_updates: Dict[str, List[tuple[StateDict, int]]] = defaultdict(list)
            for client, upd in zip(selected, updates):
                env_updates[client.environment].append(upd)
            env_states = [self._aggregate(u) for u in env_updates.values()]
            new_state = {
                k: sum(s[k] for s in env_states) / len(env_states)
                for k in env_states[0].keys()
            }
        else:
            new_state = self._aggregate(updates)
        self.model.load_state_dict(new_state)

    def train(
        self,
        rounds: int,
        epochs: int,
        eval_fn: (
            Callable[[nn.Module, Sequence["FederatedClient"]], Dict[str, float]] | None
        ) = None,
    ) -> List[Dict[str, float]]:
        """Run multiple federated rounds and collect evaluation metrics."""
        history: List[Dict[str, float]] = []
        for _ in range(rounds):
            self.train_round(epochs)
            if eval_fn is not None:
                history.append(eval_fn(self.model, self.clients))
        return history


__all__ = ["FederatedServer"]
