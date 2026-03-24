"""Tests for federated learning utilities."""

from __future__ import annotations

import copy
import sys
import types
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi_activity_recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.datasets import Dataset  # type: ignore  # noqa: E402
from wifi_activity_recognition.training import (  # type: ignore  # noqa: E402
    FederatedClient,
    FederatedServer,
    add_gaussian_noise,
    clip_gradients,
    fed_avg,
    hardware_balanced_selector,
    run_simulation,
)


def make_dataset(seed: int) -> Dataset:
    """Create deterministic dataset for a given seed."""
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(20, 1, 4, 4)).astype(np.float32)
    labels = rng.integers(0, 2, size=20)
    split = (data[:15], labels[:15]), (data[15:], labels[15:]), (data[15:], labels[15:])
    return Dataset(train=split[0], val=split[1], test=split[2])


def make_client(seed: int, hardware: str, env: str) -> FederatedClient:
    """Create a federated client with labeled hardware and environment."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(1 * 4 * 4, 2))
    return FederatedClient(
        model=model, dataset=make_dataset(seed), hardware=hardware, environment=env
    )


def test_fedavg():
    """Ensure FedAvg averages parameters correctly."""
    s1 = {"w": torch.tensor([1.0, 2.0])}
    s2 = {"w": torch.tensor([3.0, 4.0])}
    avg = fed_avg([(s1, 1), (s2, 1)])
    assert torch.allclose(avg["w"], torch.tensor([2.0, 3.0]))


def test_privacy_utils():
    """Verify clipping and noise addition modify the state as expected."""
    state = {"w": torch.ones(5)}
    clipped = clip_gradients(state, max_norm=1.0)
    assert torch.linalg.norm(torch.stack(list(clipped.values()))) <= 1.0 + 1e-6
    noised = add_gaussian_noise(state, std=0.1)
    assert not torch.allclose(state["w"], noised["w"])


def test_federated_round():
    """Ensure the server updates the global model after a training round."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(1 * 4 * 4, 2))
    clients = [
        FederatedClient(model=copy.deepcopy(model), dataset=make_dataset(0)),
        FederatedClient(model=copy.deepcopy(model), dataset=make_dataset(1)),
    ]
    server = FederatedServer(model=model, clients=clients)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    server.train_round(epochs=1)
    updated = model.state_dict()
    changed = any(not torch.allclose(initial[k], updated[k]) for k in initial)
    assert changed


def test_hardware_balanced_selection():
    """Selection strategy should include multiple hardware types."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(1 * 4 * 4, 2))
    clients = [
        make_client(0, "intel", "home"),
        make_client(1, "esp32", "office"),
        make_client(2, "intel", "lab"),
    ]
    server = FederatedServer(
        model=model,
        clients=clients,
        fraction=2 / 3,
        selector=hardware_balanced_selector,
    )
    selected = server._select_clients()
    hardwares = {c.hardware for c in selected}
    assert hardwares == {"intel", "esp32"}


def eval_fn(model: nn.Module, datasets: Dict[str, Dataset]) -> Dict[str, float]:
    """Return accuracy evaluation across environments."""
    model.eval()
    metrics: Dict[str, float] = {}
    with torch.no_grad():
        for env, ds in datasets.items():
            x, y = ds.test
            logits = model(torch.tensor(x))
            preds = logits.argmax(dim=1)
            metrics[env] = float((preds == torch.tensor(y)).float().mean())
    return metrics


def test_simulation_returns_metrics():
    """Simulation should provide per-environment metrics each round."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(1 * 4 * 4, 2))
    clients = [
        make_client(0, "intel", "home"),
        make_client(1, "esp32", "office"),
    ]
    server = FederatedServer(model=model, clients=clients)
    history = run_simulation(server, rounds=2, epochs=1, eval_fn=eval_fn)
    assert len(history) == 2
    assert set(history[0].keys()) == {"home", "office"}

