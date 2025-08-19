"""Tests for federated learning utilities."""

from __future__ import annotations

import copy
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi-activity-recognition"
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
)


def make_dataset(seed: int) -> Dataset:
    """Create deterministic dataset for a given seed."""
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(20, 1, 4, 4)).astype(np.float32)
    labels = rng.integers(0, 2, size=20)
    split = (data[:15], labels[:15]), (data[15:], labels[15:]), (data[15:], labels[15:])
    return Dataset(train=split[0], val=split[1], test=split[2])


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
