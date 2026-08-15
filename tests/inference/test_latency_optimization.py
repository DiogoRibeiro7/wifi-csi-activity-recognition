"""Tests for latency optimization utilities."""

import torch

from wifi_activity_recognition.inference import (  # type: ignore  # noqa: E402
    dynamic_batch,
    prune_model,
    quantize_model,
)


def test_quantize_and_prune_forward() -> None:
    """Models remain callable after quantization and pruning."""
    model = torch.nn.Sequential(torch.nn.Linear(4, 2), torch.nn.ReLU())
    q_model = quantize_model(model)
    pruned = prune_model(q_model, amount=0.5)
    out = pruned(torch.randn(1, 4))
    assert out.shape == (1, 2)


def test_dynamic_batch() -> None:
    """Dynamic batching groups items up to requested size."""
    items = [1, 2, 3, 4, 5]
    batches = list(dynamic_batch(items, 2))
    assert batches == [[1, 2], [3, 4], [5]]
