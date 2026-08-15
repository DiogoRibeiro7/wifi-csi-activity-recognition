"""Tests for advanced domain adaptation utilities."""

import numpy as np
import torch
from torch import nn

from wifi_activity_recognition.hardware.base import CSIData  # noqa: E402
from wifi_activity_recognition.research.domain_adaptation import (  # noqa: E402
    DomainAdapter,
    DomainAdversarialNetwork,
    coral_loss,
    mmd_loss,
)


def make_csi(
    value: float = 1.0, subcarriers: int = 30, n_rx: int = 1, n_tx: int = 1
) -> CSIData:
    """Create a synthetic CSI packet with constant amplitude."""
    amp = np.full((n_rx, n_tx, subcarriers), value, dtype=np.float32)
    phase = np.zeros_like(amp)
    return CSIData(
        timestamp=0.0,
        amplitude=amp,
        phase=phase,
        frequency=5200.0,
        bandwidth=20.0,
        n_tx=n_tx,
        n_rx=n_rx,
        n_subcarriers=subcarriers,
    )


def test_coral_and_mmd_zero_when_same():
    """CORAL and MMD losses should vanish for identical features."""
    feat = torch.randn(4, 5)
    assert float(coral_loss(feat, feat)) < 1e-6
    assert float(mmd_loss(feat, feat)) < 1e-6


def test_domain_adversarial_forward_shapes():
    """DANN forward pass should return class and domain logits."""
    feature_extractor = nn.Identity()
    class_classifier = nn.Linear(5, 2)
    model = DomainAdversarialNetwork(feature_extractor, class_classifier, feature_dim=5)
    x = torch.randn(3, 5)
    cls, dom = model(x, lambd=1.0)
    assert cls.shape == (3, 2)
    assert dom.shape == (3, 2)


def test_cross_hardware_adaptation_helpers():
    """Subcarrier and antenna adaptation should adjust shapes."""
    csi = make_csi(subcarriers=30)
    resized = DomainAdapter.match_subcarrier_count(csi, 64)
    assert resized.amplitude.shape == (1, 1, 64)
    padded = DomainAdapter.match_antenna_config(csi, 2, 2)
    assert padded.amplitude.shape == (2, 2, 30)


def test_domain_adapter_mmd_adaptation():
    """MMD-based adaptation should correct mean shifts."""
    model = nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[-1.0], [1.0]]))
    adapter = DomainAdapter(model)
    dataset = [(make_csi(4.0), 0), (make_csi(6.0), 1)]
    adapter.adapt_to_target([csi for csi, _ in dataset], method="mmd")
    acc = adapter.evaluate_adaptation(dataset)["accuracy"]
    assert acc == 1.0
