"""Tests for research utilities."""

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi-activity-recognition"
if "wifi_activity_recognition" not in sys.modules:
    package = types.ModuleType("wifi_activity_recognition")
    package.__path__ = [str(PACKAGE_ROOT)]
    sys.modules["wifi_activity_recognition"] = package
importlib.import_module("wifi_activity_recognition.hardware")
importlib.import_module("wifi_activity_recognition.research")

from wifi_activity_recognition.hardware.base import (  # type: ignore  # noqa: E402
    CSIData,
)
from wifi_activity_recognition.research import (  # type: ignore  # noqa: E402
    DomainAdapter,
    FewShotLearner,
)


def make_csi(value: float) -> CSIData:
    """Create a synthetic :class:`CSIData` packet with constant amplitude."""
    amp = np.full((1, 1, 30), value, dtype=np.float32)
    phase = np.zeros_like(amp)
    return CSIData(
        timestamp=0.0,
        amplitude=amp,
        phase=phase,
        frequency=5200.0,
        bandwidth=20.0,
        n_tx=1,
        n_rx=1,
        n_subcarriers=30,
    )


def test_domain_adapter_mean_shift():
    """Mean-centering adaptation should correct domain shift."""
    model = nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[-1.0], [1.0]]))
    adapter = DomainAdapter(model)
    dataset = [(make_csi(4.0), 0), (make_csi(6.0), 1)]
    baseline = adapter.evaluate_adaptation(dataset)["accuracy"]
    assert baseline == 0.5
    adapter.adapt_to_target([csi for csi, _ in dataset])
    adapted = adapter.evaluate_adaptation(dataset)["accuracy"]
    assert adapted == 1.0


def test_few_shot_learner_prototype_and_novel_detection():
    """Prototype-based learner should classify new activities and detect novelty."""
    learner = FewShotLearner(nn.Identity(), novelty_threshold=3.0)
    support = [make_csi(10.0 + i) for i in range(5)]
    learner.learn_new_activity(support, "jump")
    label, conf, novel = learner.predict_with_confidence(make_csi(11.0))
    assert label == "jump"
    assert conf > 0.3
    assert not novel
    _, _, novel2 = learner.predict_with_confidence(make_csi(-5.0))
    assert novel2
