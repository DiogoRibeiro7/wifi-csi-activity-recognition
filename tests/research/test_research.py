"""Tests for research utilities."""

from pathlib import Path

import numpy as np
import torch
from torch import nn


from wifi_activity_recognition.research import (  # type: ignore  # noqa: E402
    DomainAdapter,
    FewShotLearner,
    MAMLLearner,
    PrototypicalNetwork,
    RelationNetwork,
)


class _CSIData:
    def __init__(self, amplitude: np.ndarray, phase: np.ndarray) -> None:
        self.timestamp = 0.0
        self.amplitude = amplitude
        self.phase = phase
        self.frequency = 5200.0
        self.bandwidth = 20.0
        self.n_tx = 1
        self.n_rx = 1
        self.n_subcarriers = amplitude.shape[-1]


def make_csi(value: float) -> _CSIData:
    """Create a synthetic CSI packet with constant amplitude."""
    amp = np.full((1, 1, 30), value, dtype=np.float32)
    phase = np.zeros_like(amp)
    return _CSIData(amp, phase)


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


def test_prototypical_network_classification():
    """Prototypical network should classify queries by distance to prototypes."""
    proto = PrototypicalNetwork(nn.Identity())
    support = [(make_csi(1.0), 0), (make_csi(-1.0), 1)]
    proto.fit(support)
    label, conf = proto.predict(make_csi(1.2))
    assert label == 0
    assert conf > 0.5


def test_relation_network_similarity():
    """Relation network should favour classes with higher relation scores."""
    rn = RelationNetwork(nn.Identity())
    support = [(make_csi(1.0), 0), (make_csi(-1.0), 1)]
    label, score = rn.predict(support, make_csi(-1.2))
    assert label == 1
    assert 0.0 <= score <= 1.0


def test_maml_learner_adaptation():
    """MAML learner should improve query accuracy after meta-update."""
    model = nn.Linear(1, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    learner = MAMLLearner(model, inner_lr=0.5)
    support = [(make_csi(2.0), 0), (make_csi(-1.0), 1)]
    query = [(make_csi(2.0), 0), (make_csi(-1.0), 1)]
    acc = learner.adapt(support, query)
    assert acc > 0.5

