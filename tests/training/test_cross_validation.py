"""Tests for cross-validation utility."""

import numpy as np
from torch import nn

from wifi_activity_recognition.datasets import (  # type: ignore  # noqa: E402
    Dataset,
    split_dataset,
)
from wifi_activity_recognition.training import Trainer  # type: ignore  # noqa: E402


def make_dataset() -> Dataset:
    """Create dataset used for cross-validation tests."""
    data = np.random.rand(20, 1, 4, 4).astype(np.float32)
    labels = np.random.randint(0, 2, 20)
    train, val, test = split_dataset(data, labels, val_ratio=0.2, test_ratio=0.2)
    return Dataset(train=train, val=val, test=test)


def test_cross_validation_runs():
    """Ensure cross-validation returns metrics for each fold."""
    dataset = make_dataset()
    model = nn.Sequential(nn.Flatten(), nn.Linear(1 * 4 * 4, 2))
    trainer = Trainer(model=model, dataset=dataset, batch_size=4, learning_rate=1e-2)
    results = trainer.cross_validate(folds=3, epochs=1)
    for key in ["accuracy", "precision", "recall", "f1"]:
        assert key in results
        assert 0.0 <= results[key] <= 1.0
    assert len(results["per_class_accuracy"]) == 2
    assert np.array(results["confusion_matrix"]).shape == (2, 2)
