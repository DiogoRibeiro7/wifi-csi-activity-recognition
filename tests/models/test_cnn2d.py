import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Make the package importable despite repository layout using hyphenated name
# ---------------------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi-activity-recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.models.cnn2d import (  # type: ignore  # noqa: E402
    CNN2DModel,
    CNN2DTensorFlowModel,
)

try:  # noqa: E402
    import tensorflow as tf
except Exception:  # pragma: no cover - tensorflow may be unavailable
    tf = None


@pytest.mark.parametrize("subcarriers", [30, 64, 128])
def test_cnn2d_forward_shapes(subcarriers: int) -> None:
    model = CNN2DModel(num_classes=5)
    x = torch.randn(2, 1, subcarriers, 200)
    out = model(x)
    assert out.shape == (2, 5)


def test_cnn2d_training_step_decreases_loss() -> None:
    model = CNN2DModel(num_classes=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(10, 1, 30, 50)
    y = torch.randint(0, 3, (10,))
    loss1 = criterion(model(x), y)
    for _ in range(5):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    loss2 = criterion(model(x), y)
    assert loss2.item() < loss1.item()


@pytest.mark.skipif(tf is None, reason="TensorFlow not installed")
@pytest.mark.parametrize("subcarriers", [30, 64, 128])
def test_cnn2d_tf_forward_shapes(subcarriers: int) -> None:
    model = CNN2DTensorFlowModel(num_classes=5)
    x = tf.random.normal((2, subcarriers, 200, 1))
    out = model(x, training=False)
    assert out.shape == (2, 5)


@pytest.mark.skipif(tf is None, reason="TensorFlow not installed")
def test_cnn2d_tf_training_step_decreases_loss() -> None:
    model = CNN2DTensorFlowModel(num_classes=3)
    optimizer = tf.keras.optimizers.SGD(0.1)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    x = tf.random.normal((10, 30, 50, 1))
    y = tf.random.uniform((10,), maxval=3, dtype=tf.int32)
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss1 = loss_fn(y, logits)
    grads = tape.gradient(loss1, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    loss2 = loss_fn(y, model(x, training=False))
    assert loss2.numpy() <= loss1.numpy()
