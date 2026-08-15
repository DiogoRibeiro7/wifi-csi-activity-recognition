"""Tests for VisionTransformerModel."""

import pytest
import torch

from wifi_activity_recognition.models.vision_transformer import (  # noqa: E402
    VisionTransformerModel,
    VisionTransformerTensorFlowModel,
)


@pytest.mark.parametrize("shape", [(2, 1, 30, 50), (2, 1, 40, 40)])
def test_vit_forward_shapes(shape: tuple[int, int, int, int]) -> None:
    """Model should handle variable input sizes."""
    model = VisionTransformerModel(num_classes=5)
    x = torch.randn(shape)
    out = model(x)
    assert out.shape == (shape[0], 5)


def test_vit_gradients() -> None:
    """Gradients should flow through the network."""
    model = VisionTransformerModel(num_classes=3)
    x = torch.randn(4, 1, 30, 50)
    y = torch.randint(0, 3, (4,))
    loss = torch.nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None


def test_vit_seq_to_seq() -> None:
    """Sequence-to-sequence mode returns patch predictions."""
    model = VisionTransformerModel(num_classes=2, seq_to_seq=True)
    x = torch.randn(1, 1, 30, 50)
    out = model(x)
    assert out.shape[0] == 1
    # 30/5 * 50/5 = 60 patches
    assert out.shape[1] == 60
    assert out.shape[2] == 2


def test_vit_tf_forward() -> None:
    """Ensure TensorFlow implementation runs a forward pass."""
    tf = pytest.importorskip("tensorflow")
    model = VisionTransformerTensorFlowModel(num_classes=3)
    x = tf.random.normal((2, 30, 50, 1))
    out = model(x, training=False)
    assert out.shape == (2, 3)


def test_vit_tf_seq_to_seq() -> None:
    """Ensure TensorFlow sequence-to-sequence mode returns patch predictions."""
    tf = pytest.importorskip("tensorflow")
    model = VisionTransformerTensorFlowModel(num_classes=2, seq_to_seq=True)
    x = tf.random.normal((1, 30, 50, 1))
    out = model(x, training=False)
    assert out.shape[0] == 1
    assert out.shape[1] == 60
    assert out.shape[2] == 2
