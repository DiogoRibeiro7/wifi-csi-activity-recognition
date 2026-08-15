"""Tests for transformer-based model."""

import pytest
import torch
from torch import nn

from wifi_activity_recognition.models.transformer import (  # type: ignore  # noqa: E402
    PositionalEncoding,
    TransformerModel,
)


def test_forward_shape() -> None:
    """Model produces logits with correct shape."""
    model = TransformerModel(input_dim=32, num_classes=4)
    x = torch.randn(2, 15, 32)
    out = model(x)
    assert out.shape == (2, 4)


# ---------------------------------------------------------------------------
# Temporal order
#
# Self-attention is permutation-invariant. Before positional encoding was
# added these three assertions all failed: the model returned bit-identical
# logits for a sequence, its reverse and a shuffle of it, meaning it could not
# distinguish "sit down" from "stand up".
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_reversing_the_sequence_changes_the_output() -> None:
    """A sequence and its reverse must not map to the same logits."""
    torch.manual_seed(0)
    model = TransformerModel(input_dim=8, num_classes=3).eval()
    x = torch.randn(1, 12, 8)

    with torch.no_grad():
        forward = model(x)
        backward = model(torch.flip(x, dims=[1]))

    assert not torch.allclose(forward, backward, atol=1e-6), (
        "model is invariant to sequence reversal, so it cannot represent the "
        "direction of an activity"
    )


@pytest.mark.regression
def test_shuffling_the_sequence_changes_the_output() -> None:
    """Permuting timesteps must not leave the output unchanged."""
    torch.manual_seed(0)
    model = TransformerModel(input_dim=8, num_classes=3).eval()
    x = torch.randn(1, 12, 8)
    permutation = torch.randperm(12)

    with torch.no_grad():
        ordered = model(x)
        shuffled = model(x[:, permutation, :])

    assert not torch.allclose(ordered, shuffled, atol=1e-6)


@pytest.mark.regression
def test_identical_sequences_still_map_to_identical_outputs() -> None:
    """Order sensitivity must not come at the cost of determinism."""
    torch.manual_seed(0)
    model = TransformerModel(input_dim=8, num_classes=3).eval()
    x = torch.randn(2, 10, 8)

    with torch.no_grad():
        assert torch.allclose(model(x), model(x.clone()), atol=1e-7)


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------


def test_positional_encoding_is_additive_and_shape_preserving() -> None:
    """Encoding adds position without changing tensor shape."""
    encoder = PositionalEncoding(d_model=16, max_len=50)
    x = torch.zeros(2, 20, 16)
    out = encoder(x)

    assert out.shape == x.shape
    # Applied to zeros, the output is the encoding table itself.
    assert not torch.allclose(out, x)


def test_positional_encoding_differs_between_positions() -> None:
    """Distinct timesteps must receive distinct encodings."""
    encoder = PositionalEncoding(d_model=16, max_len=50)
    table = encoder(torch.zeros(1, 10, 16))[0]

    for i in range(table.size(0)):
        for j in range(i + 1, table.size(0)):
            assert not torch.allclose(
                table[i], table[j], atol=1e-6
            ), f"positions {i} and {j} share an encoding"


@pytest.mark.parametrize("d_model", [7, 15, 31, 64])
def test_positional_encoding_supports_odd_d_model(d_model: int) -> None:
    """Odd widths must work: the cosine half is one column shorter."""
    encoder = PositionalEncoding(d_model=d_model, max_len=20)
    assert encoder(torch.zeros(1, 5, d_model)).shape == (1, 5, d_model)


def test_positional_encoding_rejects_sequences_beyond_max_len() -> None:
    """Overlong sequences fail loudly rather than silently truncating."""
    encoder = PositionalEncoding(d_model=8, max_len=4)
    with pytest.raises(ValueError, match="exceeds max_len"):
        encoder(torch.zeros(1, 5, 8))


@pytest.mark.parametrize("bad", [0, -1])
def test_positional_encoding_rejects_invalid_dimensions(bad: int) -> None:
    """Constructor validates its arguments."""
    with pytest.raises(ValueError):
        PositionalEncoding(d_model=bad)
    with pytest.raises(ValueError):
        PositionalEncoding(d_model=8, max_len=bad)


@pytest.mark.regression
def test_positional_encoding_stays_out_of_the_state_dict() -> None:
    """The encoding is derived, so checkpoints must not carry it.

    Keeping the buffer non-persistent means checkpoints written before
    positional encoding existed still load with ``strict=True``.
    """
    model = TransformerModel(input_dim=8, num_classes=3)
    assert not [key for key in model.state_dict() if key.startswith("pos_encoder.")]


# ---------------------------------------------------------------------------
# Training behaviour
# ---------------------------------------------------------------------------


def test_gradients_reach_every_learnable_parameter() -> None:
    """A backward pass must not leave part of the network unconnected."""
    model = TransformerModel(input_dim=8, num_classes=3)
    logits = model(torch.randn(4, 10, 8))
    nn.CrossEntropyLoss()(logits, torch.randint(0, 3, (4,))).backward()

    unreached = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and (param.grad is None or param.grad.abs().sum() == 0)
    ]
    assert not unreached, f"no gradient reached: {unreached}"


@pytest.mark.slow
def test_model_can_overfit_two_order_reversed_classes() -> None:
    """The model must learn a task that is *only* solvable using order.

    Both classes contain the same frames; they differ only in direction. Before
    positional encoding this was unlearnable in principle, not merely hard.
    """
    torch.manual_seed(0)
    base = torch.randn(1, 16, 8)
    x = torch.cat([base, torch.flip(base, dims=[1])]).repeat(4, 1, 1)
    y = torch.tensor([0, 1]).repeat(4)

    model = TransformerModel(input_dim=8, num_classes=2, d_model=32, nhead=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    criterion = nn.CrossEntropyLoss()

    first = last = None
    for step in range(150):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        if step == 0:
            first = loss.item()
        last = loss.item()

    assert last < first, f"loss did not decrease ({first:.4f} -> {last:.4f})"

    model.eval()
    with torch.no_grad():
        accuracy = (model(x).argmax(dim=1) == y).float().mean().item()
    assert accuracy > 0.9, f"only reached {accuracy:.0%} on a direction-only task"
