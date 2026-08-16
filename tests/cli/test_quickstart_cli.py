"""Tests for the quickstart demo workflow.

The issue this covers (#9) asks that the quickstart exercise a real supported
workflow rather than placeholders. The previous documented flow generated
``rng.random`` arrays with random labels: a model cannot learn anything from
that, so it demonstrated only that the commands ran.

These tests therefore check that the pipeline actually *works* -- that the
generated task is learnable and the trained model beats chance -- not merely
that the command exits zero.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from wifi_activity_recognition.cli import cli

# Deliberately small: these assert the pipeline works, not that it reaches a
# particular score. The signal is strong enough that 3 epochs clears chance
# comfortably, and the suite stays fast.
EPOCHS = "3"
SAMPLES = "90"


def _run(tmp_path: Path, *extra: str):
    """Invoke quickstart into a temporary directory."""
    runner = CliRunner()
    return runner.invoke(
        cli,
        [
            "quickstart",
            "--output-dir",
            str(tmp_path / "demo"),
            "--epochs",
            EPOCHS,
            "--samples",
            SAMPLES,
            *extra,
        ],
    )


@pytest.mark.functional
def test_quickstart_runs_end_to_end(tmp_path: Path) -> None:
    """The whole demo completes without hardware or downloads."""
    result = _run(tmp_path)

    assert result.exit_code == 0, result.output
    for stage in ("[1/5]", "[2/5]", "[3/5]", "[4/5]", "[5/5]"):
        assert stage in result.output, f"{stage} missing from:\n{result.output}"
    assert "Quickstart complete" in result.output


@pytest.mark.functional
def test_quickstart_writes_the_artifacts_it_reports(tmp_path: Path) -> None:
    """Every file the output mentions must exist afterwards."""
    result = _run(tmp_path)
    assert result.exit_code == 0, result.output

    demo = tmp_path / "demo"
    for name in ("demo_data.npy", "demo_labels.npy", "demo_model.pt"):
        assert (demo / name).exists(), f"{name} was reported but not written"

    data = np.load(demo / "demo_data.npy")
    labels = np.load(demo / "demo_labels.npy")
    assert data.shape[0] == labels.shape[0] == int(SAMPLES)
    # Models need an explicit channel axis; a 3-D array would fail in training.
    assert data.ndim == 4, f"expected (samples, channels, H, W), got {data.shape}"


@pytest.mark.functional
def test_the_demo_task_is_actually_learnable(tmp_path: Path) -> None:
    """The model must beat chance, or the demo proves nothing.

    Three balanced classes put chance at ~0.33. The signal is a per-class sine
    frequency, so a working pipeline should land far above that. This is the
    assertion that would have failed against the old random-array quickstart.
    """
    # Uses the command's own defaults rather than the reduced settings above:
    # this asserts the experience a user actually gets from `wifi-har-quickstart`
    # with no arguments. The shorter runs elsewhere are for structure only.
    runner = CliRunner()
    result = runner.invoke(cli, ["quickstart", "--output-dir", str(tmp_path / "demo")])
    assert result.exit_code == 0, result.output

    accuracy_line = next(
        (line for line in result.output.splitlines() if "accuracy=" in line), None
    )
    assert accuracy_line is not None, f"no accuracy reported:\n{result.output}"

    accuracy = float(accuracy_line.split("accuracy=")[1].split()[0])
    assert accuracy > 0.6, (
        f"accuracy {accuracy:.2f} is near chance (0.33) for three classes; "
        "the default quickstart is not learnable"
    )


@pytest.mark.functional
def test_quickstart_reports_the_class_it_predicted(tmp_path: Path) -> None:
    """The final step must reload the saved artifact and predict with it."""
    result = _run(tmp_path)
    assert result.exit_code == 0, result.output
    assert "predicted class" in result.output
    assert "actual" in result.output


@pytest.mark.functional
def test_quickstart_is_deterministic_for_a_fixed_seed(tmp_path: Path) -> None:
    """The same seed must generate the same dataset.

    A demo that produces different data each run cannot be documented with
    expected output, and makes support questions unanswerable.
    """
    first = _run(tmp_path / "a", "--seed", "7")
    second = _run(tmp_path / "b", "--seed", "7")
    assert first.exit_code == second.exit_code == 0

    a = np.load(tmp_path / "a" / "demo" / "demo_data.npy")
    b = np.load(tmp_path / "b" / "demo" / "demo_data.npy")
    assert np.array_equal(a, b), "same seed produced different data"


@pytest.mark.functional
def test_saved_artifact_records_its_provenance(tmp_path: Path) -> None:
    """Metadata must identify the artifact as a quickstart product."""
    import torch

    result = _run(tmp_path)
    assert result.exit_code == 0, result.output

    payload = torch.load(
        tmp_path / "demo" / "demo_model.pt", map_location="cpu", weights_only=False
    )
    assert payload["metadata"]["source"] == "quickstart"
    assert payload["model_name"] == "cnn2d"
    # Guards against a demo that silently reports an accuracy it did not reach.
    assert 0.0 <= float(payload["metadata"]["accuracy"]) <= 1.0
    json.dumps(payload["model_kwargs"])  # kwargs must stay plain data
