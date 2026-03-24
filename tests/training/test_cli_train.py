"""CLI integration tests for training."""

from pathlib import Path

import numpy as np
from click.testing import CliRunner


from wifi_activity_recognition.cli import cli  # type: ignore  # noqa: E402


def test_cli_train(tmp_path: Path):
    """Run the CLI training command and ensure a model is saved."""
    data = np.random.rand(20, 1, 8, 8).astype(np.float32)
    labels = np.random.randint(0, 2, 20)
    data_path = tmp_path / "data.npy"
    labels_path = tmp_path / "labels.npy"
    np.save(data_path, data)
    np.save(labels_path, labels)
    model_path = tmp_path / "model.pt"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "train",
            "--data",
            str(data_path),
            "--labels",
            str(labels_path),
            "--model",
            "cnn2d",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--output",
            str(model_path),
            "--hardware",
            "esp32",
        ],
    )
    assert result.exit_code == 0, result.output
    assert model_path.exists()

