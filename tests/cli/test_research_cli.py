"""CLI workflow integration tests."""

from pathlib import Path

import numpy as np
from click.testing import CliRunner


from wifi_activity_recognition.cli import cli  # type: ignore  # noqa: E402


def _make_data(tmp_path: Path):
    """Create dummy data and label arrays."""
    data = np.random.rand(20, 1, 8, 8).astype(np.float32)
    labels = np.random.randint(0, 2, 20)
    data_path = tmp_path / "data.npy"
    labels_path = tmp_path / "labels.npy"
    np.save(data_path, data)
    np.save(labels_path, labels)
    return data_path, labels_path


def test_collect(tmp_path: Path):
    """Collect synthetic CSI packets via the CLI."""
    out_path = tmp_path / "csi.h5"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "collect",
            "--hardware",
            "intel_5300",
            "--packets",
            "2",
            "--output",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_path.exists()


def test_autotrain(tmp_path: Path):
    """Run automated hyperparameter search."""
    data_path, labels_path = _make_data(tmp_path)
    model_path = tmp_path / "best.pt"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "autotrain",
            "--data",
            str(data_path),
            "--labels",
            str(labels_path),
            "--model",
            "cnn2d",
            "--hardware",
            "intel_5300",
            "--epochs",
            "1",
            "--learning-rates",
            "0.01,0.02",
            "--batch-sizes",
            "2,4",
            "--output",
            str(model_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert model_path.exists()


def test_visualize(tmp_path: Path):
    """Render a CSI heatmap via the CLI."""
    img_path = tmp_path / "vis.png"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "visualize",
            "--hardware",
            "intel_5300",
            "--num-packets",
            "1",
            "--save",
            str(img_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert img_path.exists()


def test_benchmark(tmp_path: Path):
    """Generate a performance report."""
    data_path, labels_path = _make_data(tmp_path)
    import torch
    import torch.nn as nn

    net = nn.Sequential(nn.Flatten(), nn.Linear(64, 2))
    torch.save(net, tmp_path / "model.pth")

    runner = CliRunner()
    report = tmp_path / "report.json"
    result = runner.invoke(
        cli,
        [
            "benchmark",
            "--model",
            str(tmp_path / "model.pth"),
            "--data",
            str(data_path),
            "--labels",
            str(labels_path),
            "--hardware",
            "intel_5300",
            "--output",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    assert report.exists()


def test_export(tmp_path: Path):
    """Export a model to ONNX via the CLI."""
    import torch
    import torch.nn as nn

    model = nn.Sequential(nn.Flatten(), nn.Linear(64, 2))
    model_path = tmp_path / "model.pth"
    torch.save(model, model_path)
    onnx_path = tmp_path / "model.onnx"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "export",
            "--model",
            str(model_path),
            "--target",
            "edge",
            "--input-shape",
            "1,1,8,8",
            "--output",
            str(onnx_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert onnx_path.exists()

