"""Integration tests for the CLI streaming command."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
from click.testing import CliRunner

from wifi_activity_recognition.cli import cli
from wifi_activity_recognition.hardware.base import CSIData


class DummyReader:
    """Minimal CSI reader producing synthetic packets."""

    def __enter__(self) -> "DummyReader":
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Exit the context manager."""
        return False

    def stream(self) -> Iterator[CSIData]:
        """Yield a couple of synthetic CSI packets."""
        amplitude = np.ones((1, 1, 4))
        phase = np.zeros((1, 1, 4))
        metadata = {"firmware_version": "v1"}
        for _ in range(2):
            yield CSIData(
                timestamp=0.0,
                amplitude=amplitude,
                phase=phase,
                frequency=2400.0,
                bandwidth=20.0,
                n_tx=1,
                n_rx=1,
                n_subcarriers=4,
                metadata=metadata,
            )


def test_stream_command(monkeypatch, tmp_path: Path) -> None:
    """Running the stream command saves collected packets."""

    def fake_csi_reader(*_args, **_kwargs):
        return DummyReader()

    # Patch factory function and saving utility
    monkeypatch.setattr("wifi_activity_recognition.hardware.CSIReader", fake_csi_reader)
    monkeypatch.setattr(
        "wifi_activity_recognition.utils.io.save_csi_data",
        lambda data, path: Path(path).write_text(str(len(data))),
    )

    out_file = tmp_path / "stream.json"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "stream",
            "--hardware",
            "esp32",
            "--duration",
            "1",
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_file.read_text() == "2"
