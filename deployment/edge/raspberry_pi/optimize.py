"""Optimize models for Raspberry Pi deployment.

This script loads a TorchScript model, applies dynamic quantization,
and saves the optimized model for ARM devices.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def optimize_model(model_path: Path, output_path: Path) -> None:
    """Quantize and save a TorchScript model for edge devices.

    Parameters
    ----------
    model_path:
        Path to the input TorchScript model.
    output_path:
        File path to store the optimized model.
    """
    model = torch.jit.load(model_path)
    optimized = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    optimized.save(str(output_path))


def main() -> None:
    """CLI entry point for model optimization."""
    parser = argparse.ArgumentParser(
        description="Optimize model for Raspberry Pi deployment"
    )
    parser.add_argument("model", type=Path, help="Input TorchScript model path")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model_optimized.pt"),
        help="Path to save optimized model",
    )
    args = parser.parse_args()
    optimize_model(args.model, args.output)


if __name__ == "__main__":
    main()
