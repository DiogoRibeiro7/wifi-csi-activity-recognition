from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))
from deployment.edge.raspberry_pi.optimize import optimize_model


class SimpleModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_optimize_model(tmp_path: Path) -> None:
    model = SimpleModel()
    scripted = torch.jit.script(model)
    model_path = tmp_path / "model.pt"
    scripted.save(str(model_path))

    output_path = tmp_path / "model_optimized.pt"
    optimize_model(model_path, output_path)

    assert output_path.exists()
    optimized = torch.jit.load(output_path)
    sample = torch.randn(1, 2)
    _ = optimized(sample)
