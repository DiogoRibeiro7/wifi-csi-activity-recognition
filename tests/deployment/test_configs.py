"""Tests for deployment configuration and optimization utilities."""
# noqa: D100
import sys
from pathlib import Path

import torch
import yaml

BASE_DIR = Path(__file__).resolve().parents[2] / "deployment"
sys.path.append(str(BASE_DIR.parent))

from deployment.edge.raspberry_pi.optimize import optimize_model  # noqa: E402


def test_dockerfile_stages() -> None:
    """Ensure Dockerfile defines dev, prod and edge stages."""
    dockerfile = (BASE_DIR / "docker" / "Dockerfile").read_text()
    assert "AS dev" in dockerfile
    assert "AS prod" in dockerfile
    assert "AS edge" in dockerfile


def test_docker_compose_services() -> None:
    """Verify docker-compose exposes dev, prod and edge services."""
    compose_path = BASE_DIR / "docker" / "docker-compose.yml"
    services = yaml.safe_load(compose_path.read_text())["services"]
    assert {"dev", "prod", "edge"}.issubset(services)


def test_kubernetes_deployment() -> None:
    """Check Kubernetes manifest points to correct container image."""
    k8s_path = BASE_DIR / "kubernetes" / "deployment.yaml"
    deployment = yaml.safe_load(k8s_path.read_text())
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    # Matches the distribution name; Kubernetes object names are RFC-1123
    # labels, so the underscored import name is not valid here.
    assert container["image"] == "wifi-activity-recognition:latest"


def test_optimize_model(tmp_path: Path) -> None:
    """Quantization script should emit an optimized model file."""
    model = torch.nn.Linear(4, 2)
    scripted = torch.jit.script(model)
    input_model = tmp_path / "model.pt"
    scripted.save(str(input_model))
    output_model = tmp_path / "model_opt.pt"
    optimize_model(input_model, output_model)
    assert output_model.exists()
