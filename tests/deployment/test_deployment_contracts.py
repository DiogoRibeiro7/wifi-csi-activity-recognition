"""Consistency between deployment assets and what the package can actually do.

These assets described a service that does not exist. The Dockerfile's `prod`
and `edge` stages ran ``python -m wifi_activity_recognition`` with no
``__main__`` module, so both containers exited immediately. The Kubernetes
manifest probed ``/health`` on port 8080 while nothing in the package binds a
socket, which would have driven the pod into CrashLoopBackOff even had the
container started.

Each test below pins one of those contracts so the assets cannot drift back
out of step with the code.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT = REPO_ROOT / "deployment"
DOCKERFILE = DEPLOYMENT / "docker" / "Dockerfile"
COMPOSE = DEPLOYMENT / "docker" / "docker-compose.yml"
K8S = DEPLOYMENT / "kubernetes" / "deployment.yaml"


# ---------------------------------------------------------------------------
# The entry point the container images rely on
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_package_is_executable_with_python_dash_m() -> None:
    """``python -m wifi_activity_recognition`` must run.

    Both container images invoke the package this way. Without a ``__main__``
    module this exits non-zero and the container dies on start.
    """
    result = subprocess.run(
        [sys.executable, "-m", "wifi_activity_recognition", "--version"],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, f"python -m failed with:\n{result.stderr.strip()}"
    assert "wifi" in result.stdout.lower()


@pytest.mark.regression
def test_dash_m_exposes_the_same_commands_as_the_console_scripts() -> None:
    """The module entry point must not be a reduced surface."""
    result = subprocess.run(
        [sys.executable, "-m", "wifi_activity_recognition", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0
    for command in ("train", "predict", "stream"):
        assert command in result.stdout, f"'{command}' missing from -m help"


# ---------------------------------------------------------------------------
# Dockerfile
# ---------------------------------------------------------------------------


def test_dockerfile_defines_the_expected_stages() -> None:
    """Compose targets these three by name."""
    dockerfile = DOCKERFILE.read_text()
    for stage in ("AS base", "AS dev", "AS prod", "AS edge"):
        assert stage in dockerfile


@pytest.mark.regression
@pytest.mark.parametrize("stage", ["prod", "edge"])
def test_runtime_stages_install_the_package(stage: str) -> None:
    """Runtime images must install the package, not rely on the workdir.

    Copying the source and setting WORKDIR makes imports resolve by accident
    and leaves the console scripts absent from the image entirely.
    """
    body = DOCKERFILE.read_text().split(f"AS {stage}")[1].split("\nFROM ")[0]
    assert (
        "pip install" in body and " ." in body
    ), f"the {stage} stage does not install the package itself"


def test_dockerfile_entrypoints_use_a_module_that_exists() -> None:
    """Every ``python -m`` CMD must name an importable module."""
    for line in DOCKERFILE.read_text().splitlines():
        if "python" in line and "-m" in line and line.strip().startswith("CMD"):
            assert "wifi_activity_recognition" in line
            assert (
                REPO_ROOT / "wifi_activity_recognition" / "__main__.py"
            ).exists(), "CMD runs the package as a module but __main__.py is absent"


# ---------------------------------------------------------------------------
# Compose and Kubernetes must not advertise a server that does not exist
# ---------------------------------------------------------------------------


def test_compose_exposes_the_expected_services() -> None:
    """Service names are part of the documented interface."""
    services = yaml.safe_load(COMPOSE.read_text())["services"]
    assert {"dev", "prod", "edge"}.issubset(services)


@pytest.mark.regression
def test_compose_does_not_publish_ports_for_a_package_that_never_listens() -> None:
    """Publishing a port advertises a listener. There isn't one."""
    services = yaml.safe_load(COMPOSE.read_text())["services"]

    publishing = [name for name, spec in services.items() if spec.get("ports")]
    assert not publishing, (
        f"{publishing} publish ports, but no module in the package binds a "
        "socket; add a real server interface before advertising one"
    )


@pytest.mark.regression
def test_kubernetes_probes_do_not_assume_an_http_endpoint() -> None:
    """httpGet probes against /health would fail forever.

    Nothing serves HTTP, so an httpGet liveness probe guarantees
    CrashLoopBackOff. Probes must exercise something the image can do.
    """
    container = yaml.safe_load(K8S.read_text())["spec"]["template"]["spec"][
        "containers"
    ][0]

    for probe_name in ("livenessProbe", "readinessProbe"):
        probe = container.get(probe_name)
        assert probe is not None, f"{probe_name} missing"
        assert (
            "httpGet" not in probe
        ), f"{probe_name} uses httpGet, but the package serves no HTTP"
        assert "exec" in probe, f"{probe_name} should exec a command"


@pytest.mark.regression
def test_kubernetes_probe_commands_are_runnable() -> None:
    """The probe command must actually succeed against the package."""
    container = yaml.safe_load(K8S.read_text())["spec"]["template"]["spec"][
        "containers"
    ][0]
    command = container["livenessProbe"]["exec"]["command"]

    # Run the same command locally, swapping the interpreter for this one.
    assert command[0] == "python"
    result = subprocess.run(
        [sys.executable, *command[1:]], capture_output=True, text=True, timeout=120
    )
    assert (
        result.returncode == 0
    ), f"liveness probe command {command} fails: {result.stderr.strip()}"


def test_kubernetes_image_matches_the_distribution_name() -> None:
    """Kubernetes object names are RFC-1123 labels; underscores are invalid."""
    deployment = yaml.safe_load(K8S.read_text())
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == "wifi-activity-recognition:latest"
    assert deployment["metadata"]["name"] == "wifi-activity-recognition"
