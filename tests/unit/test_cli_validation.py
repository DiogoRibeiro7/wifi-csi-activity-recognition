"""Unit tests for lightweight CLI validation helpers."""

from __future__ import annotations

import pytest

from wifi_activity_recognition import cli as cli_module

pytestmark = [pytest.mark.unit, pytest.mark.smoke]


def test_validate_hardware_option_accepts_registered_value(monkeypatch) -> None:
    """Registered hardware names should be accepted unchanged."""
    monkeypatch.setattr(
        cli_module,
        "_list_cli_hardware",
        lambda include_all=False: ["esp32", "qualcomm", *(["all"] if include_all else [])],
    )

    value = cli_module._validate_hardware_option(None, None, "qualcomm")

    assert value == "qualcomm"


def test_validate_hardware_option_rejects_unknown_value(monkeypatch) -> None:
    """Unknown hardware names should raise a click validation error."""
    monkeypatch.setattr(cli_module, "_list_cli_hardware", lambda include_all=False: ["esp32"])

    with pytest.raises(Exception, match="Unsupported hardware 'intel_5300'"):
        cli_module._validate_hardware_option(None, None, "intel_5300")


def test_validate_info_hardware_option_accepts_all(monkeypatch) -> None:
    """The info command should allow the synthetic 'all' selector."""
    monkeypatch.setattr(
        cli_module,
        "_list_cli_hardware",
        lambda include_all=False: ["esp32", "qualcomm", *(["all"] if include_all else [])],
    )

    value = cli_module._validate_info_hardware_option(None, None, "all")

    assert value == "all"
