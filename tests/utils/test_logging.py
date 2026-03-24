"""Tests for logging utilities."""

import json
import logging
import sys
import types
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi_activity_recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.utils.logging import (  # type: ignore  # noqa: E402
    setup_logging,
)


def test_setup_logging_emits_messages(caplog):
    """Logger emits messages at requested level."""
    logger = setup_logging("test", level=logging.DEBUG)
    with caplog.at_level(logging.DEBUG, logger="test"):
        logger.debug("hello")
    assert "hello" in caplog.text
    assert logger.level == logging.DEBUG


def test_json_logging_format(capfd):
    """JSON formatter produces parseable output."""
    logger = setup_logging("json", json_format=True)
    logger.propagate = False
    logger.info("world")
    err = capfd.readouterr().err.strip()
    parsed = json.loads(err)
    assert parsed["message"] == "world"

