"""Tests for logging utilities."""

import json
import logging

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
