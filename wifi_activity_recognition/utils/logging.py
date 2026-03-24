"""Structured logging utilities."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

_LOG_FORMAT = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"


class _JsonFormatter(logging.Formatter):
    """Simple JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        """Serialize a log record into a compact JSON string.

        Args:
            record: Logging record produced by the Python logging framework.

        Returns:
            JSON string containing timestamp, level, logger name, and message.
        """
        return json.dumps(
            {
                "time": self.formatTime(record),
                "level": record.levelname,
                "name": record.name,
                "message": record.getMessage(),
            }
        )


def setup_logging(
    name: Optional[str] = None,
    level: int = logging.INFO,
    *,
    logfile: str | Path | None = None,
    json_format: bool = False,
) -> logging.Logger:
    """Configure and return a logger with optional JSON and file output.

    Parameters
    ----------
    name:
        Name of the logger. Defaults to the root logger when ``None``.
    level:
        Logging level, e.g. ``logging.INFO``.
    logfile:
        Optional path to a log file. When provided, messages are written both to
        the stream and the file.
    json_format:
        Emit logs in JSON when ``True``. Otherwise use a human-readable format.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    formatter: logging.Formatter = (
        _JsonFormatter() if json_format else logging.Formatter(_LOG_FORMAT)
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if logfile is not None:
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.setLevel(level)
    return logger
