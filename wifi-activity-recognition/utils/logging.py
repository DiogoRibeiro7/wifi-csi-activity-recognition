"""Structured logging utilities."""

from __future__ import annotations

import logging
from typing import Optional

_LOG_FORMAT = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"


def setup_logging(
    name: Optional[str] = None, level: int = logging.INFO
) -> logging.Logger:
    """Configure and return a logger with a standard format.

    Parameters
    ----------
    name: str, optional
        Name of the logger. Defaults to the root logger when ``None``.
    level: int
        Logging level, e.g., ``logging.INFO``.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
