"""Shared logging helpers for ai_arch_toolkit."""

from __future__ import annotations

import logging

PACKAGE_LOGGER_NAME = "ai_arch_toolkit"


def _configure_package_logger() -> logging.Logger:
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.addHandler(logging.NullHandler())
    return logger


package_logger = _configure_package_logger()
