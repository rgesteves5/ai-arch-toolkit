"""Lazy SDK import guards — clear error messages when extras not installed."""

from __future__ import annotations


def require_sdk(package: str, extra: str) -> None:
    """Raise ``ImportError`` with install instructions if *package* is missing."""
    try:
        __import__(package)
    except ImportError:
        msg = f"Install the {extra} extra: pip install ai-arch-toolkit[{extra}]"
        raise ImportError(msg) from None
