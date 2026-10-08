"""How the geo, weather and natural-event tools write the values they read (T00, output point 5).

Numbers come as their digits, never in scientific notation (``1e-05`` is ``0.00001``), and
instants in ISO 8601, in UTC (``2026-10-08T14:00:00Z``), whatever form the source sends them in.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def plain(value: object) -> str:
    """A number as its digits, exactly as the source sent it; any other value as one line of
    text, and ``None`` as nothing."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(Decimal(repr(value)), "f") if math.isfinite(value) else str(value)
    return " ".join(str(value).split())


def utc(seconds: float) -> str:
    """The instant ``seconds`` after the Unix epoch, in ISO 8601 UTC, to the millisecond when it
    has one.

    Raises:
        OverflowError: The instant is outside the years a ``datetime`` holds.
        ValueError: ``seconds`` is not a number (a NaN).
    """
    instant = _EPOCH + timedelta(seconds=seconds)
    millis = instant.microsecond // 1000
    fraction = f".{millis:03d}" if millis else ""
    return f"{instant:%Y-%m-%dT%H:%M:%S}{fraction}Z"
