"""How the tools write the values they read (T00, output point 5).

Numbers come as their digits, never in scientific notation (``1e-05`` is ``0.00001``), and
instants in ISO 8601, in UTC (``2026-10-08T14:00:00Z``), whatever form the source sends them in.
"""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime, timedelta
from decimal import Decimal

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
# A decimal number as text, in ASCII digits (``+1.750``, ``1.0E7``, ``.5``).
_DECIMAL_RE = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
# The furthest a number sent as text is written out from its first digit: one SPARQL cell of
# ``1E1000000`` would be a line of a million digits, so past this it keeps the source's text.
_MAX_EXPONENT = 100


def plain(value: object) -> str:
    """A number as its digits, never in scientific notation, as the source's JSON had it: an
    integer as one, a float with its point (``21.0``, ``1e16`` as ``10000000000000000.0``); any
    other value as one line of text, ``true``/``false`` as JSON writes them, ``None`` as nothing.

    Text stays as it is (whitespace aside): a code such as ``"007"`` keeps its zeros. A number
    the source sends as text goes through :func:`decimal_text`.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        text = format(Decimal(repr(abs(value) if value == 0 else value)), "f")
        return text if "." in text else f"{text}.0"
    return " ".join(str(value).split())


def decimal_text(text: str) -> str:
    """A number the source sends as text (``"+1.750"``, ``"1.0E7"``) in plain digits, as precise
    as it was sent: without its ``+`` and its exponent, with the digits it gave.

    Text that is no decimal number in ASCII digits comes back as it was, and so does one whose
    first digit lies more than ``_MAX_EXPONENT`` places from the point.
    """
    stripped = text.strip()
    if not _DECIMAL_RE.fullmatch(stripped):
        return text
    number = Decimal(stripped)
    if abs(number.adjusted()) > _MAX_EXPONENT:
        return text
    return format(number, "f")


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
