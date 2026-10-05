"""What a paid tool's service billed during one tool call (D56).

The executor opens a record for each call it meters (``billing``); the code that reaches a paid
service (the toolkit's HTTP door) adds the units the service billed (``bill``), and the executor
charges them at the table's price when the call ends. The record follows the context, into the
thread a synchronous tool runs in. Outside a metered call, ``bill`` records nothing.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar

type Billed = list[tuple[str, int]]
"""The units billed during one call, as (price name, units) in order."""

_RECORD: ContextVar[Billed | None] = ContextVar("tool_billing", default=None)


def bill(name: str, units: int = 1) -> None:
    """Record that the service priced as ``name`` billed ``units`` for a request just made."""
    record = _RECORD.get()
    if record is not None and units > 0:
        record.append((name, units))


@contextlib.contextmanager
def billing() -> Iterator[Billed]:
    """Open the record of one call; it collects what the code inside bills."""
    record: Billed = []
    token = _RECORD.set(record)
    try:
        yield record
    finally:
        _RECORD.reset(token)
