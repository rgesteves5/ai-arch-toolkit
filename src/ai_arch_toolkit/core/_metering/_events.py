"""The optional usage audit stream: one immutable event per terminal operation.

Events are a *projection*, never the source of truth — the meter's counters are. A run with a
retaining :class:`UsageSink` satisfies ``replay(events) == projection``; with no sink, no event
is built at all (zero overhead on the default measure-only path). The store builds each event
*under* its lock and emits it to sinks *outside* the lock, so no foreign code runs under the lock.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from ai_arch_toolkit.core._metering._cost import Cost
from ai_arch_toolkit.core._response import Usage

__all__ = ["EventStatus", "UsageEvent", "UsageSink"]

type EventStatus = Literal["settled", "failed", "incomplete", "aborted"]


@dataclass(frozen=True, slots=True, kw_only=True)
class UsageEvent:
    """One terminal operation's audit record. ``metadata`` is already redacted by the store."""

    seq: int
    op_id: str
    span_id: str
    kind: Literal["llm", "tool", "custom"]
    status: EventStatus
    usage: Usage
    cost: Cost
    model: str | None = None
    provider: str | None = None
    mode: str | None = None
    at_s: float = 0.0
    metadata: Mapping[str, str | int | float | bool] = field(default_factory=dict)


class UsageSink(Protocol):
    """A consumer of usage events (logging, OpenTelemetry, a durable audit store).

    Called *outside* the store lock, in completion order. An ``emit`` that raises is caught and
    logged by the store — a faulty sink never breaks a run.
    """

    def emit(self, event: UsageEvent) -> None: ...
