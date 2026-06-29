"""Ambient binding for the meter: a run-level :class:`MeterScope` reached via ``ContextVar``.

A charge site never threads a meter through call signatures — it reads the *current* scope from
the context. Three modes fall out of this:

* no scope bound          -> ``current_meter()`` is ``None``  -> nothing measured, nothing blocked;
* scope, ``controller=None`` -> measure-only (the default for a Flow/Agent run);
* scope + controller       -> measure **and** enforce.

Spans nest through a second ``ContextVar`` so ``open_span`` composes (a step under the run, a tool
under a step) without passing ids around. The scope owns the :class:`MeterStore`; ``__exit__``
closes it (PENDING -> aborted, STARTED -> incomplete).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ai_arch_toolkit.core._metering._store import MeterStore

if TYPE_CHECKING:
    from ai_arch_toolkit.core._metering._admission import AdmissionController, MeterSnapshot
    from ai_arch_toolkit.core._metering._cost import Cost
    from ai_arch_toolkit.core._metering._events import UsageSink
    from ai_arch_toolkit.core._metering._operation import MeterOperation, OperationRequest
    from ai_arch_toolkit.core._redaction import Redactor
    from ai_arch_toolkit.core._response import Usage

__all__ = [
    "MeterScope",
    "Pricer",
    "RunConfig",
    "bind_meter",
    "current_meter",
    "current_span_id",
    "open_span",
]

_scope_var: ContextVar[MeterScope | None] = ContextVar("ai_arch_meter_scope", default=None)
_span_var: ContextVar[str | None] = ContextVar("ai_arch_meter_span", default=None)


class Pricer(Protocol):
    """Turns an operation's facts + observed usage into a :class:`Cost`. Lives in ``toolkit``."""

    def price(self, request: OperationRequest, usage: Usage) -> Cost: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class RunConfig:
    """How a run wires its meter. All optional — the bare default is measure-only."""

    controller: AdmissionController | None = None
    sinks: Sequence[UsageSink] = ()
    redactor: Redactor | None = None
    pricer: Pricer | None = None
    clock: Callable[[], float] | None = None


class MeterScope:
    """A run's meter: owns a :class:`MeterStore`, binds itself to the context, closes on exit."""

    def __init__(self, config: RunConfig | None = None) -> None:
        cfg = config or RunConfig()
        self._store = MeterStore(
            clock=cfg.clock or time.monotonic,
            sinks=cfg.sinks,
            redactor=cfg.redactor,
        )
        self._controller = cfg.controller
        self.pricer = cfg.pricer
        self._scope_token: object | None = None
        self._span_token: object | None = None

    @property
    def store(self) -> MeterStore:
        return self._store

    @property
    def run_span_id(self) -> str:
        return self._store.run_span_id

    @property
    def controller(self) -> AdmissionController | None:
        return self._controller

    def open(self, request: OperationRequest) -> MeterOperation:
        """Reserve an operation against this run's controller (None -> measure-only)."""
        return self._store.open(request, self._controller)

    def open_span(self, scope_type: str, parent_span_id: str | None = None) -> str:
        return self._store.open_span(scope_type, parent_span_id)

    def snapshot(self) -> MeterSnapshot:
        return self._store.snapshot()

    def for_span(self, span_id: str) -> MeterSnapshot:
        return self._store.for_span(span_id)

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> MeterScope:
        self._scope_token = _scope_var.set(self)
        self._span_token = _span_var.set(self._store.run_span_id)
        return self

    def __exit__(self, *exc: object) -> bool:
        try:
            self._store.close()
        finally:
            if self._span_token is not None:
                _span_var.reset(self._span_token)  # type: ignore[arg-type]
            if self._scope_token is not None:
                _scope_var.reset(self._scope_token)  # type: ignore[arg-type]
        return False

    async def __aenter__(self) -> MeterScope:
        return self.__enter__()

    async def __aexit__(self, *exc: object) -> bool:
        return self.__exit__(*exc)


def current_meter() -> MeterScope | None:
    """The scope bound to the current context, or ``None`` (unmetered)."""
    return _scope_var.get()


def current_span_id() -> str | None:
    """The span a charge site should attach to, or ``None`` when no scope is bound."""
    return _span_var.get()


@contextmanager
def bind_meter(
    scope: MeterScope | None, span_id: str | None = None
) -> Iterator[MeterScope | None]:
    """Bind a (possibly captured) scope to this context — for finalizers / worker threads."""
    span = span_id or (scope.run_span_id if scope is not None else None)
    scope_token = _scope_var.set(scope)
    span_token = _span_var.set(span)
    try:
        yield scope
    finally:
        _span_var.reset(span_token)
        _scope_var.reset(scope_token)


@contextmanager
def open_span(scope_type: str) -> Iterator[str | None]:
    """Open a child span under the current one and make it current. No-op when unmetered."""
    scope = current_meter()
    if scope is None:
        yield None
        return
    span_id = scope.open_span(scope_type, current_span_id())
    token = _span_var.set(span_id)
    try:
        yield span_id
    finally:
        _span_var.reset(token)
