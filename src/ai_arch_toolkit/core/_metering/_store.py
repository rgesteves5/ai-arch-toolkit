"""The single-writer meter store: operation lifecycle, per-span accounting, TOCTOU admission.

One writer, one ``threading.Lock`` (NOT ``asyncio`` — a stream finalizer may settle from
another OS thread). Every mutation is atomic under the lock; the controller's ``admit`` runs
*outside* the lock (no foreign code under it) and its verdict is re-validated under the lock
against the *current* run-level state, so a stale admit can never overshoot a cap.

Accounting is a **span tree**: each operation's deltas are applied to its parent span and every
ancestor up to the run root. The run span is the global aggregate, so ``snapshot()`` ==
``for_span(run_span_id)``. ``ResourceLimits`` caps are run-level (re-validated at the root);
per-span aggregates exist for reporting and the flow's per-step ``Policy.max_cost``.

A terminal transition optionally builds a :class:`UsageEvent` (only when a sink is attached),
*under* the lock, and emits it to sinks *outside* the lock. ``metadata`` is run through a
:class:`Redactor` before it ever leaves the store (F1). The idempotency tombstone is LRU-bounded
(F5) so a long-running meter cannot leak memory.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace

from ai_arch_toolkit.core._metering._admission import (
    AdmissionController,
    AdmissionDenied,
    MeterSnapshot,
    Reservation,
    ResourceLimits,
)
from ai_arch_toolkit.core._metering._cost import Cost
from ai_arch_toolkit.core._metering._events import EventStatus, UsageEvent, UsageSink
from ai_arch_toolkit.core._metering._money import Money
from ai_arch_toolkit.core._metering._operation import MeterOperation, OperationRequest
from ai_arch_toolkit.core._redaction import Redactor
from ai_arch_toolkit.core._response import Usage

__all__ = ["MeterStore"]

logger = logging.getLogger(__name__)

_RUN_SPAN = "run"
_TOMBSTONE_MAX = (
    50_000  # LRU bound: a long run can terminate millions of ops; keep the recent tail
)
_ZERO_COST = Cost.known(Money.zero())
_NO_USAGE = Usage()

# Terminal record kept for idempotency (settle/fail/abort after an op already ended).
type _Terminal = tuple[str, int | None]  # (status, settle-payload hash | None)


@dataclass(slots=True)
class _Counters:
    """One span's per-dimension committed (started/settled) vs outstanding (reserved) tallies."""

    c_llm: int = 0
    c_tool: int = 0
    c_input: int = 0
    c_output: int = 0
    c_cache_read: int = 0
    c_cache_write: int = 0
    c_cost: Money = field(default_factory=Money.zero)
    unknown: int = 0
    o_llm: int = 0
    o_tool: int = 0
    o_input: int = 0
    o_output: int = 0
    o_cost: Money = field(default_factory=Money.zero)


@dataclass(slots=True)
class _Span:
    span_id: str
    parent: str | None
    scope_type: str
    started_at: float
    counters: _Counters = field(default_factory=_Counters)


@dataclass(slots=True)
class _LiveOp:
    """A PENDING or STARTED operation's accounting + audit state (mutable, lock-guarded)."""

    op_id: str
    kind: str
    count: int
    reservation: Reservation
    parent_span_id: str
    model: str | None = None
    provider: str | None = None
    mode: str | None = None
    metadata: Mapping[str, str | int | float | bool] = field(default_factory=dict)
    started: bool = False


# --- counter mutations (pure; applied to every ancestor span under the lock) ---


def _reserve(c: _Counters, op: _LiveOp) -> None:
    if op.kind == "llm":
        c.o_llm += op.count
    elif op.kind == "tool":
        c.o_tool += op.count
    c.o_input += op.reservation.input_tokens
    c.o_output += op.reservation.output_tokens
    c.o_cost += op.reservation.cost


def _start(c: _Counters, op: _LiveOp) -> None:
    if op.kind == "llm":
        c.o_llm -= op.count
        c.c_llm += op.count
    elif op.kind == "tool":
        c.o_tool -= op.count
        c.c_tool += op.count


def _release_holds(c: _Counters, op: _LiveOp) -> None:
    c.o_input -= op.reservation.input_tokens
    c.o_output -= op.reservation.output_tokens
    c.o_cost -= op.reservation.cost


def _settle(c: _Counters, op: _LiveOp, usage: Usage, cost: Cost) -> None:
    _release_holds(c, op)
    c.c_input += usage.input_tokens
    c.c_output += usage.output_tokens
    c.c_cache_read += usage.cache_read_tokens
    c.c_cache_write += usage.cache_write_tokens
    if cost.kind == "known":
        assert cost.amount is not None  # guaranteed by Cost.__post_init__
        c.c_cost += cost.amount
    else:  # unknown -> fail-closed bookkeeping, never silently $0
        c.unknown += 1


def _fail_started(c: _Counters, op: _LiveOp) -> None:
    # count stays committed; llm/custom may have unobservable cost; a failed tool is free
    _release_holds(c, op)
    if op.kind in ("llm", "custom"):
        c.unknown += 1


def _abort(c: _Counters, op: _LiveOp) -> None:
    _release_holds(c, op)
    if op.kind == "llm":
        c.o_llm -= op.count
    elif op.kind == "tool":
        c.o_tool -= op.count


def _fail_cost(kind: str) -> Cost:
    """Cost ascribed to a non-settling op: Unknown for llm/custom (fail-closed), free for tool."""
    if kind in ("llm", "custom"):
        return Cost.unknown("operation did not settle")
    return _ZERO_COST


def _payload_hash(usage: Usage, cost: Cost) -> int:
    """A stable hash of a settle payload, for double-settle detection."""
    return hash(
        (
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_read_tokens,
            usage.cache_write_tokens,
            cost.kind,
            cost.amount.pico if cost.amount is not None else None,
            cost.reason,
        )
    )


class MeterStore:
    """The authoritative meter. Counters are the source of truth; everything else is a view."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        sinks: Sequence[UsageSink] = (),
        redactor: Redactor | None = None,
        sink_error_policy: str = "log",
    ) -> None:
        self._lock = threading.Lock()
        self._clock = clock
        self._started_at = clock()
        self._sinks = tuple(sinks)
        self._redactor = redactor or Redactor()
        self._sink_error_policy = sink_error_policy
        self._seq = 0
        self._next_op = 0
        self._next_span = 0
        self._ops: dict[str, _LiveOp] = {}
        self._tombstones: dict[str, _Terminal] = {}
        self._spans: dict[str, _Span] = {
            _RUN_SPAN: _Span(_RUN_SPAN, None, "run", self._started_at)
        }

    @property
    def run_span_id(self) -> str:
        """The root span — its aggregate is the whole-run meter."""
        return _RUN_SPAN

    # ------------------------------------------------------------------ spans
    def open_span(self, scope_type: str, parent_span_id: str | None = None) -> str:
        """Register a child span (e.g. a flow step / tool). Returns its id."""
        with self._lock:
            parent = parent_span_id or _RUN_SPAN
            if parent not in self._spans:
                raise ValueError(f"unknown parent span {parent}")
            self._next_span += 1
            span_id = f"span-{self._next_span}"
            self._spans[span_id] = _Span(span_id, parent, scope_type, self._clock())
            return span_id

    # ------------------------------------------------------------------ reads
    def snapshot(self) -> MeterSnapshot:
        """An atomic immutable read of the whole-run meter (the root span's aggregate)."""
        with self._lock:
            return self._snapshot_of(self._spans[_RUN_SPAN])

    def for_span(self, span_id: str) -> MeterSnapshot:
        """The aggregate for one span's subtree (it + every descendant operation)."""
        with self._lock:
            span = self._spans.get(span_id)
            if span is None:
                raise ValueError(f"unknown span {span_id}")
            return self._snapshot_of(span)

    def close_span(self, span_id: str) -> None:
        """Reclaim a finished span node so ``_spans`` can't grow O(iterations) in cyclic/LATS runs.

        Its counters already live in every ancestor (``_apply`` walks to the root on each op
        transition), so removing the node loses no accounting. Refuses to drop the run root, an
        unknown span, or any span with a LIVE op still in its subtree — otherwise that op's later
        settle/fail would ``_apply`` through a missing ancestor.
        """
        if span_id == _RUN_SPAN:
            return
        with self._lock:
            if span_id not in self._spans:
                return
            for op in self._ops.values():
                sid: str | None = op.parent_span_id
                while sid is not None:
                    if sid == span_id:
                        return  # a live op sits under this span — keep it reachable
                    parent = self._spans.get(sid)
                    sid = parent.parent if parent is not None else None
            del self._spans[span_id]

    def _snapshot_of(self, span: _Span) -> MeterSnapshot:
        c = span.counters
        return MeterSnapshot(
            llm_calls=c.c_llm,
            tool_calls=c.c_tool,
            input_tokens=c.c_input,
            output_tokens=c.c_output,
            cache_read_tokens=c.c_cache_read,
            cache_write_tokens=c.c_cache_write,
            cost=c.c_cost,
            unknown_cost_count=c.unknown,
            out_llm_calls=c.o_llm,
            out_tool_calls=c.o_tool,
            out_input_tokens=c.o_input,
            out_output_tokens=c.o_output,
            out_cost=c.o_cost,
            elapsed_s=self._clock() - span.started_at,
        )

    def _apply(self, parent_span_id: str, mutate: Callable[[_Counters], None]) -> None:
        """Apply a counter delta to the op's span and every ancestor up to the run root."""
        sid: str | None = parent_span_id
        while sid is not None:
            span = self._spans[sid]
            mutate(span.counters)
            sid = span.parent

    # ------------------------------------------------------------------ open
    def open(
        self, request: OperationRequest, controller: AdmissionController | None
    ) -> MeterOperation:
        """Reserve an operation. Raises :class:`AdmissionDenied` if a cap is (or would be) hit.

        ``controller=None`` is measure-only: always admitted, no reservation, no caps. With a
        controller, ``admit`` runs on a snapshot *outside* the lock; its :class:`ResourceLimits`
        are then re-checked against the live run-level state under the lock (TOCTOU close).
        """
        if controller is None:
            reservation = Reservation()
            limits: ResourceLimits | None = None
        else:
            decision = controller.admit(self.snapshot(), request)
            if not decision.admitted:
                raise decision.denial or AdmissionDenied()
            reservation = decision.reservation
            limits = decision.limits

        with self._lock:
            if request.parent_span_id not in self._spans:
                raise ValueError(f"unknown parent span {request.parent_span_id}")
            denial = self._would_exceed_unlocked(limits, request, reservation)
            if denial is not None:
                raise denial
            self._next_op += 1
            op_id = f"op-{self._next_op}"
            op = _LiveOp(
                op_id=op_id,
                kind=request.kind,
                count=request.count,
                reservation=reservation,
                parent_span_id=request.parent_span_id,
                model=request.model,
                provider=request.provider,
                mode=request.mode,
                metadata=request.metadata,
            )
            self._ops[op_id] = op
            self._apply(op.parent_span_id, lambda c: _reserve(c, op))
            return MeterOperation(self, op_id)

    def _would_exceed_unlocked(
        self,
        limits: ResourceLimits | None,
        request: OperationRequest,
        reservation: Reservation,
    ) -> AdmissionDenied | None:
        """Re-validate hard caps vs run-level committed + outstanding + this op (lock held)."""
        if limits is None:
            return None
        c = self._spans[_RUN_SPAN].counters
        add_llm = request.count if request.kind == "llm" else 0
        add_tool = request.count if request.kind == "tool" else 0
        committed_tokens = c.c_input + c.c_output + c.c_cache_read + c.c_cache_write
        rows = (
            ("llm_calls", limits.max_llm_calls, c.c_llm + c.o_llm, add_llm),
            ("tool_calls", limits.max_tool_calls, c.c_tool + c.o_tool, add_tool),
            (
                "input_tokens",
                limits.max_input_tokens,
                c.c_input + c.c_cache_read + c.c_cache_write + c.o_input,
                reservation.input_tokens,
            ),
            (
                "output_tokens",
                limits.max_output_tokens,
                c.c_output + c.o_output,
                reservation.output_tokens,
            ),
            (
                "total_tokens",
                limits.max_total_tokens,
                committed_tokens + c.o_input + c.o_output,
                reservation.input_tokens + reservation.output_tokens,
            ),
        )
        for dim, cap, current, add in rows:
            if cap is not None and current + add > cap:
                return AdmissionDenied(dimension=dim, limit=cap, current=current, attempted=add)
        if limits.max_cost is not None:
            current_cost = c.c_cost + c.o_cost
            if current_cost + reservation.cost > limits.max_cost:
                return AdmissionDenied(
                    dimension="cost",
                    limit=limits.max_cost.to_float(),
                    current=current_cost.to_float(),
                    attempted=reservation.cost.to_float(),
                )
        if limits.max_wall_s is not None:
            elapsed = self._clock() - self._started_at
            if elapsed > limits.max_wall_s:
                return AdmissionDenied(
                    dimension="wall_s", limit=limits.max_wall_s, current=elapsed, attempted=0.0
                )
        return None

    # ------------------------------------------------------------- transitions
    def mark_started(self, op_id: str) -> None:
        """PENDING -> STARTED: move the base call count from outstanding to committed."""
        with self._lock:
            op = self._ops.get(op_id)
            if op is None or op.started:
                return  # terminal/unknown, or already started -> idempotent no-op
            op.started = True
            self._apply(op.parent_span_id, lambda c: _start(c, op))

    def settle(self, op_id: str, *, usage: Usage, cost: Cost) -> None:
        """STARTED -> SETTLED: release holds, record actual usage + cost. Idempotent on replay."""
        if cost.kind == "estimated":
            raise ValueError("settle() needs an actual cost (known|unknown), not an estimate")
        event = None
        with self._lock:
            op = self._ops.get(op_id)
            if op is None:
                self._replay_terminal(op_id, "settled", _payload_hash(usage, cost))
                return
            if not op.started:
                raise ValueError(f"cannot settle operation {op_id} before mark_started()")
            self._apply(op.parent_span_id, lambda c: _settle(c, op, usage, cost))
            event = self._make_event(op, "settled", usage, cost)
            self._terminalize(op, "settled", _payload_hash(usage, cost))
        self._dispatch(event)

    def fail(self, op_id: str) -> None:
        """A started op errored: release holds, keep the count, charge cost-on-fail kind-aware.

        Never raises — runs in error-cleanup paths. A never-started op is fully released.
        """
        event = None
        with self._lock:
            op = self._ops.get(op_id)
            if op is None:
                return  # terminal/unknown -> no-op (cleanup safety)
            if op.started:
                self._apply(op.parent_span_id, lambda c: _fail_started(c, op))
                event = self._make_event(op, "failed", _NO_USAGE, _fail_cost(op.kind))
                self._terminalize(op, "failed", None)
            else:
                self._apply(op.parent_span_id, lambda c: _abort(c, op))
                event = self._make_event(op, "aborted", _NO_USAGE, _ZERO_COST)
                self._terminalize(op, "aborted", None)
        self._dispatch(event)

    def abort(self, op_id: str) -> None:
        """PENDING -> ABORTED: fully release an operation that never started."""
        event = None
        with self._lock:
            op = self._ops.get(op_id)
            if op is None:
                return  # terminal/unknown -> no-op
            if op.started:
                raise ValueError(f"cannot abort started operation {op_id}; use fail()")
            self._apply(op.parent_span_id, lambda c: _abort(c, op))
            event = self._make_event(op, "aborted", _NO_USAGE, _ZERO_COST)
            self._terminalize(op, "aborted", None)
        self._dispatch(event)

    def close(self) -> None:
        """End the scope: PENDING -> ABORTED, STARTED -> INCOMPLETE (count kept, cost Unknown)."""
        events: list[UsageEvent] = []
        with self._lock:
            for op in list(self._ops.values()):
                if op.started:
                    self._apply(op.parent_span_id, lambda c, op=op: _fail_started(c, op))
                    event = self._make_event(op, "incomplete", _NO_USAGE, _fail_cost(op.kind))
                    self._terminalize(op, "incomplete", None)
                else:
                    self._apply(op.parent_span_id, lambda c, op=op: _abort(c, op))
                    event = self._make_event(op, "aborted", _NO_USAGE, _ZERO_COST)
                    self._terminalize(op, "aborted", None)
                if event is not None:
                    events.append(event)
        for event in events:
            self._dispatch(event)

    # ------------------------------------------------------------- helpers (locked)
    def _make_event(
        self, op: _LiveOp, status: EventStatus, usage: Usage, cost: Cost
    ) -> UsageEvent | None:
        """Build the audit event (only when a sink is attached). Caller holds the lock."""
        if not self._sinks:
            return None
        self._seq += 1
        return UsageEvent(
            seq=self._seq,
            op_id=op.op_id,
            span_id=op.parent_span_id,
            kind=op.kind,  # type: ignore[arg-type]  # op.kind is one of the literal kinds
            status=status,
            usage=usage,
            cost=cost,
            model=op.model,
            provider=op.provider,
            mode=op.mode,
            at_s=self._clock() - self._started_at,
            metadata=op.metadata,  # raw; _dispatch redacts OUTSIDE the lock (redactor is foreign)
        )

    def _terminalize(self, op: _LiveOp, status: str, payload: int | None) -> None:
        self._tombstones[op.op_id] = (status, payload)
        del self._ops[op.op_id]
        if len(self._tombstones) > _TOMBSTONE_MAX:
            # dicts preserve insertion order -> evict the oldest tombstone (F5: bounded memory)
            del self._tombstones[next(iter(self._tombstones))]

    def _was_ever_issued(self, op_id: str) -> bool:
        """True if this op_id was handed out by open() at some point (id counter is monotonic)."""
        if not op_id.startswith("op-"):
            return False
        try:
            return 1 <= int(op_id[3:]) <= self._next_op
        except ValueError:
            return False

    def _replay_terminal(self, op_id: str, action: str, payload: int | None) -> None:
        """Handle a transition on an already-terminal (or unknown) op: idempotent no-op or warn."""
        tomb = self._tombstones.get(op_id)
        if tomb is None:
            # Not live, not tombstoned. If it was ever issued, its tombstone was LRU-evicted — a
            # benign late replay (a finalizer settling after close must not raise). Otherwise the
            # op_id was never opened, which is a real programming error.
            if not self._was_ever_issued(op_id):
                raise ValueError(f"unknown operation {op_id}")
            return
        status, prev = tomb
        if status != action or prev != payload:
            logger.warning(
                "%s on already-%s operation %s ignored (keeping the first outcome)",
                action,
                status,
                op_id,
            )

    def _dispatch(self, event: UsageEvent | None) -> None:
        """Redact + emit to sinks OUTSIDE the lock. A foreign redactor/sink can't stall the meter
        OR break the (already-settled, already-paid) call that triggered the event."""
        if event is None:
            return
        try:
            metadata = self._redactor.redact(dict(event.metadata))
        except Exception:
            logger.exception("usage redactor %r raised; dropping event metadata", self._redactor)
            metadata = {}
        event = replace(event, metadata=metadata)
        for sink in self._sinks:
            try:
                sink.emit(event)
            except Exception:
                if self._sink_error_policy == "raise":
                    raise
                logger.exception("usage sink %r raised emitting %s", sink, event.op_id)
