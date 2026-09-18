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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

from ai_arch_toolkit.core._exceptions import Delivery
from ai_arch_toolkit.core._metering._admission import (
    AdmissionController,
    AdmissionDenied,
    FailureBoundController,
    MeterSnapshot,
    Reservation,
    ResourceLimits,
    limit_denial,
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
type _Terminal = tuple[str, tuple[object, ...] | None]  # (status, settle-payload key | None)


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
    uncertain: Money = field(default_factory=Money.zero)
    uncertain_count: int = 0
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


@dataclass(frozen=True, slots=True, kw_only=True)
class _LiveOp:
    """One admitted operation, retaining its immutable facts and optional failure sizing."""

    op_id: str
    request: OperationRequest
    reservation: Reservation
    controller: AdmissionController | None
    failure_request: Callable[[], OperationRequest] | None = None
    started: bool = False


# --- counter mutations (pure; applied to every ancestor span under the lock) ---


def _reserve(c: _Counters, op: _LiveOp) -> None:
    if op.request.kind == "llm":
        c.o_llm += op.request.count
    elif op.request.kind == "tool":
        c.o_tool += op.request.count
    c.o_input += op.reservation.input_tokens
    c.o_output += op.reservation.output_tokens
    c.o_cost += op.reservation.cost


def _start(c: _Counters, op: _LiveOp) -> None:
    if op.request.kind == "llm":
        c.o_llm -= op.request.count
        c.c_llm += op.request.count
    elif op.request.kind == "tool":
        c.o_tool -= op.request.count
        c.c_tool += op.request.count


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
    elif cost.at_most is None:
        c.unknown += 1
    else:
        c.uncertain += cost.at_most
        c.uncertain_count += 1


def _abort(c: _Counters, op: _LiveOp) -> None:
    _release_holds(c, op)
    if op.request.kind == "llm":
        c.o_llm -= op.request.count
    elif op.request.kind == "tool":
        c.o_tool -= op.request.count


def _fail_cost(op: _LiveOp, delivery: Delivery) -> Cost:
    """Price a failure outside the store lock; foreign estimation must never break cleanup."""
    if delivery in ("not_sent", "unbilled") or op.request.kind == "tool":
        return _ZERO_COST
    bound = None
    if isinstance(op.controller, FailureBoundController):
        try:
            request = op.request
            if request.content_size_hint is None and op.failure_request is not None:
                request = op.failure_request()
            bound = op.controller.failure_bound(request, op.reservation)
        except Exception:
            logger.exception("failure bound could not be estimated")
    return Cost.unknown("operation did not settle", at_most=bound)


def _payload_key(usage: Usage, cost: Cost) -> tuple[object, ...]:
    """The identifying tuple of a settle payload, for double-settle detection.

    Stored (not hashed) so re-settle comparison is EXACT — a hash collision can't suppress the
    ``keeping the first outcome`` warning. Bounded by the tombstone LRU.
    """
    return (
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_read_tokens,
        usage.cache_write_tokens,
        cost.kind,
        cost.amount.pico if cost.amount is not None else None,
        cost.reason,
        cost.at_most,
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

    def has_live_ops(self, span_id: str) -> bool:
        """True if a PENDING or STARTED op sits in this span's subtree — its cost isn't known yet.

        A per-step cost cap uses this to fail closed when a step returns while a metered op it
        opened is still in flight (e.g. a stream opened but never drained): the committed span cost
        misses that op, so the cap must treat the span's spend as indeterminate rather than as $0.
        """
        with self._lock:
            return self._has_live_op_under(span_id)

    def _has_live_op_under(self, span_id: str) -> bool:
        """Walk every live op to the root; True if one is anchored in ``span_id``'s subtree.

        Caller holds ``self._lock``. O(live ops * span depth) — both tiny in practice.
        """
        for op in self._ops.values():
            sid: str | None = op.request.parent_span_id
            while sid is not None:
                if sid == span_id:
                    return True
                parent = self._spans.get(sid)
                sid = parent.parent if parent is not None else None
        return False

    def close_span(self, span_id: str) -> None:
        """Reclaim a finished span node so ``_spans`` can't grow O(iterations) in cyclic/LATS runs.

        Its counters already live in every ancestor (``_apply`` walks to the root on each op
        transition), so removing the node loses no accounting. Refuses to drop the run root, an
        unknown span, any span with a LIVE op still in its subtree, or any span that still has a
        CHILD span — otherwise a later ``_apply`` would climb through the missing node (KeyError).
        """
        if span_id == _RUN_SPAN:
            return
        with self._lock:
            if span_id not in self._spans:
                return
            if self._has_live_op_under(span_id):
                return  # a live op sits under this span — keep it reachable
            # Keep it while any child span references it as parent: dropping it would orphan the
            # child, and the child's next op would _apply through the vanished parent. (The
            # open_span CM closes children first, so this only guards non-LIFO/public-API use.)
            if any(span.parent == span_id for span in self._spans.values()):
                return
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
            uncertain_cost=c.uncertain,
            uncertain_cost_count=c.uncertain_count,
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
        self,
        request: OperationRequest,
        controller: AdmissionController | None,
        *,
        failure_request: Callable[[], OperationRequest] | None = None,
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
                request=request,
                reservation=reservation,
                controller=controller,
                failure_request=failure_request,
            )
            self._ops[op_id] = op
            self._apply(op.request.parent_span_id, lambda c: _reserve(c, op))
            return MeterOperation(self, op_id)

    def _would_exceed_unlocked(
        self,
        limits: ResourceLimits | None,
        request: OperationRequest,
        reservation: Reservation,
    ) -> AdmissionDenied | None:
        """Re-validate hard caps vs run-level committed + outstanding + this op (lock held)."""
        return limit_denial(
            self._snapshot_of(self._spans[_RUN_SPAN]), limits, request, reservation
        )

    # ------------------------------------------------------------- transitions
    def mark_started(self, op_id: str) -> None:
        """PENDING -> STARTED: move the base call count from outstanding to committed."""
        with self._lock:
            op = self._ops.get(op_id)
            if op is None or op.started:
                return  # terminal/unknown, or already started -> idempotent no-op
            op = replace(op, started=True)
            self._ops[op_id] = op
            self._apply(op.request.parent_span_id, lambda c: _start(c, op))

    def settle(self, op_id: str, *, usage: Usage, cost: Cost) -> None:
        """STARTED -> SETTLED: release holds, record actual usage + cost. Idempotent on replay."""
        if cost.kind == "estimated":
            raise ValueError("settle() needs an actual cost (known|unknown), not an estimate")
        event = None
        with self._lock:
            op = self._ops.get(op_id)
            if op is None:
                self._replay_terminal(op_id, "settled", _payload_key(usage, cost))
                return
            if not op.started:
                raise ValueError(f"cannot settle operation {op_id} before mark_started()")
            self._apply(op.request.parent_span_id, lambda c: _settle(c, op, usage, cost))
            event = self._make_event(op, "settled", usage, cost)
            self._terminalize(op, "settled", _payload_key(usage, cost))
        self._dispatch(event)

    def fail(self, op_id: str, disposition: Delivery) -> None:
        """Finalize failed work with a classified cost; pending work is aborted."""
        self._finish_failure(op_id, "failed", disposition)

    def _finish_failure(self, op_id: str, status: EventStatus, delivery: Delivery) -> None:
        while True:
            with self._lock:
                op = self._ops.get(op_id)
            if op is None:
                return
            cost = _fail_cost(op, delivery) if op.started else _ZERO_COST
            with self._lock:
                if self._ops.get(op_id) is not op:
                    continue
                if op.started:
                    self._apply(
                        op.request.parent_span_id,
                        lambda c, op=op, cost=cost: _settle(c, op, _NO_USAGE, cost),
                    )
                else:
                    status = "aborted"
                    delivery = "not_sent"
                    self._apply(op.request.parent_span_id, lambda c, op=op: _abort(c, op))
                event = self._make_event(op, status, _NO_USAGE, cost, delivery)
                self._terminalize(op, status, None)
                break
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
            self._apply(op.request.parent_span_id, lambda c, op=op: _abort(c, op))
            event = self._make_event(op, "aborted", _NO_USAGE, _ZERO_COST)
            self._terminalize(op, "aborted", None)
        self._dispatch(event)

    def close(self) -> None:
        """Finalize every live operation; bounds are estimated outside the store lock."""
        with self._lock:
            op_ids = tuple(self._ops)
        for op_id in op_ids:
            self._finish_failure(op_id, "incomplete", "indeterminate")

    # ------------------------------------------------------------- helpers (locked)
    def _make_event(
        self,
        op: _LiveOp,
        status: EventStatus,
        usage: Usage,
        cost: Cost,
        delivery: Delivery | None = None,
    ) -> UsageEvent | None:
        """Build the audit event (only when a sink is attached). Caller holds the lock."""
        if not self._sinks:
            return None
        self._seq += 1
        return UsageEvent(
            seq=self._seq,
            op_id=op.op_id,
            span_id=op.request.parent_span_id,
            kind=op.request.kind,
            status=status,
            delivery=delivery,
            usage=usage,
            cost=cost,
            model=op.request.model,
            provider=op.request.provider,
            mode=op.request.mode,
            at_s=self._clock() - self._started_at,
            metadata=op.request.metadata,  # _dispatch redacts outside the lock
        )

    def _terminalize(self, op: _LiveOp, status: str, payload: tuple[object, ...] | None) -> None:
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

    def _replay_terminal(
        self, op_id: str, action: str, payload: tuple[object, ...] | None
    ) -> None:
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
