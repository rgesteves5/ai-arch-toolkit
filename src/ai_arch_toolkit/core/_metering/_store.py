"""The single-writer meter store: operation lifecycle, per-dimension accounting, TOCTOU admission.

One writer, one ``threading.Lock`` (NOT ``asyncio`` — a stream finalizer may settle from
another OS thread). Every mutation is atomic under the lock; the controller's ``admit`` runs
*outside* the lock (no foreign code under it) and its verdict is re-validated under the lock
against the *current* state, so a stale admit can never overshoot a :class:`ResourceLimits` cap.

This increment is **run-level** (one flat meter). Per-span projections (``for_span``) and the
``UsageEvent`` audit stream are deferred to the next increment; the accounting they will project
from lives here already.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from ai_arch_toolkit.core._metering._admission import (
    AdmissionController,
    AdmissionDenied,
    MeterSnapshot,
    Reservation,
    ResourceLimits,
)
from ai_arch_toolkit.core._metering._cost import Cost
from ai_arch_toolkit.core._metering._money import Money
from ai_arch_toolkit.core._metering._operation import MeterOperation, OperationRequest
from ai_arch_toolkit.core._response import Usage

__all__ = ["MeterStore"]

logger = logging.getLogger(__name__)

# Terminal statuses recorded in the tombstone for idempotency (settle/fail/abort after end).
type _Terminal = tuple[str, int | None]  # (status, settle-payload hash | None)


@dataclass(slots=True)
class _LiveOp:
    """A PENDING or STARTED operation's accounting state (mutable, lock-guarded)."""

    op_id: str
    kind: str
    count: int
    reservation: Reservation
    parent_span_id: str
    started: bool = False


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
    """The authoritative meter. Counters are the single source of truth; everything else is a view.

    Accounting per dimension is split into **committed** (started physical attempts / settled
    actuals) and **outstanding** (reserved, not yet started/settled). Caps are always checked
    against committed + outstanding, never committed alone.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._lock = threading.Lock()
        self._clock = clock
        self._started_at = clock()
        self._next = 0
        self._ops: dict[str, _LiveOp] = {}
        self._tombstones: dict[str, _Terminal] = {}
        # committed
        self._c_llm = 0
        self._c_tool = 0
        self._c_input = 0
        self._c_output = 0
        self._c_cache_read = 0
        self._c_cache_write = 0
        self._c_cost = Money.zero()
        self._unknown = 0
        # outstanding
        self._o_llm = 0
        self._o_tool = 0
        self._o_input = 0
        self._o_output = 0
        self._o_cost = Money.zero()

    # ------------------------------------------------------------------ reads
    def snapshot(self) -> MeterSnapshot:
        """An atomic immutable read of the whole meter."""
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> MeterSnapshot:
        return MeterSnapshot(
            llm_calls=self._c_llm,
            tool_calls=self._c_tool,
            input_tokens=self._c_input,
            output_tokens=self._c_output,
            cache_read_tokens=self._c_cache_read,
            cache_write_tokens=self._c_cache_write,
            cost=self._c_cost,
            unknown_cost_count=self._unknown,
            out_llm_calls=self._o_llm,
            out_tool_calls=self._o_tool,
            out_input_tokens=self._o_input,
            out_output_tokens=self._o_output,
            out_cost=self._o_cost,
            elapsed_s=self._clock() - self._started_at,
        )

    # ------------------------------------------------------------------ open
    def open(
        self, request: OperationRequest, controller: AdmissionController | None
    ) -> MeterOperation:
        """Reserve an operation. Raises :class:`AdmissionDenied` if a cap is (or would be) hit.

        ``controller=None`` is measure-only: always admitted, no reservation, no caps. With a
        controller, ``admit`` runs on a snapshot *outside* the lock; its :class:`ResourceLimits`
        are then re-checked against the live state under the lock (TOCTOU close).
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
            denial = self._would_exceed_unlocked(limits, request, reservation)
            if denial is not None:
                raise denial
            self._next += 1
            op_id = f"op-{self._next}"
            self._ops[op_id] = _LiveOp(
                op_id=op_id,
                kind=request.kind,
                count=request.count,
                reservation=reservation,
                parent_span_id=request.parent_span_id,
            )
            if request.kind == "llm":
                self._o_llm += request.count
            elif request.kind == "tool":
                self._o_tool += request.count
            self._o_input += reservation.input_tokens
            self._o_output += reservation.output_tokens
            self._o_cost += reservation.cost
            return MeterOperation(self, op_id)

    def _would_exceed_unlocked(
        self,
        limits: ResourceLimits | None,
        request: OperationRequest,
        reservation: Reservation,
    ) -> AdmissionDenied | None:
        """Re-validate hard caps vs committed + outstanding + this op (caller holds the lock)."""
        if limits is None:
            return None
        add_llm = request.count if request.kind == "llm" else 0
        add_tool = request.count if request.kind == "tool" else 0
        committed_tokens = (
            self._c_input + self._c_output + self._c_cache_read + self._c_cache_write
        )
        rows = (
            ("llm_calls", limits.max_llm_calls, self._c_llm + self._o_llm, add_llm),
            ("tool_calls", limits.max_tool_calls, self._c_tool + self._o_tool, add_tool),
            (
                "input_tokens",
                limits.max_input_tokens,
                self._c_input + self._c_cache_read + self._c_cache_write + self._o_input,
                reservation.input_tokens,
            ),
            (
                "output_tokens",
                limits.max_output_tokens,
                self._c_output + self._o_output,
                reservation.output_tokens,
            ),
            (
                "total_tokens",
                limits.max_total_tokens,
                committed_tokens + self._o_input + self._o_output,
                reservation.input_tokens + reservation.output_tokens,
            ),
        )
        for dim, cap, current, add in rows:
            if cap is not None and current + add > cap:
                return AdmissionDenied(dimension=dim, limit=cap, current=current, attempted=add)
        if limits.max_cost is not None:
            current_cost = self._c_cost + self._o_cost
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
            if op.kind == "llm":
                self._o_llm -= op.count
                self._c_llm += op.count
            elif op.kind == "tool":
                self._o_tool -= op.count
                self._c_tool += op.count

    def settle(self, op_id: str, *, usage: Usage, cost: Cost) -> None:
        """STARTED -> SETTLED: release holds, record actual usage + cost. Idempotent on replay."""
        if cost.kind == "estimated":
            raise ValueError("settle() needs an actual cost (known|unknown), not an estimate")
        with self._lock:
            op = self._ops.get(op_id)
            if op is None:
                self._replay_terminal(op_id, "settled", _payload_hash(usage, cost))
                return
            if not op.started:
                raise ValueError(f"cannot settle operation {op_id} before mark_started()")
            self._release_holds(op)
            self._c_input += usage.input_tokens
            self._c_output += usage.output_tokens
            self._c_cache_read += usage.cache_read_tokens
            self._c_cache_write += usage.cache_write_tokens
            if cost.kind == "known":
                assert cost.amount is not None  # guaranteed by Cost.__post_init__
                self._c_cost += cost.amount
            else:  # unknown -> fail-closed bookkeeping, never silently $0
                self._unknown += 1
            self._terminalize(op, "settled", _payload_hash(usage, cost))

    def fail(self, op_id: str) -> None:
        """A started op errored: release holds, keep the count, charge cost-on-fail kind-aware.

        Never raises — runs in error-cleanup paths. A never-started op is fully released.
        """
        with self._lock:
            op = self._ops.get(op_id)
            if op is None:
                return  # terminal/unknown -> no-op (cleanup safety)
            self._release_holds(op)
            if op.started:
                # llm/custom may have incurred real cost we can't observe -> Unknown (fail-closed);
                # a failed tool is free.
                if op.kind in ("llm", "custom"):
                    self._unknown += 1
                self._terminalize(op, "failed", None)
            else:
                self._release_base_count(op)
                self._terminalize(op, "aborted", None)

    def abort(self, op_id: str) -> None:
        """PENDING -> ABORTED: fully release an operation that never started."""
        with self._lock:
            op = self._ops.get(op_id)
            if op is None:
                return  # terminal/unknown -> no-op
            if op.started:
                raise ValueError(f"cannot abort started operation {op_id}; use fail()")
            self._release_holds(op)
            self._release_base_count(op)
            self._terminalize(op, "aborted", None)

    def close(self) -> None:
        """End the scope: PENDING -> ABORTED, STARTED -> INCOMPLETE (count kept, cost Unknown)."""
        with self._lock:
            for op in list(self._ops.values()):
                self._release_holds(op)
                if op.started:
                    if op.kind in ("llm", "custom"):
                        self._unknown += 1
                    self._terminalize(op, "incomplete", None)
                else:
                    self._release_base_count(op)
                    self._terminalize(op, "aborted", None)

    # ------------------------------------------------------------- helpers (locked)
    def _release_holds(self, op: _LiveOp) -> None:
        self._o_input -= op.reservation.input_tokens
        self._o_output -= op.reservation.output_tokens
        self._o_cost -= op.reservation.cost

    def _release_base_count(self, op: _LiveOp) -> None:
        if op.kind == "llm":
            self._o_llm -= op.count
        elif op.kind == "tool":
            self._o_tool -= op.count

    def _terminalize(self, op: _LiveOp, status: str, payload: int | None) -> None:
        self._tombstones[op.op_id] = (status, payload)
        del self._ops[op.op_id]

    def _replay_terminal(self, op_id: str, action: str, payload: int | None) -> None:
        """Handle a transition on an already-terminal (or unknown) op: idempotent no-op or warn."""
        tomb = self._tombstones.get(op_id)
        if tomb is None:
            raise ValueError(f"unknown operation {op_id}")
        status, prev = tomb
        if status != action or prev != payload:
            logger.warning(
                "%s on already-%s operation %s ignored (keeping the first outcome)",
                action,
                status,
                op_id,
            )
