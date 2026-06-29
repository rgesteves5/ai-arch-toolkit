"""MeterStore lifecycle + accounting oracle, incl. TOCTOU re-validation under real threads."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import pytest

from ai_arch_toolkit.core._metering._admission import (
    AdmissionDecision,
    AdmissionDenied,
    MeterSnapshot,
    Reservation,
    ResourceLimits,
)
from ai_arch_toolkit.core._metering._cost import Cost
from ai_arch_toolkit.core._metering._money import Money
from ai_arch_toolkit.core._metering._operation import OperationRequest
from ai_arch_toolkit.core._metering._store import MeterStore
from ai_arch_toolkit.core._response import Usage

# --------------------------------------------------------------------- helpers


def llm(count: int = 1) -> OperationRequest:
    return OperationRequest(kind="llm", parent_span_id="run", count=count)


def tool(count: int = 1) -> OperationRequest:
    return OperationRequest(kind="tool", parent_span_id="run", count=count)


def usd(x: float) -> Money:
    return Money.from_usd(x)


@dataclass
class BlindAllow:
    """Admits everything but returns limits — the store's re-validate is the only enforcer."""

    limits: ResourceLimits | None = None
    reservation: Reservation = field(default_factory=Reservation)

    def admit(self, snapshot: MeterSnapshot, request: OperationRequest) -> AdmissionDecision:
        return AdmissionDecision.allow(self.reservation, self.limits)


class Deny:
    def admit(self, snapshot: MeterSnapshot, request: OperationRequest) -> AdmissionDecision:
        return AdmissionDecision.deny(AdmissionDenied(dimension="test"))


# --------------------------------------------------------------- measure-only


def test_open_reserves_base_count():
    store = MeterStore()
    store.open(llm(), controller=None)
    snap = store.snapshot()
    assert snap.out_llm_calls == 1 and snap.llm_calls == 0


def test_mark_started_transfers_outstanding_to_committed():
    store = MeterStore()
    op = store.open(llm(), controller=None)
    op.mark_started()
    snap = store.snapshot()
    assert snap.llm_calls == 1 and snap.out_llm_calls == 0


def test_settle_records_actuals_and_releases_holds():
    store = MeterStore()
    op = store.open(llm(), controller=None)
    op.mark_started()
    op.settle(usage=Usage(input_tokens=10, output_tokens=5), cost=Cost.known(usd(0.001)))
    snap = store.snapshot()
    assert snap.input_tokens == 10 and snap.output_tokens == 5
    assert snap.cost == usd(0.001) and snap.unknown_cost_count == 0
    assert snap.out_llm_calls == 0 and snap.out_cost == Money.zero()


def test_settle_unknown_cost_is_counted_not_zeroed():
    store = MeterStore()
    op = store.open(llm(), controller=None)
    op.mark_started()
    op.settle(usage=Usage(input_tokens=1), cost=Cost.unknown("no pricing"))
    snap = store.snapshot()
    assert snap.unknown_cost_count == 1 and snap.cost == Money.zero()


def test_settle_rejects_an_estimate():
    store = MeterStore()
    op = store.open(llm(), controller=None)
    op.mark_started()
    with pytest.raises(ValueError, match="actual cost"):
        op.settle(usage=Usage(), cost=Cost.estimated(usd(0.01)))


def test_settle_before_start_raises():
    store = MeterStore()
    op = store.open(llm(), controller=None)
    with pytest.raises(ValueError, match="before mark_started"):
        op.settle(usage=Usage(), cost=Cost.known(usd(0)))


# ------------------------------------------------------------------- fail/abort


def test_fail_keeps_the_count_and_charges_unknown_for_llm():
    store = MeterStore()
    op = store.open(llm(), controller=None)
    op.mark_started()
    op.fail()
    snap = store.snapshot()
    assert snap.llm_calls == 1 and snap.unknown_cost_count == 1
    assert snap.out_llm_calls == 0 and snap.out_cost == Money.zero()


def test_failed_tool_is_free():
    store = MeterStore()
    op = store.open(tool(), controller=None)
    op.mark_started()
    op.fail()
    snap = store.snapshot()
    assert snap.tool_calls == 1 and snap.unknown_cost_count == 0


def test_fail_before_start_releases_the_count():
    store = MeterStore()
    op = store.open(llm(), controller=None)
    op.fail()
    snap = store.snapshot()
    assert snap.llm_calls == 0 and snap.out_llm_calls == 0


def test_abort_pending_fully_releases():
    store = MeterStore()
    op = store.open(llm(), controller=None)
    op.abort()
    snap = store.snapshot()
    assert snap.out_llm_calls == 0 and snap.llm_calls == 0


def test_abort_started_raises():
    store = MeterStore()
    op = store.open(llm(), controller=None)
    op.mark_started()
    with pytest.raises(ValueError, match="use fail"):
        op.abort()


# ----------------------------------------------------------------- idempotency


def test_double_settle_same_payload_is_a_noop():
    store = MeterStore()
    op = store.open(llm(), controller=None)
    op.mark_started()
    payload = dict(usage=Usage(input_tokens=10), cost=Cost.known(usd(0.002)))
    op.settle(**payload)
    op.settle(**payload)  # replay -> no double count, no raise
    snap = store.snapshot()
    assert snap.input_tokens == 10 and snap.cost == usd(0.002)


def test_double_settle_different_payload_keeps_the_first():
    store = MeterStore()
    op = store.open(llm(), controller=None)
    op.mark_started()
    op.settle(usage=Usage(input_tokens=10), cost=Cost.known(usd(0.002)))
    op.settle(usage=Usage(input_tokens=999), cost=Cost.known(usd(9)))  # ignored
    snap = store.snapshot()
    assert snap.input_tokens == 10 and snap.cost == usd(0.002)


def test_settle_unknown_op_raises():
    store = MeterStore()
    with pytest.raises(ValueError, match="unknown operation"):
        store.settle("op-404", usage=Usage(), cost=Cost.known(usd(0)))


def test_fail_unknown_op_is_silent():
    MeterStore().fail("op-404")  # cleanup safety: never raises


# ---------------------------------------------------------------------- close


def test_close_aborts_pending_and_incompletes_started():
    store = MeterStore()
    store.open(llm(), controller=None)  # pending
    started = store.open(llm(), controller=None)
    started.mark_started()
    store.close()
    snap = store.snapshot()
    assert snap.llm_calls == 1  # the started one kept its count
    assert snap.unknown_cost_count == 1  # incomplete llm -> Unknown
    assert snap.out_llm_calls == 0  # the pending one was released


# ---------------------------------------------------------- admission / TOCTOU


def test_controller_denial_changes_no_state():
    store = MeterStore()
    with pytest.raises(AdmissionDenied):
        store.open(llm(), controller=Deny())
    snap = store.snapshot()
    assert snap.out_llm_calls == 0 and snap.llm_calls == 0


def test_store_revalidates_call_cap_even_when_controller_is_blind():
    store = MeterStore()
    ctrl = BlindAllow(limits=ResourceLimits(max_llm_calls=1))
    store.open(llm(), controller=ctrl)  # ok
    with pytest.raises(AdmissionDenied) as ei:
        store.open(llm(), controller=ctrl)
    assert ei.value.dimension == "llm_calls"
    assert store.snapshot().out_llm_calls == 1  # second never reserved


def test_store_revalidates_cost_cap():
    store = MeterStore()
    ctrl = BlindAllow(
        limits=ResourceLimits(max_cost=usd(0.05)),
        reservation=Reservation(cost=usd(0.04)),
    )
    store.open(llm(), controller=ctrl)  # out_cost = 0.04
    with pytest.raises(AdmissionDenied) as ei:
        store.open(llm(), controller=ctrl)  # 0.04 + 0.04 > 0.05
    assert ei.value.dimension == "cost"


def test_reservation_is_applied_then_released_on_settle():
    store = MeterStore()
    ctrl = BlindAllow(reservation=Reservation(input_tokens=100, cost=usd(0.02)))
    op = store.open(llm(), controller=ctrl)
    mid = store.snapshot()
    assert mid.out_input_tokens == 100 and mid.out_cost == usd(0.02)
    op.mark_started()
    op.settle(usage=Usage(input_tokens=42), cost=Cost.known(usd(0.011)))
    snap = store.snapshot()
    assert snap.out_input_tokens == 0 and snap.out_cost == Money.zero()
    assert snap.input_tokens == 42 and snap.cost == usd(0.011)


def test_full_lifecycle_drains_outstanding_to_zero():
    store = MeterStore()
    for _ in range(5):
        op = store.open(llm(), controller=None)
        op.mark_started()
        op.settle(usage=Usage(input_tokens=3, output_tokens=2), cost=Cost.known(usd(0.001)))
    snap = store.snapshot()
    assert snap.llm_calls == 5 and snap.input_tokens == 15 and snap.output_tokens == 10
    assert snap.out_llm_calls == 0 and snap.out_input_tokens == 0 and snap.out_cost == Money.zero()


# ------------------------------------------------------------- concurrency


def test_concurrent_opens_never_overshoot_the_cap():
    cap = 5
    n = 40
    store = MeterStore()
    ctrl = BlindAllow(limits=ResourceLimits(max_llm_calls=cap))
    barrier = threading.Barrier(n)
    admitted = 0
    denied = 0
    guard = threading.Lock()

    def worker() -> None:
        nonlocal admitted, denied
        barrier.wait()  # maximize contention on the store lock
        try:
            store.open(llm(), controller=ctrl)
        except AdmissionDenied:
            with guard:
                denied += 1
        else:
            with guard:
                admitted += 1

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert admitted == cap and denied == n - cap
    assert store.snapshot().out_llm_calls == cap  # exactly cap reservations, no over-admit
