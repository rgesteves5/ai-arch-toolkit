"""MeterScope ambient binding: ContextVar scope/span, span nesting, modes, async, RunConfig."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._metering._admission import (
    AdmissionDecision,
    AdmissionDenied,
    MeterSnapshot,
    ResourceLimits,
)
from ai_arch_toolkit.core._metering._cost import Cost
from ai_arch_toolkit.core._metering._events import UsageEvent
from ai_arch_toolkit.core._metering._money import Money
from ai_arch_toolkit.core._metering._operation import OperationRequest
from ai_arch_toolkit.core._metering._scope import (
    MeterScope,
    RunConfig,
    bind_meter,
    current_meter,
    current_span_id,
    open_span,
)
from ai_arch_toolkit.core._response import Usage


class CapController:
    """Blindly admits but returns run-level limits — exercises enforce mode through the scope."""

    def __init__(self, **limits: int) -> None:
        self._limits = ResourceLimits(**limits)

    def admit(self, snapshot: MeterSnapshot, request: OperationRequest) -> AdmissionDecision:
        return AdmissionDecision.allow(limits=self._limits)


def here() -> OperationRequest:
    return OperationRequest(kind="llm", parent_span_id=current_span_id() or "run")


def test_unbound_context_is_unmetered():
    assert current_meter() is None
    assert current_span_id() is None


def test_scope_binds_and_restores():
    with MeterScope() as scope:
        assert current_meter() is scope
        assert current_span_id() == scope.run_span_id
    assert current_meter() is None
    assert current_span_id() is None


def test_measure_only_is_the_default():
    with MeterScope() as scope:
        op = scope.open(here())
        op.mark_started()
        op.settle(usage=Usage(input_tokens=8), cost=Cost.known(Money.from_usd(0.01)))
    assert scope.snapshot().input_tokens == 8  # store persists after exit


def test_scope_enforces_when_given_a_controller():
    cfg = RunConfig(controller=CapController(max_llm_calls=1))
    with MeterScope(cfg) as scope:
        scope.open(here())  # ok
        with pytest.raises(AdmissionDenied):
            scope.open(here())  # over the run cap


def test_close_on_exit_incompletes_a_started_op():
    with MeterScope() as scope:
        scope.open(here()).mark_started()  # never settled
    snap = scope.snapshot()
    assert snap.llm_calls == 1 and snap.unknown_cost_count == 1


def test_open_span_nests_through_the_contextvar():
    with MeterScope() as scope:
        run = current_span_id()
        with open_span("step") as step:
            assert current_span_id() == step and step != run
            op = scope.open(here())  # attaches to the step span
            op.mark_started()
            op.settle(usage=Usage(input_tokens=5), cost=Cost.known(Money.zero()))
            with open_span("tool") as tool:
                assert current_span_id() == tool
            assert current_span_id() == step  # inner restored
            assert scope.for_span(step).input_tokens == 5  # queryable while the span is live
        assert current_span_id() == run  # outer restored
        # the step span is reclaimed on context-manager exit (bounded memory); its totals survive
        # rolled up into the run root.
        assert scope.for_span(run).input_tokens == 5


def test_open_span_is_a_noop_when_unmetered():
    with open_span("step") as sid:
        assert sid is None  # no scope -> nothing opened, no crash


def test_bind_meter_rebinds_a_captured_scope():
    # Simulates a stream finalizer settling from a context that lost the binding.
    scope = MeterScope()
    op = scope.open(OperationRequest(kind="llm", parent_span_id=scope.run_span_id))
    op.mark_started()
    assert current_meter() is None
    with bind_meter(scope):
        assert current_meter() is scope
        assert current_span_id() == scope.run_span_id
    assert current_meter() is None


def test_runconfig_wires_sinks_through_the_scope():
    events: list[UsageEvent] = []

    class Rec:
        def emit(self, event: UsageEvent) -> None:
            events.append(event)

    with MeterScope(RunConfig(sinks=[Rec()])) as scope:
        op = scope.open(here())
        op.mark_started()
        op.settle(usage=Usage(), cost=Cost.known(Money.zero()))
    assert len(events) == 1 and events[0].status == "settled"


async def test_async_context_manager_binds():
    async with MeterScope() as scope:
        assert current_meter() is scope
    assert current_meter() is None
