"""Unbilled failures are free; bounded uncertainty consumes the cap without inventing spend."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from tests.fake_provider import MODEL, fake_llm

from ai_arch_toolkit.core import (
    APIError,
    Cost,
    MeterScope,
    MeterSnapshot,
    Money,
    OperationRequest,
    Response,
    RetryConfig,
    RunConfig,
    Usage,
)
from ai_arch_toolkit.core._metering._admission import AdmissionDenied, Reservation
from ai_arch_toolkit.core._pricing import _estimate_response_cost
from ai_arch_toolkit.toolkit.budget import BudgetController, BudgetPolicy, BudgetReport


@dataclass(frozen=True, slots=True)
class FixedEstimate:
    def estimate(self, request: OperationRequest) -> Reservation:
        return Reservation(cost=Money.from_usd(0.6))


@pytest.mark.parametrize("delivery", ["not_sent", "unbilled"])
def test_free_failure_preserves_started_count_and_allows_next_operation(delivery):
    scope = MeterScope(
        RunConfig(
            controller=BudgetController(BudgetPolicy(max_cost=1.0), FixedEstimate()),
            retain_meter_events=True,
        )
    )
    with scope:
        request = OperationRequest(kind="llm", parent_span_id="run", model="priced")
        op = scope.open(request)
        op.mark_started()
        op.fail(delivery)
        scope.open(request).abort()
    snap = scope.snapshot()
    assert snap.llm_calls == 1
    assert snap.unknown_cost_count == 0
    assert snap.cost == Money.zero()
    assert scope.events()[0].delivery == delivery
    assert scope.events()[0].cost == Cost.known(Money.zero())


@pytest.mark.parametrize("reserve", ["none", "strict"])
def test_uncertain_bound_is_retained_and_enforced_without_counting_as_spend(reserve):
    controller = BudgetController(BudgetPolicy(max_cost=1.0, reserve=reserve), FixedEstimate())
    with MeterScope(RunConfig(controller=controller, retain_meter_events=True)) as scope:
        request = OperationRequest(kind="llm", parent_span_id="run", model="priced")
        op = scope.open(request)
        op.mark_started()
        op.fail("indeterminate")
        snap = scope.snapshot()
        assert snap.cost == Money.zero()
        assert snap.uncertain_cost == Money.from_usd(0.6)
        assert snap.uncertain_cost_count == 1
        assert snap.unknown_cost_count == 0
        assert snap.out_cost == Money.zero()
        report = BudgetReport.from_snapshot(snap)
        assert report.cost == 0.0 and report.cost_at_most == 0.6 and report.cost_uncertain
        if reserve == "strict":
            with pytest.raises(AdmissionDenied):
                scope.open(request)
        else:
            second = scope.open(request)
            second.mark_started()
            second.fail("indeterminate")
            with pytest.raises(AdmissionDenied):
                scope.open(request)
    assert scope.events()[0].cost.at_most == Money.from_usd(0.6)


def test_failure_sizing_and_estimator_run_outside_store_lock():
    class PeekingEstimator:
        def estimate(self, request):
            assert scope.snapshot().llm_calls == 1
            assert request.content_size_hint == 123
            return Reservation(cost=Money.from_usd(0.1))

    controller = BudgetController(BudgetPolicy(max_cost=1.0), PeekingEstimator())
    with MeterScope(RunConfig(controller=controller)) as scope:

        def sized():
            assert scope.snapshot().llm_calls == 1
            return OperationRequest(kind="llm", parent_span_id="run", content_size_hint=123)

        op = scope.open(OperationRequest(kind="llm", parent_span_id="run"), failure_request=sized)
        op.mark_started()
        op.fail("indeterminate")
    assert scope.snapshot().uncertain_cost == Money.from_usd(0.1)


def test_composite_bounded_cost_includes_known_components_in_its_bound():
    merged = Cost.merged(
        Cost.known(Money.from_usd(0.2)),
        Cost.unknown("lost reply", at_most=Money.from_usd(0.3)),
    )
    assert merged.kind == "unknown" and merged.at_most == Money.from_usd(0.5)
    assert Cost.merged(merged, Cost.unknown("unpriced")).at_most is None


def test_close_preserves_uncertain_strict_hold_and_releases_pending_reservation():
    config = RunConfig(
        controller=BudgetController(
            BudgetPolicy(max_cost=2.0, reserve="strict"),
            FixedEstimate(),
        ),
        retain_meter_events=True,
    )
    with MeterScope(config) as scope:
        request = OperationRequest(kind="llm", parent_span_id="run")
        scope.open(request).mark_started()
        scope.open(request)
    snap = scope.snapshot()
    assert snap.llm_calls == 1 and snap.out_llm_calls == 0
    assert snap.uncertain_cost == Money.from_usd(0.6)
    assert snap.out_cost == Money.zero()
    assert [event.status for event in scope.events()] == ["incomplete", "aborted"]


@pytest.mark.parametrize("raising", [False, True])
async def test_strict_tool_reserves_custom_price_before_execution(raising):
    from ai_arch_toolkit.core import ToolCall, ToolGroup, tool
    from ai_arch_toolkit.toolkit.budget import budget_scope

    class Pricer:
        def price(self, request, usage):
            if raising:
                raise RuntimeError("price unavailable")
            return Cost.known(Money.from_usd(0.016))

    calls = []

    @tool
    def paid() -> str:
        calls.append("paid")
        return "ok"

    group = ToolGroup(paid)
    with budget_scope(BudgetPolicy(max_cost=0.02, reserve="strict"), pricer=Pricer()):
        if raising:
            with pytest.raises(AdmissionDenied):
                await group.async_execute(ToolCall(id="one", name="paid", input={}))
            assert calls == []
        else:
            assert (await group.async_execute(ToolCall(id="one", name="paid", input={}))).ok
            with pytest.raises(AdmissionDenied):
                await group.async_execute(ToolCall(id="two", name="paid", input={}))
            assert calls == ["paid"]


def test_budget_report_constructor_keeps_existing_keywords() -> None:
    report = BudgetReport.from_snapshot(MeterSnapshot(cost=Money.from_usd(0.3)), BudgetPolicy())
    arguments = report.to_dict()
    arguments.pop("cost_at_most")
    legacy = BudgetReport(**arguments)
    assert legacy.cost == 0.3
    assert legacy.cost_at_most == 0.3


# A failure the provider reported usage for (Meta's response.failed): the meter keeps it (R02).
FAILED_USAGE = Usage(input_tokens=100, output_tokens=50)


def test_a_failure_with_reported_usage_settles_that_usage_and_cost() -> None:
    with MeterScope(RunConfig(retain_meter_events=True)) as scope:
        op = scope.open(OperationRequest(kind="llm", parent_span_id="run", model="priced"))
        op.mark_started()
        op.fail("indeterminate", usage=FAILED_USAGE, cost=Cost.known(Money.from_usd(0.01)))
    snap = scope.snapshot()
    assert (snap.llm_calls, snap.input_tokens, snap.output_tokens) == (1, 100, 50)
    assert snap.cost == Money.from_usd(0.01)
    assert (snap.unknown_cost_count, snap.uncertain_cost_count) == (0, 0)
    (event,) = scope.events()
    assert (event.status, event.delivery, event.usage) == ("failed", "indeterminate", FAILED_USAGE)


async def test_llm_meters_the_usage_a_failed_response_reported() -> None:
    llm, _ = fake_llm(APIError(503, "failed", usage=FAILED_USAGE))
    with MeterScope(RunConfig(retain_meter_events=True)) as scope, pytest.raises(APIError):
        await llm.complete("hi")
    snap = scope.snapshot()
    expected = _estimate_response_cost(MODEL, FAILED_USAGE)
    assert expected is not None
    assert snap.cost == Money.from_usd(expected)
    assert (snap.input_tokens, snap.unknown_cost_count, snap.uncertain_cost_count) == (100, 0, 0)


async def test_the_failed_attempt_keeps_its_reported_usage() -> None:
    retry = RetryConfig(max_retries=1, base_delay=0.001)
    llm, _ = fake_llm(
        APIError(503, "failed", usage=FAILED_USAGE), Response(text="ok"), retry=retry
    )
    response = await llm.complete("hi")
    assert [(a.status, a.usage) for a in response.attempts] == [
        ("failed", FAILED_USAGE),
        ("ok", Usage()),
    ]
