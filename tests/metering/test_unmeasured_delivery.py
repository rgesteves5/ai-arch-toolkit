"""A result delivered without its usage: bounded as a failure that may have been billed (D49), and
counted apart, so an app that bills can raise an alarm instead of charging the bound (D67)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest
from tests.fake_provider import MODEL, USAGE, Reply, fake_llm

from ai_arch_toolkit.core import (
    APIError,
    Cost,
    MeterScope,
    MeterSnapshot,
    Money,
    OperationRequest,
    Response,
    RunConfig,
    Usage,
)
from ai_arch_toolkit.core._metering._worst_case import worst_case
from ai_arch_toolkit.core._pricing import pricing
from ai_arch_toolkit.toolkit.budget import (
    BudgetController,
    BudgetPolicy,
    BudgetReport,
    SharedBudget,
)

NO_USAGE = Reply(response=Response(text="ok", model=MODEL), usage_reported=False)
PATHS = ["complete", "stream", "stream_events"]


async def _call(llm: Any, path: str) -> Response:
    if path == "complete":
        return await llm.complete("hi")
    stream = getattr(llm, path)("hi")
    async for _ in stream:
        pass
    return stream.response


@pytest.mark.parametrize("path", PATHS)
async def test_a_reply_without_usage_is_bounded_like_a_billable_failure(path: str) -> None:
    delivered, _ = fake_llm(NO_USAGE, max_tokens=1000, retry=False)
    failed, _ = fake_llm(APIError(503, "failed"), max_tokens=1000, retry=False)
    with MeterScope(RunConfig(retain_meter_events=True)) as scope:
        response = await _call(delivered, path)
    with MeterScope(RunConfig(retain_meter_events=True)) as failure, pytest.raises(APIError):
        await _call(failed, path)
    assert response.text == "ok" and response.cost is None
    (event,) = scope.events()
    (failed_event,) = failure.events()
    assert (event.status, event.cost.kind) == ("settled", "unknown")
    assert event.cost.reason == "the provider reported no usage"
    assert event.cost.at_most is not None and event.cost.at_most > Money.zero()
    assert event.cost.at_most == failed_event.cost.at_most


async def test_the_meter_bounds_it_and_counts_it_apart() -> None:
    llm, _ = fake_llm(NO_USAGE, max_tokens=1000, retry=False)
    with MeterScope() as scope:
        await llm.complete("hi")
    snap = scope.snapshot()
    assert snap.cost == Money.zero()  # never a known $0: the bound is uncertain spend
    assert snap.uncertain_cost > Money.zero()
    assert (snap.unknown_cost_count, snap.uncertain_cost_count) == (0, 1)
    assert snap.unmeasured_delivery_count == 1
    assert MeterSnapshot.from_dict(snap.to_dict()) == snap
    report = BudgetReport.from_snapshot(snap)
    assert report.cost_uncertain
    assert report.cost_at_most == snap.uncertain_cost.to_float()
    assert report.unmeasured_delivery_count == 1
    assert report.to_dict()["unmeasured_delivery_count"] == 1


async def test_only_a_delivery_without_its_cost_counts() -> None:
    llm, _ = fake_llm(
        Response(text="measured", usage=USAGE, model=MODEL),
        # The provider's own cost is a measured cost even without the usage (xAI reports both).
        Reply(
            response=Response(text="billed", model=MODEL, provider_cost=0.002),
            usage_reported=False,
        ),
        APIError(503, "failed"),  # bounded too, but nothing was delivered
        max_tokens=1000,
        retry=False,
    )
    with MeterScope() as scope:
        await llm.complete("hi")
        await llm.complete("hi")
        with pytest.raises(APIError):
            await llm.complete("hi")
    snap = scope.snapshot()
    assert snap.llm_calls == 3
    assert snap.uncertain_cost_count == 1
    assert snap.unmeasured_delivery_count == 0


async def test_under_a_strict_budget_the_bound_is_the_hold() -> None:
    release = asyncio.Event()
    llm, provider = fake_llm(
        Reply(response=Response(text="ok", model=MODEL), usage_reported=False, hold=release),
        max_tokens=1000,
        retry=False,
    )
    controller = BudgetController(BudgetPolicy(max_cost=1.0, reserve="strict"))
    with MeterScope(RunConfig(controller=controller, retain_meter_events=True)) as scope:
        call = asyncio.create_task(llm.complete("hi"))
        await provider.entered.wait()
        hold = scope.snapshot().out_cost
        release.set()
        await call
    assert hold > Money.zero()
    (event,) = scope.events()
    assert event.cost.at_most == hold
    assert scope.snapshot().out_cost == Money.zero()


async def test_a_cost_cap_counts_the_bound_and_does_not_fail_closed() -> None:
    llm, _ = fake_llm(NO_USAGE, Response(text="ok", usage=USAGE, model=MODEL), max_tokens=1000)
    controller = BudgetController(BudgetPolicy(max_cost=1.0))  # unpriced="fail_closed"
    with MeterScope(RunConfig(controller=controller)) as scope:
        await llm.complete("hi")
        await llm.complete("hi")  # admitted: the first call's cost is bounded, not unpriced
    report = BudgetReport.from_snapshot(scope.snapshot(), controller.policy)
    assert report.llm_calls == 2
    assert report.cost_at_most is not None and report.cost_at_most > report.cost > 0
    assert report.unmeasured_delivery_count == 1
    assert not report.over_budget


async def test_a_shared_ceiling_counts_it_and_stays_open() -> None:
    shared = SharedBudget(BudgetPolicy(max_cost=1.0))
    llm, _ = fake_llm(NO_USAGE, max_tokens=1000, retry=False)
    for _ in range(2):  # the second run is admitted: nothing went unbounded
        with MeterScope(RunConfig(shared=shared)):
            await llm.complete("hi")
    snap = shared.snapshot()
    assert (snap.unknown_cost_count, snap.uncertain_cost_count) == (0, 2)
    assert snap.unmeasured_delivery_count == 2


def _sized_llm_request() -> OperationRequest:
    return OperationRequest(
        kind="llm",
        parent_span_id="run",
        model=MODEL,
        content_size_hint=400,
        declared_max_output_tokens=100,
    )


def test_the_bound_is_the_worst_case_of_the_operation_facts(caplog) -> None:
    request = _sized_llm_request()
    with MeterScope(RunConfig(retain_meter_events=True)) as scope:
        op = scope.open(request)
        op.mark_started()
        with caplog.at_level(logging.WARNING):
            op.settle_unmeasured(usage=Usage(), reason="no usage")
            op.settle_unmeasured(usage=Usage(), reason="no usage")  # a replay: one settlement
    assert not caplog.records
    (event,) = scope.events()
    expected = worst_case(request, pricing)
    assert expected is not None
    assert event.cost == Cost.unknown("no usage", at_most=expected.cost)
    assert scope.snapshot().unmeasured_delivery_count == 1


def test_an_unmeasured_delivery_no_one_can_bound_stays_unbounded_and_counted() -> None:
    class Unpriced:
        def price(self, request: OperationRequest, usage: Usage) -> Cost:
            return Cost.unknown("no price")

    with MeterScope(RunConfig(pricer=Unpriced())) as scope:
        op = scope.open(_sized_llm_request())
        op.mark_started()
        op.settle_unmeasured(usage=Usage(), reason="no usage")
    snap = scope.snapshot()
    assert (snap.unknown_cost_count, snap.uncertain_cost_count) == (1, 0)
    assert snap.unmeasured_delivery_count == 1
    assert BudgetReport.from_snapshot(snap).cost_at_most is None


def test_it_settles_only_a_started_operation() -> None:
    with MeterScope() as scope:
        op = scope.open(_sized_llm_request())
        with pytest.raises(ValueError, match="before mark_started"):
            op.settle_unmeasured(usage=Usage(), reason="no usage")
        op.abort()
