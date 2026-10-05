"""A ceiling several runs spend from at once (G-28, D57)."""

from __future__ import annotations

import asyncio

import pytest
from tests.fake_provider import MODEL, Reply, fake_llm

from ai_arch_toolkit.core import (
    AdmissionDenied,
    APIError,
    MeterScope,
    Money,
    Response,
    RunConfig,
    Usage,
)
from ai_arch_toolkit.toolkit.budget import BudgetController, BudgetPolicy, SharedBudget

# claude-sonnet-4-6: $3 in, $15 out per million tokens. A call says "hi" and allows 1000 output
# tokens: its worst case is about $0.015; it spends 10 in and 5 out, $0.000105.
ANSWER = Response(text="ok", usage=Usage(input_tokens=10, output_tokens=5), model=MODEL)
SPENT = Money.from_usd(0.000105)


def _llm(*script: Reply | Response) -> object:
    llm, _ = fake_llm(*(script or (ANSWER,)), max_tokens=1000, retry=False)
    return llm


async def test_a_call_in_flight_in_one_run_holds_the_ceiling_against_the_other() -> None:
    shared = SharedBudget(BudgetPolicy(max_cost=0.02))
    release = asyncio.Event()
    held = _llm(Reply(response=ANSWER, hold=release))
    other = _llm()

    async def first_run() -> None:
        with MeterScope(RunConfig(shared=shared)):
            await held.complete("hi")  # type: ignore[attr-defined]

    first = asyncio.create_task(first_run())
    await asyncio.sleep(0.01)  # the first run's call is in flight, holding its worst case

    with MeterScope(RunConfig(shared=shared)):
        with pytest.raises(AdmissionDenied) as denied:
            await other.complete("hi")  # type: ignore[attr-defined]
        assert denied.value.dimension == "cost"

        release.set()
        await first  # settled: its hold gives way to what it spent
        await other.complete("hi")  # type: ignore[attr-defined]

    assert shared.snapshot().cost == SPENT + SPENT


async def test_what_the_app_already_spent_counts() -> None:
    shared = SharedBudget(BudgetPolicy(max_cost=0.02), spent=0.01)

    with MeterScope(RunConfig(shared=shared)), pytest.raises(AdmissionDenied):
        await _llm().complete("hi")  # type: ignore[attr-defined]

    assert shared.snapshot().cost == Money.from_usd(0.01)


async def test_a_run_with_a_budget_of_its_own_spends_from_both() -> None:
    shared = SharedBudget(BudgetPolicy(max_cost=1.0), spent=0.5)
    own = BudgetController(BudgetPolicy(max_cost=0.1))

    with MeterScope(RunConfig(controller=own, shared=shared)) as scope:
        await _llm().complete("hi")  # type: ignore[attr-defined]

    assert scope.snapshot().cost == SPENT
    assert shared.snapshot().cost == Money.from_usd(0.5) + SPENT


async def test_two_runs_calling_at_once_never_pass_the_ceiling_together() -> None:
    ceiling = 0.001
    shared = SharedBudget(BudgetPolicy(max_cost=ceiling))
    answered: list[int] = []

    async def run() -> None:
        llm, _ = fake_llm(Reply(response=ANSWER, delay=0.001), max_tokens=10, retry=False)
        with MeterScope(RunConfig(shared=shared)):
            for _ in range(20):
                try:
                    await llm.complete("hi")
                except AdmissionDenied:
                    await asyncio.sleep(0.001)
                    continue
                answered.append(1)

    await asyncio.gather(run(), run())

    assert shared.snapshot().cost <= Money.from_usd(ceiling)
    assert len(answered) >= 2


async def test_a_failed_call_leaves_its_ceiling_in_the_shared_meter() -> None:
    shared = SharedBudget(BudgetPolicy(max_cost=1.0))
    broken = _llm(Reply(error=APIError(503, "down")))

    with MeterScope(RunConfig(shared=shared)), pytest.raises(APIError):
        await broken.complete("hi")  # type: ignore[attr-defined]

    snapshot = shared.snapshot()
    assert snapshot.uncertain_cost_count == 1
    assert Money.zero() < snapshot.uncertain_cost < Money.from_usd(0.02)


def test_the_report_projects_the_shared_spend_against_its_caps() -> None:
    shared = SharedBudget(BudgetPolicy(max_cost=1.0, max_llm_calls=10), spent=0.25)

    report = shared.report()

    assert report.cost == pytest.approx(0.25)
    assert report.llm_calls == 0


@pytest.mark.parametrize(
    "policy",
    [
        BudgetPolicy(max_cost=1.0, max_wall_s=60),
        BudgetPolicy(max_cost=1.0, max_total_tokens=1000),
        BudgetPolicy(max_input_tokens=1000),
    ],
)
def test_time_and_tokens_are_not_shared(policy: BudgetPolicy) -> None:
    with pytest.raises(ValueError, match="time and tokens"):
        SharedBudget(policy)


def test_the_seed_cannot_be_negative() -> None:
    with pytest.raises(ValueError, match="spent"):
        SharedBudget(BudgetPolicy(max_cost=1.0), spent=-1)


def test_runs_in_threads_never_pass_the_ceiling_together() -> None:
    import threading

    ceiling = 0.002
    shared = SharedBudget(BudgetPolicy(max_cost=ceiling))
    answered: list[int] = []

    def run() -> None:
        llm, _ = fake_llm(Reply(response=ANSWER, delay=0.001), max_tokens=10, retry=False)
        with MeterScope(RunConfig(shared=shared)):
            for _ in range(30):
                try:
                    llm.complete_sync("hi")
                except AdmissionDenied:
                    continue
                answered.append(1)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    snapshot = shared.snapshot()
    assert answered
    assert snapshot.cost <= Money.from_usd(ceiling)
    assert snapshot.out_cost == Money.zero()  # every hold gave way to what was spent
    assert snapshot.cost == SPENT * len(answered)
