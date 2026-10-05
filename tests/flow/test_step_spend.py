"""What the meter measured in each step (``StepTrace.metered``), and the public spans (G-32)."""

from __future__ import annotations

import asyncio
import threading

from ai_arch_toolkit.core import (
    LLM,
    MeterScope,
    MeterSnapshot,
    Money,
    Result,
    RunConfig,
    State,
    StateSnapshot,
    Step,
    StepTrace,
    Trace,
    bind_meter,
    current_meter,
    current_span_id,
    open_span,
)
from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._step_engine import execute_step
from ai_arch_toolkit.toolkit.budget import BudgetPolicy
from ai_arch_toolkit.toolkit.flow import Flow, FlowStep
from tests.fake_provider import fake_llm

MODEL = "claude-sonnet-4-6"
USAGE = Usage(input_tokens=12, output_tokens=4)


def make_llm() -> LLM:
    llm, _ = fake_llm(Response(text="ok", usage=USAGE, model=MODEL), model=MODEL)
    return llm


def calling(name: str, llm: LLM, calls: int, *, then_sleep: float = 0.0) -> Step:
    """A step that makes ``calls`` LLM calls, letting the others run between them."""

    async def fn(snap: StateSnapshot) -> Result:
        for _ in range(calls):
            await llm.complete("hi")
            await asyncio.sleep(0)
        if then_sleep:
            await asyncio.sleep(then_sleep)
        return Result(value=name)

    return Step(name=name, fn=fn)


def metered(trace: StepTrace | None) -> MeterSnapshot:
    assert trace is not None and trace.metered is not None
    return trace.metered


async def test_each_step_carries_what_the_meter_measured_in_it() -> None:
    llm = make_llm()
    flow = Flow(calling("a", llm, 1), calling("b", llm, 2), name="two")

    result = await flow.run(State())

    a, b = metered(result.trace.step("a")), metered(result.trace.step("b"))
    assert (a.llm_calls, a.input_tokens, a.output_tokens) == (1, 12, 4)
    assert (b.llm_calls, b.input_tokens, b.output_tokens) == (2, 24, 8)
    assert a.cost > Money.zero() and b.cost == a.cost * 2
    assert result.meter_scope is not None
    assert a.cost + b.cost == result.meter_scope.snapshot().cost


async def test_parallel_steps_do_not_mix() -> None:
    llm = make_llm()
    flow = Flow(
        calling("start", llm, 0),
        FlowStep(step=calling("one", llm, 1), after=("start",)),
        FlowStep(step=calling("three", llm, 3), after=("start",)),
        name="fan-out",
    )

    result = await flow.run(State())

    assert metered(result.trace.step("one")).llm_calls == 1
    assert metered(result.trace.step("three")).llm_calls == 3
    assert metered(result.trace.step("start")).llm_calls == 0


async def test_a_nested_flow_counts_in_the_step_that_runs_it() -> None:
    llm = make_llm()
    inner = Flow(calling("x", llm, 1), calling("y", llm, 2), name="inner")
    outer = Flow(calling("a", llm, 1), inner, name="outer")

    result = await outer.run(State())

    assert metered(result.trace.step("a")).llm_calls == 1
    runs_inner = [step for step in result.trace.steps if step.name == "inner"]
    assert len(runs_inner) == 1 and metered(runs_inner[0]).llm_calls == 3
    assert metered(result.trace.step("x")).llm_calls == 1
    assert metered(result.trace.step("y")).llm_calls == 2


async def test_each_flow_a_step_runs_itself_carries_its_own_spend() -> None:
    llm = make_llm()
    inner = Flow(calling("x", llm, 1), name="inner")

    async def runs_it_twice(snap: StateSnapshot) -> Result:
        await inner.run(State())
        await inner.run(State())
        return Result(value="twice")

    result = await Flow(Step(name="s", fn=runs_it_twice), name="outer").run(State())

    step = result.trace.steps[0]
    assert metered(step).llm_calls == 2
    assert [child.name for child in step.children] == ["inner", "inner"]
    assert [metered(child).llm_calls for child in step.children] == [1, 1]
    assert [metered(child.children[0]).llm_calls for child in step.children] == [1, 1]


async def test_a_step_cut_by_the_budget_keeps_what_it_spent() -> None:
    llm = make_llm()
    flow = Flow(calling("s", llm, 3), name="capped", budget_policy=BudgetPolicy(max_llm_calls=2))

    result = await flow.run(State())

    cut = result.trace.step("s")
    assert cut is not None and cut.error is not None
    assert cut.error.startswith("stopped by the budget")
    assert metered(cut).llm_calls == 2


async def test_a_step_cut_by_the_flows_timeout_keeps_what_it_spent() -> None:
    llm = make_llm()
    flow = Flow(calling("slow", llm, 1, then_sleep=10.0), name="late", timeout=0.2)

    result = await flow.run(State())

    cut = result.trace.step("slow")
    assert cut is not None and cut.error is not None
    assert cut.error.startswith("cut by the flow's timeout")
    assert metered(cut).llm_calls == 1


async def test_a_step_with_no_meter_has_nothing_metered() -> None:
    llm = make_llm()

    _, trace = await execute_step(calling("s", llm, 1), StateSnapshot())

    assert current_meter() is None and trace.metered is None


async def test_execute_step_measures_under_a_bound_meter() -> None:
    llm = make_llm()

    with MeterScope(RunConfig()) as scope:
        await llm.complete("before")
        _, trace = await execute_step(calling("s", llm, 2), StateSnapshot())

    assert metered(trace).llm_calls == 2 and scope.snapshot().llm_calls == 3


async def test_what_was_metered_survives_serialization() -> None:
    llm = make_llm()
    result = await Flow(calling("a", llm, 2), name="saved").run(State())

    again = Trace.from_dict(result.trace.to_dict())

    step = again.step("a")
    assert step is not None and step.metered == result.trace.step("a").metered  # type: ignore[union-attr]
    assert StepTrace.from_dict(StepTrace(name="plain").to_dict()).metered is None


async def test_open_span_measures_a_block_outside_a_flow() -> None:
    llm = make_llm()

    with MeterScope(RunConfig()) as scope:
        await llm.complete("the turn")
        with open_span("delegate") as span_id:
            assert span_id is not None and current_span_id() == span_id
            await llm.complete("the subagent")
            await llm.complete("the subagent again")
            meter = current_meter()
            assert meter is scope
            block = meter.for_span(span_id)

    assert block.llm_calls == 2 and scope.snapshot().llm_calls == 3


async def test_open_span_is_a_no_op_without_a_meter() -> None:
    with open_span("delegate") as span_id:
        assert span_id is None and current_span_id() is None


async def test_bind_meter_carries_a_span_into_a_thread() -> None:
    llm = make_llm()

    with MeterScope(RunConfig()) as scope, open_span("worker") as span_id:
        assert span_id is not None
        captured = (current_meter(), current_span_id())

        def work() -> None:
            with bind_meter(*captured):
                llm.complete_sync("in a thread")

        thread = threading.Thread(target=work)
        thread.start()
        thread.join()
        block = scope.for_span(span_id)

    assert block.llm_calls == 1 and scope.snapshot().llm_calls == 1
