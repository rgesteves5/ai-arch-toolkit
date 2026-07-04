"""The flow opens a run-scoped meter; LLM calls inside steps are metered and projected."""

from __future__ import annotations

import asyncio

import pytest

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._metering._scope import MeterScope, RunConfig
from ai_arch_toolkit.core._policy import Policy
from ai_arch_toolkit.core._providers._base import StreamState
from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._state import State, StateSnapshot
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.core._step_engine import execute_step
from ai_arch_toolkit.toolkit.budget import BudgetPolicy
from ai_arch_toolkit.toolkit.flow._flow import Flow, FlowStep

MODEL = "claude-sonnet-4-6"


class FakeProvider:
    async def complete(self, messages, *, system=None, tools=None, **kwargs) -> Response:
        return Response(text="ok", usage=Usage(input_tokens=12, output_tokens=4), model=MODEL)

    def stream(self, messages, *, system=None, tools=None, **kwargs):
        state = StreamState()
        state.usage = Usage(input_tokens=12, output_tokens=4)

        async def _aiter():
            yield "ok"

        return _aiter(), state


def make_llm() -> LLM:
    llm = LLM(MODEL, api_key="test")
    llm._provider = FakeProvider()  # type: ignore[assignment]
    return llm


def make_unpriced_llm() -> LLM:
    # A claude-prefixed model (so the provider constructs) that is absent from the pricing table,
    # so the charge site prices it as Cost.unknown.
    llm = LLM("claude-does-not-exist-9999", api_key="test")
    llm._provider = FakeProvider()  # type: ignore[assignment]
    return llm


async def test_flow_meters_an_llm_call_inside_a_step():
    llm = make_llm()

    async def call_model(snap: StateSnapshot) -> Result:
        await llm.complete("hi")
        return Result(value="done")

    result = await Flow(Step(name="s", fn=call_model), name="metered").run(State())
    meter = result.trace.metadata["meter"]
    assert meter["llm_calls"] == 1
    assert meter["input_tokens"] == 12 and meter["output_tokens"] == 4
    assert meter["cost"] > 0 and not meter["over_budget"]


async def test_flow_accumulates_metering_across_steps():
    llm = make_llm()

    async def call_model(snap: StateSnapshot) -> Result:
        await llm.complete("hi")
        return Result(value="done")

    flow = Flow(
        Step(name="a", fn=call_model),
        Step(name="b", fn=call_model),
        name="two-step",
    )
    meter = (await flow.run(State())).trace.metadata["meter"]
    assert meter["llm_calls"] == 2 and meter["input_tokens"] == 24 and meter["output_tokens"] == 8


async def test_flow_without_an_llm_reports_zero_metering():
    async def noop(snap: StateSnapshot) -> Result:
        return Result(value="done")

    result = await Flow(Step(name="s", fn=noop), name="plain").run(State())
    meter = result.trace.metadata["meter"]
    assert meter["llm_calls"] == 0 and meter["cost"] == 0.0 and not meter["over_budget"]


async def test_parallel_dag_budget_denial_surfaces_and_does_not_hang():
    # Regression: the parallel DAG path used to swallow AdmissionDenied, leaving the denied node
    # un-marked -> infinite loop. wait_for guards against a re-introduced hang.
    llm = make_llm()

    async def call_model(snap: StateSnapshot) -> Result:
        await llm.complete("hi")
        return Result(value="ok")

    flow = Flow(
        FlowStep(step=Step(name="a", fn=call_model)),
        FlowStep(step=Step(name="b", fn=call_model)),
        FlowStep(step=Step(name="c", fn=call_model), after=("a", "b")),  # makes it a DAG
        name="par",
        budget_policy=BudgetPolicy(max_llm_calls=1),
    )
    result = await asyncio.wait_for(flow.run(State()), timeout=10)
    assert "budget_exceeded" in result.results


async def test_nested_flow_as_step_meters_without_crashing():
    # Regression: the outer flow writes _meter_scope into the world layer, which as_step used to
    # share read-only (MappingProxyType) -> TypeError swallowed -> inner flow silently failed.
    llm = make_llm()

    async def call_model(snap: StateSnapshot) -> Result:
        await llm.complete("hi")
        return Result(value="inner-done")

    inner = Flow(Step(name="inner_call", fn=call_model), name="inner")
    result = await Flow(inner, name="parent").run(State())
    # The nested call ran and metered under the parent scope (0 if the nested run had crashed).
    assert result.trace.metadata["meter"]["llm_calls"] == 1


async def test_iter_flow_abandonment_finalizes_the_scope():
    # Regression: scope.close() sat AFTER the try, so a GeneratorExit (consumer abandons the
    # stream) skipped it entirely — the STARTED op was never finalized. Now it's in a finally, so
    # aclose() runs it. (Driven through iter_flow directly so aclose hits the finally sync; via the
    # public flow.iter wrapper it runs at async-gen finalization instead of never.)
    from ai_arch_toolkit.toolkit.flow._executor import iter_flow

    llm = make_llm()

    async def start_stream(snap: StateSnapshot) -> Result:
        llm.stream("hi")  # opens+starts a stream op, never drained
        return Result(value="started")

    state = State()
    gen = iter_flow(Flow(Step(name="s", fn=start_stream), name="f"), state)
    async for ev in gen:
        if ev.type == "step_end":
            break  # abandon before flow_end
    await gen.aclose()  # runs the finally -> scope.close()

    snap = state.get("_meter_scope").snapshot()
    assert snap.llm_calls == 1 and snap.unknown_cost_count == 1  # close() incompleted the op


async def test_policy_max_cost_trips_on_metered_spend():
    # fase 6: a per-step Policy.max_cost is enforced against the step's METERED span cost, even
    # when the Result carries no manual cost annotation.
    llm = make_llm()

    async def call(snap: StateSnapshot) -> Result:
        await llm.complete("hi")
        return Result(value="done")  # no cost= annotation; the meter is the only source

    step = Step(name="s", fn=call, policy=Policy(max_cost=1e-6))  # priced call exceeds this
    with MeterScope(RunConfig()) as scope:  # measure-only; the step opens its own span
        result, trace = await execute_step(step, StateSnapshot())
    assert result.is_error and "cost_exceeded" in trace.policy_decisions
    assert scope.snapshot().llm_calls == 1


async def test_policy_max_cost_isolates_to_the_step_span():
    # A per-step max_cost counts only THAT step's metered span, not sibling/prior steps sharing the
    # run — the reason it projects a span instead of a run-cumulative delta.
    llm = make_llm()

    async def call(snap: StateSnapshot) -> Result:
        await llm.complete("hi")
        return Result(value="ok")

    one = (await Flow(Step(name="p", fn=call), name="probe").run(State())).total_cost
    assert one > 0

    # Step "a" (uncapped) settles its cost into the shared run first; step "b"'s cap admits its own
    # single call (~one) but the run cumulative (a+b ≈ 2·one) would trip it if the span leaked.
    flow = Flow(
        Step(name="a", fn=call),
        Step(name="b", fn=call, policy=Policy(max_cost=one * 1.5)),
        name="iso",
    )
    result = await flow.run(State())
    b = result.trace.step("b")
    assert b is not None and "cost_exceeded" not in b.policy_decisions  # counted only b's own call
    assert result.meter.llm_calls == 2  # both ran


async def test_policy_max_cost_sums_annotation_and_metered_span():
    # Policy.max_cost checks result.cost (manual annotation) PLUS the step's metered span cost —
    # neither alone exceeds the cap here, their sum does.
    llm = make_llm()

    async def probe_call(snap: StateSnapshot) -> Result:
        await llm.complete("hi")
        return Result(value="ok")

    one = (await Flow(Step(name="p", fn=probe_call), name="probe").run(State())).total_cost
    assert one > 0

    async def call_and_annotate(snap: StateSnapshot) -> Result:
        await llm.complete("hi")  # metered span ≈ one
        return Result(value="ok", cost=one)  # manual annotation ≈ one

    step = Step(name="s", fn=call_and_annotate, policy=Policy(max_cost=one * 1.5))
    with MeterScope(RunConfig()) as scope:  # noqa: F841 — binds the ambient meter
        result, trace = await execute_step(step, StateSnapshot())
    assert result.is_error and "cost_exceeded" in trace.policy_decisions  # one + one > 1.5·one


async def test_unpriced_model_fails_closed_under_a_cost_cap_end_to_end():
    # Phase A, end-to-end: once a call settles with an unknown cost under a max_cost cap, the next
    # op is denied (fail_closed default) — the unbounded spend can't be admitted.
    llm = make_unpriced_llm()

    async def call(snap: StateSnapshot) -> Result:
        await llm.complete("hi")
        return Result(value="ok")

    flow = Flow(
        Step(name="a", fn=call),
        Step(name="b", fn=call),
        name="unpriced",
        budget_policy=BudgetPolicy(max_cost=1.0),
    )
    result = await flow.run(State())
    assert "budget_exceeded" in result.results  # 2nd call denied: a prior cost was unknown
    meter = result.trace.metadata["meter"]
    assert meter["llm_calls"] == 1 and meter["cost_uncertain"] is True


async def test_unpriced_model_allowed_when_unpriced_is_allow():
    # The opt-out: unpriced="allow" lets both calls run even though cost can't be bounded.
    llm = make_unpriced_llm()

    async def call(snap: StateSnapshot) -> Result:
        await llm.complete("hi")
        return Result(value="ok")

    flow = Flow(
        Step(name="a", fn=call),
        Step(name="b", fn=call),
        name="unpriced-allow",
        budget_policy=BudgetPolicy(max_cost=1.0, unpriced="allow"),
    )
    result = await flow.run(State())
    assert "budget_exceeded" not in result.results
    assert result.trace.metadata["meter"]["llm_calls"] == 2


async def test_stream_context_manager_error_does_not_settle_as_success():
    # N5: an exception mid-`async for` must leave the stream op INCOMPLETE (unknown cost) at scope
    # close, not settle it as a clean (under-recorded) success.
    llm = make_llm()
    scope = MeterScope(RunConfig())
    with scope, pytest.raises(ValueError):
        async with llm.stream("hi") as s:
            async for _chunk in s:
                raise ValueError("mid-stream boom")
    snap = scope.snapshot()
    assert snap.llm_calls == 1 and snap.unknown_cost_count == 1


async def test_step_spans_are_reclaimed_not_leaked():
    # N4: each per-step cost-cap span is dropped when its context manager exits, so a cyclic run
    # can't accumulate one _Span per step per iteration. Accounting survives the reclamation.
    llm = make_llm()

    async def call(snap: StateSnapshot) -> Result:
        await llm.complete("hi")
        return Result(value="ok")

    step = Step(name="s", fn=call, policy=Policy(max_cost=1000.0))  # opens a step span each run
    with MeterScope(RunConfig()) as scope:
        for _ in range(20):
            await execute_step(step, StateSnapshot())
    assert len(scope._store._spans) == 1  # only the run root remains
    assert scope.snapshot().llm_calls == 20  # accounting intact through the drops


async def test_policy_max_cost_fails_closed_on_unknown_cost():
    # decision #4: a per-step max_cost must fail CLOSED when the step's call can't be priced
    # (unpriced model / server tool) — even under a huge cap, unbounded spend must not pass.
    llm = make_unpriced_llm()

    async def call(snap: StateSnapshot) -> Result:
        await llm.complete("hi")
        return Result(value="ok")

    step = Step(
        name="s", fn=call, policy=Policy(max_cost=1000.0)
    )  # generous cap, but cost unknown
    with MeterScope(RunConfig()) as scope:
        result, trace = await execute_step(step, StateSnapshot())
    assert result.is_error and "cost_exceeded" in trace.policy_decisions
    assert scope.snapshot().unknown_cost_count == 1


async def test_policy_max_cost_does_not_trip_when_unmetered():
    # Same step with no bound meter and no annotation -> zero cost -> no cost_exceeded.
    llm = make_llm()

    async def call(snap: StateSnapshot) -> Result:
        await llm.complete("hi")
        return Result(value="done")

    step = Step(name="s", fn=call, policy=Policy(max_cost=1e-6))
    result, trace = await execute_step(step, StateSnapshot())
    assert result.is_ok and "cost_exceeded" not in trace.policy_decisions


async def test_flow_result_accessors_are_meter_derived():
    # meter / usage / total_cost all project from the run's meter scope (single source of truth).
    llm = make_llm()

    async def call_model(snap: StateSnapshot) -> Result:
        await llm.complete("hi")
        return Result(value="done")

    result = await Flow(Step(name="s", fn=call_model), name="m").run(State())
    report = result.meter
    assert report is not None
    assert report.llm_calls == 1 and report.input_tokens == 12 and report.output_tokens == 4
    assert result.usage.input_tokens == 12 and result.usage.output_tokens == 4
    assert result.total_cost == report.cost > 0


async def test_per_run_budget_policy_enforces_without_a_construction_budget():
    # A flow built with no budget can still be capped per run.
    llm = make_llm()

    async def call_model(snap: StateSnapshot) -> Result:
        await llm.complete("hi")
        return Result(value="ok")

    flow = Flow(
        Step(name="a", fn=call_model),
        Step(name="b", fn=call_model),
        name="per-run",
    )
    result = await flow.run(State(), budget_policy=BudgetPolicy(max_llm_calls=1))
    assert "budget_exceeded" in result.results
    assert result.trace.metadata["meter"]["llm_calls"] == 1


async def test_per_run_budget_policy_overrides_the_construction_one():
    # A stricter construction budget is loosened for a single run by the per-run override.
    llm = make_llm()

    async def call_model(snap: StateSnapshot) -> Result:
        await llm.complete("hi")
        return Result(value="ok")

    flow = Flow(
        Step(name="a", fn=call_model),
        Step(name="b", fn=call_model),
        name="override",
        budget_policy=BudgetPolicy(max_llm_calls=1),  # would deny the 2nd call
    )
    result = await flow.run(State(), budget_policy=BudgetPolicy(max_llm_calls=5))
    assert "budget_exceeded" not in result.results
    assert result.trace.metadata["meter"]["llm_calls"] == 2


async def test_nested_flow_budget_denial_surfaces_at_the_owner():
    # A nested (inherited-scope) flow re-raises AdmissionDenied so the OWNING flow converts it once
    # at the top — instead of burying budget_exceeded in the nested result and running on.
    llm = make_llm()

    async def call(snap: StateSnapshot) -> Result:
        await llm.complete("hi")
        return Result(value="x")

    inner = Flow(Step(name="inner_call", fn=call), name="inner")
    parent = Flow(inner, name="parent", budget_policy=BudgetPolicy(max_llm_calls=0))
    result = await parent.run(State())
    assert "budget_exceeded" in result.results  # surfaced in the parent's results
