"""Flow-level policy and run timeout, whether the flow runs directly, nested, or streamed."""

from __future__ import annotations

import asyncio
import math
import time

import pytest

from ai_arch_toolkit.core._policy import Policy
from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._retry import RetryConfig
from ai_arch_toolkit.core._state import State, StateSnapshot
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.toolkit.flow._flow import Flow, FlowStep
from tests.fake_provider import Reply, fake_llm

_MODEL = "claude-sonnet-4-6"  # priced in _default_pricing.toml


def _sleeper(name: str, seconds: float, ran: list[str] | None = None) -> Step:
    async def fn(snap: StateSnapshot) -> Result:
        if ran is not None:
            ran.append(name)
        await asyncio.sleep(seconds)
        return Result(value=name, artifacts={name: True})

    return Step(name=name, fn=fn)


def _failing(name: str, calls: list[str] | None = None) -> Step:
    async def fn(snap: StateSnapshot) -> Result:
        if calls is not None:
            calls.append(name)
        return Result(error=f"{name} failed")

    return Step(name=name, fn=fn)


class TestFlowPolicyIsTheDefaultForEachStep:
    async def test_flow_policy_applies_when_the_flow_runs_directly(self) -> None:
        flow = Flow(_sleeper("slow", 0.5), name="f", policy=Policy(timeout=0.01))

        result = await flow.run(State())

        step = result.trace.steps[0]
        assert step.error == "Step timed out"
        assert "timeout" in step.policy_decisions

    async def test_step_policy_wins_over_flow_policy(self) -> None:
        step = Step(name="s", fn=_sleeper("s", 0.05).fn, policy=Policy(timeout=5.0))
        flow = Flow(step, name="f", policy=Policy(timeout=0.001))

        result = await flow.run(State())

        assert result.trace.steps[0].error is None

    async def test_inherited_on_exhausted_continue_keeps_the_flow_going(self) -> None:
        ran: list[str] = []
        flow = Flow(
            _failing("a"),
            _sleeper("b", 0, ran),
            name="f",
            policy=Policy(on_exhausted="continue"),
        )

        await flow.run(State())

        assert ran == ["b"]

    async def test_errors_still_halt_the_flow_by_default(self) -> None:
        ran: list[str] = []
        flow = Flow(_failing("a"), _sleeper("b", 0, ran), name="f", policy=Policy())

        await flow.run(State())

        assert ran == []

    async def test_inherited_retry_retries_each_failing_step(self) -> None:
        calls: list[str] = []
        flow = Flow(
            _failing("a", calls),
            name="f",
            policy=Policy(retry=RetryConfig(max_retries=2, base_delay=0.001)),
        )

        result = await flow.run(State())

        assert calls == ["a", "a", "a"]
        assert result.trace.steps[0].attempts == 3

    async def test_nested_flow_policy_is_applied_once_by_the_nested_flow(self) -> None:
        calls: list[str] = []
        inner = Flow(
            _failing("count", calls),
            name="inner",
            policy=Policy(retry=RetryConfig(max_retries=1, base_delay=0.001)),
        )
        outer = Flow(inner, name="outer")

        result = await outer.run(State())

        assert calls == ["count", "count"]  # retried inside, not again around the wrapper
        assert result.trace.steps[0].attempts == 1


class TestFlowTimeout:
    @pytest.mark.parametrize("bad", [0, -1.0, math.nan, math.inf])
    def test_timeout_must_be_positive_and_finite(self, bad: float) -> None:
        with pytest.raises(ValueError, match="timeout"):
            Flow(_sleeper("a", 0), timeout=bad)

    def test_timeout_is_exposed(self) -> None:
        assert Flow(_sleeper("a", 0), timeout=3.0).timeout == 3.0
        assert Flow(_sleeper("a", 0)).timeout is None

    async def test_timeout_bounds_the_whole_run(self) -> None:
        ran: list[str] = []
        flow = Flow(
            _sleeper("fast", 0, ran),
            _sleeper("slow", 10.0, ran),
            _sleeper("never", 0, ran),
            name="f",
            timeout=0.2,
        )

        started = time.monotonic()
        result = await flow.run(State())

        assert time.monotonic() - started < 3.0
        assert ran == ["fast", "slow"]
        last = result.trace.steps[-1]
        assert last.name == "flow_timeout"
        assert last.error is not None and "timed out" in last.error
        assert last.policy_decisions == ("timeout",)
        assert result.final_result is not None and result.final_result.is_error

    async def test_timeout_is_checked_between_steps(self) -> None:
        ran: list[str] = []
        flow = Flow(
            *[_sleeper(f"s{i}", 0.05, ran) for i in range(40)],
            name="f",
            timeout=0.2,
        )

        started = time.monotonic()
        result = await flow.run(State())

        assert time.monotonic() - started < 3.0
        assert 0 < len(ran) < 40
        assert result.trace.steps[-1].name == "flow_timeout"

    async def test_iter_reports_the_timeout(self) -> None:
        flow = Flow(_sleeper("slow", 10.0), name="f", timeout=0.1)

        events = [event async for event in flow.iter(State())]

        types = [event.type for event in events]
        assert "timeout" in types
        assert types[-1] == "flow_end"
        assert events[-1].trace is not None
        assert events[-1].trace.steps[-1].name == "flow_timeout"

    async def test_timeout_cancels_parallel_dag_steps(self) -> None:
        ran: list[str] = []
        flow = Flow(
            FlowStep(step=_sleeper("a", 10.0, ran)),
            FlowStep(step=_sleeper("b", 10.0, ran)),
            FlowStep(step=_sleeper("join", 0, ran), after=("a", "b")),
            name="dag",
            timeout=0.2,
        )

        started = time.monotonic()
        result = await flow.run(State())

        assert time.monotonic() - started < 3.0
        assert "join" not in ran
        assert result.trace.steps[-1].name == "flow_timeout"

    async def test_iter_timeout_cancels_parallel_dag_steps(self) -> None:
        flow = Flow(
            FlowStep(step=_sleeper("a", 10.0)),
            FlowStep(step=_sleeper("b", 10.0)),
            name="dag",
            timeout=0.2,
        )

        started = time.monotonic()
        types = [event.type async for event in flow.iter(State())]

        assert time.monotonic() - started < 3.0
        assert "timeout" in types

    async def test_timeout_fails_the_in_flight_llm_call_in_the_meter(self) -> None:
        late = Response(text="late", usage=Usage(input_tokens=10), model=_MODEL)
        llm, _ = fake_llm(Reply(response=late, delay=10.0), model=_MODEL)

        async def call(snap: StateSnapshot) -> Result:
            await llm.complete("hi")
            return Result()

        flow = Flow(Step(name="call", fn=call), name="f", timeout=0.1)

        result = await flow.run(State())

        report = result.meter
        assert report is not None
        assert report.llm_calls == 1
        assert report.cost_uncertain  # the call reached the provider; its cost is unknown

    async def test_nested_flow_timeout_is_handled_by_the_nested_flow(self) -> None:
        inner = Flow(_sleeper("slow", 10.0), name="inner", timeout=0.1)
        outer = Flow(inner, name="outer", timeout=5.0)

        started = time.monotonic()
        result = await outer.run(State())

        assert time.monotonic() - started < 3.0
        wrapper = result.trace.steps[0]
        assert wrapper.name == "inner"
        assert wrapper.error is not None and "timed out" in wrapper.error
        assert all(step.name != "flow_timeout" for step in result.trace.steps)

    async def test_outer_timeout_interrupts_a_nested_flow(self) -> None:
        inner = Flow(_sleeper("slow", 10.0), name="inner")
        outer = Flow(inner, name="outer", timeout=0.1)

        started = time.monotonic()
        result = await outer.run(State())

        assert time.monotonic() - started < 3.0
        assert result.trace.steps[-1].name == "flow_timeout"
