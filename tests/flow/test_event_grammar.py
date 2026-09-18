"""Every flow event stream follows one grammar and tells the trace's story, whatever stops it."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from ai_arch_toolkit.core._policy import Policy
from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._retry import RetryConfig
from ai_arch_toolkit.core._state import State, StateSnapshot
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.toolkit.budget import BudgetPolicy
from ai_arch_toolkit.toolkit.flow import Flow, FlowEvent, FlowResult, FlowStep, Scope
from tests.fake_provider import fake_llm
from tests.flow.event_grammar import check_grammar

_MODEL = "claude-sonnet-4-6"


def _step(name: str, *, delay: float = 0.0, error: str | None = None) -> Step:
    async def fn(snap: StateSnapshot) -> Result:
        await asyncio.sleep(delay)
        if error is not None:
            return Result(error=error)
        return Result(value=name, artifacts={name: True})

    return Step(name=name, fn=fn)


def _calls_model(name: str) -> Step:
    llm, _ = fake_llm(Response(text="ok", usage=Usage(input_tokens=1), model=_MODEL), model=_MODEL)

    async def fn(snap: StateSnapshot) -> Result:
        await llm.complete("hi")
        return Result(value=name)

    return Step(name=name, fn=fn)


def _boom(*_args: Any) -> Any:
    raise ZeroDivisionError("boom")


def _flaky(name: str) -> Step:
    calls = [0]

    async def fn(snap: StateSnapshot) -> Result:
        calls[0] += 1
        if calls[0] == 1:
            raise RuntimeError("first call fails")
        return Result(value=name)

    retry = Policy(retry=RetryConfig(max_retries=1, base_delay=1e-9, max_delay=1e-9))
    return Step(name=name, fn=fn, policy=retry)


def _timing_out_with_fallback(name: str) -> Step:
    async def slow(snap: StateSnapshot) -> Result:
        await asyncio.sleep(1)
        return Result(value="late")

    policy = Policy(timeout=0.01, on_timeout="fallback", fallback=_step("fallback"))
    return Step(name=name, fn=slow, policy=policy)


_Scenario = Callable[[], tuple[Flow, dict[str, Any]]]


def _sequential(*steps: Step | FlowStep, **kw: Any) -> Flow:
    return Flow(*steps, name="sequential", **kw)


def _dag(*steps: FlowStep, **kw: Any) -> Flow:
    return Flow(*steps, name="dag", **kw)


def _after(step: Step, *deps: str, **kw: Any) -> FlowStep:
    return FlowStep(step=step, after=deps, **kw)


_SCENARIOS: dict[str, _Scenario] = {
    "sequential, finishes": lambda: (_sequential(_step("a"), _step("b")), {}),
    "sequential, an error halts": lambda: (
        _sequential(_step("a", error="nope"), _step("b")),
        {},
    ),
    "sequential, a condition skips": lambda: (
        _sequential(
            _step("a"),
            FlowStep(step=_step("b"), when=lambda s: False),
            _step("c"),
            max_iterations=1,
        ),
        {},
    ),
    "sequential, decisions inside steps": lambda: (
        _sequential(_flaky("a"), _timing_out_with_fallback("b")),
        {},
    ),
    "sequential, a scope raises": lambda: (
        _sequential(_step("a"), FlowStep(step=_step("b"), scope=Scope(transform={"x": _boom}))),
        {},
    ),
    "sequential, wall time runs out": lambda: (
        _sequential(_step("slow", delay=0.1), _step("b")),
        {"budget_policy": BudgetPolicy(max_wall_s=0.05)},
    ),
    "sequential, the meter denies": lambda: (
        _sequential(_step("a"), _calls_model("b"), _step("c")),
        {"budget_policy": BudgetPolicy(max_llm_calls=0)},
    ),
    "sequential, the flow times out": lambda: (
        _sequential(_step("a"), _step("slow", delay=10), timeout=0.1),
        {},
    ),
    "cyclic, two passes": lambda: (
        _sequential(
            FlowStep(step=_step("a"), when=lambda s: True),
            _step("b"),
            max_iterations=2,
        ),
        {},
    ),
    "dag, a chain": lambda: (
        _dag(_after(_step("a")), _after(_step("b"), "a"), _after(_step("c"), "b")),
        {},
    ),
    "dag, siblings finish out of order": lambda: (
        _dag(
            _after(_step("slow", delay=0.05)),
            _after(_step("fast", delay=0.01)),
            _after(_step("join"), "slow", "fast"),
        ),
        {},
    ),
    "dag, a failure skips its dependents": lambda: (
        _dag(
            _after(_step("a", error="nope")),
            _after(_step("b")),
            _after(_step("c"), "a"),
            _after(_step("d"), "b"),
        ),
        {},
    ),
    "dag, a condition raises in a wave": lambda: (
        _dag(
            _after(_step("a")),
            _after(_step("b"), "a", when=_boom),
            _after(_step("b2"), "a"),
        ),
        {},
    ),
    "dag, the meter denies mid-wave": lambda: (
        _dag(
            _after(_step("a", delay=0.02)),
            _after(_calls_model("b")),
            _after(_step("join"), "a", "b"),
        ),
        {"budget_policy": BudgetPolicy(max_llm_calls=0)},
    ),
    "dag, the flow times out mid-wave": lambda: (
        _dag(
            _after(_step("fast", delay=0.01)),
            _after(_step("slow", delay=10)),
            _after(_step("join"), "fast", "slow"),
            timeout=0.2,
        ),
        {},
    ),
    "dag, wall time runs out after a wave": lambda: (
        _dag(
            _after(_step("a", delay=0.1)),
            _after(_step("b", delay=0.1)),
            _after(_step("join"), "a", "b"),
        ),
        {"budget_policy": BudgetPolicy(max_wall_s=0.05)},
    ),
}


async def _drain(flow: Flow, **kw: Any) -> tuple[list[FlowEvent], FlowResult]:
    events: list[FlowEvent] = []
    execution = flow.iter(State(), **kw)
    async for event in execution:
        events.append(event)
    assert execution.result is not None
    return events, execution.result


@pytest.mark.parametrize("scenario", sorted(_SCENARIOS))
async def test_the_stream_follows_the_grammar_and_agrees_with_the_trace(scenario: str) -> None:
    flow, kw = _SCENARIOS[scenario]()

    events, result = await _drain(flow, **kw)

    check_grammar(events, result.trace)


def test_the_checker_catches_a_step_that_never_ends_and_a_story_the_trace_does_not_tell() -> None:
    start = FlowEvent(type="flow_start", flow_name="f")
    end = FlowEvent(type="flow_end", flow_name="f")
    open_step = FlowEvent(type="step_start", flow_name="f", step_name="a")
    with pytest.raises(AssertionError, match="never closed"):
        check_grammar([start, open_step, end], _trace())
    closed = FlowEvent(type="step_end", flow_name="f", step_name="a")
    with pytest.raises(AssertionError, match="trace says"):
        check_grammar([start, open_step, closed, end], _trace())


def _trace() -> Any:
    from ai_arch_toolkit.core._trace import Trace

    return Trace(flow_name="f", steps=())
