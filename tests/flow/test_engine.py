"""The single flow engine: run() and iter() share scheduling, isolation, events, and traces."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._metering._events import UsageEvent
from ai_arch_toolkit.core._metering._scope import RunConfig
from ai_arch_toolkit.core._policy import Policy
from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._retry import RetryConfig
from ai_arch_toolkit.core._state import State, StateSnapshot
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.toolkit.agents import Agent, ReasoningSpec
from ai_arch_toolkit.toolkit.budget import BudgetPolicy
from ai_arch_toolkit.toolkit.flow import Flow, FlowEvent, FlowResult, FlowStep, Scope
from tests.fake_provider import fake_llm

_MODEL = "claude-sonnet-4-6"  # priced in _default_pricing.toml


def _llm(text: str = "ok") -> LLM:
    usage = Usage(input_tokens=100_000, output_tokens=50_000)
    llm, _ = fake_llm(Response(text=text, usage=usage, model=_MODEL), model=_MODEL)
    return llm


class _Sink:
    def __init__(self) -> None:
        self.events: list[UsageEvent] = []

    def emit(self, event: UsageEvent) -> None:
        self.events.append(event)


def _recorder(name: str, ran: list[str], delay: float = 0.0, artifacts: Any = None) -> Step:
    async def fn(snap: StateSnapshot) -> Result:
        ran.append(name)
        if delay:
            await asyncio.sleep(delay)
        return Result(value=name, artifacts=artifacts if artifacts is not None else {name: True})

    return Step(name=name, fn=fn)


def _boom(*_args: Any) -> Any:
    raise ZeroDivisionError("boom")


# --------------------------------------------------------------------------------------- 8a


def _hook_kwargs(hook: str) -> dict[str, Any]:
    if hook == "when":
        return {"when": _boom}
    if hook == "transform":
        return {"scope": Scope(transform={"x": _boom})}
    return {"scope": Scope(enrich={"y": _boom})}


def _flow_with_failing_hook(mode: str, hook: str, ran: list[str]) -> Flow:
    a, b, b2, c = (_recorder(n, ran) for n in ("a", "b", "b2", "c"))
    kw = _hook_kwargs(hook)
    if mode == "linear":
        return Flow(FlowStep(step=a), FlowStep(step=b, **kw), FlowStep(step=c), name="f")
    if mode == "cyclic":
        return Flow(
            FlowStep(step=a, when=lambda s: not s.get("a")),
            FlowStep(step=b, **kw),
            FlowStep(step=c),
            name="f",
            max_iterations=3,
        )
    if mode == "dag_single":
        return Flow(
            FlowStep(step=a),
            FlowStep(step=b, after=("a",), **kw),
            FlowStep(step=c, after=("b",)),
            name="f",
        )
    return Flow(  # dag_parallel: b and b2 form one wave
        FlowStep(step=a),
        FlowStep(step=b, after=("a",), **kw),
        FlowStep(step=b2, after=("a",)),
        FlowStep(step=c, after=("b", "b2")),
        name="f",
    )


_HOOK_CASES = [
    ("linear", "transform"),
    ("linear", "enrich"),
    ("cyclic", "when"),
    ("cyclic", "transform"),
    ("cyclic", "enrich"),
    ("dag_single", "when"),
    ("dag_single", "transform"),
    ("dag_single", "enrich"),
    ("dag_parallel", "when"),
    ("dag_parallel", "transform"),
    ("dag_parallel", "enrich"),
]


@pytest.mark.parametrize(("mode", "hook"), _HOOK_CASES)
async def test_orchestration_errors_become_trace_errors_and_halt_run(mode: str, hook: str) -> None:
    ran: list[str] = []
    flow = _flow_with_failing_hook(mode, hook, ran)

    result = await flow.run(State(operational={"x": 1}))

    errors = {step.name: step.error for step in result.trace.steps if step.error}
    assert "b" in errors
    assert ("condition" if hook == "when" else "scope") in errors["b"]
    assert "boom" in errors["b"]
    assert "b" not in ran
    assert "c" not in ran


@pytest.mark.parametrize(("mode", "hook"), _HOOK_CASES)
async def test_orchestration_errors_are_reported_by_iter(mode: str, hook: str) -> None:
    ran: list[str] = []
    flow = _flow_with_failing_hook(mode, hook, ran)

    events = [event async for event in flow.iter(State(operational={"x": 1}))]

    failed = [e for e in events if e.type == "step_end" and e.step_name == "b"]
    assert failed and failed[0].error is not None and "boom" in failed[0].error
    assert events[-1].type == "flow_end"
    assert "c" not in ran


# --------------------------------------------------------------------------------------- 8b


async def test_meter_scope_is_not_stored_in_the_state() -> None:
    state = State()

    result = await Flow(_recorder("a", []), name="f").run(state)

    assert "_meter_scope" not in state.world
    assert result.meter_scope is not None
    assert result.meter is not None


async def test_full_debug_trace_is_json_serializable() -> None:
    result = await Flow(_recorder("a", []), name="f").run(State(world={"region": "eu"}))

    json.dumps(result.trace.to_dict(trace_mode="full_debug"))


# --------------------------------------------------------------------------------------- 8c


def _isolation_flow(seen: dict[str, Any]) -> Flow:
    async def writer(snap: StateSnapshot) -> Result:
        await asyncio.sleep(0.01)
        return Result(artifacts={"written_by_a": "A"})

    async def reader(snap: StateSnapshot) -> Result:
        await asyncio.sleep(0.05)
        seen["b_saw"] = snap.get("written_by_a", "<nothing>")
        return Result()

    async def join(snap: StateSnapshot) -> Result:
        return Result()

    return Flow(
        FlowStep(step=Step(name="a", fn=writer)),
        FlowStep(step=Step(name="b", fn=reader)),
        FlowStep(step=Step(name="join", fn=join), after=("a", "b")),
        name="dag",
    )


async def test_iter_keeps_parallel_siblings_isolated_like_run() -> None:
    via_run: dict[str, Any] = {}
    via_iter: dict[str, Any] = {}

    await _isolation_flow(via_run).run(State())
    async for _ in _isolation_flow(via_iter).iter(State()):
        pass

    assert via_run["b_saw"] == "<nothing>"
    assert via_iter["b_saw"] == "<nothing>"


async def test_iter_runs_a_dag_wave_in_parallel() -> None:
    ran: list[str] = []
    flow = Flow(
        *[FlowStep(step=_recorder(f"s{i}", ran, delay=0.2)) for i in range(4)],
        FlowStep(step=_recorder("join", ran), after=tuple(f"s{i}" for i in range(4))),
        name="wave",
        max_parallelism=2,
    )

    started = time.monotonic()
    async for _ in flow.iter(State()):
        pass

    assert time.monotonic() - started < 0.7  # two at a time: ~0.4 s, sequential would be 0.8 s


async def test_iter_exposes_the_flow_result_after_draining() -> None:
    execution = Flow(_recorder("a", []), name="f").iter(State())

    assert execution.result is None
    events = [event async for event in execution]

    assert isinstance(execution.result, FlowResult)
    assert events[-1].trace is execution.result.trace
    assert execution.result.results["a"].value == "a"


def test_iter_sync_exposes_the_flow_result() -> None:
    execution = Flow(_recorder("a", []), name="f").iter_sync(State())

    events = list(execution)

    assert events[-1].type == "flow_end"
    assert execution.result is not None
    assert execution.result.results["a"].value == "a"


async def test_breaking_out_of_iter_without_aclose_stops_the_flow() -> None:
    ran: list[str] = []
    sink = _Sink()
    llm = _llm()

    async def open_stream(snap: StateSnapshot) -> Result:
        ran.append("a")
        llm.stream("hi")  # a metered op that only the scope's close can finalize
        return Result(value="a")

    flow = Flow(Step(name="a", fn=open_stream), _recorder("b", ran), name="f")

    async for event in flow.iter(State(), config=RunConfig(sinks=[sink])):
        if event.type == "step_end":
            break
    await asyncio.sleep(0.3)  # the loop finalizes the abandoned run

    assert ran == ["a"]
    assert sink.events, "the abandoned run's scope was never closed"
    assert sink.events[-1].status in {"incomplete", "aborted"}


async def test_retry_event_arrives_while_the_step_is_still_running() -> None:
    attempts = 0

    async def flaky(snap: StateSnapshot) -> Result:
        nonlocal attempts
        attempts += 1
        return Result(error="try again") if attempts == 1 else Result(value="ok")

    policy = Policy(retry=RetryConfig(max_retries=1, base_delay=0.4))
    flow = Flow(Step(name="flaky", fn=flaky, policy=policy), name="f")

    arrivals: dict[str, float] = {}
    async for event in flow.iter(State()):
        arrivals.setdefault(event.type, time.monotonic())

    assert arrivals["step_end"] - arrivals["retry"] >= 0.3


async def test_agent_iter_exposes_the_agent_result() -> None:
    agent = Agent(ReasoningSpec(strategy="completion"), _llm("final answer"))

    execution = agent.iter("hi")
    events = [event async for event in execution]

    assert events[-1].type == "flow_end"
    assert execution.result is not None
    assert execution.result.text == "final answer"


# --------------------------------------------------------------------------------------- 8d


async def _event_types(flow: Flow) -> list[FlowEvent]:
    return [event async for event in flow.iter(State())]


async def test_step_timeout_is_streamed_as_a_timeout_event() -> None:
    flow = Flow(
        Step(name="slow", fn=_recorder("slow", [], delay=1.0).fn, policy=Policy(timeout=0.01)),
        name="f",
    )

    events = await _event_types(flow)

    timeouts = [e for e in events if e.type == "timeout"]
    assert timeouts and timeouts[0].step_name == "slow"


async def test_fallback_is_streamed_as_a_fallback_event() -> None:
    fallback = _recorder("cheap", [])
    policy = Policy(timeout=0.01, on_timeout="fallback", fallback=fallback)
    flow = Flow(Step(name="slow", fn=_recorder("slow", [], delay=1.0).fn, policy=policy), name="f")

    types = [e.type for e in await _event_types(flow)]

    assert types.index("timeout") < types.index("fallback") < types.index("step_end")


async def test_low_confidence_is_streamed_as_policy_decisions() -> None:
    async def unsure(snap: StateSnapshot) -> Result:
        return Result(value="maybe", confidence=0.1)

    policy = Policy(confidence_threshold=0.9, on_low_confidence="escalate")
    flow = Flow(Step(name="unsure", fn=unsure, policy=policy), name="f")

    decisions = [
        e.policy_decision for e in await _event_types(flow) if e.type == "policy_decision"
    ]

    assert decisions == ["low_confidence", "escalate"]


# --------------------------------------------------------------------------------------- 8e


async def test_nested_flow_trace_is_linked_as_children() -> None:
    inner = Flow(_recorder("a", []), _recorder("b", []), name="inner")
    outer = Flow(inner, _recorder("c", []), name="outer")

    result = await outer.run(State())

    nested = result.trace.flow("inner")
    assert nested is not None
    assert [child.name for child in nested.children] == ["a", "b"]


async def test_flow_run_inside_a_step_is_linked_as_children() -> None:
    inner = Flow(_recorder("a", []), name="inner")

    async def wrapper(snap: StateSnapshot) -> Result:
        await inner.run(State())
        return Result(value="done")

    result = await Flow(Step(name="wrapper", fn=wrapper), name="outer").run(State())

    step = result.trace.step("wrapper")
    assert step is not None
    assert [child.name for child in step.children] == ["inner"]
    assert [child.name for child in step.children[0].children] == ["a"]


async def test_reflexion_trace_exposes_its_inner_react_run() -> None:
    agent = Agent(ReasoningSpec(strategy="reflexion"), _llm("an answer"))

    result = await agent.run("task")

    attempt = result.flow_result.trace.step("attempt")
    assert attempt is not None
    inner = [child for child in attempt.children if child.name == "react"]
    assert inner and any(step.name == "llm_call" for step in inner[0].children)


# --------------------------------------------------------------------------------------- 8f


@pytest.mark.parametrize("denied_position", [0, 1, 2])
async def test_denial_in_a_parallel_wave_keeps_the_finished_siblings(denied_position: int) -> None:
    llm = _llm()

    async def call_model(snap: StateSnapshot) -> Result:
        await llm.complete("hi")
        return Result(value="unreachable")

    steps: list[Step] = []
    for index in range(3):
        if index == denied_position:
            steps.append(Step(name=f"s{index}", fn=call_model))
        else:
            steps.append(_recorder(f"s{index}", [], delay=0.05))
    flow = Flow(
        *[FlowStep(step=step) for step in steps],
        FlowStep(step=_recorder("join", []), after=tuple(step.name for step in steps)),
        name="wave",
    )
    state = State()

    result = await flow.run(state, budget_policy=BudgetPolicy(max_llm_calls=0))

    siblings = [f"s{index}" for index in range(3) if index != denied_position]
    traced = {step.name for step in result.trace.steps}
    assert set(siblings) <= traced
    assert all(state.get(name) is True for name in siblings)
    assert "budget_exceeded" in result.results
    assert "join" not in traced


# --------------------------------------------------------------------------------------- 8g


async def test_step_max_cost_counts_spend_inside_a_nested_flow() -> None:
    llm = _llm()

    async def call(snap: StateSnapshot) -> Result:
        await llm.complete("hi")
        return Result(value="done")

    inner = Flow(Step(name="inner_call", fn=call), name="inner")
    capped = Step(name="wrapper", fn=inner.as_step().fn, policy=Policy(max_cost=0.0001))

    result = await Flow(capped, name="outer").run(State())

    wrapper = result.trace.step("wrapper")
    assert wrapper is not None
    assert wrapper.error is not None and "Cost exceeded" in wrapper.error


def _sleeper(name: str, delay: float, log: list[str]) -> Step:
    async def fn(snap: StateSnapshot) -> Result:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            log.append(f"{name} cancelled")
            raise
        log.append(f"{name} done")
        return Result(value=name, artifacts={name: True})

    return Step(name=name, fn=fn)


async def test_a_timeout_in_a_parallel_wave_keeps_the_siblings_that_finished() -> None:
    log: list[str] = []
    flow = Flow(
        FlowStep(step=_sleeper("a", 0.01, log)),
        FlowStep(step=_sleeper("b", 10.0, log)),
        FlowStep(step=_sleeper("join", 0.0, log), after=("a", "b")),
        name="dag",
        timeout=0.3,
    )
    state = State()

    result = await flow.run(state)

    assert result.results["a"].value == "a"
    assert state["a"] is True
    assert [st.name for st in result.trace.steps] == ["a", "flow_timeout"]
    assert log == ["a done", "b cancelled"]


async def test_the_timeout_event_arrives_after_the_steps_in_flight_were_cancelled() -> None:
    log: list[str] = []
    flow = Flow(_sleeper("slow", 10.0, log), name="f", timeout=0.1)
    cancelled_by_then: list[str] | None = None

    async with flow.iter(State()) as execution:
        async for event in execution:
            if event.type == "timeout":
                cancelled_by_then = list(log)
                break

    assert cancelled_by_then == ["slow cancelled"]


async def test_events_after_the_deadline_do_not_postpone_the_timeout() -> None:
    async def always_fails(snap: StateSnapshot) -> Result:
        raise RuntimeError("again")

    retries = Policy(retry=RetryConfig(max_retries=1_000, base_delay=1e-9, max_delay=1e-9))
    flow = Flow(Step(name="busy", fn=always_fails, policy=retries), name="f", timeout=0.01)

    result = await flow.run(State())

    assert result.trace.steps[-1].name == "flow_timeout"


async def test_a_cyclic_flow_with_max_iterations_zero_runs_no_pass() -> None:
    ran: list[str] = []
    flow = Flow(
        FlowStep(step=_recorder("a", ran), when=lambda s: True), name="f", max_iterations=0
    )

    await flow.run(State())

    assert ran == []


async def test_a_held_execution_keeps_running_after_break_until_it_is_closed() -> None:
    log: list[str] = []
    flow = Flow(_sleeper("slow", 10.0, log), name="f")

    async with flow.iter(State()) as execution:
        async for event in execution:
            if event.type == "step_start":
                break
        await asyncio.sleep(0.05)
        assert log == []  # break alone does not stop a run that is still referenced

    assert log == ["slow cancelled"]


def test_a_sync_execution_used_as_a_context_manager_stops_the_run_on_exit() -> None:
    log: list[str] = []
    flow = Flow(_sleeper("slow", 10.0, log), name="f")

    with flow.iter_sync(State()) as execution:
        for event in execution:
            if event.type == "step_start":
                break

    assert log == ["slow cancelled"]


async def test_a_second_cancellation_during_cleanup_still_closes_the_meter_scope() -> None:
    async def slow_to_cancel(snap: StateSnapshot) -> Result:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            await asyncio.sleep(0.2)  # slow cleanup
            raise
        return Result()

    execution = Flow(Step(name="s", fn=slow_to_cancel), name="f").iter(State())

    async def drive() -> None:
        async for _ in execution:
            pass

    driver = asyncio.create_task(drive())
    await asyncio.sleep(0.05)
    scope = execution.meter_scope
    assert scope is not None
    closed: list[bool] = []
    original_close = scope.close

    def recording_close() -> None:
        closed.append(True)
        original_close()

    scope.close = recording_close  # type: ignore[method-assign]
    driver.cancel()
    await asyncio.sleep(0.05)  # the engine's cleanup is waiting for the step to finish cancelling
    driver.cancel()
    with pytest.raises(asyncio.CancelledError):
        await driver
    await asyncio.sleep(0.3)  # let the step's own cleanup finish

    assert closed == [True]


def test_sync_flow_execution_is_exported_like_flow_execution() -> None:
    import ai_arch_toolkit
    from ai_arch_toolkit import toolkit
    from ai_arch_toolkit.toolkit.flow import SyncFlowExecution

    for module in (ai_arch_toolkit, toolkit):
        assert getattr(module, "SyncFlowExecution", None) is SyncFlowExecution
        assert "SyncFlowExecution" in module.__all__


@pytest.mark.parametrize("mode", ["sequential", "dag"])
async def test_wall_budget_emits_completed_step_before_denial(mode: str) -> None:
    flow = Flow(
        FlowStep(step=_recorder("slow", [], delay=0.1)),
        FlowStep(step=_recorder("next", []), after=("slow",) if mode == "dag" else ()),
    )
    events = [
        event async for event in flow.iter(State(), budget_policy=BudgetPolicy(max_wall_s=0.05))
    ]
    kinds = [event.type for event in events]
    assert "step_end" in kinds
    end = next(event for event in events if event.type == "step_end")
    assert end.step_name == "slow"
    denied = next(
        i for i, event in enumerate(events) if event.policy_decision == "budget_exceeded"
    )
    assert kinds.index("step_end") < denied
