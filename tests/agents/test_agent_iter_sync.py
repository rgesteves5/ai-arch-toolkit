"""Agent.iter_sync: the async iter()'s events and result, from synchronous code."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

from ai_arch_toolkit.core._metering._events import UsageEvent
from ai_arch_toolkit.core._metering._scope import RunConfig
from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.toolkit.agents import Agent, ReasoningSpec, SyncAgentExecution
from ai_arch_toolkit.toolkit.budget import BudgetPolicy
from tests.fake_provider import fake_llm

_MODEL = "claude-sonnet-4-6"


def _agent() -> Agent:
    answer = Response(text="answer", usage=Usage(input_tokens=10, output_tokens=5), model=_MODEL)
    llm, _ = fake_llm(answer, model=_MODEL)
    return Agent(ReasoningSpec(strategy="completion"), llm)


def test_the_events_stream_then_the_result_is_the_one_run_sync_gives() -> None:
    execution = _agent().iter_sync("hi")

    assert isinstance(execution, SyncAgentExecution)
    events = list(execution)
    reference = _agent().run_sync("hi")

    assert events[0].type == "flow_start"
    assert events[-1].type == "flow_end"
    tokens = [e.llm_event.text for e in events if e.type == "llm_event" and e.llm_event]
    assert "".join(tokens) == "answer"
    assert execution.result is not None
    assert (execution.result.text, execution.result.cost, execution.result.usage) == (
        reference.text,
        reference.cost,
        reference.usage,
    )


def test_the_result_is_none_until_the_run_has_finished() -> None:
    execution = _agent().iter_sync("hi")

    assert execution.result is None
    assert next(execution).type == "flow_start"
    assert execution.result is None
    for _event in execution:
        pass
    assert execution.result is not None
    assert execution.result is execution.result  # read once, kept


def test_a_run_config_reaches_the_run() -> None:
    class Sink:
        def __init__(self) -> None:
            self.events: list[UsageEvent] = []

        def emit(self, event: UsageEvent) -> None:
            self.events.append(event)

    sink = Sink()

    list(_agent().iter_sync("hi", config=RunConfig(sinks=[sink])))

    assert [event.kind for event in sink.events] == ["llm"]


def test_a_budget_policy_caps_the_run() -> None:
    execution = _agent().iter_sync("hi", budget_policy=BudgetPolicy(max_llm_calls=0))

    events = list(execution)

    assert not any(event.type == "llm_event" for event in events)
    assert execution.result is not None
    assert execution.result.text == ""
    assert execution.result.report is not None
    assert execution.result.report.llm_calls == 0


def _hanging_agent(log: list[str]) -> Agent:
    async def complete(*_args: Any, **_kwargs: Any) -> Response:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            log.append("cancelled")
            raise
        return Response(text="late")

    llm = AsyncMock()
    llm.complete = complete
    return Agent(ReasoningSpec(strategy="completion"), llm)


def test_close_stops_the_run() -> None:
    log: list[str] = []
    execution = _hanging_agent(log).iter_sync("hi")

    for event in execution:
        if event.type == "step_start":
            break
    execution.close()

    assert log == ["cancelled"]
    assert execution.result is None


def test_leaving_the_with_block_stops_the_run() -> None:
    log: list[str] = []

    with _hanging_agent(log).iter_sync("hi") as execution:
        for event in execution:
            if event.type == "step_start":
                break

    assert log == ["cancelled"]
    assert execution.result is None
