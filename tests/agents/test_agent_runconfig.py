"""Agent.run / run_sync / iter accept a full RunConfig per run, like Flow.run."""

from __future__ import annotations

from ai_arch_toolkit.core._metering._events import UsageEvent
from ai_arch_toolkit.core._metering._scope import RunConfig
from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.toolkit.agents import Agent, ReasoningSpec
from ai_arch_toolkit.toolkit.budget import BudgetController, BudgetPolicy
from tests.fake_provider import fake_llm

_MODEL = "claude-sonnet-4-6"


class _Sink:
    def __init__(self) -> None:
        self.events: list[UsageEvent] = []

    def emit(self, event: UsageEvent) -> None:
        self.events.append(event)


def _agent() -> Agent:
    answer = Response(text="answer", usage=Usage(input_tokens=10, output_tokens=5), model=_MODEL)
    llm, _ = fake_llm(answer, model=_MODEL)
    return Agent(ReasoningSpec(strategy="completion"), llm)


async def test_run_config_sinks_receive_the_run_events() -> None:
    sink = _Sink()

    result = await _agent().run("hi", config=RunConfig(sinks=[sink]))

    assert result.text == "answer"
    assert [event.kind for event in sink.events] == ["llm"]


async def test_run_config_takes_precedence_over_budget_policy() -> None:
    deny_all = RunConfig(controller=BudgetController(BudgetPolicy(max_llm_calls=0)))

    result = await _agent().run(
        "hi", budget_policy=BudgetPolicy(max_llm_calls=10), config=deny_all
    )

    assert result.text == ""
    assert any("llm_calls" in error for error in result.errors)


def test_run_sync_accepts_a_run_config() -> None:
    sink = _Sink()

    _agent().run_sync("hi", config=RunConfig(sinks=[sink]))

    assert len(sink.events) == 1


async def test_iter_accepts_a_run_config() -> None:
    sink = _Sink()

    events = [event async for event in _agent().iter("hi", config=RunConfig(sinks=[sink]))]

    assert events[-1].type == "flow_end"
    assert len(sink.events) == 1
