"""Each strategy leaves its answer under ``answer`` and the response behind it under ``response``.

The runner reads those two, or the last step's value when a flow leaves no answer (D35).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock

import pytest

from ai_arch_toolkit.core._response import Response, ToolCall
from ai_arch_toolkit.core._state import State
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.core._tools._decorator import tool
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.core._trace import StepTrace, Trace
from ai_arch_toolkit.toolkit.agents import Agent, AgentResult, ReasoningSpec, extract_text
from ai_arch_toolkit.toolkit.agents.flows._keys import ANSWER, MESSAGES, RESPONSE, TASK
from ai_arch_toolkit.toolkit.flow._flow import Flow, FlowResult

_STRATEGIES = (
    "react",
    "completion",
    "plan_execute",
    "rewoo",
    "reflexion",
    "generate_review",
    "self_discovery",
    "llm_compiler",
    "tot",
    "lats",
)


@tool
def lookup(query: str) -> str:
    """Look something up."""
    return f"found {query}"


def _numbered_llm(text: str) -> AsyncMock:
    """A model that answers ``text`` and the call's number, as a new ``Response`` each time.

    ``text`` is read by every phase: ACCEPT ends a review, "1." is a plan step and a score, and
    "$1. ... [deps: none]" is a one-task DAG.
    """
    calls = 0

    async def complete(*_args: Any, **_kwargs: Any) -> Response:
        nonlocal calls
        calls += 1
        return Response(text=f"{text}\n(call {calls})")

    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=complete)
    return llm


_ANSWERS = "ACCEPT\n1. Do the thing\n$1. Do the thing [deps: none]"


def _kept(result: AgentResult) -> tuple[Any, Any]:
    state = result.flow_result.state
    return state.get(ANSWER), state.get(RESPONSE)


def test_the_shared_keys_are_the_documented_names() -> None:
    assert (TASK, MESSAGES, ANSWER, RESPONSE) == ("task", "messages", "answer", "response")


@pytest.mark.parametrize("strategy", _STRATEGIES)
async def test_a_strategy_leaves_its_answer_and_the_response_behind_it(strategy: str) -> None:
    agent = Agent(ReasoningSpec(strategy=strategy, max_iterations=2), _numbered_llm(_ANSWERS))

    result = await agent.run("task")

    answer, response = _kept(result)
    assert isinstance(response, Response)
    assert answer == response.text == result.text
    assert result.response is response
    assert result.text.startswith("ACCEPT")


async def test_a_reflexion_that_never_passes_leaves_its_last_attempt() -> None:
    agent = Agent(
        ReasoningSpec(strategy="reflexion", knobs={"max_retries": 2}),
        _numbered_llm("attempt"),
        deps={"evaluator": lambda _task, _answer: 0.0},
    )

    result = await agent.run("task")

    answer, response = _kept(result)
    assert isinstance(response, Response)
    assert answer == response.text == result.text
    assert result.response is response
    assert result.flow_result.state["passed"] is False


async def test_a_generate_review_that_never_accepts_leaves_its_last_draft() -> None:
    agent = Agent(
        ReasoningSpec(strategy="generate_review", knobs={"max_cycles": 2}),
        _numbered_llm("draft; RETRY with more detail"),
    )

    result = await agent.run("task")

    answer, response = _kept(result)
    assert isinstance(response, Response)
    assert answer == response.text == result.text
    assert result.response is response
    assert result.flow_result.state["accepted"] is False


async def test_a_react_run_that_ends_on_a_tool_turn_answers_with_that_turns_text() -> None:
    turn = Response(text="", tool_calls=[ToolCall(id="1", name="lookup", input={"query": "x"})])
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=turn)
    spec = ReasoningSpec(strategy="react", max_iterations=1, knobs={"final_answer_hint": False})

    result = await Agent(spec, llm, ToolGroup(lookup)).run("task")

    assert _kept(result) == ("", turn)
    assert result.text == ""
    assert result.response is turn


def _finished(state: State, value: Any) -> FlowResult:
    """A run whose last step returned ``value``."""
    trace = Trace(flow_name="hand_built", steps=(StepTrace(name="last"),))
    return FlowResult(state=state, trace=trace, results={"last": Result(value=value)})


def test_the_answer_is_read_from_answer_alone() -> None:
    state = State(operational={RESPONSE: Response(text="from response"), ANSWER: "the answer"})

    assert extract_text(state, _finished(state, "last value")) == "the answer"


def test_an_empty_answer_is_the_answer() -> None:
    state = State(operational={ANSWER: "", RESPONSE: Response(text="")})

    assert extract_text(state, _finished(state, 1.0)) == ""


def test_without_an_answer_the_last_steps_value_answers_not_another_key() -> None:
    state = State(operational={RESPONSE: Response(text="from response"), "last_answer": "old"})

    assert extract_text(state, _finished(state, "last value")) == "last value"
    assert extract_text(state, _finished(state, Response(text="last response"))) == (
        "last response"
    )
    assert extract_text(state, _finished(state, None)) == ""


def _hand_built(value: Callable[[], Any]) -> Flow:
    async def last(_snap: Any) -> Result:
        return Result(value=value())

    return Flow(Step(name="last", fn=last), name="hand_built")


async def test_a_hand_built_flow_answers_with_its_last_steps_response() -> None:
    response = Response(text="hand built")
    agent = Agent.from_flow(_hand_built(lambda: response))

    result = await agent.run("task")

    assert result.text == "hand built"
    assert result.response is response


async def test_a_hand_built_flow_without_a_response_has_none() -> None:
    agent = Agent.from_flow(_hand_built(lambda: {"total": 3}))

    result = await agent.run("task")

    assert result.text == "{'total': 3}"
    assert result.response is None
