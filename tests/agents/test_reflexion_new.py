"""Tests for ReflexionAgent — ReAct + self-critique retry loop."""

from __future__ import annotations

from unittest.mock import AsyncMock

from ai_arch_toolkit.core._response import Response
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents._base import AgentConfig, AgentEvent, AgentResult
from ai_arch_toolkit.toolkit.agents._reflexion import ReflexionAgent, ReflexionConfig
from tests.agents.conftest import make_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    llm_side_effect: list[Response],
    *,
    config: AgentConfig | None = None,
    reflexion: ReflexionConfig | None = None,
) -> ReflexionAgent:
    """Build a ReflexionAgent with mocked LLM."""

    def dummy_tool(x: str = "") -> str:
        """A test tool."""
        return f"result:{x}"

    tools = ToolGroup(dummy_tool)
    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=llm_side_effect)
    return ReflexionAgent(
        llm,
        tools,
        config=config or AgentConfig(),
        reflexion=reflexion or ReflexionConfig(evaluator=lambda _t, _a: 1.0),
    )


# ---------------------------------------------------------------------------
# 1. Passes first attempt — evaluator returns 1.0
# ---------------------------------------------------------------------------


async def test_passes_first_attempt():
    agent = _make_agent(
        [make_response(text="Good answer")],
        reflexion=ReflexionConfig(evaluator=lambda _t, _a: 1.0),
    )
    result = await agent.run("Solve this")

    assert result.stop_reason == "completed"
    assert result.answer == "Good answer"
    assert len(result.steps) == 1


# ---------------------------------------------------------------------------
# 2. Fails then passes — 0.3 → 1.0, reflection LLM called
# ---------------------------------------------------------------------------


async def test_fails_then_passes():
    scores = iter([0.3, 1.0])

    def evaluator(_t: str, _a: str) -> float:
        return next(scores)

    responses = [
        # Attempt 1: ReAct answer
        make_response(text="Bad answer"),
        # Reflection LLM call
        make_response(text="You should try harder"),
        # Attempt 2: ReAct answer
        make_response(text="Good answer"),
    ]
    agent = _make_agent(
        responses,
        reflexion=ReflexionConfig(evaluator=evaluator, max_retries=3),
    )
    result = await agent.run("Solve this")

    assert result.stop_reason == "completed"
    assert result.answer == "Good answer"
    # LLM called 3 times: attempt1, reflect, attempt2
    assert agent.llm.complete.call_count == 3


# ---------------------------------------------------------------------------
# 3. All retries exhausted — always 0.1 → max_iterations
# ---------------------------------------------------------------------------


async def test_all_retries_exhausted():
    responses = [
        make_response(text="Bad 1"),
        make_response(text="Reflect 1"),
        make_response(text="Bad 2"),
        make_response(text="Reflect 2"),
        make_response(text="Bad 3"),
        make_response(text="Reflect 3"),
    ]
    agent = _make_agent(
        responses,
        reflexion=ReflexionConfig(evaluator=lambda _t, _a: 0.1, max_retries=3),
    )
    result = await agent.run("Impossible task")

    assert result.stop_reason == "max_iterations"


# ---------------------------------------------------------------------------
# 4. Reflection prompt contains task + answer + score
# ---------------------------------------------------------------------------


async def test_reflection_prompt_contents():
    scores = iter([0.3, 1.0])

    def evaluator(_t: str, _a: str) -> float:
        return next(scores)

    responses = [
        make_response(text="My bad answer"),
        make_response(text="Reflection text"),
        make_response(text="Good answer"),
    ]
    agent = _make_agent(
        responses,
        reflexion=ReflexionConfig(evaluator=evaluator, max_retries=3),
    )
    await agent.run("Find the capital of France")

    # Second LLM call is the reflection
    reflect_call = agent.llm.complete.call_args_list[1]
    reflect_messages = reflect_call.args[0]
    prompt_text = reflect_messages[0]["content"]

    assert "Find the capital of France" in prompt_text
    assert "My bad answer" in prompt_text
    assert "0.30" in prompt_text


# ---------------------------------------------------------------------------
# 5. Timeout across retries (shared timeout)
# ---------------------------------------------------------------------------


async def test_timeout_across_retries(monkeypatch):
    import ai_arch_toolkit.toolkit.agents._react as react_mod

    _counter = {"n": 0}

    def fake_monotonic():
        _counter["n"] += 1
        return _counter["n"] * 100.0

    monkeypatch.setattr(react_mod.time, "monotonic", fake_monotonic)

    responses = [
        make_response(text="Answer"),
        make_response(text="Reflect"),
        make_response(text="Answer2"),
    ]
    agent = _make_agent(
        responses,
        config=AgentConfig(timeout=0.001),
        reflexion=ReflexionConfig(evaluator=lambda _t, _a: 0.1, max_retries=3),
    )
    result = await agent.run("Test timeout")

    assert result.stop_reason == "timeout"


# ---------------------------------------------------------------------------
# 6. Event sequence continuity — step numbers increase
# ---------------------------------------------------------------------------


async def test_event_sequence_continuity():
    scores = iter([0.3, 1.0])

    def evaluator(_t: str, _a: str) -> float:
        return next(scores)

    responses = [
        make_response(text="Bad"),
        make_response(text="Reflection"),
        make_response(text="Good"),
    ]
    events: list[AgentEvent] = []
    config = AgentConfig(on_event=events.append)
    agent = _make_agent(
        responses,
        config=config,
        reflexion=ReflexionConfig(evaluator=evaluator, max_retries=3),
    )
    await agent.run("Test")

    step_starts = [e.step for e in events if e.type == "step_start"]
    # Steps should be monotonically increasing
    assert step_starts == sorted(step_starts)
    assert len(set(step_starts)) == len(step_starts)  # no duplicates


# ---------------------------------------------------------------------------
# 7. run_sync works
# ---------------------------------------------------------------------------


def test_run_sync():
    agent = _make_agent(
        [make_response(text="Sync answer")],
        reflexion=ReflexionConfig(evaluator=lambda _t, _a: 1.0),
    )
    result = agent.run_sync("Sync test")

    assert isinstance(result, AgentResult)
    assert result.answer == "Sync answer"
    assert result.stop_reason == "completed"


# ---------------------------------------------------------------------------
# 8. Evaluator receives correct (task, answer) args
# ---------------------------------------------------------------------------


async def test_evaluator_args():
    received: list[tuple[str, str]] = []

    def tracking_evaluator(task: str, answer: str) -> float:
        received.append((task, answer))
        return 1.0

    agent = _make_agent(
        [make_response(text="The answer is 42")],
        reflexion=ReflexionConfig(evaluator=tracking_evaluator),
    )
    await agent.run("What is the meaning of life?")

    assert len(received) == 1
    assert received[0] == ("What is the meaning of life?", "The answer is 42")
