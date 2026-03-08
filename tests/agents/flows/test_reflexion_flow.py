"""Tests for reflexion_flow factory."""

from __future__ import annotations

from unittest.mock import AsyncMock

from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._state import State
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents.flows._reflexion import (
    reflexion_flow,
    reflexion_initial_state,
)


def _make_response(text: str = "", cost: float = 0.001) -> Response:
    return Response(
        text=text,
        usage=Usage(input_tokens=10, output_tokens=5),
        cost=cost,
    )


class TestReflexionFlow:
    async def test_passes_on_first_attempt(self) -> None:
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_response(text="good answer"))
        tools = ToolGroup()

        def evaluator(task: str, answer: str) -> float:
            return 0.9  # Above threshold

        flow = reflexion_flow(llm, tools, evaluator=evaluator, threshold=0.7)
        state = State(operational=reflexion_initial_state("task"))
        await flow.run(state)

        assert state.get("score") == 0.9
        assert state.get("passed") is True

    async def test_reflects_and_retries(self) -> None:
        call_count = 0

        async def mock_complete(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # First attempt + first reflection
                return _make_response(text=f"attempt {call_count}")
            return _make_response(text="better answer")

        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=mock_complete)
        tools = ToolGroup()

        scores = iter([0.3, 0.9])

        def evaluator(task: str, answer: str) -> float:
            return next(scores)

        flow = reflexion_flow(llm, tools, evaluator=evaluator, threshold=0.7, max_retries=3)
        state = State(operational=reflexion_initial_state("task"))
        await flow.run(state)

        # Should have reflections
        reflections = state.get("reflections", [])
        assert len(reflections) >= 1

    async def test_max_retries_exhausted(self) -> None:
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_response(text="bad answer"))
        tools = ToolGroup()

        def evaluator(task: str, answer: str) -> float:
            return 0.1  # Always below threshold

        flow = reflexion_flow(llm, tools, evaluator=evaluator, threshold=0.7, max_retries=2)
        state = State(operational=reflexion_initial_state("task"))
        await flow.run(state)

        # Should have exhausted retries
        assert state.get("passed") is False


class TestReflexionInitialState:
    def test_creates_initial_state(self) -> None:
        init = reflexion_initial_state("my task")
        assert init["task"] == "my task"
        assert init["reflections"] == []
        assert init["passed"] is False
