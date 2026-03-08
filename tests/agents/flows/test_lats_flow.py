"""Tests for lats_flow factory."""

from __future__ import annotations

from unittest.mock import AsyncMock

from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._state import State
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents.flows._lats import lats_flow, lats_initial_state


def _make_response(text: str = "", cost: float = 0.001) -> Response:
    return Response(
        text=text,
        usage=Usage(input_tokens=10, output_tokens=5),
        cost=cost,
    )


class TestLATSFlow:
    async def test_high_score_solves(self) -> None:
        llm = AsyncMock()
        llm.complete = AsyncMock(
            side_effect=[
                # Inner ReAct response (no tool calls → completes)
                _make_response(text="Good answer"),
                # Solver
                _make_response(text="Final answer"),
            ]
        )
        tools = ToolGroup()

        def evaluator(task: str, answer: str) -> float:
            return 0.95  # High score

        flow = lats_flow(
            llm,
            tools,
            evaluator_fn=evaluator,
            max_rollouts=5,
        )
        state = State(operational=lats_initial_state("test task"))
        await flow.run(state)

        assert state.get("search_done") is True
        assert state.get("answer") is not None

    async def test_low_score_reflects(self) -> None:
        call_count = 0

        async def mock_complete(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            return _make_response(text=f"response {call_count}")

        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=mock_complete)
        tools = ToolGroup()

        scores = iter([0.3, 0.95])

        def evaluator(task: str, answer: str) -> float:
            return next(scores)

        flow = lats_flow(
            llm,
            tools,
            evaluator_fn=evaluator,
            max_rollouts=3,
        )
        state = State(operational=lats_initial_state("test"))
        await flow.run(state)

        # Should have done at least 2 rollouts
        assert state.get("rollout_num", 0) >= 2

    async def test_max_rollouts_exhausted(self) -> None:
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_response(text="answer"))
        tools = ToolGroup()

        def evaluator(task: str, answer: str) -> float:
            return 0.5  # Never passes threshold

        flow = lats_flow(
            llm,
            tools,
            evaluator_fn=evaluator,
            max_rollouts=2,
        )
        state = State(operational=lats_initial_state("test"))
        result = await flow.run(state)

        assert result.trace.flow_name == "lats"


class TestLATSInitialState:
    def test_creates_initial_state(self) -> None:
        init = lats_initial_state("task")
        assert init["task"] == "task"
        assert init["mcts_root"] is not None
        assert init["search_done"] is False
