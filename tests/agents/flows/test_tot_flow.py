"""Tests for tot_flow factory."""

from __future__ import annotations

from unittest.mock import AsyncMock

from ai_arch_toolkit.core._budget import BudgetPolicy
from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._state import State
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents.flows._tot import tot_flow, tot_initial_state


def _make_response(text: str = "", cost: float = 0.001) -> Response:
    return Response(
        text=text,
        usage=Usage(input_tokens=10, output_tokens=5),
        cost=cost,
    )


class TestToTFlow:
    async def test_high_confidence_solves(self) -> None:
        """If evaluator gives high score, should solve immediately."""
        llm = AsyncMock()
        llm.complete = AsyncMock(
            side_effect=[
                # Generate candidates
                _make_response(text="1. Think about it\n2. Consider options\n3. Analyze"),
                # Evaluate candidate 1 — high score
                _make_response(text="0.95"),
                # Evaluate candidate 2
                _make_response(text="0.3"),
                # Evaluate candidate 3
                _make_response(text="0.4"),
                # Solve with high-confidence candidate
                _make_response(text="The answer is 42"),
            ]
        )
        tools = ToolGroup()

        flow = tot_flow(llm, tools, n_candidates=3, max_iterations=5)
        state = State(operational=tot_initial_state("What is the meaning of life?"))
        await flow.run(state)

        assert state.get("answer") == "The answer is 42"

    async def test_max_depth_solves(self) -> None:
        """At max depth, should solve directly."""
        llm = AsyncMock()
        llm.complete = AsyncMock(
            side_effect=[
                # Generate (depth 0)
                _make_response(text="1. Step A"),
                # Evaluate
                _make_response(text="0.6"),
                # At depth 1 (max_depth=1), solve
                _make_response(text="Final answer"),
            ]
        )
        tools = ToolGroup()

        flow = tot_flow(llm, tools, n_candidates=1, max_depth=1, max_iterations=5)
        state = State(operational=tot_initial_state("test"))
        await flow.run(state)

        assert state.get("answer") is not None

    async def test_max_iterations_stops(self) -> None:
        """Should stop after max_iterations even without high confidence."""
        llm = AsyncMock()
        llm.complete = AsyncMock(
            return_value=_make_response(text="1. Think\n2. More\n3. Ideas\n0.5")
        )
        tools = ToolGroup()

        flow = tot_flow(llm, tools, n_candidates=1, max_iterations=2, max_depth=10)
        state = State(operational=tot_initial_state("test"))
        result = await flow.run(state)

        # Should complete without error
        assert result.trace.flow_name == "tot"

    async def test_llm_call_budget_stops_search(self) -> None:
        llm = AsyncMock()
        llm.complete = AsyncMock(
            side_effect=[
                _make_response(text="1. Think"),
                _make_response(text="0.5"),
            ]
        )
        tools = ToolGroup()

        flow = tot_flow(
            llm,
            tools,
            n_candidates=1,
            max_iterations=5,
            budget_policy=BudgetPolicy(max_llm_calls=1),
        )
        state = State(operational=tot_initial_state("test"))
        result = await flow.run(state)

        assert llm.complete.call_count == 1
        assert "budget_exceeded" in result.results
        assert result.trace.metadata["budget"]["exceeded"]["limit"] == "llm_calls"


class TestToTInitialState:
    def test_creates_initial_state(self) -> None:
        init = tot_initial_state("task")
        assert init["task"] == "task"
        assert len(init["frontier"]) == 1
        assert init["search_done"] is False
