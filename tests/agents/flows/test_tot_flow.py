"""Tests for tot_flow factory."""

from __future__ import annotations

from unittest.mock import AsyncMock

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._state import State
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents.flows._tot import tot_flow, tot_initial_state
from ai_arch_toolkit.toolkit.budget import BudgetPolicy


def _make_response(text: str = "", cost: float = 0.001) -> Response:
    return Response(
        text=text,
        usage=Usage(input_tokens=10, output_tokens=5),
        cost=cost,
    )


class _FakeProvider:
    """A real LLM's provider stand-in, so LLM.complete runs its metering charge site."""

    def __init__(self, *responses: Response) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def complete(self, messages, *, system=None, tools=None, **kwargs) -> Response:
        self.calls += 1
        return self._responses[min(self.calls - 1, len(self._responses) - 1)]


def _metered_llm(*responses: Response) -> tuple[LLM, _FakeProvider]:
    llm = LLM("claude-sonnet-4-6", api_key="test")
    provider = _FakeProvider(*responses)
    llm._provider = provider  # type: ignore[assignment]
    return llm, provider


def _budget_dimension(result) -> str:
    info = result.results["budget_exceeded"].artifacts["budget_exceeded"]
    return info.get("dimension") or (info.get("breached") or [None])[0]


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
        llm, provider = _metered_llm(
            _make_response(text="1. Think"),
            _make_response(text="0.5"),
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

        assert provider.calls == 1  # generate ran; evaluate was denied at the charge site
        assert "budget_exceeded" in result.results
        assert _budget_dimension(result) == "llm_calls"


class TestToTInitialState:
    def test_creates_initial_state(self) -> None:
        init = tot_initial_state("task")
        assert init["task"] == "task"
        assert len(init["frontier"]) == 1
        assert init["search_done"] is False
