"""Tests for tot_flow factory."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock

import pytest

from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._state import State
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents.flows._tot import tot_flow, tot_initial_state
from ai_arch_toolkit.toolkit.budget import BudgetPolicy
from tests.fake_provider import fake_llm


def _make_response(text: str = "", cost: float = 0.001) -> Response:
    return Response(
        text=text,
        usage=Usage(input_tokens=10, output_tokens=5),
        cost=cost,
    )


def _budget_dimension(result) -> str:
    info = result.results["budget_exceeded"].artifacts["budget_exceeded"]
    return info.get("dimension") or (info.get("breached") or [None])[0]


_THOUGHTS = "1. GOOD\n2. MID\n3. BAD"
_SCORES = {"GOOD": "0.8", "MID": "0.5", "BAD": "0.2"}


class _ScriptedLLM:
    """Proposes ``generate(reasoning)`` as the next thoughts and scores each by its text.

    Records how many calls it answered and the reasoning each final answer was solved from.
    """

    def __init__(
        self, scores: dict[str, str], generate: Callable[[str], str] = lambda _: _THOUGHTS
    ) -> None:
        self.scores = scores
        self.generate = generate
        self.calls = 0
        self.solved_from: list[str] = []

    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> Response:
        self.calls += 1
        prompt: str = messages[-1]["content"]
        if prompt.endswith("Score (0.0-1.0):"):
            return _make_response(self.scores[prompt.split("Next step: ")[1].split("\n")[0]])
        if prompt.endswith("Provide the final answer."):
            reasoning = prompt.split(":\n", 1)[1].removesuffix("\n\nProvide the final answer.")
            self.solved_from.append(reasoning)
            return _make_response("final answer")
        return _make_response(
            self.generate(prompt.split("Current reasoning:\n")[1].split("\n\n")[0])
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

    async def test_dfs_expands_the_most_promising_child_first(self) -> None:
        llm = _ScriptedLLM(_SCORES)

        flow = tot_flow(llm, ToolGroup(), strategy="dfs")
        state = State(operational=tot_initial_state("task"))
        await flow.run(state)

        assert llm.solved_from == ["task\nGOOD\nGOOD\nGOOD"]
        assert llm.calls == 13  # three expansions of one generation and three scores, one solve
        assert state.get("answer") == "final answer"

    async def test_bfs_keeps_the_best_states_of_each_level_and_reaches_max_depth(self) -> None:
        llm = _ScriptedLLM(_SCORES)

        flow = tot_flow(llm, ToolGroup(), strategy="bfs")
        state = State(operational=tot_initial_state("task"))
        await flow.run(state)

        # The root, then three states on each of the next two levels (not 3 and 9), then a solve.
        assert llm.calls == 29
        assert llm.solved_from == ["task\nGOOD\nGOOD\nGOOD"]
        assert state.get("answer") == "final answer"

    @pytest.mark.parametrize("strategy", ["dfs", "bfs"])
    async def test_out_of_iterations_it_answers_from_the_best_state_found(
        self, strategy: str
    ) -> None:
        llm = _ScriptedLLM(_SCORES)

        flow = tot_flow(llm, ToolGroup(), strategy=strategy, max_iterations=2)
        state = State(operational=tot_initial_state("task"))
        await flow.run(state)

        assert llm.solved_from == ["task\nGOOD\nGOOD"]
        assert state.get("answer") == "final answer"

    @pytest.mark.parametrize("strategy", ["dfs", "bfs"])
    async def test_out_of_states_it_answers_from_the_best_state_found(self, strategy: str) -> None:
        # Only the task yields thoughts: each of its children is a dead end.
        llm = _ScriptedLLM(
            _SCORES, generate=lambda reasoning: _THOUGHTS if reasoning == "task" else "No idea."
        )

        flow = tot_flow(llm, ToolGroup(), strategy=strategy)
        state = State(operational=tot_initial_state("task"))
        await flow.run(state)

        assert llm.solved_from == ["task\nGOOD"]
        assert state.get("answer") == "final answer"

    @pytest.mark.parametrize(
        ("scores", "best"),
        [
            # The prompt's "Score (0.0-1.0):" repeated before the score is not the score…
            ({"A": "Score (0.0-1.0): 0.8", "B": "0.5"}, "A"),
            # …nor is a step's number.
            ({"A": "Step 2 looks strong: 0.9", "B": "0.95"}, "B"),
        ],
    )
    async def test_a_score_is_the_last_number_in_range_of_the_evaluator_reply(
        self, scores: dict[str, str], best: str
    ) -> None:
        llm = _ScriptedLLM(scores, generate=lambda _: "1. A\n2. B")

        flow = tot_flow(llm, ToolGroup(), n_candidates=2, max_depth=1, strategy="bfs")
        state = State(operational=tot_initial_state("task"))
        await flow.run(state)

        assert llm.solved_from == [f"task\n{best}"]

    async def test_llm_call_budget_stops_search(self) -> None:
        llm, provider = fake_llm(
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
