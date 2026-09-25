"""Tests for lats_flow factory."""

from __future__ import annotations

from unittest.mock import AsyncMock

from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._state import State
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents.flows._lats import lats_flow, lats_initial_state
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

    async def test_llm_call_budget_stops_after_inner_rollout(self) -> None:
        llm, provider = fake_llm(_make_response(text="answer"))
        tools = ToolGroup()

        def evaluator(task: str, answer: str) -> float:
            return 0.95

        flow = lats_flow(
            llm,
            tools,
            evaluator_fn=evaluator,
            max_rollouts=2,
            budget_policy=BudgetPolicy(max_llm_calls=1),
        )
        state = State(operational=lats_initial_state("test"))
        result = await flow.run(state)

        assert provider.calls == 1  # inner ReAct ran once; the solver was denied
        assert "budget_exceeded" in result.results
        assert _budget_dimension(result) == "llm_calls"


class TestLATSInitialState:
    def test_creates_initial_state(self) -> None:
        init = lats_initial_state("task")
        assert init["task"] == "task"
        assert init["mcts_root"] is not None
        assert init["search_done"] is False


def _scripted_llm(prompts: list[str]) -> AsyncMock:
    """An LLM for the tree tests. A ReAct attempt answers ``attempt-N`` and records its prompt, the
    expanded node's state, in ``prompts``; a reflection answers ``reflection``; the solver answers
    ``final: <the best answer it was given>``."""

    async def complete(messages, **kwargs):
        content = messages[-1]["content"]
        if "tools" in kwargs:  # a turn of the inner ReAct
            prompts.append(messages[0]["content"])
            return _make_response(text=f"attempt-{len(prompts)}")
        if "Provide feedback." in content:
            return _make_response(text="reflection")
        return _make_response(text="final: " + content.split("Best answer: ")[1].split("\n")[0])

    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=complete)
    return llm


def _count_nodes(node) -> int:
    return sum(1 + _count_nodes(child) for child in node.children)


class TestLATSTree:
    """A node takes ``n_candidates`` sibling attempts before the search goes below it."""

    async def test_the_root_takes_n_candidates_children(self) -> None:
        prompts: list[str] = []

        def evaluator(task: str, answer: str) -> float:
            return 0.6  # neither solved nor reflected on

        flow = lats_flow(
            _scripted_llm(prompts),
            ToolGroup(),
            n_candidates=3,
            max_rollouts=3,
            evaluator_fn=evaluator,
        )
        state = State(operational=lats_initial_state("task"))
        await flow.run(state)

        root = state.get("mcts_root")
        assert [child.answer for child in root.children] == ["attempt-1", "attempt-2", "attempt-3"]
        assert all(not child.children for child in root.children)
        assert prompts == ["task", "task", "task"]  # every attempt started from the root

    async def test_uct_returns_to_a_sibling_when_a_branch_disappoints(self) -> None:
        prompts: list[str] = []
        scores = {"attempt-1": 0.6, "attempt-2": 0.55, "attempt-3": 0.1, "attempt-4": 0.7}

        def evaluator(task: str, answer: str) -> float:
            return scores[answer]

        flow = lats_flow(
            _scripted_llm(prompts),
            ToolGroup(),
            n_candidates=2,
            max_rollouts=4,
            exploration_weight=0.0,  # UCT is then the mean score, so the path is deterministic
            evaluator_fn=evaluator,
        )
        state = State(operational=lats_initial_state("task"))
        await flow.run(state)

        first, second = state.get("mcts_root").children
        # The better child (0.6) was expanded first; its attempt scored 0.1, which pulled its mean
        # to 0.35, below its sibling's 0.55, so the next rollout expanded the sibling.
        assert [child.answer for child in first.children] == ["attempt-3"]
        assert [child.answer for child in second.children] == ["attempt-4"]
        assert prompts[2:] == ["task\nAttempt: attempt-1", "task\nAttempt: attempt-2"]

    async def test_max_rollouts_caps_the_attempts(self) -> None:
        prompts: list[str] = []

        def evaluator(task: str, answer: str) -> float:
            return 0.6

        flow = lats_flow(
            _scripted_llm(prompts),
            ToolGroup(),
            n_candidates=2,
            max_rollouts=5,
            evaluator_fn=evaluator,
        )
        state = State(operational=lats_initial_state("task"))
        await flow.run(state)

        root = state.get("mcts_root")
        assert len(prompts) == 5
        assert state.get("rollout_num") == 5
        assert len(root.children) == 2
        assert _count_nodes(root) == 5
        assert state.get("search_done") is True

    async def test_the_answer_comes_from_the_best_attempt(self) -> None:
        scores = {"attempt-1": 0.6, "attempt-2": 0.8, "attempt-3": 0.7}

        def evaluator(task: str, answer: str) -> float:
            return scores[answer]

        flow = lats_flow(
            _scripted_llm([]),
            ToolGroup(),
            n_candidates=3,
            max_rollouts=3,
            evaluator_fn=evaluator,
        )
        state = State(operational=lats_initial_state("task"))
        await flow.run(state)

        assert state.get("best_answer") == "attempt-2"
        assert state.get("answer") == "final: attempt-2"
