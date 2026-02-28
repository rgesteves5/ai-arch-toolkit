"""Tests for LATSAgent — Language Agent Tree Search."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ai_arch_toolkit.core._response import Response
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents._base import AgentConfig, AgentResult
from ai_arch_toolkit.toolkit.agents._lats import LATSAgent, LATSConfig
from tests.agents.conftest import make_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    llm_side_effect: list[Response],
    *,
    config: AgentConfig | None = None,
    lats: LATSConfig | None = None,
) -> LATSAgent:
    """Build a LATSAgent with mocked LLM."""

    def dummy_tool(x: str = "") -> str:
        """A test tool."""
        return f"result:{x}"

    tools = ToolGroup(dummy_tool)
    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=llm_side_effect)
    return LATSAgent(
        llm,
        tools,
        config=config or AgentConfig(),
        lats=lats or LATSConfig(evaluator_fn=lambda _t, _a: 0.5),
    )


# ---------------------------------------------------------------------------
# 1. Finds solution on first rollout (high score)
# ---------------------------------------------------------------------------


async def test_finds_solution_first_rollout():
    responses = [
        make_response(text="The answer is 42"),  # Inner ReAct answer
        make_response(text="42 is correct"),  # Final answer
    ]
    agent = _make_agent(
        responses,
        lats=LATSConfig(evaluator_fn=lambda _t, _a: 0.95),
    )
    result = await agent.run("What is 6*7?")

    assert result.stop_reason == "completed"
    assert "42" in result.answer


# ---------------------------------------------------------------------------
# 2. Multiple rollouts before success
# ---------------------------------------------------------------------------


async def test_multiple_rollouts():
    scores = iter([0.3, 0.95])

    def evaluator(_t: str, _a: str) -> float:
        return next(scores)

    responses = [
        make_response(text="Bad answer"),  # Rollout 1 inner ReAct
        make_response(text="What went wrong: not enough info"),  # Reflection
        make_response(text="Good answer"),  # Rollout 2 inner ReAct
        make_response(text="Final good answer"),  # Final answer
    ]
    agent = _make_agent(
        responses,
        lats=LATSConfig(evaluator_fn=evaluator, max_rollouts=5),
    )
    result = await agent.run("Solve this")

    assert result.stop_reason == "completed"


# ---------------------------------------------------------------------------
# 3. All rollouts exhausted
# ---------------------------------------------------------------------------


async def test_all_rollouts_exhausted():
    responses = [
        make_response(text="Bad answer 1"),
        make_response(text="Reflection 1"),
        make_response(text="Bad answer 2"),
        make_response(text="Reflection 2"),
    ]
    agent = _make_agent(
        responses,
        lats=LATSConfig(evaluator_fn=lambda _t, _a: 0.3, max_rollouts=2),
    )
    result = await agent.run("Impossible task")

    assert result.stop_reason == "max_iterations"


# ---------------------------------------------------------------------------
# 4. External evaluator used when provided
# ---------------------------------------------------------------------------


async def test_external_evaluator():
    eval_calls: list[tuple[str, str]] = []

    def tracking_eval(task: str, answer: str) -> float:
        eval_calls.append((task, answer))
        return 0.95

    responses = [
        make_response(text="My answer"),
        make_response(text="Polished answer"),
    ]
    agent = _make_agent(
        responses,
        lats=LATSConfig(evaluator_fn=tracking_eval),
    )
    result = await agent.run("Test task")

    assert result.stop_reason == "completed"
    assert len(eval_calls) == 1
    assert eval_calls[0][0] == "Test task"
    assert eval_calls[0][1] == "My answer"


# ---------------------------------------------------------------------------
# 5. UCT selection prefers high-value + under-explored nodes
# ---------------------------------------------------------------------------


async def test_uct_selection():
    from ai_arch_toolkit.toolkit.agents._lats import _Node, _select_uct, _uct_score

    root = _Node(state="root")
    root.visits = 10

    child_a = _Node(state="a", parent=root)
    child_a.visits = 5
    child_a.value = 4.0  # avg 0.8

    child_b = _Node(state="b", parent=root)
    child_b.visits = 1
    child_b.value = 0.3  # avg 0.3 but under-explored

    root.children = [child_a, child_b]

    # Under-explored node (child_b) should have higher UCT with high exploration
    score_b = _uct_score(child_b, exploration_weight=2.0)
    score_a = _uct_score(child_a, exploration_weight=2.0)
    assert score_b > score_a

    # With low exploration weight, high-value node wins
    score_a_low = _uct_score(child_a, exploration_weight=0.01)
    score_b_low = _uct_score(child_b, exploration_weight=0.01)
    assert score_a_low > score_b_low

    # Unvisited node should have infinite UCT
    child_c = _Node(state="c", parent=root)
    child_c.visits = 0
    root.children.append(child_c)
    assert _uct_score(child_c, exploration_weight=1.0) == float("inf")

    # select_uct should pick unvisited child
    selected = _select_uct(root, exploration_weight=1.0)
    assert selected is child_c


# ---------------------------------------------------------------------------
# 6. Reflection stored on low-score rollouts
# ---------------------------------------------------------------------------


async def test_reflection_on_low_score():
    responses = [
        make_response(text="Bad answer"),  # Rollout 1
        make_response(text="Need to think more carefully"),  # Reflection
        make_response(text="Better answer"),  # Rollout 2
        make_response(text="Final answer"),  # Final
    ]
    scores = iter([0.2, 0.95])
    agent = _make_agent(
        responses,
        lats=LATSConfig(evaluator_fn=lambda _t, _a: next(scores), max_rollouts=5),
    )
    result = await agent.run("Test reflection")

    assert result.stop_reason == "completed"


# ---------------------------------------------------------------------------
# 7. run_sync works
# ---------------------------------------------------------------------------


def test_run_sync():
    responses = [
        make_response(text="Quick answer"),
        make_response(text="Final"),
    ]
    agent = _make_agent(
        responses,
        lats=LATSConfig(evaluator_fn=lambda _t, _a: 0.95),
    )
    result = agent.run_sync("Sync test")

    assert isinstance(result, AgentResult)
    assert result.stop_reason == "completed"


# ---------------------------------------------------------------------------
# 8. LLM-based evaluator (default path, no external evaluator)
# ---------------------------------------------------------------------------


async def test_llm_based_evaluator():
    """When no external evaluator provided, LLM is used for evaluation."""
    responses = [
        make_response(text="My answer"),  # Inner ReAct
        make_response(text="0.95"),  # LLM evaluation
        make_response(text="Polished answer"),  # Final answer
    ]
    agent = _make_agent(
        responses,
        config=AgentConfig(),
        lats=LATSConfig(evaluator_fn=None, max_rollouts=5),  # no external evaluator
    )
    result = await agent.run("Test LLM eval")

    assert result.stop_reason == "completed"
    # 3 LLM calls: inner ReAct, evaluation, final answer
    assert agent.llm.complete.call_count == 3
    # Evaluation call (2nd) should use evaluator_system
    eval_call = agent.llm.complete.call_args_list[1]
    assert "system" in eval_call.kwargs


# ---------------------------------------------------------------------------
# 9. Mid-range score (0.5-0.9) — no reflection, no completion
# ---------------------------------------------------------------------------


async def test_mid_range_score_no_reflection():
    """Score in [0.5, 0.9) should not trigger reflection or completion."""
    scores = iter([0.7, 0.7])

    def evaluator(_t: str, _a: str) -> float:
        return next(scores)

    responses = [
        make_response(text="Answer 1"),  # Rollout 1
        make_response(text="Answer 2"),  # Rollout 2
    ]
    agent = _make_agent(
        responses,
        lats=LATSConfig(evaluator_fn=evaluator, max_rollouts=2),
    )
    result = await agent.run("Mid score test")

    # Should exhaust rollouts — not high enough to complete, not low enough to reflect
    assert result.stop_reason == "max_iterations"
    # No reflection calls — only 2 LLM calls (inner ReAct for each rollout)
    assert agent.llm.complete.call_count == 2


# ---------------------------------------------------------------------------
# 10. Backpropagation updates node statistics
# ---------------------------------------------------------------------------


def test_backprop_updates_statistics():
    from ai_arch_toolkit.toolkit.agents._lats import _backprop, _Node

    root = _Node(state="root")
    child = _Node(state="child", parent=root)
    root.children.append(child)

    _backprop(child, 0.8)

    assert child.visits == 1
    assert child.value == 0.8
    assert root.visits == 1
    assert root.value == 0.8

    # Second backprop
    _backprop(child, 0.4)
    assert child.visits == 2
    assert child.value == pytest.approx(1.2)
    assert root.visits == 2
    assert root.value == pytest.approx(1.2)


# ---------------------------------------------------------------------------
# 11. Score parsing edge cases
# ---------------------------------------------------------------------------


def test_lats_parse_score_edge_cases():
    from ai_arch_toolkit.toolkit.agents._lats import _parse_score

    assert _parse_score("0.5") == 0.5
    assert _parse_score("no number") == 0.0
    assert _parse_score("") == 0.0
    assert _parse_score("1.5") == 1.0
    assert _parse_score("Score: 0.75") == 0.75


# ---------------------------------------------------------------------------
# 12. Custom evaluator_system forwarded to LLM eval
# ---------------------------------------------------------------------------


async def test_custom_evaluator_system():
    """Custom evaluator_system should be used in LLM-based evaluation."""
    custom_sys = "Rate accuracy 0-1."
    responses = [
        make_response(text="My answer"),  # Inner ReAct
        make_response(text="0.95"),  # LLM evaluation
        make_response(text="Final"),  # Final answer
    ]
    agent = _make_agent(
        responses,
        config=AgentConfig(),
        lats=LATSConfig(evaluator_fn=None, evaluator_system=custom_sys, max_rollouts=5),
    )
    await agent.run("Test custom eval system")

    eval_call = agent.llm.complete.call_args_list[1]
    assert eval_call.kwargs.get("system") == custom_sys
