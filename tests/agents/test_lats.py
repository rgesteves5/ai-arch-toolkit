"""Tests for LATSAgent."""

from __future__ import annotations

import asyncio
import math
from unittest.mock import AsyncMock, MagicMock

from ai_arch_toolkit.agents._base import AgentConfig, AgentEvent
from ai_arch_toolkit.agents._lats import LATSAgent, _backpropagate, _TreeNode
from ai_arch_toolkit.llm._types import Response, Usage
from ai_arch_toolkit.tools._registry import ToolRegistry


def _make_client(responses: list[Response]) -> MagicMock:
    client = MagicMock()
    client.chat = MagicMock(side_effect=responses)
    return client


def _make_async_client(responses: list[Response]) -> MagicMock:
    client = MagicMock()
    client.chat = AsyncMock(side_effect=responses)
    return client


def test_lats_finds_answer_first_iteration():
    """LATS finds a successful answer on the first iteration."""
    responses = [
        # Expand: generate 2 candidates
        Response(text="1. Approach A\n2. Approach B", usage=Usage()),
        # Evaluate candidate 1
        Response(text="0.8", usage=Usage()),
        # Evaluate candidate 2
        Response(text="0.6", usage=Usage()),
        # Simulate best child (inner ReAct direct answer)
        Response(text="Solved!", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = ToolRegistry()
    config = AgentConfig(max_iterations=1)
    agent = LATSAgent(client, reg, config=config)
    result = agent.run("Solve it", num_expansions=2)

    assert result.answer == "Solved!"


def test_lats_multiple_iterations():
    """LATS runs multiple MCTS iterations."""
    responses = [
        # Iteration 1: expand + evaluate + simulate
        Response(text="1. Try A", usage=Usage()),
        Response(text="0.9", usage=Usage()),
        Response(text="Good result", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = ToolRegistry()
    config = AgentConfig(max_iterations=1)
    agent = LATSAgent(client, reg, config=config)
    result = agent.run("test", num_expansions=1)

    assert result.answer == "Good result"


def test_lats_backpropagation():
    """Backpropagation updates value/visits from leaf to root."""
    root = _TreeNode(state="root", visits=1, value=0.0)
    child = _TreeNode(state="child", visits=1, value=0.5, parent=root)
    root.children.append(child)
    leaf = _TreeNode(state="leaf", visits=1, value=0.0, parent=child)
    child.children.append(leaf)

    _backpropagate(leaf, 0.8)

    assert leaf.visits == 2
    assert leaf.value == 0.8
    assert child.visits == 2
    assert child.value == 1.3  # 0.5 + 0.8
    assert root.visits == 2
    assert root.value == 0.8


def test_lats_uct_selection():
    """UCT selects unvisited nodes first, then balances explore/exploit."""
    root = _TreeNode(state="root", visits=10)
    c1 = _TreeNode(state="c1", visits=5, value=3.0, parent=root)
    c2 = _TreeNode(state="c2", visits=0, value=0.0, parent=root)
    root.children = [c1, c2]

    # Unvisited node should have inf UCT
    assert c2.uct(math.sqrt(2)) == float("inf")
    # Visited node should have finite UCT
    assert c1.uct(math.sqrt(2)) < float("inf")
    # best_child should select c2 (unvisited)
    assert root.best_child(math.sqrt(2)) is c2


def test_lats_reflection_on_failure():
    """LATS generates reflection when simulation fails."""
    events: list[AgentEvent] = []
    responses = [
        # Expand
        Response(text="1. Bad approach", usage=Usage()),
        # Evaluate
        Response(text="0.5", usage=Usage()),
        # Simulate (inner ReAct) — returns failure marker
        Response(text="[max iterations reached]", usage=Usage()),
        # Reflect
        Response(text="Should try different approach", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = ToolRegistry()
    config = AgentConfig(max_iterations=1, on_event=events.append)
    agent = LATSAgent(client, reg, config=config)
    agent.run("test", num_expansions=1)

    # Should have reflection event
    types = [e.type for e in events]
    assert "reflection" in types


def test_lats_max_iterations():
    """LATS respects max_iterations."""
    responses = [
        # Iteration 1
        Response(text="1. Approach", usage=Usage()),
        Response(text="0.8", usage=Usage()),
        Response(text="result", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = ToolRegistry()
    config = AgentConfig(max_iterations=1)
    agent = LATSAgent(client, reg, config=config)
    result = agent.run("test", num_expansions=1)

    assert result.answer is not None


def test_lats_async():
    """Async LATS works correctly."""
    responses = [
        Response(text="1. Async approach", usage=Usage()),
        Response(text="0.9", usage=Usage()),
        Response(text="Async solved!", usage=Usage()),
    ]
    client = _make_async_client(responses)
    reg = ToolRegistry()
    config = AgentConfig(max_iterations=1)
    agent = LATSAgent(client, reg, config=config)
    result = asyncio.run(agent.async_run("test", num_expansions=1))

    assert result.answer == "Async solved!"


def test_lats_child_initial_visits_zero():
    """Newly created child nodes start with visits=0 (backpropagate handles first increment)."""
    root = _TreeNode(state="root", visits=0, value=0.0)
    child = _TreeNode(state="child", visits=0, value=0.5, parent=root)
    root.children.append(child)

    # Before backpropagation, child has 0 visits
    assert child.visits == 0

    _backpropagate(child, 0.8)

    # After backpropagation, child has 1 visit
    assert child.visits == 1
    assert child.value == 1.3  # 0.5 + 0.8
    assert root.visits == 1
    assert root.value == 0.8


def test_lats_timeout():
    """LATS returns [timeout exceeded] when timeout is near-zero."""
    responses = [
        Response(text="1. Approach A", usage=Usage()),
        Response(text="0.8", usage=Usage()),
        Response(text="result", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = ToolRegistry()
    # timeout=0 ensures _check_timeout fires before first iteration
    config = AgentConfig(timeout=0, max_iterations=10)
    agent = LATSAgent(client, reg, config=config)
    result = agent.run("test", num_expansions=1)

    assert result.answer == "[timeout exceeded]"
    assert client.chat.call_count == 0


def test_lats_custom_evaluator():
    """Custom evaluator overrides default heuristic."""
    responses = [
        # Expand
        Response(text="1. Approach A", usage=Usage()),
        # Evaluate
        Response(text="0.8", usage=Usage()),
        # Simulate (inner ReAct direct answer)
        Response(text="Solved!", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = ToolRegistry()
    config = AgentConfig(max_iterations=1)
    agent = LATSAgent(client, reg, config=config)
    result = agent.run("test", num_expansions=1, evaluator=lambda x: 0.9)

    assert result.answer == "Solved!"


def test_lats_default_evaluator_fallback():
    """Default heuristic evaluator is used when no evaluator kwarg."""
    responses = [
        Response(text="1. Try", usage=Usage()),
        Response(text="0.8", usage=Usage()),
        # Inner ReAct returns failure marker -> heuristic gives 0.0
        Response(text="[max iterations reached]", usage=Usage()),
        # Reflect (because heuristic score < 0.5)
        Response(text="reflection", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = ToolRegistry()
    config = AgentConfig(max_iterations=1)
    agent = LATSAgent(client, reg, config=config)
    result = agent.run("test", num_expansions=1)

    # Default heuristic returns 0.0 for answers starting with "[",
    # triggering reflection. No successful terminal -> fallback.
    assert result.answer == "[no solution found]"
