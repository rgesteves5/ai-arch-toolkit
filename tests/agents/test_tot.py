"""Tests for TreeOfThoughtsAgent."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from ai_arch_toolkit.agents._base import AgentConfig
from ai_arch_toolkit.agents._tot import TreeOfThoughtsAgent
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


def _bfs_responses(depth: int, k: int) -> list[Response]:
    """Build mock responses for BFS with k thoughts per depth level.

    At each level: 1 generate call + k evaluate calls.
    """
    responses: list[Response] = []
    for d in range(depth):
        # Generate call
        thoughts = "\n".join(f"{i + 1}. Thought {d + 1}.{i + 1}" for i in range(k))
        responses.append(Response(text=thoughts, usage=Usage()))
        # Evaluate calls (one per thought)
        for i in range(k):
            score = 0.9 - (i * 0.1) - (d * 0.05)
            responses.append(Response(text=f"{score:.1f}", usage=Usage()))
    return responses


def test_tot_bfs_finds_answer():
    """BFS finds the best-scored thought."""
    # depth=1, branching_factor=2, beam_width=1
    responses = _bfs_responses(depth=1, k=2)
    client = _make_client(responses)
    reg = ToolRegistry()
    agent = TreeOfThoughtsAgent(client, reg)
    result = agent.run(
        "Solve it",
        max_depth=1,
        branching_factor=2,
        beam_width=1,
    )

    assert result.answer != "(start)"
    assert "Thought" in result.answer


def test_tot_dfs_finds_answer():
    """DFS explores the tree and finds a good thought."""
    # depth=1, branching_factor=2
    responses = _bfs_responses(depth=1, k=2)
    client = _make_client(responses)
    reg = ToolRegistry()
    agent = TreeOfThoughtsAgent(client, reg)
    result = agent.run(
        "Solve it",
        max_depth=1,
        branching_factor=2,
        beam_width=1,
        search_strategy="dfs",
    )

    assert result.answer != "(start)"


def test_tot_pruning():
    """BFS prunes to beam_width candidates at each level."""
    # depth=2, branching_factor=3, beam_width=1
    # Level 1: generate(3) + eval(3) = 4 calls, keeps top 1
    # Level 2: generate(3) + eval(3) = 4 calls for 1 surviving node
    responses = _bfs_responses(depth=1, k=3)  # level 1
    # level 2: 1 surviving node expanded
    thoughts_l2 = "1. Final A\n2. Final B\n3. Final C"
    responses.append(Response(text=thoughts_l2, usage=Usage()))
    responses.extend(Response(text=f"{0.9 - i * 0.1:.1f}", usage=Usage()) for i in range(3))
    client = _make_client(responses)
    reg = ToolRegistry()
    agent = TreeOfThoughtsAgent(client, reg)
    result = agent.run(
        "Solve",
        max_depth=2,
        branching_factor=3,
        beam_width=1,
    )

    assert result.answer != "(start)"


def test_tot_async():
    """Async ToT works correctly."""
    responses = _bfs_responses(depth=1, k=2)
    client = _make_async_client(responses)
    reg = ToolRegistry()
    agent = TreeOfThoughtsAgent(client, reg)
    result = asyncio.run(
        agent.async_run(
            "Solve",
            max_depth=1,
            branching_factor=2,
            beam_width=1,
        )
    )

    assert result.answer != "(start)"


def test_tot_max_depth_reached():
    """ToT produces a result even at max depth."""
    # depth=1, k=1 — minimal case
    responses = [
        Response(text="1. Only thought", usage=Usage()),
        Response(text="0.8", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = ToolRegistry()
    agent = TreeOfThoughtsAgent(client, reg)
    result = agent.run(
        "Simple",
        max_depth=1,
        branching_factor=1,
        beam_width=1,
    )

    assert result.answer == "Only thought"


def test_tot_timeout():
    """ToT returns [timeout exceeded] when timeout is near-zero."""

    responses = [
        Response(text="1. Thought A\n2. Thought B", usage=Usage()),
        Response(text="0.8", usage=Usage()),
        Response(text="0.6", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = ToolRegistry()
    # timeout=0 ensures _check_timeout fires before any BFS depth
    config = AgentConfig(timeout=0)
    agent = TreeOfThoughtsAgent(client, reg, config=config)
    result = agent.run(
        "test",
        max_depth=5,
        branching_factor=2,
        beam_width=2,
    )

    assert result.answer == "[timeout exceeded]"
    # No LLM calls should have been made
    assert client.chat.call_count == 0
