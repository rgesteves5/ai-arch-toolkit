"""Tests for ReWOOAgent."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from ai_arch_toolkit.agents._base import AgentConfig, AgentEvent
from ai_arch_toolkit.agents._rewoo import ReWOOAgent
from ai_arch_toolkit.llm._types import Response, Tool, Usage
from ai_arch_toolkit.tools._registry import ToolRegistry


def _make_client(responses: list[Response]) -> MagicMock:
    client = MagicMock()
    client.chat = MagicMock(side_effect=responses)
    return client


def _make_async_client(responses: list[Response]) -> MagicMock:
    client = MagicMock()
    client.chat = AsyncMock(side_effect=responses)
    return client


def _make_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        "search",
        lambda input: f"result for '{input}'",
        Tool(name="search", description="Search", parameters={}),
    )
    reg.register(
        "summarize",
        lambda input: f"summary of '{input}'",
        Tool(name="summarize", description="Summarize", parameters={}),
    )
    return reg


def test_rewoo_plan_and_solve():
    """ReWOO plans, executes, and solves in 3 phases."""
    plan_text = "#E1 = search[python best practices]\n#E2 = summarize[#E1]"
    responses = [
        Response(text=plan_text, usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15)),
        Response(
            text="Final answer.", usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15)
        ),
    ]
    client = _make_client(responses)
    reg = _make_registry()
    agent = ReWOOAgent(client, reg)
    result = agent.run("Learn about Python")

    assert result.answer == "Final answer."
    assert len(result.steps) == 3
    assert client.chat.call_count == 2  # plan + solve


def test_rewoo_placeholder_substitution():
    """#E references are substituted with prior results."""
    plan_text = "#E1 = search[topic]\n#E2 = summarize[#E1]"
    responses = [
        Response(text=plan_text, usage=Usage()),
        Response(text="done", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = _make_registry()
    agent = ReWOOAgent(client, reg)
    result = agent.run("test")

    # Step 2 has tool results
    tool_results = result.steps[1].tool_results
    assert len(tool_results) == 2
    # First tool gets raw input
    assert tool_results[0].content == "result for 'topic'"
    # Second tool gets substituted result
    assert "result for 'topic'" in tool_results[1].content


def test_rewoo_tool_error_in_worker():
    """Errors during tool execution are captured gracefully."""

    def failing_tool(input: str) -> str:
        msg = "network error"
        raise ConnectionError(msg)

    plan_text = "#E1 = fail[something]"
    responses = [
        Response(text=plan_text, usage=Usage()),
        Response(text="handled", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = ToolRegistry()
    reg.register(
        "fail",
        failing_tool,
        Tool(name="fail", description="", parameters={}),
    )
    agent = ReWOOAgent(client, reg)
    result = agent.run("test")

    assert result.answer == "handled"
    assert "Error" in result.steps[1].tool_results[0].content


def test_rewoo_async():
    """Async ReWOO works correctly."""
    plan_text = "#E1 = search[query]"
    responses = [
        Response(text=plan_text, usage=Usage()),
        Response(text="async done", usage=Usage()),
    ]
    client = _make_async_client(responses)
    reg = _make_registry()
    agent = ReWOOAgent(client, reg)
    result = asyncio.run(agent.async_run("test"))

    assert result.answer == "async done"
    assert len(result.steps) == 3


def test_rewoo_events_fired():
    """Events are fired for plan, tool calls, and solve."""
    events: list[AgentEvent] = []
    plan_text = "#E1 = search[query]"
    responses = [
        Response(text=plan_text, usage=Usage()),
        Response(text="done", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = _make_registry()
    config = AgentConfig(on_event=events.append)
    agent = ReWOOAgent(client, reg, config=config)
    agent.run("test")

    types = [e.type for e in events]
    assert "plan_created" in types
    assert "tool_call" in types
    assert "tool_result" in types
    assert types.count("step_start") == 3
    assert types.count("step_end") == 3


def test_rewoo_timeout():
    """ReWOO returns [timeout exceeded] when timeout is exceeded."""

    def slow_chat(*args: object, **kwargs: object) -> Response:
        time.sleep(0.05)
        return Response(text="#E1 = search[q]", usage=Usage())

    client = MagicMock()
    client.chat = MagicMock(side_effect=slow_chat)
    reg = _make_registry()
    config = AgentConfig(timeout=0.05)
    agent = ReWOOAgent(client, reg, config=config)
    result = agent.run("test")

    assert result.answer == "[timeout exceeded]"
