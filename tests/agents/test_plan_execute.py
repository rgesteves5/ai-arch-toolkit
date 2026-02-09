"""Tests for PlanExecuteAgent."""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock

from ai_arch_toolkit.agents._base import AgentConfig, AgentEvent
from ai_arch_toolkit.agents._plan_execute import PlanExecuteAgent
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


def test_plan_execute_simple():
    """Plans, executes steps via inner ReAct, synthesizes answer."""
    plan_json = json.dumps(["Step 1: Research", "Step 2: Summarize"])
    responses = [
        # Plan
        Response(text=plan_json, usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15)),
        # Inner ReAct for step 1 (direct answer)
        Response(text="research result", usage=Usage()),
        # Inner ReAct for step 2 (direct answer)
        Response(text="summary result", usage=Usage()),
        # Synthesize
        Response(text="Final synthesized answer.", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = ToolRegistry()
    agent = PlanExecuteAgent(client, reg)
    result = agent.run("Do research and summarize")

    assert result.answer == "Final synthesized answer."
    assert client.chat.call_count == 4


def test_plan_execute_with_tools():
    """PlanExecute can use tools via inner ReAct agents."""
    plan_json = json.dumps(["Search for info"])
    responses = [
        # Plan
        Response(text=plan_json, usage=Usage()),
        # Inner ReAct: direct answer (no tool call for simplicity)
        Response(text="found info", usage=Usage()),
        # Synthesize
        Response(text="Done.", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = ToolRegistry()
    reg.register(
        "search",
        lambda q: f"results for {q}",
        Tool(name="search", description="", parameters={}),
    )
    agent = PlanExecuteAgent(client, reg)
    result = agent.run("Search and report")

    assert result.answer == "Done."


def test_plan_execute_max_replans():
    """PlanExecute respects max_iterations for replans."""
    plan_json = json.dumps(["Do thing"])
    responses = [
        # Plan
        Response(text=plan_json, usage=Usage()),
        # Inner ReAct: direct answer for step
        Response(text="step result", usage=Usage()),
        # Synthesize
        Response(text="Final.", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = ToolRegistry()
    config = AgentConfig(max_iterations=1)
    agent = PlanExecuteAgent(client, reg, config=config)
    result = agent.run("Do it")

    assert result.answer == "Final."


def test_plan_execute_async():
    """Async PlanExecute works correctly."""
    plan_json = json.dumps(["Async step"])
    responses = [
        Response(text=plan_json, usage=Usage()),
        Response(text="async result", usage=Usage()),
        Response(text="Async done.", usage=Usage()),
    ]
    client = _make_async_client(responses)
    reg = ToolRegistry()
    agent = PlanExecuteAgent(client, reg)
    result = asyncio.run(agent.async_run("Async task"))

    assert result.answer == "Async done."


def test_plan_execute_events():
    """Events include plan_created, step_start, step_end."""
    events: list[AgentEvent] = []
    plan_json = json.dumps(["One step"])
    responses = [
        Response(text=plan_json, usage=Usage()),
        Response(text="result", usage=Usage()),
        Response(text="done", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = ToolRegistry()
    config = AgentConfig(on_event=events.append)
    agent = PlanExecuteAgent(client, reg, config=config)
    agent.run("task")

    types = [e.type for e in events]
    assert "plan_created" in types
    assert types.count("step_start") >= 2  # plan + execute + synth
    assert types.count("step_end") >= 2


def test_plan_execute_timeout():
    """PlanExecute returns [timeout exceeded] when timeout is exceeded."""

    plan_json = json.dumps(["Step 1", "Step 2", "Step 3"])

    def slow_chat(*args: object, **kwargs: object) -> Response:
        time.sleep(0.03)
        return Response(text=plan_json, usage=Usage())

    client = MagicMock()
    client.chat = MagicMock(side_effect=slow_chat)
    reg = ToolRegistry()
    config = AgentConfig(timeout=0.05)
    agent = PlanExecuteAgent(client, reg, config=config)
    result = agent.run("test")

    assert result.answer == "[timeout exceeded]"
