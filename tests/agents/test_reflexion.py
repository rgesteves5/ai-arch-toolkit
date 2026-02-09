"""Tests for ReflexionAgent."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from ai_arch_toolkit.agents._base import AgentConfig, AgentEvent
from ai_arch_toolkit.agents._reflexion import ReflexionAgent
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


def test_reflexion_passes_first_attempt():
    """Returns immediately when evaluator score >= threshold."""
    responses = [
        Response(
            text="good answer", usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15)
        ),
    ]
    client = _make_client(responses)
    reg = ToolRegistry()
    agent = ReflexionAgent(client, reg)
    result = agent.run("task", evaluator=lambda x: 1.0)

    assert result.answer == "good answer"
    assert client.chat.call_count == 1


def test_reflexion_retries_then_passes():
    """Retries with reflection when first attempt scores below threshold."""
    responses = [
        # Attempt 1: inner ReAct direct answer (low score)
        Response(text="bad answer", usage=Usage()),
        # Reflection
        Response(text="try harder", usage=Usage()),
        # Attempt 2: inner ReAct direct answer (high score)
        Response(text="good answer", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = ToolRegistry()
    scores = iter([0.3, 0.9])
    agent = ReflexionAgent(client, reg)
    result = agent.run("task", evaluator=lambda x: next(scores))

    assert result.answer == "good answer"
    assert client.chat.call_count == 3


def test_reflexion_max_retries_exceeded():
    """Returns last answer when max iterations reached."""
    responses = [
        Response(text="bad", usage=Usage()),
        Response(text="reflect1", usage=Usage()),
        Response(text="still bad", usage=Usage()),
        Response(text="reflect2", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = ToolRegistry()
    config = AgentConfig(max_iterations=2)
    agent = ReflexionAgent(client, reg, config=config)
    result = agent.run("task", evaluator=lambda x: 0.1)

    # After 2 iterations, returns last answer
    assert result.answer == "still bad"


def test_reflexion_async():
    """Async Reflexion works correctly."""
    responses = [
        Response(text="async good", usage=Usage()),
    ]
    client = _make_async_client(responses)
    reg = ToolRegistry()
    agent = ReflexionAgent(client, reg)
    result = asyncio.run(agent.async_run("task", evaluator=lambda x: 1.0))

    assert result.answer == "async good"


def test_reflexion_events():
    """Events include step_start, step_end, and reflection."""
    events: list[AgentEvent] = []
    responses = [
        Response(text="bad", usage=Usage()),
        Response(text="reflection text", usage=Usage()),
        Response(text="good", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = ToolRegistry()
    scores = iter([0.3, 0.9])
    config = AgentConfig(on_event=events.append)
    agent = ReflexionAgent(client, reg, config=config)
    agent.run("task", evaluator=lambda x: next(scores))

    types = [e.type for e in events]
    assert "reflection" in types
    reflection_event = next(e for e in events if e.type == "reflection")
    assert reflection_event.result == "reflection text"


def test_reflexion_timeout():
    """Reflexion returns [timeout exceeded] when timeout is exceeded."""

    call_count = 0

    def slow_chat(*args: object, **kwargs: object) -> Response:
        nonlocal call_count
        call_count += 1
        time.sleep(0.05)
        return Response(text="bad answer", usage=Usage())

    client = MagicMock()
    client.chat = MagicMock(side_effect=slow_chat)
    reg = ToolRegistry()
    config = AgentConfig(timeout=0.05, max_iterations=10)
    agent = ReflexionAgent(client, reg, config=config)
    result = agent.run("task", evaluator=lambda x: 0.1)

    assert result.answer == "[timeout exceeded]"
