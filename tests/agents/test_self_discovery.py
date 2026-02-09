"""Tests for SelfDiscoveryAgent."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from ai_arch_toolkit.agents._base import AgentConfig, AgentEvent
from ai_arch_toolkit.agents._self_discovery import SelfDiscoveryAgent
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


def test_self_discovery_four_phase():
    """SelfDiscovery makes exactly 4 LLM calls (select, adapt, implement, solve)."""
    responses = [
        Response(
            text="Critical Thinking",
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        ),
        Response(
            text="Adapted: Analyze critically",
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        ),
        Response(
            text='{"step1": "Analyze", "step2": "Evaluate"}',
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        ),
        Response(
            text="The answer is 42.",
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        ),
    ]
    client = _make_client(responses)
    reg = ToolRegistry()
    agent = SelfDiscoveryAgent(client, reg)
    result = agent.run("What is the meaning of life?")

    assert result.answer == "The answer is 42."
    assert len(result.steps) == 4
    assert result.total_usage.input_tokens == 40
    assert client.chat.call_count == 4


def test_self_discovery_custom_modules():
    """Custom reasoning modules can be passed via kwargs."""
    responses = [
        Response(text="Math", usage=Usage()),
        Response(text="Adapted Math", usage=Usage()),
        Response(text='{"step1": "compute"}', usage=Usage()),
        Response(text="result", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = ToolRegistry()
    agent = SelfDiscoveryAgent(client, reg)
    result = agent.run(
        "Calculate something",
        reasoning_modules=["Math", "Logic"],
    )

    assert result.answer == "result"
    # Verify the first call included the custom modules
    first_call_args = client.chat.call_args_list[0]
    prompt = first_call_args[0][0][0].content
    assert "Math" in prompt
    assert "Logic" in prompt


def test_self_discovery_async():
    """Async SelfDiscovery works correctly."""
    responses = [
        Response(text="selected", usage=Usage()),
        Response(text="adapted", usage=Usage()),
        Response(text="plan", usage=Usage()),
        Response(text="async result", usage=Usage()),
    ]
    client = _make_async_client(responses)
    reg = ToolRegistry()
    agent = SelfDiscoveryAgent(client, reg)
    result = asyncio.run(agent.async_run("Think about something"))

    assert result.answer == "async result"
    assert len(result.steps) == 4


def test_self_discovery_events():
    """Events are fired for each phase."""
    events: list[AgentEvent] = []
    responses = [
        Response(text="s", usage=Usage()),
        Response(text="a", usage=Usage()),
        Response(text="p", usage=Usage()),
        Response(text="r", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = ToolRegistry()
    config = AgentConfig(on_event=events.append)
    agent = SelfDiscoveryAgent(client, reg, config=config)
    agent.run("task")

    types = [e.type for e in events]
    # 4 phases x (step_start + step_end) = 8 events
    assert types == [
        "step_start",
        "step_end",
        "step_start",
        "step_end",
        "step_start",
        "step_end",
        "step_start",
        "step_end",
    ]


def test_self_discovery_timeout():
    """SelfDiscovery returns [timeout exceeded] when timeout is exceeded."""

    def slow_chat(*args: object, **kwargs: object) -> Response:
        time.sleep(0.05)
        return Response(text="phase result", usage=Usage())

    client = MagicMock()
    client.chat = MagicMock(side_effect=slow_chat)
    reg = ToolRegistry()
    config = AgentConfig(timeout=0.05)
    agent = SelfDiscoveryAgent(client, reg, config=config)
    result = agent.run("test")

    assert result.answer == "[timeout exceeded]"


def test_self_discovery_adapt_prompt_includes_selected():
    """The ADAPT call (2nd) receives the output of SELECT (1st)."""
    responses = [
        Response(text="Critical Thinking\nSystems Thinking", usage=Usage()),
        Response(text="adapted", usage=Usage()),
        Response(text="plan", usage=Usage()),
        Response(text="answer", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = ToolRegistry()
    agent = SelfDiscoveryAgent(client, reg)
    agent.run("test task")

    # 2nd call is ADAPT — its prompt should contain SELECT output
    adapt_call = client.chat.call_args_list[1]
    adapt_prompt = adapt_call[0][0][0].content
    assert "Critical Thinking" in adapt_prompt
    assert "Systems Thinking" in adapt_prompt


def test_self_discovery_implement_mentions_json():
    """The IMPLEMENT call (3rd) prompt mentions JSON structure."""
    responses = [
        Response(text="selected", usage=Usage()),
        Response(text="adapted modules", usage=Usage()),
        Response(text='{"step": "do it"}', usage=Usage()),
        Response(text="answer", usage=Usage()),
    ]
    client = _make_client(responses)
    reg = ToolRegistry()
    agent = SelfDiscoveryAgent(client, reg)
    agent.run("test task")

    # 3rd call is IMPLEMENT — its prompt should mention JSON
    impl_call = client.chat.call_args_list[2]
    impl_prompt = impl_call[0][0][0].content
    assert "JSON" in impl_prompt
