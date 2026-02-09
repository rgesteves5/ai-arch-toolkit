"""Tests for ReActAgent."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from ai_arch_toolkit.agents._base import AgentConfig, AgentEvent
from ai_arch_toolkit.agents._react import ReActAgent
from ai_arch_toolkit.llm._types import Response, Tool, ToolCall, Usage
from ai_arch_toolkit.tools._registry import ToolRegistry


def _make_client(responses: list[Response]) -> MagicMock:
    """Create a mock Client that returns the given responses in order."""
    client = MagicMock()
    client.chat = MagicMock(side_effect=responses)
    return client


def _make_async_client(responses: list[Response]) -> MagicMock:
    """Create a mock AsyncClient that returns the given responses in order."""
    client = MagicMock()
    client.chat = AsyncMock(side_effect=responses)
    return client


def test_direct_answer_no_tools():
    """Agent returns immediately when LLM responds without tool calls."""
    response = Response(
        text="The answer is 42.",
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    client = _make_client([response])
    reg = ToolRegistry()
    agent = ReActAgent(client, reg)
    result = agent.run("What is the answer?")

    assert result.answer == "The answer is 42."
    assert len(result.steps) == 1
    assert result.steps[0].step_number == 1
    assert result.steps[0].tool_calls == ()
    assert result.total_usage.input_tokens == 10
    assert result.total_usage.output_tokens == 5


def test_tool_call_then_answer():
    """Agent executes a tool call and then returns the final answer."""
    tool_response = Response(
        text="",
        tool_calls=(ToolCall(id="tc_1", name="add", arguments={"a": 2, "b": 3}),),
        usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
    )
    final_response = Response(
        text="2 + 3 = 5",
        usage=Usage(input_tokens=30, output_tokens=8, total_tokens=38),
    )
    client = _make_client([tool_response, final_response])
    reg = ToolRegistry()
    tool_def = Tool(name="add", description="Add numbers", parameters={})
    reg.register("add", lambda a, b: a + b, tool_def)
    agent = ReActAgent(client, reg)
    result = agent.run("What is 2 + 3?")

    assert result.answer == "2 + 3 = 5"
    assert len(result.steps) == 2
    # First step has tool calls
    assert result.steps[0].tool_calls == tool_response.tool_calls
    assert len(result.steps[0].tool_results) == 1
    assert result.steps[0].tool_results[0].content == "5"
    # Second step is the final answer
    assert result.steps[1].tool_calls == ()
    # Usage is aggregated
    assert result.total_usage.input_tokens == 50
    assert result.total_usage.output_tokens == 18


def test_max_iterations_reached():
    """Agent stops after max_iterations and returns a fallback answer."""
    tool_call = ToolCall(id="tc_1", name="noop", arguments={})
    responses = [
        Response(
            text="",
            tool_calls=(tool_call,),
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        for _ in range(3)
    ]
    client = _make_client(responses)
    reg = ToolRegistry()
    reg.register("noop", lambda: "ok", Tool(name="noop", description="", parameters={}))
    agent = ReActAgent(client, reg, config=AgentConfig(max_iterations=3))
    result = agent.run("Loop forever")

    assert result.answer == "[max iterations reached]"
    assert len(result.steps) == 3
    assert result.total_usage.input_tokens == 30


def test_kwargs_passed_to_client():
    """Extra kwargs are forwarded to client.chat()."""
    response = Response(text="ok", usage=Usage())
    client = _make_client([response])
    reg = ToolRegistry()
    agent = ReActAgent(client, reg)
    agent.run("test", temperature=0.5, max_tokens=100)

    _, kwargs = client.chat.call_args
    assert kwargs["temperature"] == 0.5
    assert kwargs["max_tokens"] == 100


def test_system_prompt_passed():
    """System prompt from AgentConfig is passed to client.chat()."""
    response = Response(text="ok", usage=Usage())
    client = _make_client([response])
    reg = ToolRegistry()
    config = AgentConfig(system="You are a helpful assistant.")
    agent = ReActAgent(client, reg, config=config)
    agent.run("test")

    _, kwargs = client.chat.call_args
    assert kwargs["system"] == "You are a helpful assistant."


# --- Phase 3: Tool error handling ---


def test_tool_error_feeds_back_to_llm():
    """Tool errors are caught and fed back as ToolResult content."""

    def failing_tool(**kwargs: object) -> str:
        msg = "connection refused"
        raise ConnectionError(msg)

    tool_response = Response(
        text="",
        tool_calls=(ToolCall(id="tc_1", name="fetch", arguments={"url": "x"}),),
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    final_response = Response(
        text="Sorry, the tool failed.",
        usage=Usage(input_tokens=20, output_tokens=5, total_tokens=25),
    )
    client = _make_client([tool_response, final_response])
    reg = ToolRegistry()
    reg.register("fetch", failing_tool, Tool(name="fetch", description="", parameters={}))
    agent = ReActAgent(client, reg)
    result = agent.run("Fetch x")

    assert result.answer == "Sorry, the tool failed."
    assert len(result.steps) == 2
    assert "Error executing tool 'fetch'" in result.steps[0].tool_results[0].content
    assert "connection refused" in result.steps[0].tool_results[0].content


# --- Phase 4: Token budget ---


def test_token_budget_stops_early():
    """Agent stops when token budget is exceeded."""
    tool_response = Response(
        text="",
        tool_calls=(ToolCall(id="tc_1", name="noop", arguments={}),),
        usage=Usage(input_tokens=50, output_tokens=50, total_tokens=100),
    )
    client = _make_client([tool_response])
    reg = ToolRegistry()
    reg.register("noop", lambda: "ok", Tool(name="noop", description="", parameters={}))
    agent = ReActAgent(client, reg, config=AgentConfig(max_tokens=80, max_iterations=10))
    result = agent.run("Do something")

    assert result.answer == "[token budget exceeded]"
    assert len(result.steps) == 1
    assert result.total_usage.total_tokens == 100


def test_token_budget_none_no_limit():
    """When max_tokens is None, no token budget limit is applied."""
    tool_response = Response(
        text="",
        tool_calls=(ToolCall(id="tc_1", name="noop", arguments={}),),
        usage=Usage(input_tokens=500, output_tokens=500, total_tokens=1000),
    )
    final_response = Response(
        text="done",
        usage=Usage(input_tokens=10, output_tokens=10, total_tokens=20),
    )
    client = _make_client([tool_response, final_response])
    reg = ToolRegistry()
    reg.register("noop", lambda: "ok", Tool(name="noop", description="", parameters={}))
    agent = ReActAgent(client, reg, config=AgentConfig(max_tokens=None))
    result = agent.run("Do something")

    assert result.answer == "done"
    assert result.total_usage.total_tokens == 1020


# --- Phase 5: Async agent ---


def test_async_direct_answer():
    """Async agent returns immediately when LLM responds without tool calls."""
    response = Response(
        text="async answer",
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    client = _make_async_client([response])
    reg = ToolRegistry()
    agent = ReActAgent(client, reg)
    result = asyncio.run(agent.async_run("test"))

    assert result.answer == "async answer"
    assert len(result.steps) == 1


def test_async_tool_call_then_answer():
    """Async agent executes a tool call and then returns the final answer."""
    tool_response = Response(
        text="",
        tool_calls=(ToolCall(id="tc_1", name="add", arguments={"a": 1, "b": 2}),),
        usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
    )
    final_response = Response(
        text="1 + 2 = 3",
        usage=Usage(input_tokens=30, output_tokens=8, total_tokens=38),
    )
    client = _make_async_client([tool_response, final_response])
    reg = ToolRegistry()
    reg.register("add", lambda a, b: a + b, Tool(name="add", description="", parameters={}))
    agent = ReActAgent(client, reg)
    result = asyncio.run(agent.async_run("What is 1 + 2?"))

    assert result.answer == "1 + 2 = 3"
    assert len(result.steps) == 2
    assert result.steps[0].tool_results[0].content == "3"


def test_async_max_iterations():
    """Async agent stops after max_iterations."""
    tool_call = ToolCall(id="tc_1", name="noop", arguments={})
    responses = [
        Response(
            text="",
            tool_calls=(tool_call,),
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        for _ in range(2)
    ]
    client = _make_async_client(responses)
    reg = ToolRegistry()
    reg.register("noop", lambda: "ok", Tool(name="noop", description="", parameters={}))
    agent = ReActAgent(client, reg, config=AgentConfig(max_iterations=2))
    result = asyncio.run(agent.async_run("Loop"))

    assert result.answer == "[max iterations reached]"
    assert len(result.steps) == 2


def test_async_tool_error_recovery():
    """Async agent catches tool errors and feeds them back."""

    def failing(**kwargs: object) -> str:
        msg = "boom"
        raise RuntimeError(msg)

    tool_response = Response(
        text="",
        tool_calls=(ToolCall(id="tc_1", name="fail", arguments={}),),
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    final_response = Response(
        text="recovered",
        usage=Usage(input_tokens=20, output_tokens=5, total_tokens=25),
    )
    client = _make_async_client([tool_response, final_response])
    reg = ToolRegistry()
    reg.register("fail", failing, Tool(name="fail", description="", parameters={}))
    agent = ReActAgent(client, reg)
    result = asyncio.run(agent.async_run("Do it"))

    assert result.answer == "recovered"
    assert "Error executing tool 'fail'" in result.steps[0].tool_results[0].content


# --- Phase 6: Streaming execution ---


def test_run_stream_yields_steps():
    """run_stream yields each step including tool calls."""
    tool_response = Response(
        text="",
        tool_calls=(ToolCall(id="tc_1", name="noop", arguments={}),),
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    final_response = Response(
        text="done",
        usage=Usage(input_tokens=20, output_tokens=5, total_tokens=25),
    )
    client = _make_client([tool_response, final_response])
    reg = ToolRegistry()
    reg.register("noop", lambda: "ok", Tool(name="noop", description="", parameters={}))
    agent = ReActAgent(client, reg)
    steps = list(agent.run_stream("test"))

    assert len(steps) == 2
    assert steps[0].step_number == 1
    assert len(steps[0].tool_calls) == 1
    assert steps[1].step_number == 2
    assert steps[1].tool_calls == ()


def test_run_stream_stops_on_answer():
    """run_stream stops after the first response without tool calls."""
    response = Response(
        text="immediate answer",
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    client = _make_client([response])
    reg = ToolRegistry()
    agent = ReActAgent(client, reg)
    steps = list(agent.run_stream("test"))

    assert len(steps) == 1
    assert steps[0].response.text == "immediate answer"


def test_async_run_stream_yields_steps():
    """async_run_stream yields each step."""
    tool_response = Response(
        text="",
        tool_calls=(ToolCall(id="tc_1", name="noop", arguments={}),),
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    final_response = Response(
        text="done",
        usage=Usage(input_tokens=20, output_tokens=5, total_tokens=25),
    )
    client = _make_async_client([tool_response, final_response])
    reg = ToolRegistry()
    reg.register("noop", lambda: "ok", Tool(name="noop", description="", parameters={}))
    agent = ReActAgent(client, reg)

    async def collect_steps() -> list:
        return [step async for step in agent.async_run_stream("test")]

    steps = asyncio.run(collect_steps())

    assert len(steps) == 2
    assert steps[0].step_number == 1
    assert steps[1].step_number == 2
    assert steps[1].response.text == "done"


# --- Phase 7: Events, tool_choice, parallel, timeout ---


def test_events_fired_on_direct_answer():
    """step_start + step_end events are fired for a direct answer."""
    events: list[AgentEvent] = []
    response = Response(text="hi", usage=Usage())
    client = _make_client([response])
    reg = ToolRegistry()
    config = AgentConfig(on_event=events.append)
    agent = ReActAgent(client, reg, config=config)
    agent.run("test")

    types = [e.type for e in events]
    assert types == ["step_start", "step_end"]
    assert events[0].step_number == 1


def test_events_fired_on_tool_call():
    """tool_call + tool_result events are fired when tools are used."""
    events: list[AgentEvent] = []
    tool_response = Response(
        text="",
        tool_calls=(ToolCall(id="tc_1", name="add", arguments={"a": 1, "b": 2}),),
        usage=Usage(),
    )
    final_response = Response(text="3", usage=Usage())
    client = _make_client([tool_response, final_response])
    reg = ToolRegistry()
    reg.register("add", lambda a, b: a + b, Tool(name="add", description="", parameters={}))
    config = AgentConfig(on_event=events.append)
    agent = ReActAgent(client, reg, config=config)
    agent.run("add 1 and 2")

    types = [e.type for e in events]
    assert "tool_call" in types
    assert "tool_result" in types
    tc_event = next(e for e in events if e.type == "tool_call")
    assert tc_event.tool_name == "add"
    assert tc_event.tool_args == {"a": 1, "b": 2}


def test_events_fired_on_tool_error():
    """error event is fired when a tool execution fails."""
    events: list[AgentEvent] = []

    def fail_tool(**kw: object) -> str:
        msg = "boom"
        raise RuntimeError(msg)

    tool_response = Response(
        text="",
        tool_calls=(ToolCall(id="tc_1", name="fail", arguments={}),),
        usage=Usage(),
    )
    final_response = Response(text="recovered", usage=Usage())
    client = _make_client([tool_response, final_response])
    reg = ToolRegistry()
    reg.register("fail", fail_tool, Tool(name="fail", description="", parameters={}))
    config = AgentConfig(on_event=events.append)
    agent = ReActAgent(client, reg, config=config)
    agent.run("do it")

    error_events = [e for e in events if e.type == "error"]
    assert len(error_events) == 1
    assert error_events[0].error == "boom"


def test_tool_choice_passed_to_client():
    """tool_choice from AgentConfig is forwarded to client.chat()."""
    response = Response(text="ok", usage=Usage())
    client = _make_client([response])
    reg = ToolRegistry()
    config = AgentConfig(tool_choice="auto")
    agent = ReActAgent(client, reg, config=config)
    agent.run("test")

    _, kwargs = client.chat.call_args
    assert kwargs["tool_choice"] == "auto"


def test_parallel_tool_execution_sync():
    """Multiple tool calls execute (in parallel by default)."""
    tool_response = Response(
        text="",
        tool_calls=(
            ToolCall(id="tc_1", name="add", arguments={"a": 1, "b": 2}),
            ToolCall(id="tc_2", name="add", arguments={"a": 3, "b": 4}),
        ),
        usage=Usage(),
    )
    final_response = Response(text="done", usage=Usage())
    client = _make_client([tool_response, final_response])
    reg = ToolRegistry()
    reg.register("add", lambda a, b: a + b, Tool(name="add", description="", parameters={}))
    agent = ReActAgent(client, reg)
    result = agent.run("add stuff")

    assert result.answer == "done"
    assert len(result.steps[0].tool_results) == 2
    assert result.steps[0].tool_results[0].content == "3"
    assert result.steps[0].tool_results[1].content == "7"


def test_parallel_tool_execution_disabled():
    """When parallel_tool_execution=False, tools execute sequentially."""
    call_order: list[str] = []

    def tracked_add(a: int, b: int) -> int:
        call_order.append(f"{a}+{b}")
        return a + b

    tool_response = Response(
        text="",
        tool_calls=(
            ToolCall(id="tc_1", name="add", arguments={"a": 1, "b": 2}),
            ToolCall(id="tc_2", name="add", arguments={"a": 3, "b": 4}),
        ),
        usage=Usage(),
    )
    final_response = Response(text="done", usage=Usage())
    client = _make_client([tool_response, final_response])
    reg = ToolRegistry()
    reg.register("add", tracked_add, Tool(name="add", description="", parameters={}))
    config = AgentConfig(parallel_tool_execution=False)
    agent = ReActAgent(client, reg, config=config)
    result = agent.run("add stuff")

    assert result.answer == "done"
    assert call_order == ["1+2", "3+4"]


def test_timeout_exceeded():
    """Agent returns [timeout exceeded] when timeout is exceeded."""

    def slow_chat(*args: object, **kwargs: object) -> Response:
        time.sleep(0.05)
        return Response(
            text="",
            tool_calls=(ToolCall(id="tc_1", name="noop", arguments={}),),
            usage=Usage(),
        )

    client = MagicMock()
    client.chat = MagicMock(side_effect=slow_chat)
    reg = ToolRegistry()
    reg.register("noop", lambda: "ok", Tool(name="noop", description="", parameters={}))
    config = AgentConfig(timeout=0.05, max_iterations=100)
    agent = ReActAgent(client, reg, config=config)
    result = agent.run("loop")

    assert result.answer == "[timeout exceeded]"


def test_timeout_none_no_limit():
    """When timeout is None, no time limit is applied."""
    response = Response(text="done", usage=Usage())
    client = _make_client([response])
    reg = ToolRegistry()
    config = AgentConfig(timeout=None)
    agent = ReActAgent(client, reg, config=config)
    result = agent.run("test")

    assert result.answer == "done"


def test_react_event_ordering_tool_call_before_result():
    """tool_call events fire before tool execution; tool_result events fire after."""
    events: list[AgentEvent] = []
    tool_response = Response(
        text="",
        tool_calls=(
            ToolCall(id="tc_1", name="add", arguments={"a": 1, "b": 2}),
            ToolCall(id="tc_2", name="add", arguments={"a": 3, "b": 4}),
        ),
        usage=Usage(),
    )
    final_response = Response(text="done", usage=Usage())
    client = _make_client([tool_response, final_response])
    reg = ToolRegistry()
    reg.register("add", lambda a, b: a + b, Tool(name="add", description="", parameters={}))
    config = AgentConfig(on_event=events.append)
    agent = ReActAgent(client, reg, config=config)
    agent.run("add stuff")

    # All tool_call events should appear before all tool_result events
    tool_events = [e for e in events if e.type in ("tool_call", "tool_result")]
    types = [e.type for e in tool_events]
    # With 2 tools: [tool_call, tool_call, tool_result, tool_result]
    assert types == ["tool_call", "tool_call", "tool_result", "tool_result"]
