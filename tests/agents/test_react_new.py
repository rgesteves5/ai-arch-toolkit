"""Tests for the new ReActAgent built on core/ primitives."""

from __future__ import annotations

from unittest.mock import AsyncMock

from ai_arch_toolkit.agents._base import AgentConfig, AgentEvent, AgentResult, _add_usage
from ai_arch_toolkit.agents._react import ReActAgent
from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._tools._group import ToolGroup
from tests.agents.conftest import make_response, make_tool_call

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    side_effect: list[Response],
    *,
    config: AgentConfig | None = None,
    tool_fn=None,
) -> ReActAgent:
    """Create a ReActAgent with a mocked LLM and a simple tool."""

    def dummy_tool(x: str = "") -> str:
        """A test tool."""
        return f"result:{x}"

    fn = tool_fn or dummy_tool
    tools = ToolGroup(fn)
    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=side_effect)
    return ReActAgent(llm, tools, config=config)


# ---------------------------------------------------------------------------
# 1. Direct answer — no tool calls
# ---------------------------------------------------------------------------


async def test_direct_answer():
    agent = _make_agent([make_response(text="Hello!")])
    result = await agent.run("Hi")

    assert isinstance(result, AgentResult)
    assert result.answer == "Hello!"
    assert result.stop_reason == "completed"
    assert len(result.steps) == 1
    assert result.steps[0].step == 1
    assert result.total_usage.input_tokens == 10
    assert result.total_cost == 0.001


# ---------------------------------------------------------------------------
# 2. One tool call cycle
# ---------------------------------------------------------------------------


async def test_one_tool_call_cycle():
    tc = make_tool_call(name="dummy_tool", input={"x": "hi"})
    responses = [
        make_response(tool_calls=(tc,)),
        make_response(text="The result was result:hi"),
    ]
    agent = _make_agent(responses)
    result = await agent.run("Use the tool")

    assert result.stop_reason == "completed"
    assert len(result.steps) == 2
    assert result.answer == "The result was result:hi"
    assert result.total_usage.input_tokens == 20  # 10 + 10
    # Verify tool call info propagated to step
    assert len(result.steps[0].tool_calls) == 1
    assert result.steps[0].tool_calls[0].id == "tc_1"
    assert result.steps[0].tool_calls[0].name == "dummy_tool"


# ---------------------------------------------------------------------------
# 3. Multiple parallel tool calls — verify both events emitted
# ---------------------------------------------------------------------------


async def test_parallel_tool_calls():
    tc1 = make_tool_call(name="dummy_tool", input={"x": "a"}, id="tc_1")
    tc2 = make_tool_call(name="dummy_tool", input={"x": "b"}, id="tc_2")
    responses = [
        make_response(tool_calls=(tc1, tc2)),
        make_response(text="Done"),
    ]
    events_received: list[AgentEvent] = []
    config = AgentConfig(parallel_tool_calls=True, on_event=events_received.append)
    agent = _make_agent(responses, config=config)
    result = await agent.run("Use tools")

    assert result.stop_reason == "completed"
    assert len(result.steps) == 2

    # Both tool_call events emitted
    tc_events = [e for e in events_received if e.type == "tool_call"]
    assert len(tc_events) == 2
    tc_ids = {e.tool_call_id for e in tc_events}
    assert tc_ids == {"tc_1", "tc_2"}

    # Both tool_result events emitted
    tr_events = [e for e in events_received if e.type == "tool_result"]
    assert len(tr_events) == 2


# ---------------------------------------------------------------------------
# 4. Max iterations
# ---------------------------------------------------------------------------


async def test_max_iterations():
    tc = make_tool_call(name="dummy_tool")
    # Always returns tool calls — never a final answer
    responses = [make_response(tool_calls=(tc,)) for _ in range(5)]
    agent = _make_agent(responses, config=AgentConfig(max_iterations=3))
    result = await agent.run("Loop forever")

    assert result.stop_reason == "max_iterations"


# ---------------------------------------------------------------------------
# 5. Timeout
# ---------------------------------------------------------------------------


async def test_timeout(monkeypatch):
    import ai_arch_toolkit.agents._react as react_mod

    # Make monotonic() return increasing values so (now - start) > timeout
    _counter = {"n": 0}

    def fake_monotonic():
        _counter["n"] += 1
        return _counter["n"] * 100.0

    monkeypatch.setattr(react_mod.time, "monotonic", fake_monotonic)

    tc = make_tool_call(name="dummy_tool")
    responses = [
        make_response(tool_calls=(tc,)),
        make_response(text="done"),
    ]
    agent = _make_agent(responses, config=AgentConfig(timeout=0.001))
    result = await agent.run("Test timeout")

    assert result.stop_reason == "timeout"


# ---------------------------------------------------------------------------
# 6. Budget exhausted
# ---------------------------------------------------------------------------


async def test_budget_exhausted():
    tc = make_tool_call(name="dummy_tool")
    big_usage = Usage(input_tokens=500, output_tokens=500)
    responses = [
        make_response(tool_calls=(tc,), usage=big_usage),
        make_response(text="done"),
    ]
    agent = _make_agent(responses, config=AgentConfig(max_tokens=100))
    result = await agent.run("Test budget")

    assert result.stop_reason == "budget_exhausted"


# ---------------------------------------------------------------------------
# 7. Tool error — error event emitted, agent continues
# ---------------------------------------------------------------------------


async def test_tool_error():
    tc = make_tool_call(name="dummy_tool", input={"x": "hi"}, id="tc_err")

    def dummy_tool(x: str = "") -> str:
        """A test tool."""
        raise ValueError("tool broke")

    events_received: list[AgentEvent] = []
    config = AgentConfig(on_event=events_received.append)
    responses = [
        make_response(tool_calls=(tc,)),
        make_response(text="Handled error"),
    ]
    agent = _make_agent(responses, tool_fn=dummy_tool, config=config)
    result = await agent.run("Use broken tool")

    assert result.stop_reason == "completed"
    assert result.answer == "Handled error"

    # Verify error event was emitted with correct info
    error_events = [e for e in events_received if e.type == "error"]
    assert len(error_events) == 1
    assert "tool broke" in error_events[0].error
    assert error_events[0].tool_name == "dummy_tool"
    assert error_events[0].tool_call_id == "tc_err"


# ---------------------------------------------------------------------------
# 8. Event callback — exact event sequence, each fires exactly once
# ---------------------------------------------------------------------------


async def test_event_callback_exact_sequence():
    events_received: list[AgentEvent] = []
    config = AgentConfig(on_event=events_received.append)

    tc = make_tool_call(name="dummy_tool", input={"x": "a"}, id="tc_seq")
    responses = [
        make_response(tool_calls=(tc,)),
        make_response(text="Final"),
    ]
    agent = _make_agent(responses, config=config)
    await agent.run("Test events")

    types = [e.type for e in events_received]
    # Step 1: start → tool_call → tool_result → end (mid-run)
    # Step 2: start → end (final answer)
    assert types == [
        "step_start",
        "tool_call",
        "tool_result",
        "step_end",
        "step_start",
        "step_end",
    ]

    # Verify no double-firing — count should match exactly
    assert len(events_received) == 6

    # Verify tool_call event carries correct IDs
    tc_event = events_received[1]
    assert tc_event.tool_name == "dummy_tool"
    assert tc_event.tool_call_id == "tc_seq"
    assert tc_event.tool_args == {"x": "a"}


# ---------------------------------------------------------------------------
# 9. run(stream=True) — yields events, does NOT fire on_event
# ---------------------------------------------------------------------------


async def test_stream_mode_no_callback():
    callback_events: list[AgentEvent] = []
    config = AgentConfig(on_event=callback_events.append)

    tc = make_tool_call(name="dummy_tool")
    responses = [
        make_response(tool_calls=(tc,)),
        make_response(text="Done"),
    ]
    agent = _make_agent(responses, config=config)
    stream_events = []
    async for event in await agent.run("Stream", stream=True):
        stream_events.append(event)

    # Events are yielded to the caller
    assert len(stream_events) > 0
    assert any(e.type == "step_start" for e in stream_events)
    assert any(e.type == "step_end" for e in stream_events)

    # on_event callback should NOT be called in stream mode
    assert len(callback_events) == 0


# ---------------------------------------------------------------------------
# 10. run_sync() — sync wrapper works
# ---------------------------------------------------------------------------


def test_run_sync():
    agent = _make_agent([make_response(text="Sync answer")])
    result = agent.run_sync("Sync test")

    assert isinstance(result, AgentResult)
    assert result.answer == "Sync answer"
    assert result.stop_reason == "completed"


# ---------------------------------------------------------------------------
# 11. run_sync(stream=True) — sync stream works
# ---------------------------------------------------------------------------


def test_run_sync_stream():
    agent = _make_agent([make_response(text="Streamed")])
    events = list(agent.run_sync("Stream sync", stream=True))

    assert len(events) > 0
    assert any(e.type == "step_end" for e in events)


# ---------------------------------------------------------------------------
# 12. System prompt forwarding
# ---------------------------------------------------------------------------


async def test_system_prompt_forwarded():
    agent = _make_agent(
        [make_response(text="ok")],
        config=AgentConfig(system="You are helpful"),
    )
    await agent.run("Test system")

    call_kwargs = agent.llm.complete.call_args
    assert call_kwargs.kwargs["system"] == "You are helpful"


# ---------------------------------------------------------------------------
# 13. tool_choice forwarding
# ---------------------------------------------------------------------------


async def test_tool_choice_forwarded():
    agent = _make_agent(
        [make_response(text="ok")],
        config=AgentConfig(tool_choice="auto"),
    )
    await agent.run("Test tool_choice")

    call_kwargs = agent.llm.complete.call_args
    assert call_kwargs.kwargs["tool_choice"] == "auto"


# ---------------------------------------------------------------------------
# 14. _add_usage helper
# ---------------------------------------------------------------------------


def test_add_usage():
    a = Usage(input_tokens=10, output_tokens=5, cache_write_tokens=2, cache_read_tokens=1)
    b = Usage(input_tokens=20, output_tokens=10, cache_write_tokens=3, cache_read_tokens=4)
    result = _add_usage(a, b)

    assert result.input_tokens == 30
    assert result.output_tokens == 15
    assert result.cache_write_tokens == 5
    assert result.cache_read_tokens == 5


# ---------------------------------------------------------------------------
# 15. LLM error during complete()
# ---------------------------------------------------------------------------


async def test_llm_error():
    events_received: list[AgentEvent] = []
    config = AgentConfig(on_event=events_received.append)
    agent = _make_agent([], config=config)
    agent.llm.complete = AsyncMock(side_effect=RuntimeError("API down"))
    result = await agent.run("Fail")

    assert result.stop_reason == "error"

    # Verify error event was emitted
    error_events = [e for e in events_received if e.type == "error"]
    assert len(error_events) == 1
    assert "API down" in error_events[0].error


# ---------------------------------------------------------------------------
# 16. Intermediate step_end has no stop_reason
# ---------------------------------------------------------------------------


async def test_intermediate_step_end_no_stop_reason():
    """Intermediate step_end events (mid-run tool steps) should not have stop_reason."""
    events_received: list[AgentEvent] = []
    config = AgentConfig(on_event=events_received.append)

    tc = make_tool_call(name="dummy_tool")
    responses = [
        make_response(tool_calls=(tc,)),
        make_response(text="Final"),
    ]
    agent = _make_agent(responses, config=config)
    await agent.run("Test intermediate")

    step_ends = [e for e in events_received if e.type == "step_end"]
    assert len(step_ends) == 2

    # First step_end is intermediate (tool call step) — no stop_reason
    assert step_ends[0].stop_reason is None
    # Last step_end is final — has stop_reason
    assert step_ends[1].stop_reason == "completed"


# ---------------------------------------------------------------------------
# 17. tool_call_id propagated through AgentStep
# ---------------------------------------------------------------------------


async def test_tool_call_id_in_step():
    tc = make_tool_call(name="dummy_tool", input={"x": "v"}, id="tc_abc")
    responses = [
        make_response(tool_calls=(tc,)),
        make_response(text="ok"),
    ]
    agent = _make_agent(responses)
    result = await agent.run("Check IDs")

    step = result.steps[0]
    assert step.tool_calls[0].id == "tc_abc"
    assert step.tool_results[0]["tool_use_id"] == "tc_abc"
