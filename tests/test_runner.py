"""Tests for _tools/_runner.py — run_tools and run_tools_sync."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._response import Response, ToolCall, Usage
from ai_arch_toolkit.core._tools._decorator import tool
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit._runner import run_tools, run_tools_sync

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"


@tool
def get_time(tz: str) -> dict:
    """Get the current time."""
    return {"time": "12:00", "tz": tz}


@tool
async def async_search(query: str) -> str:
    """Search for something."""
    return f"Results for: {query}"


def _make_response(*tool_calls: ToolCall) -> Response:
    return Response(
        text="",
        tool_calls=tuple(tool_calls),
        usage=Usage(input_tokens=10, output_tokens=5),
    )


# ---------------------------------------------------------------------------
# run_tools (async)
# ---------------------------------------------------------------------------


class TestRunTools:
    async def test_no_tool_calls(self):
        r = Response(text="Hello")
        results = await run_tools(r, [get_weather])
        assert results == []

    async def test_single_tool_call(self):
        r = _make_response(ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"}))
        results = await run_tools(r, [get_weather])
        assert len(results) == 1
        assert results[0]["role"] == "tool"
        assert results[0]["tool_use_id"] == "tc_1"
        assert results[0]["content"] == "Sunny in NYC"

    async def test_multiple_tool_calls(self):
        r = _make_response(
            ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"}),
            ToolCall(id="tc_2", name="get_time", input={"tz": "UTC"}),
        )
        results = await run_tools(r, [get_weather, get_time])
        assert len(results) == 2
        assert results[0]["tool_use_id"] == "tc_1"
        assert results[0]["content"] == "Sunny in NYC"
        assert results[1]["tool_use_id"] == "tc_2"
        assert results[1]["content"] == '{"time": "12:00", "tz": "UTC"}'

    async def test_async_tool(self):
        r = _make_response(ToolCall(id="tc_1", name="async_search", input={"query": "test"}))
        results = await run_tools(r, [async_search])
        assert len(results) == 1
        assert results[0]["content"] == "Results for: test"

    async def test_unknown_tool_raises(self):
        r = _make_response(ToolCall(id="tc_1", name="nonexistent", input={}))

        with pytest.raises(KeyError, match="nonexistent"):
            await run_tools(r, [get_weather])

    async def test_result_format_is_tool_result(self):
        """Results match the _content.tool_result() format."""
        r = _make_response(ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"}))
        results = await run_tools(r, [get_weather])
        msg = results[0]
        assert msg["role"] == "tool"
        assert "tool_use_id" in msg
        assert msg["name"] == "get_weather"
        assert "content" in msg


# ---------------------------------------------------------------------------
# run_tools_sync
# ---------------------------------------------------------------------------


class TestRunToolsSync:
    def test_no_tool_calls(self):
        r = Response(text="Hello")
        results = run_tools_sync(r, [get_weather])
        assert results == []

    def test_single_tool_call(self):
        r = _make_response(ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"}))
        results = run_tools_sync(r, [get_weather])
        assert len(results) == 1
        assert results[0]["role"] == "tool"
        assert results[0]["tool_use_id"] == "tc_1"
        assert results[0]["content"] == "Sunny in NYC"

    def test_multiple_tool_calls(self):
        r = _make_response(
            ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"}),
            ToolCall(id="tc_2", name="get_time", input={"tz": "UTC"}),
        )
        results = run_tools_sync(r, [get_weather, get_time])
        assert len(results) == 2
        assert results[0]["content"] == "Sunny in NYC"
        assert results[1]["content"] == '{"time": "12:00", "tz": "UTC"}'

    def test_non_string_result_json_serialized(self):
        r = _make_response(ToolCall(id="tc_1", name="get_time", input={"tz": "UTC"}))
        results = run_tools_sync(r, [get_time])
        assert results[0]["content"] == '{"time": "12:00", "tz": "UTC"}'


# ---------------------------------------------------------------------------
# ToolGroup support
# ---------------------------------------------------------------------------


class TestRunToolsWithToolGroup:
    async def test_async_with_tool_group(self):
        group = ToolGroup(get_weather, get_time)
        r = _make_response(ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"}))
        results = await run_tools(r, group)
        assert len(results) == 1
        assert results[0]["content"] == "Sunny in NYC"

    def test_sync_with_tool_group(self):
        group = ToolGroup(get_weather, get_time)
        r = _make_response(ToolCall(id="tc_1", name="get_time", input={"tz": "UTC"}))
        results = run_tools_sync(r, group)
        assert len(results) == 1
        assert results[0]["content"] == '{"time": "12:00", "tz": "UTC"}'


# ---------------------------------------------------------------------------
# End-to-end roundtrip
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_to_message_through_anthropic_wire(self):
        """Response → to_message → Anthropic _messages_to_wire → correct format."""
        from ai_arch_toolkit.core._providers._anthropic import _messages_to_sdk as anthropic_wire

        r = Response(
            text="Let me check.",
            tool_calls=(ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"}),),
        )
        # Step 1: to_message
        assistant_msg = r.to_message()

        # Step 2: run_tools
        tool_results = run_tools_sync(r, [get_weather])

        # Step 3: build conversation and convert to wire
        conversation = [
            {"role": "user", "content": "What's the weather?"},
            assistant_msg,
            *tool_results,
        ]
        _sys, wire = anthropic_wire(conversation)

        # Verify user message
        assert wire[0] == {"role": "user", "content": "What's the weather?"}
        # Verify assistant message has content blocks
        assert wire[1]["role"] == "assistant"
        content = wire[1]["content"]
        assert content[0] == {"type": "text", "text": "Let me check."}
        assert content[1]["type"] == "tool_use"
        assert content[1]["name"] == "get_weather"
        assert content[1]["input"] == {"city": "NYC"}
        # Verify tool result
        assert wire[2]["role"] == "user"
        assert wire[2]["content"][0]["type"] == "tool_result"
        assert wire[2]["content"][0]["tool_use_id"] == "tc_1"

    def test_to_message_through_openai_wire(self):
        """Response → to_message → OpenAI _messages_to_sdk → correct format."""
        from ai_arch_toolkit.core._providers._openai import _messages_to_sdk as openai_wire

        r = Response(
            text="Let me check.",
            tool_calls=(ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"}),),
        )
        assistant_msg = r.to_message()
        tool_results = run_tools_sync(r, [get_weather])

        conversation = [
            {"role": "user", "content": "What's the weather?"},
            assistant_msg,
            *tool_results,
        ]
        wire = openai_wire(conversation)

        # Verify user message
        assert wire[0] == {"role": "user", "content": "What's the weather?"}
        # Verify assistant message has function tool_calls
        assert wire[1]["role"] == "assistant"
        assert wire[1]["content"] == "Let me check."
        tc = wire[1]["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == '{"city": "NYC"}'
        # Verify tool result
        assert wire[2]["role"] == "tool"
        assert wire[2]["tool_call_id"] == "tc_1"


# ---------------------------------------------------------------------------
# Governance + metering (run_tools now routes through the gated/metered executor)
# ---------------------------------------------------------------------------


@tool(requires_approval=True)
def deploy(target: str) -> str:
    """Deploy to a target (requires approval)."""
    return f"deployed to {target}"


class TestRunToolsGovernance:
    async def test_approval_gate_blocks_without_a_handler(self):
        # The old run_tools ran this ungated — the bypass. Now the gate applies.
        r = _make_response(ToolCall(id="tc_1", name="deploy", input={"target": "prod"}))
        results = await run_tools(r, [deploy])
        assert "deployed to prod" not in results[0]["content"]  # blocked, not executed

    async def test_approval_handler_lets_it_run(self):
        from ai_arch_toolkit.core._tools._approval import ApprovalDecision

        def approve(_request):
            return ApprovalDecision.approve(reviewer="human")

        r = _make_response(ToolCall(id="tc_1", name="deploy", input={"target": "prod"}))
        results = await run_tools(r, [deploy], approval_handler=approve)
        assert results[0]["content"] == "deployed to prod"

    async def test_run_tools_is_metered_under_a_scope(self):
        from ai_arch_toolkit.core._metering._scope import MeterScope

        r = _make_response(
            ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"}),
            ToolCall(id="tc_2", name="get_time", input={"tz": "UTC"}),
        )
        with MeterScope() as scope:
            await run_tools(r, [get_weather, get_time])
        assert scope.snapshot().tool_calls == 2

    def test_run_tools_sync_is_metered_under_a_scope(self):
        from ai_arch_toolkit.core._metering._scope import MeterScope

        r = _make_response(ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"}))
        with MeterScope() as scope:
            run_tools_sync(r, [get_weather])
        assert scope.snapshot().tool_calls == 1
