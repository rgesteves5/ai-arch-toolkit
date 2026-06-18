"""Tests for _tools/_executor.py — tool execution."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._response import ToolCall
from ai_arch_toolkit.core._tools._decorator import tool
from ai_arch_toolkit.core._tools._executor import (
    async_execute_tool,
    async_execute_tool_result,
    execute_tool,
    execute_tool_result,
)


@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny in {city}"


@tool
def multiply(a: int, b: int) -> dict:
    """Multiply two numbers."""
    return {"result": a * b}


@tool
async def async_lookup(key: str) -> str:
    """Async lookup."""
    return f"value_for_{key}"


@tool
def fail_hard() -> str:
    """Raise a runtime error."""
    raise RuntimeError("boom")


class TestExecuteTool:
    def test_basic_execution(self):
        tc = ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"})
        result = execute_tool(tc, [get_weather])
        assert result == "Sunny in NYC"

    def test_non_string_result_json_dumped(self):
        tc = ToolCall(id="tc_1", name="multiply", input={"a": 3, "b": 4})
        result = execute_tool(tc, [multiply])
        assert result == '{"result": 12}'

    def test_unknown_tool_raises(self):
        tc = ToolCall(id="tc_1", name="unknown", input={})
        with pytest.raises(KeyError, match="unknown"):
            execute_tool(tc, [get_weather])

    def test_finds_by_tool_name(self):
        """Finds function via __tool__['name'], not __name__."""

        @tool(name="custom_name")
        def fn(x: str) -> str:
            """Do stuff."""
            return x

        tc = ToolCall(id="tc_1", name="custom_name", input={"x": "hello"})
        result = execute_tool(tc, [fn])
        assert result == "hello"

    def test_multiple_tools(self):
        tc = ToolCall(id="tc_1", name="multiply", input={"a": 2, "b": 5})
        result = execute_tool(tc, [get_weather, multiply])
        assert result == '{"result": 10}'


class TestExecuteToolResult:
    def test_success_result(self):
        tc = ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"})
        result = execute_tool_result(tc, [get_weather])
        assert result.ok is True
        assert result.value == "Sunny in NYC"
        assert result.error is None

    def test_validation_error_result(self):
        tc = ToolCall(id="tc_1", name="get_weather", input={})
        result = execute_tool_result(tc, [get_weather])
        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "validation_error"
        assert result.error.details["tool_name"] == "get_weather"

    def test_runtime_error_result(self):
        tc = ToolCall(id="tc_1", name="fail_hard", input={})
        result = execute_tool_result(tc, [fail_hard])
        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "runtime_error"
        assert result.error.retryable is True
        assert result.error.details["exception_type"] == "RuntimeError"

    def test_unknown_tool_result(self):
        tc = ToolCall(id="tc_1", name="unknown", input={})
        result = execute_tool_result(tc, [get_weather])
        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "unknown_tool"
        assert result.error.details["tool_name"] == "unknown"


class TestAsyncExecuteTool:
    async def test_sync_function(self):
        tc = ToolCall(id="tc_1", name="get_weather", input={"city": "LA"})
        result = await async_execute_tool(tc, [get_weather])
        assert result == "Sunny in LA"

    async def test_async_function(self):
        tc = ToolCall(id="tc_1", name="async_lookup", input={"key": "foo"})
        result = await async_execute_tool(tc, [async_lookup])
        assert result == "value_for_foo"

    async def test_unknown_tool_raises(self):
        tc = ToolCall(id="tc_1", name="missing", input={})
        with pytest.raises(KeyError, match="missing"):
            await async_execute_tool(tc, [get_weather])

    async def test_async_success_result(self):
        tc = ToolCall(id="tc_1", name="async_lookup", input={"key": "foo"})
        result = await async_execute_tool_result(tc, [async_lookup])
        assert result.ok is True
        assert result.value == "value_for_foo"

    async def test_async_unknown_tool_result(self):
        tc = ToolCall(id="tc_1", name="missing", input={})
        result = await async_execute_tool_result(tc, [get_weather])
        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "unknown_tool"
