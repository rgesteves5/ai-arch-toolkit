"""Tests for _tools/_executor.py — tool execution."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._response import ToolCall
from ai_arch_toolkit.core._tools._decorator import tool
from ai_arch_toolkit.core._tools._executor import async_execute_tool, execute_tool


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
