"""Tests for _tools/_group.py — ToolGroup."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._response import ToolCall
from ai_arch_toolkit.core._tools._decorator import tool
from ai_arch_toolkit.core._tools._group import ToolGroup


@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny in {city}"


@tool
def search(query: str) -> dict:
    """Search the web."""
    return {"results": [query]}


@tool
async def async_fetch(url: str) -> str:
    """Fetch a URL."""
    return f"content_of_{url}"


def plain_function(x: int) -> int:
    """Double a number."""
    return x * 2


class TestToolGroup:
    def test_from_decorated(self):
        group = ToolGroup(get_weather, search)
        assert len(group) == 2
        assert "get_weather" in group
        assert "search" in group

    def test_definitions(self):
        group = ToolGroup(get_weather)
        defs = group.definitions
        assert len(defs) == 1
        assert defs[0]["name"] == "get_weather"
        assert "input_schema" in defs[0]

    def test_execute(self):
        group = ToolGroup(get_weather, search)
        tc = ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"})
        result = group.execute(tc)
        assert result == "Sunny in NYC"

    def test_execute_json_result(self):
        group = ToolGroup(search)
        tc = ToolCall(id="tc_1", name="search", input={"query": "test"})
        result = group.execute(tc)
        assert result == '{"results": ["test"]}'

    def test_execute_unknown_raises(self):
        group = ToolGroup(get_weather)
        tc = ToolCall(id="tc_1", name="missing", input={})
        with pytest.raises(KeyError, match="missing"):
            group.execute(tc)

    async def test_async_execute_sync_fn(self):
        group = ToolGroup(get_weather)
        tc = ToolCall(id="tc_1", name="get_weather", input={"city": "LA"})
        result = await group.async_execute(tc)
        assert result == "Sunny in LA"

    async def test_async_execute_async_fn(self):
        group = ToolGroup(async_fetch)
        tc = ToolCall(id="tc_1", name="async_fetch", input={"url": "http://example.com"})
        result = await group.async_execute(tc)
        assert result == "content_of_http://example.com"

    async def test_async_execute_unknown_raises(self):
        group = ToolGroup(get_weather)
        tc = ToolCall(id="tc_1", name="missing", input={})
        with pytest.raises(KeyError, match="missing"):
            await group.async_execute(tc)

    def test_plain_function_auto_inferred(self):
        group = ToolGroup(plain_function)
        assert "plain_function" in group
        defs = group.definitions
        assert defs[0]["name"] == "plain_function"
        assert defs[0]["description"] == "Double a number."

    def test_plain_function_execute(self):
        group = ToolGroup(plain_function)
        tc = ToolCall(id="tc_1", name="plain_function", input={"x": 5})
        result = group.execute(tc)
        assert result == "10"

    def test_add(self):
        group = ToolGroup()
        assert len(group) == 0
        group.add(get_weather)
        assert len(group) == 1

    def test_contains(self):
        group = ToolGroup(get_weather)
        assert "get_weather" in group
        assert "missing" not in group

    def test_repr(self):
        group = ToolGroup(get_weather, search)
        r = repr(group)
        assert "ToolGroup" in r
        assert "get_weather" in r
        assert "search" in r
