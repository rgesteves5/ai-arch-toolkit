"""Tests for _tools/__init__.py — prepare_tools."""

from __future__ import annotations

from ai_arch_toolkit.core._tools import ToolGroup, prepare_tools
from ai_arch_toolkit.core._tools._decorator import tool


@tool
def get_weather(city: str) -> str:
    """Get weather."""
    return f"Sunny in {city}"


@tool
def search(query: str) -> str:
    """Search."""
    return query


def plain_fn(x: int) -> int:
    """Double."""
    return x * 2


class TestPrepareTools:
    def test_none(self):
        assert prepare_tools(None) is None

    def test_single_decorated(self):
        result = prepare_tools(get_weather)
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"

    def test_list_of_decorated(self):
        result = prepare_tools([get_weather, search])
        assert result is not None
        assert len(result) == 2
        names = {d["name"] for d in result}
        assert names == {"get_weather", "search"}

    def test_tool_group(self):
        group = ToolGroup(get_weather, search)
        result = prepare_tools(group)
        assert result is not None
        assert len(result) == 2

    def test_list_of_dicts(self):
        dicts = [
            {"name": "fn1", "description": "d1", "input_schema": {"type": "object"}},
            {"name": "fn2", "description": "d2", "input_schema": {"type": "object"}},
        ]
        result = prepare_tools(dicts)
        assert result is not None
        assert len(result) == 2
        assert result[0]["name"] == "fn1"

    def test_mixed_list(self):
        raw = {"name": "raw", "description": "Raw tool", "input_schema": {"type": "object"}}
        result = prepare_tools([get_weather, raw])
        assert result is not None
        assert len(result) == 2
        names = {d["name"] for d in result}
        assert names == {"get_weather", "raw"}

    def test_list_with_tool_group(self):
        group = ToolGroup(search)
        result = prepare_tools([get_weather, group])
        assert result is not None
        assert len(result) == 2

    def test_plain_callable_auto_inferred(self):
        result = prepare_tools([plain_fn])
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "plain_fn"

    def test_empty_list_returns_none(self):
        assert prepare_tools([]) is None

    def test_input_schema_key_used(self):
        """All prepared tools use input_schema (not parameters)."""
        result = prepare_tools([get_weather])
        assert result is not None
        assert "input_schema" in result[0]
        assert "parameters" not in result[0]
