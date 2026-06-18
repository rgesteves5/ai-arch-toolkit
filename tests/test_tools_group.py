"""Tests for _tools/_group.py — ToolGroup."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._response import ToolCall
from ai_arch_toolkit.core._tools._approval import ApprovalDecision
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


@tool
def explode() -> str:
    """Raise a runtime error."""
    raise RuntimeError("boom")


@tool(capability="shell", risk_level="critical", requires_approval=True)
def dangerous_echo(command: str) -> str:
    """Echo a dangerous command."""
    return command


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

    def test_execute_result_success(self):
        group = ToolGroup(get_weather)
        tc = ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"})
        result = group.execute_result(tc)
        assert result.ok is True
        assert result.value == "Sunny in NYC"

    def test_execute_result_validation_error(self):
        group = ToolGroup(get_weather)
        tc = ToolCall(id="tc_1", name="get_weather", input={})
        result = group.execute_result(tc)
        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "validation_error"

    def test_execute_result_runtime_error(self):
        group = ToolGroup(explode)
        tc = ToolCall(id="tc_1", name="explode", input={})
        result = group.execute_result(tc)
        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "runtime_error"

    def test_execute_result_unknown(self):
        group = ToolGroup(get_weather)
        tc = ToolCall(id="tc_1", name="missing", input={})
        result = group.execute_result(tc)
        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "unknown_tool"

    def test_execute_result_missing_approval_handler_denies(self):
        group = ToolGroup(dangerous_echo)
        tc = ToolCall(id="tc_1", name="dangerous_echo", input={"command": "rm -rf /"})
        result = group.execute_result(tc)
        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "approval_denied"

    def test_execute_result_approved(self):
        group = ToolGroup(
            dangerous_echo,
            approval_handler=lambda _: ApprovalDecision.approve(reviewer="human"),
        )
        tc = ToolCall(id="tc_1", name="dangerous_echo", input={"command": "echo ok"})
        result = group.execute_result(tc)
        assert result.ok is True
        assert result.value == "echo ok"
        assert result.metadata["approval_decision"]["reviewer"] == "human"

    def test_execute_string_api_denied_raises_permission_error(self):
        group = ToolGroup(dangerous_echo)
        tc = ToolCall(id="tc_1", name="dangerous_echo", input={"command": "rm -rf /"})
        with pytest.raises(PermissionError, match="requires approval"):
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

    async def test_async_execute_result_success(self):
        group = ToolGroup(async_fetch)
        tc = ToolCall(id="tc_1", name="async_fetch", input={"url": "http://example.com"})
        result = await group.async_execute_result(tc)
        assert result.ok is True
        assert result.value == "content_of_http://example.com"

    async def test_async_execute_result_unknown(self):
        group = ToolGroup(get_weather)
        tc = ToolCall(id="tc_1", name="missing", input={})
        result = await group.async_execute_result(tc)
        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "unknown_tool"

    async def test_async_execute_result_approved(self):
        async def approve(_request):
            return ApprovalDecision.approve(modified_args={"command": "echo safe"})

        group = ToolGroup(dangerous_echo, approval_handler=approve)
        tc = ToolCall(id="tc_1", name="dangerous_echo", input={"command": "rm -rf /"})
        result = await group.async_execute_result(tc)
        assert result.ok is True
        assert result.value == "echo safe"

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
