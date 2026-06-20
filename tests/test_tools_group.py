"""Tests for _tools/_group.py — ToolGroup with governance."""

from __future__ import annotations

import asyncio

from ai_arch_toolkit.core._response import ToolCall
from ai_arch_toolkit.core._tools._approval import ApprovalDecision
from ai_arch_toolkit.core._tools._decorator import tool
from ai_arch_toolkit.core._tools._governance import DangerousToolGate, DryRunGate
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


class TestToolGroupBasics:
    def test_from_decorated(self):
        group = ToolGroup(get_weather, search)
        assert len(group) == 2
        assert "get_weather" in group
        assert "search" in group

    def test_definitions_are_provider_safe(self):
        group = ToolGroup(dangerous_echo)
        defs = group.definitions
        assert len(defs) == 1
        assert defs[0]["name"] == "dangerous_echo"
        assert "input_schema" in defs[0]
        # No runtime governance metadata leaks into provider schemas.
        assert "requires_approval" not in defs[0]
        assert "risk_level" not in defs[0]
        assert "capability" not in defs[0]

    def test_runtime_definitions_carry_policy(self):
        group = ToolGroup(dangerous_echo)
        rt = group.runtime_definitions
        assert rt[0].policy.requires_approval is True
        assert rt[0].policy.risk_level == "critical"

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

    def test_plain_function_auto_inferred(self):
        group = ToolGroup(plain_function)
        assert "plain_function" in group
        defs = group.definitions
        assert defs[0]["name"] == "plain_function"
        assert defs[0]["description"] == "Double a number."


class TestExecute:
    def test_success(self):
        group = ToolGroup(get_weather, search)
        tc = ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"})
        result = group.execute(tc)
        assert result.ok is True
        assert result.value == "Sunny in NYC"

    def test_json_value(self):
        group = ToolGroup(search)
        tc = ToolCall(id="tc_1", name="search", input={"query": "test"})
        result = group.execute(tc)
        assert result.to_model_text() == '{"results": ["test"]}'

    def test_unknown(self):
        group = ToolGroup(get_weather)
        tc = ToolCall(id="tc_1", name="missing", input={})
        result = group.execute(tc)
        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "unknown_tool"

    def test_validation_error(self):
        group = ToolGroup(get_weather)
        tc = ToolCall(id="tc_1", name="get_weather", input={})
        result = group.execute(tc)
        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "validation_error"

    def test_runtime_error(self):
        group = ToolGroup(explode)
        tc = ToolCall(id="tc_1", name="explode", input={})
        result = group.execute(tc)
        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "runtime_error"

    def test_plain_function(self):
        group = ToolGroup(plain_function)
        tc = ToolCall(id="tc_1", name="plain_function", input={"x": 5})
        result = group.execute(tc)
        assert result.value == 10

    async def test_async_sync_fn(self):
        group = ToolGroup(get_weather)
        tc = ToolCall(id="tc_1", name="get_weather", input={"city": "LA"})
        result = await group.async_execute(tc)
        assert result.value == "Sunny in LA"

    async def test_async_async_fn(self):
        group = ToolGroup(async_fetch)
        tc = ToolCall(id="tc_1", name="async_fetch", input={"url": "http://example.com"})
        result = await group.async_execute(tc)
        assert result.value == "content_of_http://example.com"


class TestApproval:
    def test_missing_handler_denies(self):
        group = ToolGroup(dangerous_echo)
        tc = ToolCall(id="tc_1", name="dangerous_echo", input={"command": "rm -rf /"})
        result = group.execute(tc)
        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "approval_denied"

    def test_approved_with_audit(self):
        group = ToolGroup(
            dangerous_echo,
            approval_handler=lambda _: ApprovalDecision.approve(reviewer="human"),
        )
        tc = ToolCall(id="tc_1", name="dangerous_echo", input={"command": "echo ok"})
        result = group.execute(tc)
        assert result.ok is True
        assert result.value == "echo ok"
        assert result.metadata["audit"]["approval"]["decision"]["reviewer"] == "human"

    async def test_async_approved_modified_args(self):
        async def approve(_request):
            return ApprovalDecision.approve(modified_args={"command": "echo safe"})

        group = ToolGroup(dangerous_echo, approval_handler=approve)
        tc = ToolCall(id="tc_1", name="dangerous_echo", input={"command": "rm -rf /"})
        result = await group.async_execute(tc)
        assert result.ok is True
        assert result.value == "echo safe"

    async def test_handler_preserved_when_composed_with_gates(self):
        """Regression: composing governance gates must not drop the handler."""

        async def approve(_request):
            return ApprovalDecision.approve(reviewer="human")

        group = ToolGroup(
            dangerous_echo,
            approval_handler=approve,
            gates=(DangerousToolGate(blocked={"other"}, allow=False),),
        )
        tc = ToolCall(id="tc_1", name="dangerous_echo", input={"command": "echo ok"})
        result = await group.async_execute(tc)
        assert result.ok is True
        assert result.value == "echo ok"


class TestGovernanceGates:
    def test_dangerous_blocked(self):
        group = ToolGroup(
            get_weather,
            gates=(DangerousToolGate(blocked={"get_weather"}, allow=False),),
        )
        tc = ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"})
        result = group.execute(tc)
        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "dangerous_tool_blocked"

    def test_dangerous_allowed(self):
        group = ToolGroup(
            get_weather,
            gates=(DangerousToolGate(blocked={"get_weather"}, allow=True),),
        )
        tc = ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"})
        result = group.execute(tc)
        assert result.ok is True

    def test_dry_run_does_not_execute(self):
        calls: list[str] = []

        @tool
        def record(text: str) -> str:
            """Record a call."""
            calls.append(text)
            return text

        group = ToolGroup(record, gates=(DryRunGate(dry_run=True),))
        tc = ToolCall(id="tc_1", name="record", input={"text": "hi"})
        result = group.execute(tc)
        assert result.ok is True
        assert result.metadata["governance"]["outcome"] == "dry_run"
        assert result.metadata["governance"]["executed"] is False
        assert calls == []
        # The model-facing text never includes raw arguments.
        assert "hi" not in result.to_model_text()


class TestCallBudget:
    def test_max_calls_blocks_after_limit(self):
        group = ToolGroup(get_weather, max_calls=1)
        tc = ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"})
        first = group.execute(tc)
        second = group.execute(tc)
        assert first.ok is True
        assert second.ok is False
        assert second.error is not None
        assert second.error.type == "max_calls_exceeded"

    def test_blocked_call_does_not_consume_budget(self):
        # dry-run short-circuits before the budget commit, so it never counts.
        group = ToolGroup(get_weather, max_calls=1, gates=(DryRunGate(dry_run=True),))
        tc = ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"})
        first = group.execute(tc)
        second = group.execute(tc)
        assert first.metadata["governance"]["outcome"] == "dry_run"
        assert second.metadata["governance"]["outcome"] == "dry_run"

    def test_reset_restores_budget(self):
        group = ToolGroup(get_weather, max_calls=1)
        tc = ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"})
        group.execute(tc)
        assert group.execute(tc).ok is False
        group.reset()
        assert group.execute(tc).ok is True

    def test_approval_denied_does_not_consume_budget(self):
        # No handler → dangerous_echo is denied by approval before the budget commit.
        group = ToolGroup(get_weather, dangerous_echo, max_calls=1)
        denied = group.execute(
            ToolCall(id="t1", name="dangerous_echo", input={"command": "echo x"})
        )
        assert denied.error is not None
        assert denied.error.type == "approval_denied"
        # The denied call must not have consumed the single budget slot.
        allowed = group.execute(ToolCall(id="t2", name="get_weather", input={"city": "NYC"}))
        assert allowed.ok is True

    async def test_max_calls_atomic_under_gather(self):
        """Concurrent calls must not exceed the budget (atomic reserve)."""
        limit = 3
        group = ToolGroup(async_fetch, max_calls=limit)
        tc = ToolCall(id="tc", name="async_fetch", input={"url": "x"})
        results = await asyncio.gather(*[group.async_execute(tc) for _ in range(limit + 5)])
        executed = [r for r in results if r.ok]
        blocked = [r for r in results if not r.ok]
        assert len(executed) == limit
        assert all(r.error.type == "max_calls_exceeded" for r in blocked if r.error)
