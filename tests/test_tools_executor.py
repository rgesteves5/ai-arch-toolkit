"""Tests for _tools/_executor.py — the single structured execution pipeline."""

from __future__ import annotations

import json

from ai_arch_toolkit.core._response import ToolCall
from ai_arch_toolkit.core._tools._approval import ApprovalDecision
from ai_arch_toolkit.core._tools._decorator import tool
from ai_arch_toolkit.core._tools._executor import (
    async_execute_tool,
    execute_tool,
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


@tool
def leak_secret() -> str:
    """Raise an error containing a secret."""
    raise RuntimeError("auth failed token=sk-supersecretvalue123456")


@tool(
    capability="shell",
    risk_level="critical",
    requires_approval=True,
    approval_reason="Needs review.",
)
def dangerous_echo(command: str) -> str:
    """Echo a dangerous command."""
    return command


def plain_function(x: int) -> int:
    """Double a number (undecorated)."""
    return x * 2


class TestExecuteTool:
    def test_success_result(self):
        tc = ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"})
        result = execute_tool(tc, [get_weather])
        assert result.ok is True
        assert result.value == "Sunny in NYC"
        assert result.error is None

    def test_non_string_value_preserved(self):
        tc = ToolCall(id="tc_1", name="multiply", input={"a": 3, "b": 4})
        result = execute_tool(tc, [multiply])
        assert result.ok is True
        assert result.value == {"result": 12}
        assert result.to_model_text() == '{"result": 12}'

    def test_validation_error(self):
        tc = ToolCall(id="tc_1", name="get_weather", input={})
        result = execute_tool(tc, [get_weather])
        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "validation_error"
        assert result.error.details["tool_name"] == "get_weather"

    def test_runtime_error(self):
        tc = ToolCall(id="tc_1", name="fail_hard", input={})
        result = execute_tool(tc, [fail_hard])
        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "runtime_error"
        assert result.error.retryable is True
        assert result.error.details["exception_type"] == "RuntimeError"
        # Exception text is redacted, not hidden — useful message survives.
        assert "boom" in result.error.message

    def test_runtime_error_redacts_secret_in_message(self):
        tc = ToolCall(id="tc_1", name="leak_secret", input={})
        result = execute_tool(tc, [leak_secret])
        assert result.ok is False
        assert "sk-supersecretvalue123456" not in result.to_model_text()
        assert "sk-supersecretvalue123456" not in (result.error.message if result.error else "")

    def test_unknown_tool(self):
        tc = ToolCall(id="tc_1", name="unknown", input={})
        result = execute_tool(tc, [get_weather])
        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "unknown_tool"
        assert result.error.details["tool_name"] == "unknown"

    def test_finds_by_tool_name(self):
        @tool(name="custom_name")
        def fn(x: str) -> str:
            """Do stuff."""
            return x

        tc = ToolCall(id="tc_1", name="custom_name", input={"x": "hello"})
        result = execute_tool(tc, [fn])
        assert result.value == "hello"

    def test_plain_undecorated_callable(self):
        tc = ToolCall(id="tc_1", name="plain_function", input={"x": 5})
        result = execute_tool(tc, [plain_function])
        assert result.ok is True
        assert result.value == 10


class TestApprovalGate:
    def test_missing_handler_denies_by_default(self):
        tc = ToolCall(id="tc_1", name="dangerous_echo", input={"command": "rm -rf /tmp/x"})
        result = execute_tool(tc, [dangerous_echo])
        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "approval_denied"
        assert "approval" in result.metadata["audit"]

    def test_approved_tool_executes_with_audit(self):
        tc = ToolCall(id="tc_1", name="dangerous_echo", input={"command": "echo ok"})

        def approve(request):
            assert request.tool_name == "dangerous_echo"
            assert request.capability == "shell"
            assert request.risk_level == "critical"
            return ApprovalDecision.approve(reviewer="human")

        result = execute_tool(tc, [dangerous_echo], approval_handler=approve)
        assert result.ok is True
        assert result.value == "echo ok"
        assert result.metadata["audit"]["approval"]["decision"]["reviewer"] == "human"

    def test_approval_can_modify_arguments(self):
        tc = ToolCall(id="tc_1", name="dangerous_echo", input={"command": "rm -rf /tmp/x"})
        result = execute_tool(
            tc,
            [dangerous_echo],
            approval_handler=lambda _: ApprovalDecision.approve(
                modified_args={"command": "echo safe"}
            ),
        )
        assert result.ok is True
        assert result.value == "echo safe"

    def test_denied_tool_does_not_execute(self):
        tc = ToolCall(id="tc_1", name="dangerous_echo", input={"command": "echo no"})
        result = execute_tool(
            tc,
            [dangerous_echo],
            approval_handler=lambda _: ApprovalDecision.deny(reason="not allowed"),
        )
        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "approval_denied"
        assert result.metadata["audit"]["approval"]["decision"]["reason"] == "not allowed"

    def test_sync_path_denies_async_handler(self):
        tc = ToolCall(id="tc_1", name="dangerous_echo", input={"command": "echo ok"})

        async def approve(_request):
            return ApprovalDecision.approve()

        result = execute_tool(tc, [dangerous_echo], approval_handler=approve)
        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "approval_denied"

    def test_handler_receives_unredacted_arguments(self):
        """Trust boundary: the handler sees real args; only stored audit is redacted."""
        secret = "sk-supersecretvalue123456"
        tc = ToolCall(id="tc_1", name="dangerous_echo", input={"command": f"deploy {secret}"})
        seen: dict[str, str] = {}

        def approve(request):
            seen["command"] = request.arguments["command"]
            return ApprovalDecision.approve(reviewer="human")

        result = execute_tool(tc, [dangerous_echo], approval_handler=approve)
        assert result.ok is True
        assert secret in seen["command"]

    def test_secret_in_argument_redacted_in_audit(self):
        """A secret passed as a tool argument is stripped from stored audit metadata."""
        secret = "sk-supersecretvalue123456"
        tc = ToolCall(id="tc_1", name="dangerous_echo", input={"command": f"deploy {secret}"})
        result = execute_tool(
            tc,
            [dangerous_echo],
            approval_handler=lambda _: ApprovalDecision.approve(reviewer="human"),
        )
        assert result.ok is True
        audit_blob = json.dumps(result.metadata["audit"])
        assert secret not in audit_blob


class TestAsyncExecuteTool:
    async def test_sync_function(self):
        tc = ToolCall(id="tc_1", name="get_weather", input={"city": "LA"})
        result = await async_execute_tool(tc, [get_weather])
        assert result.ok is True
        assert result.value == "Sunny in LA"

    async def test_async_function(self):
        tc = ToolCall(id="tc_1", name="async_lookup", input={"key": "foo"})
        result = await async_execute_tool(tc, [async_lookup])
        assert result.ok is True
        assert result.value == "value_for_foo"

    async def test_async_unknown_tool(self):
        tc = ToolCall(id="tc_1", name="missing", input={})
        result = await async_execute_tool(tc, [get_weather])
        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "unknown_tool"

    async def test_async_approval_handler(self):
        tc = ToolCall(id="tc_1", name="dangerous_echo", input={"command": "echo ok"})

        async def approve(_request):
            return ApprovalDecision.approve(reviewer="async-human")

        result = await async_execute_tool(tc, [dangerous_echo], approval_handler=approve)
        assert result.ok is True
        assert result.value == "echo ok"
        assert result.metadata["audit"]["approval"]["decision"]["reviewer"] == "async-human"
