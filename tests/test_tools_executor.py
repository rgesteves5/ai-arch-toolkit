"""Tests for _tools/_executor.py — the single structured execution pipeline."""

from __future__ import annotations

import asyncio
import gc
import json
import warnings
from collections.abc import Generator
from typing import Any

from ai_arch_toolkit.core._response import ToolCall
from ai_arch_toolkit.core._tools._approval import ApprovalDecision
from ai_arch_toolkit.core._tools._decorator import tool
from ai_arch_toolkit.core._tools._executor import (
    async_execute_tool,
    execute_tool,
)
from ai_arch_toolkit.core._tools._group import ToolGroup


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


# --- Awaitables on either path (async tools run from sync code, and vice versa) ------------


@tool
async def async_greet(name: str) -> str:
    """Greet someone after yielding to the event loop."""
    await asyncio.sleep(0)
    return f"hello {name}"


def deferred_lookup(key: str) -> Any:
    """A sync function that hands back a coroutine instead of a value."""
    return async_lookup(key)


class _Deferred:
    """A minimal awaitable that is not a coroutine."""

    def __init__(self, value: object) -> None:
        self._value = value

    def __await__(self) -> Generator[Any, None, object]:
        return self._resolve().__await__()

    async def _resolve(self) -> object:
        await asyncio.sleep(0)
        return self._value


def deferred_object(key: str) -> _Deferred:
    """A sync function that returns a non-coroutine awaitable."""
    return _Deferred(f"object_for_{key}")


def _never_awaited(caught: list[warnings.WarningMessage]) -> list[str]:
    return [str(w.message) for w in caught if "never awaited" in str(w.message)]


class TestAsyncToolOnSyncPath:
    def test_group_execute_awaits_async_tool(self):
        group = ToolGroup(async_greet)
        tc = ToolCall(id="tc_1", name="async_greet", input={"name": "ada"})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = group.execute(tc)
            assert result.ok is True
            assert result.value == "hello ada"
            assert result.to_model_text() == "hello ada"
            del result
            gc.collect()
        assert _never_awaited(caught) == []

    def test_execute_tool_awaits_async_tool(self):
        tc = ToolCall(id="tc_1", name="async_lookup", input={"key": "foo"})
        result = execute_tool(tc, [async_lookup])
        assert result.ok is True
        assert result.value == "value_for_foo"

    async def test_group_execute_inside_a_running_loop(self):
        # A running loop sends the sync path through _run_sync's worker thread.
        group = ToolGroup(async_greet)
        tc = ToolCall(id="tc_1", name="async_greet", input={"name": "bob"})
        result = group.execute(tc)
        assert result.ok is True
        assert result.value == "hello bob"

    def test_sync_function_returning_a_coroutine(self):
        tc = ToolCall(id="tc_1", name="deferred_lookup", input={"key": "k"})
        result = execute_tool(tc, [deferred_lookup])
        assert result.ok is True
        assert result.value == "value_for_k"

    def test_sync_function_returning_a_non_coroutine_awaitable(self):
        tc = ToolCall(id="tc_1", name="deferred_object", input={"key": "k"})
        result = execute_tool(tc, [deferred_object])
        assert result.ok is True
        assert result.value == "object_for_k"

    def test_async_tool_that_raises_is_a_runtime_error(self):
        @tool
        async def async_explode() -> str:
            """Fail after yielding to the loop."""
            await asyncio.sleep(0)
            raise RuntimeError("async boom")

        tc = ToolCall(id="tc_1", name="async_explode", input={})
        result = ToolGroup(async_explode).execute(tc)
        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "runtime_error"
        assert "async boom" in result.error.message


class TestAwaitableOnAsyncPath:
    async def test_sync_function_returning_a_coroutine(self):
        tc = ToolCall(id="tc_1", name="deferred_lookup", input={"key": "k"})
        result = await async_execute_tool(tc, [deferred_lookup])
        assert result.ok is True
        assert result.value == "value_for_k"

    async def test_sync_function_returning_a_non_coroutine_awaitable(self):
        tc = ToolCall(id="tc_1", name="deferred_object", input={"key": "k"})
        result = await async_execute_tool(tc, [deferred_object])
        assert result.ok is True
        assert result.value == "object_for_k"


# --- Positional-only parameters ------------------------------------------------------------


@tool
def square(x: int, /) -> int:
    """Square a number."""
    return x * x


@tool
async def async_square(x: int, /) -> int:
    """Square a number asynchronously."""
    return x * x


def plain_square(x: int, /) -> int:
    """Square a number (undecorated)."""
    return x * x


@tool
def scale(value: float, /, factor: float = 2.0, *, offset: float = 0.0) -> float:
    """Scale a value, then shift it."""
    return value * factor + offset


@tool
def span(start: int = 0, stop: int = 10, /) -> int:
    """Length of a range."""
    return stop - start


class TestPositionalOnlyParameters:
    def test_decorated_tool_on_sync_path(self):
        tc = ToolCall(id="tc_1", name="square", input={"x": 3})
        result = ToolGroup(square).execute(tc)
        assert result.ok is True
        assert result.value == 9

    async def test_decorated_tool_on_async_path(self):
        tc = ToolCall(id="tc_1", name="square", input={"x": 3})
        result = await ToolGroup(square).async_execute(tc)
        assert result.ok is True
        assert result.value == 9

    async def test_async_tool_on_both_paths(self):
        tc = ToolCall(id="tc_1", name="async_square", input={"x": 4})
        group = ToolGroup(async_square)
        assert (await group.async_execute(tc)).value == 16
        assert group.execute(tc).value == 16

    async def test_undecorated_callable_on_both_paths(self):
        tc = ToolCall(id="tc_1", name="plain_square", input={"x": 5})
        assert execute_tool(tc, [plain_square]).value == 25
        assert (await async_execute_tool(tc, [plain_square])).value == 25

    def test_mixed_with_keyword_parameters(self):
        tc = ToolCall(id="tc_1", name="scale", input={"offset": 1.0, "factor": 3.0, "value": 3.0})
        result = execute_tool(tc, [scale])
        assert result.ok is True
        assert result.value == 10.0

    def test_omitted_positional_only_parameter_takes_its_default(self):
        # `stop` can only be passed by position, so the omitted `start` before it is filled in.
        tc = ToolCall(id="tc_1", name="span", input={"stop": 4})
        result = execute_tool(tc, [span])
        assert result.ok is True
        assert result.value == 4

    def test_missing_required_positional_only_is_a_validation_error(self):
        tc = ToolCall(id="tc_1", name="square", input={})
        result = execute_tool(tc, [square])
        assert result.ok is False
        assert result.error is not None
        assert result.error.type == "validation_error"
