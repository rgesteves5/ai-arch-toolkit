"""Tool execution — call functions from LLM tool_call results."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

from ai_arch_toolkit.core._response import ToolCall
from ai_arch_toolkit.core._tools._result import ToolResult, _format_value


def _resolve_fn(tool_call: ToolCall, tools: list[Callable[..., Any]]) -> Callable[..., Any]:
    """Find the callable matching a tool call name.

    Matches by ``__tool__["name"]`` first, then falls back to ``__name__``.
    The ``__tool__`` attribute is set by the ``@tool`` decorator; plain
    functions without it are matched by their Python name.
    """
    # First pass: prefer __tool__["name"] (explicit, unambiguous)
    for fn in tools:
        tool_def = getattr(fn, "__tool__", None)
        if tool_def is not None and tool_def.get("name") == tool_call.name:
            return fn
    # Second pass: fall back to __name__ for non-decorated functions
    for fn in tools:
        if not hasattr(fn, "__tool__") and getattr(fn, "__name__", None) == tool_call.name:
            return fn
    msg = f"Unknown tool: {tool_call.name!r}"
    raise KeyError(msg)


def _format_result(result: Any) -> str:
    """Convert a tool result to string for LLM consumption."""
    if isinstance(result, ToolResult):
        return result.to_model_text()
    return _format_value(result)


def _result_from_exception(tool_name: str, exc: Exception) -> ToolResult:
    """Convert an exception raised during tool execution to a structured result."""
    if isinstance(exc, TypeError):
        return ToolResult.failure(
            "validation_error",
            f"Tool {tool_name!r} argument mismatch: {exc}",
            details={"tool_name": tool_name},
        )
    return ToolResult.failure(
        "runtime_error",
        str(exc),
        retryable=True,
        details={
            "tool_name": tool_name,
            "exception_type": type(exc).__name__,
        },
    )


def _coerce_result(value: Any) -> ToolResult:
    """Normalize a tool return value into a ToolResult."""
    if isinstance(value, ToolResult):
        return value
    return ToolResult.success(value)


def execute_tool_result(
    tool_call: ToolCall,
    tools: list[Callable[..., Any]],
) -> ToolResult:
    """Execute a tool call synchronously, returning a structured result.

    Args:
        tool_call: The ToolCall from an LLM response.
        tools: List of decorated tool functions to search.
    """
    try:
        fn = _resolve_fn(tool_call, tools)
    except KeyError:
        return ToolResult.failure(
            "unknown_tool",
            f"Unknown tool: {tool_call.name!r}",
            details={"tool_name": tool_call.name},
        )

    try:
        return _coerce_result(fn(**tool_call.input))
    except Exception as exc:
        return _result_from_exception(tool_call.name, exc)


async def async_execute_tool_result(
    tool_call: ToolCall,
    tools: list[Callable[..., Any]],
) -> ToolResult:
    """Execute a tool call asynchronously, returning a structured result.

    Args:
        tool_call: The ToolCall from an LLM response.
        tools: List of decorated tool functions to search.
    """
    try:
        fn = _resolve_fn(tool_call, tools)
    except KeyError:
        return ToolResult.failure(
            "unknown_tool",
            f"Unknown tool: {tool_call.name!r}",
            details={"tool_name": tool_call.name},
        )

    try:
        if inspect.iscoroutinefunction(fn):
            return _coerce_result(await fn(**tool_call.input))
        return _coerce_result(await asyncio.to_thread(fn, **tool_call.input))
    except Exception as exc:
        return _result_from_exception(tool_call.name, exc)


def execute_tool(tool_call: ToolCall, tools: list[Callable[..., Any]]) -> str:
    """Execute a tool call synchronously, returning a string result.

    Args:
        tool_call: The ToolCall from an LLM response.
        tools: List of decorated tool functions to search.
    """
    fn = _resolve_fn(tool_call, tools)
    try:
        result = fn(**tool_call.input)
    except TypeError as exc:
        raise TypeError(f"Tool {tool_call.name!r} argument mismatch: {exc}") from exc
    return _format_result(result)


async def async_execute_tool(tool_call: ToolCall, tools: list[Callable[..., Any]]) -> str:
    """Execute a tool call asynchronously, returning a string result.

    Args:
        tool_call: The ToolCall from an LLM response.
        tools: List of decorated tool functions to search.
    """
    fn = _resolve_fn(tool_call, tools)
    try:
        if inspect.iscoroutinefunction(fn):
            result = await fn(**tool_call.input)
        else:
            result = await asyncio.to_thread(fn, **tool_call.input)
    except TypeError as exc:
        raise TypeError(f"Tool {tool_call.name!r} argument mismatch: {exc}") from exc
    return _format_result(result)
