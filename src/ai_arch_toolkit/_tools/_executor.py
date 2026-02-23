"""Tool execution — call functions from LLM tool_call results."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable
from typing import Any

from ai_arch_toolkit._response import ToolCall


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
    if isinstance(result, str):
        return result
    return json.dumps(result)


def execute_tool(tool_call: ToolCall, tools: list[Callable[..., Any]]) -> str:
    """Execute a tool call synchronously, returning a string result.

    Args:
        tool_call: The ToolCall from an LLM response.
        tools: List of decorated tool functions to search.
    """
    fn = _resolve_fn(tool_call, tools)
    result = fn(**tool_call.input)
    return _format_result(result)


async def async_execute_tool(tool_call: ToolCall, tools: list[Callable[..., Any]]) -> str:
    """Execute a tool call asynchronously, returning a string result.

    Args:
        tool_call: The ToolCall from an LLM response.
        tools: List of decorated tool functions to search.
    """
    fn = _resolve_fn(tool_call, tools)
    if inspect.iscoroutinefunction(fn):
        result = await fn(**tool_call.input)
    else:
        result = await asyncio.to_thread(fn, **tool_call.input)
    return _format_result(result)
