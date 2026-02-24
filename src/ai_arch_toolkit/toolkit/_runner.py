"""Tool runner — execute all tool calls from a Response and return results."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

from ai_arch_toolkit.core._content import tool_result
from ai_arch_toolkit.core._response import Response
from ai_arch_toolkit.core._tools._executor import _format_result, _resolve_fn
from ai_arch_toolkit.core._tools._group import ToolGroup


def _normalize_tools(tools: list[Callable[..., Any]] | ToolGroup) -> list[Callable[..., Any]]:
    """Accept a ToolGroup or list of callables."""
    if isinstance(tools, ToolGroup):
        return list(tools._fns.values())
    return tools


async def run_tools(
    response: Response,
    tools: list[Callable[..., Any]] | ToolGroup,
) -> list[dict[str, Any]]:
    """Execute all tool calls in a response and return tool_result messages.

    Args:
        response: An LLM response (potentially containing tool_calls).
        tools: List of callable tools or a ToolGroup to search.

    Returns:
        A list of tool_result message dicts, one per tool call.
        Empty list if the response has no tool calls.
    """
    if not response.has_tool_calls:
        return []

    fns = _normalize_tools(tools)
    results: list[dict[str, Any]] = []
    for tc in response.tool_calls:
        fn = _resolve_fn(tc, fns)
        if inspect.iscoroutinefunction(fn):
            result = await fn(**tc.input)
        else:
            result = await asyncio.to_thread(fn, **tc.input)
        results.append(tool_result(_format_result(result), tool_use_id=tc.id))
    return results


def run_tools_sync(
    response: Response,
    tools: list[Callable[..., Any]] | ToolGroup,
) -> list[dict[str, Any]]:
    """Execute all tool calls synchronously and return tool_result messages.

    Args:
        response: An LLM response (potentially containing tool_calls).
        tools: List of callable tools or a ToolGroup to search.

    Returns:
        A list of tool_result message dicts, one per tool call.
        Empty list if the response has no tool calls.
    """
    if not response.has_tool_calls:
        return []

    fns = _normalize_tools(tools)
    results: list[dict[str, Any]] = []
    for tc in response.tool_calls:
        fn = _resolve_fn(tc, fns)
        result = fn(**tc.input)
        results.append(tool_result(_format_result(result), tool_use_id=tc.id))
    return results
