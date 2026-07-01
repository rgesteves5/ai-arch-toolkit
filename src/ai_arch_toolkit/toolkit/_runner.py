"""Tool runner — execute all tool calls from a Response and return results.

Routes every call through the common governed + metered executor (``execute_tool`` /
``async_execute_tool``): the same approval gate, dry-run, and metering a Flow gets. It never calls
the raw function directly — that path bypassed both governance and the meter.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ai_arch_toolkit.core._content import tool_result
from ai_arch_toolkit.core._response import Response
from ai_arch_toolkit.core._tools._approval import ApprovalHandler
from ai_arch_toolkit.core._tools._executor import (
    _format_result,
    _resolve_fn,
    async_execute_tool,
    execute_tool,
)
from ai_arch_toolkit.core._tools._group import ToolGroup


def _normalize_tools(tools: list[Callable[..., Any]] | ToolGroup) -> list[Callable[..., Any]]:
    """Accept a ToolGroup or list of callables."""
    if isinstance(tools, ToolGroup):
        return tools.tools
    return tools


async def run_tools(
    response: Response,
    tools: list[Callable[..., Any]] | ToolGroup,
    *,
    approval_handler: ApprovalHandler | None = None,
) -> list[dict[str, Any]]:
    """Execute all tool calls in a response and return tool_result messages.

    Each call goes through ``async_execute_tool`` — approval gate, dry-run, and metering (when a
    :class:`~ai_arch_toolkit.core.MeterScope` is bound) all apply. An unknown tool still raises
    ``KeyError``; a tool that raises is returned as an error result (never propagated).

    Args:
        response: An LLM response (potentially containing tool_calls).
        tools: List of callable tools or a ToolGroup to search.
        approval_handler: Optional handler for tools that require approval.

    Returns:
        A list of tool_result message dicts, one per tool call. Empty if there are none.
    """
    if not response.has_tool_calls:
        return []

    fns = _normalize_tools(tools)
    results: list[dict[str, Any]] = []
    for tc in response.tool_calls:
        _resolve_fn(tc, fns)  # preserve the KeyError-on-unknown-tool contract (pre-flight)
        result = await async_execute_tool(tc, fns, approval_handler=approval_handler)
        results.append(tool_result(_format_result(result), tool_use_id=tc.id, name=tc.name))
    return results


def run_tools_sync(
    response: Response,
    tools: list[Callable[..., Any]] | ToolGroup,
    *,
    approval_handler: ApprovalHandler | None = None,
) -> list[dict[str, Any]]:
    """Execute all tool calls synchronously and return tool_result messages.

    Sync counterpart of :func:`run_tools` — same governance and metering via ``execute_tool``.

    Args:
        response: An LLM response (potentially containing tool_calls).
        tools: List of callable tools or a ToolGroup to search.
        approval_handler: Optional handler for tools that require approval.

    Returns:
        A list of tool_result message dicts, one per tool call. Empty if there are none.
    """
    if not response.has_tool_calls:
        return []

    fns = _normalize_tools(tools)
    results: list[dict[str, Any]] = []
    for tc in response.tool_calls:
        _resolve_fn(tc, fns)  # preserve the KeyError-on-unknown-tool contract (pre-flight)
        result = execute_tool(tc, fns, approval_handler=approval_handler)
        results.append(tool_result(_format_result(result), tool_use_id=tc.id, name=tc.name))
    return results
