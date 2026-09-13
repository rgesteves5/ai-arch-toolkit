"""Tool runner — execute all tool calls from a Response and return results.

Every call goes through the common governed + metered executor, never the raw function (that path
bypassed both governance and the meter). A ``ToolGroup`` runs each call through its own governance
(``group.async_execute`` / ``group.execute``: its gates, approval handler, and ``max_calls``
budget); a list of callables goes through ``async_execute_tool`` / ``execute_tool`` and the
approval gate.
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


def _reject_handler_with_group(
    tools: list[Callable[..., Any]] | ToolGroup, approval_handler: ApprovalHandler | None
) -> None:
    """Refuse ``approval_handler=`` next to a ``ToolGroup``: the handler belongs to the group."""
    if isinstance(tools, ToolGroup) and approval_handler is not None:
        msg = (
            "approval_handler= cannot be combined with a ToolGroup, which applies its own "
            "governance; pass it to ToolGroup(..., approval_handler=...) instead."
        )
        raise ValueError(msg)


def _require_known_tools(response: Response, tools: list[Callable[..., Any]] | ToolGroup) -> None:
    """Raise ``KeyError`` for an unknown tool before any call in the response runs.

    Checked up front so a response naming an unknown tool never half-executes: an earlier call's
    side effects would otherwise happen and its ``tool_result`` be lost with the exception.
    """
    for tool_call in response.tool_calls:
        if isinstance(tools, ToolGroup):
            if tool_call.name not in tools:
                msg = f"Unknown tool: {tool_call.name!r}"
                raise KeyError(msg)
        else:
            _resolve_fn(tool_call, tools)


async def run_tools(
    response: Response,
    tools: list[Callable[..., Any]] | ToolGroup,
    *,
    approval_handler: ApprovalHandler | None = None,
) -> list[dict[str, Any]]:
    """Execute all tool calls in a response and return tool_result messages.

    With a :class:`ToolGroup`, each call runs through ``group.async_execute`` — the group's gates,
    approval handler, and ``max_calls`` budget. With a list of callables, each call goes through
    ``async_execute_tool`` and its approval gate. Metering applies either way when a
    :class:`~ai_arch_toolkit.core.MeterScope` is bound. A tool that raises is returned as an error
    result (never propagated).

    Args:
        response: An LLM response (potentially containing tool_calls).
        tools: List of callable tools, or a ToolGroup whose governance applies.
        approval_handler: Handler for tools that require approval when ``tools`` is a list. A
            ToolGroup takes its handler at construction.

    Returns:
        A list of tool_result message dicts, one per tool call. Empty if there are none.

    Raises:
        KeyError: A tool call names a tool that ``tools`` does not contain; no call runs.
        ValueError: ``approval_handler`` is passed together with a ToolGroup.
    """
    _reject_handler_with_group(tools, approval_handler)
    if not response.has_tool_calls:
        return []

    _require_known_tools(response, tools)
    results: list[dict[str, Any]] = []
    for tc in response.tool_calls:
        if isinstance(tools, ToolGroup):
            result = await tools.async_execute(tc)
        else:
            result = await async_execute_tool(tc, tools, approval_handler=approval_handler)
        results.append(tool_result(_format_result(result), tool_use_id=tc.id, name=tc.name))
    return results


def run_tools_sync(
    response: Response,
    tools: list[Callable[..., Any]] | ToolGroup,
    *,
    approval_handler: ApprovalHandler | None = None,
) -> list[dict[str, Any]]:
    """Execute all tool calls synchronously and return tool_result messages.

    Sync counterpart of :func:`run_tools`: a ToolGroup runs each call through ``group.execute``, a
    list of callables through ``execute_tool``, with the same governance and metering.

    Args:
        response: An LLM response (potentially containing tool_calls).
        tools: List of callable tools, or a ToolGroup whose governance applies.
        approval_handler: Handler for tools that require approval when ``tools`` is a list. A
            ToolGroup takes its handler at construction.

    Returns:
        A list of tool_result message dicts, one per tool call. Empty if there are none.

    Raises:
        KeyError: A tool call names a tool that ``tools`` does not contain; no call runs.
        ValueError: ``approval_handler`` is passed together with a ToolGroup.
    """
    _reject_handler_with_group(tools, approval_handler)
    if not response.has_tool_calls:
        return []

    _require_known_tools(response, tools)
    results: list[dict[str, Any]] = []
    for tc in response.tool_calls:
        if isinstance(tools, ToolGroup):
            result = tools.execute(tc)
        else:
            result = execute_tool(tc, tools, approval_handler=approval_handler)
        results.append(tool_result(_format_result(result), tool_use_id=tc.id, name=tc.name))
    return results
