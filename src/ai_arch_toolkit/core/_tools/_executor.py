"""Tool execution — call functions from LLM tool_call results."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any, NoReturn

from ai_arch_toolkit.core._response import ToolCall
from ai_arch_toolkit.core._tools._approval import (
    ApprovalDecision,
    ApprovalHandler,
    ApprovalRequest,
    approval_request_for,
    resolve_approval,
    resolve_approval_sync,
)
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


def _tool_def(fn: Callable[..., Any]) -> dict[str, Any]:
    return getattr(fn, "__tool__", {})


def _approval_error(
    error_type: str,
    message: str,
    *,
    request: ApprovalRequest | None = None,
    decision: ApprovalDecision | None = None,
) -> ToolResult:
    details: dict[str, Any] = {}
    if request is not None:
        details["approval_request"] = request.to_dict()
    if decision is not None:
        details["approval_decision"] = decision.to_dict()
    return ToolResult.failure(
        error_type,
        message,
        safe_to_show=True,
        details=details,
        metadata=details,
    )


def _approved_args_or_error(
    tool_call: ToolCall,
    fn: Callable[..., Any],
    approval_handler: ApprovalHandler | None,
) -> tuple[dict[str, Any] | None, ToolResult | None, dict[str, Any]]:
    tool_def = _tool_def(fn)
    if not tool_def.get("requires_approval", False):
        return dict(tool_call.input), None, {}

    request = approval_request_for(tool_call, tool_def)
    decision = resolve_approval_sync(request, approval_handler)
    if not decision.approved or decision.denied:
        return (
            None,
            _approval_error(
                "approval_denied",
                f"Tool {tool_call.name!r} requires approval and was denied",
                request=request,
                decision=decision,
            ),
            {},
        )
    metadata = {
        "approval_request": request.to_dict(),
        "approval_decision": decision.to_dict(),
    }
    return decision.modified_args or dict(tool_call.input), None, metadata


async def _approved_args_or_error_async(
    tool_call: ToolCall,
    fn: Callable[..., Any],
    approval_handler: ApprovalHandler | None,
) -> tuple[dict[str, Any] | None, ToolResult | None, dict[str, Any]]:
    tool_def = _tool_def(fn)
    if not tool_def.get("requires_approval", False):
        return dict(tool_call.input), None, {}

    request = approval_request_for(tool_call, tool_def)
    decision = await resolve_approval(request, approval_handler)
    if not decision.approved or decision.denied:
        return (
            None,
            _approval_error(
                "approval_denied",
                f"Tool {tool_call.name!r} requires approval and was denied",
                request=request,
                decision=decision,
            ),
            {},
        )
    metadata = {
        "approval_request": request.to_dict(),
        "approval_decision": decision.to_dict(),
    }
    return decision.modified_args or dict(tool_call.input), None, metadata


def execute_tool_result(
    tool_call: ToolCall,
    tools: list[Callable[..., Any]],
    *,
    approval_handler: ApprovalHandler | None = None,
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

    approved_args, approval_error, approval_metadata = _approved_args_or_error(
        tool_call,
        fn,
        approval_handler,
    )
    if approval_error is not None:
        return approval_error

    try:
        result = _coerce_result(fn(**(approved_args or {})))
        if approval_metadata:
            return ToolResult(
                ok=result.ok,
                value=result.value,
                error=result.error,
                metadata={**result.metadata, **approval_metadata},
            )
        return result
    except Exception as exc:
        return _result_from_exception(tool_call.name, exc)


async def async_execute_tool_result(
    tool_call: ToolCall,
    tools: list[Callable[..., Any]],
    *,
    approval_handler: ApprovalHandler | None = None,
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

    approved_args, approval_error, approval_metadata = await _approved_args_or_error_async(
        tool_call,
        fn,
        approval_handler,
    )
    if approval_error is not None:
        return approval_error

    try:
        if inspect.iscoroutinefunction(fn):
            result = _coerce_result(await fn(**(approved_args or {})))
        else:
            result = _coerce_result(await asyncio.to_thread(fn, **(approved_args or {})))
        if approval_metadata:
            return ToolResult(
                ok=result.ok,
                value=result.value,
                error=result.error,
                metadata={**result.metadata, **approval_metadata},
            )
        return result
    except Exception as exc:
        return _result_from_exception(tool_call.name, exc)


def execute_tool(
    tool_call: ToolCall,
    tools: list[Callable[..., Any]],
    *,
    approval_handler: ApprovalHandler | None = None,
) -> str:
    """Execute a tool call synchronously, returning a string result.

    Args:
        tool_call: The ToolCall from an LLM response.
        tools: List of decorated tool functions to search.
    """
    result = execute_tool_result(tool_call, tools, approval_handler=approval_handler)
    if result.ok:
        return result.to_model_text()
    _raise_legacy_error(tool_call.name, result)


async def async_execute_tool(
    tool_call: ToolCall,
    tools: list[Callable[..., Any]],
    *,
    approval_handler: ApprovalHandler | None = None,
) -> str:
    """Execute a tool call asynchronously, returning a string result.

    Args:
        tool_call: The ToolCall from an LLM response.
        tools: List of decorated tool functions to search.
    """
    result = await async_execute_tool_result(
        tool_call,
        tools,
        approval_handler=approval_handler,
    )
    if result.ok:
        return result.to_model_text()
    _raise_legacy_error(tool_call.name, result)


def _raise_legacy_error(tool_name: str, result: ToolResult) -> NoReturn:
    error = result.error
    if error is None:
        raise RuntimeError(f"Tool {tool_name!r} failed")
    if error.type == "unknown_tool":
        raise KeyError(error.message)
    if error.type == "validation_error":
        raise TypeError(error.message)
    if error.type == "approval_denied":
        raise PermissionError(error.message)
    raise RuntimeError(error.message)
