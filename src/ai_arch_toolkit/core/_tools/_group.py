"""ToolGroup — a collection of tool functions with lookup and execution."""

from __future__ import annotations

import asyncio
import inspect
import warnings
from collections.abc import Callable
from typing import Any

from ai_arch_toolkit.core._response import ToolCall
from ai_arch_toolkit.core._tools._approval import ApprovalHandler
from ai_arch_toolkit.core._tools._executor import (
    _approved_args_or_error,
    _approved_args_or_error_async,
    _coerce_result,
    _format_result,
    _result_from_exception,
)
from ai_arch_toolkit.core._tools._result import ToolResult
from ai_arch_toolkit.core._tools._schema import infer_schema


class ToolGroup:
    """A named collection of tool functions.

    Provides tool definitions (for passing to LLM) and execution (for
    handling tool calls from LLM responses).

    Functions passed to the constructor can be ``@tool``-decorated (their
    ``__tool__`` schema is reused) or plain callables (schema is inferred).

    Usage::

        group = ToolGroup(get_weather, search_web)
        response = await llm.complete("...", tools=group)
        for tc in response.tool_calls:
            result = group.execute(tc)
    """

    __slots__ = ("_approval_handler", "_definitions", "_fns")

    def __init__(
        self,
        *fns: Callable[..., Any],
        approval_handler: ApprovalHandler | None = None,
    ) -> None:
        self._approval_handler = approval_handler
        self._fns: dict[str, Callable[..., Any]] = {}
        self._definitions: dict[str, dict[str, Any]] = {}
        for fn in fns:
            self.add(fn)

    def add(self, fn: Callable[..., Any]) -> None:
        """Add a function to the group."""
        tool_def = getattr(fn, "__tool__", None)
        if tool_def is None:
            tool_def = infer_schema(fn)
        name = tool_def["name"]
        if name in self._fns:
            warnings.warn(
                f"Duplicate tool name {name!r} in ToolGroup; overwriting previous",
                stacklevel=2,
            )
        self._fns[name] = fn
        self._definitions[name] = tool_def

    @property
    def tools(self) -> list[Callable[..., Any]]:
        """Return the registered tool functions."""
        return list(self._fns.values())

    @property
    def definitions(self) -> list[dict[str, Any]]:
        """Return tool definitions suitable for passing to LLM APIs."""
        return list(self._definitions.values())

    def execute(self, tool_call: ToolCall) -> str:
        """Execute a tool call synchronously."""
        fn = self._fns.get(tool_call.name)
        if fn is None:
            msg = f"Unknown tool: {tool_call.name!r}"
            raise KeyError(msg)
        approved_args, approval_error, _ = _approved_args_or_error(
            tool_call,
            fn,
            self._approval_handler,
        )
        if approval_error is not None:
            raise PermissionError(approval_error.error.message if approval_error.error else "")
        return _format_result(fn(**(approved_args or {})))

    def execute_result(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call synchronously, returning a structured result."""
        fn = self._fns.get(tool_call.name)
        if fn is None:
            return ToolResult.failure(
                "unknown_tool",
                f"Unknown tool: {tool_call.name!r}",
                details={"tool_name": tool_call.name},
            )
        approved_args, approval_error, approval_metadata = _approved_args_or_error(
            tool_call,
            fn,
            self._approval_handler,
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

    async def async_execute(self, tool_call: ToolCall) -> str:
        """Execute a tool call asynchronously."""
        fn = self._fns.get(tool_call.name)
        if fn is None:
            msg = f"Unknown tool: {tool_call.name!r}"
            raise KeyError(msg)
        approved_args, approval_error, _ = await _approved_args_or_error_async(
            tool_call,
            fn,
            self._approval_handler,
        )
        if approval_error is not None:
            raise PermissionError(approval_error.error.message if approval_error.error else "")
        if inspect.iscoroutinefunction(fn):
            result = await fn(**(approved_args or {}))
        else:
            result = await asyncio.to_thread(fn, **(approved_args or {}))
        return _format_result(result)

    async def async_execute_result(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call asynchronously, returning a structured result."""
        fn = self._fns.get(tool_call.name)
        if fn is None:
            return ToolResult.failure(
                "unknown_tool",
                f"Unknown tool: {tool_call.name!r}",
                details={"tool_name": tool_call.name},
            )
        approved_args, approval_error, approval_metadata = await _approved_args_or_error_async(
            tool_call,
            fn,
            self._approval_handler,
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

    def __contains__(self, name: str) -> bool:
        return name in self._fns

    def __len__(self) -> int:
        return len(self._fns)

    def __repr__(self) -> str:
        names = ", ".join(self._fns)
        return f"ToolGroup({names})"
