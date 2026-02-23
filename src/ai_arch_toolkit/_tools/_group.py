"""ToolGroup — a collection of tool functions with lookup and execution."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

from ai_arch_toolkit._response import ToolCall
from ai_arch_toolkit._tools._executor import _format_result
from ai_arch_toolkit._tools._schema import infer_schema


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

    __slots__ = ("_definitions", "_fns")

    def __init__(self, *fns: Callable[..., Any]) -> None:
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
        self._fns[name] = fn
        self._definitions[name] = tool_def

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
        return _format_result(fn(**tool_call.input))

    async def async_execute(self, tool_call: ToolCall) -> str:
        """Execute a tool call asynchronously."""
        fn = self._fns.get(tool_call.name)
        if fn is None:
            msg = f"Unknown tool: {tool_call.name!r}"
            raise KeyError(msg)
        if inspect.iscoroutinefunction(fn):
            result = await fn(**tool_call.input)
        else:
            result = await asyncio.to_thread(fn, **tool_call.input)
        return _format_result(result)

    def __contains__(self, name: str) -> bool:
        return name in self._fns

    def __len__(self) -> int:
        return len(self._fns)

    def __repr__(self) -> str:
        names = ", ".join(self._fns)
        return f"ToolGroup({names})"
