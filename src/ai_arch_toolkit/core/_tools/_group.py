"""ToolGroup — a collection of tools with lookup, governance, and execution."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

from ai_arch_toolkit.core._response import ToolCall
from ai_arch_toolkit.core._tools._approval import ApprovalHandler
from ai_arch_toolkit.core._tools._definition import ToolDefinition
from ai_arch_toolkit.core._tools._executor import (
    _arun_tool,
    _definition_for,
    _run_tool_sync,
)
from ai_arch_toolkit.core._tools._governance import (
    ApprovalGate,
    RunState,
    ToolGate,
    default_redactor,
)
from ai_arch_toolkit.core._tools._result import ToolResult


class ToolGroup:
    """A named collection of tools with execution-time governance.

    Stores canonical :class:`ToolDefinition` objects. ``execute`` /
    ``async_execute`` run a single governed pipeline and return a structured
    :class:`ToolResult`. Governance is configured at construction — an optional
    approval handler, extra pre-execution ``gates`` (e.g. dangerous-tool
    blocking, dry-run), and a call-count budget (``max_calls``).

    Usage::

        group = ToolGroup(get_weather, search)
        response = await llm.complete("...", tools=group)
        for tc in response.tool_calls:
            result = group.execute(tc)            # -> ToolResult
            text = result.to_model_text()
    """

    __slots__ = ("_defs", "_gates", "_max_calls", "_redactor", "_run_state")

    def __init__(
        self,
        *fns: Callable[..., Any],
        approval_handler: ApprovalHandler | None = None,
        gates: Sequence[ToolGate] = (),
        max_calls: int | None = None,
    ) -> None:
        self._defs: dict[str, ToolDefinition] = {}
        for fn in fns:
            self.add(fn)
        # Approval always runs last so dangerous-blocking / dry-run short-circuit
        # before a (potentially human) approval prompt.
        self._gates: tuple[ToolGate, ...] = (*gates, ApprovalGate(approval_handler))
        self._max_calls = max_calls
        self._run_state = RunState()
        self._redactor = default_redactor()

    def add(self, fn: Callable[..., Any]) -> None:
        """Add a function to the group."""
        definition = _definition_for(fn)
        name = definition.schema.name
        if name in self._defs:
            warnings.warn(
                f"Duplicate tool name {name!r} in ToolGroup; overwriting previous",
                stacklevel=2,
            )
        self._defs[name] = definition

    @property
    def tools(self) -> list[Callable[..., Any]]:
        """Return the registered tool callables."""
        return [d.fn for d in self._defs.values()]

    @property
    def definitions(self) -> list[dict[str, Any]]:
        """Return provider-safe tool definitions (no governance metadata)."""
        return [d.schema.to_provider_dict() for d in self._defs.values()]

    @property
    def runtime_definitions(self) -> list[ToolDefinition]:
        """Return the canonical runtime definitions (internal)."""
        return list(self._defs.values())

    def reset(self) -> None:
        """Reset the call-count budget so the group can be reused for a new run."""
        self._run_state.reset()

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call synchronously, returning a structured result."""
        definition = self._defs.get(tool_call.name)
        if definition is None:
            return ToolResult.failure(
                "unknown_tool",
                f"Unknown tool: {tool_call.name!r}",
                details={"tool_name": tool_call.name},
            )
        return _run_tool_sync(
            definition,
            tool_call,
            gates=self._gates,
            run_state=self._run_state,
            max_calls=self._max_calls,
            redactor=self._redactor,
        )

    async def async_execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call asynchronously, returning a structured result."""
        definition = self._defs.get(tool_call.name)
        if definition is None:
            return ToolResult.failure(
                "unknown_tool",
                f"Unknown tool: {tool_call.name!r}",
                details={"tool_name": tool_call.name},
            )
        return await _arun_tool(
            definition,
            tool_call,
            gates=self._gates,
            run_state=self._run_state,
            max_calls=self._max_calls,
            redactor=self._redactor,
        )

    def __contains__(self, name: str) -> bool:
        return name in self._defs

    def __len__(self) -> int:
        return len(self._defs)

    def __repr__(self) -> str:
        names = ", ".join(self._defs)
        return f"ToolGroup({names})"
