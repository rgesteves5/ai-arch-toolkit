"""ToolGroup — a collection of tools with lookup, governance, and execution."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ai_arch_toolkit.core._response import ToolCall
from ai_arch_toolkit.core._server_tools import ServerTool
from ai_arch_toolkit.core._tools._approval import ApprovalHandler
from ai_arch_toolkit.core._tools._definition import ToolDefinition, check_bounds, name_clash
from ai_arch_toolkit.core._tools._executor import (
    _arun_tool,
    _definition_for,
    _Limits,
    _run_tool_sync,
)
from ai_arch_toolkit.core._tools._governance import (
    ApprovalGate,
    RunState,
    ToolGate,
    default_redactor,
)
from ai_arch_toolkit.core._tools._result import ToolResult

# One writer at a time, so no change to a group is lost. Shared by every group (writes are rare
# and short) rather than held by each, so a group can still be copied (``copy.deepcopy``).
_WRITING = threading.Lock()


class ToolGroup:
    """A named collection of tools with execution-time governance.

    Stores canonical :class:`ToolDefinition` objects. ``execute`` /
    ``async_execute`` run a single governed pipeline and return a structured
    :class:`ToolResult`. Governance is configured at construction — an optional
    approval handler, extra pre-execution ``gates`` (e.g. dangerous-tool
    blocking, dry-run), a call-count budget (``max_calls``), and ceilings on each
    tool's ``max_output_chars`` and ``timeout_s``: the stricter of the group's and
    the tool's own applies, so a group tightens its tools and never widens them.

    One name, one tool. :meth:`add` and :meth:`remove` may change the group while it runs.

    Usage::

        group = ToolGroup(get_weather, search)
        response = await llm.complete("...", tools=group)
        for tc in response.tool_calls:
            result = group.execute(tc)            # -> ToolResult
            text = result.to_model_text()
    """

    __slots__ = ("_ceiling", "_defs", "_gates", "_max_calls", "_redactor", "_run_state")

    def __init__(
        self,
        *fns: Callable[..., Any],
        approval_handler: ApprovalHandler | None = None,
        gates: Sequence[ToolGate] = (),
        max_calls: int | None = None,
        max_output_chars: int | None = None,
        timeout_s: float | None = None,
    ) -> None:
        check_bounds(max_output_chars, timeout_s)
        self._ceiling = _Limits(max_output_chars, timeout_s)
        # Copy-on-write: a change puts a new table here and never edits the one in place, so a
        # reader (a turn listing the tools, a call starting) works on the table it read.
        self._defs: Mapping[str, ToolDefinition] = {}
        for fn in fns:
            self.add(fn)
        # Approval always runs last so dangerous-blocking / dry-run short-circuit
        # before a (potentially human) approval prompt.
        self._gates: tuple[ToolGate, ...] = (*gates, ApprovalGate(approval_handler))
        self._max_calls = max_calls
        self._run_state = RunState()
        self._redactor = default_redactor()

    def add(self, fn: Callable[..., Any], *, replace: bool = False) -> None:
        """Add a tool to the group; adding a tool the group holds changes nothing.

        The group may change while it runs: the next turn lists the tools the group holds then,
        and a call already running ends with the tool it started with. ``max_calls`` keeps
        counting.

        Args:
            fn: A ``@tool`` function, a :func:`tool_from_schema` tool, or a plain callable.
            replace: Put ``fn`` in the place of the tool the group holds under its name.

        Raises:
            TypeError: If ``fn`` is a provider-hosted :class:`ServerTool` (pass it to the LLM
                next to the group instead) or is not callable.
            ValueError: If the group holds another tool with the same name and ``replace`` is
                false, or the name is not portable (see :class:`ToolSchema`).
        """
        if isinstance(fn, ServerTool):
            msg = (
                f"ToolGroup cannot hold server tool {fn.type!r}: server tools are executed by "
                "the provider, not by the group. Pass it next to the group instead, e.g. "
                "llm.complete(..., tools=[group, web_search()])"
            )
            raise TypeError(msg)
        if not callable(fn):
            msg = f"ToolGroup tools must be callable, got {type(fn).__name__}"
            raise TypeError(msg)
        definition = _definition_for(fn)
        name = definition.schema.name
        with _WRITING:
            held = self._defs.get(name)
            # Equal, not identical: each read of ``obj.method`` makes a new bound method.
            if held is not None and held.fn == definition.fn:
                return
            if held is not None and not replace:
                msg = (
                    f"{name_clash(name)} To swap the tool the group holds, add(..., replace=True)."
                )
                raise ValueError(msg)
            self._defs = {**self._defs, name: definition}

    def remove(self, name: str) -> ToolDefinition:
        """Take the tool named ``name`` out of the group, and return its definition.

        As with :meth:`add`, a call already running ends with the tool it started with.

        Raises:
            KeyError: The group holds no tool named ``name``.
        """
        with _WRITING:
            if name not in self._defs:
                msg = f"ToolGroup holds no tool named {name!r}"
                raise KeyError(msg)
            kept = dict(self._defs)
            removed = kept.pop(name)
            self._defs = kept
        return removed

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
            ceiling=self._ceiling,
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
            ceiling=self._ceiling,
        )

    def __contains__(self, name: str) -> bool:
        return name in self._defs

    def __len__(self) -> int:
        return len(self._defs)

    def __repr__(self) -> str:
        names = ", ".join(self._defs)
        return f"ToolGroup({names})"
