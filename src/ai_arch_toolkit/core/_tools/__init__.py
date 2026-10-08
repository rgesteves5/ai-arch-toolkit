"""Tools — schema inference, decorator, execution, grouping, and governance."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

from ai_arch_toolkit.core._server_tools import ServerTool
from ai_arch_toolkit.core._tools._approval import (
    ApprovalDecision,
    ApprovalHandler,
    ApprovalRequest,
)
from ai_arch_toolkit.core._tools._decorator import tool
from ai_arch_toolkit.core._tools._definition import (
    RiskLevel,
    ToolDefinition,
    ToolRuntimePolicy,
    ToolSchema,
    check_tool_name,
    one_per_name,
)
from ai_arch_toolkit.core._tools._dynamic import tool_from_schema
from ai_arch_toolkit.core._tools._executor import (
    _definition_for,
    async_execute_tool,
    execute_tool,
)
from ai_arch_toolkit.core._tools._governance import (
    ApprovalGate,
    DangerousToolGate,
    DryRunGate,
    ExecutionContext,
    GateBlock,
    GateDryRun,
    GateModify,
    GateResult,
    GovernanceOutcome,
    RunState,
    ToolGate,
)
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.core._tools._result import ToolError, ToolFailure, ToolFailureType, ToolResult
from ai_arch_toolkit.core._tools._schema import Range, infer_schema, tool_schema

__all__ = [
    "ApprovalDecision",
    "ApprovalGate",
    "ApprovalHandler",
    "ApprovalRequest",
    "DangerousToolGate",
    "DryRunGate",
    "ExecutionContext",
    "GateBlock",
    "GateDryRun",
    "GateModify",
    "GateResult",
    "GovernanceOutcome",
    "Range",
    "RiskLevel",
    "RunState",
    "ToolDefinition",
    "ToolError",
    "ToolFailure",
    "ToolFailureType",
    "ToolGate",
    "ToolGroup",
    "ToolResult",
    "ToolRuntimePolicy",
    "ToolSchema",
    "async_execute_tool",
    "execute_tool",
    "infer_schema",
    "prepare_tools",
    "tool",
    "tool_from_schema",
    "tool_schema",
]


def prepare_tools(
    tools: list[Any] | ToolGroup | Callable[..., Any] | None,
) -> list[dict[str, Any]] | None:
    """Normalize tool inputs into a list of provider-facing definition dicts.

    Accepts:
    - ``None`` → ``None``
    - A ``ToolGroup`` → its ``.definitions`` (provider-safe)
    - A single tool → list with one provider dict
    - A list containing any mix of:
        - tools: ``@tool`` functions, :func:`tool_from_schema` tools, plain callables
        - Plain dicts (``{"name": ..., "input_schema": ...}``)
        - ``ToolGroup`` instances (flattened)
        - server tools (``web_search()``, ...)

    One name, one tool: a tool met again, in the list or in a group in it, is sent once.

    Raises:
        ValueError: Two different tools share a name, or a name is not portable (see
            :class:`ToolSchema`), before anything is sent.
    """
    if tools is None:
        return None
    if isinstance(tools, ToolGroup):
        return tools.definitions
    if callable(tools):
        tools = [tools]
    if not isinstance(tools, list):
        warnings.warn(
            "Unsupported tools input type "
            f"{type(tools).__name__}; expected list, ToolGroup, or tool",
            stacklevel=3,
        )
        return None
    return one_per_name([entry for item in tools for entry in _wire_entries(item)])


# A tool on its way to a provider: its name (none for a server tool), what makes it the same
# tool, and the dict the adapters take.
type _Wire = tuple[str | None, object, dict[str, Any]]


def _wire_entries(item: object) -> list[_Wire]:
    """The tools ``item`` holds, as wire entries; none for an entry that is skipped."""
    if isinstance(item, ServerTool):
        return [(None, item, {"_server_tool": True, "type": item.type, **item.config})]
    if isinstance(item, dict):
        return _dict_entries(item)
    if isinstance(item, ToolGroup):
        return [_definition_entry(definition) for definition in item.runtime_definitions]
    if callable(item):
        return [_definition_entry(_definition_for(item))]
    warnings.warn(f"Skipping unsupported tool entry of type {type(item).__name__}", stacklevel=4)
    return []


def _definition_entry(definition: ToolDefinition) -> _Wire:
    return definition.schema.name, definition.fn, definition.schema.to_provider_dict()


def _dict_entries(item: dict[str, Any]) -> list[_Wire]:
    if item.get("_server_tool"):  # already in wire form (e.g. a request after middleware)
        return [(None, item, item)]
    name = item.get("name")
    if not name:
        warnings.warn("Tool dict missing 'name' field; skipping", stacklevel=5)
        return []
    check_tool_name(name)
    return [(name, item, item)]
