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
)
from ai_arch_toolkit.core._tools._executor import (
    async_execute_tool,
    execute_tool,
)
from ai_arch_toolkit.core._tools._governance import (
    ApprovalGate,
    DangerousToolGate,
    DryRunGate,
    GovernanceOutcome,
    RunState,
)
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.core._tools._result import ToolError, ToolResult
from ai_arch_toolkit.core._tools._schema import infer_schema, tool_schema

__all__ = [
    "ApprovalDecision",
    "ApprovalGate",
    "ApprovalHandler",
    "ApprovalRequest",
    "DangerousToolGate",
    "DryRunGate",
    "GovernanceOutcome",
    "RiskLevel",
    "RunState",
    "ToolDefinition",
    "ToolError",
    "ToolGroup",
    "ToolResult",
    "ToolRuntimePolicy",
    "ToolSchema",
    "async_execute_tool",
    "execute_tool",
    "infer_schema",
    "prepare_tools",
    "tool",
    "tool_schema",
]


def prepare_tools(
    tools: list[Any] | ToolGroup | Callable[..., Any] | None,
) -> list[dict[str, Any]] | None:
    """Normalize tool inputs into a list of provider-facing definition dicts.

    Accepts:
    - ``None`` → ``None``
    - A single ``@tool``-decorated function → list with one provider dict
    - A ``ToolGroup`` → its ``.definitions`` (provider-safe)
    - A list containing any mix of:
        - ``@tool``-decorated functions (have ``__tool_definition__``)
        - Plain dicts (``{"name": ..., "input_schema": ...}``)
        - ``ToolGroup`` instances (flattened)
    """
    if tools is None:
        return None

    # Single decorated function
    if callable(tools) and hasattr(tools, "__tool_definition__"):
        return [tools.__tool_definition__.schema.to_provider_dict()]  # type: ignore[union-attr]

    # ToolGroup
    if isinstance(tools, ToolGroup):
        return tools.definitions

    # List of mixed items
    if not isinstance(tools, list):
        warnings.warn(
            "Unsupported tools input type "
            f"{type(tools).__name__}; expected list, ToolGroup, or tool",
            stacklevel=3,
        )
        return None

    result: list[dict[str, Any]] = []
    for item in tools:
        if isinstance(item, ServerTool):
            result.append({"_server_tool": True, "type": item.type, **item.config})
        elif isinstance(item, dict):
            if "name" not in item or not item["name"]:
                warnings.warn(
                    "Tool dict missing 'name' field; skipping",
                    stacklevel=3,
                )
                continue
            result.append(item)
        elif isinstance(item, ToolGroup):
            result.extend(item.definitions)
        elif callable(item):
            definition = getattr(item, "__tool_definition__", None)
            if definition is not None:
                result.append(definition.schema.to_provider_dict())
            else:
                result.append(infer_schema(item))
        else:
            warnings.warn(
                f"Skipping unsupported tool entry of type {type(item).__name__}",
                stacklevel=3,
            )
    return result
