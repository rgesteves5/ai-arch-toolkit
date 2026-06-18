"""Tools — schema inference, decorator, execution, and grouping."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

from ai_arch_toolkit.core._server_tools import ServerTool
from ai_arch_toolkit.core._tools._decorator import tool
from ai_arch_toolkit.core._tools._executor import (
    async_execute_tool,
    async_execute_tool_result,
    execute_tool,
    execute_tool_result,
)
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.core._tools._result import ToolError, ToolResult
from ai_arch_toolkit.core._tools._schema import infer_schema

__all__ = [
    "ToolError",
    "ToolGroup",
    "ToolResult",
    "async_execute_tool",
    "async_execute_tool_result",
    "execute_tool",
    "execute_tool_result",
    "infer_schema",
    "prepare_tools",
    "tool",
]


def prepare_tools(
    tools: list[Any] | ToolGroup | Callable[..., Any] | None,
) -> list[dict[str, Any]] | None:
    """Normalize tool inputs into a list of tool definition dicts.

    Accepts:
    - ``None`` → ``None``
    - A single ``@tool``-decorated function → list with one def
    - A ``ToolGroup`` → its ``.definitions``
    - A list containing any mix of:
        - ``@tool``-decorated functions (have ``__tool__`` attr)
        - Plain dicts (``{"name": ..., "input_schema": ...}``)
        - ``ToolGroup`` instances (flattened)
    """
    if tools is None:
        return None

    # Single decorated function
    if callable(tools) and hasattr(tools, "__tool__"):
        return [tools.__tool__]  # type: ignore[union-attr]

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
            tool_def = getattr(item, "__tool__", None)
            if tool_def is not None:
                result.append(tool_def)
            else:
                result.append(infer_schema(item))
        else:
            warnings.warn(
                f"Skipping unsupported tool entry of type {type(item).__name__}",
                stacklevel=3,
            )
    return result
