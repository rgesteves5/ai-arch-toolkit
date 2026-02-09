"""Tool registry for managing and executing tool functions."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable
from typing import Any

from ai_arch_toolkit.llm._types import Tool, ToolCall


class ValidationError(Exception):
    """Raised when tool arguments fail validation."""


_SCHEMA_TYPE_TO_PYTHON: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
}


def _validate_args(tool_def: Tool, arguments: dict[str, object]) -> None:
    """Validate arguments against a tool's parameter schema."""
    params = tool_def.parameters
    required = params.get("required", [])
    properties = params.get("properties", {})

    # Check required keys
    if isinstance(required, list):
        for key in required:
            if key not in arguments:
                msg = f"Missing required argument '{key}' for tool '{tool_def.name}'"
                raise ValidationError(msg)

    # Type-check each provided argument
    if isinstance(properties, dict):
        for key, value in arguments.items():
            prop_schema = properties.get(key)
            if not isinstance(prop_schema, dict):
                continue
            schema_type = prop_schema.get("type")
            if not isinstance(schema_type, str):
                continue
            expected = _SCHEMA_TYPE_TO_PYTHON.get(schema_type)
            if expected is None:
                continue
            if not isinstance(value, expected):
                msg = (
                    f"Argument '{key}' for tool '{tool_def.name}' "
                    f"expected {schema_type}, got "
                    f"{type(value).__name__}"
                )
                raise ValidationError(msg)


class ToolRegistry:
    """Registry that maps tool names to callables and their LLM-compatible schemas."""

    def __init__(self) -> None:
        self._callables: dict[str, Callable[..., Any]] = {}
        self._definitions: dict[str, Tool] = {}
        self._disabled: set[str] = set()

    def register(self, name: str, fn: Callable[..., Any], tool_def: Tool) -> None:
        """Register a callable with its tool definition."""
        self._callables[name] = fn
        self._definitions[name] = tool_def

    def execute(self, tool_call: ToolCall) -> str:
        """Execute a tool call synchronously, returning a JSON string result."""
        if tool_call.name in self._disabled:
            raise KeyError(f"Tool {tool_call.name!r} is disabled")
        fn = self._callables.get(tool_call.name)
        if fn is None:
            raise KeyError(f"Unknown tool: {tool_call.name!r}")
        tool_def = self._definitions[tool_call.name]
        _validate_args(tool_def, tool_call.arguments)
        result = fn(**tool_call.arguments)
        return result if isinstance(result, str) else json.dumps(result)

    async def async_execute(self, tool_call: ToolCall) -> str:
        """Execute a tool call asynchronously, returning a JSON string result."""
        if tool_call.name in self._disabled:
            raise KeyError(f"Tool {tool_call.name!r} is disabled")
        fn = self._callables.get(tool_call.name)
        if fn is None:
            raise KeyError(f"Unknown tool: {tool_call.name!r}")
        tool_def = self._definitions[tool_call.name]
        _validate_args(tool_def, tool_call.arguments)
        if inspect.iscoroutinefunction(fn):
            result = await fn(**tool_call.arguments)
        else:
            result = await asyncio.to_thread(fn, **tool_call.arguments)
        return result if isinstance(result, str) else json.dumps(result)

    def definitions(self, group: str | None = None) -> list[Tool]:
        """Return registered tool definitions (for passing to LLM context).

        If group is set, return only tools whose names start with
        ``group + "."``.
        """
        if group is not None:
            prefix = group + "."
            return [
                t
                for name, t in self._definitions.items()
                if name.startswith(prefix) and name not in self._disabled
            ]
        return [t for name, t in self._definitions.items() if name not in self._disabled]

    def disable(self, name: str) -> None:
        """Disable a registered tool (excluded from definitions and execution)."""
        if name not in self._callables:
            raise KeyError(f"Unknown tool: {name!r}")
        self._disabled.add(name)

    def enable(self, name: str) -> None:
        """Re-enable a previously disabled tool."""
        self._disabled.discard(name)

    def alias(self, original: str, alias_name: str) -> None:
        """Register an alias for an existing tool."""
        if original not in self._callables:
            raise KeyError(f"Unknown tool: {original!r}")
        fn = self._callables[original]
        orig_def = self._definitions[original]
        alias_def = Tool(
            name=alias_name,
            description=orig_def.description,
            parameters=orig_def.parameters,
        )
        self._callables[alias_name] = fn
        self._definitions[alias_name] = alias_def

    def group(self, prefix: str) -> ToolRegistryView:
        """Return a lightweight view that filters tools by prefix."""
        return ToolRegistryView(self, prefix)

    def __contains__(self, name: str) -> bool:
        return name in self._callables

    def __len__(self) -> int:
        return len(self._callables)


class ToolRegistryView:
    """A filtered view of a ToolRegistry that delegates to the parent."""

    def __init__(self, parent: ToolRegistry, prefix: str) -> None:
        self._parent = parent
        self._prefix = prefix + "."

    def definitions(self) -> list[Tool]:
        """Return tool definitions matching the group prefix."""
        return self._parent.definitions(group=self._prefix[:-1])

    def execute(self, tool_call: ToolCall) -> str:
        """Execute a tool call if it belongs to this group."""
        if not tool_call.name.startswith(self._prefix):
            raise KeyError(f"Tool {tool_call.name!r} not in group {self._prefix[:-1]!r}")
        return self._parent.execute(tool_call)

    async def async_execute(self, tool_call: ToolCall) -> str:
        """Execute a tool call asynchronously if it belongs to this group."""
        if not tool_call.name.startswith(self._prefix):
            raise KeyError(f"Tool {tool_call.name!r} not in group {self._prefix[:-1]!r}")
        return await self._parent.async_execute(tool_call)

    def __contains__(self, name: str) -> bool:
        return name.startswith(self._prefix) and name in self._parent

    def __len__(self) -> int:
        return sum(1 for name in self._parent._callables if name.startswith(self._prefix))
