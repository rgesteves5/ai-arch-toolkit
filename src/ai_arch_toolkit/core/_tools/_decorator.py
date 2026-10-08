"""@tool decorator — auto-generates a tool definition from hints and docstring."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping
from typing import Any, overload

from ai_arch_toolkit.core._tools._definition import (
    DEFAULT_MAX_OUTPUT_CHARS,
    DEFAULT_TIMEOUT_S,
    RiskLevel,
    ToolDefinition,
    ToolRuntimePolicy,
)
from ai_arch_toolkit.core._tools._schema import tool_schema


@overload
def tool(fn: Callable[..., Any], /) -> Callable[..., Any]: ...


@overload
def tool(
    *,
    name: str | None = None,
    schema: dict[str, dict[str, object]] | None = None,
    capability: str | None = None,
    risk_level: RiskLevel = "low",
    requires_approval: bool = False,
    approval_reason: str = "",
    max_output_chars: int | None = DEFAULT_MAX_OUTPUT_CHARS,
    timeout_s: float | None = DEFAULT_TIMEOUT_S,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...


def tool(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    schema: dict[str, dict[str, object]] | None = None,
    capability: str | None = None,
    risk_level: RiskLevel = "low",
    requires_approval: bool = False,
    approval_reason: str = "",
    max_output_chars: int | None = DEFAULT_MAX_OUTPUT_CHARS,
    timeout_s: float | None = DEFAULT_TIMEOUT_S,
) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that builds a ``ToolDefinition`` from hints and docstring.

    Can be used bare (``@tool``) or with arguments
    (``@tool(name=..., requires_approval=...)``). Attaches the canonical
    ``__tool_definition__`` (a :class:`ToolDefinition`) to the decorated function.
    ``schema`` maps a parameter's name to JSON Schema keywords merged into what was inferred for
    it; a tool described by a complete JSON Schema is :func:`tool_from_schema`'s.
    ``max_output_chars`` and ``timeout_s`` bound each call (see :class:`ToolRuntimePolicy`).

    Raises:
        TypeError: ``schema`` is not a mapping of parameter names to mappings.
        ValueError: The tool's name is not portable (see :class:`ToolSchema`).
    """
    _check_overrides(schema)
    policy = ToolRuntimePolicy(
        capability=capability,
        risk_level=risk_level,
        requires_approval=requires_approval,
        approval_reason=approval_reason,
        max_output_chars=max_output_chars,
        timeout_s=timeout_s,
    )

    def _wrap(f: Callable[..., Any]) -> Callable[..., Any]:
        schema_obj = tool_schema(f, name=name, overrides=schema)

        if inspect.iscoroutinefunction(f):

            @functools.wraps(f)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await f(*args, **kwargs)

            async_wrapper.__tool_definition__ = ToolDefinition(  # type: ignore[attr-defined]
                fn=async_wrapper, schema=schema_obj, policy=policy
            )
            return async_wrapper

        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return f(*args, **kwargs)

        wrapper.__tool_definition__ = ToolDefinition(  # type: ignore[attr-defined]
            fn=wrapper, schema=schema_obj, policy=policy
        )
        return wrapper

    if fn is not None:
        return _wrap(fn)
    return _wrap


def _check_overrides(schema: object) -> None:
    """Raise ``TypeError`` unless ``schema`` maps each parameter's name to JSON Schema keywords.

    A complete schema (``{"type": "object", "properties": ...}``) would otherwise be merged in as
    parameters named ``type`` and ``properties``.
    """
    if schema is None:
        return
    if isinstance(schema, Mapping):
        wrong = [key for key, value in schema.items() if not isinstance(value, Mapping)]
        if not wrong:
            return
        found = f"{wrong[0]!r} mapped to a {type(schema[wrong[0]]).__name__}"
    else:
        found = f"a {type(schema).__name__}"
    msg = (
        "@tool(schema=...) maps each parameter's name to the JSON Schema keywords merged into "
        f"what was inferred for it; got {found}. A tool described by a complete JSON Schema is "
        "built with tool_from_schema()."
    )
    raise TypeError(msg)
