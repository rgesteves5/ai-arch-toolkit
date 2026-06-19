"""@tool decorator — auto-generates a tool definition from hints and docstring."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, overload

from ai_arch_toolkit.core._tools._definition import (
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
) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that builds a ``ToolDefinition`` from hints and docstring.

    Can be used bare (``@tool``) or with arguments
    (``@tool(name=..., requires_approval=...)``). Attaches the canonical
    ``__tool_definition__`` (a :class:`ToolDefinition`) to the decorated function.
    """

    def _wrap(f: Callable[..., Any]) -> Callable[..., Any]:
        schema_obj = tool_schema(f, name=name, overrides=schema)
        policy = ToolRuntimePolicy(
            capability=capability,
            risk_level=risk_level,
            requires_approval=requires_approval,
            approval_reason=approval_reason,
        )

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
