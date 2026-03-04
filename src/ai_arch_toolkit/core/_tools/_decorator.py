"""@tool decorator — auto-generates tool schema from type hints and docstring."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, overload

from ai_arch_toolkit.core._tools._schema import infer_schema


@overload
def tool(fn: Callable[..., Any], /) -> Callable[..., Any]: ...


@overload
def tool(
    *,
    name: str | None = None,
    schema: dict[str, dict[str, object]] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...


def tool(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    schema: dict[str, dict[str, object]] | None = None,
) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that auto-generates a tool schema from type hints and docstring.

    Can be used bare (``@tool``) or with arguments
    (``@tool(name=..., schema=...)``).
    Attaches a ``__tool__`` attribute (dict) to the decorated function.
    """

    def _wrap(f: Callable[..., Any]) -> Callable[..., Any]:
        tool_def = infer_schema(f, name=name, overrides=schema)

        if inspect.iscoroutinefunction(f):

            @functools.wraps(f)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await f(*args, **kwargs)

            async_wrapper.__tool__ = tool_def  # type: ignore[attr-defined]
            return async_wrapper

        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return f(*args, **kwargs)

        wrapper.__tool__ = tool_def  # type: ignore[attr-defined]
        return wrapper

    if fn is not None:
        return _wrap(fn)
    return _wrap
