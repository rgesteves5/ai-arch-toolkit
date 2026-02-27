"""Middleware protocol for request/response hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ai_arch_toolkit.core._response import Response


@dataclass(frozen=True, slots=True, kw_only=True)
class Request:
    """Snapshot of an LLM request, passed to middleware hooks."""

    messages: list[dict[str, Any]]
    system: str | None
    tools: list[dict[str, Any]] | None
    model: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Middleware(Protocol):
    """Protocol for request/response middleware.

    Implement ``before`` and ``after`` for sync hooks. Optionally add
    ``abefore`` / ``aafter`` for async-only hooks (detected via hasattr).
    """

    def before(self, request: Request) -> Request:
        """Modify or inspect the request before it reaches the provider."""
        ...

    def after(self, request: Request, response: Response) -> Response:
        """Modify or inspect the response before it's returned to the caller."""
        ...


def _run_before(middleware: list[Any], request: Request) -> Request:
    """Run all ``before`` hooks in order."""
    for mw in middleware:
        request = mw.before(request)
    return request


def _run_after(middleware: list[Any], request: Request, response: Response) -> Response:
    """Run all ``after`` hooks in reverse order."""
    for mw in reversed(middleware):
        response = mw.after(request, response)
    return response


async def _run_abefore(middleware: list[Any], request: Request) -> Request:
    """Run async ``abefore`` hooks, falling back to sync ``before``."""
    for mw in middleware:
        if hasattr(mw, "abefore"):
            request = await mw.abefore(request)
        else:
            request = mw.before(request)
    return request


async def _run_aafter(middleware: list[Any], request: Request, response: Response) -> Response:
    """Run async ``aafter`` hooks in reverse, falling back to sync ``after``."""
    for mw in reversed(middleware):
        if hasattr(mw, "aafter"):
            response = await mw.aafter(request, response)
        else:
            response = mw.after(request, response)
    return response
