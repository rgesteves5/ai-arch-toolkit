"""Middleware contracts and request envelope for client pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ai_arch_toolkit._legacy.llm._types import ConversationItem, JsonSchema, Tool


@dataclass(slots=True)
class Request:
    """Normalized request object passed through middleware hooks."""

    operation: str
    provider: str
    model: str
    messages: list[ConversationItem]
    system: str | None = None
    tools: list[Tool] | None = None
    json_schema: JsonSchema | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)


class Middleware(Protocol):
    """Hook contract for request/response interception."""

    def before(self, request: Request) -> Request: ...

    def after(self, request: Request, result: Any) -> Any: ...

    async def abefore(self, request: Request) -> Request: ...

    async def aafter(self, request: Request, result: Any) -> Any: ...
