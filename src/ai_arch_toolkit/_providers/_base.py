"""Abstract base for LLM providers — async-only, 2 methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from ai_arch_toolkit._response import Response


class BaseProvider(ABC):
    """Interface that every provider must implement (async-only)."""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Response: ...

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]: ...
