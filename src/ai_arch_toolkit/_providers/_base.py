"""Abstract base for LLM providers — async-only."""

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
    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        **kwargs: Any,
    ) -> tuple[AsyncIterator[str], Any]: ...

    # ------------------------------------------------------------------
    # Lifecycle — concrete no-ops, providers override if needed
    # ------------------------------------------------------------------

    async def close(self) -> None:  # noqa: B027
        """Release resources. Override in providers that hold clients."""

    async def __aenter__(self) -> BaseProvider:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
