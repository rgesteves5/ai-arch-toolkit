"""Abstract base for LLM providers — async-only."""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from typing import Any

from ai_arch_toolkit.core._response import Response, StreamEvent, ThinkingBlock, ToolCall, Usage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared constants — used by providers that map effort → token budget
# ---------------------------------------------------------------------------

# Default budget when thinking_effort maps to token budget (Anthropic, Gemini 2.5)
THINKING_EFFORT_BUDGETS: dict[str, int] = {
    "low": 2048,
    "medium": 5000,
    "high": 10000,
}
DEFAULT_THINKING_BUDGET: int = 10000


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a retry-after header value to seconds.

    Handles numeric seconds (int or float). Returns None if missing or unparseable.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        logger.debug("Could not parse retry-after header: %r", value)
        return None


class LoopAwareClientCache:
    """Rebuild a cached async SDK client whose pool died with its event loop.

    The sync wrappers drive every call through a fresh ``asyncio.run()`` loop
    that is closed afterwards. An async SDK client binds its connection pool
    (httpx, gRPC aio) to the loop that served its first request, so the next
    call — on a new loop — fails with a connection error. Providers install
    their client with ``_install_client``; the ``_client`` property rebuilds it
    from the factory once the loop it served is closed. A client assigned
    directly (``provider._client = mock`` in tests) has no factory and is never
    replaced. Using one provider from two concurrently *live* loops remains
    unsupported.
    """

    _client_value: Any
    _client_factory: Callable[[], Any] | None = None
    _client_loop: asyncio.AbstractEventLoop | None = None

    def _install_client(self, factory: Callable[[], Any]) -> None:
        """Install an SDK client that ``_client`` may rebuild after loop turnover."""
        self._client_value = factory()
        self._client_factory = factory
        self._client_loop = None

    @property
    def _client(self) -> Any:
        if self._client_factory is not None:
            try:
                loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                if self._client_loop is None:
                    self._client_loop = loop
                elif self._client_loop is not loop and self._client_loop.is_closed():
                    # The dead loop's pool cannot be closed without its loop —
                    # drop the old client and start fresh on this one.
                    self._client_value = self._client_factory()
                    self._client_loop = loop
        return self._client_value

    @_client.setter
    def _client(self, value: Any) -> None:
        self._client_value = value
        self._client_factory = None
        self._client_loop = None


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
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[AsyncIterator[str], StreamState]: ...

    def stream_events(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[AsyncIterator[StreamEvent], StreamState]:
        """Stream structured events. Default wraps ``stream()``.

        Note: the default implementation yields thinking and tool_call events
        *after* the text stream is exhausted (non-real-time). Providers that
        support real-time structured events (e.g. Anthropic) override this.
        """
        text_stream, state = self.stream(messages, system=system, tools=tools, **kwargs)

        async def _events() -> AsyncIterator[StreamEvent]:
            async for chunk in text_stream:
                yield StreamEvent(kind="text", text=chunk)
            for block in state.thinking:
                yield StreamEvent(kind="thinking", thinking=block)
            for tool_call in state.tool_calls:
                yield StreamEvent(kind="tool_call", tool_call=tool_call)

        return _events(), state

    async def batch_submit(
        self,
        requests: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        """Submit a batch of requests. Returns a batch ID."""
        raise NotImplementedError(f"{type(self).__name__} does not support batch API")

    async def batch_status(self, batch_id: str) -> str:
        """Check batch status. Returns status string."""
        raise NotImplementedError(f"{type(self).__name__} does not support batch API")

    async def batch_results(self, batch_id: str) -> list[Any]:
        """Retrieve batch results."""
        raise NotImplementedError(f"{type(self).__name__} does not support batch API")

    async def count_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        """Count tokens for the given messages. Override in providers that support it."""
        raise NotImplementedError(f"{type(self).__name__} does not support token counting")

    # ------------------------------------------------------------------
    # Lifecycle — concrete no-ops, providers override if needed
    # ------------------------------------------------------------------

    async def close(self) -> None:  # noqa: B027
        """Release resources. Override in providers that hold clients."""

    async def __aenter__(self) -> BaseProvider:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


# ---------------------------------------------------------------------------
# Shared provider utilities
# ---------------------------------------------------------------------------


class StreamState:
    """Per-stream metadata accumulator (one per call). Used by all providers."""

    __slots__ = ("model", "raw", "stop_reason", "thinking", "tool_calls", "usage")

    def __init__(self) -> None:
        self.usage: Usage | None = None
        self.model: str = ""
        self.stop_reason: str = ""
        self.raw: Any = None
        self.tool_calls: list[ToolCall] = []
        self.thinking: list[ThinkingBlock] = []


def parse_tool_args(raw_args: str | dict[str, Any]) -> dict[str, Any]:
    """Parse tool call arguments (may be JSON string or dict)."""
    if isinstance(raw_args, dict):
        return raw_args
    try:
        return json.loads(raw_args)
    except (json.JSONDecodeError, TypeError):
        return {"_raw": raw_args}
