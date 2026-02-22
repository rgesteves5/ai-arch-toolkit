"""User-facing LLM class — stateless function: content → content."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from ai_arch_toolkit._content import user
from ai_arch_toolkit._providers import create_provider
from ai_arch_toolkit._response import Response
from ai_arch_toolkit._sync import _run_sync, _stream_sync


class LLM:
    """Lightweight LLM client.

    Async-first with convenient sync wrappers.
    """

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._model = model
        self._defaults: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        self._provider = create_provider(model, api_key=api_key, base_url=base_url)

    _INIT_DEFAULTS: dict[str, Any] = {"temperature": 0.0, "max_tokens": 4096}

    def __repr__(self) -> str:
        non_default = {
            k: v for k, v in self._defaults.items() if v != self._INIT_DEFAULTS.get(k)
        }
        params = ", ".join(f"{k}={v!r}" for k, v in non_default.items())
        if params:
            return f"LLM(model={self._model!r}, {params})"
        return f"LLM(model={self._model!r})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(messages: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Accept a bare string as shorthand for a single user message."""
        if isinstance(messages, str):
            return [user(messages)]
        return messages

    def _merge_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        merged = dict(self._defaults)
        merged.update({k: v for k, v in kwargs.items() if v is not None})
        return merged

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: str | list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Response:
        """Send messages and return a Response."""
        normalized = self._normalize(messages)
        merged = self._merge_kwargs(**kwargs)
        return await self._provider.complete(normalized, system=system, tools=tools, **merged)

    async def stream(
        self,
        messages: str | list[dict[str, Any]],
        *,
        system: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream text chunks."""
        normalized = self._normalize(messages)
        merged = self._merge_kwargs(**kwargs)
        async for chunk in self._provider.stream(normalized, system=system, **merged):
            yield chunk

    async def __call__(
        self,
        messages: str | list[dict[str, Any]],
        **kwargs: Any,
    ) -> Response:
        """Alias for ``complete()``."""
        return await self.complete(messages, **kwargs)

    # ------------------------------------------------------------------
    # Sync wrappers
    # ------------------------------------------------------------------

    def complete_sync(
        self,
        messages: str | list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Response:
        """Synchronous version of ``complete()``."""
        return _run_sync(self.complete(messages, system=system, tools=tools, **kwargs))

    def stream_sync(
        self,
        messages: str | list[dict[str, Any]],
        *,
        system: str | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Synchronous version of ``stream()``."""
        return _stream_sync(lambda: self.stream(messages, system=system, **kwargs))
