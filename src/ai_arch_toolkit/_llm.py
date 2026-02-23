"""User-facing LLM class — stateless function: content → content."""

from __future__ import annotations

from typing import Any, ClassVar

from ai_arch_toolkit._content import user
from ai_arch_toolkit._pricing import pricing
from ai_arch_toolkit._providers import create_provider
from ai_arch_toolkit._response import Response, StreamResponse, SyncStreamResponse, Usage
from ai_arch_toolkit._sync import _run_sync, _stream_sync
from ai_arch_toolkit._tools import prepare_tools


class LLM:
    """Lightweight LLM client.

    Async-first with convenient sync wrappers.
    Supports context managers for client reuse.
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

    _INIT_DEFAULTS: ClassVar[dict[str, Any]] = {"temperature": 0.0, "max_tokens": 4096}

    def __repr__(self) -> str:
        non_default = {k: v for k, v in self._defaults.items() if v != self._INIT_DEFAULTS.get(k)}
        params = ", ".join(f"{k}={v!r}" for k, v in non_default.items())
        if params:
            return f"LLM(model={self._model!r}, {params})"
        return f"LLM(model={self._model!r})"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._provider.close()

    async def __aenter__(self) -> LLM:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def __enter__(self) -> LLM:
        return self

    def __exit__(self, *args: Any) -> None:
        _run_sync(self.close())

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
        tools: Any | None = None,
        **kwargs: Any,
    ) -> Response:
        """Send messages and return a Response."""
        normalized = self._normalize(messages)
        merged = self._merge_kwargs(**kwargs)
        wire_tools = prepare_tools(tools)
        return await self._provider.complete(normalized, system=system, tools=wire_tools, **merged)

    def stream(
        self,
        messages: str | list[dict[str, Any]],
        *,
        system: str | None = None,
        **kwargs: Any,
    ) -> StreamResponse:
        """Stream text chunks, with metadata available after consumption."""
        normalized = self._normalize(messages)
        merged = self._merge_kwargs(**kwargs)
        aiter, state = self._provider.stream(normalized, system=system, **merged)
        model = self._model

        def _finalize(text: str) -> Response:
            usage = state.usage or Usage()
            cost, cost_estimated = pricing.estimate_cost(
                model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_write_tokens=usage.cache_write_tokens,
                cache_read_tokens=usage.cache_read_tokens,
            )
            return Response(
                text=text,
                usage=usage,
                cost=cost,
                cost_estimated=cost_estimated,
                stop_reason=state.stop_reason,
                model=state.model or model,
            )

        return StreamResponse(aiter, _finalize)

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
        tools: Any | None = None,
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
    ) -> SyncStreamResponse:
        """Synchronous version of ``stream()``."""
        normalized = self._normalize(messages)
        merged = self._merge_kwargs(**kwargs)

        # We need the state from the provider, but the provider call happens
        # in the async world. Bridge: create a holder, populate in the async
        # factory, read after iteration.
        state_holder: list[Any] = []
        model = self._model

        def _async_factory():
            aiter, state = self._provider.stream(normalized, system=system, **merged)
            state_holder.append(state)
            return aiter

        sync_iter = _stream_sync(_async_factory)

        def _finalize(text: str) -> Response:
            state = state_holder[0] if state_holder else None
            usage = (state.usage if state else None) or Usage()
            cost, cost_estimated = pricing.estimate_cost(
                model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_write_tokens=usage.cache_write_tokens,
                cache_read_tokens=usage.cache_read_tokens,
            )
            return Response(
                text=text,
                usage=usage,
                cost=cost,
                cost_estimated=cost_estimated,
                stop_reason=getattr(state, "stop_reason", ""),
                model=getattr(state, "model", "") or model,
            )

        return SyncStreamResponse(sync_iter, _finalize)
