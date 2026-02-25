"""User-facing LLM class — stateless function: content → content."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from ai_arch_toolkit.core._content import user
from ai_arch_toolkit.core._pricing import _estimate_response_cost
from ai_arch_toolkit.core._providers import create_provider
from ai_arch_toolkit.core._response import (
    OutputSchema,
    Response,
    StreamResponse,
    SyncStreamResponse,
    Usage,
    _resolve_output_schema,
)
from ai_arch_toolkit.core._sync import _run_sync, _stream_sync
from ai_arch_toolkit.core._tools import prepare_tools
from ai_arch_toolkit.core._tools._group import ToolGroup


class _StateRef:
    """Mutable container for stream state (replaces list hack)."""

    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value: Any = None


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

    _REPR_DEFAULTS: ClassVar[dict[str, Any]] = {"temperature": 0.0, "max_tokens": 4096}

    def __repr__(self) -> str:
        non_default = {k: v for k, v in self._defaults.items() if v != self._REPR_DEFAULTS.get(k)}
        params = ", ".join(f"{k}={v!r}" for k, v in non_default.items())
        if params:
            return f"LLM(model={self._model!r}, {params})"
        return f"LLM(model={self._model!r})"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying provider client."""
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

    @staticmethod
    def _prepare_provider_kwargs(
        *,
        thinking: bool,
        thinking_effort: str | None,
        thinking_budget: int | None,
        output_schema: OutputSchema | type | None,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        """Build kwargs to forward to the provider."""
        kwargs = dict(extra)
        if thinking_effort is not None and not thinking_effort:
            raise ValueError("thinking_effort must be a non-empty string")
        if thinking_budget is not None and thinking_budget < 0:
            raise ValueError(f"thinking_budget must be non-negative, got {thinking_budget}")
        if thinking:
            kwargs["thinking"] = True
        if thinking_effort is not None:
            kwargs["thinking_effort"] = thinking_effort
        if thinking_budget is not None:
            kwargs["thinking_budget"] = thinking_budget
        if output_schema is not None:
            kwargs["output_schema"] = _resolve_output_schema(output_schema)
        return kwargs

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: str | list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | ToolGroup | Callable[..., Any] | None = None,
        thinking: bool = False,
        thinking_effort: str | None = None,
        thinking_budget: int | None = None,
        output_schema: OutputSchema | type | None = None,
        **kwargs: Any,
    ) -> Response:
        """Send messages and return a Response."""
        normalized = self._normalize(messages)
        merged = self._merge_kwargs(**kwargs)
        wire_tools = prepare_tools(tools)
        provider_kwargs = self._prepare_provider_kwargs(
            thinking=thinking,
            thinking_effort=thinking_effort,
            thinking_budget=thinking_budget,
            output_schema=output_schema,
            extra=merged,
        )
        return await self._provider.complete(
            normalized, system=system, tools=wire_tools, **provider_kwargs
        )

    def stream(
        self,
        messages: str | list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | ToolGroup | Callable[..., Any] | None = None,
        thinking: bool = False,
        thinking_effort: str | None = None,
        thinking_budget: int | None = None,
        output_schema: OutputSchema | type | None = None,
        **kwargs: Any,
    ) -> StreamResponse:
        """Stream text chunks, with metadata available after consumption."""
        normalized = self._normalize(messages)
        merged = self._merge_kwargs(**kwargs)
        wire_tools = prepare_tools(tools)
        provider_kwargs = self._prepare_provider_kwargs(
            thinking=thinking,
            thinking_effort=thinking_effort,
            thinking_budget=thinking_budget,
            output_schema=output_schema,
            extra=merged,
        )
        aiter, state = self._provider.stream(
            normalized, system=system, tools=wire_tools, **provider_kwargs
        )
        model = self._model

        def _finalize(text: str) -> Response:
            usage = state.usage or Usage()
            thinking_blocks = tuple(getattr(state, "thinking", []))
            cost, cost_estimated = _estimate_response_cost(model, usage)
            return Response(
                text=text,
                tool_calls=tuple(state.tool_calls),
                thinking=thinking_blocks,
                usage=usage,
                cost=cost,
                cost_estimated=cost_estimated,
                stop_reason=state.stop_reason,
                model=state.model or model,
                raw=state.raw,
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
        tools: list[dict[str, Any]] | ToolGroup | Callable[..., Any] | None = None,
        thinking: bool = False,
        thinking_effort: str | None = None,
        thinking_budget: int | None = None,
        output_schema: OutputSchema | type | None = None,
        **kwargs: Any,
    ) -> Response:
        """Synchronous version of ``complete()``."""
        return _run_sync(
            self.complete(
                messages,
                system=system,
                tools=tools,
                thinking=thinking,
                thinking_effort=thinking_effort,
                thinking_budget=thinking_budget,
                output_schema=output_schema,
                **kwargs,
            )
        )

    def stream_sync(
        self,
        messages: str | list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | ToolGroup | Callable[..., Any] | None = None,
        thinking: bool = False,
        thinking_effort: str | None = None,
        thinking_budget: int | None = None,
        output_schema: OutputSchema | type | None = None,
        **kwargs: Any,
    ) -> SyncStreamResponse:
        """Synchronous version of ``stream()``."""
        normalized = self._normalize(messages)
        merged = self._merge_kwargs(**kwargs)
        wire_tools = prepare_tools(tools)
        provider_kwargs = self._prepare_provider_kwargs(
            thinking=thinking,
            thinking_effort=thinking_effort,
            thinking_budget=thinking_budget,
            output_schema=output_schema,
            extra=merged,
        )

        state_ref = _StateRef()
        model = self._model

        def _async_factory():
            aiter, state = self._provider.stream(
                normalized, system=system, tools=wire_tools, **provider_kwargs
            )
            state_ref.value = state
            return aiter

        sync_iter = _stream_sync(_async_factory)

        def _finalize(text: str) -> Response:
            state = state_ref.value
            usage = (state.usage if state else None) or Usage()
            tool_calls = tuple(state.tool_calls) if state else ()
            thinking_blocks = tuple(getattr(state, "thinking", [])) if state else ()
            cost, cost_estimated = _estimate_response_cost(model, usage)
            return Response(
                text=text,
                tool_calls=tool_calls,
                thinking=thinking_blocks,
                usage=usage,
                cost=cost,
                cost_estimated=cost_estimated,
                stop_reason=getattr(state, "stop_reason", ""),
                model=getattr(state, "model", "") or model,
                raw=getattr(state, "raw", None),
            )

        return SyncStreamResponse(sync_iter, _finalize)
