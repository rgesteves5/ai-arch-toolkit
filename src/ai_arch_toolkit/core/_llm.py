"""User-facing LLM class — stateless function: content → content."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, ClassVar

from ai_arch_toolkit.core._content import user
from ai_arch_toolkit.core._exceptions import APIError
from ai_arch_toolkit.core._middleware import (
    Request,
    _run_aafter,
    _run_abefore,
    _run_after,
    _run_before,
)
from ai_arch_toolkit.core._pricing import _estimate_response_cost
from ai_arch_toolkit.core._providers import create_provider
from ai_arch_toolkit.core._response import (
    OutputSchema,
    Response,
    RichStreamResponse,
    StreamResponse,
    SyncRichStreamResponse,
    SyncStreamResponse,
    Usage,
    _resolve_output_schema,
)
from ai_arch_toolkit.core._retry import RetryConfig, with_retry
from ai_arch_toolkit.core._sync import _run_sync, _stream_sync
from ai_arch_toolkit.core._tools import prepare_tools
from ai_arch_toolkit.core._tools._group import ToolGroup

logger = logging.getLogger(__name__)


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
        retry: RetryConfig | bool | None = None,
        middleware: list[Any] | None = None,
        fallback: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._model = model
        self._defaults: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        self._provider = create_provider(model, api_key=api_key, base_url=base_url)
        self._retry: RetryConfig | None = RetryConfig() if retry is True else retry
        self._middleware: list[Any] = list(middleware) if middleware else []
        self._fallback_provider = (
            create_provider(fallback, api_key=api_key, base_url=base_url) if fallback else None
        )
        self._fallback_model = fallback

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
        """Close the underlying provider client(s)."""
        await self._provider.close()
        if self._fallback_provider:
            await self._fallback_provider.close()

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
    def _normalize(messages: str | list[dict[str, Any]] | list) -> list[dict[str, Any]]:
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
        tool_choice: str | None,
        json_mode: bool,
        logprobs: bool,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        """Build kwargs to forward to the provider."""
        kwargs = dict(extra)
        if thinking_effort is not None and not thinking_effort:
            raise ValueError("thinking_effort must be a non-empty string")
        if thinking_budget is not None and thinking_budget < 0:
            raise ValueError(f"thinking_budget must be non-negative, got {thinking_budget}")
        if json_mode and output_schema is not None:
            raise ValueError("json_mode and output_schema are mutually exclusive")
        if thinking:
            kwargs["thinking"] = True
        if thinking_effort is not None:
            kwargs["thinking_effort"] = thinking_effort
        if thinking_budget is not None:
            kwargs["thinking_budget"] = thinking_budget
        if output_schema is not None:
            kwargs["output_schema"] = _resolve_output_schema(output_schema)
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if json_mode:
            kwargs["json_mode"] = True
        if logprobs:
            kwargs["logprobs"] = True
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
        tool_choice: str | None = None,
        json_mode: bool = False,
        logprobs: bool = False,
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
            tool_choice=tool_choice,
            json_mode=json_mode,
            logprobs=logprobs,
            extra=merged,
        )
        # Middleware before hooks
        req: Request | None = None
        if self._middleware:
            req = Request(
                messages=normalized,
                system=system,
                tools=wire_tools,
                model=self._model,
                kwargs=provider_kwargs,
            )
            req = await _run_abefore(self._middleware, req)
            normalized = req.messages
            system = req.system
            wire_tools = req.tools
            provider_kwargs = req.kwargs

        async def _call() -> Response:
            return await self._provider.complete(
                normalized, system=system, tools=wire_tools, **provider_kwargs
            )

        try:
            if self._retry:
                response = await with_retry(_call, self._retry)
            else:
                response = await _call()
        except APIError:
            if not self._fallback_provider:
                raise
            logger.info("Primary provider failed, trying fallback: %s", self._fallback_model)

            async def _fallback_call() -> Response:
                return await self._fallback_provider.complete(  # type: ignore[union-attr]
                    normalized, system=system, tools=wire_tools, **provider_kwargs
                )

            if self._retry:
                response = await with_retry(_fallback_call, self._retry)
            else:
                response = await _fallback_call()

        # Middleware after hooks
        if self._middleware and req is not None:
            response = await _run_aafter(self._middleware, req, response)

        return response

    def _build_stream_finalizer(self, state: Any) -> Callable[[str], Response]:
        """Build a finalization callback for stream wrappers."""
        model = self._model

        def _finalize(text: str) -> Response:
            usage = state.usage or Usage()
            thinking_blocks = tuple(getattr(state, "thinking", []))
            cost = _estimate_response_cost(model, usage)
            resp = Response(
                text=text,
                tool_calls=tuple(state.tool_calls),
                thinking=thinking_blocks,
                usage=usage,
                cost=cost,
                stop_reason=state.stop_reason,
                model=state.model or model,
                raw=state.raw,
            )
            return resp

        return _finalize

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
        tool_choice: str | None = None,
        json_mode: bool = False,
        logprobs: bool = False,
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
            tool_choice=tool_choice,
            json_mode=json_mode,
            logprobs=logprobs,
            extra=merged,
        )

        # Middleware before hooks
        req: Request | None = None
        if self._middleware:
            req = Request(
                messages=normalized,
                system=system,
                tools=wire_tools,
                model=self._model,
                kwargs=provider_kwargs,
            )
            req = _run_before(self._middleware, req)
            normalized = req.messages
            system = req.system
            wire_tools = req.tools
            provider_kwargs = req.kwargs

        try:
            aiter, state = self._provider.stream(
                normalized, system=system, tools=wire_tools, **provider_kwargs
            )
        except APIError:
            if not self._fallback_provider:
                raise
            logger.info("Primary stream failed, trying fallback: %s", self._fallback_model)
            aiter, state = self._fallback_provider.stream(
                normalized, system=system, tools=wire_tools, **provider_kwargs
            )

        finalizer = self._build_stream_finalizer(state)

        if self._middleware and req is not None:
            inner_finalizer = finalizer
            mw_list = self._middleware
            mw_req = req

            def _mw_finalize(text: str) -> Response:
                resp = inner_finalizer(text)
                return _run_after(mw_list, mw_req, resp)

            finalizer = _mw_finalize

        return StreamResponse(aiter, finalizer)

    def stream_events(
        self,
        messages: str | list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | ToolGroup | Callable[..., Any] | None = None,
        thinking: bool = False,
        thinking_effort: str | None = None,
        thinking_budget: int | None = None,
        output_schema: OutputSchema | type | None = None,
        tool_choice: str | None = None,
        json_mode: bool = False,
        logprobs: bool = False,
        **kwargs: Any,
    ) -> RichStreamResponse:
        """Stream structured events (text, thinking, tool_call)."""
        normalized = self._normalize(messages)
        merged = self._merge_kwargs(**kwargs)
        wire_tools = prepare_tools(tools)
        provider_kwargs = self._prepare_provider_kwargs(
            thinking=thinking,
            thinking_effort=thinking_effort,
            thinking_budget=thinking_budget,
            output_schema=output_schema,
            tool_choice=tool_choice,
            json_mode=json_mode,
            logprobs=logprobs,
            extra=merged,
        )

        # Middleware before hooks
        req: Request | None = None
        if self._middleware:
            req = Request(
                messages=normalized,
                system=system,
                tools=wire_tools,
                model=self._model,
                kwargs=provider_kwargs,
            )
            req = _run_before(self._middleware, req)
            normalized = req.messages
            system = req.system
            wire_tools = req.tools
            provider_kwargs = req.kwargs

        try:
            aiter, state = self._provider.stream_events(
                normalized, system=system, tools=wire_tools, **provider_kwargs
            )
        except APIError:
            if not self._fallback_provider:
                raise
            logger.info("Primary stream_events failed, trying fallback: %s", self._fallback_model)
            aiter, state = self._fallback_provider.stream_events(
                normalized, system=system, tools=wire_tools, **provider_kwargs
            )

        finalizer = self._build_stream_finalizer(state)

        if self._middleware and req is not None:
            inner_finalizer = finalizer
            mw_list = self._middleware
            mw_req = req

            def _mw_finalize(text: str) -> Response:
                resp = inner_finalizer(text)
                return _run_after(mw_list, mw_req, resp)

            finalizer = _mw_finalize

        return RichStreamResponse(aiter, finalizer)

    def stream_events_sync(
        self,
        messages: str | list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | ToolGroup | Callable[..., Any] | None = None,
        thinking: bool = False,
        thinking_effort: str | None = None,
        thinking_budget: int | None = None,
        output_schema: OutputSchema | type | None = None,
        tool_choice: str | None = None,
        json_mode: bool = False,
        logprobs: bool = False,
        **kwargs: Any,
    ) -> SyncRichStreamResponse:
        """Synchronous version of ``stream_events()``."""
        normalized = self._normalize(messages)
        merged = self._merge_kwargs(**kwargs)
        wire_tools = prepare_tools(tools)
        provider_kwargs = self._prepare_provider_kwargs(
            thinking=thinking,
            thinking_effort=thinking_effort,
            thinking_budget=thinking_budget,
            output_schema=output_schema,
            tool_choice=tool_choice,
            json_mode=json_mode,
            logprobs=logprobs,
            extra=merged,
        )

        state_ref = _StateRef()

        def _async_factory():
            aiter, state = self._provider.stream_events(
                normalized, system=system, tools=wire_tools, **provider_kwargs
            )
            state_ref.value = state
            return aiter

        sync_iter = _stream_sync(_async_factory)

        model = self._model

        def _finalize(text: str) -> Response:
            state = state_ref.value
            usage = (state.usage if state else None) or Usage()
            tool_calls = tuple(state.tool_calls) if state else ()
            thinking_blocks = tuple(getattr(state, "thinking", [])) if state else ()
            cost = _estimate_response_cost(model, usage)
            return Response(
                text=text,
                tool_calls=tool_calls,
                thinking=thinking_blocks,
                usage=usage,
                cost=cost,
                stop_reason=getattr(state, "stop_reason", ""),
                model=getattr(state, "model", "") or model,
                raw=getattr(state, "raw", None),
            )

        return SyncRichStreamResponse(sync_iter, _finalize)

    async def __call__(
        self,
        messages: str | list[dict[str, Any]],
        **kwargs: Any,
    ) -> Response:
        """Alias for ``complete()``."""
        return await self.complete(messages, **kwargs)

    # ------------------------------------------------------------------
    # Batch API
    # ------------------------------------------------------------------

    async def batch_submit(self, requests: list[dict[str, Any]]) -> str:
        """Submit a batch of requests. Returns a batch ID."""
        return await self._provider.batch_submit(requests)

    async def batch_status(self, batch_id: str) -> str:
        """Check batch processing status."""
        return await self._provider.batch_status(batch_id)

    async def batch_results(self, batch_id: str) -> list[Any]:
        """Retrieve batch results."""
        return await self._provider.batch_results(batch_id)

    def batch_submit_sync(self, requests: list[dict[str, Any]]) -> str:
        """Synchronous version of ``batch_submit()``."""
        return _run_sync(self.batch_submit(requests))

    def batch_status_sync(self, batch_id: str) -> str:
        """Synchronous version of ``batch_status()``."""
        return _run_sync(self.batch_status(batch_id))

    def batch_results_sync(self, batch_id: str) -> list[Any]:
        """Synchronous version of ``batch_results()``."""
        return _run_sync(self.batch_results(batch_id))

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    async def count_tokens(
        self,
        messages: str | list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | ToolGroup | Callable[..., Any] | None = None,
    ) -> int:
        """Count tokens for the given messages (provider-dependent)."""
        normalized = self._normalize(messages)
        wire_tools = prepare_tools(tools)
        return await self._provider.count_tokens(normalized, system=system, tools=wire_tools)

    def count_tokens_sync(
        self,
        messages: str | list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | ToolGroup | Callable[..., Any] | None = None,
    ) -> int:
        """Synchronous version of ``count_tokens()``."""
        return _run_sync(self.count_tokens(messages, system=system, tools=tools))

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
        tool_choice: str | None = None,
        json_mode: bool = False,
        logprobs: bool = False,
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
                tool_choice=tool_choice,
                json_mode=json_mode,
                logprobs=logprobs,
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
        tool_choice: str | None = None,
        json_mode: bool = False,
        logprobs: bool = False,
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
            tool_choice=tool_choice,
            json_mode=json_mode,
            logprobs=logprobs,
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
            cost = _estimate_response_cost(model, usage)
            return Response(
                text=text,
                tool_calls=tool_calls,
                thinking=thinking_blocks,
                usage=usage,
                cost=cost,
                stop_reason=getattr(state, "stop_reason", ""),
                model=getattr(state, "model", "") or model,
                raw=getattr(state, "raw", None),
            )

        return SyncStreamResponse(sync_iter, _finalize)
