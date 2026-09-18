"""LLM facade: public call signatures and provider configuration."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any, ClassVar, Literal, cast

from ai_arch_toolkit.core._attempts import Arguments, Execution
from ai_arch_toolkit.core._content import user
from ai_arch_toolkit.core._exceptions import ProviderError, RequestError
from ai_arch_toolkit.core._metering._admission import NotMeteredOperationError
from ai_arch_toolkit.core._metering._scope import current_meter
from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._providers import _match_provider, create_provider, resolve_provider_name
from ai_arch_toolkit.core._response import (
    OutputSchema,
    Response,
    RichStreamResponse,
    StreamEvent,
    StreamResponse,
    SyncRichStreamResponse,
    SyncStreamResponse,
    _resolve_output_schema,
)
from ai_arch_toolkit.core._retry import RetryConfig
from ai_arch_toolkit.core._sync import _run_sync, _stream_sync
from ai_arch_toolkit.core._tools import prepare_tools
from ai_arch_toolkit.core._tools._group import ToolGroup

PROVIDER_ERRORS: tuple[type[Exception], ...] = (ProviderError,)


def _normalize_fallbacks(
    fallback: str | LLM | list[str | LLM] | None,
    api_key: str | None,
    base_url: str | None,
    provider: str | None = None,
) -> tuple[list[LLM], list[LLM]]:
    """Normalize fallback param into (all_fallbacks, owned_fallbacks).

    Strings are converted to new ``LLM`` instances (owned for lifecycle). A
    string fallback is routed by its own name: a recognizable model (e.g.
    ``claude-...``) is built standalone with its own connection, so a local
    primary can fail over to a cloud model; a bare, unroutable tag (e.g.
    ``llama3:8b``) inherits the parent's ``api_key``/``base_url``/``provider``,
    assuming it lives on the same server. Pass ``LLM`` instances for full
    per-fallback control.

    A fallback that has fallbacks of its own contributes its whole chain, in order, and a model
    reachable twice appears once. Nothing the caller passed is modified: a nested ``LLM`` keeps
    its chain (so it still falls back when used on its own, or under another parent) and keeps
    ownership of the fallbacks it created. ``owned`` holds only the instances created here.
    """
    if fallback is None:
        return [], []
    items: list[str | LLM] = fallback if isinstance(fallback, list) else [fallback]
    all_fbs: list[LLM] = []
    owned: list[LLM] = []
    for item in items:
        if isinstance(item, str):
            if _match_provider(item):
                fb = LLM(item)  # recognizable model → route by its own name
            else:
                fb = LLM(item, api_key=api_key, base_url=base_url, provider=provider)
            owned.append(fb)
        else:
            fb = item
        # ``fb._fallbacks`` is already flat: every LLM flattens its chain when it is built.
        for candidate in (fb, *fb._fallbacks):
            if not any(candidate is seen for seen in all_fbs):
                all_fbs.append(candidate)
    return all_fbs, owned


class LLM:
    """Async-first LLM client with convenient sync wrappers and reusable SDK clients."""

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        retry: RetryConfig | bool | None = None,
        middleware: list[Any] | None = None,
        fallback: str | LLM | list[str | LLM] | None = None,
        fallback_on: tuple[type[Exception], ...] | None = None,
        **kwargs: Any,
    ) -> None:
        self._validate_defaults(temperature, max_tokens, timeout)
        self._model = model
        self._timeout = timeout
        self._defaults: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        self._provider = create_provider(
            model, provider=provider, api_key=api_key, base_url=base_url, timeout=timeout
        )
        self._provider_name = self._resolve_provider_name(model, provider, base_url)
        self._retry = RetryConfig() if retry is True else None if retry is False else retry
        self._middleware: list[Any] = list(middleware) if middleware else []
        self._fallback_on = fallback_on or PROVIDER_ERRORS
        self._fallbacks, self._owned_fallbacks = _normalize_fallbacks(
            fallback, api_key=api_key, base_url=base_url, provider=provider
        )

    @staticmethod
    def _validate_defaults(temperature: float, max_tokens: int, timeout: float | None) -> None:
        if not (0.0 <= temperature <= 2.0):
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {temperature}")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError(f"max_tokens must be a positive integer, got {max_tokens}")
        if timeout is not None and timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")

    @staticmethod
    def _resolve_provider_name(
        model: str, provider: str | None, base_url: str | None
    ) -> str | None:
        try:
            return resolve_provider_name(model, provider=provider, base_url=base_url)
        except ValueError:
            return None

    _REPR_DEFAULTS: ClassVar[dict[str, Any]] = {"temperature": 0.0, "max_tokens": 4096}

    def __repr__(self) -> str:
        non_default = {k: v for k, v in self._defaults.items() if v != self._REPR_DEFAULTS.get(k)}
        parts = [f"model={self._model!r}"]
        parts.extend(f"{k}={v!r}" for k, v in non_default.items())
        if self._timeout is not None:
            parts.append(f"timeout={self._timeout!r}")
        if self._fallbacks:
            fb_models = [fb._model for fb in self._fallbacks]
            parts.append(f"fallback={fb_models!r}")
        return f"LLM({', '.join(parts)})"

    async def close(self) -> None:
        """Close the underlying provider client(s)."""
        await self._provider.close()
        for fb in self._owned_fallbacks:
            await fb.close()

    async def __aenter__(self) -> LLM:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def __enter__(self) -> LLM:
        return self

    def __exit__(self, *args: Any) -> None:
        _run_sync(self.close())

    @staticmethod
    def _normalize(messages: str | list[dict[str, Any]] | list) -> list[dict[str, Any]]:
        """Accept a bare string as shorthand for a single user message."""
        if isinstance(messages, str):
            return [user(messages)]
        if not isinstance(messages, list):
            raise TypeError(
                f"messages must be a string or list of dicts, got {type(messages).__name__}"
            )
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise TypeError(f"messages[{i}] must be a dict, got {type(msg).__name__}")
            if "role" not in msg:
                raise ValueError(f"messages[{i}] missing required 'role' key")
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
            raise RequestError("thinking_effort must be a non-empty string")
        if thinking_budget is not None and thinking_budget < 0:
            raise RequestError(f"thinking_budget must be non-negative, got {thinking_budget}")
        if json_mode and output_schema is not None:
            raise RequestError("json_mode and output_schema are mutually exclusive")
        for name, enabled in {
            "thinking": thinking,
            "json_mode": json_mode,
            "logprobs": logprobs,
        }.items():
            if enabled:
                kwargs[name] = True
        for name, value in {
            "thinking_effort": thinking_effort,
            "thinking_budget": thinking_budget,
            "tool_choice": tool_choice,
        }.items():
            if value is not None:
                kwargs[name] = value
        if output_schema is not None:
            kwargs["output_schema"] = _resolve_output_schema(output_schema)
        return kwargs

    def _prepare_call(
        self,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict[str, Any]] | None,
        arguments: Arguments,
    ) -> Request:
        """Validate common options before reserving a physical attempt."""
        return Request(
            messages=messages,
            system=system,
            tools=tools,
            model=self._model,
            kwargs=self._prepare_provider_kwargs(
                **arguments.options, extra=self._merge_kwargs(**arguments.extra)
            ),
        )

    def _execution(
        self,
        messages: str | list[dict[str, Any]],
        *,
        path: Literal["complete", "stream", "stream_events"],
        system: str | None,
        tools: list[dict[str, Any]] | ToolGroup | Callable[..., Any] | None,
        options: dict[str, Any],
        extra: dict[str, Any],
    ) -> Execution:
        arguments = Arguments(options=options, extra=extra)
        request = self._prepare_call(
            self._normalize(messages), system, prepare_tools(tools), arguments
        )
        return Execution(self, request, arguments, path)

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
        execution = self._execution(
            messages,
            path="complete",
            system=system,
            tools=tools,
            options={
                "thinking": thinking,
                "thinking_effort": thinking_effort,
                "thinking_budget": thinking_budget,
                "output_schema": output_schema,
                "tool_choice": tool_choice,
                "json_mode": json_mode,
                "logprobs": logprobs,
            },
            extra=kwargs,
        )
        return await execution.complete()

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
        """Stream with retry/fallback permitted before the first observable item."""
        execution = self._execution(
            messages,
            path="stream",
            system=system,
            tools=tools,
            options={
                "thinking": thinking,
                "thinking_effort": thinking_effort,
                "thinking_budget": thinking_budget,
                "output_schema": output_schema,
                "tool_choice": tool_choice,
                "json_mode": json_mode,
                "logprobs": logprobs,
            },
            extra=kwargs,
        )
        return StreamResponse(
            cast("AsyncIterator[str]", execution.items()),
            execution.finalize,
            lifecycle=execution,
        )

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
        """Stream with retry/fallback permitted before the first observable item."""
        execution = self._execution(
            messages,
            path="stream_events",
            system=system,
            tools=tools,
            options={
                "thinking": thinking,
                "thinking_effort": thinking_effort,
                "thinking_budget": thinking_budget,
                "output_schema": output_schema,
                "tool_choice": tool_choice,
                "json_mode": json_mode,
                "logprobs": logprobs,
            },
            extra=kwargs,
        )
        return RichStreamResponse(
            cast("AsyncIterator[StreamEvent]", execution.items()),
            execution.finalize,
            lifecycle=execution,
        )

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
        rich_stream = self.stream_events(
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
        sync_iter = _stream_sync(lambda: rich_stream._aiter)
        return SyncRichStreamResponse(
            sync_iter, rich_stream._finalizer, lifecycle=rich_stream._lifecycle
        )

    async def __call__(
        self,
        messages: str | list[dict[str, Any]],
        **kwargs: Any,
    ) -> Response:
        """Alias for ``complete()``."""
        return await self.complete(messages, **kwargs)

    def _reject_batch_under_enforcement(self) -> None:
        """Batch bypasses per-attempt metering; refuse it inside an enforcing budget (fail-closed).

        Measure-only (``controller=None``) and unmetered runs are unaffected — batch simply is
        not metered there (a documented gap, reconciled post-hoc from provider records).
        """
        scope = current_meter()
        if scope is not None and scope.controller is not None and not scope.allow_unmetered_batch:
            raise NotMeteredOperationError(
                "batch operations bypass per-attempt metering and are not allowed under an "
                "enforcing budget; use complete()/stream(), set RunConfig.allow_unmetered_batch, "
                "or run the batch without a controller"
            )

    async def batch_submit(self, requests: list[dict[str, Any]]) -> str:
        """Submit a batch of requests. Returns a batch ID."""
        self._reject_batch_under_enforcement()
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
        async_stream = self.stream(
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
        sync_iter = _stream_sync(lambda: async_stream._aiter)
        return SyncStreamResponse(
            sync_iter, async_stream._finalizer, lifecycle=async_stream._lifecycle
        )
