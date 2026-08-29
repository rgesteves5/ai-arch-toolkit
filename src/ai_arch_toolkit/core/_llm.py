"""User-facing LLM class — stateless function: content → content."""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, ClassVar, Literal, cast

from ai_arch_toolkit.core._concurrency import inference_slot
from ai_arch_toolkit.core._content import user
from ai_arch_toolkit.core._exceptions import APIError
from ai_arch_toolkit.core._metering._admission import AdmissionDenied, NotMeteredOperationError
from ai_arch_toolkit.core._metering._cost import Cost
from ai_arch_toolkit.core._metering._money import Money
from ai_arch_toolkit.core._metering._operation import MeterOperation, OperationRequest
from ai_arch_toolkit.core._metering._scope import current_meter, current_span_id
from ai_arch_toolkit.core._middleware import (
    Request,
    _run_aafter,
    _run_abefore,
    _run_after,
    _run_before,
)
from ai_arch_toolkit.core._pricing import _estimate_response_cost, pricing
from ai_arch_toolkit.core._providers import _match_provider, create_provider
from ai_arch_toolkit.core._providers._base import StreamState
from ai_arch_toolkit.core._response import (
    Attempt,
    OutputSchema,
    Response,
    RichStreamResponse,
    StreamResponse,
    SyncRichStreamResponse,
    SyncStreamResponse,
    Usage,
    _resolve_output_schema,
)
from ai_arch_toolkit.core._retry import RetryConfig, _wait_before_retry, with_retry
from ai_arch_toolkit.core._sync import _run_sync, _stream_sync
from ai_arch_toolkit.core._tools import prepare_tools
from ai_arch_toolkit.core._tools._group import ToolGroup

logger = logging.getLogger(__name__)


def _price_or_unknown(pricer: Any, request: OperationRequest, usage: Usage) -> Cost:
    """Price a settled call defensively. A raising or estimate-returning (foreign) pricer must not
    turn a successful, already-served response into a failure — fall back to an unknown cost so the
    op still settles (the store rejects estimates and would otherwise raise out of a success path).
    """
    try:
        cost = pricer.price(request, usage)
    except Exception:
        logger.exception("pricer %r raised while pricing a settled call; using unknown", pricer)
        return Cost.unknown("pricer raised")
    if cost.kind == "estimated":
        logger.warning("pricer %r returned an estimate at settle; recording unknown", pricer)
        return Cost.unknown("pricer returned an estimate at settle")
    return cost


def _settlement_cost(pricer: Any, request: OperationRequest, response: Response) -> Cost:
    """Prefer an exact provider cost under the default pricer, then estimate from usage.

    An explicitly configured custom pricer remains authoritative. Provider costs are actual
    per-request charges, while the default registry is only a token-based fallback.
    """
    if pricer is pricing and response.provider_cost is not None:
        return Cost.known(Money.from_usd(response.provider_cost))
    return _price_or_unknown(pricer, request, response.usage)


PROVIDER_ERRORS: tuple[type[Exception], ...] = (
    APIError,
    ConnectionError,  # subclass of OSError, listed explicitly for clarity
    TimeoutError,
    OSError,
)


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
    per-fallback control. Nested fallbacks are flattened into the parent chain.

    .. note:: Flattening **clears** the nested LLM's ``_fallbacks`` list so
       that the parent owns the full chain. Passing the same ``LLM`` instance
       as a nested fallback to multiple parents is not supported — only the
       first parent will receive the nested chain.
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
            all_fbs.append(fb)
            owned.append(fb)
        else:
            all_fbs.append(item)
        # Flatten nested fallbacks from the just-added LLM
        fb_llm = all_fbs[-1]
        if fb_llm._fallbacks:
            # Copy before clearing — .extend() reads before the mutation
            all_fbs.extend(list(fb_llm._fallbacks))
            owned.extend(list(fb_llm._owned_fallbacks))
            fb_llm._fallbacks = []
            fb_llm._owned_fallbacks = []
    return all_fbs, owned


class _StreamRun:
    """Thread-safe lifecycle for one lazily-started stream and all of its retries."""

    __slots__ = (
        "_lock",
        "attempts",
        "finalized",
        "meter",
        "model",
        "retry_number",
        "started_at",
        "state",
        "terminal",
    )

    def __init__(self, model: str) -> None:
        self._lock = threading.Lock()
        self.model = model
        self.attempts: list[Attempt] = []
        self.state: StreamState | None = None
        self.meter: tuple[MeterOperation, OperationRequest, Any] | None = None
        self.started_at = 0.0
        self.retry_number = 0
        self.terminal = True
        self.finalized = True

    def begin(
        self,
        retry_number: int,
        meter: tuple[MeterOperation, OperationRequest, Any] | None,
    ) -> None:
        with self._lock:
            self.state = None
            self.meter = meter
            self.started_at = time.time()
            self.retry_number = retry_number
            self.terminal = False
            self.finalized = False

    def start_attempt(self) -> bool:
        """Latch the physical-attempt start time; false if already abandoned."""
        with self._lock:
            if self.terminal:
                return False
            self.started_at = time.time()
            return True

    def set_state(self, state: StreamState) -> None:
        with self._lock:
            if not self.terminal:
                self.state = state

    def succeed(self) -> None:
        with self._lock:
            if self.terminal:
                return
            usage = self.state.usage if self.state is not None else Usage()
            self.attempts.append(
                Attempt(
                    model=self.model,
                    status="ok",
                    usage=usage,
                    duration=time.time() - self.started_at,
                    timestamp=self.started_at,
                    retry_number=self.retry_number,
                )
            )
            self.terminal = True

    def fail(self, exc: BaseException) -> None:
        with self._lock:
            if self.terminal:
                return
            self.attempts.append(
                Attempt(
                    model=self.model,
                    status="failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                    status_code=getattr(exc, "status_code", None),
                    duration=time.time() - self.started_at,
                    timestamp=self.started_at,
                    retry_number=self.retry_number,
                )
            )
            if self.meter is not None:
                self.meter[0].fail()
            self.terminal = True
            self.finalized = True

    def abandon(self) -> None:
        """Fail an observable stream that the consumer did not fully drain."""
        with self._lock:
            if self.finalized:
                return
            if self.terminal and self.attempts and self.attempts[-1].status == "ok":
                self.attempts.pop()
            self.attempts.append(
                Attempt(
                    model=self.model,
                    status="failed",
                    error="stream abandoned before full consumption",
                    error_type="StreamAbandoned",
                    duration=time.time() - self.started_at,
                    timestamp=self.started_at,
                    retry_number=self.retry_number,
                )
            )
            if self.meter is not None:
                self.meter[0].mark_abandoned()
                self.meter[0].fail()
            self.terminal = True
            self.finalized = True

    def finalization_snapshot(
        self,
    ) -> tuple[
        StreamState | None,
        tuple[MeterOperation, OperationRequest, Any] | None,
        tuple[Attempt, ...],
    ]:
        """Return one stable finalization view shared by async and sync wrappers."""
        with self._lock:
            self.finalized = True
            return self.state, self.meter, tuple(self.attempts)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _StreamFinalizer:
    """Callable stream finalizer with explicit lifecycle metadata."""

    callback: Callable[[str], Response]
    _stream_abandon: Callable[[], None]
    _stream_attempts: list[Attempt]
    _stream_request: Request | None
    _stream_system: str | None

    def __call__(self, text: str) -> Response:
        return self.callback(text)


class _FallbackStreamRun:
    """Thread-safe active-candidate and prior-attempt state for a fallback stream."""

    __slots__ = ("_abandoned", "_lock", "owner", "prior_attempts", "stream")

    def __init__(
        self,
        owner: LLM,
        stream: StreamResponse | RichStreamResponse,
    ) -> None:
        self._abandoned = False
        self._lock = threading.Lock()
        self.owner = owner
        self.stream = stream
        self.prior_attempts: list[Attempt] = []

    def activate(
        self,
        owner: LLM,
        stream: StreamResponse | RichStreamResponse,
    ) -> bool:
        with self._lock:
            if self._abandoned:
                return False
            self.owner = owner
            self.stream = stream
            return True

    def record_attempts(self, attempts: tuple[Attempt, ...]) -> None:
        with self._lock:
            self.prior_attempts.extend(attempts)

    def snapshot(
        self,
    ) -> tuple[LLM, StreamResponse | RichStreamResponse, tuple[Attempt, ...]]:
        with self._lock:
            return self.owner, self.stream, tuple(self.prior_attempts)

    def abandon(self) -> None:
        """Latch abandonment and stop whichever candidate is currently active."""
        with self._lock:
            self._abandoned = True
            abandon = getattr(self.stream._finalizer, "_stream_abandon", None)
            if callable(abandon):
                abandon()


def _content_chars(
    normalized: list[dict[str, Any]],
    system: str | None,
    wire_tools: list[dict[str, Any]] | None,
    output_schema: Any = None,
) -> int:
    """Rough char count of the whole request — a fact for the estimator (over-counts a bit).

    Includes the output schema: it is sent to the model as input, so a large planning/analysis
    schema meaningfully raises the input-token estimate for a strict reservation.
    """
    total = len(system) if system else 0
    total += len(str(normalized))
    if wire_tools:
        total += len(str(wire_tools))
    if output_schema is not None:
        total += len(str(output_schema))
    return total


def _count_non_text_parts(normalized: list[dict[str, Any]]) -> int:
    """Number of non-text content parts (images/documents) across all messages."""
    count = 0
    for msg in normalized:
        content = msg.get("content")
        if isinstance(content, list):
            count += sum(
                1
                for part in content
                if isinstance(part, dict) and part.get("type") not in (None, "text")
            )
    return count


def _has_server_tools(wire_tools: list[dict[str, Any]] | None) -> bool:
    """Whether any provider-hosted server tool (web_search / code_execution) is present."""
    return bool(wire_tools) and any(
        isinstance(t, dict) and t.get("_server_tool") for t in wire_tools
    )


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
        if not (0.0 <= temperature <= 2.0):
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {temperature}")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError(f"max_tokens must be a positive integer, got {max_tokens}")
        if timeout is not None and timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")

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
        if retry is True:
            self._retry: RetryConfig | None = RetryConfig()
        elif retry is False:
            self._retry = None
        else:
            self._retry = retry
        self._middleware: list[Any] = list(middleware) if middleware else []
        self._fallback_on = fallback_on or PROVIDER_ERRORS
        self._fallbacks, self._owned_fallbacks = _normalize_fallbacks(
            fallback, api_key=api_key, base_url=base_url, provider=provider
        )

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
    # Internal: attempt tracking + fallback helpers
    # ------------------------------------------------------------------

    async def _try_with_tracking(
        self,
        call: Callable[[], Awaitable[Response]],
        model: str,
        retry: RetryConfig | None,
        attempts: list[Attempt],
        request: OperationRequest | None = None,
    ) -> Response:
        """Execute *call* with optional retry, recording + metering each physical attempt."""
        retry_number = 0

        async def _tracked() -> Response:
            nonlocal retry_number
            t0 = time.time()
            # One metered op per physical attempt, so retries count against max_llm_calls.
            # open() may raise AdmissionDenied — terminal by design: with_retry treats it as
            # non-retryable and it is not a PROVIDER_ERROR, so it never falls back.
            scope = current_meter()
            op: MeterOperation | None = (
                scope.open(request) if scope is not None and request is not None else None
            )
            if op is not None:
                op.mark_started()
            settled = False
            try:
                # Leaf-level inference-concurrency slot (no-op unless inference_limit is
                # active). Held only around the provider call — never across orchestration.
                async with inference_slot():
                    response = await call()
                attempts.append(
                    Attempt(
                        model=model,
                        status="ok",
                        usage=response.usage,
                        duration=time.time() - t0,
                        timestamp=t0,
                        retry_number=retry_number,
                    )
                )
                if op is not None and scope is not None and request is not None:
                    pricer = scope.pricer or pricing
                    cost = _settlement_cost(pricer, request, response)
                    op.settle(usage=response.usage, cost=cost)
                    settled = True
                return response
            except Exception as exc:
                attempts.append(
                    Attempt(
                        model=model,
                        status="failed",
                        error=str(exc),
                        error_type=type(exc).__name__,
                        status_code=getattr(exc, "status_code", None),
                        duration=time.time() - t0,
                        timestamp=t0,
                        retry_number=retry_number,
                    )
                )
                raise
            finally:
                # Fail on ANY non-settled exit (incl. cancellation / BaseException) so a started
                # op is never leaked until scope close; op.fail() is a no-op once settled.
                if op is not None and not settled:
                    op.fail()
                retry_number += 1

        if retry:
            return await with_retry(_tracked, retry)
        return await _tracked()

    def _meter_request(
        self,
        mode: Literal["complete", "stream"],
        provider_kwargs: dict[str, Any],
        *,
        normalized: list[dict[str, Any]],
        system: str | None,
        wire_tools: list[dict[str, Any]] | None,
    ) -> OperationRequest | None:
        """Build the metering facts for an LLM call, or ``None`` when no scope is bound."""
        scope = current_meter()
        if scope is None:
            return None
        # The content-size hint is consumed ONLY by a strict-reserve estimator. Computing it
        # stringifies the whole request (every message, every base64 image) — so skip it unless the
        # bound controller says it wants it. Measure-only and soft-budget runs (the common case)
        # never read it. A controller without a callable wants_request_size gets the hint (safe).
        controller = scope.controller
        if controller is None:
            wants_size = False
        else:
            wants = getattr(controller, "wants_request_size", None)
            wants_size = wants() if callable(wants) else True
        schema = provider_kwargs.get("output_schema")
        return OperationRequest(
            kind="llm",
            parent_span_id=current_span_id() or scope.run_span_id,
            mode=mode,
            model=self._model,
            declared_max_output_tokens=provider_kwargs.get("max_tokens"),
            content_size_hint=(
                _content_chars(normalized, system, wire_tools, schema) if wants_size else None
            ),
            non_text_parts=_count_non_text_parts(normalized) if wants_size else 0,
            has_server_tools=_has_server_tools(wire_tools),
        )

    def _open_stream_op(
        self,
        provider_kwargs: dict[str, Any],
        *,
        normalized: list[dict[str, Any]],
        system: str | None,
        wire_tools: list[dict[str, Any]] | None,
    ) -> tuple[MeterOperation, OperationRequest, Any] | None:
        """Open + start a metered op for one stream attempt; ``None`` when unmetered.

        Returns the handle, its request, and the pricer to settle with. ``AdmissionDenied`` from
        ``open`` propagates (terminal — the stream never starts). The handle carries the store
        reference, so the finalizer can settle from any thread without re-binding the scope.
        """
        request = self._meter_request(
            "stream", provider_kwargs, normalized=normalized, system=system, wire_tools=wire_tools
        )
        scope = current_meter()
        if scope is None or request is None:
            return None
        op = scope.open(request)
        op.mark_started()
        return op, request, scope.pricer or pricing

    async def _try_fallbacks(
        self,
        messages: str | list[dict[str, Any]],
        *,
        attempts: list[Attempt],
        last_error: Exception,
        **kwargs: Any,
    ) -> Response:
        """Walk fallback chain, delegating full complete() to each."""
        last_exc = last_error
        for i, fb in enumerate(self._fallbacks):
            logger.info("Fallback %d/%d: trying %s", i + 1, len(self._fallbacks), fb._model)
            try:
                response = await fb.complete(messages, **kwargs)
                # Merge fallback's tracked attempts into ours
                attempts.extend(response.attempts)
                return response
            except self._fallback_on as exc:
                if isinstance(exc, AdmissionDenied):
                    raise  # budget/admission denial is terminal — never masked by a later fallback
                last_exc = exc
        raise last_exc

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

        meter_request = self._meter_request(
            "complete",
            provider_kwargs,
            normalized=normalized,
            system=system,
            wire_tools=wire_tools,
        )
        attempts: list[Attempt] = []

        try:
            response = await self._try_with_tracking(
                _call, self._model, self._retry, attempts, meter_request
            )
        except self._fallback_on as primary_err:
            if isinstance(primary_err, AdmissionDenied):
                raise  # terminal: never fall back after a budget/admission denial
            if not self._fallbacks:
                raise
            response = await self._try_fallbacks(
                messages,
                attempts=attempts,
                last_error=primary_err,
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

        # Middleware after hooks
        if self._middleware and req is not None:
            response = await _run_aafter(self._middleware, req, response)

        return dataclasses.replace(response, attempts=tuple(attempts))

    def _build_stream_finalizer(self, state: StreamState) -> Callable[[str], Response]:
        """Build a finalization callback for stream wrappers."""
        model = self._model

        def _finalize(text: str) -> Response:
            usage = state.usage or Usage()
            provider_cost = state.provider_cost
            return Response(
                text=text,
                tool_calls=tuple(state.tool_calls),
                thinking=tuple(state.thinking),
                usage=usage,
                cost=(
                    provider_cost
                    if provider_cost is not None
                    else _estimate_response_cost(model, usage)
                ),
                provider_cost=provider_cost,
                stop_reason=state.stop_reason,
                model=state.model or model,
                raw=state.raw,
            )

        return _finalize

    def _prepare_stream_request(
        self,
        messages: str | list[dict[str, Any]],
        *,
        system: str | None,
        tools: list[dict[str, Any]] | ToolGroup | Callable[..., Any] | None,
        thinking: bool,
        thinking_effort: str | None,
        thinking_budget: int | None,
        output_schema: OutputSchema | type | None,
        tool_choice: str | None,
        json_mode: bool,
        logprobs: bool,
        kwargs: dict[str, Any],
    ) -> tuple[
        list[dict[str, Any]],
        str | None,
        list[dict[str, Any]] | None,
        dict[str, Any],
        Request | None,
    ]:
        normalized = self._normalize(messages)
        wire_tools = prepare_tools(tools)
        provider_kwargs = self._prepare_provider_kwargs(
            thinking=thinking,
            thinking_effort=thinking_effort,
            thinking_budget=thinking_budget,
            output_schema=output_schema,
            tool_choice=tool_choice,
            json_mode=json_mode,
            logprobs=logprobs,
            extra=self._merge_kwargs(**kwargs),
        )
        req: Request | None = None
        if self._middleware:
            req = _run_before(
                self._middleware,
                Request(
                    messages=normalized,
                    system=system,
                    tools=wire_tools,
                    model=self._model,
                    kwargs=provider_kwargs,
                ),
            )
            normalized = req.messages
            system = req.system
            wire_tools = req.tools
            provider_kwargs = req.kwargs
        return normalized, system, wire_tools, provider_kwargs, req

    def _single_stream(
        self,
        messages: str | list[dict[str, Any]],
        *,
        events: bool,
        system: str | None,
        tools: list[dict[str, Any]] | ToolGroup | Callable[..., Any] | None,
        thinking: bool,
        thinking_effort: str | None,
        thinking_budget: int | None,
        output_schema: OutputSchema | type | None,
        tool_choice: str | None,
        json_mode: bool,
        logprobs: bool,
        kwargs: dict[str, Any],
    ) -> StreamResponse | RichStreamResponse:
        """Build one provider stream with retries before its first observable item."""
        normalized, prepared_system, wire_tools, provider_kwargs, req = (
            self._prepare_stream_request(
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
                kwargs=kwargs,
            )
        )
        run = _StreamRun(self._model)
        # Preserve the established stream contract: admission and call-count reservation happen
        # when the stream object is created, even though provider I/O remains lazy until iteration.
        first_meter = self._open_stream_op(
            provider_kwargs,
            normalized=normalized,
            system=prepared_system,
            wire_tools=wire_tools,
        )
        run.begin(0, first_meter)

        async def _items() -> AsyncIterator[Any]:
            max_retries = self._retry.max_retries if self._retry is not None else 0
            for retry_number in range(max_retries + 1):
                if retry_number:
                    # Admission denials are deliberately outside the retry handler: they must
                    # never retry or fall back.
                    meter = self._open_stream_op(
                        provider_kwargs,
                        normalized=normalized,
                        system=prepared_system,
                        wire_tools=wire_tools,
                    )
                    run.begin(retry_number, meter)
                if not run.start_attempt():
                    return
                emitted = False
                iterator: AsyncIterator[Any] | None = None
                try:
                    # Streaming deliberately stays outside inference_limit: unlike complete(),
                    # its provider call remains live across caller-controlled yields. Holding the
                    # run semaphore there would let an abandoned stream starve unrelated calls.
                    if events:
                        iterator, state = self._provider.stream_events(
                            normalized,
                            system=prepared_system,
                            tools=wire_tools,
                            **provider_kwargs,
                        )
                    else:
                        iterator, state = self._provider.stream(
                            normalized,
                            system=prepared_system,
                            tools=wire_tools,
                            **provider_kwargs,
                        )
                    run.set_state(state)
                    try:
                        first = await iterator.__anext__()
                    except StopAsyncIteration:
                        run.succeed()
                        return
                    emitted = True
                    yield first
                    async for item in iterator:
                        yield item
                    run.succeed()
                    return
                except Exception as exc:
                    run.fail(exc)
                    # Once output is observable, replaying on another attempt would duplicate or
                    # splice content. Retry is therefore safe only before the first emitted item.
                    if (
                        emitted
                        or self._retry is None
                        or not await _wait_before_retry(exc, retry_number, self._retry)
                    ):
                        raise
                except BaseException as exc:
                    run.fail(exc)
                    raise
                finally:
                    close = getattr(iterator, "aclose", None)
                    if callable(close):
                        # Closing transport resources is best-effort and must never replace the
                        # provider outcome that selected retry/fallback behaviour.
                        with contextlib.suppress(Exception):
                            await cast(Callable[[], Awaitable[Any]], close)()

        def _finalize(text: str) -> Response:
            state, meter, attempts = run.finalization_snapshot()
            state = state or StreamState()
            response = self._build_stream_finalizer(state)(text)
            response = dataclasses.replace(response, attempts=attempts)
            if (
                meter is not None
                and attempts
                and attempts[-1].status == "ok"
                and not meter[0].abandoned
            ):
                op, request, pricer = meter
                op.settle(
                    usage=response.usage,
                    cost=_settlement_cost(pricer, request, response),
                )
            if self._middleware and req is not None:
                response = _run_after(self._middleware, req, response)
            return response

        finalizer = _StreamFinalizer(
            callback=_finalize,
            _stream_abandon=run.abandon,
            _stream_attempts=run.attempts,
            _stream_request=req,
            _stream_system=prepared_system,
        )
        if events:
            return RichStreamResponse(_items(), finalizer)
        return StreamResponse(_items(), finalizer)

    def _stream_with_fallbacks(
        self,
        messages: str | list[dict[str, Any]],
        *,
        events: bool,
        system: str | None,
        tools: list[dict[str, Any]] | ToolGroup | Callable[..., Any] | None,
        thinking: bool,
        thinking_effort: str | None,
        thinking_budget: int | None,
        output_schema: OutputSchema | type | None,
        tool_choice: str | None,
        json_mode: bool,
        logprobs: bool,
        kwargs: dict[str, Any],
        _fallback_ancestry: frozenset[int] = frozenset(),
    ) -> StreamResponse | RichStreamResponse:
        if id(self) in _fallback_ancestry:
            raise ValueError(f"stream fallback cycle detected at model {self._model!r}")
        fallback_ancestry = _fallback_ancestry | {id(self)}
        primary = self._single_stream(
            messages,
            events=events,
            system=system,
            tools=tools,
            thinking=thinking,
            thinking_effort=thinking_effort,
            thinking_budget=thinking_budget,
            output_schema=output_schema,
            tool_choice=tool_choice,
            json_mode=json_mode,
            logprobs=logprobs,
            kwargs=kwargs,
        )
        if not self._fallbacks:
            return primary

        fallback_run = _FallbackStreamRun(self, primary)
        parent_req = getattr(primary._finalizer, "_stream_request", None)
        fallback_system = getattr(primary._finalizer, "_stream_system", system)

        def _candidate_stream(
            owner: LLM,
        ) -> StreamResponse | RichStreamResponse:
            if owner is self:
                return primary
            # Preserve the candidate LLM's complete fallback policy, middleware nesting, and
            # retry configuration. Calling _single_stream() here silently truncated A -> B -> C.
            return owner._stream_with_fallbacks(
                messages,
                events=events,
                system=fallback_system,
                tools=tools,
                thinking=thinking,
                thinking_effort=thinking_effort,
                thinking_budget=thinking_budget,
                output_schema=output_schema,
                tool_choice=tool_choice,
                json_mode=json_mode,
                logprobs=logprobs,
                kwargs=kwargs,
                _fallback_ancestry=fallback_ancestry,
            )

        async def _items() -> AsyncIterator[Any]:
            last_exc: Exception | None = None
            owners = (self, *self._fallbacks)
            candidate: StreamResponse | RichStreamResponse | None = None
            try:
                for index, owner in enumerate(owners):
                    if index:
                        logger.info(
                            "Fallback %s %d/%d: trying %s",
                            "stream_events" if events else "stream",
                            index,
                            len(self._fallbacks),
                            owner._model,
                        )
                    candidate = _candidate_stream(owner)
                    if not fallback_run.activate(owner, candidate):
                        abandon = getattr(candidate._finalizer, "_stream_abandon", None)
                        if callable(abandon):
                            abandon()
                        return
                    try:
                        first = await candidate.__anext__()
                    except StopAsyncIteration:
                        return
                    except self._fallback_on as exc:
                        if isinstance(exc, AdmissionDenied):
                            raise
                        attempts = tuple(getattr(candidate._finalizer, "_stream_attempts", ()))
                        fallback_run.record_attempts(attempts)
                        last_exc = exc
                        continue
                    yield first
                    # A failure after this point is terminal: some output is already observable.
                    async for item in candidate:
                        yield item
                    return
                assert last_exc is not None
                raise last_exc
            finally:
                if candidate is not None:
                    # If the outer stream is abandoned while suspended at a yield, recursively
                    # close the active candidate and, ultimately, its provider iterator.
                    with contextlib.suppress(Exception):
                        await candidate._close_iterator()

        def _finalize(text: str) -> Response:
            owner, candidate, prior_attempts = fallback_run.snapshot()
            response = (
                candidate.response
                if candidate.response is not None
                else candidate._finalizer(text)
            )
            if owner is not self and self._middleware and parent_req is not None:
                response = _run_after(self._middleware, parent_req, response)
            response = dataclasses.replace(
                response,
                attempts=prior_attempts + response.attempts,
            )
            return response

        def _abandon() -> None:
            fallback_run.abandon()

        finalizer = _StreamFinalizer(
            callback=_finalize,
            _stream_abandon=_abandon,
            _stream_attempts=fallback_run.prior_attempts,
            _stream_request=parent_req,
            _stream_system=fallback_system,
        )
        if events:
            return RichStreamResponse(_items(), finalizer)
        return StreamResponse(_items(), finalizer)

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
        """Stream text; retry/fallback is safe only before the first emitted chunk."""
        stream = self._stream_with_fallbacks(
            messages,
            events=False,
            system=system,
            tools=tools,
            thinking=thinking,
            thinking_effort=thinking_effort,
            thinking_budget=thinking_budget,
            output_schema=output_schema,
            tool_choice=tool_choice,
            json_mode=json_mode,
            logprobs=logprobs,
            kwargs=kwargs,
        )
        assert isinstance(stream, StreamResponse)
        return stream

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
        """Stream events; retry/fallback is safe only before the first emitted event."""
        stream = self._stream_with_fallbacks(
            messages,
            events=True,
            system=system,
            tools=tools,
            thinking=thinking,
            thinking_effort=thinking_effort,
            thinking_budget=thinking_budget,
            output_schema=output_schema,
            tool_choice=tool_choice,
            json_mode=json_mode,
            logprobs=logprobs,
            kwargs=kwargs,
        )
        assert isinstance(stream, RichStreamResponse)
        return stream

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
        return SyncRichStreamResponse(sync_iter, rich_stream._finalizer)

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
        return SyncStreamResponse(sync_iter, async_stream._finalizer)
