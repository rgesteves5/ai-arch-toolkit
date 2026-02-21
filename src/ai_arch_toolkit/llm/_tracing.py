"""OpenTelemetry-style tracing middleware (optional dependency)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any, Protocol, runtime_checkable

from ai_arch_toolkit.llm._middleware import Request
from ai_arch_toolkit.llm._types import Response, StreamEvent

_SPAN_CM_KEY = "tracing.span_cm"
_SPAN_KEY = "tracing.span"

try:
    from opentelemetry.trace import get_tracer
except Exception:  # pragma: no cover - optional dependency
    get_tracer = None


@runtime_checkable
class _SpanLike(Protocol):
    def set_attribute(self, key: str, value: Any) -> None: ...

    def record_exception(self, exception: BaseException) -> None: ...


@runtime_checkable
class _SpanContextManagerLike(Protocol):
    def __enter__(self) -> _SpanLike: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> bool | None: ...


@runtime_checkable
class _TracerLike(Protocol):
    def start_as_current_span(self, name: str) -> _SpanContextManagerLike: ...


class TracingMiddleware:
    """Middleware that emits tracing spans for LLM operations."""

    def __init__(
        self,
        tracer: _TracerLike | None = None,
        *,
        tracer_name: str = "ai_arch_toolkit.llm",
    ) -> None:
        if tracer is not None:
            self._tracer = tracer
        elif get_tracer is not None:
            self._tracer = get_tracer(tracer_name)
        else:
            self._tracer = None

    def before(self, request: Request) -> Request:
        if self._tracer is None:
            return request
        span_cm = self._tracer.start_as_current_span(f"llm.{request.operation}")
        span = span_cm.__enter__()
        request.context[_SPAN_CM_KEY] = span_cm
        request.context[_SPAN_KEY] = span
        self._set_request_attributes(request, span)
        return request

    def after(self, request: Request, result: Any) -> Any:
        span = request.context.get(_SPAN_KEY)
        if span is None:
            return result
        if isinstance(result, Response):
            self._set_response_attributes(span, result)
            self._end_span(request, None)
            return result
        if request.operation == "stream" and isinstance(result, Iterator):
            return self._wrap_stream(request, result)
        if request.operation == "stream_events" and isinstance(result, Iterator):
            return self._wrap_stream_events(request, result)
        self._end_span(request, None)
        return result

    async def abefore(self, request: Request) -> Request:
        return self.before(request)

    async def aafter(self, request: Request, result: Any) -> Any:
        span = request.context.get(_SPAN_KEY)
        if span is None:
            return result
        if isinstance(result, Response):
            self._set_response_attributes(span, result)
            self._end_span(request, None)
            return result
        if request.operation == "stream" and isinstance(result, AsyncIterator):
            return self._awrap_stream(request, result)
        if request.operation == "stream_events" and isinstance(result, AsyncIterator):
            return self._awrap_stream_events(request, result)
        self._end_span(request, None)
        return result

    def _set_request_attributes(self, request: Request, span: _SpanLike) -> None:
        span.set_attribute("gen_ai.operation.name", request.operation)
        span.set_attribute("gen_ai.system", request.provider)
        span.set_attribute("gen_ai.request.model", request.model)
        span.set_attribute("gen_ai.request.message_count", len(request.messages))
        span.set_attribute("gen_ai.request.tool_count", len(request.tools or ()))

    def _set_response_attributes(self, span: _SpanLike, response: Response) -> None:
        span.set_attribute("gen_ai.response.finish_reason", response.stop_reason)
        span.set_attribute("gen_ai.usage.input_tokens", response.usage.input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", response.usage.output_tokens)
        span.set_attribute("gen_ai.usage.total_tokens", response.usage.total_tokens)

    def _wrap_stream(self, request: Request, stream: Iterator[str]) -> Iterator[str]:
        try:
            yield from stream
            self._end_span(request, None)
        except Exception as exc:  # pragma: no cover - exercised in caller tests
            self._end_span(request, exc)
            raise

    def _wrap_stream_events(
        self, request: Request, stream: Iterator[StreamEvent]
    ) -> Iterator[StreamEvent]:
        try:
            span = request.context.get(_SPAN_KEY)
            for event in stream:
                if span is not None and event.type == "usage" and event.usage is not None:
                    span.set_attribute("gen_ai.usage.input_tokens", event.usage.input_tokens)
                    span.set_attribute("gen_ai.usage.output_tokens", event.usage.output_tokens)
                    span.set_attribute("gen_ai.usage.total_tokens", event.usage.total_tokens)
                yield event
            self._end_span(request, None)
        except Exception as exc:  # pragma: no cover - exercised in caller tests
            self._end_span(request, exc)
            raise

    async def _awrap_stream(
        self, request: Request, stream: AsyncIterator[str]
    ) -> AsyncIterator[str]:
        try:
            async for chunk in stream:
                yield chunk
            self._end_span(request, None)
        except Exception as exc:  # pragma: no cover - exercised in caller tests
            self._end_span(request, exc)
            raise

    async def _awrap_stream_events(
        self, request: Request, stream: AsyncIterator[StreamEvent]
    ) -> AsyncIterator[StreamEvent]:
        try:
            span = request.context.get(_SPAN_KEY)
            async for event in stream:
                if span is not None and event.type == "usage" and event.usage is not None:
                    span.set_attribute("gen_ai.usage.input_tokens", event.usage.input_tokens)
                    span.set_attribute("gen_ai.usage.output_tokens", event.usage.output_tokens)
                    span.set_attribute("gen_ai.usage.total_tokens", event.usage.total_tokens)
                yield event
            self._end_span(request, None)
        except Exception as exc:  # pragma: no cover - exercised in caller tests
            self._end_span(request, exc)
            raise

    def _end_span(self, request: Request, exc: BaseException | None) -> None:
        span_cm = request.context.pop(_SPAN_CM_KEY, None)
        span = request.context.pop(_SPAN_KEY, None)
        if span_cm is None:
            return
        if exc is not None and span is not None and hasattr(span, "record_exception"):
            span.record_exception(exc)
        span_cm.__exit__(type(exc) if exc is not None else None, exc, None)
