"""OpenTelemetry tracing middleware — no-op when otel is not installed."""

from __future__ import annotations

import time
from typing import Any

from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._response import Response

try:
    from opentelemetry import trace

    _HAS_OTEL = True
except ImportError:
    trace = None  # type: ignore[assignment]
    _HAS_OTEL = False


class TracingMiddleware:
    """Creates OpenTelemetry spans for LLM calls. No-op if otel not installed."""

    def __init__(self, tracer_name: str = "ai_arch_toolkit") -> None:
        if _HAS_OTEL and trace is not None:
            self._tracer: Any = trace.get_tracer(tracer_name)
        else:
            self._tracer = None

    def before(self, request: Request) -> Request:
        if not self._tracer:
            return request
        span = self._tracer.start_span(
            f"llm.{request.model}",
            attributes={
                "llm.model": request.model,
                "llm.message_count": len(request.messages),
            },
        )
        return Request(
            messages=request.messages,
            system=request.system,
            tools=request.tools,
            model=request.model,
            kwargs={**request.kwargs, "_otel_span": span, "_otel_start": time.monotonic()},
        )

    def after(self, request: Request, response: Response) -> Response:
        span = request.kwargs.get("_otel_span")
        if not span:
            return response
        start = request.kwargs.get("_otel_start", time.monotonic())
        span.set_attribute("llm.input_tokens", response.usage.input_tokens)
        span.set_attribute("llm.output_tokens", response.usage.output_tokens)
        span.set_attribute("llm.cost", response.cost or 0)
        span.set_attribute("llm.duration_s", time.monotonic() - start)
        span.end()
        return response

    async def abefore(self, request: Request) -> Request:
        return self.before(request)

    async def aafter(self, request: Request, response: Response) -> Response:
        return self.after(request, response)
