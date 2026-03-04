from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._telemetry import TracingMiddleware


def _make_request(**overrides):
    defaults = dict(
        messages=[{"role": "user", "content": "hi"}],
        system=None,
        tools=None,
        model="claude-sonnet-4-20250514",
        kwargs={},
    )
    defaults.update(overrides)
    return Request(**defaults)


def _make_response(**overrides):
    defaults = dict(
        text="hello",
        usage=Usage(input_tokens=10, output_tokens=20),
        cost=0.001,
    )
    defaults.update(overrides)
    return Response(**defaults)


class TestTracingMiddlewareNoOtel:
    """TracingMiddleware is a no-op when OpenTelemetry is not installed."""

    @patch("ai_arch_toolkit.core._telemetry._HAS_OTEL", False)
    def test_noop_when_otel_not_installed(self):
        mw = TracingMiddleware()
        request = _make_request()

        result = mw.before(request)

        assert result is request
        assert "_otel_span" not in result.kwargs

        response = _make_response()
        result_resp = mw.after(request, response)
        assert result_resp is response


class TestTracingMiddlewareWithOtel:
    """TracingMiddleware creates spans when OpenTelemetry is available."""

    @patch("ai_arch_toolkit.core._telemetry._HAS_OTEL", True)
    @patch("ai_arch_toolkit.core._telemetry.trace", create=True)
    def test_span_created_when_otel_available(self, mock_trace):
        mock_tracer = MagicMock()
        mock_trace.get_tracer.return_value = mock_tracer

        mw = TracingMiddleware()
        request = _make_request()

        result = mw.before(request)

        mock_tracer.start_span.assert_called_once()
        assert "_otel_span" in result.kwargs
        assert "_otel_start" in result.kwargs

    def test_after_sets_attributes_and_ends_span(self):
        mock_span = MagicMock()
        request = _make_request(
            kwargs={"_otel_span": mock_span, "_otel_start": time.monotonic()},
        )
        response = _make_response()

        mw = TracingMiddleware.__new__(TracingMiddleware)
        mw._tracer = None  # not needed for after()

        result = mw.after(request, response)

        assert result is response
        mock_span.set_attribute.assert_any_call("llm.input_tokens", 10)
        mock_span.set_attribute.assert_any_call("llm.output_tokens", 20)
        mock_span.set_attribute.assert_any_call("llm.cost", 0.001)
        mock_span.end.assert_called_once()


class TestTracingMiddlewareAsync:
    """Async hooks delegate to sync implementations."""

    @patch("ai_arch_toolkit.core._telemetry._HAS_OTEL", False)
    async def test_abefore_delegates_to_before(self):
        mw = TracingMiddleware()
        request = _make_request()

        result = await mw.abefore(request)

        assert result is request
        assert "_otel_span" not in result.kwargs
