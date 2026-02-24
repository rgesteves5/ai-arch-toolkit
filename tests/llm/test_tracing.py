"""Tests for tracing middleware."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_arch_toolkit._legacy.llm import AsyncClient, Client
from ai_arch_toolkit._legacy.llm._middleware import Request
from ai_arch_toolkit._legacy.llm._tracing import TracingMiddleware
from ai_arch_toolkit._legacy.llm._types import Message, Response, StreamEvent, Usage


class _FakeSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}
        self.exceptions: list[BaseException] = []

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def record_exception(self, exception: BaseException) -> None:
        self.exceptions.append(exception)


class _FakeSpanCM:
    def __init__(self, span: _FakeSpan) -> None:
        self.span = span
        self.ended = False

    def __enter__(self) -> _FakeSpan:
        return self.span

    def __exit__(self, exc_type, exc, tb) -> None:
        self.ended = True


class _FakeTracer:
    def __init__(self) -> None:
        self.spans: list[_FakeSpan] = []
        self.cms: list[_FakeSpanCM] = []
        self.span_names: list[str] = []

    def start_as_current_span(self, name: str) -> _FakeSpanCM:
        span = _FakeSpan()
        cm = _FakeSpanCM(span)
        self.spans.append(span)
        self.cms.append(cm)
        self.span_names.append(name)
        return cm


@patch("ai_arch_toolkit._legacy.llm._client.create_provider")
def test_tracing_chat_records_attributes_and_ends_span(mock_create: MagicMock) -> None:
    provider = MagicMock()
    provider.complete.return_value = Response(
        text="ok",
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        stop_reason="stop",
    )
    mock_create.return_value = provider
    tracer = _FakeTracer()
    middleware = TracingMiddleware(tracer)
    client = Client("openai", model="gpt-4o", api_key="sk-test", middleware=[middleware])

    _ = client.chat("hello")

    assert tracer.span_names == ["llm.chat"]
    span = tracer.spans[0]
    assert span.attributes["gen_ai.operation.name"] == "chat"
    assert span.attributes["gen_ai.request.model"] == "gpt-4o"
    assert span.attributes["gen_ai.usage.total_tokens"] == 15
    assert tracer.cms[0].ended is True


@patch("ai_arch_toolkit._legacy.llm._client.create_provider")
def test_tracing_stream_events_records_usage(mock_create: MagicMock) -> None:
    provider = MagicMock()
    provider.stream_events.return_value = iter(
        [
            StreamEvent(type="text", text="hi"),
            StreamEvent(
                type="usage",
                usage=Usage(input_tokens=3, output_tokens=4, total_tokens=7),
            ),
            StreamEvent(type="done"),
        ]
    )
    mock_create.return_value = provider
    tracer = _FakeTracer()
    middleware = TracingMiddleware(tracer)
    client = Client("openai", model="gpt-4o", api_key="sk-test", middleware=[middleware])

    _ = list(client.stream_events("hello"))

    span = tracer.spans[0]
    assert span.attributes["gen_ai.usage.total_tokens"] == 7
    assert tracer.cms[0].ended is True


@patch("ai_arch_toolkit._legacy.llm._async_client.create_provider")
@pytest.mark.asyncio
async def test_tracing_async_chat_records_attributes(mock_create: MagicMock) -> None:
    provider = MagicMock()
    provider.acomplete = AsyncMock(
        return_value=Response(
            text="ok",
            usage=Usage(input_tokens=8, output_tokens=2, total_tokens=10),
            stop_reason="done",
        )
    )
    mock_create.return_value = provider
    tracer = _FakeTracer()
    middleware = TracingMiddleware(tracer)
    client = AsyncClient("openai", model="gpt-4o-mini", api_key="sk-test", middleware=[middleware])

    _ = await client.chat("hello")

    assert tracer.span_names == ["llm.chat"]
    span = tracer.spans[0]
    assert span.attributes["gen_ai.request.model"] == "gpt-4o-mini"
    assert span.attributes["gen_ai.usage.total_tokens"] == 10
    assert tracer.cms[0].ended is True


def test_tracing_end_span_clears_context_keys() -> None:
    tracer = _FakeTracer()
    middleware = TracingMiddleware(tracer)
    request = Request(
        operation="chat",
        provider="openai",
        model="gpt-4o",
        messages=[Message(role="user", content="hello")],
    )

    request = middleware.before(request)
    assert "tracing.span_cm" in request.context
    assert "tracing.span" in request.context

    _ = middleware.after(request, Response(text="ok"))
    assert "tracing.span_cm" not in request.context
    assert "tracing.span" not in request.context
