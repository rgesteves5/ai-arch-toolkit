"""Tests for stream fallback and stream middleware."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import pytest

from ai_arch_toolkit.core._exceptions import APIError
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._retry import RetryConfig
from tests.fake_provider import FakeProvider, Reply

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Each test builds its LLMs through a patched ``create_provider``; every fake answers for the
# model of the LLM it is handed to. A scripted exception fails a stream before its first chunk.


def _answer(text: str, usage: Usage | None = None) -> Response:
    return Response(text=text, usage=usage or Usage(input_tokens=10, output_tokens=5))


# ---------------------------------------------------------------------------
# 1. Stream fallback triggers on APIError
# ---------------------------------------------------------------------------


async def test_stream_fallback_on_error():
    with patch("ai_arch_toolkit.core._llm.create_provider") as mock_cp:
        primary = FakeProvider(APIError(500, "Server error"), model="test-model")
        fallback = FakeProvider(_answer("fallback text"), model="fallback-model")

        mock_cp.side_effect = [primary, fallback]

        llm = LLM("test-model", fallback="fallback-model")
        stream = llm.stream("Hello")

        chunks: list[str] = []
        async for chunk in stream:
            chunks.append(chunk)

    assert chunks == ["fallback text"]
    assert stream.response is not None


# ---------------------------------------------------------------------------
# 2. Stream fallback raises if no fallback configured
# ---------------------------------------------------------------------------


async def test_stream_raises_without_fallback():
    with patch("ai_arch_toolkit.core._llm.create_provider") as mock_cp:
        primary = FakeProvider(APIError(500, "Server error"), model="test-model")
        mock_cp.return_value = primary

        llm = LLM("test-model")
        stream = llm.stream("Hello")
        with pytest.raises(APIError):
            async for _ in stream:
                pass


# ---------------------------------------------------------------------------
# 3. Stream middleware before modifies request
# ---------------------------------------------------------------------------


async def test_stream_middleware_before():
    @dataclass
    class AddSystemMW:
        def before(self, request: Request) -> Request:
            return Request(
                messages=request.messages,
                system="injected system",
                tools=request.tools,
                model=request.model,
                kwargs=request.kwargs,
            )

        def after(self, request: Request, response: Response) -> Response:
            return response

    with patch("ai_arch_toolkit.core._llm.create_provider") as mock_cp:
        provider = FakeProvider(_answer("modified"), model="test-model")
        mock_cp.return_value = provider

        llm = LLM("test-model", middleware=[AddSystemMW()])
        stream = llm.stream("Hello")

        async for _ in stream:
            pass

    # Provider should have been called with the injected system
    assert provider.last.system == "injected system"


# ---------------------------------------------------------------------------
# 4. Stream middleware after fires on finalized response
# ---------------------------------------------------------------------------


async def test_stream_middleware_after():
    after_called: list[bool] = []

    @dataclass
    class TrackAfterMW:
        def before(self, request: Request) -> Request:
            return request

        def after(self, request: Request, response: Response) -> Response:
            after_called.append(True)
            return response

    with patch("ai_arch_toolkit.core._llm.create_provider") as mock_cp:
        mock_cp.return_value = FakeProvider(_answer("hello"), model="test-model")

        llm = LLM("test-model", middleware=[TrackAfterMW()])
        stream = llm.stream("Hello")

        async for _ in stream:
            pass

    # After hook fires after stream is consumed (finalization)
    assert stream.response is not None
    assert len(after_called) == 1


# ---------------------------------------------------------------------------
# 5. stream_events fallback on APIError
# ---------------------------------------------------------------------------


async def test_stream_events_fallback_on_error():
    with patch("ai_arch_toolkit.core._llm.create_provider") as mock_cp:
        primary = FakeProvider(APIError(500, "Server error"), model="test-model")
        usage = Usage(input_tokens=5, output_tokens=3)
        fallback = FakeProvider(_answer("fallback", usage), model="fallback-model")

        mock_cp.side_effect = [primary, fallback]

        llm = LLM("test-model", fallback="fallback-model")
        stream = llm.stream_events("Hello")

        collected = []
        async for event in stream:
            collected.append(event)

    assert len(collected) == 1
    assert collected[0].text == "fallback"
    assert stream.response is not None


# ---------------------------------------------------------------------------
# 6. Non-APIError does NOT trigger fallback
# ---------------------------------------------------------------------------


async def test_non_api_error_does_not_fallback():
    with patch("ai_arch_toolkit.core._llm.create_provider") as mock_cp:
        primary = FakeProvider(ValueError("Not an API error"), model="test-model")
        fallback = FakeProvider(_answer("fallback"), model="fallback-model")
        mock_cp.side_effect = [primary, fallback]

        llm = LLM("test-model", fallback="fallback-model")
        stream = llm.stream("Hello")
        with pytest.raises(ValueError, match="Not an API error"):
            async for _ in stream:
                pass

    assert fallback.calls == 0


# ---------------------------------------------------------------------------
# 7. Combined fallback + middleware
# ---------------------------------------------------------------------------


async def test_fallback_with_middleware():
    before_calls: list[bool] = []
    after_calls: list[bool] = []

    @dataclass
    class TrackMW:
        def before(self, request: Request) -> Request:
            before_calls.append(True)
            return request

        def after(self, request: Request, response: Response) -> Response:
            after_calls.append(True)
            return response

    with patch("ai_arch_toolkit.core._llm.create_provider") as mock_cp:
        primary = FakeProvider(APIError(500, "Down"), model="test-model")
        fallback = FakeProvider(_answer("from fallback"), model="fallback-model")

        mock_cp.side_effect = [primary, fallback]

        llm = LLM("test-model", fallback="fallback-model", middleware=[TrackMW()])
        stream = llm.stream("Hello")

        async for _ in stream:
            pass

    # Middleware before should fire (once)
    assert len(before_calls) == 1
    # Middleware after fires on finalized response
    assert stream.response is not None
    assert len(after_calls) == 1


async def test_stream_retries_lazy_error_before_first_chunk(monkeypatch):
    second_usage = Usage(input_tokens=7, output_tokens=2)

    async def _no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("ai_arch_toolkit.core._retry.asyncio.sleep", _no_sleep)
    with patch("ai_arch_toolkit.core._llm.create_provider") as mock_cp:
        provider = FakeProvider(
            APIError(500, "temporary"),  # fails before any output
            _answer("recovered", second_usage),
            model="test-model",
        )
        mock_cp.return_value = provider

        llm = LLM(
            "test-model",
            retry=RetryConfig(max_retries=1, base_delay=0.01),
        )
        stream = llm.stream("Hello")
        chunks = [chunk async for chunk in stream]

    assert chunks == ["recovered"]
    assert provider.calls == 2
    assert stream.response is not None
    assert [attempt.status for attempt in stream.response.attempts] == ["failed", "ok"]
    assert [attempt.retry_number for attempt in stream.response.attempts] == [0, 1]
    assert stream.response.attempts[1].usage == second_usage


async def test_stream_events_retries_lazy_error_before_first_event(monkeypatch):
    async def _no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("ai_arch_toolkit.core._retry.asyncio.sleep", _no_sleep)
    with patch("ai_arch_toolkit.core._llm.create_provider") as mock_cp:
        provider = FakeProvider(
            APIError(503, "temporary"),  # fails before any output
            _answer("recovered"),
            model="test-model",
        )
        mock_cp.return_value = provider

        llm = LLM(
            "test-model",
            retry=RetryConfig(max_retries=1, base_delay=0.01),
        )
        stream = llm.stream_events("Hello")
        events = [event async for event in stream]

    assert [event.text for event in events] == ["recovered"]
    assert provider.calls == 2
    assert stream.response is not None
    assert [attempt.status for attempt in stream.response.attempts] == ["failed", "ok"]


async def test_stream_falls_back_after_lazy_error_before_first_chunk():
    with patch("ai_arch_toolkit.core._llm.create_provider") as mock_cp:
        primary = FakeProvider(APIError(500, "primary down"), model="test-model")
        fallback = FakeProvider(_answer("fallback"), model="fallback-model")
        mock_cp.side_effect = [primary, fallback]

        llm = LLM("test-model", fallback="fallback-model")
        stream = llm.stream("Hello")
        chunks = [chunk async for chunk in stream]

    assert chunks == ["fallback"]
    assert stream.response is not None
    assert [attempt.status for attempt in stream.response.attempts] == ["failed", "ok"]
    assert [attempt.model for attempt in stream.response.attempts] == [
        "test-model",
        "fallback-model",
    ]


async def test_stream_preserves_nested_llm_fallback_chains():
    with patch("ai_arch_toolkit.core._llm.create_provider") as mock_cp:
        nested_provider = FakeProvider(_answer("nested fallback"), model="nested-model")
        middle_provider = FakeProvider(APIError(500, "middle down"), model="middle-model")
        primary_provider = FakeProvider(APIError(500, "primary down"), model="primary-model")
        mock_cp.side_effect = [nested_provider, middle_provider, primary_provider]

        nested = LLM("nested-model")
        middle = LLM("middle-model", fallback=nested)
        primary = LLM("primary-model", fallback=middle)

        stream = primary.stream("Hello")
        chunks = [chunk async for chunk in stream]

    assert chunks == ["nested fallback"]
    assert nested_provider.calls == 1
    assert stream.response is not None
    assert [attempt.model for attempt in stream.response.attempts] == [
        "primary-model",
        "middle-model",
        "nested-model",
    ]


async def test_stream_does_not_retry_or_fallback_after_first_chunk(monkeypatch):
    async def _no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("ai_arch_toolkit.core._retry.asyncio.sleep", _no_sleep)
    with patch("ai_arch_toolkit.core._llm.create_provider") as mock_cp:
        partial_then_error = Reply(
            chunks=["partial"], error=APIError(500, "failed after output"), error_after=1
        )
        primary = FakeProvider(partial_then_error, model="test-model")
        fallback = FakeProvider(_answer("fallback"), model="fallback-model")
        mock_cp.side_effect = [primary, fallback]

        llm = LLM(
            "test-model",
            retry=RetryConfig(max_retries=2, base_delay=0.01),
            fallback="fallback-model",
        )
        stream = llm.stream("Hello")
        chunks: list[str] = []
        with pytest.raises(APIError, match="failed after output"):
            async for chunk in stream:
                chunks.append(chunk)

    assert chunks == ["partial"]
    assert primary.calls == 1
    assert fallback.calls == 0
