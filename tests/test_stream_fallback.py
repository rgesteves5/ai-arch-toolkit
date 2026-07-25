"""Tests for stream fallback and stream middleware."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from ai_arch_toolkit.core._exceptions import APIError
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._providers._base import StreamState
from ai_arch_toolkit.core._response import Response, StreamEvent, Usage
from ai_arch_toolkit.core._retry import RetryConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(usage: Usage | None = None) -> StreamState:
    state = StreamState()
    state.usage = usage or Usage(input_tokens=10, output_tokens=5)
    state.model = "test-model"
    state.stop_reason = "end_turn"
    return state


# ---------------------------------------------------------------------------
# 1. Stream fallback triggers on APIError
# ---------------------------------------------------------------------------


async def test_stream_fallback_on_error():
    state = _make_state()

    async def _fake_stream() -> AsyncIterator[str]:
        yield "fallback text"

    with patch("ai_arch_toolkit.core._llm.create_provider") as mock_cp:
        primary = MagicMock()
        primary.stream.side_effect = APIError(500, "Server error")

        fallback = MagicMock()
        fallback.stream.return_value = (_fake_stream(), state)

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
        primary = MagicMock()
        primary.stream.side_effect = APIError(500, "Server error")
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
    state = _make_state()

    async def _fake_stream() -> AsyncIterator[str]:
        yield "modified"

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
        provider = MagicMock()
        provider.stream.return_value = (_fake_stream(), state)
        mock_cp.return_value = provider

        llm = LLM("test-model", middleware=[AddSystemMW()])
        stream = llm.stream("Hello")

        async for _ in stream:
            pass

    # Provider should have been called with the injected system
    call_kwargs = provider.stream.call_args
    assert call_kwargs.kwargs.get("system") == "injected system"


# ---------------------------------------------------------------------------
# 4. Stream middleware after fires on finalized response
# ---------------------------------------------------------------------------


async def test_stream_middleware_after():
    state = _make_state()

    async def _fake_stream() -> AsyncIterator[str]:
        yield "hello"

    after_called: list[bool] = []

    @dataclass
    class TrackAfterMW:
        def before(self, request: Request) -> Request:
            return request

        def after(self, request: Request, response: Response) -> Response:
            after_called.append(True)
            return response

    with patch("ai_arch_toolkit.core._llm.create_provider") as mock_cp:
        provider = MagicMock()
        provider.stream.return_value = (_fake_stream(), state)
        mock_cp.return_value = provider

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
    from ai_arch_toolkit.core._providers._base import StreamState
    from ai_arch_toolkit.core._response import StreamEvent

    state = StreamState()
    state.usage = Usage(input_tokens=5, output_tokens=3)
    state.model = "fallback-model"
    state.stop_reason = "end_turn"
    state.tool_calls = []
    state.thinking = []

    async def _fake_events():
        yield StreamEvent(kind="text", text="fallback")

    with patch("ai_arch_toolkit.core._llm.create_provider") as mock_cp:
        primary = MagicMock()
        primary.stream_events.side_effect = APIError(500, "Server error")

        fallback = MagicMock()
        fallback.stream_events.return_value = (_fake_events(), state)

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
        primary = MagicMock()
        primary.stream.side_effect = ValueError("Not an API error")

        fallback = MagicMock()
        mock_cp.side_effect = [primary, fallback]

        llm = LLM("test-model", fallback="fallback-model")
        stream = llm.stream("Hello")
        with pytest.raises(ValueError, match="Not an API error"):
            async for _ in stream:
                pass


# ---------------------------------------------------------------------------
# 7. Combined fallback + middleware
# ---------------------------------------------------------------------------


async def test_fallback_with_middleware():
    state = _make_state()

    async def _fake_stream():
        yield "from fallback"

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
        primary = MagicMock()
        primary.stream.side_effect = APIError(500, "Down")

        fallback = MagicMock()
        fallback.stream.return_value = (_fake_stream(), state)

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
    first_state = _make_state()
    second_state = _make_state(Usage(input_tokens=7, output_tokens=2))

    async def _fails_before_output():
        raise APIError(500, "temporary")
        yield  # pragma: no cover

    async def _succeeds():
        yield "recovered"

    async def _no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("ai_arch_toolkit.core._retry.asyncio.sleep", _no_sleep)
    with patch("ai_arch_toolkit.core._llm.create_provider") as mock_cp:
        provider = MagicMock()
        provider.stream.side_effect = [
            (_fails_before_output(), first_state),
            (_succeeds(), second_state),
        ]
        mock_cp.return_value = provider

        llm = LLM(
            "test-model",
            retry=RetryConfig(max_retries=1, base_delay=0.01),
        )
        stream = llm.stream("Hello")
        chunks = [chunk async for chunk in stream]

    assert chunks == ["recovered"]
    assert provider.stream.call_count == 2
    assert stream.response is not None
    assert [attempt.status for attempt in stream.response.attempts] == ["failed", "ok"]
    assert [attempt.retry_number for attempt in stream.response.attempts] == [0, 1]
    assert stream.response.attempts[1].usage == second_state.usage


async def test_stream_events_retries_lazy_error_before_first_event(monkeypatch):
    first_state = _make_state()
    second_state = _make_state()

    async def _fails_before_output():
        raise APIError(503, "temporary")
        yield  # pragma: no cover

    async def _succeeds():
        yield StreamEvent(kind="text", text="recovered")

    async def _no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("ai_arch_toolkit.core._retry.asyncio.sleep", _no_sleep)
    with patch("ai_arch_toolkit.core._llm.create_provider") as mock_cp:
        provider = MagicMock()
        provider.stream_events.side_effect = [
            (_fails_before_output(), first_state),
            (_succeeds(), second_state),
        ]
        mock_cp.return_value = provider

        llm = LLM(
            "test-model",
            retry=RetryConfig(max_retries=1, base_delay=0.01),
        )
        stream = llm.stream_events("Hello")
        events = [event async for event in stream]

    assert [event.text for event in events] == ["recovered"]
    assert provider.stream_events.call_count == 2
    assert stream.response is not None
    assert [attempt.status for attempt in stream.response.attempts] == ["failed", "ok"]


async def test_stream_falls_back_after_lazy_error_before_first_chunk():
    primary_state = _make_state()
    fallback_state = _make_state()

    async def _fails_before_output():
        raise APIError(500, "primary down")
        yield  # pragma: no cover

    async def _fallback_stream():
        yield "fallback"

    with patch("ai_arch_toolkit.core._llm.create_provider") as mock_cp:
        primary = MagicMock()
        primary.stream.return_value = (_fails_before_output(), primary_state)
        fallback = MagicMock()
        fallback.stream.return_value = (_fallback_stream(), fallback_state)
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
    async def _nested_success():
        yield "nested fallback"

    with patch("ai_arch_toolkit.core._llm.create_provider") as mock_cp:
        nested_provider = MagicMock()
        nested_provider.stream.return_value = (_nested_success(), _make_state())
        middle_provider = MagicMock()
        middle_provider.stream.side_effect = APIError(500, "middle down")
        primary_provider = MagicMock()
        primary_provider.stream.side_effect = APIError(500, "primary down")
        mock_cp.side_effect = [nested_provider, middle_provider, primary_provider]

        nested = LLM("nested-model")
        middle = LLM("middle-model", fallback=nested)
        primary = LLM("primary-model", fallback=middle)

        stream = primary.stream("Hello")
        chunks = [chunk async for chunk in stream]

    assert chunks == ["nested fallback"]
    assert nested_provider.stream.call_count == 1
    assert stream.response is not None
    assert [attempt.model for attempt in stream.response.attempts] == [
        "primary-model",
        "middle-model",
        "nested-model",
    ]


async def test_stream_does_not_retry_or_fallback_after_first_chunk(monkeypatch):
    state = _make_state()

    async def _partial_then_error():
        yield "partial"
        raise APIError(500, "failed after output")

    async def _no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("ai_arch_toolkit.core._retry.asyncio.sleep", _no_sleep)
    with patch("ai_arch_toolkit.core._llm.create_provider") as mock_cp:
        primary = MagicMock()
        primary.stream.return_value = (_partial_then_error(), state)
        fallback = MagicMock()
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
    assert primary.stream.call_count == 1
    assert fallback.stream.call_count == 0
