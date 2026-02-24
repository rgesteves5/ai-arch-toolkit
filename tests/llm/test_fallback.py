"""Tests for fallback client behavior."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import pytest

from ai_arch_toolkit._legacy.llm._exceptions import APIError
from ai_arch_toolkit._legacy.llm._fallback import FallbackClient
from ai_arch_toolkit._legacy.llm._types import Response, StreamEvent


class _FailingSyncClient:
    def chat(self, *_args, **_kwargs):
        raise APIError(500, {"error": "boom"})

    def stream(self, *_args, **_kwargs) -> Iterator[str]:
        raise APIError(500, {"error": "boom"})
        yield

    def stream_events(self, *_args, **_kwargs) -> Iterator[StreamEvent]:
        raise APIError(500, {"error": "boom"})
        yield


class _SuccessSyncClient:
    def chat(self, *_args, **_kwargs) -> Response:
        return Response(text="ok")

    def stream(self, *_args, **_kwargs) -> Iterator[str]:
        yield "a"
        yield "b"

    def stream_events(self, *_args, **_kwargs) -> Iterator[StreamEvent]:
        yield StreamEvent(type="text", text="hi")
        yield StreamEvent(type="done")


class _PartialFailSyncClient:
    def chat(self, *_args, **_kwargs) -> Response:
        return Response(text="x")

    def stream(self, *_args, **_kwargs) -> Iterator[str]:
        yield "first"
        raise APIError(500, {"error": "after-first"})

    def stream_events(self, *_args, **_kwargs) -> Iterator[StreamEvent]:
        yield StreamEvent(type="text", text="first")
        raise APIError(500, {"error": "after-first"})


class _FailingAsyncClient:
    async def chat(self, *_args, **_kwargs):
        raise APIError(500, {"error": "boom"})

    async def stream(self, *_args, **_kwargs) -> AsyncIterator[str]:
        raise APIError(500, {"error": "boom"})
        yield

    async def stream_events(self, *_args, **_kwargs) -> AsyncIterator[StreamEvent]:
        raise APIError(500, {"error": "boom"})
        yield


class _SuccessAsyncClient:
    async def chat(self, *_args, **_kwargs) -> Response:
        return Response(text="ok-async")

    async def stream(self, *_args, **_kwargs) -> AsyncIterator[str]:
        yield "x"
        yield "y"

    async def stream_events(self, *_args, **_kwargs) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(type="text", text="async")
        yield StreamEvent(type="done")


class _AStreamEventsOnlyAsyncClient:
    async def astream_events(self, *_args, **_kwargs) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(type="text", text="async-astream-events")
        yield StreamEvent(type="done")


def test_fallback_chat_uses_second_client() -> None:
    client = FallbackClient([_FailingSyncClient(), _SuccessSyncClient()])
    response = client.chat("hello")
    assert response.text == "ok"


def test_fallback_stream_uses_second_client() -> None:
    client = FallbackClient([_FailingSyncClient(), _SuccessSyncClient()])
    chunks = list(client.stream("hello"))
    assert chunks == ["a", "b"]


def test_fallback_stream_events_uses_second_client() -> None:
    client = FallbackClient([_FailingSyncClient(), _SuccessSyncClient()])
    events = list(client.stream_events("hello"))
    assert [e.type for e in events] == ["text", "done"]


def test_fallback_does_not_retry_after_partial_stream_output() -> None:
    client = FallbackClient([_PartialFailSyncClient(), _SuccessSyncClient()])
    iterator = client.stream("hello")
    assert next(iterator) == "first"
    with pytest.raises(APIError):
        next(iterator)


@pytest.mark.asyncio
async def test_async_fallback_chat_uses_second_client() -> None:
    client = FallbackClient([_FailingAsyncClient(), _SuccessAsyncClient()])
    response = await client.achat("hello")
    assert response.text == "ok-async"


@pytest.mark.asyncio
async def test_async_fallback_stream_uses_second_client() -> None:
    client = FallbackClient([_FailingAsyncClient(), _SuccessAsyncClient()])
    chunks = []
    async for chunk in client.astream("hello"):
        chunks.append(chunk)
    assert chunks == ["x", "y"]


@pytest.mark.asyncio
async def test_async_fallback_stream_events_uses_second_client() -> None:
    client = FallbackClient([_FailingAsyncClient(), _SuccessAsyncClient()])
    events = []
    async for event in client.astream_events("hello"):
        events.append(event)
    assert [e.type for e in events] == ["text", "done"]


@pytest.mark.asyncio
async def test_async_fallback_stream_events_supports_astream_events_method() -> None:
    client = FallbackClient([_FailingAsyncClient(), _AStreamEventsOnlyAsyncClient()])
    events = []
    async for event in client.astream_events("hello"):
        events.append(event)
    assert [e.type for e in events] == ["text", "done"]
    assert events[0].text == "async-astream-events"


def test_fallback_requires_at_least_one_client() -> None:
    with pytest.raises(ValueError, match="at least one"):
        FallbackClient([])
