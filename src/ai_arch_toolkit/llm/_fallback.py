"""Fallback wrapper that tries multiple clients in order."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from ai_arch_toolkit.llm._exceptions import APIError, RateLimitError
from ai_arch_toolkit.llm._types import Message, Response, StreamEvent, ToolResult

logger = logging.getLogger(__name__)


class FallbackClient:
    """Try multiple clients in order until one succeeds."""

    def __init__(
        self,
        clients: Sequence[Any],
        *,
        fallback_on: tuple[type[BaseException], ...] = (
            RateLimitError,
            TimeoutError,
            APIError,
        ),
    ) -> None:
        if not clients:
            raise ValueError("FallbackClient requires at least one client")
        self._clients = list(clients)
        self._fallback_on = fallback_on

    def chat(
        self,
        prompt_or_messages: str | Sequence[dict[str, str] | Message | ToolResult],
        **kwargs: Any,
    ) -> Response:
        last_error: BaseException | None = None
        for index, client in enumerate(self._clients, start=1):
            try:
                return client.chat(prompt_or_messages, **kwargs)
            except self._fallback_on as exc:
                last_error = exc
                logger.warning(
                    "Client %s failed for chat; trying fallback: %s",
                    index,
                    type(exc).__name__,
                )
        assert last_error is not None
        raise last_error

    def stream(
        self,
        prompt_or_messages: str | Sequence[dict[str, str] | Message | ToolResult],
        **kwargs: Any,
    ) -> Iterator[str]:
        return self._stream_with_fallback(prompt_or_messages, stream_method="stream", **kwargs)

    def stream_events(
        self,
        prompt_or_messages: str | Sequence[dict[str, str] | Message | ToolResult],
        **kwargs: Any,
    ) -> Iterator[StreamEvent]:
        return self._stream_events_with_fallback(prompt_or_messages, **kwargs)

    async def achat(
        self,
        prompt_or_messages: str | Sequence[dict[str, str] | Message | ToolResult],
        **kwargs: Any,
    ) -> Response:
        last_error: BaseException | None = None
        for index, client in enumerate(self._clients, start=1):
            try:
                return await client.chat(prompt_or_messages, **kwargs)
            except self._fallback_on as exc:
                last_error = exc
                logger.warning(
                    "Client %s failed for async chat; trying fallback: %s",
                    index,
                    type(exc).__name__,
                )
        assert last_error is not None
        raise last_error

    async def astream(
        self,
        prompt_or_messages: str | Sequence[dict[str, str] | Message | ToolResult],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        async for chunk in self._astream_with_fallback(
            prompt_or_messages, stream_method="stream", **kwargs
        ):
            yield chunk

    async def astream_events(
        self,
        prompt_or_messages: str | Sequence[dict[str, str] | Message | ToolResult],
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        async for event in self._astream_events_with_fallback(prompt_or_messages, **kwargs):
            yield event

    def _stream_with_fallback(
        self,
        prompt_or_messages: str | Sequence[dict[str, str] | Message | ToolResult],
        *,
        stream_method: str,
        **kwargs: Any,
    ) -> Iterator[str]:
        last_error: BaseException | None = None
        for index, client in enumerate(self._clients, start=1):
            emitted = False
            try:
                stream = getattr(client, stream_method)(prompt_or_messages, **kwargs)
                for chunk in stream:
                    emitted = True
                    yield chunk
                return
            except self._fallback_on as exc:
                if emitted:
                    raise
                last_error = exc
                logger.warning(
                    "Client %s failed for %s; trying fallback: %s",
                    index,
                    stream_method,
                    type(exc).__name__,
                )
        assert last_error is not None
        raise last_error

    def _stream_events_with_fallback(
        self,
        prompt_or_messages: str | Sequence[dict[str, str] | Message | ToolResult],
        **kwargs: Any,
    ) -> Iterator[StreamEvent]:
        last_error: BaseException | None = None
        for index, client in enumerate(self._clients, start=1):
            emitted = False
            try:
                events = client.stream_events(prompt_or_messages, **kwargs)
                for event in events:
                    emitted = True
                    yield event
                return
            except self._fallback_on as exc:
                if emitted:
                    raise
                last_error = exc
                logger.warning(
                    "Client %s failed for stream_events; trying fallback: %s",
                    index,
                    type(exc).__name__,
                )
        assert last_error is not None
        raise last_error

    async def _astream_with_fallback(
        self,
        prompt_or_messages: str | Sequence[dict[str, str] | Message | ToolResult],
        *,
        stream_method: str,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        last_error: BaseException | None = None
        for index, client in enumerate(self._clients, start=1):
            emitted = False
            try:
                stream = getattr(client, stream_method)(prompt_or_messages, **kwargs)
                async for chunk in stream:
                    emitted = True
                    yield chunk
                return
            except self._fallback_on as exc:
                if emitted:
                    raise
                last_error = exc
                logger.warning(
                    "Client %s failed for async %s; trying fallback: %s",
                    index,
                    stream_method,
                    type(exc).__name__,
                )
        assert last_error is not None
        raise last_error

    async def _astream_events_with_fallback(
        self,
        prompt_or_messages: str | Sequence[dict[str, str] | Message | ToolResult],
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        last_error: BaseException | None = None
        for index, client in enumerate(self._clients, start=1):
            emitted = False
            try:
                events = self._get_async_stream_events(client, prompt_or_messages, **kwargs)
                async for event in events:
                    emitted = True
                    yield event
                return
            except self._fallback_on as exc:
                if emitted:
                    raise
                last_error = exc
                logger.warning(
                    "Client %s failed for async stream_events; trying fallback: %s",
                    index,
                    type(exc).__name__,
                )
        assert last_error is not None
        raise last_error

    @staticmethod
    def _get_async_stream_events(
        client: Any,
        prompt_or_messages: str | Sequence[dict[str, str] | Message | ToolResult],
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Support both async client styles: ``astream_events`` and ``stream_events``."""
        stream_fn = getattr(client, "astream_events", None)
        if stream_fn is None:
            stream_fn = getattr(client, "stream_events", None)
        if stream_fn is None:
            raise AttributeError(
                "Fallback client target must define astream_events or stream_events"
            )
        stream = stream_fn(prompt_or_messages, **kwargs)
        if not hasattr(stream, "__aiter__"):
            raise TypeError("Async stream_events method must return an async iterator")
        return stream
