"""Tests for the AsyncClient."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_arch_toolkit.llm._async_client import AsyncClient
from ai_arch_toolkit.llm._middleware import Request
from ai_arch_toolkit.llm._types import Message, Response, StreamEvent, ToolCall, Usage


@pytest.fixture
def mock_provider():
    """Create a mock provider with async methods."""
    provider = MagicMock()
    provider.acomplete = AsyncMock(return_value=Response(text="Hello!", usage=Usage(10, 5, 15)))

    async def _astream(*args, **kwargs):
        for chunk in ["Hello", " world"]:
            yield chunk

    provider.astream = MagicMock(side_effect=_astream)

    async def _astream_events(*args, **kwargs):
        yield StreamEvent(type="text", text="Hi")
        yield StreamEvent(
            type="tool_call",
            tool_call=ToolCall(id="c1", name="f", arguments={"a": 1}),
        )
        yield StreamEvent(type="usage", usage=Usage(10, 5, 15))
        yield StreamEvent(type="done")

    provider.astream_events = MagicMock(side_effect=_astream_events)
    return provider


@patch("ai_arch_toolkit.llm._async_client.create_provider")
async def test_async_chat_string(mock_create: MagicMock, mock_provider: MagicMock) -> None:
    mock_create.return_value = mock_provider
    client = AsyncClient("openai", model="gpt-4o", api_key="sk-test")
    resp = await client.chat("Hello!")

    assert resp.text == "Hello!"
    mock_provider.acomplete.assert_called_once()
    call_args = mock_provider.acomplete.call_args
    messages = call_args[0][0]
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "Hello!"


@patch("ai_arch_toolkit.llm._async_client.create_provider")
async def test_async_chat_messages(mock_create: MagicMock, mock_provider: MagicMock) -> None:
    mock_create.return_value = mock_provider
    client = AsyncClient("openai", model="gpt-4o", api_key="sk-test")
    resp = await client.chat([Message("user", "Hi"), Message("assistant", "Hello")])

    assert resp.text == "Hello!"
    call_args = mock_provider.acomplete.call_args
    messages = call_args[0][0]
    assert len(messages) == 2


@patch("ai_arch_toolkit.llm._async_client.create_provider")
async def test_async_stream(mock_create: MagicMock, mock_provider: MagicMock) -> None:
    mock_create.return_value = mock_provider
    client = AsyncClient("openai", model="gpt-4o", api_key="sk-test")

    chunks = []
    async for chunk in client.stream("Tell me a story"):
        chunks.append(chunk)
    assert chunks == ["Hello", " world"]


@patch("ai_arch_toolkit.llm._async_client.create_provider")
async def test_async_chat_with_system(mock_create: MagicMock, mock_provider: MagicMock) -> None:
    mock_create.return_value = mock_provider
    client = AsyncClient("openai", model="gpt-4o", api_key="sk-test")
    await client.chat("Hi", system="Be helpful")

    call_kwargs = mock_provider.acomplete.call_args[1]
    assert call_kwargs["system"] == "Be helpful"


@patch("ai_arch_toolkit.llm._async_client.create_provider")
async def test_async_stream_events(mock_create: MagicMock, mock_provider: MagicMock) -> None:
    mock_create.return_value = mock_provider
    client = AsyncClient("openai", model="gpt-4o", api_key="sk-test")

    events = []
    async for event in client.stream_events("Hello"):
        events.append(event)

    assert len(events) == 4
    assert events[0].type == "text"
    assert events[0].text == "Hi"
    assert events[1].type == "tool_call"
    assert events[1].tool_call.name == "f"
    assert events[2].type == "usage"
    assert events[2].usage.input_tokens == 10
    assert events[3].type == "done"


@patch("ai_arch_toolkit.llm._async_client.create_provider")
async def test_async_chat_forwards_timeout(
    mock_create: MagicMock, mock_provider: MagicMock
) -> None:
    mock_create.return_value = mock_provider
    client = AsyncClient("openai", model="gpt-4o", api_key="sk-test")

    await client.chat("Hello!", timeout=17)

    call_kwargs = mock_provider.acomplete.call_args[1]
    assert call_kwargs["timeout"] == 17


@patch("ai_arch_toolkit.llm._async_client.create_provider")
async def test_async_stream_forwards_timeout(
    mock_create: MagicMock, mock_provider: MagicMock
) -> None:
    mock_create.return_value = mock_provider
    client = AsyncClient("openai", model="gpt-4o", api_key="sk-test")

    chunks = []
    async for chunk in client.stream("Hello!", timeout=21):
        chunks.append(chunk)

    assert chunks == ["Hello", " world"]
    call_kwargs = mock_provider.astream.call_args[1]
    assert call_kwargs["timeout"] == 21


class _AsyncRecorderMiddleware:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def abefore(self, request: Request) -> Request:
        self._events.append(f"before:{request.operation}")
        request.kwargs["middleware_marker"] = "async"
        return request

    async def aafter(self, request: Request, result: Any) -> Any:
        self._events.append(f"after:{request.operation}")
        if request.operation == "chat":
            return Response(text=f"{result.text}!", usage=result.usage)
        if request.operation == "stream":
            return _aupper_stream(result)
        return result


async def _aupper_stream(stream: AsyncIterator[str]) -> AsyncIterator[str]:
    async for chunk in stream:
        yield chunk.upper()


@patch("ai_arch_toolkit.llm._async_client.create_provider")
async def test_async_chat_runs_middleware_before_and_after(
    mock_create: MagicMock, mock_provider: MagicMock
) -> None:
    mock_create.return_value = mock_provider
    events: list[str] = []
    middleware = _AsyncRecorderMiddleware(events)
    client = AsyncClient("openai", model="gpt-4o", api_key="sk-test", middleware=[middleware])

    resp = await client.chat("Hello!")

    assert resp.text == "Hello!!"
    assert events == ["before:chat", "after:chat"]
    call_kwargs = mock_provider.acomplete.call_args[1]
    assert call_kwargs["middleware_marker"] == "async"


@patch("ai_arch_toolkit.llm._async_client.create_provider")
async def test_async_stream_can_be_transformed_by_middleware(
    mock_create: MagicMock, mock_provider: MagicMock
) -> None:
    mock_create.return_value = mock_provider
    events: list[str] = []
    middleware = _AsyncRecorderMiddleware(events)
    client = AsyncClient("openai", model="gpt-4o", api_key="sk-test", middleware=[middleware])

    chunks = []
    async for chunk in client.stream("Tell me a story"):
        chunks.append(chunk)

    assert chunks == ["HELLO", " WORLD"]
    assert events == ["before:stream", "after:stream"]
