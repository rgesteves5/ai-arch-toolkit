"""Tests for the synchronous Client."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

from ai_arch_toolkit.llm._client import Client
from ai_arch_toolkit.llm._middleware import Request
from ai_arch_toolkit.llm._types import Message, Response, StreamEvent, ToolCall, Usage


def _mock_provider() -> MagicMock:
    provider = MagicMock()
    provider.complete = MagicMock(return_value=Response(text="Hi!", usage=Usage(10, 5, 15)))
    provider.stream = MagicMock(return_value=iter(["Hello", " world"]))
    provider.stream_events = MagicMock(
        return_value=iter(
            [
                StreamEvent(type="text", text="Hi"),
                StreamEvent(
                    type="tool_call",
                    tool_call=ToolCall(id="c1", name="f", arguments={"a": 1}),
                ),
                StreamEvent(type="usage", usage=Usage(10, 5, 15)),
                StreamEvent(type="done"),
            ]
        )
    )
    return provider


@patch("ai_arch_toolkit.llm._client.create_provider")
def test_chat_string(mock_create: MagicMock) -> None:
    mock_create.return_value = _mock_provider()
    client = Client("openai", model="gpt-4o", api_key="sk-test")
    resp = client.chat("Hello!")

    assert resp.text == "Hi!"
    provider = mock_create.return_value
    call_args = provider.complete.call_args
    messages = call_args[0][0]
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "Hello!"


@patch("ai_arch_toolkit.llm._client.create_provider")
def test_chat_with_system(mock_create: MagicMock) -> None:
    mock_create.return_value = _mock_provider()
    client = Client("openai", model="gpt-4o", api_key="sk-test")
    client.chat("Hi", system="Be helpful")

    provider = mock_create.return_value
    call_kwargs = provider.complete.call_args[1]
    assert call_kwargs["system"] == "Be helpful"


@patch("ai_arch_toolkit.llm._client.create_provider")
def test_chat_messages(mock_create: MagicMock) -> None:
    mock_create.return_value = _mock_provider()
    client = Client("openai", model="gpt-4o", api_key="sk-test")
    client.chat([Message("user", "Hi"), Message("assistant", "Hello")])

    provider = mock_create.return_value
    call_args = provider.complete.call_args
    messages = call_args[0][0]
    assert len(messages) == 2


@patch("ai_arch_toolkit.llm._client.create_provider")
def test_stream(mock_create: MagicMock) -> None:
    mock_create.return_value = _mock_provider()
    client = Client("openai", model="gpt-4o", api_key="sk-test")
    chunks = list(client.stream("Tell me a story"))
    assert chunks == ["Hello", " world"]


@patch("ai_arch_toolkit.llm._client.create_provider")
def test_stream_events(mock_create: MagicMock) -> None:
    mock_create.return_value = _mock_provider()
    client = Client("openai", model="gpt-4o", api_key="sk-test")
    events = list(client.stream_events("Hello"))

    assert len(events) == 4
    assert events[0].type == "text"
    assert events[0].text == "Hi"
    assert events[1].type == "tool_call"
    assert events[1].tool_call.name == "f"
    assert events[2].type == "usage"
    assert events[2].usage.input_tokens == 10
    assert events[3].type == "done"


@patch("ai_arch_toolkit.llm._client.create_provider")
def test_chat_forwards_timeout(mock_create: MagicMock) -> None:
    mock_create.return_value = _mock_provider()
    client = Client("openai", model="gpt-4o", api_key="sk-test")

    client.chat("Hello!", timeout=17)

    provider = mock_create.return_value
    call_kwargs = provider.complete.call_args[1]
    assert call_kwargs["timeout"] == 17


@patch("ai_arch_toolkit.llm._client.create_provider")
def test_stream_forwards_timeout(mock_create: MagicMock) -> None:
    mock_create.return_value = _mock_provider()
    client = Client("openai", model="gpt-4o", api_key="sk-test")

    _ = list(client.stream("Hello!", timeout=21))

    provider = mock_create.return_value
    call_kwargs = provider.stream.call_args[1]
    assert call_kwargs["timeout"] == 21


class _SyncRecorderMiddleware:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def before(self, request: Request) -> Request:
        self._events.append(f"before:{request.operation}")
        request.kwargs["middleware_marker"] = "sync"
        return request

    def after(self, request: Request, result: Any) -> Any:
        self._events.append(f"after:{request.operation}")
        if request.operation == "chat":
            return Response(text=f"{result.text}!", usage=result.usage)
        if request.operation == "stream":
            return _upper_stream(result)
        return result


def _upper_stream(stream: Iterator[str]) -> Iterator[str]:
    for chunk in stream:
        yield chunk.upper()


@patch("ai_arch_toolkit.llm._client.create_provider")
def test_chat_runs_middleware_before_and_after(mock_create: MagicMock) -> None:
    mock_create.return_value = _mock_provider()
    events: list[str] = []
    middleware = _SyncRecorderMiddleware(events)
    client = Client("openai", model="gpt-4o", api_key="sk-test", middleware=[middleware])

    resp = client.chat("Hello!")

    assert resp.text == "Hi!!"
    assert events == ["before:chat", "after:chat"]
    provider = mock_create.return_value
    call_kwargs = provider.complete.call_args[1]
    assert call_kwargs["middleware_marker"] == "sync"


@patch("ai_arch_toolkit.llm._client.create_provider")
def test_stream_can_be_transformed_by_middleware(mock_create: MagicMock) -> None:
    mock_create.return_value = _mock_provider()
    events: list[str] = []
    middleware = _SyncRecorderMiddleware(events)
    client = Client("openai", model="gpt-4o", api_key="sk-test", middleware=[middleware])

    chunks = list(client.stream("Tell me a story"))

    assert chunks == ["HELLO", " WORLD"]
    assert events == ["before:stream", "after:stream"]
