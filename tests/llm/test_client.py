"""Tests for the synchronous Client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ai_arch_toolkit.llm._client import Client
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
