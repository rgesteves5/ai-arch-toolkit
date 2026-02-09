"""Tests for the Anthropic provider."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ai_arch_toolkit.llm._providers._anthropic import AnthropicProvider
from ai_arch_toolkit.llm._types import (
    AudioPart,
    DocumentPart,
    ImagePart,
    JsonSchema,
    Message,
    TextPart,
    ThinkingConfig,
    Tool,
    ToolCall,
    ToolResult,
)
from tests.conftest import MockResponse

_TEXT_RESPONSE = {
    "id": "msg_123",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5-20250929",
    "content": [{"type": "text", "text": "Hello there!"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 5},
}

_TOOL_RESPONSE = {
    "id": "msg_456",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5-20250929",
    "content": [
        {"type": "text", "text": "Let me check the weather."},
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "get_weather",
            "input": {"city": "Paris"},
        },
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 20, "output_tokens": 15},
}

_THINKING_RESPONSE = {
    "id": "msg_789",
    "type": "message",
    "role": "assistant",
    "content": [
        {"type": "thinking", "thinking": "Let me reason about this..."},
        {"type": "thinking", "thinking": "The answer should be 42."},
        {"type": "text", "text": "The answer is 42."},
    ],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 50, "output_tokens": 30},
}

_CACHE_RESPONSE = {
    "id": "msg_cache",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Cached!"}],
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 100,
        "output_tokens": 10,
        "cache_creation_input_tokens": 50,
        "cache_read_input_tokens": 30,
    },
}


def _make_provider() -> AnthropicProvider:
    return AnthropicProvider("claude-sonnet-4-5-20250929", "sk-ant-test")


def test_complete_text(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    resp = provider.complete([Message("user", "Hi")])

    assert resp.text == "Hello there!"
    assert resp.usage.input_tokens == 10
    assert resp.usage.output_tokens == 5
    assert resp.usage.total_tokens == 15
    assert resp.stop_reason == "end_turn"


def test_system_is_top_level(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    provider.complete([Message("user", "Hi")], system="Be helpful")

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert payload["system"] == "Be helpful"
    # system should NOT appear as a message
    assert all(m["role"] != "system" for m in payload["messages"])


def test_max_tokens_required(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    provider.complete([Message("user", "Hi")])

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert "max_tokens" in payload


def test_complete_with_tools(mock_post: MagicMock, weather_tool: Tool) -> None:
    mock_post.return_value = MockResponse(json_data=_TOOL_RESPONSE)
    provider = _make_provider()
    resp = provider.complete([Message("user", "Weather?")], tools=[weather_tool])

    assert len(resp.tool_calls) == 1
    assert isinstance(resp.tool_calls, tuple)
    tc = resp.tool_calls[0]
    assert tc.id == "toolu_1"
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "Paris"}

    # Check tool format uses input_schema
    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    tool_def = payload["tools"][0]
    assert "input_schema" in tool_def
    assert "parameters" not in tool_def


def test_headers(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    provider.complete([Message("user", "Hi")])

    call_kwargs = mock_post.call_args
    headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
    assert headers["x-api-key"] == "sk-ant-test"
    assert headers["anthropic-version"] == "2023-06-01"


def test_complete_passes_timeout(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    provider.complete([Message("user", "Hi")], timeout=30)

    call_kwargs = mock_post.call_args
    assert call_kwargs.kwargs.get("timeout") == 30
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert "timeout" not in payload


def test_stream(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(
        lines=[
            'data: {"type":"message_start","message":{"id":"msg_1"}}',
            'data: {"type":"content_block_start","index":0}',
            "data: "
            '{"type":"content_block_delta","index":0,'
            '"delta":{"type":"text_delta","text":"Hello"}}',
            "data: "
            '{"type":"content_block_delta","index":0,'
            '"delta":{"type":"text_delta","text":" world"}}',
            'data: {"type":"message_stop"}',
        ]
    )
    provider = _make_provider()
    chunks = list(provider.stream([Message("user", "Hi")]))
    assert chunks == ["Hello", " world"]


def test_stream_skips_empty_text(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(
        lines=[
            "data: "
            '{"type":"content_block_delta","index":0,'
            '"delta":{"type":"text_delta","text":""}}',
            "data: "
            '{"type":"content_block_delta","index":0,'
            '"delta":{"type":"text_delta","text":"Hello"}}',
        ]
    )
    provider = _make_provider()
    chunks = list(provider.stream([Message("user", "Hi")]))
    assert chunks == ["Hello"]


# --- Multi-turn tool use tests ---


def test_tool_result_wire_format(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    tc = ToolCall(id="toolu_1", name="get_weather", arguments={"city": "Paris"})
    provider.complete(
        [
            Message("user", "Weather?"),
            Message("assistant", "Let me check.", tool_calls=(tc,)),
            ToolResult(tool_call_id="toolu_1", name="get_weather", content='{"temp": 20}'),
        ]
    )

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    msgs = payload["messages"]

    # user message
    assert msgs[0] == {"role": "user", "content": "Weather?"}

    # assistant with tool_use blocks
    assert msgs[1]["role"] == "assistant"
    content_blocks = msgs[1]["content"]
    assert content_blocks[0] == {"type": "text", "text": "Let me check."}
    assert content_blocks[1] == {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "get_weather",
        "input": {"city": "Paris"},
    }

    # tool result wrapped in user message
    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": '{"temp": 20}',
        }
    ]


def test_multiple_tool_results_combine(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    tc1 = ToolCall(id="t1", name="f1", arguments={})
    tc2 = ToolCall(id="t2", name="f2", arguments={})
    provider.complete(
        [
            Message("assistant", tool_calls=(tc1, tc2)),
            ToolResult(tool_call_id="t1", name="f1", content="r1"),
            ToolResult(tool_call_id="t2", name="f2", content="r2"),
        ]
    )

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    msgs = payload["messages"]

    # Two tool results should combine into one user message
    assert len(msgs) == 2
    assert msgs[1]["role"] == "user"
    assert len(msgs[1]["content"]) == 2


# --- JSON mode tests ---


def test_json_schema_appends_to_system(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    schema = JsonSchema(
        name="person",
        schema={"type": "object", "properties": {"name": {"type": "string"}}},
    )
    provider.complete([Message("user", "Extract")], system="You extract data.", json_schema=schema)

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert "You extract data." in payload["system"]
    assert "json" in payload["system"].lower()


def test_json_schema_without_system(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    schema = JsonSchema(name="data", schema={"type": "object"})
    provider.complete([Message("user", "Extract")], json_schema=schema)

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert "system" in payload
    assert "json" in payload["system"].lower()


def test_no_system_without_json_schema(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    provider.complete([Message("user", "Hi")])

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert "system" not in payload


# --- Multimodal tests ---


def test_image_base64_wire_format(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    parts = (TextPart("Describe"), ImagePart(data="abc123", media_type="image/jpeg"))
    provider.complete([Message("user", parts)])

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    content = payload["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "Describe"}
    assert content[1]["type"] == "image"
    assert content[1]["source"]["type"] == "base64"
    assert content[1]["source"]["media_type"] == "image/jpeg"
    assert content[1]["source"]["data"] == "abc123"


def test_image_url_wire_format(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    parts = (ImagePart(url="https://example.com/img.png"),)
    provider.complete([Message("user", parts)])

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["type"] == "url"
    assert content[0]["source"]["url"] == "https://example.com/img.png"


def test_document_wire_format(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    parts = (DocumentPart(data="pdfdata"),)
    provider.complete([Message("user", parts)])

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "document"
    assert content[0]["source"]["type"] == "base64"
    assert content[0]["source"]["media_type"] == "application/pdf"


def test_audio_raises_error() -> None:
    provider = _make_provider()
    parts = (AudioPart(data="audiodata", media_type="audio/wav"),)
    with pytest.raises(ValueError, match="does not support audio"):
        provider._build_payload([Message("user", parts)])


# --- Thinking tests ---


def test_thinking_config_extended(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    provider.complete(
        [Message("user", "Think")],
        thinking=ThinkingConfig(budget_tokens=16384),
    )

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert payload["thinking"]["type"] == "enabled"
    assert payload["thinking"]["budget_tokens"] == 16384


def test_thinking_config_adaptive(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    provider.complete(
        [Message("user", "Think")],
        thinking=ThinkingConfig(effort="high"),
    )

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert payload["thinking"]["type"] == "adaptive"
    assert payload["thinking"]["effort"] == "high"


def test_thinking_blocks_parsed(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_THINKING_RESPONSE)
    provider = _make_provider()
    resp = provider.complete([Message("user", "Think deeply")])

    assert resp.text == "The answer is 42."
    assert len(resp.thinking_blocks) == 2
    assert resp.thinking_blocks[0].text == "Let me reason about this..."
    assert "42" in resp.thinking


# --- Cache tests ---


def test_cache_control_payload(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    provider.complete(
        [Message("user", "Hi")],
        system="System prompt",
        cache_control=True,
    )

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert isinstance(payload["system"], list)
    assert payload["system"][0]["cache_control"]["type"] == "ephemeral"

    headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
    assert headers["anthropic-beta"] == "prompt-caching-2024-07-31"


def test_cache_usage_parsed(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_CACHE_RESPONSE)
    provider = _make_provider()
    resp = provider.complete([Message("user", "Hi")])

    assert resp.usage.cache_creation_tokens == 50
    assert resp.usage.cache_read_tokens == 30


# --- Computer use test ---


def test_computer_use_beta_header(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    provider.complete([Message("user", "Click button")], computer_use=True)

    call_kwargs = mock_post.call_args
    headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
    assert headers["anthropic-beta"] == "computer-use-2025-01-24"


# --- Stream events tests ---


def test_stream_events_text_and_done(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(
        lines=[
            'data: {"type":"content_block_start","content_block":{"type":"text"}}',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":" world"}}',
            'data: {"type":"content_block_stop"}',
            'data: {"type":"message_stop"}',
        ]
    )
    provider = _make_provider()
    events = list(provider.stream_events([Message("user", "Hi")]))
    text_events = [e for e in events if e.type == "text"]
    assert len(text_events) == 2
    assert text_events[0].text == "Hello"
    assert text_events[1].text == " world"
    assert events[-1].type == "done"


def test_stream_events_thinking(mock_post: MagicMock) -> None:
    thinking_delta = (
        'data: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"hmm"}}'
    )
    mock_post.return_value = MockResponse(
        lines=[
            'data: {"type":"content_block_start","content_block":{"type":"thinking"}}',
            thinking_delta,
            'data: {"type":"content_block_stop"}',
            'data: {"type":"message_stop"}',
        ]
    )
    provider = _make_provider()
    events = list(provider.stream_events([Message("user", "Think")]))
    thinking_events = [e for e in events if e.type == "thinking"]
    assert len(thinking_events) == 1
    assert thinking_events[0].thinking == "hmm"


def test_stream_events_tool_call(mock_post: MagicMock) -> None:
    block_start = (
        "data: "
        '{"type":"content_block_start","content_block":'
        '{"type":"tool_use","id":"t1","name":"f"}}'
    )
    json_delta = (
        "data: "
        '{"type":"content_block_delta","delta":'
        '{"type":"input_json_delta",'
        '"partial_json":"{\\"a\\": 1}"}}'
    )
    mock_post.return_value = MockResponse(
        lines=[
            block_start,
            json_delta,
            'data: {"type":"content_block_stop"}',
            'data: {"type":"message_stop"}',
        ]
    )
    provider = _make_provider()
    events = list(provider.stream_events([Message("user", "Call")]))
    tc_events = [e for e in events if e.type == "tool_call"]
    assert len(tc_events) == 1
    assert tc_events[0].tool_call.id == "t1"
    assert tc_events[0].tool_call.name == "f"
    assert tc_events[0].tool_call.arguments == {"a": 1}


def test_stream_events_usage(mock_post: MagicMock) -> None:
    msg_delta = (
        "data: "
        '{"type":"message_delta",'
        '"delta":{"stop_reason":"end_turn"},'
        '"usage":{"output_tokens":10}}'
    )
    mock_post.return_value = MockResponse(
        lines=[
            msg_delta,
            'data: {"type":"message_stop"}',
        ]
    )
    provider = _make_provider()
    events = list(provider.stream_events([Message("user", "Hi")]))
    usage_events = [e for e in events if e.type == "usage"]
    assert len(usage_events) == 1
    assert usage_events[0].usage.output_tokens == 10


# --- Header non-mutation regression test ---


def test_cache_control_does_not_leak_to_next_call(mock_post: MagicMock) -> None:
    """Regression: cache_control on one call must not pollute subsequent calls."""
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()

    # First call with cache_control
    provider.complete([Message("user", "Hi")], system="sys", cache_control=True)
    first_headers = mock_post.call_args.kwargs.get("headers", {})
    assert "anthropic-beta" in first_headers

    # Second call without cache_control
    provider.complete([Message("user", "Hi")])
    second_headers = mock_post.call_args.kwargs.get("headers", {})
    assert "anthropic-beta" not in second_headers
