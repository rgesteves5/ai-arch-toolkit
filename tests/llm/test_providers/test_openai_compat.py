"""Tests for the OpenAI-compatible provider."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from ai_arch_toolkit.llm._providers._openai_compat import OpenAICompatProvider, _parse_tool_args
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

_CHAT_RESPONSE = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello there!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

_TOOL_RESPONSE = {
    "id": "chatcmpl-456",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
}


def _make_provider() -> OpenAICompatProvider:
    return OpenAICompatProvider("openai", "gpt-4o", "sk-test")


def test_complete_text(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_CHAT_RESPONSE)
    provider = _make_provider()
    resp = provider.complete([Message("user", "Hi")])

    assert resp.text == "Hello there!"
    assert resp.usage.input_tokens == 10
    assert resp.usage.output_tokens == 5
    assert resp.usage.total_tokens == 15
    assert resp.stop_reason == "stop"
    assert resp.tool_calls == ()


def test_complete_with_system(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_CHAT_RESPONSE)
    provider = _make_provider()
    provider.complete([Message("user", "Hi")], system="Be helpful")

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert payload["messages"][0] == {"role": "system", "content": "Be helpful"}
    assert payload["messages"][1] == {"role": "user", "content": "Hi"}


def test_complete_with_tools(mock_post: MagicMock, weather_tool: Tool) -> None:
    mock_post.return_value = MockResponse(json_data=_TOOL_RESPONSE)
    provider = _make_provider()
    resp = provider.complete([Message("user", "Weather?")], tools=[weather_tool])

    assert len(resp.tool_calls) == 1
    assert isinstance(resp.tool_calls, tuple)
    tc = resp.tool_calls[0]
    assert tc.id == "call_1"
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "Paris"}


def test_parse_tool_args_valid_json() -> None:
    assert _parse_tool_args('{"city": "Paris"}') == {"city": "Paris"}


def test_parse_tool_args_dict_passthrough() -> None:
    d = {"city": "Paris"}
    assert _parse_tool_args(d) is d


def test_parse_tool_args_malformed_json() -> None:
    result = _parse_tool_args("not valid json{{{")
    assert result == {"_raw": "not valid json{{{"}


def test_complete_passes_timeout(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_CHAT_RESPONSE)
    provider = _make_provider()
    provider.complete([Message("user", "Hi")], timeout=30)

    call_kwargs = mock_post.call_args
    assert call_kwargs.kwargs.get("timeout") == 30
    # timeout should NOT leak into the payload
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert "timeout" not in payload


def test_stream(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(
        lines=[
            'data: {"choices":[{"delta":{"role":"assistant"}}]}',
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            "data: [DONE]",
        ]
    )
    provider = _make_provider()
    chunks = list(provider.stream([Message("user", "Hi")]))
    assert chunks == ["Hello", " world"]


def test_provider_urls() -> None:
    """Each provider configures the right base URL."""
    for name, expected_path in [
        ("openai", "https://api.openai.com/v1/chat/completions"),
        ("xai", "https://api.x.ai/v1/chat/completions"),
        ("mistral", "https://api.mistral.ai/v1/chat/completions"),
        ("groq", "https://api.groq.com/openai/v1/chat/completions"),
    ]:
        p = OpenAICompatProvider(name, "model", "key")
        assert p._url == expected_path


# --- Multi-turn tool use tests ---


def test_tool_result_wire_format(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_CHAT_RESPONSE)
    provider = _make_provider()
    tc = ToolCall(id="call_1", name="get_weather", arguments={"city": "Paris"})
    provider.complete(
        [
            Message("user", "Weather?"),
            Message("assistant", tool_calls=(tc,)),
            ToolResult(tool_call_id="call_1", name="get_weather", content='{"temp": 20}'),
        ]
    )

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    msgs = payload["messages"]

    # user message
    assert msgs[0] == {"role": "user", "content": "Weather?"}

    # assistant with tool_calls
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] is None
    assert len(msgs[1]["tool_calls"]) == 1
    assert msgs[1]["tool_calls"][0]["id"] == "call_1"
    assert msgs[1]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(msgs[1]["tool_calls"][0]["function"]["arguments"]) == {"city": "Paris"}

    # tool result
    assert msgs[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"temp": 20}',
    }


def test_assistant_message_with_text_and_tool_calls(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_CHAT_RESPONSE)
    provider = _make_provider()
    tc = ToolCall(id="call_1", name="f", arguments={})
    provider.complete(
        [
            Message("assistant", "thinking", tool_calls=(tc,)),
        ]
    )

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    msg = payload["messages"][0]
    assert msg["content"] == "thinking"
    assert msg["tool_calls"][0]["id"] == "call_1"


# --- JSON mode tests ---


def test_json_schema_in_payload(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_CHAT_RESPONSE)
    provider = _make_provider()
    schema = JsonSchema(
        name="person",
        schema={"type": "object", "properties": {"name": {"type": "string"}}},
    )
    provider.complete([Message("user", "Extract")], json_schema=schema)

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    rf = payload["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "person"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"] == {
        "type": "object",
        "properties": {"name": {"type": "string"}},
    }


def test_no_json_schema_by_default(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_CHAT_RESPONSE)
    provider = _make_provider()
    provider.complete([Message("user", "Hi")])

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert "response_format" not in payload


# --- Multimodal tests ---


def test_image_url_wire_format(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_CHAT_RESPONSE)
    provider = _make_provider()
    parts = (TextPart("Describe"), ImagePart(url="https://example.com/img.png"))
    provider.complete([Message("user", parts)])

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    content = payload["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "Describe"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "https://example.com/img.png"
    assert content[1]["image_url"]["detail"] == "auto"


def test_image_base64_wire_format(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_CHAT_RESPONSE)
    provider = _make_provider()
    parts = (ImagePart(data="abc123", media_type="image/jpeg", detail="high"),)
    provider.complete([Message("user", parts)])

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    content = payload["messages"][0]["content"]
    assert content[0]["image_url"]["url"] == "data:image/jpeg;base64,abc123"
    assert content[0]["image_url"]["detail"] == "high"


def test_audio_input_wire_format(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_CHAT_RESPONSE)
    provider = _make_provider()
    parts = (AudioPart(data="audiodata", media_type="audio/wav"),)
    provider.complete([Message("user", parts)])

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "input_audio"
    assert content[0]["input_audio"]["data"] == "audiodata"
    assert content[0]["input_audio"]["format"] == "wav"


def test_document_raises_error() -> None:
    provider = _make_provider()
    parts = (DocumentPart(data="pdfdata"),)
    with pytest.raises(ValueError, match="does not support document"):
        provider._build_payload([Message("user", parts)])


# --- Thinking tests ---


def test_thinking_config_reasoning_effort(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_CHAT_RESPONSE)
    provider = _make_provider()
    provider.complete([Message("user", "Think")], thinking=ThinkingConfig(effort="high"))

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert payload["reasoning_effort"] == "high"
    assert "include_reasoning" not in payload


def test_thinking_config_groq_include_reasoning(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_CHAT_RESPONSE)
    provider = OpenAICompatProvider("groq", "llama-3", "key")
    provider.complete([Message("user", "Think")], thinking=ThinkingConfig(effort="medium"))

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert payload["reasoning_effort"] == "medium"
    assert payload["include_reasoning"] is True


# --- Stream events tests ---


def test_stream_events_text_and_done(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(
        lines=[
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            "data: [DONE]",
        ]
    )
    provider = _make_provider()
    events = list(provider.stream_events([Message("user", "Hi")]))
    assert events[0].type == "text"
    assert events[0].text == "Hello"
    assert events[1].type == "text"
    assert events[1].text == " world"
    assert events[-1].type == "done"


def test_stream_events_tool_calls(mock_post: MagicMock) -> None:
    tc_start = (
        "data: "
        '{"choices":[{"delta":{"tool_calls":'
        '[{"index":0,"id":"call_1","function":'
        '{"name":"get_weather","arguments":""}}]}}]}'
    )
    tc_arg1 = (
        "data: "
        '{"choices":[{"delta":{"tool_calls":'
        '[{"index":0,"function":{"arguments":"{\\"city\\""}}]}}]}'
    )
    tc_arg2 = (
        "data: "
        '{"choices":[{"delta":{"tool_calls":'
        '[{"index":0,"function":{"arguments":'
        '": \\"Paris\\"}"}}]}}]}'
    )
    mock_post.return_value = MockResponse(
        lines=[
            tc_start,
            tc_arg1,
            tc_arg2,
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]
    )
    provider = _make_provider()
    events = list(provider.stream_events([Message("user", "Weather?")]))
    tc_events = [e for e in events if e.type == "tool_call"]
    assert len(tc_events) == 1
    assert tc_events[0].tool_call.name == "get_weather"
    assert tc_events[0].tool_call.arguments == {"city": "Paris"}


def test_stream_events_usage(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(
        lines=[
            'data: {"choices":[{"delta":{"content":"Hi"}}]}',
            'data: {"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}',
            "data: [DONE]",
        ]
    )
    provider = _make_provider()
    events = list(provider.stream_events([Message("user", "Hi")]))
    usage_events = [e for e in events if e.type == "usage"]
    assert len(usage_events) == 1
    assert usage_events[0].usage.input_tokens == 10
    assert usage_events[0].usage.total_tokens == 15


# --- Audio kwargs test ---


def test_audio_kwarg(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_CHAT_RESPONSE)
    provider = _make_provider()
    provider.complete(
        [Message("user", "Hi")],
        audio={"voice": "alloy", "format": "wav"},
    )

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert payload["modalities"] == ["text", "audio"]
    assert payload["audio"] == {"voice": "alloy", "format": "wav"}
