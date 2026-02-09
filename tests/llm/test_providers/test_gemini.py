"""Tests for the Gemini provider."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from ai_arch_toolkit.llm._providers._gemini import GeminiProvider
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
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [{"text": "Hello there!"}],
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 10,
        "candidatesTokenCount": 5,
        "totalTokenCount": 15,
    },
}

_TOOL_RESPONSE = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "Paris"},
                        }
                    }
                ],
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 20,
        "candidatesTokenCount": 10,
        "totalTokenCount": 30,
    },
}


def _make_provider() -> GeminiProvider:
    return GeminiProvider("gemini-2.5-flash", "test-key")


def test_complete_text(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    resp = provider.complete([Message("user", "Hi")])

    assert resp.text == "Hello there!"
    assert resp.usage.input_tokens == 10
    assert resp.usage.output_tokens == 5
    assert resp.usage.total_tokens == 15
    assert resp.stop_reason == "STOP"


def test_contents_format(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    provider.complete([Message("user", "Hi"), Message("assistant", "Hello")])

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert "contents" in payload
    assert payload["contents"][0] == {"role": "user", "parts": [{"text": "Hi"}]}
    # assistant -> model
    assert payload["contents"][1] == {"role": "model", "parts": [{"text": "Hello"}]}


def test_system_instruction(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    provider.complete([Message("user", "Hi")], system="Be helpful")

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert payload["systemInstruction"] == {"parts": [{"text": "Be helpful"}]}


def test_rejects_system_role_message() -> None:
    provider = _make_provider()
    with pytest.raises(ValueError, match="does not support role 'system'"):
        provider._build_payload([Message("system", "Be helpful")])


def test_complete_with_tools(mock_post: MagicMock, weather_tool: Tool) -> None:
    mock_post.return_value = MockResponse(json_data=_TOOL_RESPONSE)
    provider = _make_provider()
    resp = provider.complete([Message("user", "Weather?")], tools=[weather_tool])

    assert len(resp.tool_calls) == 1
    assert isinstance(resp.tool_calls, tuple)
    tc = resp.tool_calls[0]
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "Paris"}

    # Check tool format uses functionDeclarations
    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert "functionDeclarations" in payload["tools"][0]


def test_generation_config(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    provider.complete([Message("user", "Hi")], temperature=0.5, max_tokens=100, top_p=0.9)

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    gen_config = payload["generationConfig"]
    assert gen_config["temperature"] == 0.5
    assert gen_config["maxOutputTokens"] == 100
    assert gen_config["topP"] == 0.9


def test_url_includes_model() -> None:
    provider = _make_provider()
    assert "gemini-2.5-flash" in provider._url
    assert provider._url.endswith(":generateContent")
    assert ":streamGenerateContent?alt=sse" in provider._stream_url


def test_no_dead_attributes() -> None:
    """Verify no unused ENV_KEY attribute exists."""
    provider = _make_provider()
    assert not hasattr(type(provider), "ENV_KEY")
    # _api_key is now used by create_cache/complete_with_cache
    assert hasattr(provider, "_api_key")


def test_complete_passes_timeout(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    provider.complete([Message("user", "Hi")], timeout=30)

    call_kwargs = mock_post.call_args
    assert call_kwargs.kwargs.get("timeout") == 30


def test_stream(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(
        lines=[
            'data: {"candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}',
            'data: {"candidates":[{"content":{"parts":[{"text":" world"}]}}]}',
        ]
    )
    provider = _make_provider()
    chunks = list(provider.stream([Message("user", "Hi")]))
    assert chunks == ["Hello", " world"]


# --- Multi-turn tool use tests ---


def test_tool_result_wire_format(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    tc = ToolCall(id="", name="get_weather", arguments={"city": "Paris"})
    provider.complete(
        [
            Message("user", "Weather?"),
            Message("assistant", tool_calls=(tc,)),
            ToolResult(tool_call_id="", name="get_weather", content='{"temp": 20}'),
        ]
    )

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    contents = payload["contents"]

    # user message
    assert contents[0] == {"role": "user", "parts": [{"text": "Weather?"}]}

    # assistant with functionCall
    assert contents[1]["role"] == "model"
    assert contents[1]["parts"][0] == {
        "functionCall": {"name": "get_weather", "args": {"city": "Paris"}}
    }

    # tool result with functionResponse
    assert contents[2]["role"] == "user"
    fn_resp = contents[2]["parts"][0]["functionResponse"]
    assert fn_resp["name"] == "get_weather"
    assert fn_resp["response"] == {"temp": 20}


def test_tool_result_non_json_content(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    tc = ToolCall(id="", name="search", arguments={"q": "test"})
    provider.complete(
        [
            Message("assistant", tool_calls=(tc,)),
            ToolResult(tool_call_id="", name="search", content="plain text result"),
        ]
    )

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    fn_resp = payload["contents"][1]["parts"][0]["functionResponse"]
    assert fn_resp["response"] == {"result": "plain text result"}


def test_assistant_message_with_text_and_tool_calls(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    tc = ToolCall(id="", name="f", arguments={"a": 1})
    provider.complete(
        [
            Message("assistant", "thinking", tool_calls=(tc,)),
        ]
    )

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    parts = payload["contents"][0]["parts"]
    assert parts[0] == {"text": "thinking"}
    assert parts[1] == {"functionCall": {"name": "f", "args": {"a": 1}}}


# --- JSON mode tests ---


def test_json_schema_in_generation_config(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    schema = JsonSchema(
        name="person",
        schema={"type": "object", "properties": {"name": {"type": "string"}}},
    )
    provider.complete([Message("user", "Extract")], json_schema=schema)

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    gen_config = payload["generationConfig"]
    assert gen_config["responseMimeType"] == "application/json"
    assert gen_config["responseSchema"] == {
        "type": "object",
        "properties": {"name": {"type": "string"}},
    }


def test_no_generation_config_by_default(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    provider.complete([Message("user", "Hi")])

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert "generationConfig" not in payload


# --- Multimodal content tests ---


def test_inline_image(mock_post: MagicMock) -> None:
    """Inline base64 image is sent as inlineData."""
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    parts = (
        TextPart("Describe this image"),
        ImagePart(media_type="image/png", data="iVBORw0KGgo="),
    )
    provider.complete([Message("user", parts)])

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    content_parts = payload["contents"][0]["parts"]

    assert content_parts[0] == {"text": "Describe this image"}
    assert content_parts[1] == {"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}}


def test_file_data_pdf(mock_post: MagicMock) -> None:
    """A DocumentPart with a URI is sent as fileData."""
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    parts = (
        TextPart("Summarize this PDF"),
        DocumentPart(uri="gs://my-bucket/report.pdf", media_type="application/pdf"),
    )
    provider.complete([Message("user", parts)])

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    content_parts = payload["contents"][0]["parts"]

    assert content_parts[0] == {"text": "Summarize this PDF"}
    assert content_parts[1] == {
        "fileData": {"mimeType": "application/pdf", "fileUri": "gs://my-bucket/report.pdf"}
    }


def test_audio_inline(mock_post: MagicMock) -> None:
    """Inline base64 audio is sent as inlineData."""
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    parts = (
        TextPart("Transcribe this audio"),
        AudioPart(data="UklGRi4A", media_type="audio/wav"),
    )
    provider.complete([Message("user", parts)])

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    content_parts = payload["contents"][0]["parts"]

    assert content_parts[0] == {"text": "Transcribe this audio"}
    assert content_parts[1] == {"inlineData": {"mimeType": "audio/wav", "data": "UklGRi4A"}}


# --- Thinking config tests ---


def test_thinking_config_in_generation_config(mock_post: MagicMock) -> None:
    """ThinkingConfig produces thinkingConfig inside generationConfig."""
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    provider.complete(
        [Message("user", "Think about this")],
        thinking=ThinkingConfig(budget_tokens=4096),
    )

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    gen_config = payload["generationConfig"]
    assert "thinkingConfig" in gen_config
    assert gen_config["thinkingConfig"] == {"thinkingBudget": 4096}


def test_thinking_config_default_budget(mock_post: MagicMock) -> None:
    """ThinkingConfig with no budget_tokens falls back to 8192."""
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    provider.complete(
        [Message("user", "Think")],
        thinking=ThinkingConfig(),
    )

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    gen_config = payload["generationConfig"]
    assert gen_config["thinkingConfig"] == {"thinkingBudget": 8192}


# --- Stream events tests ---


def test_stream_events_text(mock_post: MagicMock) -> None:
    """stream_events yields text events from SSE chunks."""
    mock_post.return_value = MockResponse(
        lines=[
            "data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}),
            "data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": " world"}]}}]}),
        ]
    )
    provider = _make_provider()
    events = list(provider.stream_events([Message("user", "Hi")]))

    text_events = [e for e in events if e.type == "text"]
    assert len(text_events) == 2
    assert text_events[0].text == "Hello"
    assert text_events[1].text == " world"


def test_stream_events_tool_call(mock_post: MagicMock) -> None:
    """stream_events yields tool_call events from functionCall chunks."""
    mock_post.return_value = MockResponse(
        lines=[
            "data: "
            + json.dumps(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "functionCall": {
                                            "name": "get_weather",
                                            "args": {"city": "London"},
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            ),
        ]
    )
    provider = _make_provider()
    events = list(provider.stream_events([Message("user", "Weather?")]))

    tc_events = [e for e in events if e.type == "tool_call"]
    assert len(tc_events) == 1
    assert tc_events[0].tool_call is not None
    assert tc_events[0].tool_call.name == "get_weather"
    assert tc_events[0].tool_call.arguments == {"city": "London"}


def test_stream_events_usage(mock_post: MagicMock) -> None:
    """stream_events yields usage events from usageMetadata."""
    mock_post.return_value = MockResponse(
        lines=[
            "data: "
            + json.dumps(
                {
                    "candidates": [{"content": {"parts": [{"text": "Hi"}]}}],
                    "usageMetadata": {
                        "promptTokenCount": 5,
                        "candidatesTokenCount": 2,
                        "totalTokenCount": 7,
                    },
                }
            ),
        ]
    )
    provider = _make_provider()
    events = list(provider.stream_events([Message("user", "Hi")]))

    usage_events = [e for e in events if e.type == "usage"]
    assert len(usage_events) == 1
    assert usage_events[0].usage is not None
    assert usage_events[0].usage.input_tokens == 5
    assert usage_events[0].usage.output_tokens == 2
    assert usage_events[0].usage.total_tokens == 7


def test_stream_events_done(mock_post: MagicMock) -> None:
    """stream_events always ends with a done event."""
    mock_post.return_value = MockResponse(
        lines=[
            "data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": "Done"}]}}]}),
        ]
    )
    provider = _make_provider()
    events = list(provider.stream_events([Message("user", "Hi")]))

    assert events[-1].type == "done"


# --- Google search grounding tests ---


def test_google_search_grounding(mock_post: MagicMock) -> None:
    """google_search kwarg adds googleSearchRetrieval to tools."""
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    provider.complete(
        [Message("user", "Latest news")],
        google_search=True,
    )

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    tools = payload["tools"]
    assert len(tools) == 1
    assert "googleSearchRetrieval" in tools[0]
    retrieval = tools[0]["googleSearchRetrieval"]
    assert retrieval["dynamicRetrievalConfig"]["mode"] == "MODE_DYNAMIC"
    assert retrieval["dynamicRetrievalConfig"]["dynamicThreshold"] == 0.7


def test_google_search_grounding_custom_threshold(mock_post: MagicMock) -> None:
    """google_search with dict allows custom threshold."""
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    provider.complete(
        [Message("user", "Latest news")],
        google_search={"threshold": 0.5},
    )

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    retrieval = payload["tools"][0]["googleSearchRetrieval"]
    assert retrieval["dynamicRetrievalConfig"]["dynamicThreshold"] == 0.5


def test_google_search_grounding_with_function_tools(
    mock_post: MagicMock, weather_tool: Tool
) -> None:
    """google_search is appended alongside function tools."""
    mock_post.return_value = MockResponse(json_data=_TOOL_RESPONSE)
    provider = _make_provider()
    provider.complete(
        [Message("user", "Weather and news")],
        tools=[weather_tool],
        google_search=True,
    )

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    tools = payload["tools"]
    assert len(tools) == 2
    assert "functionDeclarations" in tools[0]
    assert "googleSearchRetrieval" in tools[1]


# --- Cache tests ---


def test_create_cache_payload(mock_post: MagicMock) -> None:
    """create_cache sends correct payload and returns cache name."""
    mock_post.return_value = MockResponse(json_data={"name": "cachedContents/abc123"})
    provider = _make_provider()
    cache_name = provider.create_cache(
        [Message("user", "Long document content")],
        system="You are a document analyst",
        ttl="600s",
    )

    assert cache_name == "cachedContents/abc123"

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert payload["model"] == "models/gemini-2.5-flash"
    assert payload["ttl"] == "600s"
    assert payload["contents"][0] == {
        "role": "user",
        "parts": [{"text": "Long document content"}],
    }
    assert payload["systemInstruction"] == {"parts": [{"text": "You are a document analyst"}]}


def test_create_cache_default_ttl(mock_post: MagicMock) -> None:
    """create_cache uses default TTL of 300s when not specified."""
    mock_post.return_value = MockResponse(json_data={"name": "cachedContents/def456"})
    provider = _make_provider()
    provider.create_cache([Message("user", "Content")])

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert payload["ttl"] == "300s"
    assert "systemInstruction" not in payload


def test_complete_with_cache_payload(mock_post: MagicMock) -> None:
    """complete_with_cache includes cachedContent key in payload."""
    mock_post.return_value = MockResponse(json_data=_TEXT_RESPONSE)
    provider = _make_provider()
    resp = provider.complete_with_cache(
        "cachedContents/abc123",
        [Message("user", "Summarize the cached document")],
        temperature=0.3,
    )

    assert resp.text == "Hello there!"

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert "cachedContent" in payload
    assert payload["cachedContent"] == "cachedContents/abc123"
    assert payload["contents"][0] == {
        "role": "user",
        "parts": [{"text": "Summarize the cached document"}],
    }
    assert payload["generationConfig"]["temperature"] == 0.3
