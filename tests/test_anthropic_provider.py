"""Tests for _providers/_anthropic.py — SDK adapter."""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_arch_toolkit.core._content import CachePart, DocumentPart, ImagePart
from ai_arch_toolkit.core._exceptions import APIError, RateLimitError
from ai_arch_toolkit.core._providers._anthropic import (
    AnthropicProvider,
    _build_output_config,
    _build_thinking_param,
    _content_to_sdk,
    _extract_usage,
    _messages_to_sdk,
    _parse_sdk_response,
    _tool_to_sdk,
)
from ai_arch_toolkit.core._providers._base import StreamState
from ai_arch_toolkit.core._response import OutputSchema, Response, ToolCall

# ---------------------------------------------------------------------------
# Helpers — build fake SDK objects
# ---------------------------------------------------------------------------


def _sdk_message(
    *,
    text: str = "Hello!",
    tool_calls: list[dict] | None = None,
    thinking: list[str] | None = None,
    model: str = "claude-sonnet-4-20250514",
    stop_reason: str = "end_turn",
    input_tokens: int = 10,
    output_tokens: int = 5,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> SimpleNamespace:
    """Build a fake anthropic.types.Message-like object."""
    content = []
    if thinking:
        for t in thinking:
            content.append(SimpleNamespace(type="thinking", thinking=t, signature="sig"))
    if text:
        content.append(SimpleNamespace(type="text", text=text, citations=None))
    if tool_calls:
        for tc in tool_calls:
            content.append(
                SimpleNamespace(
                    type="tool_use",
                    id=tc["id"],
                    name=tc["name"],
                    input=tc.get("input", {}),
                )
            )
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
    )
    return SimpleNamespace(
        content=content,
        model=model,
        stop_reason=stop_reason,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestMessagesToSdk:
    def test_extracts_system(self):
        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ]
        sys, wire = _messages_to_sdk(msgs)
        assert sys == "Be helpful."
        assert len(wire) == 1
        assert wire[0] == {"role": "user", "content": "Hi"}

    def test_no_system(self):
        msgs = [{"role": "user", "content": "Hi"}]
        sys, wire = _messages_to_sdk(msgs)
        assert sys is None
        assert len(wire) == 1

    def test_multiple_system_messages_joined(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ]
        sys, wire = _messages_to_sdk(msgs)
        assert sys == "You are helpful.\n\nBe concise."
        assert len(wire) == 1

    def test_tool_result_message(self):
        msgs = [{"role": "user", "content": "42", "tool_use_id": "call_1"}]
        _, wire = _messages_to_sdk(msgs)
        assert wire[0]["role"] == "user"
        assert wire[0]["content"][0]["type"] == "tool_result"
        assert wire[0]["content"][0]["tool_use_id"] == "call_1"

    def test_assistant_with_tool_calls(self):
        msgs = [
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {"id": "tc_1", "name": "get_weather", "input": {"city": "NYC"}},
                ],
            },
        ]
        _, wire = _messages_to_sdk(msgs)
        assert wire[0]["role"] == "assistant"
        content = wire[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "Let me check."}
        assert content[1] == {
            "type": "tool_use",
            "id": "tc_1",
            "name": "get_weather",
            "input": {"city": "NYC"},
        }

    def test_assistant_with_tool_calls_no_text(self):
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc_1", "name": "search", "input": {"q": "test"}},
                ],
            },
        ]
        _, wire = _messages_to_sdk(msgs)
        content = wire[0]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "tool_use"

    def test_assistant_with_multiple_tool_calls(self):
        msgs = [
            {
                "role": "assistant",
                "content": "Checking both.",
                "tool_calls": [
                    {"id": "tc_1", "name": "get_weather", "input": {"city": "NYC"}},
                    {"id": "tc_2", "name": "get_time", "input": {"tz": "UTC"}},
                ],
            },
        ]
        _, wire = _messages_to_sdk(msgs)
        content = wire[0]["content"]
        assert len(content) == 3
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "tool_use"
        assert content[2]["type"] == "tool_use"


class TestToolToSdk:
    def test_maps_parameters_to_input_schema(self):
        tool = {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
        result = _tool_to_sdk(tool)
        assert result["input_schema"] == tool["parameters"]
        assert "parameters" not in result

    def test_falls_back_to_input_schema_key(self):
        tool = {"name": "fn", "description": "desc", "input_schema": {"type": "object"}}
        result = _tool_to_sdk(tool)
        assert result["input_schema"] == {"type": "object"}

    def test_prefers_input_schema_over_parameters(self):
        tool = {
            "name": "fn",
            "description": "desc",
            "input_schema": {"type": "object", "properties": {"a": {"type": "string"}}},
            "parameters": {"type": "object", "properties": {"b": {"type": "integer"}}},
        }
        result = _tool_to_sdk(tool)
        assert "a" in result["input_schema"]["properties"]
        assert "b" not in result["input_schema"]["properties"]


class TestBuildThinkingParam:
    def test_disabled(self):
        assert _build_thinking_param(False, None, None) is None

    def test_enabled_default(self):
        cfg = _build_thinking_param(True, None, None)
        assert cfg == {"type": "enabled", "budget_tokens": 10000}

    def test_explicit_budget(self):
        cfg = _build_thinking_param(True, None, 5000)
        assert cfg == {"type": "enabled", "budget_tokens": 5000}

    def test_effort_high(self):
        cfg = _build_thinking_param(True, "high", None)
        assert cfg["budget_tokens"] == 10000

    def test_effort_low(self):
        cfg = _build_thinking_param(True, "low", None)
        assert cfg["budget_tokens"] == 2048

    def test_effort_medium(self):
        cfg = _build_thinking_param(True, "medium", None)
        assert cfg["budget_tokens"] == 5000


class TestBuildOutputConfig:
    def test_creates_output_config(self):
        schema = OutputSchema(
            name="Person",
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
        )
        cfg = _build_output_config(schema)
        assert cfg["format"]["type"] == "json_schema"
        assert cfg["format"]["schema"] == schema.schema


class TestExtractUsage:
    def test_basic(self):
        sdk_usage = SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=20,
            cache_read_input_tokens=10,
        )
        usage = _extract_usage(sdk_usage)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cache_write_tokens == 20
        assert usage.cache_read_tokens == 10

    def test_missing_cache_fields(self):
        sdk_usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        usage = _extract_usage(sdk_usage)
        assert usage.cache_write_tokens == 0
        assert usage.cache_read_tokens == 0


class TestParseSdkResponse:
    def test_text_response(self):
        msg = _sdk_message(text="Hello!")
        r = _parse_sdk_response(msg, "claude-sonnet-4-20250514")
        assert r.text == "Hello!"
        assert r.usage.input_tokens == 10
        assert r.usage.output_tokens == 5
        assert r.stop_reason == "end_turn"
        assert isinstance(r, Response)

    def test_tool_calls(self):
        msg = _sdk_message(
            text="Let me check.",
            tool_calls=[{"id": "tc_1", "name": "get_weather", "input": {"city": "NYC"}}],
        )
        r = _parse_sdk_response(msg, "claude-sonnet-4-20250514")
        assert r.text == "Let me check."
        assert len(r.tool_calls) == 1
        assert isinstance(r.tool_calls[0], ToolCall)
        assert r.tool_calls[0].name == "get_weather"
        assert r.tool_calls[0].input == {"city": "NYC"}

    def test_cost_is_computed(self):
        msg = _sdk_message(input_tokens=1000, output_tokens=500)
        r = _parse_sdk_response(msg, "claude-sonnet-4-20250514")
        assert r.cost is not None
        assert r.cost > 0

    def test_cost_unknown_model(self):
        msg = _sdk_message(input_tokens=1000, output_tokens=500)
        r = _parse_sdk_response(msg, "unknown-model-v9")
        assert r.cost is None

    def test_cache_tokens(self):
        msg = _sdk_message(cache_creation_input_tokens=20, cache_read_input_tokens=10)
        r = _parse_sdk_response(msg, "claude-sonnet-4-20250514")
        assert r.usage.cache_write_tokens == 20
        assert r.usage.cache_read_tokens == 10

    def test_thinking_blocks(self):
        msg = _sdk_message(text="Answer", thinking=["Let me reason..."])
        r = _parse_sdk_response(msg, "claude-sonnet-4-20250514")
        assert len(r.thinking) == 1
        assert r.thinking[0].text == "Let me reason..."

    def test_structured_output_parsed(self):
        schema = OutputSchema(name="Person", schema={"type": "object"})
        msg = _sdk_message(
            text='{"name": "Alice"}',
            stop_reason="end_turn",
        )
        r = _parse_sdk_response(msg, "claude-sonnet-4-20250514", output_schema=schema)
        assert r.parsed == {"name": "Alice"}

    def test_raw_is_preserved(self):
        msg = _sdk_message()
        r = _parse_sdk_response(msg, "claude-sonnet-4-20250514")
        assert r.raw is msg


class TestStreamState:
    def test_initial_state(self):
        state = StreamState()
        assert state.usage is None
        assert state.model == ""
        assert state.stop_reason == ""
        assert state.raw is None
        assert state.tool_calls == []
        assert state.thinking == []


# ---------------------------------------------------------------------------
# Provider integration tests (mocked SDK client)
# ---------------------------------------------------------------------------


class TestAnthropicProviderComplete:
    @patch("ai_arch_toolkit.core._providers._anthropic.anthropic")
    async def test_complete(self, mock_sdk):
        mock_client = AsyncMock()
        mock_sdk.AsyncAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _sdk_message(text="Hello!")

        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        provider._client = mock_client
        result = await provider.complete([{"role": "user", "content": "Hi"}])
        assert result.text == "Hello!"
        assert isinstance(result, Response)
        mock_client.messages.create.assert_called_once()

    @patch("ai_arch_toolkit.core._providers._anthropic.anthropic")
    async def test_complete_with_tools(self, mock_sdk):
        mock_client = AsyncMock()
        mock_sdk.AsyncAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _sdk_message(
            text="",
            tool_calls=[{"id": "tc_1", "name": "search", "input": {"q": "test"}}],
            stop_reason="tool_use",
        )

        tools = [{"name": "search", "description": "Search", "parameters": {"type": "object"}}]
        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        provider._client = mock_client
        result = await provider.complete([{"role": "user", "content": "Hi"}], tools=tools)
        assert result.has_tool_calls

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" in call_kwargs

    @patch("ai_arch_toolkit.core._providers._anthropic.anthropic")
    async def test_system_from_messages(self, mock_sdk):
        mock_client = AsyncMock()
        mock_sdk.AsyncAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _sdk_message(text="Ok")

        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        provider._client = mock_client
        msgs = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ]
        await provider.complete(msgs)
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "Be brief."
        assert all(m["role"] != "system" for m in call_kwargs["messages"])

    @patch("ai_arch_toolkit.core._providers._anthropic.anthropic")
    async def test_explicit_system_overrides(self, mock_sdk):
        mock_client = AsyncMock()
        mock_sdk.AsyncAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _sdk_message(text="Ok")

        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        provider._client = mock_client
        msgs = [
            {"role": "system", "content": "From message."},
            {"role": "user", "content": "Hi"},
        ]
        await provider.complete(msgs, system="Explicit system.")
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "Explicit system."

    @patch("ai_arch_toolkit.core._providers._anthropic.anthropic")
    async def test_thinking_forwarded(self, mock_sdk):
        mock_client = AsyncMock()
        mock_sdk.AsyncAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _sdk_message(
            text="Answer", thinking=["reasoning"]
        )

        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        provider._client = mock_client
        result = await provider.complete(
            [{"role": "user", "content": "Hi"}],
            thinking=True,
            thinking_effort="high",
        )
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "thinking" in call_kwargs
        assert call_kwargs["thinking"]["type"] == "enabled"
        assert "temperature" not in call_kwargs  # removed when thinking
        assert len(result.thinking) == 1

    @patch("ai_arch_toolkit.core._providers._anthropic.anthropic")
    async def test_output_schema_forwarded(self, mock_sdk):
        mock_client = AsyncMock()
        mock_sdk.AsyncAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _sdk_message(
            text='{"name": "Alice"}',
            stop_reason="end_turn",
        )

        schema = OutputSchema(
            name="Person",
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
        )
        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        provider._client = mock_client
        result = await provider.complete(
            [{"role": "user", "content": "Hi"}],
            output_schema=schema,
        )
        assert result.parsed == {"name": "Alice"}

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "output_config" in call_kwargs
        assert call_kwargs["output_config"]["format"]["type"] == "json_schema"

    @patch("ai_arch_toolkit.core._providers._anthropic.anthropic")
    async def test_output_schema_with_tools_coexist(self, mock_sdk):
        """Anthropic now supports both tools and output_schema via native JSON mode."""
        mock_client = AsyncMock()
        mock_sdk.AsyncAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _sdk_message(
            text='{"answer": "42"}',
        )

        schema = OutputSchema(name="X", schema={"type": "object"})
        tools = [{"name": "search", "description": "Search", "parameters": {"type": "object"}}]
        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        provider._client = mock_client
        result = await provider.complete(
            [{"role": "user", "content": "Hi"}],
            tools=tools,
            output_schema=schema,
        )
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert "output_config" in call_kwargs
        assert result.parsed == {"answer": "42"}

    @patch("ai_arch_toolkit.core._providers._anthropic.anthropic")
    async def test_unknown_kwargs_warn(self, mock_sdk):
        mock_client = AsyncMock()
        mock_sdk.AsyncAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _sdk_message()

        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        provider._client = mock_client
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await provider.complete(
                [{"role": "user", "content": "Hi"}],
                topp=0.9,
                typo_param=True,
            )
            assert len(w) == 1
            assert "topp" in str(w[0].message)

    @patch("ai_arch_toolkit.core._providers._anthropic.anthropic")
    async def test_known_kwargs_no_warn(self, mock_sdk):
        mock_client = AsyncMock()
        mock_sdk.AsyncAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _sdk_message()

        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        provider._client = mock_client
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await provider.complete(
                [{"role": "user", "content": "Hi"}],
                temperature=0.5,
                top_p=0.9,
            )
            assert len(w) == 0

    @patch("ai_arch_toolkit.core._providers._anthropic.anthropic")
    async def test_deprecated_temperature_model_drops_temperature(self, mock_sdk):
        mock_client = AsyncMock()
        mock_sdk.AsyncAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _sdk_message()

        provider = AnthropicProvider("claude-opus-4-7", "test-key")
        provider._client = mock_client
        await provider.complete(
            [{"role": "user", "content": "Hi"}],
            temperature=0.0,
        )
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "temperature" not in call_kwargs

    @patch("ai_arch_toolkit.core._providers._anthropic.anthropic")
    async def test_does_not_inject_max_tokens(self, mock_sdk):
        mock_client = AsyncMock()
        mock_sdk.AsyncAnthropic.return_value = mock_client
        mock_client.messages.create.return_value = _sdk_message()

        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        provider._client = mock_client
        await provider.complete([{"role": "user", "content": "Hi"}])
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "max_tokens" not in call_kwargs


class TestAnthropicProviderErrors:
    async def test_rate_limit_error(self):
        import anthropic as anthropic_sdk
        import httpx

        mock_client = AsyncMock()
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        resp = httpx.Response(429, headers={"retry-after": "5.0"}, request=request)
        mock_client.messages.create.side_effect = anthropic_sdk.RateLimitError(
            "rate limited", response=resp, body={"error": "too many requests"}
        )

        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        provider._client = mock_client
        with pytest.raises(RateLimitError) as exc_info:
            await provider.complete([{"role": "user", "content": "Hi"}])
        assert exc_info.value.status_code == 429
        assert exc_info.value.retry_after == 5.0

    async def test_api_status_error(self):
        import anthropic as anthropic_sdk
        import httpx

        mock_client = AsyncMock()
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        resp = httpx.Response(500, request=request)
        mock_client.messages.create.side_effect = anthropic_sdk.APIStatusError(
            "server error", response=resp, body="internal"
        )

        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        provider._client = mock_client
        with pytest.raises(APIError) as exc_info:
            await provider.complete([{"role": "user", "content": "Hi"}])
        assert exc_info.value.status_code == 500


class TestAnthropicProviderLifecycle:
    async def test_close(self):
        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        await provider.close()

    async def test_context_manager(self):
        async with AnthropicProvider("claude-sonnet-4-20250514", "test-key") as provider:
            assert provider._client is not None


class _FakeAnthropicStream:
    def __init__(self, events, final_message):
        self._events = iter(events)
        self._final = final_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._events)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def get_final_message(self):
        return self._final


class TestAnthropicProviderStreamEvents:
    async def test_stream_events_yield_thinking_deltas(self):
        events = [
            SimpleNamespace(
                type="content_block_start",
                content_block=SimpleNamespace(type="thinking"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="thinking_delta", thinking="step1 "),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="thinking_delta", thinking="step2"),
            ),
            SimpleNamespace(type="content_block_stop"),
        ]
        final_message = SimpleNamespace(content=[])

        mock_client = MagicMock()
        mock_client.messages.stream.return_value = _FakeAnthropicStream(events, final_message)

        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        provider._client = mock_client

        event_iter, state = provider.stream_events(
            [{"role": "user", "content": "Hi"}],
            max_tokens=64,
        )
        collected = [event async for event in event_iter]

        thinking_events = [event for event in collected if event.kind == "thinking"]
        # Deltas are buffered and emitted as a single complete block on content_block_stop
        assert [e.thinking.text for e in thinking_events if e.thinking] == ["step1 step2"]
        assert [b.text for b in state.thinking] == ["step1 step2"]


# ---------------------------------------------------------------------------
# Multimodal content conversion
# ---------------------------------------------------------------------------


class TestContentToSdk:
    def test_string_passthrough(self):
        assert _content_to_sdk("hello") == "hello"

    def test_image_url(self):
        parts = [ImagePart(source="https://example.com/img.png", media_type="image/png")]
        result = _content_to_sdk(parts)
        assert result == [
            {"type": "image", "source": {"type": "url", "url": "https://example.com/img.png"}}
        ]

    def test_image_b64(self):
        parts = [ImagePart(source="abc123", media_type="image/jpeg")]
        result = _content_to_sdk(parts)
        assert result[0]["type"] == "image"
        assert result[0]["source"]["type"] == "base64"
        assert result[0]["source"]["media_type"] == "image/jpeg"
        assert result[0]["source"]["data"] == "abc123"

    def test_image_bytes(self):
        parts = [ImagePart(source=b"\x89PNG", media_type="image/png")]
        result = _content_to_sdk(parts)
        assert result[0]["source"]["type"] == "base64"

    def test_document(self):
        parts = [DocumentPart(source="b64data", media_type="application/pdf", name="doc.pdf")]
        result = _content_to_sdk(parts)
        assert result[0]["type"] == "document"
        assert result[0]["name"] == "doc.pdf"

    def test_cache_part(self):
        parts = [CachePart(content="cached text")]
        result = _content_to_sdk(parts)
        assert result[0] == {
            "type": "text",
            "text": "cached text",
            "cache_control": {"type": "ephemeral"},
        }

    def test_mixed_content(self):
        parts = [
            "Describe this:",
            ImagePart(source="https://example.com/img.png"),
        ]
        result = _content_to_sdk(parts)
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "Describe this:"}
        assert result[1]["type"] == "image"

    def test_multimodal_messages_to_sdk(self):
        """Multimodal content flows through _messages_to_sdk."""
        msgs = [
            {
                "role": "user",
                "content": ["hello", ImagePart(source="https://img.com/a.png")],
            }
        ]
        _, wire = _messages_to_sdk(msgs)
        assert len(wire) == 1
        assert isinstance(wire[0]["content"], list)
        assert wire[0]["content"][0] == {"type": "text", "text": "hello"}
