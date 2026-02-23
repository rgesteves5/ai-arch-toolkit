"""Tests for _providers/_anthropic.py."""

from __future__ import annotations

import warnings
from unittest.mock import AsyncMock, patch

import pytest

from ai_arch_toolkit._providers._anthropic import (
    AnthropicProvider,
    _StreamState,
    _build_payload,
    _messages_to_wire,
    _parse_response,
    _parse_stream_usage,
    _tool_to_anthropic,
)
from ai_arch_toolkit._response import Response, ToolCall, Usage


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestMessagesToWire:
    def test_extracts_system(self):
        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ]
        sys, wire = _messages_to_wire(msgs)
        assert sys == "Be helpful."
        assert len(wire) == 1
        assert wire[0] == {"role": "user", "content": "Hi"}

    def test_no_system(self):
        msgs = [{"role": "user", "content": "Hi"}]
        sys, wire = _messages_to_wire(msgs)
        assert sys is None
        assert len(wire) == 1

    def test_multiple_system_messages_joined(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ]
        sys, wire = _messages_to_wire(msgs)
        assert sys == "You are helpful.\n\nBe concise."
        assert len(wire) == 1

    def test_tool_result_message(self):
        msgs = [{"role": "user", "content": "42", "tool_use_id": "call_1"}]
        _, wire = _messages_to_wire(msgs)
        assert wire[0]["role"] == "user"
        assert wire[0]["content"][0]["type"] == "tool_result"
        assert wire[0]["content"][0]["tool_use_id"] == "call_1"


class TestParseResponse:
    def test_text_response(self):
        raw = {
            "content": [{"type": "text", "text": "Hello!"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "end_turn",
            "model": "claude-sonnet-4-20250514",
        }
        r = _parse_response(raw, "claude-sonnet-4-20250514")
        assert r.text == "Hello!"
        assert r.usage.input_tokens == 10
        assert r.usage.output_tokens == 5
        assert r.stop_reason == "end_turn"
        assert isinstance(r, Response)

    def test_tool_calls(self):
        raw = {
            "content": [
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool_use",
                    "id": "tc_1",
                    "name": "get_weather",
                    "input": {"city": "NYC"},
                },
            ],
            "usage": {"input_tokens": 20, "output_tokens": 15},
            "stop_reason": "tool_use",
            "model": "claude-sonnet-4-20250514",
        }
        r = _parse_response(raw, "claude-sonnet-4-20250514")
        assert r.text == "Let me check."
        assert len(r.tool_calls) == 1
        assert isinstance(r.tool_calls[0], ToolCall)
        assert r.tool_calls[0].name == "get_weather"
        assert r.tool_calls[0].input == {"city": "NYC"}
        assert r.has_tool_calls

    def test_cost_is_computed(self):
        raw = {
            "content": [{"type": "text", "text": "Hi"}],
            "usage": {"input_tokens": 1000, "output_tokens": 500},
            "stop_reason": "end_turn",
        }
        r = _parse_response(raw, "claude-sonnet-4-20250514")
        assert r.cost > 0
        assert r.cost_estimated is True

    def test_cost_unknown_model(self):
        raw = {
            "content": [{"type": "text", "text": "Hi"}],
            "usage": {"input_tokens": 1000, "output_tokens": 500},
            "stop_reason": "end_turn",
        }
        r = _parse_response(raw, "unknown-model-v9")
        assert r.cost == 0.0
        assert r.cost_estimated is False

    def test_cache_tokens(self):
        raw = {
            "content": [{"type": "text", "text": "Hi"}],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 20,
                "cache_read_input_tokens": 10,
            },
            "stop_reason": "end_turn",
        }
        r = _parse_response(raw, "claude-sonnet-4-20250514")
        assert r.usage.cache_write_tokens == 20
        assert r.usage.cache_read_tokens == 10


class TestBuildPayloadWarning:
    def test_unknown_kwargs_warn(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _build_payload(
                [{"role": "user", "content": "Hi"}],
                model="claude-sonnet-4-20250514",
                topp=0.9,
                typo_param=True,
            )
            assert len(w) == 1
            assert "topp" in str(w[0].message)
            assert "typo_param" in str(w[0].message)

    def test_known_kwargs_no_warn(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _build_payload(
                [{"role": "user", "content": "Hi"}],
                model="claude-sonnet-4-20250514",
                temperature=0.5,
                top_p=0.9,
            )
            assert len(w) == 0


class TestToolToAnthropic:
    def test_maps_parameters_to_input_schema(self):
        tool = {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
        result = _tool_to_anthropic(tool)
        assert result["input_schema"] == tool["parameters"]
        assert "parameters" not in result

    def test_falls_back_to_input_schema_key(self):
        tool = {
            "name": "fn",
            "description": "desc",
            "input_schema": {"type": "object"},
        }
        result = _tool_to_anthropic(tool)
        assert result["input_schema"] == {"type": "object"}


class TestParseStreamUsage:
    def test_extracts_usage(self):
        event = {"usage": {"input_tokens": 100, "output_tokens": 50}}
        usage = _parse_stream_usage(event)
        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_no_usage(self):
        event = {"type": "content_block_delta"}
        usage = _parse_stream_usage(event)
        assert usage is None


class TestStreamState:
    def test_initial_state(self):
        state = _StreamState()
        assert state.usage is None
        assert state.model == ""
        assert state.stop_reason == ""
        assert state.raw is None


# ---------------------------------------------------------------------------
# Provider integration tests (mocked HTTP)
# ---------------------------------------------------------------------------


class TestAnthropicProviderComplete:
    @patch("ai_arch_toolkit._providers._anthropic.async_post_json", new_callable=AsyncMock)
    async def test_complete(self, mock_post):
        mock_post.return_value = {
            "content": [{"type": "text", "text": "Hello!"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "end_turn",
            "model": "claude-sonnet-4-20250514",
        }
        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        result = await provider.complete([{"role": "user", "content": "Hi"}])
        assert result.text == "Hello!"
        assert isinstance(result, Response)
        mock_post.assert_called_once()

    @patch("ai_arch_toolkit._providers._anthropic.async_post_json", new_callable=AsyncMock)
    async def test_complete_passes_client(self, mock_post):
        """Verify the provider passes its httpx client to async_post_json."""
        mock_post.return_value = {
            "content": [{"type": "text", "text": "Ok"}],
            "usage": {"input_tokens": 5, "output_tokens": 3},
            "stop_reason": "end_turn",
        }
        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        await provider.complete([{"role": "user", "content": "Hi"}])
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["client"] is provider._client

    @patch("ai_arch_toolkit._providers._anthropic.async_post_json", new_callable=AsyncMock)
    async def test_complete_with_tools(self, mock_post):
        mock_post.return_value = {
            "content": [
                {"type": "tool_use", "id": "tc_1", "name": "search", "input": {"q": "test"}},
            ],
            "usage": {"input_tokens": 20, "output_tokens": 10},
            "stop_reason": "tool_use",
        }
        tools = [{"name": "search", "description": "Search", "parameters": {"type": "object"}}]
        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        result = await provider.complete([{"role": "user", "content": "Hi"}], tools=tools)
        assert result.has_tool_calls

        # Verify tools were forwarded in payload
        call_args = mock_post.call_args
        payload = call_args[1]["payload"]
        assert "tools" in payload

    @patch("ai_arch_toolkit._providers._anthropic.async_post_json", new_callable=AsyncMock)
    async def test_system_from_messages(self, mock_post):
        mock_post.return_value = {
            "content": [{"type": "text", "text": "Ok"}],
            "usage": {"input_tokens": 5, "output_tokens": 3},
            "stop_reason": "end_turn",
        }
        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        msgs = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ]
        await provider.complete(msgs)
        payload = mock_post.call_args[1]["payload"]
        assert payload["system"] == "Be brief."
        # System should not appear in messages
        assert all(m["role"] != "system" for m in payload["messages"])

    @patch("ai_arch_toolkit._providers._anthropic.async_post_json", new_callable=AsyncMock)
    async def test_explicit_system_overrides(self, mock_post):
        mock_post.return_value = {
            "content": [{"type": "text", "text": "Ok"}],
            "usage": {"input_tokens": 5, "output_tokens": 3},
            "stop_reason": "end_turn",
        }
        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        msgs = [
            {"role": "system", "content": "From message."},
            {"role": "user", "content": "Hi"},
        ]
        await provider.complete(msgs, system="Explicit system.")
        payload = mock_post.call_args[1]["payload"]
        assert payload["system"] == "Explicit system."


class TestAnthropicProviderStream:
    @patch("ai_arch_toolkit._providers._anthropic.async_stream_sse")
    async def test_stream_text_deltas(self, mock_stream):
        events = [
            '{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}',
            '{"type":"content_block_delta","delta":{"type":"text_delta","text":" world"}}',
            '{"type":"message_stop"}',
        ]

        async def _fake_stream(*args, **kwargs):
            for e in events:
                yield e

        mock_stream.return_value = _fake_stream()

        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        chunks = []
        async for chunk in aiter:
            chunks.append(chunk)
        assert chunks == ["Hello", " world"]

    @patch("ai_arch_toolkit._providers._anthropic.async_stream_sse")
    async def test_stream_captures_usage(self, mock_stream):
        events = [
            '{"type":"message_start","message":{"model":"claude-sonnet-4-20250514","usage":{"input_tokens":25,"output_tokens":0}}}',
            '{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}',
            '{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}',
        ]

        async def _fake_stream(*args, **kwargs):
            for e in events:
                yield e

        mock_stream.return_value = _fake_stream()

        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        chunks = []
        async for chunk in aiter:
            chunks.append(chunk)

        assert chunks == ["Hi"]
        assert state.usage is not None
        assert state.usage.input_tokens == 25
        assert state.usage.output_tokens == 10
        assert state.stop_reason == "end_turn"
        assert state.model == "claude-sonnet-4-20250514"

    @patch("ai_arch_toolkit._providers._anthropic.async_stream_sse")
    async def test_stream_passes_client(self, mock_stream):
        """Verify the provider passes its httpx client to async_stream_sse."""

        async def _fake_stream(*args, **kwargs):
            yield '{"type":"message_stop"}'

        mock_stream.return_value = _fake_stream()

        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        async for _ in aiter:
            pass
        call_kwargs = mock_stream.call_args[1]
        assert call_kwargs["client"] is provider._client


class TestAnthropicProviderLifecycle:
    async def test_close(self):
        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        assert provider._client is not None
        await provider.close()

    async def test_context_manager(self):
        async with AnthropicProvider("claude-sonnet-4-20250514", "test-key") as provider:
            assert provider._client is not None
