"""Tests for tool call streaming — Phase 4."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from ai_arch_toolkit.core._providers._anthropic import AnthropicProvider
from ai_arch_toolkit.core._providers._anthropic import (
    _StreamState as AnthropicStreamState,
)
from ai_arch_toolkit.core._providers._openai import OpenAIProvider
from ai_arch_toolkit.core._providers._openai import (
    _StreamState as OpenAIStreamState,
)
from ai_arch_toolkit.core._response import ToolCall, Usage

# ---------------------------------------------------------------------------
# Helpers — build SSE event strings via json.dumps (avoids long lines)
# ---------------------------------------------------------------------------


def _anth(obj: dict) -> str:
    return json.dumps(obj, separators=(",", ":"))


def _oai(obj: dict) -> str:
    return json.dumps(obj, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Anthropic stream tool calls
# ---------------------------------------------------------------------------


class TestAnthropicStreamToolCalls:
    @patch("ai_arch_toolkit.core._providers._anthropic.async_stream_sse")
    async def test_single_tool_call(self, mock_stream):
        events = [
            _anth(
                {
                    "type": "message_start",
                    "message": {
                        "model": "claude-sonnet-4-20250514",
                        "usage": {"input_tokens": 25, "output_tokens": 0},
                    },
                }
            ),
            _anth(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "tc_1",
                        "name": "get_weather",
                    },
                }
            ),
            _anth(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": '{"city"',
                    },
                }
            ),
            _anth(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": ': "NYC"}',
                    },
                }
            ),
            _anth({"type": "content_block_stop", "index": 0}),
            _anth(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                    "usage": {"output_tokens": 15},
                }
            ),
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

        assert chunks == []  # no text, only tool call
        assert len(state.tool_calls) == 1
        tc = state.tool_calls[0]
        assert isinstance(tc, ToolCall)
        assert tc.id == "tc_1"
        assert tc.name == "get_weather"
        assert tc.input == {"city": "NYC"}
        assert state.stop_reason == "tool_use"

    @patch("ai_arch_toolkit.core._providers._anthropic.async_stream_sse")
    async def test_text_then_tool_call(self, mock_stream):
        events = [
            _anth(
                {
                    "type": "message_start",
                    "message": {
                        "model": "claude-sonnet-4-20250514",
                        "usage": {"input_tokens": 10, "output_tokens": 0},
                    },
                }
            ),
            _anth(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }
            ),
            _anth(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Let me check."},
                }
            ),
            _anth({"type": "content_block_stop", "index": 0}),
            _anth(
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {
                        "type": "tool_use",
                        "id": "tc_1",
                        "name": "get_weather",
                    },
                }
            ),
            _anth(
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": '{"city": "NYC"}',
                    },
                }
            ),
            _anth({"type": "content_block_stop", "index": 1}),
            _anth(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                    "usage": {"output_tokens": 20},
                }
            ),
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

        assert chunks == ["Let me check."]
        assert len(state.tool_calls) == 1
        assert state.tool_calls[0].name == "get_weather"
        assert state.tool_calls[0].input == {"city": "NYC"}

    @patch("ai_arch_toolkit.core._providers._anthropic.async_stream_sse")
    async def test_multiple_tool_calls(self, mock_stream):
        events = [
            _anth(
                {
                    "type": "message_start",
                    "message": {
                        "model": "claude-sonnet-4-20250514",
                        "usage": {"input_tokens": 10, "output_tokens": 0},
                    },
                }
            ),
            _anth(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "tc_1",
                        "name": "get_weather",
                    },
                }
            ),
            _anth(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": '{"city": "NYC"}',
                    },
                }
            ),
            _anth({"type": "content_block_stop", "index": 0}),
            _anth(
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {
                        "type": "tool_use",
                        "id": "tc_2",
                        "name": "get_time",
                    },
                }
            ),
            _anth(
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": '{"tz": "UTC"}',
                    },
                }
            ),
            _anth({"type": "content_block_stop", "index": 1}),
            _anth(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                    "usage": {"output_tokens": 20},
                }
            ),
        ]

        async def _fake_stream(*args, **kwargs):
            for e in events:
                yield e

        mock_stream.return_value = _fake_stream()

        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        async for _ in aiter:
            pass

        assert len(state.tool_calls) == 2
        assert state.tool_calls[0].id == "tc_1"
        assert state.tool_calls[0].name == "get_weather"
        assert state.tool_calls[1].id == "tc_2"
        assert state.tool_calls[1].name == "get_time"
        assert state.tool_calls[1].input == {"tz": "UTC"}

    @patch("ai_arch_toolkit.core._providers._anthropic.async_stream_sse")
    async def test_malformed_tool_args(self, mock_stream):
        events = [
            _anth(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "tc_1",
                        "name": "fn",
                    },
                }
            ),
            _anth(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": "not valid json",
                    },
                }
            ),
            _anth({"type": "content_block_stop", "index": 0}),
        ]

        async def _fake_stream(*args, **kwargs):
            for e in events:
                yield e

        mock_stream.return_value = _fake_stream()

        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        async for _ in aiter:
            pass

        assert len(state.tool_calls) == 1
        assert state.tool_calls[0].input == {"_raw": "not valid json"}

    @patch("ai_arch_toolkit.core._providers._anthropic.async_stream_sse")
    async def test_tools_passed_in_payload(self, mock_stream):
        async def _fake_stream(*args, **kwargs):
            yield '{"type":"message_stop"}'

        mock_stream.return_value = _fake_stream()

        tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {"type": "object"},
            }
        ]
        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        aiter, _ = provider.stream([{"role": "user", "content": "Hi"}], tools=tools)
        async for _ in aiter:
            pass

        payload = mock_stream.call_args[1]["payload"]
        assert "tools" in payload
        assert payload["tools"][0]["name"] == "get_weather"

    @patch("ai_arch_toolkit.core._providers._anthropic.async_stream_sse")
    async def test_empty_tool_args(self, mock_stream):
        """Tool call with no arguments (empty partial_json)."""
        events = [
            _anth(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "tc_1",
                        "name": "get_status",
                    },
                }
            ),
            _anth({"type": "content_block_stop", "index": 0}),
        ]

        async def _fake_stream(*args, **kwargs):
            for e in events:
                yield e

        mock_stream.return_value = _fake_stream()

        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        async for _ in aiter:
            pass

        assert len(state.tool_calls) == 1
        assert state.tool_calls[0].input == {}


class TestAnthropicStreamState:
    def test_initial_state_has_tool_calls(self):
        state = AnthropicStreamState()
        assert state.tool_calls == []
        assert state.usage is None
        assert state.model == ""
        assert state.stop_reason == ""


# ---------------------------------------------------------------------------
# OpenAI stream tool calls
# ---------------------------------------------------------------------------


def _oai_tc_delta(
    index: int,
    *,
    tc_id: str = "",
    name: str = "",
    arguments: str = "",
    finish: str | None = None,
) -> str:
    """Build an OpenAI streaming chunk with a tool_calls delta."""
    tc: dict = {"index": index}
    if tc_id:
        tc["id"] = tc_id
    fn: dict = {}
    if name:
        fn["name"] = name
    fn["arguments"] = arguments
    tc["function"] = fn
    choice: dict = {
        "delta": {"tool_calls": [tc]},
        "finish_reason": finish,
    }
    return _oai({"choices": [choice], "model": "gpt-4o"})


class TestOpenAIStreamToolCalls:
    @patch("ai_arch_toolkit.core._providers._openai.async_stream_sse")
    async def test_single_tool_call(self, mock_stream):
        events = [
            _oai_tc_delta(0, tc_id="tc_1", name="get_weather"),
            _oai_tc_delta(0, arguments='{"city"'),
            _oai_tc_delta(0, arguments=': "NYC"}', finish="tool_calls"),
            _oai(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 25,
                        "completion_tokens": 15,
                    },
                }
            ),
        ]

        async def _fake_stream(*args, **kwargs):
            for e in events:
                yield e

        mock_stream.return_value = _fake_stream()

        provider = OpenAIProvider("gpt-4o", "test-key")
        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        chunks = []
        async for chunk in aiter:
            chunks.append(chunk)

        assert chunks == []  # no text, only tool call
        assert len(state.tool_calls) == 1
        tc = state.tool_calls[0]
        assert isinstance(tc, ToolCall)
        assert tc.id == "tc_1"
        assert tc.name == "get_weather"
        assert tc.input == {"city": "NYC"}
        assert state.stop_reason == "tool_calls"

    @patch("ai_arch_toolkit.core._providers._openai.async_stream_sse")
    async def test_multiple_tool_calls(self, mock_stream):
        events = [
            _oai_tc_delta(0, tc_id="tc_1", name="get_weather"),
            _oai_tc_delta(0, arguments='{"city": "NYC"}'),
            _oai_tc_delta(1, tc_id="tc_2", name="get_time"),
            _oai_tc_delta(1, arguments='{"tz": "UTC"}', finish="tool_calls"),
            _oai(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 30,
                        "completion_tokens": 20,
                    },
                }
            ),
        ]

        async def _fake_stream(*args, **kwargs):
            for e in events:
                yield e

        mock_stream.return_value = _fake_stream()

        provider = OpenAIProvider("gpt-4o", "test-key")
        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        async for _ in aiter:
            pass

        assert len(state.tool_calls) == 2
        assert state.tool_calls[0].id == "tc_1"
        assert state.tool_calls[0].name == "get_weather"
        assert state.tool_calls[0].input == {"city": "NYC"}
        assert state.tool_calls[1].id == "tc_2"
        assert state.tool_calls[1].name == "get_time"
        assert state.tool_calls[1].input == {"tz": "UTC"}

    @patch("ai_arch_toolkit.core._providers._openai.async_stream_sse")
    async def test_text_then_tool_call(self, mock_stream):
        events = [
            _oai(
                {
                    "choices": [
                        {
                            "delta": {"content": "Let me check."},
                            "finish_reason": None,
                        }
                    ],
                    "model": "gpt-4o",
                }
            ),
            _oai_tc_delta(
                0,
                tc_id="tc_1",
                name="get_weather",
                arguments='{"city": "NYC"}',
                finish="tool_calls",
            ),
            _oai(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 15,
                    },
                }
            ),
        ]

        async def _fake_stream(*args, **kwargs):
            for e in events:
                yield e

        mock_stream.return_value = _fake_stream()

        provider = OpenAIProvider("gpt-4o", "test-key")
        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        chunks = []
        async for chunk in aiter:
            chunks.append(chunk)

        assert chunks == ["Let me check."]
        assert len(state.tool_calls) == 1
        assert state.tool_calls[0].name == "get_weather"

    @patch("ai_arch_toolkit.core._providers._openai.async_stream_sse")
    async def test_tools_passed_in_payload(self, mock_stream):
        async def _fake_stream(*args, **kwargs):
            yield _oai(
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                }
            )

        mock_stream.return_value = _fake_stream()

        tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object"},
            }
        ]
        provider = OpenAIProvider("gpt-4o", "test-key")
        aiter, _ = provider.stream([{"role": "user", "content": "Hi"}], tools=tools)
        async for _ in aiter:
            pass

        payload = mock_stream.call_args[1]["payload"]
        assert "tools" in payload
        assert payload["tools"][0]["type"] == "function"
        assert payload["tools"][0]["function"]["name"] == "get_weather"


class TestOpenAIStreamState:
    def test_initial_state_has_tool_calls(self):
        state = OpenAIStreamState()
        assert state.tool_calls == []


# ---------------------------------------------------------------------------
# LLM-level stream tool calls
# ---------------------------------------------------------------------------


class TestLLMStreamToolCalls:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_stream_with_tools(self, mock_create):
        """StreamResponse.response.tool_calls after consumption."""

        async def _fake_gen():
            yield "Let me check."

        state = MagicMock()
        state.usage = Usage(input_tokens=10, output_tokens=5)
        state.model = "claude-sonnet-4-20250514"
        state.stop_reason = "tool_use"
        state.tool_calls = [
            ToolCall(
                id="tc_1",
                name="get_weather",
                input={"city": "NYC"},
            ),
        ]

        mock_provider = MagicMock()
        mock_provider.stream.return_value = (_fake_gen(), state)
        mock_create.return_value = mock_provider

        from ai_arch_toolkit.core._llm import LLM

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        stream = llm.stream("Hi", tools=[{"name": "get_weather"}])

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert chunks == ["Let me check."]
        assert stream.response is not None
        assert len(stream.response.tool_calls) == 1
        assert stream.response.tool_calls[0].name == "get_weather"
        assert stream.response.stop_reason == "tool_use"

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_stream_tools_forwarded_to_provider(self, mock_create):
        """Verify tools param is passed to provider.stream()."""

        async def _fake_gen():
            yield "Hi"

        state = MagicMock()
        state.usage = Usage(input_tokens=5, output_tokens=3)
        state.model = "claude-sonnet-4-20250514"
        state.stop_reason = "end_turn"
        state.tool_calls = []

        mock_provider = MagicMock()
        mock_provider.stream.return_value = (_fake_gen(), state)
        mock_create.return_value = mock_provider

        from ai_arch_toolkit.core._llm import LLM

        tools = [
            {
                "name": "search",
                "description": "Search",
                "input_schema": {"type": "object"},
            }
        ]
        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        stream = llm.stream("Hi", tools=tools)
        async for _ in stream:
            pass

        call_kwargs = mock_provider.stream.call_args[1]
        assert call_kwargs["tools"] == tools

    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_stream_sync_with_tools(self, mock_create):
        """SyncStreamResponse.response.tool_calls after consumption."""

        async def _fake_gen():
            yield "Checking."

        state = MagicMock()
        state.usage = Usage(input_tokens=10, output_tokens=5)
        state.model = "claude-sonnet-4-20250514"
        state.stop_reason = "tool_use"
        state.tool_calls = [
            ToolCall(id="tc_1", name="get_time", input={"tz": "UTC"}),
        ]

        mock_provider = MagicMock()
        mock_provider.stream.return_value = (_fake_gen(), state)
        mock_create.return_value = mock_provider

        from ai_arch_toolkit.core._llm import LLM

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        stream = llm.stream_sync("Hi", tools=[{"name": "get_time"}])

        chunks = list(stream)
        assert chunks == ["Checking."]
        assert stream.response is not None
        assert len(stream.response.tool_calls) == 1
        assert stream.response.tool_calls[0].name == "get_time"
        assert stream.response.tool_calls[0].input == {"tz": "UTC"}
