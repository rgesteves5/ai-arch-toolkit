"""Tests for _providers/_openai.py."""

from __future__ import annotations

import warnings
from unittest.mock import AsyncMock, patch

from ai_arch_toolkit.core._providers._openai import (
    OpenAIProvider,
    _build_payload,
    _messages_to_wire,
    _parse_response,
    _parse_tool_args,
    _tool_to_openai,
)
from ai_arch_toolkit.core._response import Response, ToolCall

# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestMessagesToWire:
    def test_system_as_regular_message(self):
        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ]
        wire = _messages_to_wire(msgs)
        assert wire[0] == {"role": "system", "content": "Be helpful."}
        assert wire[1] == {"role": "user", "content": "Hi"}

    def test_explicit_system_prepended(self):
        msgs = [{"role": "user", "content": "Hi"}]
        wire = _messages_to_wire(msgs, system="Be helpful.")
        assert wire[0] == {"role": "system", "content": "Be helpful."}
        assert wire[1] == {"role": "user", "content": "Hi"}

    def test_explicit_system_overrides_list_system(self):
        """Explicit system discards system messages from the list (same as Anthropic)."""
        msgs = [
            {"role": "system", "content": "From list."},
            {"role": "user", "content": "Hi"},
        ]
        wire = _messages_to_wire(msgs, system="Explicit.")
        system_msgs = [m for m in wire if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "Explicit."

    def test_multiple_system_messages_without_explicit(self):
        """Without explicit system, all system messages from list are kept."""
        msgs = [
            {"role": "system", "content": "First."},
            {"role": "system", "content": "Second."},
            {"role": "user", "content": "Hi"},
        ]
        wire = _messages_to_wire(msgs)
        system_msgs = [m for m in wire if m["role"] == "system"]
        assert len(system_msgs) == 2

    def test_tool_result_uses_role_tool(self):
        msgs = [{"role": "tool", "content": "42", "tool_use_id": "call_1"}]
        wire = _messages_to_wire(msgs)
        assert wire[0]["role"] == "tool"
        assert wire[0]["tool_call_id"] == "call_1"
        assert wire[0]["content"] == "42"

    def test_no_system(self):
        msgs = [{"role": "user", "content": "Hi"}]
        wire = _messages_to_wire(msgs)
        assert len(wire) == 1
        assert wire[0]["role"] == "user"

    def test_assistant_with_tool_calls(self):
        """Verify generic format is converted to OpenAI wire format."""
        msgs = [
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {"id": "tc_1", "name": "get_weather", "input": {"city": "NYC"}},
                ],
            },
        ]
        wire = _messages_to_wire(msgs)
        assert wire[0]["role"] == "assistant"
        assert wire[0]["content"] == "Let me check."
        tc = wire[0]["tool_calls"][0]
        assert tc["id"] == "tc_1"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == '{"city": "NYC"}'


class TestParseResponse:
    def test_text_response(self):
        raw = {
            "choices": [
                {
                    "message": {"content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "model": "gpt-4o",
        }
        r = _parse_response(raw, "gpt-4o")
        assert r.text == "Hello!"
        assert r.usage.input_tokens == 10
        assert r.usage.output_tokens == 5
        assert r.stop_reason == "stop"
        assert isinstance(r, Response)

    def test_tool_calls(self):
        raw = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tc_1",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "NYC"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15},
            "model": "gpt-4o",
        }
        r = _parse_response(raw, "gpt-4o")
        assert len(r.tool_calls) == 1
        assert isinstance(r.tool_calls[0], ToolCall)
        assert r.tool_calls[0].name == "get_weather"
        assert r.tool_calls[0].input == {"city": "NYC"}
        assert r.has_tool_calls

    def test_empty_choices(self):
        raw = {"choices": [], "usage": {}}
        r = _parse_response(raw, "gpt-4o")
        assert r.text == ""

    def test_cost_is_computed(self):
        raw = {
            "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
            "model": "gpt-4o",
        }
        r = _parse_response(raw, "gpt-4o")
        assert r.cost > 0
        assert r.cost_estimated is True


class TestToolToOpenai:
    def test_wraps_in_function(self):
        tool = {
            "name": "search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        }
        result = _tool_to_openai(tool)
        assert result["type"] == "function"
        assert result["function"]["name"] == "search"
        assert result["function"]["description"] == "Search the web"
        assert result["function"]["parameters"] == tool["parameters"]

    def test_accepts_input_schema_key(self):
        tool = {
            "name": "search",
            "description": "Search",
            "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
        }
        result = _tool_to_openai(tool)
        assert result["function"]["parameters"] == tool["input_schema"]

    def test_prefers_input_schema_over_parameters(self):
        """When both keys present, input_schema wins (canonical format)."""
        tool = {
            "name": "fn",
            "description": "desc",
            "input_schema": {"type": "object", "properties": {"a": {"type": "string"}}},
            "parameters": {"type": "object", "properties": {"b": {"type": "integer"}}},
        }
        result = _tool_to_openai(tool)
        assert "a" in result["function"]["parameters"]["properties"]
        assert "b" not in result["function"]["parameters"]["properties"]


class TestParseToolArgs:
    def test_json_string(self):
        result = _parse_tool_args('{"city": "NYC"}')
        assert result == {"city": "NYC"}

    def test_dict_passthrough(self):
        d = {"city": "NYC"}
        assert _parse_tool_args(d) is d

    def test_invalid_json(self):
        result = _parse_tool_args("not json")
        assert result == {"_raw": "not json"}


class TestBuildPayload:
    def test_basic_payload(self):
        msgs = [{"role": "user", "content": "Hi"}]
        payload = _build_payload(msgs, model="gpt-4o")
        assert payload["model"] == "gpt-4o"
        assert payload["messages"] == msgs

    def test_with_tools(self):
        msgs = [{"role": "user", "content": "Hi"}]
        tools = [{"name": "search", "description": "Search", "parameters": {"type": "object"}}]
        payload = _build_payload(msgs, model="gpt-4o", tools=tools)
        assert "tools" in payload
        assert payload["tools"][0]["type"] == "function"

    def test_unknown_kwargs_warn(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _build_payload(
                [{"role": "user", "content": "Hi"}],
                model="gpt-4o",
                typo_param=True,
            )
            assert len(w) == 1
            assert "typo_param" in str(w[0].message)

    def test_known_kwargs_no_warn(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _build_payload(
                [{"role": "user", "content": "Hi"}],
                model="gpt-4o",
                temperature=0.5,
                top_p=0.9,
            )
            assert len(w) == 0


# ---------------------------------------------------------------------------
# Provider integration tests (mocked HTTP)
# ---------------------------------------------------------------------------


class TestOpenAIProviderComplete:
    @patch("ai_arch_toolkit.core._providers._openai.async_post_json", new_callable=AsyncMock)
    async def test_complete(self, mock_post):
        mock_post.return_value = {
            "choices": [
                {
                    "message": {"content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "model": "gpt-4o",
        }
        provider = OpenAIProvider("gpt-4o", "test-key")
        result = await provider.complete([{"role": "user", "content": "Hi"}])
        assert result.text == "Hello!"
        assert isinstance(result, Response)
        mock_post.assert_called_once()

    @patch("ai_arch_toolkit.core._providers._openai.async_post_json", new_callable=AsyncMock)
    async def test_complete_passes_client(self, mock_post):
        mock_post.return_value = {
            "choices": [{"message": {"content": "Ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
        provider = OpenAIProvider("gpt-4o", "test-key")
        await provider.complete([{"role": "user", "content": "Hi"}])
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["client"] is provider._client

    @patch("ai_arch_toolkit.core._providers._openai.async_post_json", new_callable=AsyncMock)
    async def test_complete_with_tools(self, mock_post):
        mock_post.return_value = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tc_1",
                                "function": {"name": "search", "arguments": '{"q": "test"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        }
        tools = [{"name": "search", "description": "Search", "parameters": {"type": "object"}}]
        provider = OpenAIProvider("gpt-4o", "test-key")
        result = await provider.complete(
            [{"role": "user", "content": "Hi"}],
            tools=tools,
        )
        assert result.has_tool_calls

        payload = mock_post.call_args[1]["payload"]
        assert "tools" in payload
        assert payload["tools"][0]["type"] == "function"

    @patch("ai_arch_toolkit.core._providers._openai.async_post_json", new_callable=AsyncMock)
    async def test_system_passed_as_message(self, mock_post):
        mock_post.return_value = {
            "choices": [{"message": {"content": "Ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
        provider = OpenAIProvider("gpt-4o", "test-key")
        await provider.complete(
            [{"role": "user", "content": "Hi"}],
            system="Be brief.",
        )
        payload = mock_post.call_args[1]["payload"]
        # System should be first message
        assert payload["messages"][0] == {"role": "system", "content": "Be brief."}
        assert payload["messages"][1] == {"role": "user", "content": "Hi"}

    @patch("ai_arch_toolkit.core._providers._openai.async_post_json", new_callable=AsyncMock)
    async def test_explicit_system_overrides_list_system(self, mock_post):
        mock_post.return_value = {
            "choices": [{"message": {"content": "Ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
        provider = OpenAIProvider("gpt-4o", "test-key")
        msgs = [
            {"role": "system", "content": "From list."},
            {"role": "user", "content": "Hi"},
        ]
        await provider.complete(msgs, system="Explicit.")
        payload = mock_post.call_args[1]["payload"]
        system_msgs = [m for m in payload["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "Explicit."


class TestOpenAIProviderStream:
    @patch("ai_arch_toolkit.core._providers._openai.async_stream_sse")
    async def test_stream_text_deltas(self, mock_stream):
        events = [
            '{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}],"model":"gpt-4o"}',
            '{"choices":[{"delta":{"content":" world"},"finish_reason":null}],"model":"gpt-4o"}',
            '{"choices":[{"delta":{},"finish_reason":"stop"}],"model":"gpt-4o"}',
        ]

        async def _fake_stream(*args, **kwargs):
            for e in events:
                yield e

        mock_stream.return_value = _fake_stream()

        provider = OpenAIProvider("gpt-4o", "test-key")
        aiter, _state = provider.stream([{"role": "user", "content": "Hi"}])
        chunks = []
        async for chunk in aiter:
            chunks.append(chunk)
        assert chunks == ["Hello", " world"]

    @patch("ai_arch_toolkit.core._providers._openai.async_stream_sse")
    async def test_stream_captures_usage(self, mock_stream):
        events = [
            '{"choices":[{"delta":{"content":"Hi"},"finish_reason":null}],"model":"gpt-4o"}',
            '{"choices":[{"delta":{},"finish_reason":"stop"}],"model":"gpt-4o"}',
            '{"choices":[],"usage":{"prompt_tokens":25,"completion_tokens":10},"model":"gpt-4o"}',
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

        assert chunks == ["Hi"]
        assert state.usage is not None
        assert state.usage.input_tokens == 25
        assert state.usage.output_tokens == 10
        assert state.stop_reason == "stop"

    @patch("ai_arch_toolkit.core._providers._openai.async_stream_sse")
    async def test_stream_includes_usage_option(self, mock_stream):
        """Verify stream payload includes stream_options for usage."""

        async def _fake_stream(*args, **kwargs):
            yield '{"choices":[{"delta":{},"finish_reason":"stop"}]}'

        mock_stream.return_value = _fake_stream()

        provider = OpenAIProvider("gpt-4o", "test-key")
        aiter, _ = provider.stream([{"role": "user", "content": "Hi"}])
        async for _ in aiter:
            pass

        payload = mock_stream.call_args[1]["payload"]
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}

    @patch("ai_arch_toolkit.core._providers._openai.async_stream_sse")
    async def test_stream_passes_client(self, mock_stream):

        async def _fake_stream(*args, **kwargs):
            yield '{"choices":[{"delta":{},"finish_reason":"stop"}]}'

        mock_stream.return_value = _fake_stream()

        provider = OpenAIProvider("gpt-4o", "test-key")
        aiter, _ = provider.stream([{"role": "user", "content": "Hi"}])
        async for _ in aiter:
            pass
        call_kwargs = mock_stream.call_args[1]
        assert call_kwargs["client"] is provider._client


class TestOpenAIProviderLifecycle:
    async def test_close(self):
        provider = OpenAIProvider("gpt-4o", "test-key")
        assert provider._client is not None
        await provider.close()

    async def test_context_manager(self):
        async with OpenAIProvider("gpt-4o", "test-key") as provider:
            assert provider._client is not None
