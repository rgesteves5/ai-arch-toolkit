"""Tests for _providers/_xai.py — xAI SDK adapter (gRPC)."""

from __future__ import annotations

import json
import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest

from ai_arch_toolkit.core._exceptions import APIError, RateLimitError
from ai_arch_toolkit.core._providers._xai import (
    XAIProvider,
    _build_response_format,
    _extract_usage,
    _grpc_code_to_http,
    _is_multi_agent_model,
    _messages_to_sdk,
    _parse_sdk_response,
    _tool_to_sdk,
)
from ai_arch_toolkit.core._response import OutputSchema, Response

# ---------------------------------------------------------------------------
# Helpers — build fake SDK objects
# ---------------------------------------------------------------------------


def _sdk_tool_call(tc_id="tc_1", name="get_weather", arguments='{"city": "NYC"}'):
    """Build a fake ToolCall proto-like object."""
    func = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(id=tc_id, type="function", function=func)


def _sdk_usage(prompt_tokens=10, completion_tokens=5, cached_prompt_text_tokens=0):
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_prompt_text_tokens=cached_prompt_text_tokens,
        reasoning_tokens=0,
    )


def _sdk_response(
    content="Hello",
    finish_reason="FINISH_REASON_STOP",
    tool_calls=None,
    reasoning_content="",
    usage=None,
):
    """Build a fake xAI SDK Response-like object."""
    return SimpleNamespace(
        content=content,
        finish_reason=finish_reason,
        tool_calls=tool_calls or [],
        reasoning_content=reasoning_content,
        usage=usage or _sdk_usage(),
    )


def _sdk_chunk(content="", reasoning_content="", tool_calls=None):
    """Build a fake xAI SDK Chunk-like object."""
    return SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls or [],
    )


# ---------------------------------------------------------------------------
# _messages_to_sdk
# ---------------------------------------------------------------------------


class TestMessagesToSdk:
    def test_simple_user_message(self):
        msgs, sys = _messages_to_sdk([{"role": "user", "content": "Hello"}])
        assert sys is None
        assert len(msgs) == 1

    def test_system_extracted(self):
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ]
        msgs, sys = _messages_to_sdk(messages)
        assert sys == "Be helpful."
        assert len(msgs) == 1  # only user message

    def test_multiple_system_joined(self):
        messages = [
            {"role": "system", "content": "First."},
            {"role": "system", "content": "Second."},
            {"role": "user", "content": "Hi"},
        ]
        _, sys = _messages_to_sdk(messages)
        assert sys == "First.\n\nSecond."

    def test_assistant_message(self):
        msgs, _ = _messages_to_sdk([{"role": "assistant", "content": "Hi there"}])
        assert len(msgs) == 1

    def test_tool_result(self):
        messages = [
            {"role": "tool", "tool_use_id": "tc_1", "content": "Sunny in NYC"},
        ]
        msgs, _ = _messages_to_sdk(messages)
        assert len(msgs) == 1

    def test_assistant_with_tool_calls(self):
        messages = [
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {"id": "tc_1", "name": "get_weather", "input": {"city": "NYC"}},
                ],
            }
        ]
        msgs, _ = _messages_to_sdk(messages)
        assert len(msgs) == 1
        # Should be a proto Message with tool_calls
        assert len(msgs[0].tool_calls) == 1
        assert msgs[0].tool_calls[0].function.name == "get_weather"


# ---------------------------------------------------------------------------
# _tool_to_sdk
# ---------------------------------------------------------------------------


class TestToolToSdk:
    def test_basic(self):
        tool = {
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
        sdk_tool = _tool_to_sdk(tool)
        assert sdk_tool.function.name == "get_weather"

    def test_parameters_key_fallback(self):
        tool = {"name": "fn", "description": "Do something", "parameters": {"type": "object"}}
        sdk_tool = _tool_to_sdk(tool)
        assert sdk_tool.function.name == "fn"


# ---------------------------------------------------------------------------
# _build_response_format
# ---------------------------------------------------------------------------


class TestBuildResponseFormat:
    def test_creates_json_schema_format(self):
        schema = OutputSchema(name="Person", schema={"type": "object"})
        rf = _build_response_format(schema)
        assert rf.format_type == 3  # FORMAT_TYPE_JSON_SCHEMA
        assert json.loads(rf.schema) == {"type": "object"}


# ---------------------------------------------------------------------------
# _extract_usage
# ---------------------------------------------------------------------------


class TestExtractUsage:
    def test_basic(self):
        usage = _extract_usage(_sdk_usage(prompt_tokens=100, completion_tokens=50))
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_cached_tokens(self):
        usage = _extract_usage(_sdk_usage(cached_prompt_text_tokens=20))
        assert usage.cache_read_tokens == 20

    def test_input_output_tokens_fields(self):
        """xAI SDK may use input_tokens/output_tokens field names."""
        usage_obj = SimpleNamespace(
            input_tokens=100, output_tokens=50, cached_prompt_text_tokens=0
        )
        usage = _extract_usage(usage_obj)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50


# ---------------------------------------------------------------------------
# _parse_sdk_response
# ---------------------------------------------------------------------------


class TestParseSdkResponse:
    def test_text_response(self):
        resp = _sdk_response(content="Hello world")
        r = _parse_sdk_response(resp, "grok-3")
        assert isinstance(r, Response)
        assert r.text == "Hello world"
        assert r.model == "grok-3"

    def test_tool_calls(self):
        tc = _sdk_tool_call()
        resp = _sdk_response(content="", tool_calls=[tc])
        r = _parse_sdk_response(resp, "grok-3")
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "get_weather"
        assert r.tool_calls[0].input == {"city": "NYC"}

    def test_reasoning_content(self):
        resp = _sdk_response(content="42", reasoning_content="Let me think step by step...")
        r = _parse_sdk_response(resp, "grok-3")
        assert len(r.thinking) == 1
        assert r.thinking[0].text == "Let me think step by step..."
        assert r.text == "42"

    def test_no_reasoning(self):
        resp = _sdk_response(content="Hello", reasoning_content="")
        r = _parse_sdk_response(resp, "grok-3")
        assert r.thinking == ()

    def test_structured_output_parsed(self):
        schema = OutputSchema(name="Person", schema={"type": "object"})
        resp = _sdk_response(content='{"name": "Alice"}')
        r = _parse_sdk_response(resp, "grok-3", output_schema=schema)
        assert r.parsed == {"name": "Alice"}

    def test_raw_is_preserved(self):
        resp = _sdk_response()
        r = _parse_sdk_response(resp, "grok-3")
        assert r.raw is resp

    def test_finish_reason(self):
        resp = _sdk_response(finish_reason="FINISH_REASON_STOP")
        r = _parse_sdk_response(resp, "grok-3")
        assert r.stop_reason == "FINISH_REASON_STOP"


# ---------------------------------------------------------------------------
# _grpc_code_to_http
# ---------------------------------------------------------------------------


class TestGrpcCodeToHttp:
    def test_resource_exhausted(self):
        assert _grpc_code_to_http(grpc.StatusCode.RESOURCE_EXHAUSTED) == 429

    def test_invalid_argument(self):
        assert _grpc_code_to_http(grpc.StatusCode.INVALID_ARGUMENT) == 400

    def test_internal(self):
        assert _grpc_code_to_http(grpc.StatusCode.INTERNAL) == 500

    def test_unknown_defaults_500(self):
        assert _grpc_code_to_http(grpc.StatusCode.UNKNOWN) == 500

    def test_cancelled(self):
        assert _grpc_code_to_http(grpc.StatusCode.CANCELLED) == 499

    def test_aborted(self):
        assert _grpc_code_to_http(grpc.StatusCode.ABORTED) == 409

    def test_failed_precondition(self):
        assert _grpc_code_to_http(grpc.StatusCode.FAILED_PRECONDITION) == 412


class TestModelHelpers:
    def test_is_multi_agent_model(self):
        assert _is_multi_agent_model("grok-4.20-multi-agent")
        assert not _is_multi_agent_model("grok-4.20-reasoning")


# ---------------------------------------------------------------------------
# Provider integration tests (mocked SDK client)
# ---------------------------------------------------------------------------


def _make_mock_chat(response=None):
    """Create a mock chat object with sample/stream methods."""
    mock_chat = MagicMock()
    mock_chat.sample = AsyncMock(return_value=response or _sdk_response())
    return mock_chat


class TestXAIProviderComplete:
    async def test_complete(self):
        mock_chat = _make_mock_chat()
        mock_client = MagicMock()
        mock_client.chat.create.return_value = mock_chat

        provider = XAIProvider("grok-3", "test-key")
        provider._client = mock_client

        result = await provider.complete([{"role": "user", "content": "Hi"}])
        assert isinstance(result, Response)
        assert result.text == "Hello"
        mock_client.chat.create.assert_called_once()

    async def test_system_forwarded(self):
        mock_chat = _make_mock_chat()
        mock_client = MagicMock()
        mock_client.chat.create.return_value = mock_chat

        provider = XAIProvider("grok-3", "test-key")
        provider._client = mock_client
        await provider.complete(
            [{"role": "user", "content": "Hi"}],
            system="Be helpful.",
        )
        call_kwargs = mock_client.chat.create.call_args[1]
        # System should be first message
        msgs = call_kwargs["messages"]
        assert len(msgs) >= 2  # system + user

    async def test_tools_forwarded(self):
        mock_chat = _make_mock_chat()
        mock_client = MagicMock()
        mock_client.chat.create.return_value = mock_chat

        tools = [{"name": "get_weather", "description": "Get weather", "parameters": {}}]
        provider = XAIProvider("grok-3", "test-key")
        provider._client = mock_client
        await provider.complete([{"role": "user", "content": "Hi"}], tools=tools)
        call_kwargs = mock_client.chat.create.call_args[1]
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 1

    async def test_reasoning_effort_ignored(self):
        mock_chat = _make_mock_chat()
        mock_client = MagicMock()
        mock_client.chat.create.return_value = mock_chat

        provider = XAIProvider("grok-4.20-reasoning", "test-key")
        provider._client = mock_client
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await provider.complete(
                [{"role": "user", "content": "Hi"}],
                thinking=True,
                thinking_effort="high",
            )
        call_kwargs = mock_client.chat.create.call_args[1]
        assert "reasoning_effort" not in call_kwargs
        assert any("reason automatically" in str(warning.message) for warning in w)

    async def test_thinking_true_is_ignored(self):
        mock_chat = _make_mock_chat()
        mock_client = MagicMock()
        mock_client.chat.create.return_value = mock_chat

        provider = XAIProvider("grok-4.20-reasoning", "test-key")
        provider._client = mock_client
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await provider.complete(
                [{"role": "user", "content": "Hi"}],
                thinking=True,
            )
        call_kwargs = mock_client.chat.create.call_args[1]
        assert "reasoning_effort" not in call_kwargs
        assert any("reason automatically" in str(warning.message) for warning in w)

    async def test_multi_agent_agent_count_forwarded_and_max_tokens_dropped(self):
        mock_chat = _make_mock_chat()
        mock_client = MagicMock()
        mock_client.chat.create.return_value = mock_chat

        provider = XAIProvider("grok-4.20-multi-agent", "test-key")
        provider._client = mock_client
        await provider.complete(
            [{"role": "user", "content": "Hi"}],
            agent_count=4,
            max_tokens=64,
        )
        call_kwargs = mock_client.chat.create.call_args[1]
        assert call_kwargs["agent_count"] == 4
        assert "max_tokens" not in call_kwargs

    async def test_multi_agent_thinking_effort_maps_to_agent_count(self):
        mock_chat = _make_mock_chat()
        mock_client = MagicMock()
        mock_client.chat.create.return_value = mock_chat

        provider = XAIProvider("grok-4.20-multi-agent", "test-key")
        provider._client = mock_client
        await provider.complete(
            [{"role": "user", "content": "Hi"}],
            thinking=True,
            thinking_effort="medium",
        )
        call_kwargs = mock_client.chat.create.call_args[1]
        assert call_kwargs["agent_count"] == 4
        assert "reasoning_effort" not in call_kwargs

    async def test_output_schema_forwarded(self):
        mock_chat = _make_mock_chat(response=_sdk_response(content='{"name": "Alice"}'))
        mock_client = MagicMock()
        mock_client.chat.create.return_value = mock_chat

        schema = OutputSchema(name="Person", schema={"type": "object"})
        provider = XAIProvider("grok-3", "test-key")
        provider._client = mock_client
        result = await provider.complete(
            [{"role": "user", "content": "Hi"}],
            output_schema=schema,
        )
        call_kwargs = mock_client.chat.create.call_args[1]
        assert "response_format" in call_kwargs
        assert result.parsed == {"name": "Alice"}

    async def test_unknown_kwargs_warn(self):
        mock_chat = _make_mock_chat()
        mock_client = MagicMock()
        mock_client.chat.create.return_value = mock_chat

        provider = XAIProvider("grok-3", "test-key")
        provider._client = mock_client
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await provider.complete(
                [{"role": "user", "content": "Hi"}],
                typo_param=True,
            )
            assert len(w) == 1
            assert "typo_param" in str(w[0].message)

    async def test_known_kwargs_no_warn(self):
        mock_chat = _make_mock_chat()
        mock_client = MagicMock()
        mock_client.chat.create.return_value = mock_chat

        provider = XAIProvider("grok-3", "test-key")
        provider._client = mock_client
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await provider.complete(
                [{"role": "user", "content": "Hi"}],
                temperature=0.5,
                top_p=0.9,
            )
            assert len(w) == 0


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


class TestXAIProviderErrors:
    async def test_rate_limit_error(self):
        mock_chat = MagicMock()
        error = grpc.aio.AioRpcError(
            code=grpc.StatusCode.RESOURCE_EXHAUSTED,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(),
            details="rate limited",
        )
        mock_chat.sample = AsyncMock(side_effect=error)
        mock_client = MagicMock()
        mock_client.chat.create.return_value = mock_chat

        provider = XAIProvider("grok-3", "test-key")
        provider._client = mock_client
        with pytest.raises(RateLimitError) as exc_info:
            await provider.complete([{"role": "user", "content": "Hi"}])
        assert exc_info.value.status_code == 429

    async def test_api_error(self):
        mock_chat = MagicMock()
        error = grpc.aio.AioRpcError(
            code=grpc.StatusCode.INVALID_ARGUMENT,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(),
            details="bad request",
        )
        mock_chat.sample = AsyncMock(side_effect=error)
        mock_client = MagicMock()
        mock_client.chat.create.return_value = mock_chat

        provider = XAIProvider("grok-3", "test-key")
        provider._client = mock_client
        with pytest.raises(APIError) as exc_info:
            await provider.complete([{"role": "user", "content": "Hi"}])
        assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestXAIProviderStream:
    async def test_stream_text(self):
        response = _sdk_response(finish_reason="FINISH_REASON_STOP")
        chunks = [
            (response, _sdk_chunk(content="Hello ")),
            (response, _sdk_chunk(content="world")),
        ]

        mock_chat = MagicMock()

        async def _fake_stream():
            for item in chunks:
                yield item

        mock_chat.stream = _fake_stream
        mock_client = MagicMock()
        mock_client.chat.create.return_value = mock_chat

        provider = XAIProvider("grok-3", "test-key")
        provider._client = mock_client
        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        text_chunks = []
        async for chunk in aiter:
            text_chunks.append(chunk)

        assert text_chunks == ["Hello ", "world"]
        assert state.stop_reason == "FINISH_REASON_STOP"

    async def test_stream_tool_call(self):
        tc = _sdk_tool_call()
        response = _sdk_response(finish_reason="FINISH_REASON_TOOL_CALLS")
        chunks = [
            (response, _sdk_chunk(tool_calls=[tc])),
        ]

        mock_chat = MagicMock()

        async def _fake_stream():
            for item in chunks:
                yield item

        mock_chat.stream = _fake_stream
        mock_client = MagicMock()
        mock_client.chat.create.return_value = mock_chat

        provider = XAIProvider("grok-3", "test-key")
        provider._client = mock_client
        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        async for _ in aiter:
            pass

        assert len(state.tool_calls) == 1
        assert state.tool_calls[0].name == "get_weather"

    async def test_stream_reasoning(self):
        response = _sdk_response(finish_reason="FINISH_REASON_STOP")
        chunks = [
            (response, _sdk_chunk(reasoning_content="Let me think...")),
            (response, _sdk_chunk(content="The answer is 42.")),
        ]

        mock_chat = MagicMock()

        async def _fake_stream():
            for item in chunks:
                yield item

        mock_chat.stream = _fake_stream
        mock_client = MagicMock()
        mock_client.chat.create.return_value = mock_chat

        provider = XAIProvider("grok-3", "test-key")
        provider._client = mock_client
        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        text_chunks = []
        async for chunk in aiter:
            text_chunks.append(chunk)

        assert text_chunks == ["The answer is 42."]
        assert len(state.thinking) == 1
        assert state.thinking[0].text == "Let me think..."

    async def test_stream_usage(self):
        response = _sdk_response(usage=_sdk_usage(prompt_tokens=25, completion_tokens=10))
        chunks = [
            (response, _sdk_chunk(content="Hi")),
        ]

        mock_chat = MagicMock()

        async def _fake_stream():
            for item in chunks:
                yield item

        mock_chat.stream = _fake_stream
        mock_client = MagicMock()
        mock_client.chat.create.return_value = mock_chat

        provider = XAIProvider("grok-3", "test-key")
        provider._client = mock_client
        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        async for _ in aiter:
            pass

        assert state.usage is not None
        assert state.usage.input_tokens == 25
        assert state.usage.output_tokens == 10

    async def test_stream_error_mapping(self):
        error = grpc.aio.AioRpcError(
            code=grpc.StatusCode.RESOURCE_EXHAUSTED,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(),
            details="rate limited",
        )

        mock_chat = MagicMock()

        async def _error_stream():
            raise error
            yield  # makes this an async generator

        mock_chat.stream = _error_stream
        mock_client = MagicMock()
        mock_client.chat.create.return_value = mock_chat

        provider = XAIProvider("grok-3", "test-key")
        provider._client = mock_client
        aiter, _state = provider.stream([{"role": "user", "content": "Hi"}])
        with pytest.raises(RateLimitError):
            async for _ in aiter:
                pass
