"""Tests for _providers/_gemini.py — SDK adapter."""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_arch_toolkit.core._exceptions import APIError, RateLimitError
from ai_arch_toolkit.core._providers._gemini import (
    GeminiProvider,
    _build_thinking_config,
    _extract_usage,
    _messages_to_sdk,
    _parse_sdk_response,
    _tool_to_sdk,
)
from ai_arch_toolkit.core._response import OutputSchema, Response

# ---------------------------------------------------------------------------
# Helpers — build fake SDK objects
# ---------------------------------------------------------------------------


def _sdk_part(*, text=None, thought=False, function_call=None, function_response=None):
    """Build a fake Part-like object."""
    p = SimpleNamespace(text=text, thought=thought, function_call=function_call)
    if function_response is not None:
        p.function_response = function_response
    return p


def _sdk_candidate(parts=None, finish_reason=None):
    content = SimpleNamespace(parts=parts or [])
    return SimpleNamespace(content=content, finish_reason=finish_reason)


def _sdk_response(
    text="Hello",
    parts=None,
    finish_reason="STOP",
    prompt_tokens=10,
    candidates_tokens=5,
    cached_tokens=0,
    tool_calls=None,
    thinking_parts=None,
):
    """Build a fake GenerateContentResponse."""
    all_parts = []
    if thinking_parts:
        for t in thinking_parts:
            all_parts.append(_sdk_part(text=t, thought=True))
    if text:
        all_parts.append(_sdk_part(text=text))
    if tool_calls:
        for tc in tool_calls:
            fc = SimpleNamespace(
                id=tc.get("id", ""),
                name=tc["name"],
                args=tc.get("args", {}),
            )
            all_parts.append(_sdk_part(function_call=fc))
    if parts:
        all_parts = parts

    candidate = _sdk_candidate(parts=all_parts, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_token_count=prompt_tokens,
        candidates_token_count=candidates_tokens,
        cached_content_token_count=cached_tokens,
    )
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage)


def _sdk_stream_chunk(
    text=None,
    thought_text=None,
    function_call=None,
    finish_reason=None,
    prompt_tokens=0,
    candidates_tokens=0,
):
    """Build a fake streaming chunk."""
    parts = []
    if thought_text:
        parts.append(_sdk_part(text=thought_text, thought=True))
    if text:
        parts.append(_sdk_part(text=text))
    if function_call:
        parts.append(_sdk_part(function_call=function_call))

    candidate = _sdk_candidate(parts=parts, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_token_count=prompt_tokens,
        candidates_token_count=candidates_tokens,
        cached_content_token_count=0,
    )
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage)


# ---------------------------------------------------------------------------
# _messages_to_sdk
# ---------------------------------------------------------------------------


class TestMessagesToSdk:
    def test_simple_user_message(self):
        sys, contents = _messages_to_sdk([{"role": "user", "content": "Hello"}])
        assert sys is None
        assert len(contents) == 1
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "Hello"

    def test_system_extracted(self):
        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ]
        sys, contents = _messages_to_sdk(msgs)
        assert sys == "Be helpful."
        assert len(contents) == 1
        assert contents[0].role == "user"

    def test_multiple_system_joined(self):
        msgs = [
            {"role": "system", "content": "First."},
            {"role": "system", "content": "Second."},
            {"role": "user", "content": "Hi"},
        ]
        sys, _contents = _messages_to_sdk(msgs)
        assert sys == "First.\n\nSecond."

    def test_assistant_mapped_to_model(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        _, contents = _messages_to_sdk(msgs)
        assert contents[1].role == "model"
        assert contents[1].parts[0].text == "Hello"

    def test_assistant_with_tool_calls(self):
        msgs = [
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [{"name": "get_weather", "input": {"city": "NYC"}}],
            }
        ]
        _, contents = _messages_to_sdk(msgs)
        assert contents[0].role == "model"
        assert contents[0].parts[0].text == "Let me check."
        assert contents[0].parts[1].function_call.name == "get_weather"
        assert contents[0].parts[1].function_call.args == {"city": "NYC"}

    def test_tool_result_as_function_response(self):
        msgs = [
            {
                "role": "tool",
                "tool_use_id": "tc_1",
                "name": "get_weather",
                "content": '{"temp": 72}',
            }
        ]
        _, contents = _messages_to_sdk(msgs)
        assert contents[0].role == "user"
        fr = contents[0].parts[0].function_response
        assert fr.name == "get_weather"
        assert fr.response == {"temp": 72}

    def test_tool_result_non_json_wrapped(self):
        msgs = [{"role": "tool", "tool_use_id": "tc_1", "name": "fn", "content": "plain text"}]
        _, contents = _messages_to_sdk(msgs)
        fr = contents[0].parts[0].function_response
        assert fr.response == {"result": "plain text"}


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
        fd = _tool_to_sdk(tool)
        assert fd.name == "get_weather"
        assert fd.description == "Get weather"
        # SDK auto-converts dict to types.Schema
        assert fd.parameters is not None
        assert "city" in fd.parameters.properties

    def test_parameters_key_fallback(self):
        tool = {"name": "fn", "parameters": {"type": "object"}}
        fd = _tool_to_sdk(tool)
        assert fd.parameters is not None


# ---------------------------------------------------------------------------
# _build_thinking_config
# ---------------------------------------------------------------------------


class TestBuildThinkingConfig:
    def test_disabled(self):
        assert _build_thinking_config(False, None, None, "") is None

    def test_default_budget(self):
        cfg = _build_thinking_config(True, None, None, "gemini-2.5-flash")
        assert cfg.include_thoughts is True
        assert cfg.thinking_budget == 10000

    def test_explicit_budget(self):
        cfg = _build_thinking_config(True, None, 5000, "gemini-2.5-flash")
        assert cfg.thinking_budget == 5000

    def test_effort_maps_to_budget(self):
        cfg = _build_thinking_config(True, "low", None, "gemini-2.5-flash")
        assert cfg.thinking_budget == 2048
        cfg = _build_thinking_config(True, "high", None, "gemini-2.5-flash")
        assert cfg.thinking_budget == 10000

    def test_gemini3_uses_thinking_level(self):
        cfg = _build_thinking_config(True, "medium", None, "gemini-3-flash")
        # SDK auto-converts string to ThinkingLevel enum
        assert "MEDIUM" in str(cfg.thinking_level)
        assert cfg.thinking_budget is None


# ---------------------------------------------------------------------------
# _extract_usage
# ---------------------------------------------------------------------------


class TestExtractUsage:
    def test_basic(self):
        meta = SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=50,
            cached_content_token_count=10,
        )
        usage = _extract_usage(meta)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cache_read_tokens == 10

    def test_none_values(self):
        meta = SimpleNamespace(
            prompt_token_count=None,
            candidates_token_count=None,
            cached_content_token_count=None,
        )
        usage = _extract_usage(meta)
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0


# ---------------------------------------------------------------------------
# _parse_sdk_response
# ---------------------------------------------------------------------------


class TestParseSdkResponse:
    def test_text_response(self):
        resp = _sdk_response(text="Hello world")
        r = _parse_sdk_response(resp, "gemini-2.0-flash")
        assert isinstance(r, Response)
        assert r.text == "Hello world"
        assert r.model == "gemini-2.0-flash"

    def test_tool_calls(self):
        resp = _sdk_response(
            text="",
            tool_calls=[{"name": "get_weather", "args": {"city": "NYC"}}],
        )
        r = _parse_sdk_response(resp, "gemini-2.0-flash")
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "get_weather"
        assert r.tool_calls[0].input == {"city": "NYC"}

    def test_thinking_blocks(self):
        resp = _sdk_response(text="Answer.", thinking_parts=["Let me think..."])
        r = _parse_sdk_response(resp, "gemini-2.0-flash")
        assert len(r.thinking) == 1
        assert r.thinking[0].text == "Let me think..."
        assert r.text == "Answer."

    def test_empty_candidates(self):
        resp = SimpleNamespace(candidates=[], usage_metadata=None)
        r = _parse_sdk_response(resp, "gemini-2.0-flash")
        assert r.text == ""

    def test_structured_output_parsed(self):
        schema = OutputSchema(name="Person", schema={"type": "object"})
        resp = _sdk_response(text='{"name": "Alice"}')
        r = _parse_sdk_response(resp, "gemini-2.0-flash", output_schema=schema)
        assert r.parsed == {"name": "Alice"}

    def test_cost_is_computed(self):
        resp = _sdk_response(prompt_tokens=1000, candidates_tokens=500)
        r = _parse_sdk_response(resp, "gemini-2.0-flash")
        assert r.cost is not None
        assert r.cost > 0

    def test_raw_is_preserved(self):
        resp = _sdk_response()
        r = _parse_sdk_response(resp, "gemini-2.0-flash")
        assert r.raw is resp

    def test_finish_reason_mapped(self):
        resp = _sdk_response(finish_reason="STOP")
        r = _parse_sdk_response(resp, "gemini-2.0-flash")
        assert r.stop_reason == "STOP"


# ---------------------------------------------------------------------------
# Provider integration tests (mocked SDK client)
# ---------------------------------------------------------------------------


def test_provider_timeout_is_converted_to_sdk_milliseconds(monkeypatch):
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def close(self):
            pass

    monkeypatch.setattr(
        "ai_arch_toolkit.core._providers._gemini.genai.Client",
        FakeClient,
    )

    GeminiProvider("gemini-2.0-flash", "test-key", timeout=45)

    assert captured["api_key"] == "test-key"
    assert captured["http_options"] == {"timeout": 45_000}


class TestGeminiProviderComplete:
    async def test_complete(self):
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=_sdk_response())

        provider = GeminiProvider("gemini-2.0-flash", "test-key")
        provider._client = mock_client

        result = await provider.complete([{"role": "user", "content": "Hi"}])
        assert isinstance(result, Response)
        assert result.text == "Hello"
        mock_client.aio.models.generate_content.assert_awaited_once()

    async def test_system_forwarded(self):
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=_sdk_response())

        provider = GeminiProvider("gemini-2.0-flash", "test-key")
        provider._client = mock_client
        await provider.complete(
            [{"role": "user", "content": "Hi"}],
            system="Be helpful.",
        )
        call_kwargs = mock_client.aio.models.generate_content.call_args[1]
        assert call_kwargs["config"].system_instruction == "Be helpful."

    async def test_system_from_messages(self):
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=_sdk_response())

        provider = GeminiProvider("gemini-2.0-flash", "test-key")
        provider._client = mock_client
        await provider.complete(
            [
                {"role": "system", "content": "From message."},
                {"role": "user", "content": "Hi"},
            ]
        )
        call_kwargs = mock_client.aio.models.generate_content.call_args[1]
        assert call_kwargs["config"].system_instruction == "From message."

    async def test_explicit_system_overrides_message(self):
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=_sdk_response())

        provider = GeminiProvider("gemini-2.0-flash", "test-key")
        provider._client = mock_client
        await provider.complete(
            [
                {"role": "system", "content": "From message."},
                {"role": "user", "content": "Hi"},
            ],
            system="Explicit.",
        )
        call_kwargs = mock_client.aio.models.generate_content.call_args[1]
        assert call_kwargs["config"].system_instruction == "Explicit."

    async def test_tools_forwarded(self):
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=_sdk_response())

        tools = [{"name": "get_weather", "description": "Get weather", "parameters": {}}]
        provider = GeminiProvider("gemini-2.0-flash", "test-key")
        provider._client = mock_client
        await provider.complete([{"role": "user", "content": "Hi"}], tools=tools)
        call_kwargs = mock_client.aio.models.generate_content.call_args[1]
        config = call_kwargs["config"]
        assert config.tools is not None
        fd = config.tools[0].function_declarations[0]
        assert fd.name == "get_weather"

    async def test_thinking_forwarded(self):
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=_sdk_response())

        provider = GeminiProvider("gemini-2.0-flash", "test-key")
        provider._client = mock_client
        await provider.complete(
            [{"role": "user", "content": "Hi"}],
            thinking=True,
            thinking_budget=8000,
        )
        call_kwargs = mock_client.aio.models.generate_content.call_args[1]
        tc = call_kwargs["config"].thinking_config
        assert tc is not None
        assert tc.thinking_budget == 8000
        assert tc.include_thoughts is True

    async def test_output_schema_forwarded(self):
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=_sdk_response(text='{"name": "Alice"}')
        )

        schema = OutputSchema(name="Person", schema={"type": "object"})
        provider = GeminiProvider("gemini-2.0-flash", "test-key")
        provider._client = mock_client
        result = await provider.complete(
            [{"role": "user", "content": "Hi"}],
            output_schema=schema,
        )
        call_kwargs = mock_client.aio.models.generate_content.call_args[1]
        config = call_kwargs["config"]
        assert config.response_mime_type == "application/json"
        assert config.response_json_schema == {"type": "object"}
        assert result.parsed == {"name": "Alice"}

    async def test_unknown_kwargs_warn(self):
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=_sdk_response())

        provider = GeminiProvider("gemini-2.0-flash", "test-key")
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
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=_sdk_response())

        provider = GeminiProvider("gemini-2.0-flash", "test-key")
        provider._client = mock_client
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await provider.complete(
                [{"role": "user", "content": "Hi"}],
                temperature=0.5,
                top_p=0.9,
                max_output_tokens=1000,
            )
            assert len(w) == 0


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


class TestGeminiProviderErrors:
    async def test_rate_limit_error(self):
        from google.genai import errors as genai_errors

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=genai_errors.ClientError(429, {"error": "rate limited"})
        )

        provider = GeminiProvider("gemini-2.0-flash", "test-key")
        provider._client = mock_client
        with pytest.raises(RateLimitError) as exc_info:
            await provider.complete([{"role": "user", "content": "Hi"}])
        assert exc_info.value.status_code == 429

    async def test_client_error(self):
        from google.genai import errors as genai_errors

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=genai_errors.ClientError(400, {"error": "bad request"})
        )

        provider = GeminiProvider("gemini-2.0-flash", "test-key")
        provider._client = mock_client
        with pytest.raises(APIError) as exc_info:
            await provider.complete([{"role": "user", "content": "Hi"}])
        assert exc_info.value.status_code == 400

    async def test_server_error(self):
        from google.genai import errors as genai_errors

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=genai_errors.ServerError(500, {"error": "internal"})
        )

        provider = GeminiProvider("gemini-2.0-flash", "test-key")
        provider._client = mock_client
        with pytest.raises(APIError) as exc_info:
            await provider.complete([{"role": "user", "content": "Hi"}])
        assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def _wrap_as_coroutine(async_gen_fn):
    """Wrap an async generator function to behave like the real Gemini SDK.

    The real ``generate_content_stream`` is a coroutine that returns an
    async iterator. Tests define plain async generators; this wrapper
    makes them match the SDK's calling convention.
    """

    async def _wrapper(**kwargs):
        return async_gen_fn(**kwargs)

    return _wrapper


class TestGeminiStreamToolCalls:
    async def test_stream_text(self):
        chunks = [
            _sdk_stream_chunk(text="Hello "),
            _sdk_stream_chunk(text="world", finish_reason="STOP", candidates_tokens=5),
        ]

        async def _fake_stream(**kwargs):
            for c in chunks:
                yield c

        mock_client = MagicMock()
        mock_client.aio.models.generate_content_stream = _wrap_as_coroutine(_fake_stream)

        provider = GeminiProvider("gemini-2.0-flash", "test-key")
        provider._client = mock_client
        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        text_chunks = []
        async for chunk in aiter:
            text_chunks.append(chunk)

        assert text_chunks == ["Hello ", "world"]
        assert state.stop_reason == "STOP"

    async def test_stream_tool_call(self):
        fc = SimpleNamespace(id="tc_1", name="get_weather", args={"city": "NYC"})
        chunks = [
            _sdk_stream_chunk(function_call=fc, finish_reason="STOP"),
        ]

        async def _fake_stream(**kwargs):
            for c in chunks:
                yield c

        mock_client = MagicMock()
        mock_client.aio.models.generate_content_stream = _wrap_as_coroutine(_fake_stream)

        provider = GeminiProvider("gemini-2.0-flash", "test-key")
        provider._client = mock_client
        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        text_chunks = []
        async for chunk in aiter:
            text_chunks.append(chunk)

        assert text_chunks == []
        assert len(state.tool_calls) == 1
        assert state.tool_calls[0].name == "get_weather"
        assert state.tool_calls[0].input == {"city": "NYC"}

    async def test_stream_thinking(self):
        chunks = [
            _sdk_stream_chunk(thought_text="Let me think..."),
            _sdk_stream_chunk(text="The answer.", finish_reason="STOP"),
        ]

        async def _fake_stream(**kwargs):
            for c in chunks:
                yield c

        mock_client = MagicMock()
        mock_client.aio.models.generate_content_stream = _wrap_as_coroutine(_fake_stream)

        provider = GeminiProvider("gemini-2.0-flash", "test-key")
        provider._client = mock_client
        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        text_chunks = []
        async for chunk in aiter:
            text_chunks.append(chunk)

        assert text_chunks == ["The answer."]
        assert len(state.thinking) == 1
        assert state.thinking[0].text == "Let me think..."

    async def test_stream_usage(self):
        chunks = [
            _sdk_stream_chunk(text="Hi", prompt_tokens=10, candidates_tokens=5),
        ]

        async def _fake_stream(**kwargs):
            for c in chunks:
                yield c

        mock_client = MagicMock()
        mock_client.aio.models.generate_content_stream = _wrap_as_coroutine(_fake_stream)

        provider = GeminiProvider("gemini-2.0-flash", "test-key")
        provider._client = mock_client
        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        async for _ in aiter:
            pass

        assert state.usage is not None
        assert state.usage.input_tokens == 10
        assert state.usage.output_tokens == 5

    async def test_stream_error_mapping(self):
        from google.genai import errors as genai_errors

        async def _error_stream(**kwargs):
            raise genai_errors.ClientError(429, {"error": "rate limited"})
            yield  # makes this an async generator

        mock_client = MagicMock()
        mock_client.aio.models.generate_content_stream = _wrap_as_coroutine(_error_stream)

        provider = GeminiProvider("gemini-2.0-flash", "test-key")
        provider._client = mock_client
        aiter, _state = provider.stream([{"role": "user", "content": "Hi"}])
        with pytest.raises(RateLimitError):
            async for _ in aiter:
                pass


# ---------------------------------------------------------------------------
# Roundtrip test
# ---------------------------------------------------------------------------


class TestGeminiRoundtrip:
    def test_to_message_through_gemini_wire(self):
        """Response → to_message → Gemini _messages_to_sdk → correct format."""
        from ai_arch_toolkit.core._response import Response as Resp
        from ai_arch_toolkit.core._response import ToolCall as TC

        r = Resp(
            text="Let me check.",
            tool_calls=(TC(id="tc_1", name="get_weather", input={"city": "NYC"}),),
        )
        assistant_msg = r.to_message()

        conversation = [
            {"role": "user", "content": "What's the weather?"},
            assistant_msg,
        ]
        sys, contents = _messages_to_sdk(conversation)

        assert sys is None
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "What's the weather?"
        assert contents[1].role == "model"
        assert contents[1].parts[0].text == "Let me check."
        assert contents[1].parts[1].function_call.name == "get_weather"
