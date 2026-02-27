"""Tests for _llm.py — LLM class."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._response import (
    OutputSchema,
    Response,
    StreamResponse,
    SyncStreamResponse,
    ThinkingBlock,
    Usage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(text: str = "Hello") -> Response:
    return Response(text=text, usage=Usage(input_tokens=10, output_tokens=5))


def _make_stream_state(usage=None, model="claude-sonnet-4-20250514", stop_reason="end_turn"):
    """Create a mock stream state."""
    state = MagicMock()
    state.usage = usage or Usage(input_tokens=10, output_tokens=5)
    state.model = model
    state.stop_reason = stop_reason
    state.tool_calls = []
    state.thinking = []
    return state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_string_to_user_message(self):
        result = LLM._normalize("Hello")
        assert result == [{"role": "user", "content": "Hello"}]

    def test_list_passthrough(self):
        msgs = [{"role": "user", "content": "Hi"}]
        assert LLM._normalize(msgs) is msgs


class TestComplete:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_basic(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = _make_response()
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        result = await llm.complete("Hello")
        assert result.text == "Hello"
        mock_provider.complete.assert_called_once()

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_string_normalized(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = _make_response()
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        await llm.complete("Hello")
        call_args = mock_provider.complete.call_args
        messages = call_args[0][0]
        assert messages == [{"role": "user", "content": "Hello"}]

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_list_passthrough(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = _make_response()
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        msgs = [{"role": "user", "content": "Hi"}]
        await llm.complete(msgs)
        call_args = mock_provider.complete.call_args
        assert call_args[0][0] is msgs

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_default_kwargs(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = _make_response()
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test", temperature=0.5, max_tokens=1000)
        await llm.complete("Hi")
        call_kwargs = mock_provider.complete.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 1000

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_override_kwargs(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = _make_response()
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test", temperature=0.0)
        await llm.complete("Hi", temperature=0.8)
        call_kwargs = mock_provider.complete.call_args[1]
        assert call_kwargs["temperature"] == 0.8

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_tools_forwarded(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = _make_response()
        mock_create.return_value = mock_provider

        tools = [{"name": "search", "description": "Search", "parameters": {"type": "object"}}]
        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        await llm.complete("Hi", tools=tools)
        call_kwargs = mock_provider.complete.call_args[1]
        assert call_kwargs["tools"] == tools


class TestStream:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_yields_chunks(self, mock_create):
        async def _fake_gen():
            for chunk in ["Hello", " ", "world"]:
                yield chunk

        state = _make_stream_state()
        mock_provider = MagicMock()
        mock_provider.stream.return_value = (_fake_gen(), state)
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        stream = llm.stream("Hi")
        assert isinstance(stream, StreamResponse)

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        assert chunks == ["Hello", " ", "world"]

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_stream_response_available_after_consume(self, mock_create):
        async def _fake_gen():
            for chunk in ["Hello"]:
                yield chunk

        state = _make_stream_state()
        mock_provider = MagicMock()
        mock_provider.stream.return_value = (_fake_gen(), state)
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        stream = llm.stream("Hi")
        assert stream.response is None  # not consumed yet

        async for _ in stream:
            pass

        assert stream.response is not None
        assert stream.response.text == "Hello"
        assert stream.response.usage.input_tokens == 10

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_stream_context_manager_early_exit(self, mock_create):
        async def _fake_gen():
            for chunk in ["Hello", " ", "world"]:
                yield chunk

        state = _make_stream_state()
        mock_provider = MagicMock()
        mock_provider.stream.return_value = (_fake_gen(), state)
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")

        async with llm.stream("Hi") as stream:
            async for _chunk in stream:
                break  # early exit

        # Response should be finalized with partial content
        assert stream.response is not None
        assert stream.response.text == "Hello"


class TestCall:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_call_is_alias(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = _make_response("via call")
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        result = await llm("Hello")
        assert result.text == "via call"


class TestSyncWrappers:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_complete_sync(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = _make_response("sync result")
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        result = llm.complete_sync("Hello")
        assert result.text == "sync result"

    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_stream_sync(self, mock_create):
        async def _fake_gen():
            for chunk in ["a", "b", "c"]:
                yield chunk

        state = _make_stream_state()
        mock_provider = MagicMock()
        mock_provider.stream.return_value = (_fake_gen(), state)
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        stream = llm.stream_sync("Hi")
        assert isinstance(stream, SyncStreamResponse)

        chunks = list(stream)
        assert chunks == ["a", "b", "c"]
        assert stream.response is not None
        assert stream.response.text == "abc"


class TestRepr:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_repr_custom_params(self, mock_create):
        mock_create.return_value = AsyncMock()
        llm = LLM("claude-sonnet-4-20250514", api_key="test", temperature=0.5)
        r = repr(llm)
        assert "claude-sonnet-4-20250514" in r
        assert "temperature=0.5" in r
        assert "max_tokens" not in r  # default not shown

    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_repr_defaults_only(self, mock_create):
        mock_create.return_value = AsyncMock()
        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        assert repr(llm) == "LLM(model='claude-sonnet-4-20250514')"


class TestModelRouting:
    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Cannot detect provider"):
            LLM("unknown-model-v1", api_key="test")

    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_claude_model_creates_provider(self, mock_create):
        mock_create.return_value = AsyncMock()
        LLM("claude-sonnet-4-20250514", api_key="test")
        mock_create.assert_called_once()
        assert mock_create.call_args[0][0] == "claude-sonnet-4-20250514"


class TestLifecycle:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_async_context_manager(self, mock_create):
        mock_provider = AsyncMock()
        mock_create.return_value = mock_provider

        async with LLM("claude-sonnet-4-20250514", api_key="test") as llm:
            assert llm is not None

        mock_provider.close.assert_called_once()

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_close(self, mock_create):
        mock_provider = AsyncMock()
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        await llm.close()
        mock_provider.close.assert_called_once()

    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_sync_context_manager(self, mock_create):
        mock_provider = AsyncMock()
        mock_create.return_value = mock_provider

        with LLM("claude-sonnet-4-20250514", api_key="test") as llm:
            assert llm is not None

        mock_provider.close.assert_called_once()


class TestThinkingParams:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_thinking_forwarded(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = _make_response()
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        await llm.complete("Hi", thinking=True, thinking_effort="high")
        call_kwargs = mock_provider.complete.call_args[1]
        assert call_kwargs["thinking"] is True
        assert call_kwargs["thinking_effort"] == "high"

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_thinking_budget_forwarded(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = _make_response()
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        await llm.complete("Hi", thinking=True, thinking_budget=10000)
        call_kwargs = mock_provider.complete.call_args[1]
        assert call_kwargs["thinking_budget"] == 10000

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_thinking_defaults_not_forwarded(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = _make_response()
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        await llm.complete("Hi")
        call_kwargs = mock_provider.complete.call_args[1]
        assert "thinking" not in call_kwargs
        assert "thinking_effort" not in call_kwargs
        assert "thinking_budget" not in call_kwargs


class TestThinkingValidation:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_empty_thinking_effort_raises(self, mock_create):
        mock_create.return_value = AsyncMock()
        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        with pytest.raises(ValueError, match="thinking_effort must be a non-empty string"):
            await llm.complete("Hi", thinking_effort="")

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_negative_thinking_budget_raises(self, mock_create):
        mock_create.return_value = AsyncMock()
        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        with pytest.raises(ValueError, match="thinking_budget must be non-negative"):
            await llm.complete("Hi", thinking_budget=-1)

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_zero_thinking_budget_allowed(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = _make_response()
        mock_create.return_value = mock_provider
        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        await llm.complete("Hi", thinking_budget=0)  # should not raise


class TestOutputSchema:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_output_schema_forwarded(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = _make_response()
        mock_create.return_value = mock_provider

        schema = OutputSchema(name="Person", schema={"type": "object"})
        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        await llm.complete("Hi", output_schema=schema)
        call_kwargs = mock_provider.complete.call_args[1]
        assert call_kwargs["output_schema"] is schema

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_output_schema_default_not_forwarded(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = _make_response()
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        await llm.complete("Hi")
        call_kwargs = mock_provider.complete.call_args[1]
        assert "output_schema" not in call_kwargs


class TestStreamForwarding:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_stream_forwards_thinking_params(self, mock_create):
        async def _fake_gen():
            yield "Hi"

        state = _make_stream_state()
        mock_provider = MagicMock()
        mock_provider.stream.return_value = (_fake_gen(), state)
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        stream = llm.stream("Hi", thinking=True, thinking_effort="high", thinking_budget=5000)
        async for _ in stream:
            pass
        call_kwargs = mock_provider.stream.call_args[1]
        assert call_kwargs["thinking"] is True
        assert call_kwargs["thinking_effort"] == "high"
        assert call_kwargs["thinking_budget"] == 5000

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_stream_forwards_output_schema(self, mock_create):
        async def _fake_gen():
            yield "Hi"

        state = _make_stream_state()
        mock_provider = MagicMock()
        mock_provider.stream.return_value = (_fake_gen(), state)
        mock_create.return_value = mock_provider

        schema = OutputSchema(name="X", schema={"type": "object"})
        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        stream = llm.stream("Hi", output_schema=schema)
        async for _ in stream:
            pass
        call_kwargs = mock_provider.stream.call_args[1]
        assert call_kwargs["output_schema"] is schema

    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_stream_sync_forwards_thinking_params(self, mock_create):
        async def _fake_gen():
            yield "Hi"

        state = _make_stream_state()
        mock_provider = MagicMock()
        mock_provider.stream.return_value = (_fake_gen(), state)
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        stream = llm.stream_sync("Hi", thinking=True, thinking_effort="low")
        list(stream)
        call_kwargs = mock_provider.stream.call_args[1]
        assert call_kwargs["thinking"] is True
        assert call_kwargs["thinking_effort"] == "low"


class TestStreamThinking:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_stream_includes_thinking(self, mock_create):
        async def _fake_gen():
            yield "Answer"

        state = _make_stream_state()
        state.thinking = [ThinkingBlock(text="Let me think...")]
        mock_provider = MagicMock()
        mock_provider.stream.return_value = (_fake_gen(), state)
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        stream = llm.stream("Hi")
        async for _ in stream:
            pass
        assert stream.response is not None
        assert len(stream.response.thinking) == 1
        assert stream.response.thinking[0].text == "Let me think..."


# ---------------------------------------------------------------------------
# tool_choice + json_mode
# ---------------------------------------------------------------------------


class TestToolChoiceParam:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_tool_choice_forwarded(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = Response(text="ok")
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        await llm.complete("Hi", tool_choice="required")
        call_kwargs = mock_provider.complete.call_args[1]
        assert call_kwargs["tool_choice"] == "required"

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_tool_choice_specific_name(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = Response(text="ok")
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        await llm.complete("Hi", tool_choice="get_weather")
        call_kwargs = mock_provider.complete.call_args[1]
        assert call_kwargs["tool_choice"] == "get_weather"


class TestJsonModeParam:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_json_mode_forwarded(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = Response(text='{"key": "value"}')
        mock_create.return_value = mock_provider

        llm = LLM("gpt-4o", api_key="test")
        await llm.complete("Give me JSON", json_mode=True)
        call_kwargs = mock_provider.complete.call_args[1]
        assert call_kwargs["json_mode"] is True

    def test_json_mode_and_output_schema_raises(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            LLM._prepare_provider_kwargs(
                thinking=False,
                thinking_effort=None,
                thinking_budget=None,
                output_schema=OutputSchema(name="test", schema={"type": "object"}),
                tool_choice=None,
                json_mode=True,
                logprobs=False,
                extra={},
            )
