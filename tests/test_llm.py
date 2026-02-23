"""Tests for _llm.py — LLM class."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_arch_toolkit._llm import LLM
from ai_arch_toolkit._response import Response, StreamResponse, SyncStreamResponse, Usage


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
    @patch("ai_arch_toolkit._llm.create_provider")
    async def test_basic(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = _make_response()
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        result = await llm.complete("Hello")
        assert result.text == "Hello"
        mock_provider.complete.assert_called_once()

    @patch("ai_arch_toolkit._llm.create_provider")
    async def test_string_normalized(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = _make_response()
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        await llm.complete("Hello")
        call_args = mock_provider.complete.call_args
        messages = call_args[0][0]
        assert messages == [{"role": "user", "content": "Hello"}]

    @patch("ai_arch_toolkit._llm.create_provider")
    async def test_list_passthrough(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = _make_response()
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        msgs = [{"role": "user", "content": "Hi"}]
        await llm.complete(msgs)
        call_args = mock_provider.complete.call_args
        assert call_args[0][0] is msgs

    @patch("ai_arch_toolkit._llm.create_provider")
    async def test_default_kwargs(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = _make_response()
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test", temperature=0.5, max_tokens=1000)
        await llm.complete("Hi")
        call_kwargs = mock_provider.complete.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 1000

    @patch("ai_arch_toolkit._llm.create_provider")
    async def test_override_kwargs(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = _make_response()
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test", temperature=0.0)
        await llm.complete("Hi", temperature=0.8)
        call_kwargs = mock_provider.complete.call_args[1]
        assert call_kwargs["temperature"] == 0.8

    @patch("ai_arch_toolkit._llm.create_provider")
    async def test_tools_forwarded(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = _make_response()
        mock_create.return_value = mock_provider

        tools = [{"name": "search", "description": "Search", "parameters": {"type": "object"}}]
        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        await llm.complete("Hi", tools=tools)
        call_kwargs = mock_provider.complete.call_args[1]
        assert call_kwargs["tools"] is tools


class TestStream:
    @patch("ai_arch_toolkit._llm.create_provider")
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

    @patch("ai_arch_toolkit._llm.create_provider")
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

    @patch("ai_arch_toolkit._llm.create_provider")
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
            async for chunk in stream:
                break  # early exit

        # Response should be finalized with partial content
        assert stream.response is not None
        assert stream.response.text == "Hello"


class TestCall:
    @patch("ai_arch_toolkit._llm.create_provider")
    async def test_call_is_alias(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = _make_response("via call")
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        result = await llm("Hello")
        assert result.text == "via call"


class TestSyncWrappers:
    @patch("ai_arch_toolkit._llm.create_provider")
    def test_complete_sync(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = _make_response("sync result")
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        result = llm.complete_sync("Hello")
        assert result.text == "sync result"

    @patch("ai_arch_toolkit._llm.create_provider")
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
    @patch("ai_arch_toolkit._llm.create_provider")
    def test_repr_custom_params(self, mock_create):
        mock_create.return_value = AsyncMock()
        llm = LLM("claude-sonnet-4-20250514", api_key="test", temperature=0.5)
        r = repr(llm)
        assert "claude-sonnet-4-20250514" in r
        assert "temperature=0.5" in r
        assert "max_tokens" not in r  # default not shown

    @patch("ai_arch_toolkit._llm.create_provider")
    def test_repr_defaults_only(self, mock_create):
        mock_create.return_value = AsyncMock()
        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        assert repr(llm) == "LLM(model='claude-sonnet-4-20250514')"


class TestModelRouting:
    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Cannot detect provider"):
            LLM("unknown-model-v1", api_key="test")

    @patch("ai_arch_toolkit._llm.create_provider")
    def test_claude_model_creates_provider(self, mock_create):
        mock_create.return_value = AsyncMock()
        LLM("claude-sonnet-4-20250514", api_key="test")
        mock_create.assert_called_once()
        assert mock_create.call_args[0][0] == "claude-sonnet-4-20250514"


class TestLifecycle:
    @patch("ai_arch_toolkit._llm.create_provider")
    async def test_async_context_manager(self, mock_create):
        mock_provider = AsyncMock()
        mock_create.return_value = mock_provider

        async with LLM("claude-sonnet-4-20250514", api_key="test") as llm:
            assert llm is not None

        mock_provider.close.assert_called_once()

    @patch("ai_arch_toolkit._llm.create_provider")
    async def test_close(self, mock_create):
        mock_provider = AsyncMock()
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        await llm.close()
        mock_provider.close.assert_called_once()

    @patch("ai_arch_toolkit._llm.create_provider")
    def test_sync_context_manager(self, mock_create):
        mock_provider = AsyncMock()
        mock_create.return_value = mock_provider

        with LLM("claude-sonnet-4-20250514", api_key="test") as llm:
            assert llm is not None

        mock_provider.close.assert_called_once()
