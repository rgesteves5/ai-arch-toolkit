"""Tests for _llm.py — LLM class."""

from __future__ import annotations

from unittest.mock import patch

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
from tests.fake_provider import FakeProvider, Reply

MODEL = "claude-sonnet-4-20250514"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(text: str = "Hello") -> Response:
    return Response(text=text, usage=Usage(input_tokens=10, output_tokens=5))


def _fake_provider(model: str, **_: object) -> FakeProvider:
    """Stands in for ``create_provider``: a fake provider for ``model``."""
    return FakeProvider(model=model)


class _ClosingProvider(FakeProvider):
    """A fake provider that counts how many times it is closed."""

    def __init__(self) -> None:
        super().__init__(model=MODEL)
        self.closes = 0

    async def close(self) -> None:
        self.closes += 1
        await super().close()


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
        provider = FakeProvider(_make_response(), model=MODEL)
        mock_create.return_value = provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        result = await llm.complete("Hello")
        assert result.text == "Hello"
        assert provider.calls == 1

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_string_normalized(self, mock_create):
        provider = FakeProvider(_make_response(), model=MODEL)
        mock_create.return_value = provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        await llm.complete("Hello")
        assert provider.last.messages == [{"role": "user", "content": "Hello"}]

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_list_passthrough(self, mock_create):
        provider = FakeProvider(_make_response(), model=MODEL)
        mock_create.return_value = provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        msgs = [{"role": "user", "content": "Hi"}]
        await llm.complete(msgs)
        assert provider.last.messages is msgs

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_default_kwargs(self, mock_create):
        provider = FakeProvider(_make_response(), model=MODEL)
        mock_create.return_value = provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test", temperature=0.5, max_tokens=1000)
        await llm.complete("Hi")
        assert provider.last.kwargs["temperature"] == 0.5
        assert provider.last.kwargs["max_tokens"] == 1000

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_override_kwargs(self, mock_create):
        provider = FakeProvider(_make_response(), model=MODEL)
        mock_create.return_value = provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test", temperature=0.0)
        await llm.complete("Hi", temperature=0.8)
        assert provider.last.kwargs["temperature"] == 0.8

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_tools_forwarded(self, mock_create):
        provider = FakeProvider(_make_response(), model=MODEL)
        mock_create.return_value = provider

        tools = [{"name": "search", "description": "Search", "parameters": {"type": "object"}}]
        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        await llm.complete("Hi", tools=tools)
        assert provider.last.tools == tools


class TestStream:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_yields_chunks(self, mock_create):
        reply = Reply(response=_make_response("Hello world"), chunks=["Hello", " ", "world"])
        mock_create.return_value = FakeProvider(reply, model=MODEL)

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        stream = llm.stream("Hi")
        assert isinstance(stream, StreamResponse)

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        assert chunks == ["Hello", " ", "world"]

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_stream_response_available_after_consume(self, mock_create):
        mock_create.return_value = FakeProvider(_make_response("Hello"), model=MODEL)

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
        reply = Reply(response=_make_response("Hello world"), chunks=["Hello", " ", "world"])
        mock_create.return_value = FakeProvider(reply, model=MODEL)

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
        mock_create.return_value = FakeProvider(_make_response("via call"), model=MODEL)

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        result = await llm("Hello")
        assert result.text == "via call"


class TestSyncWrappers:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_complete_sync(self, mock_create):
        mock_create.return_value = FakeProvider(_make_response("sync result"), model=MODEL)

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        result = llm.complete_sync("Hello")
        assert result.text == "sync result"

    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_stream_sync(self, mock_create):
        reply = Reply(response=_make_response("abc"), chunks=["a", "b", "c"])
        mock_create.return_value = FakeProvider(reply, model=MODEL)

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
        mock_create.side_effect = _fake_provider
        llm = LLM("claude-sonnet-4-20250514", api_key="test", temperature=0.5)
        r = repr(llm)
        assert "claude-sonnet-4-20250514" in r
        assert "temperature=0.5" in r
        assert "max_tokens" not in r  # default not shown

    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_repr_defaults_only(self, mock_create):
        mock_create.side_effect = _fake_provider
        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        assert repr(llm) == "LLM(model='claude-sonnet-4-20250514')"


class TestModelRouting:
    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Cannot detect provider"):
            LLM("unknown-model-v1", api_key="test")

    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_claude_model_creates_provider(self, mock_create):
        mock_create.side_effect = _fake_provider
        LLM("claude-sonnet-4-20250514", api_key="test")
        mock_create.assert_called_once()
        assert mock_create.call_args[0][0] == "claude-sonnet-4-20250514"

    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_provider_kwarg_forwarded(self, mock_create):
        mock_create.side_effect = _fake_provider
        LLM("gemma4:e4b", provider="openai", base_url="http://localhost:11434/v1")
        mock_create.assert_called_once()
        assert mock_create.call_args.kwargs["provider"] == "openai"
        assert mock_create.call_args.kwargs["base_url"] == "http://localhost:11434/v1"

    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_unroutable_fallback_inherits_connection(self, mock_create):
        # A bare tag is assumed to live on the same server as the primary.
        mock_create.side_effect = _fake_provider
        LLM(
            "model-a",
            provider="openai",
            base_url="http://localhost:11434/v1",
            fallback="model-b",
        )
        assert mock_create.call_count == 2
        for call in mock_create.call_args_list:
            assert call.kwargs["provider"] == "openai"
            assert call.kwargs["base_url"] == "http://localhost:11434/v1"

    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_routable_fallback_is_standalone(self, mock_create):
        # A recognizable model name fails over to its own provider/connection,
        # not the local primary's — so local→cloud failover works.
        mock_create.side_effect = _fake_provider
        LLM(
            "gemma4:e4b",
            provider="openai",
            base_url="http://localhost:11434/v1",
            fallback="claude-sonnet-4-20250514",
        )
        assert mock_create.call_count == 2
        primary, fb = mock_create.call_args_list
        assert primary.kwargs["provider"] == "openai"
        assert fb.args[0] == "claude-sonnet-4-20250514"
        assert fb.kwargs["provider"] is None
        assert fb.kwargs["base_url"] is None


class TestLifecycle:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_async_context_manager(self, mock_create):
        provider = _ClosingProvider()
        mock_create.return_value = provider

        async with LLM("claude-sonnet-4-20250514", api_key="test") as llm:
            assert llm is not None

        assert provider.closes == 1

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_close(self, mock_create):
        provider = _ClosingProvider()
        mock_create.return_value = provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        await llm.close()
        assert provider.closes == 1

    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_sync_context_manager(self, mock_create):
        provider = _ClosingProvider()
        mock_create.return_value = provider

        with LLM("claude-sonnet-4-20250514", api_key="test") as llm:
            assert llm is not None

        assert provider.closes == 1


class TestThinkingParams:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_thinking_forwarded(self, mock_create):
        provider = FakeProvider(_make_response(), model=MODEL)
        mock_create.return_value = provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        await llm.complete("Hi", thinking=True, thinking_effort="high")
        assert provider.last.kwargs["thinking"] is True
        assert provider.last.kwargs["thinking_effort"] == "high"

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_thinking_budget_forwarded(self, mock_create):
        provider = FakeProvider(_make_response(), model=MODEL)
        mock_create.return_value = provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        await llm.complete("Hi", thinking=True, thinking_budget=10000)
        assert provider.last.kwargs["thinking_budget"] == 10000

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_thinking_defaults_not_forwarded(self, mock_create):
        provider = FakeProvider(_make_response(), model=MODEL)
        mock_create.return_value = provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        await llm.complete("Hi")
        call_kwargs = provider.last.kwargs
        assert "thinking" not in call_kwargs
        assert "thinking_effort" not in call_kwargs
        assert "thinking_budget" not in call_kwargs


class TestThinkingValidation:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_empty_thinking_effort_raises(self, mock_create):
        mock_create.side_effect = _fake_provider
        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        with pytest.raises(ValueError, match="thinking_effort must be a non-empty string"):
            await llm.complete("Hi", thinking_effort="")

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_negative_thinking_budget_raises(self, mock_create):
        mock_create.side_effect = _fake_provider
        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        with pytest.raises(ValueError, match="thinking_budget must be non-negative"):
            await llm.complete("Hi", thinking_budget=-1)

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_zero_thinking_budget_allowed(self, mock_create):
        mock_create.return_value = FakeProvider(_make_response(), model=MODEL)
        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        await llm.complete("Hi", thinking_budget=0)  # should not raise


class TestOutputSchema:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_output_schema_forwarded(self, mock_create):
        provider = FakeProvider(_make_response(), model=MODEL)
        mock_create.return_value = provider

        schema = OutputSchema(name="Person", schema={"type": "object"})
        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        await llm.complete("Hi", output_schema=schema)
        assert provider.last.kwargs["output_schema"] is schema

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_output_schema_default_not_forwarded(self, mock_create):
        provider = FakeProvider(_make_response(), model=MODEL)
        mock_create.return_value = provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        await llm.complete("Hi")
        assert "output_schema" not in provider.last.kwargs


class TestStreamForwarding:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_stream_forwards_thinking_params(self, mock_create):
        provider = FakeProvider(_make_response("Hi"), model=MODEL)
        mock_create.return_value = provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        stream = llm.stream("Hi", thinking=True, thinking_effort="high", thinking_budget=5000)
        async for _ in stream:
            pass
        call_kwargs = provider.last.kwargs
        assert call_kwargs["thinking"] is True
        assert call_kwargs["thinking_effort"] == "high"
        assert call_kwargs["thinking_budget"] == 5000

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_stream_forwards_output_schema(self, mock_create):
        provider = FakeProvider(_make_response("Hi"), model=MODEL)
        mock_create.return_value = provider

        schema = OutputSchema(name="X", schema={"type": "object"})
        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        stream = llm.stream("Hi", output_schema=schema)
        async for _ in stream:
            pass
        assert provider.last.kwargs["output_schema"] is schema

    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_stream_sync_forwards_thinking_params(self, mock_create):
        provider = FakeProvider(_make_response("Hi"), model=MODEL)
        mock_create.return_value = provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        stream = llm.stream_sync("Hi", thinking=True, thinking_effort="low")
        list(stream)
        assert provider.last.kwargs["thinking"] is True
        assert provider.last.kwargs["thinking_effort"] == "low"


class TestStreamThinking:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_stream_includes_thinking(self, mock_create):
        answer = Response(
            text="Answer",
            thinking=(ThinkingBlock(text="Let me think..."),),
            usage=Usage(input_tokens=10, output_tokens=5),
        )
        mock_create.return_value = FakeProvider(answer, model=MODEL)

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
        provider = FakeProvider(Response(text="ok"), model=MODEL)
        mock_create.return_value = provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        await llm.complete("Hi", tool_choice="required")
        assert provider.last.kwargs["tool_choice"] == "required"

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_tool_choice_specific_name(self, mock_create):
        provider = FakeProvider(Response(text="ok"), model=MODEL)
        mock_create.return_value = provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        await llm.complete("Hi", tool_choice="get_weather")
        assert provider.last.kwargs["tool_choice"] == "get_weather"


class TestJsonModeParam:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_json_mode_forwarded(self, mock_create):
        provider = FakeProvider(Response(text='{"key": "value"}'), model="gpt-4o")
        mock_create.return_value = provider

        llm = LLM("gpt-4o", api_key="test")
        await llm.complete("Give me JSON", json_mode=True)
        assert provider.last.kwargs["json_mode"] is True

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
