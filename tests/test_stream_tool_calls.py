"""Tool calls, thinking and usage in streams, through the SDKs' own stream processing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_arch_toolkit.core._exceptions import ResponseError
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._providers._anthropic import AnthropicProvider
from ai_arch_toolkit.core._providers._openai import OpenAIProvider
from ai_arch_toolkit.core._response import Response, ThinkingBlock, ToolCall, Usage
from tests.fake_provider import FakeProvider, Reply
from tests.provider_calls import stream as _stream
from tests.sdk_streams import (
    AnthropicStream,
    OpenAIStream,
    anthropic_block_start,
    anthropic_block_stop,
    anthropic_json,
    anthropic_message_end,
    anthropic_message_start,
    anthropic_text,
    anthropic_thinking,
    openai_chunk,
    openai_usage_chunk,
)

HI = [{"role": "user", "content": "Hi"}]


async def stream(provider, messages, **kwargs):
    """The Messages API requires max_tokens; the LLM always sends one."""
    return await _stream(provider, messages, **{"max_tokens": 1024, **kwargs})


def _anthropic(events: list) -> AnthropicProvider:
    provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
    provider._client = MagicMock()
    provider._client.messages.stream.return_value = AnthropicStream(events)
    return provider


def _texts(events: list) -> list[str]:
    return [event.text for event in events if event.kind == "text"]


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class TestAnthropicStreamToolCalls:
    async def test_single_tool_call(self):
        provider = _anthropic(
            [
                anthropic_message_start(input_tokens=25),
                anthropic_block_start(0, "tool_use", id="tc_1", name="get_weather"),
                anthropic_json(0, '{"city"'),
                anthropic_json(0, ': "NYC"}'),
                anthropic_block_stop(0),
                *anthropic_message_end("tool_use", input_tokens=25, output_tokens=15),
            ]
        )
        events, response = await stream(provider, HI)

        assert _texts(events) == []
        call = ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"})
        assert response.tool_calls == (call,)
        assert [event.tool_call for event in events if event.kind == "tool_call"] == [call]
        assert response.stop_reason == "tool_use"
        # message_delta usage is cumulative: it replaces message_start's counts.
        assert response.usage == Usage(input_tokens=25, output_tokens=15)

    async def test_text_then_tool_call(self):
        provider = _anthropic(
            [
                anthropic_message_start(input_tokens=10),
                anthropic_block_start(0, "text"),
                anthropic_text(0, "Let me check."),
                anthropic_block_stop(0),
                anthropic_block_start(1, "tool_use", id="tc_1", name="get_weather"),
                anthropic_json(1, '{"city": "NYC"}'),
                anthropic_block_stop(1),
                *anthropic_message_end("tool_use", output_tokens=20),
            ]
        )
        events, response = await stream(provider, HI)

        assert _texts(events) == ["Let me check."]
        assert [(c.name, c.input) for c in response.tool_calls] == [
            ("get_weather", {"city": "NYC"})
        ]
        # Tool-call events follow the text.
        assert [event.kind for event in events] == ["text", "tool_call"]

    async def test_multiple_tool_calls(self):
        provider = _anthropic(
            [
                anthropic_message_start(input_tokens=10),
                anthropic_block_start(0, "tool_use", id="tc_1", name="get_weather"),
                anthropic_json(0, '{"city": "NYC"}'),
                anthropic_block_stop(0),
                anthropic_block_start(1, "tool_use", id="tc_2", name="get_time"),
                anthropic_json(1, '{"tz": "UTC"}'),
                anthropic_block_stop(1),
                *anthropic_message_end("tool_use", output_tokens=20),
            ]
        )
        _, response = await stream(provider, HI)

        assert response.tool_calls == (
            ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"}),
            ToolCall(id="tc_2", name="get_time", input={"tz": "UTC"}),
        )

    async def test_malformed_tool_args_fail_the_stream(self):
        # The SDK's accumulator refuses invalid tool JSON (the adapter used to hand the tool a
        # {"_raw": ...} input): a response that cannot be read, after the request was sent.
        provider = _anthropic(
            [
                anthropic_message_start(input_tokens=1),
                anthropic_block_start(0, "tool_use", id="tc_1", name="fn"),
                anthropic_json(0, "not valid json"),
                anthropic_block_stop(0),
                *anthropic_message_end("tool_use", output_tokens=3),
            ]
        )
        with pytest.raises(ResponseError, match="Unable to parse tool parameter JSON") as failed:
            await stream(provider, HI)
        assert failed.value.delivery == "indeterminate"

    async def test_tools_passed_in_sdk_kwargs(self):
        provider = _anthropic([anthropic_message_start(), *anthropic_message_end()])
        tools = [{"name": "get_weather", "description": "Get weather", "input_schema": {}}]
        await stream(provider, HI, tools=tools)

        call_kwargs = provider._client.messages.stream.call_args[1]
        assert call_kwargs["tools"][0]["name"] == "get_weather"

    async def test_empty_tool_args(self):
        provider = _anthropic(
            [
                anthropic_message_start(),
                anthropic_block_start(0, "tool_use", id="tc_1", name="get_status"),
                anthropic_block_stop(0),
                *anthropic_message_end("tool_use"),
            ]
        )
        _, response = await stream(provider, HI)

        assert response.tool_calls == (ToolCall(id="tc_1", name="get_status", input={}),)


class TestAnthropicStreamThinking:
    async def test_thinking_blocks_come_with_the_answer(self):
        provider = _anthropic(
            [
                anthropic_message_start(input_tokens=10),
                anthropic_block_start(0, "thinking"),
                anthropic_thinking(0, "Let me reason "),
                anthropic_thinking(0, "step by step..."),
                anthropic_block_stop(0),
                anthropic_block_start(1, "text"),
                anthropic_text(1, "The answer is 42."),
                anthropic_block_stop(1),
                *anthropic_message_end(output_tokens=20),
            ]
        )
        events, response = await stream(provider, [{"role": "user", "content": "Think"}])

        assert _texts(events) == ["The answer is 42."]
        assert response.thinking == (ThinkingBlock(text="Let me reason step by step..."),)
        assert [event.kind for event in events] == ["thinking", "text"]

    async def test_multiple_thinking_blocks(self):
        provider = _anthropic(
            [
                anthropic_message_start(),
                anthropic_block_start(0, "thinking"),
                anthropic_thinking(0, "First thought"),
                anthropic_block_stop(0),
                anthropic_block_start(1, "thinking"),
                anthropic_thinking(1, "Second thought"),
                anthropic_block_stop(1),
                anthropic_block_start(2, "text"),
                anthropic_text(2, "Answer."),
                anthropic_block_stop(2),
                *anthropic_message_end(),
            ]
        )
        _, response = await stream(provider, HI)

        assert [block.text for block in response.thinking] == ["First thought", "Second thought"]

    async def test_no_thinking_blocks(self):
        provider = _anthropic(
            [
                anthropic_message_start(),
                anthropic_block_start(0, "text"),
                anthropic_text(0, "Hello"),
                anthropic_block_stop(0),
                *anthropic_message_end(),
            ]
        )
        _, response = await stream(provider, HI)

        assert response.thinking == ()


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


def _openai(chunks: list, model: str = "gpt-4o") -> OpenAIProvider:
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = OpenAIStream(chunks)
    provider = OpenAIProvider(model, "test-key")
    provider._client = mock_client
    return provider


class TestOpenAIStreamToolCalls:
    async def test_single_tool_call(self):
        provider = _openai(
            [
                openai_chunk(tool_calls=[{"index": 0, "id": "tc_1", "name": "get_weather"}]),
                openai_chunk(tool_calls=[{"index": 0, "arguments": '{"city"'}]),
                openai_chunk(
                    tool_calls=[{"index": 0, "arguments": ': "NYC"}'}],
                    finish_reason="tool_calls",
                ),
                openai_usage_chunk(25, 15),
            ]
        )
        events, response = await stream(provider, HI)

        assert _texts(events) == []
        assert response.tool_calls == (
            ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"}),
        )
        assert response.stop_reason == "tool_calls"

    async def test_multiple_tool_calls(self):
        provider = _openai(
            [
                openai_chunk(tool_calls=[{"index": 0, "id": "tc_1", "name": "get_weather"}]),
                openai_chunk(tool_calls=[{"index": 0, "arguments": '{"city": "NYC"}'}]),
                openai_chunk(tool_calls=[{"index": 1, "id": "tc_2", "name": "get_time"}]),
                openai_chunk(
                    tool_calls=[{"index": 1, "arguments": '{"tz": "UTC"}'}],
                    finish_reason="tool_calls",
                ),
                openai_usage_chunk(30, 20),
            ]
        )
        _, response = await stream(provider, HI)

        assert response.tool_calls == (
            ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"}),
            ToolCall(id="tc_2", name="get_time", input={"tz": "UTC"}),
        )

    async def test_text_then_tool_call(self):
        provider = _openai(
            [
                openai_chunk(content="Let me check."),
                openai_chunk(
                    tool_calls=[
                        {
                            "index": 0,
                            "id": "tc_1",
                            "name": "get_weather",
                            "arguments": '{"city": "NYC"}',
                        }
                    ],
                    finish_reason="tool_calls",
                ),
                openai_usage_chunk(20, 15),
            ]
        )
        events, response = await stream(provider, HI)

        assert _texts(events) == ["Let me check."]
        assert [call.name for call in response.tool_calls] == ["get_weather"]

    async def test_tools_passed_in_sdk_kwargs(self):
        provider = _openai([openai_chunk(content="Hi", finish_reason="stop")])
        tools = [{"name": "get_weather", "description": "Get weather", "input_schema": {}}]
        await stream(provider, HI, tools=tools)

        call_kwargs = provider._client.chat.completions.create.call_args[1]
        assert call_kwargs["tools"][0]["type"] == "function"

    async def test_stream_includes_usage_option(self):
        provider = _openai([openai_chunk(content="Hi", finish_reason="stop")])
        await stream(provider, HI)

        call_kwargs = provider._client.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True
        assert call_kwargs["stream_options"] == {"include_usage": True}

    async def test_stream_captures_usage(self):
        provider = _openai(
            [
                openai_chunk(content="Hi"),
                openai_chunk(finish_reason="stop"),
                openai_usage_chunk(25, 10),
            ]
        )
        events, response = await stream(provider, HI)

        assert _texts(events) == ["Hi"]
        assert response.usage == Usage(input_tokens=25, output_tokens=10)
        assert response.stop_reason == "stop"


class TestOpenAIStreamReasoning:
    """Reasoning deltas from local OpenAI-compatible servers."""

    async def test_reasoning_not_yielded_as_text(self):
        provider = _openai(
            [
                openai_chunk(reasoning="Let me "),
                openai_chunk(reasoning="think."),
                openai_chunk(content="Hi"),
                openai_chunk(finish_reason="stop"),
            ],
            model="gemma4:e4b",
        )
        events, response = await stream(provider, HI)

        assert _texts(events) == ["Hi"]
        assert response.thinking == (ThinkingBlock(text="Let me think."),)

    async def test_reasoning_field_alt_spelling(self):
        provider = _openai(
            [
                openai_chunk(reasoning="Hmm.", reasoning_field="reasoning"),
                openai_chunk(content="Hi"),
                openai_chunk(finish_reason="stop"),
            ],
            model="gemma4:e4b",
        )
        events, response = await stream(provider, HI)

        assert _texts(events) == ["Hi"]
        assert response.thinking == (ThinkingBlock(text="Hmm."),)

    async def test_empty_reasoning_ignored(self):
        provider = _openai(
            [openai_chunk(reasoning=""), openai_chunk(content="Hi", finish_reason="stop")],
            model="gemma4:e4b",
        )
        events, response = await stream(provider, HI)

        assert [event.kind for event in events] == ["text"]
        assert response.thinking == ()

    async def test_no_reasoning_thinking_empty(self):
        provider = _openai([openai_chunk(content="Hi"), openai_chunk(finish_reason="stop")])
        _, response = await stream(provider, HI)

        assert response.thinking == ()


class TestOpenAIStreamEvents:
    async def test_thinking_events_realtime(self):
        provider = _openai(
            [
                openai_chunk(reasoning="Let me "),
                openai_chunk(reasoning="think."),
                openai_chunk(content="Hi"),
                openai_chunk(finish_reason="stop"),
            ],
            model="gemma4:e4b",
        )
        events, response = await stream(provider, HI)

        assert [e.kind for e in events] == ["thinking", "thinking", "text"]
        assert [e.thinking.text for e in events if e.thinking] == ["Let me ", "think."]
        # Reasoning events are incremental fragments.
        assert all(e.partial for e in events if e.kind == "thinking")
        assert events[2].text == "Hi"
        assert response.thinking == (ThinkingBlock(text="Let me think."),)

    async def test_reasoning_and_content_same_chunk(self):
        provider = _openai(
            [
                openai_chunk(reasoning="Thinking.", content="Hi"),
                openai_chunk(finish_reason="stop"),
            ],
            model="gemma4:e4b",
        )
        events, _ = await stream(provider, HI)

        assert [e.kind for e in events] == ["thinking", "text"]

    async def test_tool_call_events_come_from_the_final_completion(self):
        provider = _openai(
            [
                openai_chunk(tool_calls=[{"index": 0, "id": "tc_1", "name": "get_weather"}]),
                openai_chunk(
                    tool_calls=[{"index": 0, "arguments": '{"city": "NYC"}'}],
                    finish_reason="tool_calls",
                ),
            ]
        )
        events, response = await stream(provider, HI)

        tool_events = [e for e in events if e.kind == "tool_call"]
        call = ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"})
        assert [event.tool_call for event in tool_events] == [call]
        assert response.tool_calls == (call,)

    async def test_usage_and_stop_reason_tracked(self):
        provider = _openai(
            [
                openai_chunk(content="Hi"),
                openai_chunk(finish_reason="stop"),
                openai_usage_chunk(25, 10),
            ]
        )
        _, response = await stream(provider, HI)

        assert response.usage == Usage(input_tokens=25, output_tokens=10)
        assert response.stop_reason == "stop"

    async def test_thinking_preserved_on_early_break(self):
        # Reasoning precedes content; an LLM stream abandoned at the first text still reports
        # the reasoning so far in its partial response (as the Anthropic adapter does).
        llm = LLM("gemma4:e4b", base_url="http://localhost:11434/v1")
        llm._provider = _openai(
            [
                openai_chunk(reasoning="Let me "),
                openai_chunk(reasoning="think."),
                openai_chunk(content="Hi"),
                openai_chunk(finish_reason="stop"),
            ],
            model="gemma4:e4b",
        )
        async with llm.stream_events("Hi") as events:
            async for event in events:
                if event.kind == "text":
                    break

        assert events.response is not None
        assert events.response.text == "Hi"
        assert events.response.thinking == (ThinkingBlock(text="Let me think."),)

    async def test_tool_calls_flushed_on_finish_stop(self):
        # Some OpenAI-compatible servers end tool-call turns with "stop".
        provider = _openai(
            [
                openai_chunk(tool_calls=[{"index": 0, "id": "tc_1", "name": "get_weather"}]),
                openai_chunk(
                    tool_calls=[{"index": 0, "arguments": '{"city": "NYC"}'}],
                    finish_reason="stop",
                ),
            ]
        )
        events, response = await stream(provider, HI)

        call = ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"})
        assert [e.tool_call for e in events if e.kind == "tool_call"] == [call]
        assert response.tool_calls == (call,)


# ---------------------------------------------------------------------------
# LLM-level stream tool calls
# ---------------------------------------------------------------------------


def _llm_with_tool_call(call: ToolCall, text: str, stop_reason: str) -> tuple[LLM, FakeProvider]:
    final = Response(
        text=text,
        tool_calls=(call,),
        stop_reason=stop_reason,
        usage=Usage(input_tokens=10, output_tokens=5),
    )
    llm = LLM("claude-sonnet-4-20250514", api_key="test")
    provider = FakeProvider(Reply(response=final), model="claude-sonnet-4-20250514")
    llm._provider = provider
    return llm, provider


class TestLLMStreamToolCalls:
    async def test_stream_with_tools(self):
        """StreamResponse.response.tool_calls after consumption."""
        call = ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"})
        llm, _ = _llm_with_tool_call(call, "Let me check.", "tool_use")
        stream_response = llm.stream("Hi", tools=[{"name": "get_weather"}])

        chunks = [chunk async for chunk in stream_response]

        assert chunks == ["Let me check."]
        assert stream_response.response is not None
        assert stream_response.response.tool_calls == (call,)
        assert stream_response.response.stop_reason == "tool_use"

    async def test_stream_tools_forwarded_to_provider(self):
        """The tools reach the provider's request."""
        call = ToolCall(id="tc_1", name="search", input={})
        llm, provider = _llm_with_tool_call(call, "Hi", "end_turn")
        tools = [{"name": "search", "description": "Search", "input_schema": {"type": "object"}}]
        async for _ in llm.stream("Hi", tools=tools):
            pass

        assert provider.last.tools == tools

    def test_stream_sync_with_tools(self):
        """SyncStreamResponse.response.tool_calls after consumption."""
        call = ToolCall(id="tc_1", name="get_time", input={"tz": "UTC"})
        llm, _ = _llm_with_tool_call(call, "Checking.", "tool_use")
        stream_response = llm.stream_sync("Hi", tools=[{"name": "get_time"}])

        assert list(stream_response) == ["Checking."]
        assert stream_response.response is not None
        assert stream_response.response.tool_calls == (call,)
