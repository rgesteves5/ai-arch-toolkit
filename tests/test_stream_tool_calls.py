"""Tests for tool call streaming — Phase 4 (updated for SDK adapters)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from ai_arch_toolkit.core._providers._anthropic import AnthropicProvider
from ai_arch_toolkit.core._providers._base import StreamState
from ai_arch_toolkit.core._providers._openai import OpenAIProvider
from ai_arch_toolkit.core._response import ThinkingBlock, ToolCall, Usage

# ---------------------------------------------------------------------------
# Helpers — build fake SDK stream events
# ---------------------------------------------------------------------------


def _sdk_event(event_type: str, **kwargs) -> SimpleNamespace:
    """Build a fake SDK stream event."""
    ns = SimpleNamespace(type=event_type)
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns


def _content_block_start(index: int, block_type: str, **block_attrs) -> SimpleNamespace:
    block = SimpleNamespace(type=block_type, **block_attrs)
    return _sdk_event("content_block_start", index=index, content_block=block)


def _text_delta(index: int, text: str) -> SimpleNamespace:
    delta = SimpleNamespace(type="text_delta", text=text)
    return _sdk_event("content_block_delta", index=index, delta=delta)


def _input_json_delta(index: int, partial_json: str) -> SimpleNamespace:
    delta = SimpleNamespace(type="input_json_delta", partial_json=partial_json)
    return _sdk_event("content_block_delta", index=index, delta=delta)


def _block_stop(index: int) -> SimpleNamespace:
    return _sdk_event("content_block_stop", index=index)


def _message_start(model: str = "claude-sonnet-4-20250514", **usage_kwargs) -> SimpleNamespace:
    usage = SimpleNamespace(**usage_kwargs) if usage_kwargs else SimpleNamespace(input_tokens=0)
    msg = SimpleNamespace(model=model, usage=usage)
    return _sdk_event("message_start", message=msg)


def _message_delta(stop_reason: str = "", **usage_kwargs) -> SimpleNamespace:
    delta = SimpleNamespace(stop_reason=stop_reason)
    usage = SimpleNamespace(**usage_kwargs) if usage_kwargs else None
    return _sdk_event("message_delta", delta=delta, usage=usage)


def _final_message(content=None) -> SimpleNamespace:
    return SimpleNamespace(content=content or [])


class _FakeStream:
    """Fake async context manager mimicking SDK messages.stream()."""

    def __init__(self, events: list, final_message=None):
        self._events = events
        self._final = final_message or _final_message()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def __aiter__(self):
        return self._iter_events()

    async def _iter_events(self):
        for event in self._events:
            yield event

    async def get_final_message(self):
        return self._final


# ---------------------------------------------------------------------------
# Anthropic stream tool calls
# ---------------------------------------------------------------------------


class TestAnthropicStreamToolCalls:
    async def test_single_tool_call(self):
        events = [
            _message_start(input_tokens=25, output_tokens=0),
            _content_block_start(0, "tool_use", id="tc_1", name="get_weather"),
            _input_json_delta(0, '{"city"'),
            _input_json_delta(0, ': "NYC"}'),
            _block_stop(0),
            _message_delta(stop_reason="tool_use", output_tokens=15),
        ]

        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        provider._client = MagicMock()
        provider._client.messages.stream.return_value = _FakeStream(events)

        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        chunks = []
        async for chunk in aiter:
            chunks.append(chunk)

        assert chunks == []
        assert len(state.tool_calls) == 1
        tc = state.tool_calls[0]
        assert isinstance(tc, ToolCall)
        assert tc.id == "tc_1"
        assert tc.name == "get_weather"
        assert tc.input == {"city": "NYC"}
        assert state.stop_reason == "tool_use"

    async def test_text_then_tool_call(self):
        events = [
            _message_start(input_tokens=10, output_tokens=0),
            _content_block_start(0, "text", text=""),
            _text_delta(0, "Let me check."),
            _block_stop(0),
            _content_block_start(1, "tool_use", id="tc_1", name="get_weather"),
            _input_json_delta(1, '{"city": "NYC"}'),
            _block_stop(1),
            _message_delta(stop_reason="tool_use", output_tokens=20),
        ]

        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        provider._client = MagicMock()
        provider._client.messages.stream.return_value = _FakeStream(events)

        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        chunks = []
        async for chunk in aiter:
            chunks.append(chunk)

        assert chunks == ["Let me check."]
        assert len(state.tool_calls) == 1
        assert state.tool_calls[0].name == "get_weather"
        assert state.tool_calls[0].input == {"city": "NYC"}

    async def test_multiple_tool_calls(self):
        events = [
            _message_start(input_tokens=10, output_tokens=0),
            _content_block_start(0, "tool_use", id="tc_1", name="get_weather"),
            _input_json_delta(0, '{"city": "NYC"}'),
            _block_stop(0),
            _content_block_start(1, "tool_use", id="tc_2", name="get_time"),
            _input_json_delta(1, '{"tz": "UTC"}'),
            _block_stop(1),
            _message_delta(stop_reason="tool_use", output_tokens=20),
        ]

        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        provider._client = MagicMock()
        provider._client.messages.stream.return_value = _FakeStream(events)

        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        async for _ in aiter:
            pass

        assert len(state.tool_calls) == 2
        assert state.tool_calls[0].id == "tc_1"
        assert state.tool_calls[0].name == "get_weather"
        assert state.tool_calls[1].id == "tc_2"
        assert state.tool_calls[1].name == "get_time"
        assert state.tool_calls[1].input == {"tz": "UTC"}

    async def test_malformed_tool_args(self):
        events = [
            _content_block_start(0, "tool_use", id="tc_1", name="fn"),
            _input_json_delta(0, "not valid json"),
            _block_stop(0),
        ]

        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        provider._client = MagicMock()
        provider._client.messages.stream.return_value = _FakeStream(events)

        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        async for _ in aiter:
            pass

        assert len(state.tool_calls) == 1
        assert state.tool_calls[0].input == {"_raw": "not valid json"}

    async def test_tools_passed_in_sdk_kwargs(self):
        events = []
        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        provider._client = MagicMock()
        provider._client.messages.stream.return_value = _FakeStream(events)

        tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {"type": "object"},
            }
        ]
        aiter, _ = provider.stream([{"role": "user", "content": "Hi"}], tools=tools)
        async for _ in aiter:
            pass

        call_kwargs = provider._client.messages.stream.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"][0]["name"] == "get_weather"

    async def test_empty_tool_args(self):
        events = [
            _content_block_start(0, "tool_use", id="tc_1", name="get_status"),
            _block_stop(0),
        ]

        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        provider._client = MagicMock()
        provider._client.messages.stream.return_value = _FakeStream(events)

        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        async for _ in aiter:
            pass

        assert len(state.tool_calls) == 1
        assert state.tool_calls[0].input == {}


class TestAnthropicStreamThinking:
    async def test_thinking_blocks_accumulated_from_final_message(self):
        """Thinking blocks are extracted from get_final_message() into state.thinking."""
        thinking_delta = SimpleNamespace(type="thinking_delta", thinking="partial thought")
        events = [
            _message_start(input_tokens=10, output_tokens=0),
            _content_block_start(0, "thinking", thinking="", signature=""),
            _sdk_event("content_block_delta", index=0, delta=thinking_delta),
            _block_stop(0),
            _content_block_start(1, "text", text=""),
            _text_delta(1, "The answer is 42."),
            _block_stop(1),
            _message_delta(stop_reason="end_turn", output_tokens=20),
        ]

        # Final message includes the full thinking block
        final = _final_message(
            content=[
                SimpleNamespace(type="thinking", thinking="Let me reason step by step..."),
                SimpleNamespace(type="text", text="The answer is 42."),
            ]
        )

        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        provider._client = MagicMock()
        provider._client.messages.stream.return_value = _FakeStream(events, final_message=final)

        aiter, state = provider.stream([{"role": "user", "content": "Think about this"}])
        chunks = []
        async for chunk in aiter:
            chunks.append(chunk)

        assert chunks == ["The answer is 42."]
        assert len(state.thinking) == 1
        assert state.thinking[0].text == "Let me reason step by step..."

    async def test_multiple_thinking_blocks(self):
        """Multiple thinking blocks are all accumulated."""
        events = [
            _text_delta(0, "Answer."),
            _block_stop(0),
            _message_delta(stop_reason="end_turn"),
        ]

        final = _final_message(
            content=[
                SimpleNamespace(type="thinking", thinking="First thought"),
                SimpleNamespace(type="thinking", thinking="Second thought"),
                SimpleNamespace(type="text", text="Answer."),
            ]
        )

        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        provider._client = MagicMock()
        provider._client.messages.stream.return_value = _FakeStream(events, final_message=final)

        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        async for _ in aiter:
            pass

        assert len(state.thinking) == 2
        assert state.thinking[0].text == "First thought"
        assert state.thinking[1].text == "Second thought"

    async def test_no_thinking_blocks(self):
        """When no thinking, state.thinking remains empty."""
        events = [
            _text_delta(0, "Hello"),
            _block_stop(0),
            _message_delta(stop_reason="end_turn"),
        ]

        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")
        provider._client = MagicMock()
        provider._client.messages.stream.return_value = _FakeStream(events)

        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        async for _ in aiter:
            pass

        assert state.thinking == []


class TestStreamState:
    def test_initial_state_has_tool_calls(self):
        state = StreamState()
        assert state.tool_calls == []
        assert state.usage is None
        assert state.model == ""
        assert state.stop_reason == ""
        assert state.thinking == []


# ---------------------------------------------------------------------------
# OpenAI stream helpers
# ---------------------------------------------------------------------------


def _oai_chunk(
    *,
    content: str | None = None,
    tool_calls: list[dict] | None = None,
    finish_reason: str | None = None,
    model: str = "gpt-4o",
    usage: dict | None = None,
    reasoning: str | None = None,
    reasoning_field: str = "reasoning_content",
) -> SimpleNamespace:
    """Build a fake OpenAI ChatCompletionChunk-like object."""
    delta = SimpleNamespace(content=content, tool_calls=None)
    if reasoning is not None:
        setattr(delta, reasoning_field, reasoning)
    if tool_calls:
        delta.tool_calls = [
            SimpleNamespace(
                index=tc.get("index", 0),
                id=tc.get("id"),
                function=SimpleNamespace(
                    name=tc.get("name"),
                    arguments=tc.get("arguments"),
                )
                if tc.get("name") is not None or tc.get("arguments") is not None
                else None,
                type=tc.get("type", "function"),
            )
            for tc in tool_calls
        ]
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason, index=0)
    sdk_usage = None
    if usage:
        sdk_usage = SimpleNamespace(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )
    return SimpleNamespace(choices=[choice], model=model, usage=sdk_usage)


def _oai_usage_chunk(prompt_tokens: int = 0, completion_tokens: int = 0) -> SimpleNamespace:
    """Build a final usage-only chunk."""
    return SimpleNamespace(
        choices=[],
        model="gpt-4o",
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


# ---------------------------------------------------------------------------
# OpenAI stream tool calls
# ---------------------------------------------------------------------------


class TestOpenAIStreamToolCalls:
    async def test_single_tool_call(self):
        chunks = [
            _oai_chunk(tool_calls=[{"index": 0, "id": "tc_1", "name": "get_weather"}]),
            _oai_chunk(tool_calls=[{"index": 0, "arguments": '{"city"'}]),
            _oai_chunk(
                tool_calls=[{"index": 0, "arguments": ': "NYC"}'}],
                finish_reason="tool_calls",
            ),
            _oai_usage_chunk(prompt_tokens=25, completion_tokens=15),
        ]

        async def _fake_stream():
            for c in chunks:
                yield c

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _fake_stream()

        provider = OpenAIProvider("gpt-4o", "test-key")
        provider._client = mock_client
        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        text_chunks = []
        async for chunk in aiter:
            text_chunks.append(chunk)

        assert text_chunks == []
        assert len(state.tool_calls) == 1
        tc = state.tool_calls[0]
        assert isinstance(tc, ToolCall)
        assert tc.id == "tc_1"
        assert tc.name == "get_weather"
        assert tc.input == {"city": "NYC"}
        assert state.stop_reason == "tool_calls"

    async def test_multiple_tool_calls(self):
        chunks = [
            _oai_chunk(tool_calls=[{"index": 0, "id": "tc_1", "name": "get_weather"}]),
            _oai_chunk(tool_calls=[{"index": 0, "arguments": '{"city": "NYC"}'}]),
            _oai_chunk(tool_calls=[{"index": 1, "id": "tc_2", "name": "get_time"}]),
            _oai_chunk(
                tool_calls=[{"index": 1, "arguments": '{"tz": "UTC"}'}],
                finish_reason="tool_calls",
            ),
            _oai_usage_chunk(prompt_tokens=30, completion_tokens=20),
        ]

        async def _fake_stream():
            for c in chunks:
                yield c

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _fake_stream()

        provider = OpenAIProvider("gpt-4o", "test-key")
        provider._client = mock_client
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

    async def test_text_then_tool_call(self):
        chunks = [
            _oai_chunk(content="Let me check."),
            _oai_chunk(
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
            _oai_usage_chunk(prompt_tokens=20, completion_tokens=15),
        ]

        async def _fake_stream():
            for c in chunks:
                yield c

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _fake_stream()

        provider = OpenAIProvider("gpt-4o", "test-key")
        provider._client = mock_client
        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        text_chunks = []
        async for chunk in aiter:
            text_chunks.append(chunk)

        assert text_chunks == ["Let me check."]
        assert len(state.tool_calls) == 1
        assert state.tool_calls[0].name == "get_weather"

    async def test_tools_passed_in_sdk_kwargs(self):
        async def _fake_stream():
            yield _oai_chunk(finish_reason="stop")

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _fake_stream()

        tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {"type": "object"},
            }
        ]
        provider = OpenAIProvider("gpt-4o", "test-key")
        provider._client = mock_client
        aiter, _ = provider.stream([{"role": "user", "content": "Hi"}], tools=tools)
        async for _ in aiter:
            pass

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"][0]["type"] == "function"

    async def test_stream_includes_usage_option(self):
        async def _fake_stream():
            yield _oai_chunk(finish_reason="stop")

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _fake_stream()

        provider = OpenAIProvider("gpt-4o", "test-key")
        provider._client = mock_client
        aiter, _ = provider.stream([{"role": "user", "content": "Hi"}])
        async for _ in aiter:
            pass

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True
        assert call_kwargs["stream_options"] == {"include_usage": True}

    async def test_stream_captures_usage(self):
        chunks = [
            _oai_chunk(content="Hi"),
            _oai_chunk(finish_reason="stop"),
            _oai_usage_chunk(prompt_tokens=25, completion_tokens=10),
        ]

        async def _fake_stream():
            for c in chunks:
                yield c

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _fake_stream()

        provider = OpenAIProvider("gpt-4o", "test-key")
        provider._client = mock_client
        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        text_chunks = []
        async for chunk in aiter:
            text_chunks.append(chunk)

        assert text_chunks == ["Hi"]
        assert state.usage is not None
        assert state.usage.input_tokens == 25
        assert state.usage.output_tokens == 10
        assert state.stop_reason == "stop"


# ---------------------------------------------------------------------------
# OpenAI stream reasoning deltas (local OpenAI-compatible servers)
# ---------------------------------------------------------------------------


def _make_openai_provider(chunks: list[SimpleNamespace]) -> OpenAIProvider:
    """Build an OpenAIProvider whose client streams the given fake chunks."""

    async def _fake_stream():
        for c in chunks:
            yield c

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = _fake_stream()
    provider = OpenAIProvider("gemma4:e4b", "not-needed")
    provider._client = mock_client
    return provider


class TestOpenAIStreamReasoning:
    async def test_reasoning_not_yielded_as_text(self):
        provider = _make_openai_provider(
            [
                _oai_chunk(reasoning="Let me "),
                _oai_chunk(reasoning="think."),
                _oai_chunk(content="Hi"),
                _oai_chunk(finish_reason="stop"),
            ]
        )

        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        text_chunks = [chunk async for chunk in aiter]

        assert text_chunks == ["Hi"]
        assert state.thinking == [ThinkingBlock(text="Let me think.")]

    async def test_reasoning_field_alt_spelling(self):
        provider = _make_openai_provider(
            [
                _oai_chunk(reasoning="Hmm.", reasoning_field="reasoning"),
                _oai_chunk(content="Hi"),
                _oai_chunk(finish_reason="stop"),
            ]
        )

        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        text_chunks = [chunk async for chunk in aiter]

        assert text_chunks == ["Hi"]
        assert state.thinking == [ThinkingBlock(text="Hmm.")]

    async def test_empty_reasoning_ignored(self):
        provider = _make_openai_provider(
            [
                _oai_chunk(reasoning=""),
                _oai_chunk(content="Hi", finish_reason="stop"),
            ]
        )

        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        async for _ in aiter:
            pass

        assert state.thinking == []

    async def test_no_reasoning_thinking_empty(self):
        provider = _make_openai_provider(
            [
                _oai_chunk(content="Hi"),
                _oai_chunk(finish_reason="stop"),
            ]
        )

        aiter, state = provider.stream([{"role": "user", "content": "Hi"}])
        async for _ in aiter:
            pass

        assert state.thinking == []


class TestOpenAIStreamEvents:
    async def test_thinking_events_realtime(self):
        provider = _make_openai_provider(
            [
                _oai_chunk(reasoning="Let me "),
                _oai_chunk(reasoning="think."),
                _oai_chunk(content="Hi"),
                _oai_chunk(finish_reason="stop"),
            ]
        )

        event_iter, state = provider.stream_events([{"role": "user", "content": "Hi"}])
        events = [event async for event in event_iter]

        assert [e.kind for e in events] == ["thinking", "thinking", "text"]
        assert [e.thinking.text for e in events if e.thinking] == ["Let me ", "think."]
        # Reasoning events are incremental fragments.
        assert all(e.partial for e in events if e.kind == "thinking")
        assert events[2].text == "Hi"
        assert state.thinking == [ThinkingBlock(text="Let me think.")]

    async def test_reasoning_and_content_same_chunk(self):
        provider = _make_openai_provider(
            [
                _oai_chunk(reasoning="Thinking.", content="Hi"),
                _oai_chunk(finish_reason="stop"),
            ]
        )

        event_iter, _ = provider.stream_events([{"role": "user", "content": "Hi"}])
        events = [event async for event in event_iter]

        assert [e.kind for e in events] == ["thinking", "text"]

    async def test_tool_call_events_emitted_realtime(self):
        provider = _make_openai_provider(
            [
                _oai_chunk(tool_calls=[{"index": 0, "id": "tc_1", "name": "get_weather"}]),
                _oai_chunk(
                    tool_calls=[{"index": 0, "arguments": '{"city": "NYC"}'}],
                    finish_reason="tool_calls",
                ),
            ]
        )

        event_iter, state = provider.stream_events([{"role": "user", "content": "Hi"}])
        events = [event async for event in event_iter]

        tool_events = [e for e in events if e.kind == "tool_call"]
        assert len(tool_events) == 1
        assert tool_events[0].tool_call == ToolCall(
            id="tc_1", name="get_weather", input={"city": "NYC"}
        )
        assert state.tool_calls == [tool_events[0].tool_call]

    async def test_usage_and_stop_reason_tracked(self):
        provider = _make_openai_provider(
            [
                _oai_chunk(content="Hi"),
                _oai_chunk(finish_reason="stop"),
                _oai_usage_chunk(prompt_tokens=25, completion_tokens=10),
            ]
        )

        event_iter, state = provider.stream_events([{"role": "user", "content": "Hi"}])
        async for _ in event_iter:
            pass

        assert state.usage is not None
        assert state.usage.input_tokens == 25
        assert state.usage.output_tokens == 10
        assert state.stop_reason == "stop"

    async def test_thinking_preserved_on_early_break(self):
        # Reasoning precedes content; breaking before the end must still leave
        # the reasoning-so-far in state.thinking (matches Anthropic).
        provider = _make_openai_provider(
            [
                _oai_chunk(reasoning="Let me "),
                _oai_chunk(reasoning="think."),
                _oai_chunk(content="Hi"),
                _oai_chunk(finish_reason="stop"),
            ]
        )

        event_iter, state = provider.stream_events([{"role": "user", "content": "Hi"}])
        async for event in event_iter:
            if event.kind == "text":
                break

        assert state.thinking == [ThinkingBlock(text="Let me think.")]

    async def test_tool_calls_flushed_on_finish_stop(self):
        # Some OpenAI-compatible servers end tool-call turns with "stop".
        provider = _make_openai_provider(
            [
                _oai_chunk(tool_calls=[{"index": 0, "id": "tc_1", "name": "get_weather"}]),
                _oai_chunk(
                    tool_calls=[{"index": 0, "arguments": '{"city": "NYC"}'}],
                    finish_reason="stop",
                ),
            ]
        )

        event_iter, state = provider.stream_events([{"role": "user", "content": "Hi"}])
        events = [event async for event in event_iter]

        tool_events = [e for e in events if e.kind == "tool_call"]
        assert len(tool_events) == 1
        assert tool_events[0].tool_call == ToolCall(
            id="tc_1", name="get_weather", input={"city": "NYC"}
        )
        assert state.tool_calls == [tool_events[0].tool_call]


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
            ToolCall(id="tc_1", name="get_weather", input={"city": "NYC"}),
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
