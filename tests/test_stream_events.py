"""Tests for rich streaming events (StreamEvent, RichStreamResponse)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import MagicMock, patch

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._providers._base import StreamState
from ai_arch_toolkit.core._response import (
    StreamEvent,
    ThinkingBlock,
    ToolCall,
    Usage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    *,
    usage: Usage | None = None,
    model: str = "test-model",
    stop_reason: str = "end_turn",
    tool_calls: list[ToolCall] | None = None,
    thinking: list[ThinkingBlock] | None = None,
) -> StreamState:
    state = StreamState()
    state.usage = usage or Usage(input_tokens=10, output_tokens=5)
    state.model = model
    state.stop_reason = stop_reason
    state.tool_calls = tool_calls or []
    state.thinking = thinking or []
    return state


# ---------------------------------------------------------------------------
# 1. Text-only streaming yields StreamEvent(kind="text")
# ---------------------------------------------------------------------------


async def test_text_only_streaming():
    events_list = [
        StreamEvent(kind="text", text="Hello"),
        StreamEvent(kind="text", text=" world"),
    ]
    state = _make_state()

    async def _fake_events() -> AsyncIterator[StreamEvent]:
        for e in events_list:
            yield e

    with patch("ai_arch_toolkit.core._llm.create_provider") as mock_cp:
        provider = MagicMock()
        provider.stream_events.return_value = (_fake_events(), state)
        mock_cp.return_value = provider

        llm = LLM("test-model")
        stream = llm.stream_events("Hello")

        collected: list[StreamEvent] = []
        async for event in stream:
            collected.append(event)

    assert len(collected) == 2
    assert all(e.kind == "text" for e in collected)
    assert collected[0].text == "Hello"
    assert collected[1].text == " world"


# ---------------------------------------------------------------------------
# 2. Finalization produces Response
# ---------------------------------------------------------------------------


async def test_finalization_produces_response():
    events_list = [
        StreamEvent(kind="text", text="Test"),
    ]
    state = _make_state(usage=Usage(input_tokens=20, output_tokens=10))

    async def _fake_events() -> AsyncIterator[StreamEvent]:
        for e in events_list:
            yield e

    with patch("ai_arch_toolkit.core._llm.create_provider") as mock_cp:
        provider = MagicMock()
        provider.stream_events.return_value = (_fake_events(), state)
        mock_cp.return_value = provider

        llm = LLM("test-model")
        stream = llm.stream_events("Hello")

        async for _ in stream:
            pass

        assert stream.response is not None
        assert stream.response.text == "Test"
        assert stream.response.usage.input_tokens == 20


# ---------------------------------------------------------------------------
# 3. Sync wrapper works
# ---------------------------------------------------------------------------


def test_sync_wrapper():
    events_list = [
        StreamEvent(kind="text", text="Sync"),
    ]
    state = _make_state()

    async def _fake_events() -> AsyncIterator[StreamEvent]:
        for e in events_list:
            yield e

    with patch("ai_arch_toolkit.core._llm.create_provider") as mock_cp:
        provider = MagicMock()
        provider.stream_events.return_value = (_fake_events(), state)
        mock_cp.return_value = provider

        llm = LLM("test-model")
        stream = llm.stream_events_sync("Hello")

        collected = list(stream)

    assert len(collected) == 1
    assert collected[0].kind == "text"
    assert stream.response is not None


# ---------------------------------------------------------------------------
# 4. Events with thinking blocks
# ---------------------------------------------------------------------------


async def test_events_with_thinking():
    thinking_block = ThinkingBlock(text="Let me think...")
    events_list = [
        StreamEvent(kind="thinking", thinking=thinking_block),
        StreamEvent(kind="text", text="Answer"),
    ]
    state = _make_state(thinking=[thinking_block])

    async def _fake_events() -> AsyncIterator[StreamEvent]:
        for e in events_list:
            yield e

    with patch("ai_arch_toolkit.core._llm.create_provider") as mock_cp:
        provider = MagicMock()
        provider.stream_events.return_value = (_fake_events(), state)
        mock_cp.return_value = provider

        llm = LLM("test-model")
        stream = llm.stream_events("Hello")

        collected: list[StreamEvent] = []
        async for event in stream:
            collected.append(event)

    assert len(collected) == 2
    assert collected[0].kind == "thinking"
    assert collected[0].thinking is not None
    assert collected[1].kind == "text"


# ---------------------------------------------------------------------------
# 5. Events with tool calls
# ---------------------------------------------------------------------------


async def test_events_with_tool_calls():
    tc = ToolCall(id="tc_1", name="search", input={"query": "test"})
    events_list = [
        StreamEvent(kind="text", text="Let me search"),
        StreamEvent(kind="tool_call", tool_call=tc),
    ]
    state = _make_state(tool_calls=[tc])

    async def _fake_events() -> AsyncIterator[StreamEvent]:
        for e in events_list:
            yield e

    with patch("ai_arch_toolkit.core._llm.create_provider") as mock_cp:
        provider = MagicMock()
        provider.stream_events.return_value = (_fake_events(), state)
        mock_cp.return_value = provider

        llm = LLM("test-model")
        stream = llm.stream_events("Hello")

        collected: list[StreamEvent] = []
        async for event in stream:
            collected.append(event)

    assert len(collected) == 2
    assert collected[1].kind == "tool_call"
    assert collected[1].tool_call is not None
    assert collected[1].tool_call.name == "search"


# ---------------------------------------------------------------------------
# 6. Mixed event types (thinking + text + tool_call)
# ---------------------------------------------------------------------------


async def test_mixed_event_types():
    thinking_block = ThinkingBlock(text="Reasoning...")
    tc = ToolCall(id="tc_1", name="search", input={"q": "test"})
    events_list = [
        StreamEvent(kind="thinking", thinking=thinking_block),
        StreamEvent(kind="text", text="I'll search"),
        StreamEvent(kind="tool_call", tool_call=tc),
        StreamEvent(kind="text", text=" for you"),
    ]
    state = _make_state(thinking=[thinking_block], tool_calls=[tc])

    async def _fake_events() -> AsyncIterator[StreamEvent]:
        for e in events_list:
            yield e

    with patch("ai_arch_toolkit.core._llm.create_provider") as mock_cp:
        provider = MagicMock()
        provider.stream_events.return_value = (_fake_events(), state)
        mock_cp.return_value = provider

        llm = LLM("test-model")
        stream = llm.stream_events("Hello")

        collected: list[StreamEvent] = []
        async for event in stream:
            collected.append(event)

    assert len(collected) == 4
    kinds = [e.kind for e in collected]
    assert kinds == ["thinking", "text", "tool_call", "text"]
    # Text chunks should be concatenated in response
    assert stream.response is not None
    assert stream.response.text == "I'll search for you"


# ---------------------------------------------------------------------------
# 7. Multiple text chunks are concatenated in response
# ---------------------------------------------------------------------------


async def test_text_chunk_concatenation():
    events_list = [
        StreamEvent(kind="text", text="Hello"),
        StreamEvent(kind="text", text=" "),
        StreamEvent(kind="text", text="world"),
    ]
    state = _make_state()

    async def _fake_events() -> AsyncIterator[StreamEvent]:
        for e in events_list:
            yield e

    with patch("ai_arch_toolkit.core._llm.create_provider") as mock_cp:
        provider = MagicMock()
        provider.stream_events.return_value = (_fake_events(), state)
        mock_cp.return_value = provider

        llm = LLM("test-model")
        stream = llm.stream_events("Hello")
        async for _ in stream:
            pass

    assert stream.response.text == "Hello world"


# ---------------------------------------------------------------------------
# 8. Empty stream produces empty response text
# ---------------------------------------------------------------------------


async def test_empty_stream():
    state = _make_state()

    async def _fake_events() -> AsyncIterator[StreamEvent]:
        return
        yield  # makes this an async generator

    with patch("ai_arch_toolkit.core._llm.create_provider") as mock_cp:
        provider = MagicMock()
        provider.stream_events.return_value = (_fake_events(), state)
        mock_cp.return_value = provider

        llm = LLM("test-model")
        stream = llm.stream_events("Hello")
        async for _ in stream:
            pass

    assert stream.response is not None
    assert stream.response.text == ""
