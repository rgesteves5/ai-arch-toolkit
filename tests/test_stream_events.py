"""Tests for rich streaming events (StreamEvent, RichStreamResponse)."""

from __future__ import annotations

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._response import (
    Response,
    StreamEvent,
    ThinkingBlock,
    ToolCall,
    Usage,
)
from tests.fake_provider import MODEL, FakeProvider, Reply, fake_llm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

USAGE = Usage(input_tokens=10, output_tokens=5)


class _ThinkingProvider(FakeProvider):
    """Streams the reply's thinking blocks before its text, as a reasoning model does."""

    def __init__(self, reply: Reply) -> None:
        super().__init__(reply, model=MODEL)
        self._thinking = reply.response.thinking

    async def open_stream(self, prepared):
        for block in self._thinking:
            yield StreamEvent(kind="thinking", thinking=block)
        async for item in super().open_stream(prepared):
            yield item


def _thinking_llm(reply: Reply) -> LLM:
    llm = LLM(MODEL, api_key="test")
    llm._provider = _ThinkingProvider(reply)
    return llm


# ---------------------------------------------------------------------------
# 1. Text-only streaming yields StreamEvent(kind="text")
# ---------------------------------------------------------------------------


async def test_text_only_streaming():
    response = Response(text="Hello world", usage=USAGE)
    llm, _ = fake_llm(Reply(response=response, chunks=["Hello", " world"]))
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
    llm, _ = fake_llm(Response(text="Test", usage=Usage(input_tokens=20, output_tokens=10)))
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
    llm, _ = fake_llm(Response(text="Sync", usage=USAGE))
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
    response = Response(text="Answer", thinking=(thinking_block,), usage=USAGE)
    stream = _thinking_llm(Reply(response=response)).stream_events("Hello")

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
    llm, _ = fake_llm(Response(text="Let me search", tool_calls=(tc,), usage=USAGE))
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
    response = Response(
        text="I'll search for you", thinking=(thinking_block,), tool_calls=(tc,), usage=USAGE
    )
    reply = Reply(response=response, chunks=["I'll search", " for you"])
    stream = _thinking_llm(reply).stream_events("Hello")

    collected: list[StreamEvent] = []
    async for event in stream:
        collected.append(event)

    assert len(collected) == 4
    kinds = [e.kind for e in collected]
    # Tool calls come from the assembled response, so they follow the text.
    assert kinds == ["thinking", "text", "text", "tool_call"]
    # The response holds the whole text that was streamed
    assert stream.response is not None
    assert stream.response.text == "I'll search for you"
    assert "".join(e.text for e in collected if e.kind == "text") == stream.response.text


# ---------------------------------------------------------------------------
# 7. Multiple text chunks add up to the response text
# ---------------------------------------------------------------------------


async def test_text_chunk_concatenation():
    response = Response(text="Hello world", usage=USAGE)
    llm, _ = fake_llm(Reply(response=response, chunks=["Hello", " ", "world"]))
    stream = llm.stream_events("Hello")
    texts = [event.text async for event in stream]

    assert texts == ["Hello", " ", "world"]
    assert stream.response is not None
    assert stream.response.text == "".join(texts) == "Hello world"


# ---------------------------------------------------------------------------
# 8. Empty stream produces empty response text
# ---------------------------------------------------------------------------


async def test_empty_stream():
    llm, _ = fake_llm(Response(text="", usage=USAGE))
    stream = llm.stream_events("Hello")
    async for _ in stream:
        pass

    assert stream.response is not None
    assert stream.response.text == ""
