"""Tests for _providers/_base.py."""

from __future__ import annotations

from collections.abc import AsyncIterator

from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._providers._base import Answer, Done
from ai_arch_toolkit.core._response import Response, StreamEvent, ThinkingBlock, ToolCall
from tests.fake_provider import FakeProvider, Reply


class _ThinkingProvider(FakeProvider):
    """Streams a thought before the text; the tool call is in the final response."""

    async def open_stream(self, prepared: Request) -> AsyncIterator[StreamEvent | Done[Reply]]:
        yield StreamEvent(kind="thinking", thinking=ThinkingBlock(text="thought"))
        async for item in super().open_stream(prepared):
            yield item


async def test_stream_events_wraps_text_thinking_and_tool_calls() -> None:
    call = ToolCall(id="tc_1", name="search", input={"q": "x"})
    final = Response(text="hello world", tool_calls=(call,))
    provider = _ThinkingProvider(Reply(response=final, chunks=["hello", " world"]))
    request = Request(
        messages=[{"role": "user", "content": "Hi"}],
        system=None,
        tools=None,
        model=provider._model,
    )
    items = [item async for item in provider.stream(provider.prepare(request))]
    events = [item for item in items if isinstance(item, StreamEvent)]
    assert [event.kind for event in events] == ["thinking", "text", "text", "tool_call"]
    assert events[1].text == "hello"
    assert events[0].thinking is not None
    assert events[0].thinking.text == "thought"
    assert events[3].tool_call == call
    assert isinstance(items[-1], Answer)
    assert items[-1].response.tool_calls == (call,)
