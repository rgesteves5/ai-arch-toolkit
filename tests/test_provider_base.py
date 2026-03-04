"""Tests for _providers/_base.py."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ai_arch_toolkit.core._providers._base import BaseProvider, StreamState
from ai_arch_toolkit.core._response import Response, ThinkingBlock, ToolCall


class _DummyProvider(BaseProvider):
    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Response:
        return Response(text="ok")

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[AsyncIterator[str], StreamState]:
        state = StreamState()
        state.thinking.append(ThinkingBlock(text="thought"))
        state.tool_calls.append(ToolCall(id="tc_1", name="search", input={"q": "x"}))

        async def _gen() -> AsyncIterator[str]:
            yield "hello"
            yield " world"

        return _gen(), state


async def test_stream_events_wraps_text_thinking_and_tool_calls() -> None:
    provider = _DummyProvider()
    events, _state = provider.stream_events([{"role": "user", "content": "Hi"}])
    collected = [event async for event in events]
    assert [event.kind for event in collected] == ["text", "text", "thinking", "tool_call"]
    assert collected[0].text == "hello"
    assert collected[2].thinking is not None
    assert collected[2].thinking.text == "thought"
    assert collected[3].tool_call is not None
    assert collected[3].tool_call.name == "search"
