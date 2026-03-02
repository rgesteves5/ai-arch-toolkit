"""Tests for MemoryMiddleware."""

from __future__ import annotations

from unittest.mock import AsyncMock

from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.toolkit.memory._middleware import MemoryMiddleware
from ai_arch_toolkit.toolkit.memory._types import Node, SearchResult


def _make_request(user_text: str = "hello") -> Request:
    return Request(
        messages=[{"role": "user", "content": user_text}],
        system=None,
        tools=None,
        model="test",
    )


def _make_response(text: str = "reply") -> Response:
    return Response(
        text=text,
        tool_calls=[],
        usage=Usage(input_tokens=0, output_tokens=0),
        model="test",
        raw={},
    )


class TestAbefore:
    async def test_injects_memories(self):
        node = Node(content={"text": "user likes python"})
        find = AsyncMock(return_value=[SearchResult(node=node, score=0.9)])
        record = AsyncMock()
        mw = MemoryMiddleware(find=find, record=record, k=3)
        request = _make_request("what do I like?")
        result = await mw.abefore(request)
        assert "user likes python" in (result.system or "")
        find.assert_called_once_with("what do I like?", k=3)

    async def test_no_memories_found(self):
        find = AsyncMock(return_value=[])
        record = AsyncMock()
        mw = MemoryMiddleware(find=find, record=record)
        request = _make_request("test")
        result = await mw.abefore(request)
        # System should remain unchanged (None or empty)
        assert result.system is None or result.system == ""


class TestAafter:
    async def test_records_interaction(self):
        find = AsyncMock(return_value=[])
        record = AsyncMock()
        mw = MemoryMiddleware(find=find, record=record)
        request = _make_request("what is python?")
        response = _make_response("Python is a language")
        await mw.aafter(request, response)
        record.assert_called_once()
        call_args = record.call_args[0][0]
        assert "python" in call_args["query"]
        assert "Python is a language" in call_args["response_summary"]
