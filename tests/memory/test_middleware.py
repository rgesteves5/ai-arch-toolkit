"""Tests for MemoryMiddleware."""

from __future__ import annotations

from unittest.mock import AsyncMock

from ai_arch_toolkit.core._content import cache, document, image, user
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


class TestARequestWithParts:
    """A message with an image searches by its text (G-38)."""

    async def test_a_request_with_an_image_searches_by_its_text(self):
        node = Node(content={"text": "the user's cat is called Miso"})
        find = AsyncMock(return_value=[SearchResult(node=node, score=0.9)])
        mw = MemoryMiddleware(find=find, record=AsyncMock(), k=3)
        message = user(["What is my cat called?", image(b"\x89PNG", "image/png")])
        request = Request(messages=[message], system="Be brief.", tools=None, model="test")

        result = await mw.abefore(request)

        find.assert_called_once_with("What is my cat called?", k=3)
        assert result.system is not None
        assert "Miso" in result.system and result.system.endswith("Be brief.")
        assert result.messages == [message]

    async def test_every_kind_of_text_part_counts_and_nothing_else(self):
        find = AsyncMock(return_value=[])
        mw = MemoryMiddleware(find=find, record=AsyncMock())
        content = [
            "Compare",
            image("https://example.com/a.png"),
            {"type": "text", "text": "these two"},
            document(b"%PDF-1.7"),
            cache("charts"),
            {"type": "image", "source": {}},
        ]
        await mw.abefore(Request(messages=[user(content)], system=None, tools=None, model="t"))

        find.assert_called_once_with("Compare these two charts", k=3)

    async def test_a_message_with_only_an_image_searches_nothing(self):
        find = AsyncMock()
        mw = MemoryMiddleware(find=find, record=AsyncMock())
        request = Request(
            messages=[user([image(b"\x89PNG")])], system=None, tools=None, model="test"
        )

        assert await mw.abefore(request) is request
        find.assert_not_called()

    async def test_the_interaction_is_recorded_by_its_text(self):
        record = AsyncMock()
        mw = MemoryMiddleware(find=AsyncMock(return_value=[]), record=record)
        request = Request(
            messages=[user(["Describe this", image(b"\x89PNG")])],
            system=None,
            tools=None,
            model="test",
        )

        await mw.aafter(request, _make_response("A cat on a sofa."))

        assert record.call_args[0][0] == {
            "query": "Describe this",
            "response_summary": "A cat on a sofa.",
        }
