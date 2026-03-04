"""Tests for streaming structured output (partial_parsed)."""

from __future__ import annotations

from ai_arch_toolkit.core._response import Response, StreamResponse


async def _async_gen(*chunks: str):
    for c in chunks:
        yield c


class TestPartialParsed:
    async def test_partial_parsed_updates_on_valid_json(self):
        stream = StreamResponse(
            _async_gen('{"key"', ': "value"}'),
            lambda text: Response(text=text),
        )
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        # After all chunks, partial_parsed should be the full object
        assert stream.partial_parsed == {"key": "value"}

    async def test_partial_parsed_none_for_non_json(self):
        stream = StreamResponse(
            _async_gen("Hello", " world"),
            lambda text: Response(text=text),
        )
        async for _ in stream:
            pass
        assert stream.partial_parsed is None

    async def test_partial_parsed_intermediate(self):
        stream = StreamResponse(
            _async_gen("42"),
            lambda text: Response(text=text),
        )
        async for _ in stream:
            pass
        # "42" is valid JSON
        assert stream.partial_parsed == 42

    async def test_partial_parsed_starts_none(self):
        stream = StreamResponse(
            _async_gen('{"a": 1}'),
            lambda text: Response(text=text),
        )
        assert stream.partial_parsed is None
