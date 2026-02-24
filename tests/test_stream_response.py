"""Tests for StreamResponse and SyncStreamResponse."""

from __future__ import annotations

from ai_arch_toolkit.core._response import Response, StreamResponse, SyncStreamResponse, Usage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_finalizer():
    """Create a simple finalizer that builds a Response from text."""

    def finalizer(text: str) -> Response:
        return Response(
            text=text,
            usage=Usage(input_tokens=10, output_tokens=5),
            cost=0.001,
            cost_estimated=True,
            stop_reason="end_turn",
            model="test-model",
        )

    return finalizer


async def _async_gen(*chunks: str):
    for chunk in chunks:
        yield chunk


# ---------------------------------------------------------------------------
# StreamResponse (async)
# ---------------------------------------------------------------------------


class TestStreamResponse:
    async def test_yields_chunks(self):
        stream = StreamResponse(_async_gen("Hello", " ", "world"), _make_finalizer())
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        assert chunks == ["Hello", " ", "world"]

    async def test_response_none_before_consume(self):
        stream = StreamResponse(_async_gen("Hi"), _make_finalizer())
        assert stream.response is None

    async def test_response_available_after_consume(self):
        stream = StreamResponse(_async_gen("Hello", " world"), _make_finalizer())
        async for _ in stream:
            pass
        assert stream.response is not None
        assert stream.response.text == "Hello world"
        assert stream.response.usage.input_tokens == 10
        assert stream.response.cost == 0.001
        assert stream.response.model == "test-model"

    async def test_empty_stream(self):
        stream = StreamResponse(_async_gen(), _make_finalizer())
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        assert chunks == []
        assert stream.response is not None
        assert stream.response.text == ""

    async def test_context_manager_full_consume(self):
        async with StreamResponse(_async_gen("Hello"), _make_finalizer()) as stream:
            async for _ in stream:
                pass
        assert stream.response is not None
        assert stream.response.text == "Hello"

    async def test_context_manager_early_exit(self):
        async with StreamResponse(_async_gen("a", "b", "c"), _make_finalizer()) as stream:
            async for chunk in stream:
                if chunk == "a":
                    break

        # Response finalized with partial content
        assert stream.response is not None
        assert stream.response.text == "a"

    async def test_context_manager_no_iteration(self):
        """Context manager exit without iteration gives empty response."""
        async with StreamResponse(_async_gen("Hello"), _make_finalizer()) as stream:
            pass
        assert stream.response is not None
        assert stream.response.text == ""

    async def test_response_not_overwritten_on_exit(self):
        """If stream was fully consumed, __aexit__ doesn't overwrite response."""
        stream = StreamResponse(_async_gen("Hello"), _make_finalizer())
        async for _ in stream:
            pass
        assert stream.response is not None
        original = stream.response

        # Simulate __aexit__
        await stream.__aexit__(None, None, None)
        # Response should not be overwritten (it was already set)
        assert stream.response is original


# ---------------------------------------------------------------------------
# SyncStreamResponse
# ---------------------------------------------------------------------------


class TestSyncStreamResponse:
    def test_yields_chunks(self):
        stream = SyncStreamResponse(iter(["a", "b", "c"]), _make_finalizer())
        chunks = list(stream)
        assert chunks == ["a", "b", "c"]

    def test_response_available_after_consume(self):
        stream = SyncStreamResponse(iter(["Hello", " world"]), _make_finalizer())
        list(stream)
        assert stream.response is not None
        assert stream.response.text == "Hello world"

    def test_response_none_before_consume(self):
        stream = SyncStreamResponse(iter(["Hi"]), _make_finalizer())
        assert stream.response is None

    def test_empty_stream(self):
        stream = SyncStreamResponse(iter([]), _make_finalizer())
        chunks = list(stream)
        assert chunks == []
        assert stream.response is not None
        assert stream.response.text == ""

    def test_partial_consume(self):
        stream = SyncStreamResponse(iter(["a", "b", "c"]), _make_finalizer())
        for chunk in stream:
            if chunk == "a":
                break
        # Response not finalized since generator wasn't exhausted
        # (but that's Python generator behavior — break exits, no finalizer)
        # The response would be None
        assert stream.response is None
