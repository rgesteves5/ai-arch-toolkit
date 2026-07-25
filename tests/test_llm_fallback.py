"""Tests for LLM fallback chains and attempt tracking."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_arch_toolkit.core._exceptions import APIError
from ai_arch_toolkit.core._llm import LLM, PROVIDER_ERRORS
from ai_arch_toolkit.core._response import Attempt, Response, Usage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(text: str = "Hello", model: str = "") -> Response:
    return Response(text=text, usage=Usage(input_tokens=10, output_tokens=5), model=model)


def _make_stream_state(usage=None, model="test-model", stop_reason="end_turn"):
    state = MagicMock()
    state.usage = usage or Usage(input_tokens=10, output_tokens=5)
    state.model = model
    state.stop_reason = stop_reason
    state.tool_calls = []
    state.thinking = []
    state.raw = None
    return state


# ---------------------------------------------------------------------------
# Fallback chain construction
# ---------------------------------------------------------------------------


class TestFallbackConstruction:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_string_fallback_backward_compat(self, mock_create):
        mock_create.return_value = AsyncMock()
        llm = LLM("claude-sonnet-4-20250514", api_key="test", fallback="claude-haiku-4-5-20251001")
        assert len(llm._fallbacks) == 1
        assert llm._fallbacks[0]._model == "claude-haiku-4-5-20251001"
        assert len(llm._owned_fallbacks) == 1

    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_llm_instance_fallback(self, mock_create):
        mock_create.return_value = AsyncMock()
        fb = LLM("gpt-4o", api_key="test")
        llm = LLM("claude-sonnet-4-20250514", api_key="test", fallback=fb)
        assert llm._fallbacks == [fb]
        assert llm._owned_fallbacks == []  # not owned

    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_list_of_mixed_fallbacks(self, mock_create):
        mock_create.return_value = AsyncMock()
        fb_llm = LLM("gpt-4o", api_key="test")
        llm = LLM(
            "claude-sonnet-4-20250514",
            api_key="test",
            fallback=[fb_llm, "claude-haiku-4-5-20251001"],
        )
        assert len(llm._fallbacks) == 2
        assert llm._fallbacks[0] is fb_llm
        assert llm._fallbacks[1]._model == "claude-haiku-4-5-20251001"
        assert len(llm._owned_fallbacks) == 1  # only the string one

    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_nested_fallbacks_flattened(self, mock_create):
        mock_create.return_value = AsyncMock()
        inner = LLM("gpt-4o", api_key="test", fallback="gpt-4.1-nano")
        assert len(inner._fallbacks) == 1  # before flattening into parent

        llm = LLM("claude-sonnet-4-20250514", api_key="test", fallback=inner)
        # inner's fallback should be flattened into llm's chain
        assert len(llm._fallbacks) == 2
        assert llm._fallbacks[0]._model == "gpt-4o"
        assert llm._fallbacks[1]._model == "gpt-4.1-nano"
        # inner's own fallbacks should be cleared
        assert inner._fallbacks == []

    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_no_fallback(self, mock_create):
        mock_create.return_value = AsyncMock()
        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        assert llm._fallbacks == []
        assert llm._owned_fallbacks == []

    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_custom_fallback_on(self, mock_create):
        mock_create.return_value = AsyncMock()
        llm = LLM(
            "claude-sonnet-4-20250514",
            api_key="test",
            fallback_on=(APIError, ValueError),
        )
        assert llm._fallback_on == (APIError, ValueError)

    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_default_fallback_on(self, mock_create):
        mock_create.return_value = AsyncMock()
        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        assert llm._fallback_on == PROVIDER_ERRORS


class TestFallbackLifecycle:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_close_only_owned(self, mock_create):
        mock_create.return_value = AsyncMock()
        user_fb = LLM("gpt-4o", api_key="test")
        llm = LLM(
            "claude-sonnet-4-20250514",
            api_key="test",
            fallback=[user_fb, "claude-haiku-4-5-20251001"],
        )
        owned_fb = llm._owned_fallbacks[0]

        # Spy on close methods
        user_fb_close = AsyncMock()
        user_fb.close = user_fb_close
        owned_fb_close = AsyncMock()
        owned_fb.close = owned_fb_close

        await llm.close()
        user_fb_close.assert_not_called()
        owned_fb_close.assert_called_once()


# ---------------------------------------------------------------------------
# Fallback in complete()
# ---------------------------------------------------------------------------


class TestCompleteWithFallback:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_primary_succeeds_no_fallback_tried(self, mock_create):
        primary_provider = AsyncMock()
        primary_provider.complete.return_value = _make_response("primary")
        fb_provider = AsyncMock()
        fb_provider.complete.return_value = _make_response("fallback")
        mock_create.side_effect = [primary_provider, fb_provider]

        llm = LLM("claude-sonnet-4-20250514", api_key="test", fallback="claude-haiku-4-5-20251001")
        result = await llm.complete("Hi")
        assert result.text == "primary"
        fb_provider.complete.assert_not_called()

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_primary_fails_fallback_succeeds(self, mock_create):
        primary_provider = AsyncMock()
        primary_provider.complete.side_effect = APIError(500, "Server error")
        fb_provider = AsyncMock()
        fb_provider.complete.return_value = _make_response("fallback")
        mock_create.side_effect = [primary_provider, fb_provider]

        llm = LLM("claude-sonnet-4-20250514", api_key="test", fallback="claude-haiku-4-5-20251001")
        result = await llm.complete("Hi")
        assert result.text == "fallback"

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_chain_walks_multiple_fallbacks(self, mock_create):
        p1 = AsyncMock()
        p1.complete.side_effect = APIError(500, "down")
        p2 = AsyncMock()
        p2.complete.side_effect = ConnectionError("refused")
        p3 = AsyncMock()
        p3.complete.return_value = _make_response("third")
        mock_create.side_effect = [p1, p2, p3]

        llm = LLM(
            "model-a",
            api_key="test",
            fallback=["model-b", "model-c"],
        )
        result = await llm.complete("Hi")
        assert result.text == "third"

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_all_fail_raises_last(self, mock_create):
        p1 = AsyncMock()
        p1.complete.side_effect = APIError(500, "down")
        p2 = AsyncMock()
        p2.complete.side_effect = ConnectionError("also down")
        mock_create.side_effect = [p1, p2]

        llm = LLM("model-a", api_key="test", fallback="model-b")
        with pytest.raises(ConnectionError, match="also down"):
            await llm.complete("Hi")

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_non_fallback_error_propagates(self, mock_create):
        provider = AsyncMock()
        provider.complete.side_effect = ValueError("bad input")
        mock_create.return_value = provider

        llm = LLM("model-a", api_key="test", fallback="model-b")
        with pytest.raises(ValueError, match="bad input"):
            await llm.complete("Hi")

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_connection_error_triggers_fallback(self, mock_create):
        p1 = AsyncMock()
        p1.complete.side_effect = ConnectionError("refused")
        p2 = AsyncMock()
        p2.complete.return_value = _make_response("ok")
        mock_create.side_effect = [p1, p2]

        llm = LLM("model-a", api_key="test", fallback="model-b")
        result = await llm.complete("Hi")
        assert result.text == "ok"

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_timeout_error_triggers_fallback(self, mock_create):
        p1 = AsyncMock()
        p1.complete.side_effect = TimeoutError("timed out")
        p2 = AsyncMock()
        p2.complete.return_value = _make_response("ok")
        mock_create.side_effect = [p1, p2]

        llm = LLM("model-a", api_key="test", fallback="model-b")
        result = await llm.complete("Hi")
        assert result.text == "ok"

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_os_error_triggers_fallback(self, mock_create):
        p1 = AsyncMock()
        p1.complete.side_effect = OSError("network down")
        p2 = AsyncMock()
        p2.complete.return_value = _make_response("ok")
        mock_create.side_effect = [p1, p2]

        llm = LLM("model-a", api_key="test", fallback="model-b")
        result = await llm.complete("Hi")
        assert result.text == "ok"

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_no_fallback_raises_directly(self, mock_create):
        provider = AsyncMock()
        provider.complete.side_effect = APIError(500, "down")
        mock_create.return_value = provider

        llm = LLM("model-a", api_key="test")
        with pytest.raises(APIError):
            await llm.complete("Hi")


# ---------------------------------------------------------------------------
# Fallback in stream()
# ---------------------------------------------------------------------------


class TestStreamWithFallback:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_primary_stream_fails_fallback_stream_used(self, mock_create):
        async def _fake_gen():
            for chunk in ["fall", "back"]:
                yield chunk

        p1 = MagicMock()
        p1.stream.side_effect = APIError(500, "stream down")
        p2 = MagicMock()
        state = _make_stream_state(model="model-b")
        p2.stream.return_value = (_fake_gen(), state)
        mock_create.side_effect = [p1, p2]

        llm = LLM("model-a", api_key="test", fallback="model-b")
        stream = llm.stream("Hi")

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        assert chunks == ["fall", "back"]

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_all_streams_fail_raises(self, mock_create):
        p1 = MagicMock()
        p1.stream.side_effect = APIError(500, "down")
        p2 = MagicMock()
        p2.stream.side_effect = ConnectionError("also down")
        mock_create.side_effect = [p1, p2]

        llm = LLM("model-a", api_key="test", fallback="model-b")
        stream = llm.stream("Hi")
        with pytest.raises(ConnectionError):
            async for _ in stream:
                pass


# ---------------------------------------------------------------------------
# Attempt tracking — complete()
# ---------------------------------------------------------------------------


class TestAttemptTrackingComplete:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_primary_succeeds_one_attempt(self, mock_create):
        provider = AsyncMock()
        provider.complete.return_value = _make_response("ok")
        mock_create.return_value = provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        result = await llm.complete("Hi")
        assert len(result.attempts) == 1
        a = result.attempts[0]
        assert a.model == "claude-sonnet-4-20250514"
        assert a.status == "ok"
        assert a.usage is not None
        assert a.timestamp > 0
        assert a.duration >= 0
        assert a.retry_number == 0

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_primary_fails_fallback_succeeds_two_plus_attempts(self, mock_create):
        p1 = AsyncMock()
        p1.complete.side_effect = APIError(500, "down")
        p2 = AsyncMock()
        p2.complete.return_value = _make_response("ok")
        mock_create.side_effect = [p1, p2]

        llm = LLM("model-a", api_key="test", fallback="model-b")
        result = await llm.complete("Hi")
        # Should have: 1 failed attempt on primary + 1 ok attempt on fallback
        assert len(result.attempts) >= 2
        assert result.attempts[0].status == "failed"
        assert result.attempts[0].model == "model-a"
        assert result.attempts[-1].status == "ok"
        assert result.attempts[-1].model == "model-b"

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_attempt_error_type_and_status_code_on_fallback(self, mock_create):
        """Failed primary attempt records error_type and status_code."""
        p1 = AsyncMock()
        p1.complete.side_effect = APIError(503, "unavailable")
        p2 = AsyncMock()
        p2.complete.return_value = _make_response("ok")
        mock_create.side_effect = [p1, p2]

        llm = LLM("model-a", api_key="test", fallback="model-b")
        result = await llm.complete("Hi")
        failed = [a for a in result.attempts if a.status == "failed"]
        assert len(failed) >= 1
        assert failed[0].error_type == "APIError"
        assert failed[0].status_code == 503
        assert "unavailable" in failed[0].error

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_attempt_status_code_captured(self, mock_create):
        p1 = AsyncMock()
        p1.complete.side_effect = APIError(429, "rate limited")
        p2 = AsyncMock()
        p2.complete.return_value = _make_response("ok")
        mock_create.side_effect = [p1, p2]

        llm = LLM("model-a", api_key="test", fallback="model-b")
        result = await llm.complete("Hi")
        failed = [a for a in result.attempts if a.status == "failed"]
        assert len(failed) >= 1
        assert failed[0].status_code == 429
        assert failed[0].error_type == "APIError"

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_attempt_timestamps_are_wallclock(self, mock_create):
        provider = AsyncMock()
        provider.complete.return_value = _make_response("ok")
        mock_create.return_value = provider

        before = time.time()
        llm = LLM("model-a", api_key="test")
        result = await llm.complete("Hi")
        after = time.time()

        assert len(result.attempts) == 1
        assert before <= result.attempts[0].timestamp <= after


# ---------------------------------------------------------------------------
# Attempt tracking — stream()
# ---------------------------------------------------------------------------


class TestAttemptTrackingStream:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_stream_finalized_response_has_attempts(self, mock_create):
        async def _fake_gen():
            yield "hello"

        state = _make_stream_state()
        provider = MagicMock()
        provider.stream.return_value = (_fake_gen(), state)
        mock_create.return_value = provider

        llm = LLM("model-a", api_key="test")
        stream = llm.stream("Hi")
        async for _ in stream:
            pass
        assert stream.response is not None
        assert len(stream.response.attempts) == 1
        assert stream.response.attempts[0].status == "ok"

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_stream_attempt_timestamp_starts_with_iteration(self, mock_create):
        async def _fake_gen():
            yield "hello"

        provider = MagicMock()
        provider.stream.return_value = (_fake_gen(), _make_stream_state())
        mock_create.return_value = provider

        stream = LLM("model-a", api_key="test").stream("Hi")
        iteration_started = time.time()
        async for _ in stream:
            pass

        assert stream.response is not None
        assert stream.response.attempts[0].timestamp >= iteration_started

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_stream_fallback_has_merged_attempts(self, mock_create):
        async def _fake_gen():
            yield "ok"

        p1 = MagicMock()
        p1.stream.side_effect = APIError(500, "down")
        p2 = MagicMock()
        state = _make_stream_state(model="model-b")
        p2.stream.return_value = (_fake_gen(), state)
        mock_create.side_effect = [p1, p2]

        llm = LLM("model-a", api_key="test", fallback="model-b")
        stream = llm.stream("Hi")
        async for _ in stream:
            pass
        assert stream.response is not None
        attempts = stream.response.attempts
        # Should have parent's failed attempt + fallback's ok attempt
        assert len(attempts) >= 2
        assert attempts[0].model == "model-a"
        assert attempts[0].status == "failed"


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestFallbackRepr:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_repr_includes_fallbacks(self, mock_create):
        mock_create.return_value = AsyncMock()
        llm = LLM(
            "claude-sonnet-4-20250514",
            api_key="test",
            fallback=["claude-haiku-4-5-20251001", "gpt-4o"],
        )
        r = repr(llm)
        assert "claude-sonnet-4-20250514" in r
        assert "claude-haiku-4-5-20251001" in r
        assert "gpt-4o" in r
        assert "fallback=" in r

    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_repr_no_fallbacks(self, mock_create):
        mock_create.return_value = AsyncMock()
        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        assert "fallback" not in repr(llm)


# ---------------------------------------------------------------------------
# Response backward compat
# ---------------------------------------------------------------------------


class TestResponseAttemptField:
    def test_response_default_empty_attempts(self):
        r = Response(text="ok")
        assert r.attempts == ()

    def test_response_with_attempts(self):
        a = Attempt(model="m", status="ok")
        r = Response(text="ok", attempts=(a,))
        assert len(r.attempts) == 1
        assert r.attempts[0].model == "m"

    def test_attempt_defaults(self):
        a = Attempt(model="m", status="ok")
        assert a.error is None
        assert a.error_type is None
        assert a.status_code is None
        assert a.usage is None
        assert a.duration == 0.0
        assert a.timestamp == 0.0
        assert a.retry_number == 0

    def test_attempt_frozen(self):
        a = Attempt(model="m", status="ok")
        with pytest.raises(AttributeError):
            a.model = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Retry + fallback interaction
# ---------------------------------------------------------------------------


class TestRetryWithFallback:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_retry_exhausted_then_fallback_succeeds(self, mock_sleep, mock_create):
        """Primary retries N times, all fail, then fallback succeeds."""
        from ai_arch_toolkit.core._retry import RetryConfig

        p1 = AsyncMock()
        p1.complete.side_effect = APIError(500, "down")
        p2 = AsyncMock()
        p2.complete.return_value = _make_response("fallback ok")
        mock_create.side_effect = [p1, p2]

        retry = RetryConfig(max_retries=2, base_delay=0.01)
        llm = LLM("model-a", api_key="test", retry=retry, fallback="model-b")
        result = await llm.complete("Hi")
        assert result.text == "fallback ok"

        # Primary: 1 initial + 2 retries = 3 failed attempts
        # Fallback: 1 ok attempt
        primary_attempts = [a for a in result.attempts if a.model == "model-a"]
        fallback_attempts = [a for a in result.attempts if a.model == "model-b"]
        assert len(primary_attempts) == 3
        assert all(a.status == "failed" for a in primary_attempts)
        assert primary_attempts[0].retry_number == 0
        assert primary_attempts[1].retry_number == 1
        assert primary_attempts[2].retry_number == 2
        assert len(fallback_attempts) == 1
        assert fallback_attempts[0].status == "ok"


# ---------------------------------------------------------------------------
# Fallback in stream_events()
# ---------------------------------------------------------------------------


class TestStreamEventsWithFallback:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_primary_stream_events_fails_fallback_used(self, mock_create):
        from ai_arch_toolkit.core._response import StreamEvent

        async def _fake_events():
            yield StreamEvent(kind="text", text="hello")

        p1 = MagicMock()
        p1.stream_events.side_effect = APIError(500, "down")
        p2 = MagicMock()
        state = _make_stream_state(model="model-b")
        p2.stream_events.return_value = (_fake_events(), state)
        mock_create.side_effect = [p1, p2]

        llm = LLM("model-a", api_key="test", fallback="model-b")
        stream = llm.stream_events("Hi")
        events = []
        async for event in stream:
            events.append(event)
        assert len(events) == 1
        assert events[0].kind == "text"
        assert stream.response is not None
        # Should have parent failed + fallback ok attempts
        assert len(stream.response.attempts) >= 2
        assert stream.response.attempts[0].model == "model-a"
        assert stream.response.attempts[0].status == "failed"


# ---------------------------------------------------------------------------
# Fallback in stream_sync()
# ---------------------------------------------------------------------------


class TestStreamSyncWithFallback:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    def test_stream_sync_fallback(self, mock_create):
        async def _fake_gen():
            for chunk in ["sync", "fb"]:
                yield chunk

        p1 = MagicMock()
        p1.stream.side_effect = APIError(500, "down")
        p2 = MagicMock()
        state = _make_stream_state(model="model-b")
        p2.stream.return_value = (_fake_gen(), state)
        mock_create.side_effect = [p1, p2]

        llm = LLM("model-a", api_key="test", fallback="model-b")
        stream = llm.stream_sync("Hi")
        chunks = list(stream)
        assert chunks == ["sync", "fb"]
        assert stream.response is not None
        assert len(stream.response.attempts) >= 2
        assert stream.response.attempts[0].status == "failed"
