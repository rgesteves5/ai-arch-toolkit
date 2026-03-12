"""Tests for ModerationMiddleware."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._moderation import ModerationError, ModerationResult
from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.toolkit.moderation._middleware import ModerationMiddleware


def _make_request(user_text: str | list) -> Request:
    content = user_text
    return Request(
        messages=[{"role": "user", "content": content}],
        system=None,
        tools=None,
        model="test-model",
    )


def _make_response(text: str) -> Response:
    return Response(
        text=text,
        model="test-model",
        stop_reason="end_turn",
        usage=Usage(input_tokens=0, output_tokens=0),
    )


class FakeModerator:
    """Configurable mock moderator."""

    def __init__(self, result: ModerationResult) -> None:
        self._result = result
        self.called_with: str | None = None

    async def moderate(self, text: str) -> ModerationResult:
        self.called_with = text
        return self._result


UNFLAGGED = ModerationResult(flagged=False, categories=[])
FLAGGED = ModerationResult(
    flagged=True,
    categories=["violence"],
    explanation="violent content",
)


class TestModerationMiddleware:
    def test_both_none_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            ModerationMiddleware()

    # --- Input moderation ---

    async def test_input_raise_on_flagged(self) -> None:
        mod = FakeModerator(FLAGGED)
        mw = ModerationMiddleware(input=mod, on_flagged="raise")
        with pytest.raises(ModerationError):
            await mw.abefore(_make_request("bad text"))

    async def test_input_warn_on_flagged(self) -> None:
        mod = FakeModerator(FLAGGED)
        mw = ModerationMiddleware(input=mod, on_flagged="warn")
        result = await mw.abefore(_make_request("bad text"))
        assert result is not None  # passes through

    async def test_input_unflagged_passes(self) -> None:
        mod = FakeModerator(UNFLAGGED)
        mw = ModerationMiddleware(input=mod, on_flagged="raise")
        req = _make_request("hello")
        result = await mw.abefore(req)
        assert result is req

    async def test_no_input_moderator(self) -> None:
        mod = FakeModerator(FLAGGED)
        mw = ModerationMiddleware(output=mod)
        req = _make_request("text")
        result = await mw.abefore(req)
        assert result is req

    # --- Output moderation ---

    async def test_output_raise_on_flagged(self) -> None:
        mod = FakeModerator(FLAGGED)
        mw = ModerationMiddleware(output=mod, on_flagged="raise")
        req = _make_request("q")
        with pytest.raises(ModerationError):
            await mw.aafter(req, _make_response("bad response"))

    async def test_output_warn_on_flagged(self) -> None:
        mod = FakeModerator(FLAGGED)
        mw = ModerationMiddleware(output=mod, on_flagged="warn")
        req = _make_request("q")
        resp = _make_response("bad response")
        result = await mw.aafter(req, resp)
        assert result is resp

    async def test_no_output_moderator(self) -> None:
        mod = FakeModerator(FLAGGED)
        mw = ModerationMiddleware(input=mod)
        req = _make_request("q")
        resp = _make_response("text")
        result = await mw.aafter(req, resp)
        assert result is resp

    async def test_empty_output_skips(self) -> None:
        mod = FakeModerator(FLAGGED)
        mw = ModerationMiddleware(output=mod, on_flagged="raise")
        req = _make_request("q")
        resp = _make_response("")
        result = await mw.aafter(req, resp)
        assert result is resp
        assert mod.called_with is None

    # --- Sync stubs ---

    def test_sync_before_passthrough(self) -> None:
        mod = FakeModerator(UNFLAGGED)
        mw = ModerationMiddleware(input=mod)
        req = _make_request("text")
        assert mw.before(req) is req

    def test_sync_after_passthrough(self) -> None:
        mod = FakeModerator(UNFLAGGED)
        mw = ModerationMiddleware(input=mod)
        req = _make_request("text")
        resp = _make_response("reply")
        assert mw.after(req, resp) is resp

    # --- Text extraction ---

    async def test_text_extraction_string(self) -> None:
        mod = FakeModerator(UNFLAGGED)
        mw = ModerationMiddleware(input=mod)
        await mw.abefore(_make_request("hello world"))
        assert mod.called_with == "hello world"

    async def test_text_extraction_multimodal(self) -> None:
        mod = FakeModerator(UNFLAGGED)
        mw = ModerationMiddleware(input=mod)
        content = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
        await mw.abefore(_make_request(content))
        assert mod.called_with == "hello world"

    async def test_no_user_message(self) -> None:
        mod = FakeModerator(UNFLAGGED)
        mw = ModerationMiddleware(input=mod)
        req = Request(
            messages=[{"role": "assistant", "content": "hi"}],
            system=None,
            tools=None,
            model="test",
        )
        result = await mw.abefore(req)
        assert result is req
        assert mod.called_with is None
