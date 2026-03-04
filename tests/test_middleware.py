"""Tests for _middleware.py — request/response hooks."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import AsyncMock, patch

import pytest

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._middleware import (
    Request,
    _run_aafter,
    _run_abefore,
    _run_after,
    _run_before,
)
from ai_arch_toolkit.core._response import Response

# ---------------------------------------------------------------------------
# Request dataclass
# ---------------------------------------------------------------------------


class TestRequest:
    def test_fields(self):
        req = Request(
            messages=[{"role": "user", "content": "Hi"}],
            system="Be helpful",
            tools=None,
            model="gpt-4o",
        )
        assert req.model == "gpt-4o"
        assert req.system == "Be helpful"
        assert req.kwargs == {}

    def test_frozen(self):
        req = Request(messages=[], system=None, tools=None, model="gpt-4o")
        with pytest.raises(AttributeError):
            req.model = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Sync middleware runners
# ---------------------------------------------------------------------------


class _AddSystemMW:
    """Test middleware that adds a system instruction."""

    def before(self, request: Request) -> Request:
        return replace(request, system="injected system")

    def after(self, request: Request, response: Response) -> Response:
        return response


class _UpperTextMW:
    """Test middleware that uppercases response text."""

    def before(self, request: Request) -> Request:
        return request

    def after(self, request: Request, response: Response) -> Response:
        return Response(text=response.text.upper())


class TestRunBefore:
    def test_single_middleware(self):
        req = Request(messages=[], system=None, tools=None, model="x")
        result = _run_before([_AddSystemMW()], req)
        assert result.system == "injected system"

    def test_chain_order(self):
        class First:
            def before(self, r: Request) -> Request:
                return replace(r, system="first")

        class Second:
            def before(self, r: Request) -> Request:
                return replace(r, system=f"{r.system}+second")

        req = Request(messages=[], system=None, tools=None, model="x")
        result = _run_before([First(), Second()], req)
        assert result.system == "first+second"


class TestRunAfter:
    def test_reverse_order(self):
        req = Request(messages=[], system=None, tools=None, model="x")
        resp = Response(text="hello")
        result = _run_after([_UpperTextMW()], req, resp)
        assert result.text == "HELLO"


# ---------------------------------------------------------------------------
# Async middleware runners
# ---------------------------------------------------------------------------


class _AsyncMW:
    """Middleware with async hooks."""

    async def abefore(self, request: Request) -> Request:
        return replace(request, system="async_injected")

    async def aafter(self, request: Request, response: Response) -> Response:
        return Response(text=response.text + "_async")

    def before(self, request: Request) -> Request:
        return request

    def after(self, request: Request, response: Response) -> Response:
        return response


class TestRunAbefore:
    async def test_async_hook(self):
        req = Request(messages=[], system=None, tools=None, model="x")
        result = await _run_abefore([_AsyncMW()], req)
        assert result.system == "async_injected"

    async def test_falls_back_to_sync(self):
        req = Request(messages=[], system=None, tools=None, model="x")
        result = await _run_abefore([_AddSystemMW()], req)
        assert result.system == "injected system"


class TestRunAafter:
    async def test_async_hook(self):
        req = Request(messages=[], system=None, tools=None, model="x")
        resp = Response(text="hi")
        result = await _run_aafter([_AsyncMW()], req, resp)
        assert result.text == "hi_async"


# ---------------------------------------------------------------------------
# LLM integration
# ---------------------------------------------------------------------------


class TestLLMMiddlewareIntegration:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_before_hook_modifies_request(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = Response(text="ok")
        mock_create.return_value = mock_provider

        llm = LLM("gpt-4o", api_key="test", middleware=[_AddSystemMW()])
        await llm.complete("Hi")

        call_kwargs = mock_provider.complete.call_args
        assert call_kwargs[1]["system"] == "injected system"

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_after_hook_modifies_response(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = Response(text="hello")
        mock_create.return_value = mock_provider

        llm = LLM("gpt-4o", api_key="test", middleware=[_UpperTextMW()])
        result = await llm.complete("Hi")
        assert result.text == "HELLO"
