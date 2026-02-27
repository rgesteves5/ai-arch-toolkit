"""Tests for model fallback in LLM."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ai_arch_toolkit.core._exceptions import APIError
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._response import Response


class TestFallback:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_no_fallback_on_success(self, mock_create):
        primary = AsyncMock()
        primary.complete.return_value = Response(text="from primary")
        mock_create.return_value = primary

        llm = LLM("gpt-4o", api_key="test")
        result = await llm.complete("Hi")
        assert result.text == "from primary"

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_fallback_on_api_error(self, mock_create):
        primary = AsyncMock()
        primary.complete.side_effect = APIError(500, "internal error")
        primary.close = AsyncMock()

        fallback = AsyncMock()
        fallback.complete.return_value = Response(text="from fallback")
        fallback.close = AsyncMock()

        mock_create.side_effect = [primary, fallback]

        llm = LLM("gpt-4o", api_key="test", fallback="claude-sonnet-4-20250514")
        result = await llm.complete("Hi")
        assert result.text == "from fallback"

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_no_fallback_provider_raises(self, mock_create):
        primary = AsyncMock()
        primary.complete.side_effect = APIError(500, "internal error")
        mock_create.return_value = primary

        llm = LLM("gpt-4o", api_key="test")
        with pytest.raises(APIError):
            await llm.complete("Hi")

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_non_api_error_not_caught(self, mock_create):
        primary = AsyncMock()
        primary.complete.side_effect = ValueError("bad")
        fallback = AsyncMock()
        mock_create.side_effect = [primary, fallback]

        llm = LLM("gpt-4o", api_key="test", fallback="claude-sonnet-4-20250514")
        with pytest.raises(ValueError):
            await llm.complete("Hi")

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_close_closes_both(self, mock_create):
        primary = AsyncMock()
        fallback = AsyncMock()
        mock_create.side_effect = [primary, fallback]

        llm = LLM("gpt-4o", api_key="test", fallback="claude-sonnet-4-20250514")
        await llm.close()
        primary.close.assert_awaited_once()
        fallback.close.assert_awaited_once()
