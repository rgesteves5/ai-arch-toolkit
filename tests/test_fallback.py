"""Tests for model fallback in LLM."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ai_arch_toolkit.core._exceptions import APIError
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._response import Response
from tests.fake_provider import FakeProvider

FALLBACK_MODEL = "claude-sonnet-4-20250514"


class _ClosingProvider(FakeProvider):
    """A fake provider that counts how many times it is closed."""

    def __init__(self, model: str) -> None:
        super().__init__(model=model)
        self.closes = 0

    async def close(self) -> None:
        self.closes += 1
        await super().close()


class TestFallback:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_no_fallback_on_success(self, mock_create):
        mock_create.return_value = FakeProvider(Response(text="from primary"), model="gpt-4o")

        llm = LLM("gpt-4o", api_key="test")
        result = await llm.complete("Hi")
        assert result.text == "from primary"

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_fallback_on_api_error(self, mock_create):
        primary = FakeProvider(APIError(500, "internal error"), model="gpt-4o")
        fallback = FakeProvider(Response(text="from fallback"), model=FALLBACK_MODEL)
        mock_create.side_effect = [primary, fallback]

        llm = LLM("gpt-4o", api_key="test", fallback=FALLBACK_MODEL)
        result = await llm.complete("Hi")
        assert result.text == "from fallback"

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_no_fallback_provider_raises(self, mock_create):
        mock_create.return_value = FakeProvider(APIError(500, "internal error"), model="gpt-4o")

        llm = LLM("gpt-4o", api_key="test")
        with pytest.raises(APIError):
            await llm.complete("Hi")

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_non_api_error_not_caught(self, mock_create):
        primary = FakeProvider(ValueError("bad"), model="gpt-4o")
        fallback = FakeProvider(model=FALLBACK_MODEL)
        mock_create.side_effect = [primary, fallback]

        llm = LLM("gpt-4o", api_key="test", fallback=FALLBACK_MODEL)
        with pytest.raises(ValueError):
            await llm.complete("Hi")

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_close_closes_both(self, mock_create):
        primary = _ClosingProvider("gpt-4o")
        fallback = _ClosingProvider(FALLBACK_MODEL)
        mock_create.side_effect = [primary, fallback]

        llm = LLM("gpt-4o", api_key="test", fallback=FALLBACK_MODEL)
        await llm.close()
        assert primary.closes == 1
        assert fallback.closes == 1
