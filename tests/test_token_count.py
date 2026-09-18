"""Tests for token counting."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from ai_arch_toolkit.core._llm import LLM
from tests.fake_provider import FakeProvider


class _CountingProvider(FakeProvider):
    """A fake provider that counts every request as ``count`` tokens and records what it saw."""

    def __init__(self, count: int, *, model: str) -> None:
        super().__init__(model=model)
        self.count = count
        self.counted: list[dict[str, Any]] = []

    async def count_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        self.counted.append({"messages": messages, "system": system, "tools": tools})
        return self.count


class TestBaseProviderCountTokens:
    async def test_not_implemented(self):
        # FakeProvider keeps BaseProvider's count_tokens.
        with pytest.raises(NotImplementedError, match="does not support token counting"):
            await FakeProvider().count_tokens([{"role": "user", "content": "Hi"}])


class TestLLMCountTokens:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_count_tokens(self, mock_create):
        mock_create.return_value = _CountingProvider(42, model="claude-sonnet-4-20250514")

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        result = await llm.count_tokens("Hello world")
        assert result == 42

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_count_tokens_with_system(self, mock_create):
        provider = _CountingProvider(100, model="claude-sonnet-4-20250514")
        mock_create.return_value = provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        result = await llm.count_tokens("Hi", system="Be helpful")
        assert result == 100
        assert provider.counted[-1]["system"] == "Be helpful"

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_not_implemented_raises(self, mock_create):
        mock_create.return_value = FakeProvider(model="gpt-4o")  # no token counting

        llm = LLM("gpt-4o", api_key="test")
        with pytest.raises(NotImplementedError):
            await llm.count_tokens("Hi")
