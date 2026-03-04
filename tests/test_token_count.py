"""Tests for token counting."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._providers._base import BaseProvider


class TestBaseProviderCountTokens:
    async def test_not_implemented(self):
        class DummyProvider(BaseProvider):
            async def complete(self, messages, **kwargs):
                pass

            def stream(self, messages, **kwargs):
                pass

        with pytest.raises(NotImplementedError, match="does not support token counting"):
            await DummyProvider().count_tokens([{"role": "user", "content": "Hi"}])


class TestLLMCountTokens:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_count_tokens(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.count_tokens.return_value = 42
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        result = await llm.count_tokens("Hello world")
        assert result == 42

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_count_tokens_with_system(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.count_tokens.return_value = 100
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        result = await llm.count_tokens("Hi", system="Be helpful")
        assert result == 100
        call_kwargs = mock_provider.count_tokens.call_args[1]
        assert call_kwargs["system"] == "Be helpful"

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_not_implemented_raises(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.count_tokens.side_effect = NotImplementedError("unsupported")
        mock_create.return_value = mock_provider

        llm = LLM("gpt-4o", api_key="test")
        with pytest.raises(NotImplementedError):
            await llm.count_tokens("Hi")
