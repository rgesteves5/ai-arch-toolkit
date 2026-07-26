"""Tests for LoopAwareClientCache — sync wrappers survive event-loop turnover.

``_run_sync`` drives each call through its own ``asyncio.run()`` loop; a cached
async SDK client binds its connection pool to the first loop that serves a
request and must be rebuilt once that loop closes.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._providers._anthropic import AnthropicProvider
from ai_arch_toolkit.core._providers._base import LoopAwareClientCache


def _sdk_message() -> SimpleNamespace:
    usage = SimpleNamespace(
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    content = [SimpleNamespace(type="text", text="Hello!", citations=None)]
    return SimpleNamespace(
        content=content, model="claude-sonnet-4-6", stop_reason="end_turn", usage=usage
    )


class _DummyProvider(LoopAwareClientCache):
    def __init__(self) -> None:
        self.built = 0

        def factory() -> object:
            self.built += 1
            return object()

        self._install_client(factory)


class TestLoopAwareClientCache:
    def test_rebuilds_after_loop_closes(self) -> None:
        provider = _DummyProvider()

        async def use() -> object:
            return provider._client

        first = asyncio.run(use())
        second = asyncio.run(use())

        assert provider.built == 2
        assert first is not second

    def test_stable_within_one_loop(self) -> None:
        provider = _DummyProvider()

        async def use_twice() -> tuple[object, object]:
            return provider._client, provider._client

        a, b = asyncio.run(use_twice())

        assert a is b
        assert provider.built == 1

    def test_injected_client_is_never_replaced(self) -> None:
        provider = _DummyProvider()
        sentinel = object()
        provider._client = sentinel  # test-style direct injection

        async def use() -> object:
            return provider._client

        assert asyncio.run(use()) is sentinel
        assert asyncio.run(use()) is sentinel
        assert provider.built == 1  # only the install-time build

    def test_access_without_running_loop(self) -> None:
        provider = _DummyProvider()

        assert provider._client is provider._client
        assert provider.built == 1


class TestProvidersAreLoopAware:
    def test_all_adapters_install_a_factory(self) -> None:
        from ai_arch_toolkit.core._providers._gemini import GeminiProvider
        from ai_arch_toolkit.core._providers._openai import OpenAIProvider
        from ai_arch_toolkit.core._providers._xai import XAIProvider

        # Patch the SDK modules: real clients open pools (and the xAI gRPC
        # client requires a running event loop) at construction time.
        with (
            patch("ai_arch_toolkit.core._providers._anthropic.anthropic"),
            patch("ai_arch_toolkit.core._providers._openai.openai"),
            patch("ai_arch_toolkit.core._providers._gemini.genai"),
            patch("ai_arch_toolkit.core._providers._xai.xai_sdk"),
        ):
            for cls in (AnthropicProvider, OpenAIProvider, GeminiProvider, XAIProvider):
                provider = cls("some-model", "test-key")
                assert isinstance(provider, LoopAwareClientCache)
                assert provider._client_factory is not None


class TestSecondSyncCallRegression:
    @patch("ai_arch_toolkit.core._providers._anthropic.anthropic")
    def test_complete_sync_twice_rebuilds_the_sdk_client(self, mock_sdk) -> None:
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_sdk_message())
        mock_sdk.AsyncAnthropic = MagicMock(return_value=mock_client)

        llm = LLM("claude-sonnet-4-6", api_key="test")

        first = llm.complete_sync([{"role": "user", "content": "Hi"}])
        second = llm.complete_sync([{"role": "user", "content": "Hi"}])

        assert first.text == "Hello!"
        assert second.text == "Hello!"
        # One build at construction, one rebuild after the first loop closed.
        assert mock_sdk.AsyncAnthropic.call_count == 2
