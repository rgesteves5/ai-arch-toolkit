"""Tests for LoopAwareClientCache — sync wrappers survive event-loop turnover.

``_run_sync`` drives each call through its own ``asyncio.run()`` loop; a cached
async SDK client binds its connection pool to the first loop that serves a
request and must be rebuilt once that loop closes.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._providers._anthropic import AnthropicProvider
from ai_arch_toolkit.core._providers._base import LoopAwareClientCache
from ai_arch_toolkit.core._providers._xai import XAIProvider
from tests.integration import fakegrpc
from tests.provider_calls import complete


def _sdk_message() -> SimpleNamespace:
    usage = SimpleNamespace(
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    content = [SimpleNamespace(type="text", text="Hello!", citations=None)]
    return SimpleNamespace(
        id="msg_test",
        content=content,
        model="claude-sonnet-4-6",
        stop_reason="end_turn",
        usage=usage,
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
        assert provider.built == 0  # the factory never ran: the injected client came first

    def test_access_without_running_loop(self) -> None:
        provider = _DummyProvider()

        assert provider._client is provider._client
        assert provider.built == 1


class TestBuiltWithoutALoop:
    """Building a provider does no work bound to an event loop (G-17): an app may build its
    ``LLM`` in a worker thread, or before its loop runs. The SDK client is built on first use."""

    def test_the_client_is_built_on_first_use(self) -> None:
        provider = _DummyProvider()
        assert provider.built == 0

        assert provider._client is provider._client
        assert provider.built == 1

    @pytest.mark.parametrize(
        "model",
        ["grok-4.7", "claude-sonnet-4-6", "gpt-5.5", "gemini-3.8-flash", "muse-spark-1.3"],
    )
    async def test_an_llm_builds_in_a_thread_without_a_loop(self, model: str) -> None:
        llm = await asyncio.to_thread(LLM, model, api_key="test-key")

        async with llm:
            assert llm._provider._client is not None  # built here, inside the loop

    def test_the_xai_adapter_builds_in_a_plain_thread(self) -> None:
        built: list[object] = []
        failed: list[BaseException] = []

        def build() -> None:
            try:
                built.append(XAIProvider("grok-4.7", "test-key"))
            except BaseException as exc:  # reported below
                failed.append(exc)

        thread = threading.Thread(target=build)
        thread.start()
        thread.join()

        assert failed == []
        assert len(built) == 1

    async def test_an_xai_adapter_built_in_a_thread_answers_in_the_loop(self) -> None:
        adapter = await asyncio.to_thread(XAIProvider, "grok-4.7", "local-test")
        async with fakegrpc.serving() as (script, port):
            fakegrpc.point(adapter, port)
            response = await complete(adapter, [{"role": "user", "content": "Hi"}])
            await adapter.close()

        assert response.text == "ok"
        assert len(script.requests) == 1

    def test_close_drops_a_client_whose_loop_has_closed(self) -> None:
        provider = _DummyProvider()
        provider._close_client = AsyncMock()  # type: ignore[method-assign]

        async def use() -> object:
            return provider._client

        asyncio.run(use())  # a sync call's loop, closed after it
        asyncio.run(provider.close())

        provider._close_client.assert_not_awaited()
        asyncio.run(use())
        assert provider.built == 2  # a later call builds a new client

    async def test_closing_an_unused_adapter_builds_no_client(self) -> None:
        provider = _DummyProvider()
        provider._close_client = AsyncMock()  # type: ignore[method-assign]

        await provider.close()

        assert provider.built == 0
        provider._close_client.assert_not_awaited()


class TestProvidersAreLoopAware:
    def test_all_adapters_install_a_factory(self) -> None:
        from ai_arch_toolkit.core._providers._gemini import GeminiProvider
        from ai_arch_toolkit.core._providers._meta import MetaProvider
        from ai_arch_toolkit.core._providers._openai import OpenAIProvider
        from ai_arch_toolkit.core._providers._openai_compatible import OpenAICompatibleProvider
        from ai_arch_toolkit.core._providers._xai import XAIProvider

        # Patch the SDK modules: real clients open pools (and the xAI gRPC
        # client requires a running event loop) at construction time.
        with (
            patch("ai_arch_toolkit.core._providers._anthropic.anthropic"),
            patch("ai_arch_toolkit.core._providers._openai.openai"),
            patch("ai_arch_toolkit.core._providers._openai_compatible.openai"),
            patch("ai_arch_toolkit.core._providers._meta.openai"),
            patch("ai_arch_toolkit.core._providers._gemini.genai"),
            patch("ai_arch_toolkit.core._providers._xai.xai_sdk"),
        ):
            providers = [
                *(
                    cls("some-model", "test-key")
                    for cls in (
                        AnthropicProvider,
                        OpenAIProvider,
                        MetaProvider,
                        GeminiProvider,
                        XAIProvider,
                    )
                ),
                OpenAICompatibleProvider("some-model", "k", base_url="http://localhost:1/v1"),
            ]
            for provider in providers:
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
