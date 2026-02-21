"""Tests for response caching middleware."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_arch_toolkit.llm import AsyncClient, Client
from ai_arch_toolkit.llm._cache import InMemoryCacheBackend, ResponseCache
from ai_arch_toolkit.llm._types import Response, Usage


@patch("ai_arch_toolkit.llm._client.create_provider")
def test_response_cache_short_circuits_sync_chat(mock_create: MagicMock) -> None:
    provider = MagicMock()
    provider.complete.side_effect = [
        Response(text="first", usage=Usage(total_tokens=1)),
        Response(text="second", usage=Usage(total_tokens=1)),
    ]
    mock_create.return_value = provider
    cache = ResponseCache()
    client = Client("openai", model="gpt-4o", api_key="sk-test", middleware=[cache])

    first = client.chat("hello")
    second = client.chat("hello")

    assert first.text == "first"
    assert second.text == "first"
    assert provider.complete.call_count == 1


@patch("ai_arch_toolkit.llm._async_client.create_provider")
@pytest.mark.asyncio
async def test_response_cache_short_circuits_async_chat(mock_create: MagicMock) -> None:
    provider = MagicMock()
    provider.acomplete = AsyncMock(
        side_effect=[
            Response(text="first", usage=Usage(total_tokens=1)),
            Response(text="second", usage=Usage(total_tokens=1)),
        ]
    )
    mock_create.return_value = provider
    cache = ResponseCache()
    client = AsyncClient("openai", model="gpt-4o", api_key="sk-test", middleware=[cache])

    first = await client.chat("hello")
    second = await client.chat("hello")

    assert first.text == "first"
    assert second.text == "first"
    assert provider.acomplete.call_count == 1


@patch("ai_arch_toolkit.llm._client.create_provider")
def test_response_cache_respects_ttl(mock_create: MagicMock) -> None:
    provider = MagicMock()
    provider.complete.side_effect = [
        Response(text="first", usage=Usage(total_tokens=1)),
        Response(text="second", usage=Usage(total_tokens=1)),
    ]
    mock_create.return_value = provider

    now = [100.0]

    def _clock() -> float:
        return now[0]

    backend = InMemoryCacheBackend(clock=_clock)
    cache = ResponseCache(backend=backend, ttl_seconds=1.0)
    client = Client("openai", model="gpt-4o", api_key="sk-test", middleware=[cache])

    first = client.chat("hello")
    now[0] = 100.5
    hit = client.chat("hello")
    now[0] = 101.5
    expired = client.chat("hello")

    assert first.text == "first"
    assert hit.text == "first"
    assert expired.text == "second"
    assert provider.complete.call_count == 2


class _RecordingBackend:
    def __init__(self) -> None:
        self._values: dict[str, Response] = {}
        self.keys_set: list[str] = []

    def get(self, key: str) -> Response | None:
        return self._values.get(key)

    def set(self, key: str, value: Response, ttl_seconds: float | None) -> None:
        self.keys_set.append(key)
        self._values[key] = value


@patch("ai_arch_toolkit.llm._client.create_provider")
def test_response_cache_supports_custom_backend_and_key_fn(mock_create: MagicMock) -> None:
    provider = MagicMock()
    provider.complete.return_value = Response(text="ok", usage=Usage(total_tokens=1))
    mock_create.return_value = provider

    backend = _RecordingBackend()
    cache = ResponseCache(backend=backend, key_fn=lambda _request: "fixed-key")
    client = Client("openai", model="gpt-4o", api_key="sk-test", middleware=[cache])

    _ = client.chat("hello")
    _ = client.chat("hello")

    assert backend.keys_set == ["fixed-key"]
    assert provider.complete.call_count == 1

