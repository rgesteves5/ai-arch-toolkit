"""Tests for _async_http.py helpers."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from ai_arch_toolkit.llm._async_http import async_post_json
from ai_arch_toolkit.llm._exceptions import APIError, RateLimitError
from ai_arch_toolkit.llm._http import RetryConfig


def _mock_httpx_response(
    status_code: int = 200,
    json_data: dict | None = None,
    text: str = "",
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """Create a mock httpx.Response."""
    resp = httpx.Response(
        status_code=status_code,
        headers=headers or {},
        text=text if not json_data else json.dumps(json_data),
    )
    return resp


@pytest.fixture
def mock_async_client():
    """Patch httpx.AsyncClient for non-streaming tests."""
    with patch("ai_arch_toolkit.llm._async_http.httpx.AsyncClient") as mock_cls:
        mock_instance = AsyncMock()
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)
        yield mock_instance


async def test_async_post_json_success(mock_async_client: AsyncMock) -> None:
    mock_async_client.post.return_value = _mock_httpx_response(json_data={"result": "ok"})
    result = await async_post_json("https://example.com", {"Auth": "key"}, {"q": "test"})
    assert result == {"result": "ok"}


async def test_async_post_json_api_error(mock_async_client: AsyncMock) -> None:
    mock_async_client.post.return_value = _mock_httpx_response(
        status_code=400, json_data={"error": "bad"}
    )
    with pytest.raises(APIError) as exc_info:
        await async_post_json("https://example.com", {}, {})
    assert exc_info.value.status_code == 400


async def test_async_post_json_rate_limit(mock_async_client: AsyncMock) -> None:
    mock_async_client.post.return_value = _mock_httpx_response(
        status_code=429,
        json_data={"error": "rate limited"},
        headers={"Retry-After": "3"},
    )
    with pytest.raises(RateLimitError) as exc_info:
        await async_post_json("https://example.com", {}, {})
    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after == 3.0


@patch("ai_arch_toolkit.llm._async_http.asyncio.sleep", new_callable=AsyncMock)
async def test_async_post_json_retries(
    mock_sleep: AsyncMock, mock_async_client: AsyncMock
) -> None:
    mock_async_client.post.side_effect = [
        _mock_httpx_response(status_code=429, json_data={"error": "rate limited"}),
        _mock_httpx_response(json_data={"result": "ok"}),
    ]
    config = RetryConfig(max_retries=2, backoff_factor=1.0)
    result = await async_post_json("https://example.com", {}, {}, retry=config)
    assert result == {"result": "ok"}
    assert mock_async_client.post.call_count == 2
    mock_sleep.assert_called_once()
