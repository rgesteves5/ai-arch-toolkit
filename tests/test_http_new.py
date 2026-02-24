"""Tests for _http.py — unified HTTP helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ai_arch_toolkit.core._exceptions import APIError, RateLimitError
from ai_arch_toolkit.core._http import (
    RetryConfig,
    _should_retry,
    _wait_time,
    async_post_json,
    async_stream_sse,
    post_json,
    stream_sse,
)

# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestShouldRetry:
    def test_retryable_code_within_limit(self):
        cfg = RetryConfig(max_retries=3)
        assert _should_retry(429, 0, cfg) is True
        assert _should_retry(500, 2, cfg) is True

    def test_non_retryable_code(self):
        cfg = RetryConfig(max_retries=3)
        assert _should_retry(400, 0, cfg) is False

    def test_exceeded_max_retries(self):
        cfg = RetryConfig(max_retries=2)
        assert _should_retry(429, 2, cfg) is False


class TestWaitTime:
    def test_exponential_backoff(self):
        cfg = RetryConfig(backoff_factor=2.0)
        assert _wait_time(0, cfg) == 1.0
        assert _wait_time(1, cfg) == 2.0
        assert _wait_time(2, cfg) == 4.0

    def test_retry_after_overrides(self):
        cfg = RetryConfig(backoff_factor=2.0)
        assert _wait_time(0, cfg, retry_after=5.0) == 5.0

    def test_retry_after_zero_uses_backoff(self):
        cfg = RetryConfig(backoff_factor=2.0)
        assert _wait_time(1, cfg, retry_after=0.0) == 2.0


# ---------------------------------------------------------------------------
# Sync tests
# ---------------------------------------------------------------------------


class MockResponse:
    """Minimal mock for requests.Response."""

    def __init__(
        self,
        status_code: int = 200,
        json_data: dict | None = None,
        text: str = "",
        headers: dict | None = None,
        lines: list[str] | None = None,
    ):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text
        self.headers = headers or {}
        self.ok = 200 <= status_code < 400
        self._lines = lines or []

    def json(self):
        return self._json

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class TestPostJson:
    def test_success(self):
        mock_session = MagicMock()
        mock_session.post.return_value = MockResponse(json_data={"result": "ok"})
        result = post_json("http://test", {}, {}, session=mock_session)
        assert result == {"result": "ok"}

    def test_error_raises_api_error(self):
        mock_session = MagicMock()
        mock_session.post.return_value = MockResponse(status_code=400, json_data={"error": "bad"})
        with pytest.raises(APIError) as exc_info:
            post_json("http://test", {}, {}, session=mock_session)
        assert exc_info.value.status_code == 400

    def test_rate_limit_raises(self):
        mock_session = MagicMock()
        mock_session.post.return_value = MockResponse(
            status_code=429,
            json_data={"error": "rate limited"},
            headers={"Retry-After": "2"},
        )
        with pytest.raises(RateLimitError) as exc_info:
            post_json("http://test", {}, {}, session=mock_session)
        assert exc_info.value.retry_after == 2.0

    def test_retry_on_500(self):
        mock_session = MagicMock()
        mock_session.post.side_effect = [
            MockResponse(status_code=500, json_data={"error": "server"}),
            MockResponse(json_data={"result": "ok"}),
        ]
        with patch("ai_arch_toolkit.core._http.time.sleep"):
            result = post_json(
                "http://test", {}, {}, session=mock_session, retry=RetryConfig(max_retries=1)
            )
        assert result == {"result": "ok"}


class TestStreamSse:
    def test_yields_data_lines(self):
        mock_session = MagicMock()
        mock_session.post.return_value = MockResponse(
            lines=["data: line1", "data: line2", ": comment", ""]
        )
        result = list(stream_sse("http://test", {}, {}, session=mock_session))
        assert result == ["line1", "line2"]

    def test_error_raises(self):
        mock_session = MagicMock()
        mock_session.post.return_value = MockResponse(status_code=500, json_data={"error": "bad"})
        with pytest.raises(APIError):
            list(stream_sse("http://test", {}, {}, session=mock_session))


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------


class TestAsyncPostJson:
    async def test_success(self):
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {"result": "ok"}

        mock_client = MagicMock()
        mock_client.post = _async_return(mock_response)

        result = await async_post_json("http://test", {}, {}, client=mock_client)
        assert result == {"result": "ok"}

    async def test_error_raises(self):
        mock_response = MagicMock()
        mock_response.is_success = False
        mock_response.status_code = 400
        mock_response.text = "bad request"
        mock_response.json.side_effect = Exception("no json")
        mock_response.headers = {}

        mock_client = MagicMock()
        mock_client.post = _async_return(mock_response)

        with pytest.raises(APIError):
            await async_post_json("http://test", {}, {}, client=mock_client)


class TestAsyncStreamSse:
    async def test_yields_data(self):
        lines = ["data: chunk1", "data: chunk2", ": comment", ""]

        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.aiter_lines = lambda: _async_iter(lines)

        mock_client = MagicMock()
        mock_client.stream.return_value = _AsyncContextManager(mock_response)

        chunks = []
        async for chunk in async_stream_sse("http://test", {}, {}, client=mock_client):
            chunks.append(chunk)
        assert chunks == ["chunk1", "chunk2"]


# ---------------------------------------------------------------------------
# Async test helpers
# ---------------------------------------------------------------------------


def _async_return(value):
    """Create an async function that returns the given value."""

    async def _fn(*args, **kwargs):
        return value

    return _fn


async def _async_iter(items):
    """Create an async iterator from a list."""
    for item in items:
        yield item


class _AsyncContextManager:
    """Minimal async context manager wrapping a value."""

    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *args):
        pass

    def __call__(self, *args, **kwargs):
        return self
