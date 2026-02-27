"""Tests for _retry.py — retry with exponential backoff."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ai_arch_toolkit.core._exceptions import APIError, RateLimitError
from ai_arch_toolkit.core._retry import RetryConfig, _compute_delay, _is_retryable, with_retry


class TestRetryConfig:
    def test_defaults(self):
        cfg = RetryConfig()
        assert cfg.max_retries == 3
        assert cfg.base_delay == 1.0
        assert cfg.max_delay == 60.0
        assert 429 in cfg.retry_on_status

    def test_frozen(self):
        cfg = RetryConfig()
        with pytest.raises(AttributeError):
            cfg.max_retries = 5  # type: ignore[misc]


class TestIsRetryable:
    def test_rate_limit_always_retryable(self):
        exc = RateLimitError(429, "rate limited")
        assert _is_retryable(exc, RetryConfig()) is True

    def test_server_error_retryable(self):
        exc = APIError(500, "server error")
        assert _is_retryable(exc, RetryConfig()) is True

    def test_client_error_not_retryable(self):
        exc = APIError(400, "bad request")
        assert _is_retryable(exc, RetryConfig()) is False

    def test_non_api_error_not_retryable(self):
        exc = ValueError("bad")
        assert _is_retryable(exc, RetryConfig()) is False


class TestComputeDelay:
    def test_respects_retry_after(self):
        delay = _compute_delay(0, RetryConfig(), retry_after=5.0)
        assert delay == 5.0

    def test_retry_after_capped_by_max_delay(self):
        delay = _compute_delay(0, RetryConfig(max_delay=3.0), retry_after=10.0)
        assert delay == 3.0

    def test_exponential_growth(self):
        cfg = RetryConfig(base_delay=1.0, max_delay=100.0)
        d0 = _compute_delay(0, cfg, retry_after=None)
        d1 = _compute_delay(1, cfg, retry_after=None)
        d2 = _compute_delay(2, cfg, retry_after=None)
        # Should roughly double (with some jitter)
        assert d0 < d1 < d2

    def test_capped_by_max_delay(self):
        cfg = RetryConfig(base_delay=1.0, max_delay=5.0)
        delay = _compute_delay(10, cfg, retry_after=None)
        assert delay <= 5.0


class TestWithRetry:
    @patch("ai_arch_toolkit.core._retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_success_first_try(self, mock_sleep):
        factory = AsyncMock(return_value="ok")
        result = await with_retry(factory, RetryConfig())
        assert result == "ok"
        factory.assert_awaited_once()
        mock_sleep.assert_not_awaited()

    @patch("ai_arch_toolkit.core._retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_retry_on_rate_limit(self, mock_sleep):
        factory = AsyncMock(side_effect=[RateLimitError(429, "rate limited"), "ok"])
        result = await with_retry(factory, RetryConfig(max_retries=2))
        assert result == "ok"
        assert factory.await_count == 2
        mock_sleep.assert_awaited_once()

    @patch("ai_arch_toolkit.core._retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_retry_on_server_error(self, mock_sleep):
        factory = AsyncMock(side_effect=[APIError(500, "internal"), "ok"])
        result = await with_retry(factory, RetryConfig(max_retries=2))
        assert result == "ok"

    @patch("ai_arch_toolkit.core._retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_exhausted_retries_raises(self, mock_sleep):
        exc = RateLimitError(429, "rate limited")
        factory = AsyncMock(side_effect=exc)
        with pytest.raises(RateLimitError):
            await with_retry(factory, RetryConfig(max_retries=2))
        assert factory.await_count == 3  # initial + 2 retries

    @patch("ai_arch_toolkit.core._retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_non_retryable_raises_immediately(self, mock_sleep):
        factory = AsyncMock(side_effect=APIError(400, "bad request"))
        with pytest.raises(APIError):
            await with_retry(factory, RetryConfig())
        factory.assert_awaited_once()
        mock_sleep.assert_not_awaited()

    @patch("ai_arch_toolkit.core._retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_respects_retry_after(self, mock_sleep):
        exc = RateLimitError(429, "rate limited", retry_after=2.5)
        factory = AsyncMock(side_effect=[exc, "ok"])
        await with_retry(factory, RetryConfig(max_retries=1))
        mock_sleep.assert_awaited_once()
        delay = mock_sleep.call_args[0][0]
        assert delay == 2.5
