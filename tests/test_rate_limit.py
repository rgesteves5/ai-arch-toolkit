from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._rate_limit import RateLimitMiddleware


def _make_request() -> Request:
    return Request(messages=[], system=None, tools=None, model="claude-sonnet-4-20250514")


class TestRateLimitValidation:
    def test_rejects_rpm_zero(self) -> None:
        with pytest.raises(ValueError, match="requests_per_minute must be positive"):
            RateLimitMiddleware(0)

    def test_rejects_negative_rpm(self) -> None:
        with pytest.raises(ValueError, match="requests_per_minute must be positive"):
            RateLimitMiddleware(-10)

    def test_rejects_burst_zero(self) -> None:
        with pytest.raises(ValueError, match="burst must be positive"):
            RateLimitMiddleware(60, burst=0)


class TestRateLimitPassThrough:
    def test_before_is_pass_through(self) -> None:
        mw = RateLimitMiddleware(60)
        request = _make_request()
        assert mw.before(request) is request

    def test_after_is_pass_through(self) -> None:
        mw = RateLimitMiddleware(60)
        request = _make_request()
        response = MagicMock()
        assert mw.after(request, response) is response


class TestRateLimitBurst:
    async def test_burst_allows_n_instant_then_waits(self) -> None:
        mw = RateLimitMiddleware(60, burst=3)
        request = _make_request()

        # First 3 calls should complete almost instantly
        t0 = time.monotonic()
        for _ in range(3):
            await mw.abefore(request)
        elapsed_burst = time.monotonic() - t0
        assert elapsed_burst < 0.1, f"Burst phase too slow: {elapsed_burst:.3f}s"

        # 4th call must wait for a token refill (~1 s at 60 rpm = 1 req/s)
        t1 = time.monotonic()
        await mw.abefore(request)
        elapsed_wait = time.monotonic() - t1
        assert elapsed_wait > 0.5, f"4th call should have waited, elapsed={elapsed_wait:.3f}s"
