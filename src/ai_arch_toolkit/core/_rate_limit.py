"""Proactive rate limiting middleware using token bucket algorithm."""

from __future__ import annotations

import asyncio
import time

from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._response import Response


class RateLimitMiddleware:
    """Proactive rate limiter using token bucket.

    .. note::
        Rate limiting is only applied on async paths (``abefore``).  The sync
        ``before()`` hook is a pass-through, so ``LLM.stream()`` and
        ``LLM.stream_events()`` — which use sync middleware hooks — bypass the
        limiter.  TODO: add sync blocking or document this limitation.

    Args:
        requests_per_minute: Maximum sustained rate.
        burst: Maximum burst size (defaults to ``int(requests_per_minute)``).
    """

    def __init__(self, requests_per_minute: float, burst: int | None = None) -> None:
        if requests_per_minute <= 0:
            raise ValueError(f"requests_per_minute must be positive, got {requests_per_minute}")
        if burst is not None and burst <= 0:
            raise ValueError(f"burst must be positive, got {burst}")

        self._rate = requests_per_minute / 60.0  # tokens/sec
        self._burst = burst or int(requests_per_minute)
        self._tokens = float(self._burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(self._burst, self._tokens + (now - self._last_refill) * self._rate)
        self._last_refill = now

    def before(self, request: Request) -> Request:
        return request

    def after(self, request: Request, response: Response) -> Response:
        return response

    async def abefore(self, request: Request) -> Request:
        async with self._lock:
            self._refill()
            while self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self._rate
                await asyncio.sleep(wait)
                self._refill()
            self._tokens -= 1.0
        return request

    async def aafter(self, request: Request, response: Response) -> Response:
        return response
