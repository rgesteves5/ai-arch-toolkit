"""Retry with exponential backoff and jitter."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ai_arch_toolkit.core._exceptions import APIError, RateLimitError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class RetryConfig:
    """Configuration for automatic retries with exponential backoff."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    retry_on_status: tuple[int, ...] = (429, 500, 502, 503, 504)

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.base_delay <= 0:
            raise ValueError(f"base_delay must be positive, got {self.base_delay}")
        if self.max_delay <= 0:
            raise ValueError(f"max_delay must be positive, got {self.max_delay}")


def _is_retryable(exc: Exception, config: RetryConfig) -> bool:
    """Check if an exception is retryable."""
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIError):
        return exc.status_code in config.retry_on_status
    return False


def _compute_delay(attempt: int, config: RetryConfig, retry_after: float | None) -> float:
    """Compute delay for the next retry attempt."""
    if retry_after is not None and retry_after > 0:
        return min(retry_after, config.max_delay)
    delay = config.base_delay * (2**attempt)
    jitter = random.uniform(0, delay * 0.25)
    return min(delay + jitter, config.max_delay)


async def _wait_before_retry(
    exc: Exception,
    attempt: int,
    config: RetryConfig,
) -> bool:
    """Wait before the next retry, or return ``False`` when retries are exhausted."""
    if not _is_retryable(exc, config) or attempt == config.max_retries:
        return False
    retry_after = getattr(exc, "retry_after", None)
    delay = _compute_delay(attempt, config, retry_after)
    logger.info(
        "Retry %d/%d after %.1fs (error: %s)",
        attempt + 1,
        config.max_retries,
        delay,
        exc,
    )
    await asyncio.sleep(delay)
    return True


async def with_retry[T](
    coro_factory: Callable[[], Awaitable[T]],
    config: RetryConfig,
) -> T:
    """Call ``coro_factory()`` with exponential backoff on retryable errors.

    ``coro_factory`` is a zero-arg callable that returns a new awaitable on each
    invocation (since coroutines are single-use).
    """
    last_exc: Exception | None = None
    for attempt in range(config.max_retries + 1):
        try:
            return await coro_factory()
        except Exception as exc:
            if not await _wait_before_retry(exc, attempt, config):
                raise
            last_exc = exc

    # Should not be reached, but satisfy type checker
    assert last_exc is not None
    raise last_exc
