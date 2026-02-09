"""Async HTTP helpers using ``httpx``."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ai_arch_toolkit.llm._exceptions import APIError, RateLimitError
from ai_arch_toolkit.llm._http import NO_RETRY, RetryConfig, _should_retry, _wait_time


def _raise_for_status_httpx(r: httpx.Response) -> None:
    if r.is_success:
        return
    try:
        body: dict[str, Any] | str = r.json()
    except Exception:
        body = r.text
    if r.status_code == 429:
        raw_retry = r.headers.get("Retry-After")
        retry_after: float | None = None
        if raw_retry is not None:
            with contextlib.suppress(ValueError, TypeError):
                retry_after = float(raw_retry)
        raise RateLimitError(r.status_code, body, retry_after)
    raise APIError(r.status_code, body)


async def async_post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: int = 60,
    retry: RetryConfig | None = None,
) -> dict[str, Any]:
    """POST JSON asynchronously and return the parsed response."""
    config = retry or NO_RETRY
    last_exc: APIError | None = None
    for attempt in range(config.max_retries + 1):
        if attempt > 0 and last_exc is not None:
            retry_after = getattr(last_exc, "retry_after", None)
            await asyncio.sleep(_wait_time(attempt, config, retry_after))
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(url, headers=headers, json=payload, timeout=timeout)
                _raise_for_status_httpx(r)
                return r.json()
        except APIError as exc:
            last_exc = exc
            if not _should_retry(exc.status_code, attempt, config):
                raise
    raise last_exc  # type: ignore[misc]


async def async_stream_sse(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: int = 120,
    retry: RetryConfig | None = None,
) -> AsyncIterator[str]:
    """POST and yield SSE ``data:`` payloads asynchronously.

    Retries only on connection-level failures (before yielding starts).
    """
    config = retry or NO_RETRY
    last_exc: APIError | None = None
    for attempt in range(config.max_retries + 1):
        if attempt > 0 and last_exc is not None:
            retry_after = getattr(last_exc, "retry_after", None)
            await asyncio.sleep(_wait_time(attempt, config, retry_after))
        try:
            async with (
                httpx.AsyncClient() as client,
                client.stream("POST", url, headers=headers, json=payload, timeout=timeout) as r,
            ):
                if not r.is_success:
                    await r.aread()
                    _raise_for_status_httpx(r)
                async for line in r.aiter_lines():
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data: "):
                        yield line[len("data: ") :]
                return
        except APIError as exc:
            last_exc = exc
            if not _should_retry(exc.status_code, attempt, config):
                raise
    raise last_exc  # type: ignore[misc]


async def async_stream_ndjson(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: int = 120,
    retry: RetryConfig | None = None,
) -> AsyncIterator[str]:
    """POST and yield newline-delimited JSON lines asynchronously.

    Retries only on connection-level failures (before yielding starts).
    """
    config = retry or NO_RETRY
    last_exc: APIError | None = None
    for attempt in range(config.max_retries + 1):
        if attempt > 0 and last_exc is not None:
            retry_after = getattr(last_exc, "retry_after", None)
            await asyncio.sleep(_wait_time(attempt, config, retry_after))
        try:
            async with (
                httpx.AsyncClient() as client,
                client.stream("POST", url, headers=headers, json=payload, timeout=timeout) as r,
            ):
                if not r.is_success:
                    await r.aread()
                    _raise_for_status_httpx(r)
                async for line in r.aiter_lines():
                    if line:
                        yield line
                return
        except APIError as exc:
            last_exc = exc
            if not _should_retry(exc.status_code, attempt, config):
                raise
    raise last_exc  # type: ignore[misc]
