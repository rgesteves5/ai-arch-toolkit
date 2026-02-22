"""Unified HTTP helpers — async (httpx) + sync (requests)."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any

import httpx
import requests

from ai_arch_toolkit._exceptions import APIError, RateLimitError


@dataclass(frozen=True, slots=True)
class RetryConfig:
    """Configuration for automatic retries with exponential backoff."""

    max_retries: int = 3
    backoff_factor: float = 2.0
    retryable_codes: frozenset[int] = frozenset({429, 500, 502, 503, 504})


NO_RETRY = RetryConfig(max_retries=0)


def _should_retry(status_code: int, attempt: int, config: RetryConfig) -> bool:
    return attempt < config.max_retries and status_code in config.retryable_codes


def _wait_time(attempt: int, config: RetryConfig, retry_after: float | None = None) -> float:
    if retry_after is not None and retry_after > 0:
        return retry_after
    return config.backoff_factor**attempt


# ---------------------------------------------------------------------------
# Sync helpers (requests)
# ---------------------------------------------------------------------------


def _raise_for_status_requests(r: requests.Response) -> None:
    if r.ok:
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


def post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: int = 60,
    retry: RetryConfig | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """POST JSON and return the parsed response, raising on HTTP errors."""
    http = session or requests
    config = retry or NO_RETRY
    last_exc: APIError | None = None
    for attempt in range(config.max_retries + 1):
        if attempt > 0 and last_exc is not None:
            retry_after = getattr(last_exc, "retry_after", None)
            time.sleep(_wait_time(attempt, config, retry_after))
        try:
            r = http.post(url, headers=headers, json=payload, timeout=timeout)
            _raise_for_status_requests(r)
            return r.json()
        except APIError as exc:
            last_exc = exc
            if not _should_retry(exc.status_code, attempt, config):
                raise
    raise last_exc  # type: ignore[misc]


def stream_sse(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: int = 120,
    retry: RetryConfig | None = None,
    session: requests.Session | None = None,
) -> Iterator[str]:
    """POST and yield SSE ``data:`` payloads (without the prefix)."""
    http = session or requests
    config = retry or NO_RETRY
    last_exc: APIError | None = None
    for attempt in range(config.max_retries + 1):
        if attempt > 0 and last_exc is not None:
            retry_after = getattr(last_exc, "retry_after", None)
            time.sleep(_wait_time(attempt, config, retry_after))
        try:
            with http.post(url, headers=headers, json=payload, stream=True, timeout=timeout) as r:
                _raise_for_status_requests(r)
                for line in r.iter_lines(decode_unicode=True):
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data: "):
                        payload_str = line[len("data: "):]
                        # TODO: filter "data: [DONE]" for OpenAI compat (Phase 2)
                        yield payload_str
                return
        except APIError as exc:
            last_exc = exc
            if not _should_retry(exc.status_code, attempt, config):
                raise
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Async helpers (httpx)
# ---------------------------------------------------------------------------


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
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """POST JSON asynchronously and return the parsed response."""
    config = retry or NO_RETRY
    last_exc: APIError | None = None
    for attempt in range(config.max_retries + 1):
        if attempt > 0 and last_exc is not None:
            retry_after = getattr(last_exc, "retry_after", None)
            await asyncio.sleep(_wait_time(attempt, config, retry_after))
        try:
            if client is not None:
                r = await client.post(url, headers=headers, json=payload, timeout=timeout)
                _raise_for_status_httpx(r)
                return r.json()
            async with httpx.AsyncClient() as default_client:
                r = await default_client.post(url, headers=headers, json=payload, timeout=timeout)
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
    client: httpx.AsyncClient | None = None,
) -> AsyncIterator[str]:
    """POST and yield SSE ``data:`` payloads asynchronously."""
    config = retry or NO_RETRY
    last_exc: APIError | None = None
    for attempt in range(config.max_retries + 1):
        if attempt > 0 and last_exc is not None:
            retry_after = getattr(last_exc, "retry_after", None)
            await asyncio.sleep(_wait_time(attempt, config, retry_after))
        try:
            if client is not None:
                async with client.stream(
                    "POST", url, headers=headers, json=payload, timeout=timeout
                ) as r:
                    if not r.is_success:
                        await r.aread()
                        _raise_for_status_httpx(r)
                    async for line in r.aiter_lines():
                        if not line or line.startswith(":"):
                            continue
                        if line.startswith("data: "):
                            # TODO: filter "data: [DONE]" for OpenAI compat (Phase 2)
                            yield line[len("data: ") :]
                    return
            async with (
                httpx.AsyncClient() as default_client,
                default_client.stream(
                    "POST", url, headers=headers, json=payload, timeout=timeout
                ) as r,
            ):
                if not r.is_success:
                    await r.aread()
                    _raise_for_status_httpx(r)
                async for line in r.aiter_lines():
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data: "):
                        # TODO: filter "data: [DONE]" for OpenAI compat (Phase 2)
                        yield line[len("data: ") :]
                return
        except APIError as exc:
            last_exc = exc
            if not _should_retry(exc.status_code, attempt, config):
                raise
    raise last_exc  # type: ignore[misc]
