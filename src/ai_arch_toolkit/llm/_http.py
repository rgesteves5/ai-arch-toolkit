"""Thin HTTP helpers around ``requests``."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import requests

from ai_arch_toolkit.llm._exceptions import APIError, RateLimitError


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


def _raise_for_status(r: requests.Response) -> None:
    if not r.ok:
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
) -> dict[str, Any]:
    """POST JSON and return the parsed response, raising on HTTP errors."""
    config = retry or NO_RETRY
    last_exc: APIError | None = None
    for attempt in range(config.max_retries + 1):
        if attempt > 0 and last_exc is not None:
            retry_after = getattr(last_exc, "retry_after", None)
            time.sleep(_wait_time(attempt, config, retry_after))
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            _raise_for_status(r)
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
) -> Iterator[str]:
    """POST and yield SSE ``data:`` payloads (without the prefix).

    Retries only on connection-level failures (before yielding starts).
    """
    config = retry or NO_RETRY
    last_exc: APIError | None = None
    for attempt in range(config.max_retries + 1):
        if attempt > 0 and last_exc is not None:
            retry_after = getattr(last_exc, "retry_after", None)
            time.sleep(_wait_time(attempt, config, retry_after))
        try:
            with requests.post(
                url, headers=headers, json=payload, stream=True, timeout=timeout
            ) as r:
                _raise_for_status(r)
                # Connected successfully — no more retries once we start yielding.
                for line in r.iter_lines(decode_unicode=True):
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


def stream_ndjson(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: int = 120,
    retry: RetryConfig | None = None,
) -> Iterator[str]:
    """POST and yield newline-delimited JSON lines.

    Retries only on connection-level failures (before yielding starts).
    """
    config = retry or NO_RETRY
    last_exc: APIError | None = None
    for attempt in range(config.max_retries + 1):
        if attempt > 0 and last_exc is not None:
            retry_after = getattr(last_exc, "retry_after", None)
            time.sleep(_wait_time(attempt, config, retry_after))
        try:
            with requests.post(
                url, headers=headers, json=payload, stream=True, timeout=timeout
            ) as r:
                _raise_for_status(r)
                for line in r.iter_lines(decode_unicode=True):
                    if line:
                        yield line
                return
        except APIError as exc:
            last_exc = exc
            if not _should_retry(exc.status_code, attempt, config):
                raise
    raise last_exc  # type: ignore[misc]
