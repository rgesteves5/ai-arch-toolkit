"""Exceptions for LLM API errors."""

from __future__ import annotations

from typing import Any


class APIError(Exception):
    """Raised when an LLM provider returns an HTTP error."""

    def __init__(self, status_code: int, body: dict[str, Any] | str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {body}")


class RateLimitError(APIError):
    """Raised on HTTP 429 — includes optional ``retry_after`` from the server."""

    def __init__(
        self,
        status_code: int,
        body: dict[str, Any] | str,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(status_code, body)
        self.retry_after = retry_after
