"""Normalized provider failures carry an explicit delivery disposition."""

from __future__ import annotations

from typing import Literal

type Delivery = Literal["not_sent", "unbilled", "indeterminate"]


class ProviderError(Exception):
    """A provider failure whose delivery determines metering, independently of retry."""

    def __init__(self, message: str, *, delivery: Delivery) -> None:
        if delivery not in ("not_sent", "unbilled", "indeterminate"):
            raise ValueError(f"invalid delivery disposition: {delivery}")
        self.delivery: Delivery = delivery
        super().__init__(message)


class RequestError(ProviderError, ValueError):
    """The request could not be constructed; no work was sent."""

    def __init__(self, message: str) -> None:
        super().__init__(message, delivery="not_sent")


class APIError(ProviderError):
    """An HTTP error response; billing remains uncertain unless explicitly classified."""

    def __init__(
        self,
        status_code: int,
        body: dict[str, object] | str,
        *,
        delivery: Delivery = "indeterminate",
    ) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"API {status_code}: {body}", delivery=delivery)


class RateLimitError(APIError):
    """An unbilled rate-limit response, with an optional server retry delay."""

    def __init__(
        self,
        status_code: int,
        body: dict[str, object] | str,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(status_code, body, delivery="unbilled")
        self.retry_after = retry_after


class TransportError(ProviderError, ConnectionError):
    """Transport failed; callers may continue to catch ConnectionError."""

    def __init__(self, message: str, *, delivery: Delivery = "indeterminate") -> None:
        super().__init__(message, delivery=delivery)


class ProviderTimeout(ProviderError, TimeoutError):
    """Provider I/O timed out; callers may continue to catch TimeoutError."""

    def __init__(self, message: str, *, delivery: Delivery = "indeterminate") -> None:
        super().__init__(message, delivery=delivery)


class ResponseError(ProviderError):
    """A successful HTTP response or stream could not yield a usable result."""

    def __init__(self, message: str) -> None:
        super().__init__(message, delivery="indeterminate")
