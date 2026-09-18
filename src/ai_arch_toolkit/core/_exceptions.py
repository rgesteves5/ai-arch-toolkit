"""Normalized provider failures carry an explicit delivery disposition."""

from __future__ import annotations

from typing import Literal

from ai_arch_toolkit.core._response import Usage

type Delivery = Literal["not_sent", "unbilled", "indeterminate"]


class ProviderError(Exception):
    """A provider failure whose delivery determines metering, independently of retry.

    ``usage`` is what the provider reported the failed request consumed (a response that failed
    after it started, like Meta's ``response.failed``); the meter settles the failure with it.
    """

    def __init__(self, message: str, *, delivery: Delivery, usage: Usage | None = None) -> None:
        if delivery not in ("not_sent", "unbilled", "indeterminate"):
            raise ValueError(f"invalid delivery disposition: {delivery}")
        self.delivery: Delivery = delivery
        self.usage = usage
        super().__init__(message)


class RequestError(ProviderError, ValueError):
    """The request could not be constructed; no work was sent."""

    def __init__(self, message: str) -> None:
        super().__init__(message, delivery="not_sent")


class UnpricedModelError(RequestError):
    """A metered call to a model its scope cannot price; nothing was sent."""


class APIError(ProviderError):
    """An HTTP error response; billing remains uncertain unless explicitly classified."""

    def __init__(
        self,
        status_code: int,
        body: dict[str, object] | str,
        *,
        delivery: Delivery = "indeterminate",
        usage: Usage | None = None,
    ) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"API {status_code}: {body}", delivery=delivery, usage=usage)


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

    def __init__(self, message: str, *, usage: Usage | None = None) -> None:
        super().__init__(message, delivery="indeterminate", usage=usage)
