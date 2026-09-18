"""Typed delivery errors keep existing builtin exception handlers useful."""

from __future__ import annotations

import pytest

import ai_arch_toolkit as toolkit
from ai_arch_toolkit.core import _exceptions as errors
from ai_arch_toolkit.core._providers._base import transport_error
from ai_arch_toolkit.core._retry import RetryConfig, _is_retryable


@pytest.mark.parametrize(
    "cause,builtin,name",
    [
        (OSError("connection lost"), ConnectionError, "TransportError"),
        (TimeoutError("connection lost"), TimeoutError, "ProviderTimeout"),
    ],
)
def test_network_mapper_raises_normalized_error_without_losing_builtin_handler(
    cause, builtin, name
):
    exc = transport_error(cause, not_sent=(), timeouts=(TimeoutError,))
    with pytest.raises(builtin, match="connection lost"):
        raise exc
    assert isinstance(exc, errors.ProviderError)
    assert type(exc).__name__ == name
    assert exc.delivery == "indeterminate"
    assert _is_retryable(exc, RetryConfig())


def test_a_failure_while_connecting_was_never_sent():
    exc = transport_error(
        ConnectionRefusedError("refused"), not_sent=(ConnectionRefusedError,), timeouts=()
    )
    assert (type(exc), exc.delivery) == (errors.TransportError, "not_sent")
    assert _is_retryable(exc, RetryConfig())


def test_rate_limit_is_explicitly_unbilled_and_still_an_api_error():
    exc = errors.RateLimitError(429, "busy", retry_after=0.2)
    with pytest.raises(errors.APIError):
        raise exc
    assert exc.delivery == "unbilled"
    assert exc.retry_after == 0.2


def test_request_failure_keeps_valueerror_handler_and_never_retries():
    exc = errors.RequestError("unsupported option")
    with pytest.raises(ValueError, match="unsupported option"):
        raise exc
    assert exc.delivery == "not_sent"
    assert not _is_retryable(exc, RetryConfig())


def test_response_failure_is_separate_from_http_status():
    exc = errors.ResponseError("invalid response")
    assert exc.delivery == "indeterminate"
    assert not isinstance(exc, errors.APIError)
    assert isinstance(exc, errors.ProviderError)
    assert not _is_retryable(exc, RetryConfig())


def test_error_base_requires_valid_delivery():
    with pytest.raises(TypeError):
        errors.ProviderError("missing delivery")
    with pytest.raises(ValueError, match="delivery"):
        errors.ProviderError("invalid", delivery="invented")


@pytest.mark.parametrize(
    "name",
    [
        "ProviderError",
        "RequestError",
        "TransportError",
        "ProviderTimeout",
        "ResponseError",
    ],
)
def test_errors_are_public_exports(name):
    assert getattr(toolkit, name) is getattr(errors, name)
