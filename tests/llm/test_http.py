"""Tests for _http.py helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ai_arch_toolkit.llm._exceptions import APIError, RateLimitError
from ai_arch_toolkit.llm._http import (
    NO_RETRY,
    RetryConfig,
    _should_retry,
    _wait_time,
    post_json,
    stream_ndjson,
    stream_sse,
)
from tests.conftest import MockResponse


def test_post_json_success(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data={"result": "ok"})
    result = post_json("https://example.com", {"Auth": "key"}, {"q": "test"})
    assert result == {"result": "ok"}
    mock_post.assert_called_once()


def test_post_json_api_error(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data={"error": "bad"}, status_code=400)
    with pytest.raises(APIError) as exc_info:
        post_json("https://example.com", {}, {})
    assert exc_info.value.status_code == 400
    assert exc_info.value.body == {"error": "bad"}


def test_post_json_api_error_non_json(mock_post: MagicMock) -> None:
    resp = MockResponse(status_code=500, text="Internal Server Error")
    resp._json_data = None
    mock_post.return_value = resp
    with pytest.raises(APIError) as exc_info:
        post_json("https://example.com", {}, {})
    assert exc_info.value.status_code == 500
    assert exc_info.value.body == "Internal Server Error"


def test_stream_sse_data_lines(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(
        lines=[
            'data: {"text": "hello"}',
            "",
            'data: {"text": "world"}',
            "data: [DONE]",
        ]
    )
    chunks = list(stream_sse("https://example.com", {}, {}))
    assert chunks == ['{"text": "hello"}', '{"text": "world"}', "[DONE]"]


def test_stream_sse_skips_comments(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(
        lines=[": comment", 'data: {"text": "hello"}', "event: ping"]
    )
    chunks = list(stream_sse("https://example.com", {}, {}))
    # Only data: lines are yielded; comments and non-data lines are skipped
    assert chunks == ['{"text": "hello"}']


def test_stream_sse_error(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data={"error": "unauthorized"}, status_code=401)
    with pytest.raises(APIError) as exc_info:
        list(stream_sse("https://example.com", {}, {}))
    assert exc_info.value.status_code == 401


def test_stream_ndjson(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(lines=['{"text": "hello"}', "", '{"text": "world"}'])
    chunks = list(stream_ndjson("https://example.com", {}, {}))
    assert chunks == ['{"text": "hello"}', '{"text": "world"}']


def test_stream_ndjson_error(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data={"error": "forbidden"}, status_code=403)
    with pytest.raises(APIError) as exc_info:
        list(stream_ndjson("https://example.com", {}, {}))
    assert exc_info.value.status_code == 403


# --- Retry tests ---


def test_should_retry_within_limit() -> None:
    config = RetryConfig(max_retries=3)
    assert _should_retry(429, 0, config) is True
    assert _should_retry(500, 2, config) is True


def test_should_retry_at_limit() -> None:
    config = RetryConfig(max_retries=3)
    assert _should_retry(429, 3, config) is False


def test_should_retry_non_retryable_code() -> None:
    config = RetryConfig(max_retries=3)
    assert _should_retry(400, 0, config) is False
    assert _should_retry(401, 1, config) is False


def test_wait_time_exponential() -> None:
    config = RetryConfig(backoff_factor=2.0)
    assert _wait_time(0, config) == 1.0  # 2^0
    assert _wait_time(1, config) == 2.0  # 2^1
    assert _wait_time(2, config) == 4.0  # 2^2


def test_wait_time_uses_retry_after() -> None:
    config = RetryConfig(backoff_factor=2.0)
    assert _wait_time(1, config, retry_after=10.0) == 10.0


def test_wait_time_ignores_zero_retry_after() -> None:
    config = RetryConfig(backoff_factor=2.0)
    assert _wait_time(1, config, retry_after=0.0) == 2.0


def test_no_retry_constant() -> None:
    assert NO_RETRY.max_retries == 0


def test_rate_limit_error_on_429(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(
        json_data={"error": "rate limited"},
        status_code=429,
        headers={"Retry-After": "5"},
    )
    with pytest.raises(RateLimitError) as exc_info:
        post_json("https://example.com", {}, {})
    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after == 5.0


def test_rate_limit_error_without_retry_after(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data={"error": "rate limited"}, status_code=429)
    with pytest.raises(RateLimitError) as exc_info:
        post_json("https://example.com", {}, {})
    assert exc_info.value.retry_after is None


@patch("ai_arch_toolkit.llm._http.time.sleep")
def test_post_json_retries_on_429(mock_sleep: MagicMock, mock_post: MagicMock) -> None:
    mock_post.side_effect = [
        MockResponse(json_data={"error": "rate limited"}, status_code=429),
        MockResponse(json_data={"result": "ok"}),
    ]
    config = RetryConfig(max_retries=2, backoff_factor=1.0)
    result = post_json("https://example.com", {}, {}, retry=config)
    assert result == {"result": "ok"}
    assert mock_post.call_count == 2
    mock_sleep.assert_called_once()


@patch("ai_arch_toolkit.llm._http.time.sleep")
def test_post_json_retries_on_500(mock_sleep: MagicMock, mock_post: MagicMock) -> None:
    resp500 = MockResponse(status_code=500, text="Server Error")
    resp500._json_data = None
    mock_post.side_effect = [
        resp500,
        MockResponse(json_data={"result": "ok"}),
    ]
    config = RetryConfig(max_retries=2, backoff_factor=1.0)
    result = post_json("https://example.com", {}, {}, retry=config)
    assert result == {"result": "ok"}
    assert mock_post.call_count == 2


@patch("ai_arch_toolkit.llm._http.time.sleep")
def test_post_json_exhausts_retries(mock_sleep: MagicMock, mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data={"error": "rate limited"}, status_code=429)
    config = RetryConfig(max_retries=2, backoff_factor=1.0)
    with pytest.raises(RateLimitError):
        post_json("https://example.com", {}, {}, retry=config)
    assert mock_post.call_count == 3  # initial + 2 retries


def test_post_json_no_retry_on_400(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(json_data={"error": "bad"}, status_code=400)
    config = RetryConfig(max_retries=3)
    with pytest.raises(APIError):
        post_json("https://example.com", {}, {}, retry=config)
    assert mock_post.call_count == 1


@patch("ai_arch_toolkit.llm._http.time.sleep")
def test_stream_sse_retries_before_yield(mock_sleep: MagicMock, mock_post: MagicMock) -> None:
    mock_post.side_effect = [
        MockResponse(json_data={"error": "rate limited"}, status_code=429),
        MockResponse(lines=['data: {"text": "hello"}']),
    ]
    config = RetryConfig(max_retries=2, backoff_factor=1.0)
    chunks = list(stream_sse("https://example.com", {}, {}, retry=config))
    assert chunks == ['{"text": "hello"}']
    assert mock_post.call_count == 2


@patch("ai_arch_toolkit.llm._http.time.sleep")
def test_stream_ndjson_retries_before_yield(mock_sleep: MagicMock, mock_post: MagicMock) -> None:
    mock_post.side_effect = [
        MockResponse(json_data={"error": "overloaded"}, status_code=503),
        MockResponse(lines=['{"text": "hello"}']),
    ]
    config = RetryConfig(max_retries=2, backoff_factor=1.0)
    chunks = list(stream_ndjson("https://example.com", {}, {}, retry=config))
    assert chunks == ['{"text": "hello"}']
    assert mock_post.call_count == 2
