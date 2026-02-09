"""Tests for the async batch API clients."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_arch_toolkit.llm._async_batch import AsyncBatchClient
from ai_arch_toolkit.llm._batch import BatchJob, BatchRequest
from ai_arch_toolkit.llm._types import Message


def _make_openai_client() -> AsyncBatchClient:
    return AsyncBatchClient("openai", model="gpt-4o", api_key="sk-test")


def _make_anthropic_client() -> AsyncBatchClient:
    return AsyncBatchClient("anthropic", model="claude-sonnet-4-5-20250514", api_key="ant-test")


# --- Construction ---


def test_unsupported_provider() -> None:
    with pytest.raises(ValueError, match="not supported"):
        AsyncBatchClient("gemini", model="model", api_key="key")


def test_openai_headers() -> None:
    client = _make_openai_client()
    assert client._headers["Authorization"] == "Bearer sk-test"


def test_anthropic_headers() -> None:
    client = _make_anthropic_client()
    assert client._headers["x-api-key"] == "ant-test"
    assert client._headers["anthropic-version"] == "2023-06-01"


# --- OpenAI submit ---


class _MockHttpxResponse:
    """Minimal mock for httpx.Response."""

    def __init__(self, json_data: dict | None = None, text: str = "") -> None:
        self._json_data = json_data or {}
        self.text = text
        self.status_code = 200

    def json(self) -> dict:
        return self._json_data

    def raise_for_status(self) -> None:
        pass


@patch("ai_arch_toolkit.llm._async_http.httpx.AsyncClient")
@patch("httpx.AsyncClient")
async def test_openai_submit(
    mock_httpx_cls: MagicMock,
    mock_async_http_cls: MagicMock,
) -> None:
    # Mock the file upload via httpx.AsyncClient context manager
    upload_resp = _MockHttpxResponse(json_data={"id": "file-abc"})
    mock_httpx_client = AsyncMock()
    mock_httpx_client.post = AsyncMock(return_value=upload_resp)
    mock_httpx_client.__aenter__ = AsyncMock(return_value=mock_httpx_client)
    mock_httpx_client.__aexit__ = AsyncMock(return_value=False)
    mock_httpx_cls.return_value = mock_httpx_client

    # Mock async_post_json for batch creation
    batch_resp = {"id": "batch-123", "status": "in_progress"}
    mock_async_post = AsyncMock(return_value=batch_resp)
    mock_async_http_client = AsyncMock()
    mock_async_http_client.post = AsyncMock(return_value=_MockHttpxResponse(json_data=batch_resp))
    mock_async_http_client.__aenter__ = AsyncMock(return_value=mock_async_http_client)
    mock_async_http_client.__aexit__ = AsyncMock(return_value=False)
    mock_async_http_cls.return_value = mock_async_http_client

    client = _make_openai_client()
    with patch("ai_arch_toolkit.llm._async_batch.async_post_json", mock_async_post):
        job = await client.submit(
            [
                BatchRequest(
                    custom_id="req-1",
                    messages=[Message("user", "Hello")],
                    system="Be helpful",
                )
            ]
        )

    assert job.id == "batch-123"
    assert job.status == "in_progress"
    assert job.provider == "openai"


# --- OpenAI status ---


@patch("httpx.AsyncClient")
async def test_openai_status(mock_httpx_cls: MagicMock) -> None:
    resp = _MockHttpxResponse(json_data={"id": "batch-123", "status": "completed"})
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_httpx_cls.return_value = mock_client

    client = _make_openai_client()
    job = BatchJob(id="batch-123", status="in_progress", provider="openai")
    updated = await client.status(job)
    assert updated.status == "completed"


# --- OpenAI results ---


@patch("httpx.AsyncClient")
async def test_openai_results(mock_httpx_cls: MagicMock) -> None:
    result_line = json.dumps(
        {
            "custom_id": "req-1",
            "response": {
                "body": {
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hi!"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
            },
        }
    )
    resp = _MockHttpxResponse(text=result_line)
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_httpx_cls.return_value = mock_client

    client = _make_openai_client()
    job = BatchJob(
        id="batch-123",
        status="completed",
        provider="openai",
        raw={"output_file_id": "file-out"},
    )
    results = await client.results(job)

    assert len(results) == 1
    assert results[0].custom_id == "req-1"
    assert results[0].response is not None
    assert results[0].response.text == "Hi!"


# --- Anthropic submit ---


@patch("ai_arch_toolkit.llm._async_batch.async_post_json")
async def test_anthropic_submit(mock_post: AsyncMock) -> None:
    mock_post.return_value = {"id": "msgbatch-abc", "processing_status": "in_progress"}

    client = _make_anthropic_client()
    job = await client.submit(
        [
            BatchRequest(
                custom_id="req-1",
                messages=[Message("user", "Hello")],
            )
        ]
    )

    assert job.id == "msgbatch-abc"
    assert job.status == "in_progress"
    assert job.provider == "anthropic"


# --- Anthropic status ---


@patch("httpx.AsyncClient")
async def test_anthropic_status(mock_httpx_cls: MagicMock) -> None:
    resp = _MockHttpxResponse(json_data={"id": "msgbatch-abc", "processing_status": "ended"})
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_httpx_cls.return_value = mock_client

    client = _make_anthropic_client()
    job = BatchJob(id="msgbatch-abc", status="in_progress", provider="anthropic")
    updated = await client.status(job)
    assert updated.status == "ended"


# --- Anthropic results ---


@patch("httpx.AsyncClient")
async def test_anthropic_results(mock_httpx_cls: MagicMock) -> None:
    result_line = json.dumps(
        {
            "custom_id": "req-1",
            "result": {
                "type": "succeeded",
                "message": {
                    "content": [{"type": "text", "text": "Hi!"}],
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                    "stop_reason": "end_turn",
                },
            },
        }
    )
    resp = _MockHttpxResponse(text=result_line)
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_httpx_cls.return_value = mock_client

    client = _make_anthropic_client()
    job = BatchJob(id="msgbatch-abc", status="ended", provider="anthropic")
    results = await client.results(job)

    assert len(results) == 1
    assert results[0].custom_id == "req-1"
    assert results[0].response is not None
    assert results[0].response.text == "Hi!"


@patch("httpx.AsyncClient")
async def test_anthropic_results_with_error(mock_httpx_cls: MagicMock) -> None:
    result_line = json.dumps(
        {
            "custom_id": "req-2",
            "result": {
                "type": "errored",
                "error": {"message": "Server error"},
            },
        }
    )
    resp = _MockHttpxResponse(text=result_line)
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_httpx_cls.return_value = mock_client

    client = _make_anthropic_client()
    job = BatchJob(id="msgbatch-abc", status="ended", provider="anthropic")
    results = await client.results(job)

    assert len(results) == 1
    assert results[0].custom_id == "req-2"
    assert results[0].response is None
    assert results[0].error == "Server error"
