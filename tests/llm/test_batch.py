"""Tests for the batch API clients."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from ai_arch_toolkit.llm._batch import BatchClient, BatchJob, BatchRequest, BatchResult
from ai_arch_toolkit.llm._types import Message
from tests.conftest import MockResponse


def _make_openai_client() -> BatchClient:
    return BatchClient("openai", model="gpt-4o", api_key="sk-test")


def _make_anthropic_client() -> BatchClient:
    return BatchClient("anthropic", model="claude-sonnet-4-5-20250514", api_key="ant-test")


# --- Construction ---


def test_unsupported_provider() -> None:
    with pytest.raises(ValueError, match="not supported"):
        BatchClient("gemini", model="model", api_key="key")


def test_openai_headers() -> None:
    client = _make_openai_client()
    assert client._headers["Authorization"] == "Bearer sk-test"


def test_anthropic_headers() -> None:
    client = _make_anthropic_client()
    assert client._headers["x-api-key"] == "ant-test"
    assert client._headers["anthropic-version"] == "2023-06-01"


# --- OpenAI submit ---


@patch("requests.post")
def test_openai_submit(mock_post: MagicMock) -> None:
    # First call: file upload; second: batch creation (post_json)
    upload_resp = MockResponse(json_data={"id": "file-abc"})
    batch_resp = MockResponse(json_data={"id": "batch-123", "status": "in_progress"})
    mock_post.side_effect = [upload_resp, batch_resp]

    client = _make_openai_client()
    job = client.submit(
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

    # Verify file upload call
    upload_call = mock_post.call_args_list[0]
    assert "files" in upload_call.kwargs or (len(upload_call.args) > 0)


# --- OpenAI results ---


@patch("requests.get")
def test_openai_results(mock_get: MagicMock) -> None:
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
    mock_get.return_value = MockResponse(text=result_line, status_code=200)

    client = _make_openai_client()
    job = BatchJob(
        id="batch-123",
        status="completed",
        provider="openai",
        raw={"output_file_id": "file-out"},
    )
    results = client.results(job)

    assert len(results) == 1
    assert results[0].custom_id == "req-1"
    assert results[0].response is not None
    assert results[0].response.text == "Hi!"
    assert results[0].error is None


@patch("requests.get")
def test_openai_results_with_error(mock_get: MagicMock) -> None:
    result_line = json.dumps(
        {
            "custom_id": "req-2",
            "error": {"message": "rate limited", "code": 429},
        }
    )
    mock_get.return_value = MockResponse(text=result_line, status_code=200)

    client = _make_openai_client()
    job = BatchJob(
        id="batch-123",
        status="completed",
        provider="openai",
        raw={"output_file_id": "file-out"},
    )
    results = client.results(job)

    assert len(results) == 1
    assert results[0].custom_id == "req-2"
    assert results[0].response is None
    assert results[0].error is not None


# --- Anthropic submit ---


@patch("requests.post")
def test_anthropic_submit(mock_post: MagicMock) -> None:
    mock_post.return_value = MockResponse(
        json_data={"id": "msgbatch-abc", "processing_status": "in_progress"}
    )

    client = _make_anthropic_client()
    job = client.submit(
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

    # Verify payload
    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    reqs = payload["requests"]
    assert len(reqs) == 1
    assert reqs[0]["custom_id"] == "req-1"
    assert reqs[0]["params"]["model"] == "claude-sonnet-4-5-20250514"
    assert reqs[0]["params"]["messages"][0] == {"role": "user", "content": "Hello"}


# --- Anthropic results ---


@patch("requests.get")
def test_anthropic_results(mock_get: MagicMock) -> None:
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
    mock_get.return_value = MockResponse(text=result_line, status_code=200)

    client = _make_anthropic_client()
    job = BatchJob(id="msgbatch-abc", status="ended", provider="anthropic")
    results = client.results(job)

    assert len(results) == 1
    assert results[0].custom_id == "req-1"
    assert results[0].response is not None
    assert results[0].response.text == "Hi!"


@patch("requests.get")
def test_anthropic_results_with_error(mock_get: MagicMock) -> None:
    result_line = json.dumps(
        {
            "custom_id": "req-2",
            "result": {
                "type": "errored",
                "error": {"message": "Server error"},
            },
        }
    )
    mock_get.return_value = MockResponse(text=result_line, status_code=200)

    client = _make_anthropic_client()
    job = BatchJob(id="msgbatch-abc", status="ended", provider="anthropic")
    results = client.results(job)

    assert len(results) == 1
    assert results[0].custom_id == "req-2"
    assert results[0].response is None
    assert results[0].error == "Server error"


# --- Status ---


@patch("requests.get")
def test_openai_status(mock_get: MagicMock) -> None:
    mock_get.return_value = MockResponse(json_data={"id": "batch-123", "status": "completed"})
    client = _make_openai_client()
    job = BatchJob(id="batch-123", status="in_progress", provider="openai")
    updated = client.status(job)
    assert updated.status == "completed"


@patch("requests.get")
def test_anthropic_status(mock_get: MagicMock) -> None:
    mock_get.return_value = MockResponse(
        json_data={"id": "msgbatch-abc", "processing_status": "ended"}
    )
    client = _make_anthropic_client()
    job = BatchJob(id="msgbatch-abc", status="in_progress", provider="anthropic")
    updated = client.status(job)
    assert updated.status == "ended"


# --- Dataclass types ---


def test_batch_request_fields() -> None:
    req = BatchRequest(
        custom_id="test",
        messages=[Message("user", "Hi")],
        system="Be brief",
    )
    assert req.custom_id == "test"
    assert req.system == "Be brief"
    assert req.tools is None
    assert req.json_schema is None


def test_batch_result_defaults() -> None:
    r = BatchResult(custom_id="test")
    assert r.response is None
    assert r.error is None


def test_batch_job_defaults() -> None:
    j = BatchJob(id="j1", status="pending", provider="openai")
    assert j.raw == {}
