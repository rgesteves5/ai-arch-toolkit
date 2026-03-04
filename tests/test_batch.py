"""Tests for batch API — types, base provider, LLM delegation, and OpenAI provider."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_arch_toolkit.core._batch import BatchRequest, BatchResult
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._providers._base import BaseProvider
from ai_arch_toolkit.core._response import Response

# ---------------------------------------------------------------------------
# Existing tests — types, base provider, LLM delegation
# ---------------------------------------------------------------------------


class TestBatchTypes:
    def test_batch_request(self):
        req = BatchRequest(
            messages=[{"role": "user", "content": "Hi"}],
            custom_id="req-1",
        )
        assert req.custom_id == "req-1"
        assert req.system is None

    def test_batch_result_success(self):
        result = BatchResult(
            custom_id="req-1",
            response=Response(text="Hello!"),
        )
        assert result.response is not None
        assert result.error is None

    def test_batch_result_error(self):
        result = BatchResult(custom_id="req-1", error="rate limited")
        assert result.response is None
        assert result.error == "rate limited"

    def test_frozen(self):
        req = BatchRequest(messages=[], custom_id="x")
        with pytest.raises(AttributeError):
            req.custom_id = "y"  # type: ignore[misc]


class TestBaseProviderBatch:
    async def test_not_implemented(self):
        class DummyProvider(BaseProvider):
            async def complete(self, messages, **kwargs):
                pass

            def stream(self, messages, **kwargs):
                pass

        provider = DummyProvider()
        with pytest.raises(NotImplementedError, match="does not support batch API"):
            await provider.batch_submit([])
        with pytest.raises(NotImplementedError):
            await provider.batch_status("x")
        with pytest.raises(NotImplementedError):
            await provider.batch_results("x")


class TestLLMBatch:
    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_batch_submit(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.batch_submit.return_value = "batch-123"
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        batch_id = await llm.batch_submit(
            [
                {"custom_id": "r1", "messages": [{"role": "user", "content": "Hi"}]},
            ]
        )
        assert batch_id == "batch-123"

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_batch_status(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.batch_status.return_value = "completed"
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        status = await llm.batch_status("batch-123")
        assert status == "completed"

    @patch("ai_arch_toolkit.core._llm.create_provider")
    async def test_batch_results(self, mock_create):
        mock_provider = AsyncMock()
        mock_provider.batch_results.return_value = [
            BatchResult(custom_id="r1", response=Response(text="Hello")),
        ]
        mock_create.return_value = mock_provider

        llm = LLM("claude-sonnet-4-20250514", api_key="test")
        results = await llm.batch_results("batch-123")
        assert len(results) == 1
        assert results[0].response.text == "Hello"


# ---------------------------------------------------------------------------
# Helpers for OpenAI provider batch tests
# ---------------------------------------------------------------------------


def _make_openai_provider():
    """Create an OpenAIProvider with a fully mocked AsyncOpenAI client."""
    from ai_arch_toolkit.core._providers._openai import OpenAIProvider

    with patch("ai_arch_toolkit.core._providers._openai.openai.AsyncOpenAI"):
        provider = OpenAIProvider(model="gpt-4o", api_key="sk-test")

    client = MagicMock()
    client.files = MagicMock()
    client.files.create = AsyncMock()
    client.files.content = AsyncMock()
    client.batches = MagicMock()
    client.batches.create = AsyncMock()
    client.batches.retrieve = AsyncMock()
    provider._client = client
    return provider


def _sample_requests() -> list[dict[str, Any]]:
    """Two minimal batch requests for testing."""
    return [
        {
            "custom_id": "req-1",
            "messages": [{"role": "user", "content": "Hello"}],
            "system": "You are helpful.",
            "kwargs": {"max_tokens": 100},
        },
        {
            "custom_id": "req-2",
            "messages": [{"role": "user", "content": "Goodbye"}],
            "kwargs": {},
        },
    ]


def _success_jsonl_line(custom_id: str, text: str = "Hi there") -> str:
    """Build one JSONL output line with a successful chat completion body."""
    return json.dumps(
        {
            "custom_id": custom_id,
            "response": {
                "body": {
                    "choices": [
                        {
                            "message": {"content": text, "tool_calls": None},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    "model": "gpt-4o",
                }
            },
        }
    )


def _error_jsonl_line(custom_id: str, error_msg: str = "rate limit") -> str:
    """Build one JSONL output line with an error."""
    return json.dumps(
        {
            "custom_id": custom_id,
            "error": {"message": error_msg, "code": "rate_limit_exceeded"},
        }
    )


# ---------------------------------------------------------------------------
# OpenAI provider — batch_submit
# ---------------------------------------------------------------------------


class TestOpenAIBatchSubmit:
    async def test_constructs_jsonl_and_creates_batch(self):
        """batch_submit uploads JSONL, creates a batch, returns the batch id."""
        provider = _make_openai_provider()

        file_mock = MagicMock()
        file_mock.id = "file-abc123"
        provider._client.files.create.return_value = file_mock

        batch_mock = MagicMock()
        batch_mock.id = "batch-xyz789"
        provider._client.batches.create.return_value = batch_mock

        result = await provider.batch_submit(_sample_requests())

        provider._client.files.create.assert_awaited_once()
        call_kwargs = provider._client.files.create.call_args
        assert call_kwargs.kwargs["purpose"] == "batch"

        provider._client.batches.create.assert_awaited_once_with(
            input_file_id="file-abc123",
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        assert result == "batch-xyz789"

    async def test_jsonl_content_structure(self):
        """Uploaded JSONL contains correct per-request structure."""
        provider = _make_openai_provider()

        file_mock = MagicMock()
        file_mock.id = "file-abc"
        provider._client.files.create.return_value = file_mock

        batch_mock = MagicMock()
        batch_mock.id = "batch-1"
        provider._client.batches.create.return_value = batch_mock

        await provider.batch_submit(_sample_requests())

        file_arg = provider._client.files.create.call_args.kwargs["file"]
        content = file_arg.read().decode()
        lines = [json.loads(ln) for ln in content.strip().splitlines()]

        assert len(lines) == 2
        assert lines[0]["custom_id"] == "req-1"
        assert lines[0]["method"] == "POST"
        assert lines[0]["url"] == "/v1/chat/completions"
        assert lines[0]["body"]["model"] == "gpt-4o"
        assert lines[0]["body"]["max_tokens"] == 100

        # Second request uses default max_tokens (4096)
        assert lines[1]["custom_id"] == "req-2"
        assert lines[1]["body"]["max_tokens"] == 4096


# ---------------------------------------------------------------------------
# OpenAI provider — batch_status
# ---------------------------------------------------------------------------


class TestOpenAIBatchStatus:
    async def test_returns_completed(self):
        """batch_status returns the status string from the API."""
        provider = _make_openai_provider()
        batch_obj = MagicMock()
        batch_obj.status = "completed"
        provider._client.batches.retrieve.return_value = batch_obj

        status = await provider.batch_status("batch-xyz789")

        provider._client.batches.retrieve.assert_awaited_once_with("batch-xyz789")
        assert status == "completed"

    async def test_returns_in_progress(self):
        """Non-terminal statuses are returned faithfully."""
        provider = _make_openai_provider()
        batch_obj = MagicMock()
        batch_obj.status = "in_progress"
        provider._client.batches.retrieve.return_value = batch_obj

        assert await provider.batch_status("batch-123") == "in_progress"


# ---------------------------------------------------------------------------
# OpenAI provider — batch_results
# ---------------------------------------------------------------------------


class TestOpenAIBatchResults:
    async def test_parses_jsonl_output(self):
        """Successful JSONL lines become BatchResult objects with a Response."""
        provider = _make_openai_provider()

        batch_obj = MagicMock()
        batch_obj.output_file_id = "file-output-1"
        provider._client.batches.retrieve.return_value = batch_obj

        jsonl = "\n".join(
            [_success_jsonl_line("req-1", "Hello!"), _success_jsonl_line("req-2", "Bye")]
        )
        file_content = MagicMock()
        file_content.text = jsonl
        provider._client.files.content.return_value = file_content

        results = await provider.batch_results("batch-xyz")

        assert len(results) == 2
        assert isinstance(results[0], BatchResult)
        assert results[0].custom_id == "req-1"
        assert results[0].response is not None
        assert results[0].response.text == "Hello!"
        assert results[0].error is None

        assert results[1].custom_id == "req-2"
        assert results[1].response is not None
        assert results[1].response.text == "Bye"

    async def test_handles_errors_in_jsonl(self):
        """Error entries produce BatchResult with error field set."""
        provider = _make_openai_provider()

        batch_obj = MagicMock()
        batch_obj.output_file_id = "file-output-2"
        provider._client.batches.retrieve.return_value = batch_obj

        jsonl = "\n".join(
            [_success_jsonl_line("req-1"), _error_jsonl_line("req-2", "rate limit exceeded")]
        )
        file_content = MagicMock()
        file_content.text = jsonl
        provider._client.files.content.return_value = file_content

        results = await provider.batch_results("batch-err")

        assert len(results) == 2
        assert results[0].response is not None
        assert results[0].error is None

        assert results[1].response is None
        assert results[1].error is not None
        assert "rate_limit" in results[1].error or "rate limit" in results[1].error

    async def test_no_output_file_id_returns_empty(self):
        """When output_file_id is None, return an empty list."""
        provider = _make_openai_provider()

        batch_obj = MagicMock()
        batch_obj.output_file_id = None
        provider._client.batches.retrieve.return_value = batch_obj

        results = await provider.batch_results("batch-pending")

        assert results == []
        provider._client.files.content.assert_not_awaited()

    async def test_usage_is_populated(self):
        """Parsed responses carry usage information from the batch output."""
        provider = _make_openai_provider()

        batch_obj = MagicMock()
        batch_obj.output_file_id = "file-out"
        provider._client.batches.retrieve.return_value = batch_obj

        file_content = MagicMock()
        file_content.text = _success_jsonl_line("req-1")
        provider._client.files.content.return_value = file_content

        results = await provider.batch_results("batch-u")
        resp = results[0].response

        assert resp is not None
        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 5
