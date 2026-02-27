"""Tests for batch API."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ai_arch_toolkit.core._batch import BatchRequest, BatchResult
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._providers._base import BaseProvider
from ai_arch_toolkit.core._response import Response


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
