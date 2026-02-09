"""Async batch API support for OpenAI and Anthropic."""

from __future__ import annotations

import json
from typing import Any

import httpx

from ai_arch_toolkit.llm._async_http import async_post_json
from ai_arch_toolkit.llm._batch import (
    BatchJob,
    BatchRequest,
    BatchResult,
)
from ai_arch_toolkit.llm._http import RetryConfig
from ai_arch_toolkit.llm._providers._anthropic import (
    _items_to_wire as anthropic_items_to_wire,
)
from ai_arch_toolkit.llm._providers._anthropic import (
    _parse_response as anthropic_parse_response,
)
from ai_arch_toolkit.llm._providers._anthropic import (
    _tool_to_anthropic,
)
from ai_arch_toolkit.llm._providers._openai_compat import (
    _message_to_wire as openai_message_to_wire,
)
from ai_arch_toolkit.llm._providers._openai_compat import (
    _parse_response as openai_parse_response,
)
from ai_arch_toolkit.llm._providers._openai_compat import (
    _tool_to_openai,
)


class AsyncBatchClient:
    """Async batch client for OpenAI and Anthropic batch APIs."""

    def __init__(
        self,
        provider: str,
        *,
        model: str,
        api_key: str,
        retry: RetryConfig | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._retry = retry

        if provider == "openai":
            self._headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        elif provider == "anthropic":
            self._headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
        else:
            raise ValueError(f"Batch API not supported for provider {provider!r}")

    async def submit(self, batch_requests: list[BatchRequest]) -> BatchJob:
        """Submit a batch of requests."""
        if self._provider == "openai":
            return await self._submit_openai(batch_requests)
        return await self._submit_anthropic(batch_requests)

    async def status(self, job: BatchJob) -> BatchJob:
        """Check the status of a batch job."""
        if self._provider == "openai":
            return await self._status_openai(job)
        return await self._status_anthropic(job)

    async def results(self, job: BatchJob) -> list[BatchResult]:
        """Retrieve results of a completed batch job."""
        if self._provider == "openai":
            return await self._results_openai(job)
        return await self._results_anthropic(job)

    # --- OpenAI ---

    async def _submit_openai(self, batch_requests: list[BatchRequest]) -> BatchJob:
        lines: list[str] = []
        for req in batch_requests:
            msgs: list[dict[str, Any]] = []
            if req.system:
                msgs.append({"role": "system", "content": req.system})
            msgs.extend(openai_message_to_wire(m) for m in req.messages)
            body: dict[str, Any] = {"model": self._model, "messages": msgs}
            if req.tools:
                body["tools"] = [_tool_to_openai(t) for t in req.tools]
            if req.json_schema:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": req.json_schema.name,
                        "schema": req.json_schema.schema,
                        "strict": req.json_schema.strict,
                    },
                }
            line = {
                "custom_id": req.custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            lines.append(json.dumps(line))
        jsonl_content = "\n".join(lines)

        async with httpx.AsyncClient() as client:
            upload_resp = await client.post(
                "https://api.openai.com/v1/files",
                headers={"Authorization": f"Bearer {self._api_key}"},
                files={"file": ("batch.jsonl", jsonl_content.encode(), "application/jsonl")},
                data={"purpose": "batch"},
                timeout=60,
            )
            upload_resp.raise_for_status()
            file_id = upload_resp.json()["id"]

        raw = await async_post_json(
            "https://api.openai.com/v1/batches",
            self._headers,
            {
                "input_file_id": file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
            timeout=60,
            retry=self._retry,
        )
        return BatchJob(
            id=raw["id"],
            status=raw.get("status", "in_progress"),
            provider="openai",
            raw=raw,
        )

    async def _status_openai(self, job: BatchJob) -> BatchJob:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://api.openai.com/v1/batches/{job.id}",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=60,
            )
            resp.raise_for_status()
            raw = resp.json()
        return BatchJob(id=job.id, status=raw.get("status", ""), provider="openai", raw=raw)

    async def _results_openai(self, job: BatchJob) -> list[BatchResult]:
        output_file_id = job.raw.get("output_file_id", "")
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://api.openai.com/v1/files/{output_file_id}/content",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=60,
            )
            resp.raise_for_status()
        results: list[BatchResult] = []
        for line in resp.text.strip().split("\n"):
            if not line:
                continue
            row = json.loads(line)
            custom_id = row.get("custom_id", "")
            response_body = row.get("response", {}).get("body", {})
            error = row.get("error")
            if error:
                results.append(BatchResult(custom_id=custom_id, error=json.dumps(error)))
            else:
                results.append(
                    BatchResult(custom_id=custom_id, response=openai_parse_response(response_body))
                )
        return results

    # --- Anthropic ---

    async def _submit_anthropic(self, batch_requests: list[BatchRequest]) -> BatchJob:
        requests_payload: list[dict[str, Any]] = []
        for req in batch_requests:
            body: dict[str, Any] = {
                "model": self._model,
                "messages": anthropic_items_to_wire(req.messages),
                "max_tokens": 4096,
            }
            if req.system:
                body["system"] = req.system
            if req.tools:
                body["tools"] = [_tool_to_anthropic(t) for t in req.tools]
            requests_payload.append(
                {
                    "custom_id": req.custom_id,
                    "params": body,
                }
            )

        raw = await async_post_json(
            "https://api.anthropic.com/v1/messages/batches",
            self._headers,
            {"requests": requests_payload},
            timeout=60,
            retry=self._retry,
        )
        return BatchJob(
            id=raw.get("id", ""),
            status=raw.get("processing_status", "in_progress"),
            provider="anthropic",
            raw=raw,
        )

    async def _status_anthropic(self, job: BatchJob) -> BatchJob:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://api.anthropic.com/v1/messages/batches/{job.id}",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                },
                timeout=60,
            )
            resp.raise_for_status()
            raw = resp.json()
        return BatchJob(
            id=job.id,
            status=raw.get("processing_status", ""),
            provider="anthropic",
            raw=raw,
        )

    async def _results_anthropic(self, job: BatchJob) -> list[BatchResult]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://api.anthropic.com/v1/messages/batches/{job.id}/results",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                },
                timeout=60,
            )
            resp.raise_for_status()
        results: list[BatchResult] = []
        for line in resp.text.strip().split("\n"):
            if not line:
                continue
            row = json.loads(line)
            custom_id = row.get("custom_id", "")
            result = row.get("result", {})
            if result.get("type") == "succeeded":
                results.append(
                    BatchResult(
                        custom_id=custom_id,
                        response=anthropic_parse_response(result.get("message", {})),
                    )
                )
            else:
                results.append(
                    BatchResult(
                        custom_id=custom_id,
                        error=result.get("error", {}).get("message", "Unknown error"),
                    )
                )
        return results
