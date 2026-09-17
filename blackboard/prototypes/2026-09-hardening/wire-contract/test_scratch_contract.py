"""Scratch tests (outside the repo): ordinary-looking adapter tests, no contract code in them."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ai_arch_toolkit.core import LLM, document, user, web_search


def _anthropic_llm() -> tuple[LLM, AsyncMock]:
    llm = LLM("claude-sonnet-4-6", api_key="x")
    client = AsyncMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok", citations=None)],
        model="claude-sonnet-4-6",
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1, cache_creation_input_tokens=0, cache_read_input_tokens=0),
    )
    llm._provider._client = client
    return llm, client


async def test_plain_call_conforms():
    llm, client = _anthropic_llm()
    await llm.complete("Hi")
    assert client.messages.create.await_count == 1


async def test_web_search_is_sent():
    llm, client = _anthropic_llm()
    await llm.complete("Hi", tools=[web_search(max_uses=3)])
    assert client.messages.create.call_args.kwargs["tools"][0]["type"] == "web_search_20250305"


async def test_document_name_is_sent():
    llm, client = _anthropic_llm()
    await llm.complete([user(["Read", document(b"%PDF", name="spec.pdf")])])
    assert client.messages.create.await_count == 1


@pytest.mark.wire_contract(tolerate=[r"document\.name \| Extra inputs"])
async def test_document_name_deviation_documented():
    llm, client = _anthropic_llm()
    await llm.complete([user(["Read", document(b"%PDF", name="spec.pdf")])])
