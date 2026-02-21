"""Tests for guardrail middleware."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_arch_toolkit.llm import AsyncClient, Client
from ai_arch_toolkit.llm._guardrails import GuardrailMiddleware, GuardrailViolation
from ai_arch_toolkit.llm._types import Response, Usage


@patch("ai_arch_toolkit.llm._client.create_provider")
def test_guardrail_blocks_input_pattern(mock_create: MagicMock) -> None:
    mock_create.return_value = MagicMock()
    middleware = GuardrailMiddleware(blocked_patterns=["password"])
    client = Client("openai", model="gpt-4o", api_key="sk-test", middleware=[middleware])

    with pytest.raises(GuardrailViolation, match="input"):
        client.chat("my password is 123")


@patch("ai_arch_toolkit.llm._client.create_provider")
def test_guardrail_blocks_output_pattern(mock_create: MagicMock) -> None:
    provider = MagicMock()
    provider.complete.return_value = Response(text="sensitive secret", usage=Usage())
    mock_create.return_value = provider
    middleware = GuardrailMiddleware(blocked_patterns=["secret"])
    client = Client("openai", model="gpt-4o", api_key="sk-test", middleware=[middleware])

    with pytest.raises(GuardrailViolation, match="output"):
        client.chat("hello")


@patch("ai_arch_toolkit.llm._client.create_provider")
def test_guardrail_runs_custom_validators(mock_create: MagicMock) -> None:
    provider = MagicMock()
    provider.complete.return_value = Response(text="ok", usage=Usage())
    mock_create.return_value = provider

    def _input_validator(request):
        request.kwargs["validated"] = True

    def _output_validator(response):
        if response.text != "ok":
            raise GuardrailViolation("bad output")

    middleware = GuardrailMiddleware(
        input_validator=_input_validator, output_validator=_output_validator
    )
    client = Client("openai", model="gpt-4o", api_key="sk-test", middleware=[middleware])

    _ = client.chat("hello")
    call_kwargs = provider.complete.call_args[1]
    assert call_kwargs["validated"] is True


@patch("ai_arch_toolkit.llm._async_client.create_provider")
@pytest.mark.asyncio
async def test_guardrail_async_blocks_output_pattern(mock_create: MagicMock) -> None:
    provider = MagicMock()
    provider.acomplete = AsyncMock(return_value=Response(text="top secret", usage=Usage()))
    mock_create.return_value = provider
    middleware = GuardrailMiddleware(blocked_patterns=["secret"])
    client = AsyncClient("openai", model="gpt-4o", api_key="sk-test", middleware=[middleware])

    with pytest.raises(GuardrailViolation, match="output"):
        await client.chat("hello")

