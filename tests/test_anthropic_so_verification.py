"""Verification test: Anthropic provider uses native output_config for structured output."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from ai_arch_toolkit.core._providers._anthropic import (
    AnthropicProvider,
    _build_output_config,
)
from ai_arch_toolkit.core._response import OutputSchema

# ---------------------------------------------------------------------------
# 1. _build_output_config produces correct structure
# ---------------------------------------------------------------------------


def test_build_output_config_structure():
    schema = OutputSchema(name="MyResponse", schema={"type": "object", "properties": {}})
    config = _build_output_config(schema)

    assert "format" in config
    assert config["format"]["type"] == "json_schema"
    assert config["format"]["schema"] == schema.schema


# ---------------------------------------------------------------------------
# 2. output_schema flows through to output_config in SDK kwargs
# ---------------------------------------------------------------------------


async def test_output_schema_flows_to_output_config():
    schema = OutputSchema(
        name="Result",
        schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    )

    # Build a fake SDK response
    fake_message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text='{"answer": "42"}', citations=None)],
        model="claude-sonnet-4-20250514",
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )

    with patch("ai_arch_toolkit.core._providers._anthropic.anthropic") as mock_anthropic:
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=fake_message)
        mock_anthropic.AsyncAnthropic.return_value = mock_client

        provider = AnthropicProvider(model="claude-sonnet-4-20250514", api_key="test-key")
        await provider.complete(
            [{"role": "user", "content": "What is 6*7?"}],
            output_schema=schema,
        )

        # Verify the SDK call included output_config
        sdk_call = mock_client.messages.create.call_args
        assert "output_config" in sdk_call.kwargs
        output_config = sdk_call.kwargs["output_config"]
        assert output_config["format"]["type"] == "json_schema"
        assert output_config["format"]["schema"] == schema.schema


# ---------------------------------------------------------------------------
# 3. Verify no tool trick — output_schema does NOT create a tool
# ---------------------------------------------------------------------------


async def test_output_schema_does_not_create_tool():
    schema = OutputSchema(
        name="Result",
        schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    )

    fake_message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text='{"answer": "42"}', citations=None)],
        model="claude-sonnet-4-20250514",
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )

    with patch("ai_arch_toolkit.core._providers._anthropic.anthropic") as mock_anthropic:
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=fake_message)
        mock_anthropic.AsyncAnthropic.return_value = mock_client

        provider = AnthropicProvider(model="claude-sonnet-4-20250514", api_key="test-key")
        await provider.complete(
            [{"role": "user", "content": "Structured output test"}],
            output_schema=schema,
        )

        sdk_call = mock_client.messages.create.call_args
        # Should NOT have tools when only output_schema is used
        assert sdk_call.kwargs.get("tools") is None or sdk_call.kwargs.get("tools") == []
