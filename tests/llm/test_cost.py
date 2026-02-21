"""Tests for cost tracking middleware."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_arch_toolkit.llm import AsyncClient, Client
from ai_arch_toolkit.llm._cost import (
    CostTracker,
    ModelPricing,
    estimate_usage_cost,
    preview_conversation_usage_and_cost_for_models,
    preview_text_usage_and_cost,
    preview_text_usage_and_cost_for_models,
    resolve_model_pricing,
)
from ai_arch_toolkit.llm._tokens import CLAUDE_3_CORRECTION_FACTOR, TokenCorrectionConfig
from ai_arch_toolkit.llm._types import Message, Response, StreamEvent, Usage


@patch("ai_arch_toolkit.llm._client.create_provider")
def test_cost_tracker_accumulates_chat_usage_and_cost(mock_create: MagicMock) -> None:
    provider = MagicMock()
    provider.complete.return_value = Response(
        text="ok",
        usage=Usage(input_tokens=1000, output_tokens=2000, total_tokens=3000),
    )
    mock_create.return_value = provider
    tracker = CostTracker(pricing={"openai:gpt-4o": ModelPricing(1.0, 2.0)})
    client = Client("openai", model="gpt-4o", api_key="sk-test", middleware=[tracker])

    _ = client.chat("hello")

    snapshot = tracker.snapshot()
    assert snapshot.request_count == 1
    assert snapshot.total_usage.input_tokens == 1000
    assert snapshot.total_usage.output_tokens == 2000
    assert snapshot.per_model_usage["openai:gpt-4o"].total_tokens == 3000
    assert snapshot.total_cost_usd == pytest.approx(0.005)
    assert snapshot.per_model_cost_usd["openai:gpt-4o"] == pytest.approx(0.005)


@patch("ai_arch_toolkit.llm._client.create_provider")
def test_cost_tracker_accumulates_usage_from_stream_events(mock_create: MagicMock) -> None:
    provider = MagicMock()
    provider.stream_events.return_value = iter(
        [
            StreamEvent(type="text", text="hi"),
            StreamEvent(
                type="usage",
                usage=Usage(input_tokens=300, output_tokens=700, total_tokens=1000),
            ),
            StreamEvent(type="done"),
        ]
    )
    mock_create.return_value = provider
    tracker = CostTracker(pricing={"openai:gpt-4o-mini": ModelPricing(0.1, 0.2)})
    client = Client("openai", model="gpt-4o-mini", api_key="sk-test", middleware=[tracker])

    _ = list(client.stream_events("hello"))

    snapshot = tracker.snapshot()
    assert snapshot.request_count == 1
    assert snapshot.total_usage.total_tokens == 1000
    assert snapshot.total_cost_usd == pytest.approx(0.00017)


@patch("ai_arch_toolkit.llm._client.create_provider")
def test_cost_tracker_warns_once_for_unknown_pricing(
    mock_create: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    provider = MagicMock()
    provider.complete.return_value = Response(text="ok", usage=Usage(total_tokens=10))
    mock_create.return_value = provider
    tracker = CostTracker(pricing={})
    client = Client("openai", model="unknown-model", api_key="sk-test", middleware=[tracker])

    with caplog.at_level("WARNING", logger="ai_arch_toolkit"):
        _ = client.chat("a")
        _ = client.chat("b")

    assert caplog.text.count("No pricing configured for provider=openai model=unknown-model") == 1


@patch("ai_arch_toolkit.llm._async_client.create_provider")
@pytest.mark.asyncio
async def test_async_cost_tracker_accumulates_chat_usage_and_cost(mock_create: MagicMock) -> None:
    provider = MagicMock()
    provider.acomplete = AsyncMock(
        return_value=Response(
            text="ok",
            usage=Usage(input_tokens=2000, output_tokens=1000, total_tokens=3000),
        )
    )
    mock_create.return_value = provider
    tracker = CostTracker(pricing={"openai:gpt-4o": ModelPricing(2.0, 1.0)})
    client = AsyncClient("openai", model="gpt-4o", api_key="sk-test", middleware=[tracker])

    _ = await client.chat("hello")

    snapshot = tracker.snapshot()
    assert snapshot.request_count == 1
    assert snapshot.total_usage.total_tokens == 3000
    assert snapshot.total_cost_usd == pytest.approx(0.005)


@patch("ai_arch_toolkit.llm._async_client.create_provider")
@pytest.mark.asyncio
async def test_async_cost_tracker_accumulates_usage_from_stream_events(
    mock_create: MagicMock,
) -> None:
    provider = MagicMock()

    async def _astream_events(*args, **kwargs):
        yield StreamEvent(type="text", text="hi")
        yield StreamEvent(
            type="usage",
            usage=Usage(input_tokens=100, output_tokens=400, total_tokens=500),
        )
        yield StreamEvent(type="done")

    provider.astream_events = MagicMock(side_effect=_astream_events)
    mock_create.return_value = provider
    tracker = CostTracker(pricing={"openai:gpt-4o-mini": ModelPricing(0.2, 0.3)})
    client = AsyncClient("openai", model="gpt-4o-mini", api_key="sk-test", middleware=[tracker])

    events = []
    async for event in client.stream_events("hello"):
        events.append(event)

    assert len(events) == 3
    snapshot = tracker.snapshot()
    assert snapshot.request_count == 1
    assert snapshot.total_usage.total_tokens == 500
    assert snapshot.total_cost_usd == pytest.approx(0.00014)


def test_resolve_model_pricing_prefers_provider_specific_key() -> None:
    pricing = {
        "gpt-4o": ModelPricing(input_per_million=10.0, output_per_million=20.0),
        "openai:gpt-4o": ModelPricing(input_per_million=1.0, output_per_million=2.0),
    }
    resolved = resolve_model_pricing("gpt-4o", provider="openai", pricing=pricing)
    assert resolved == ModelPricing(input_per_million=1.0, output_per_million=2.0)


def test_estimate_usage_cost_returns_zero_when_pricing_missing() -> None:
    usage = Usage(input_tokens=100, output_tokens=50, total_tokens=150)
    assert estimate_usage_cost(usage, model="unknown-model", pricing={}) == 0.0


def test_preview_text_usage_and_cost_applies_correction_and_pricing() -> None:
    preview = preview_text_usage_and_cost(
        "hello world",
        model="claude-3-5-sonnet-latest",
        provider="anthropic",
        expected_output_tokens=50,
        pricing={
            "anthropic:claude-3-5-sonnet-latest": ModelPricing(
                input_per_million=1.0,
                output_per_million=2.0,
            )
        },
        raw_token_counter=lambda _text, _model: 100,
    )
    assert preview.usage.input_tokens == int(100 * CLAUDE_3_CORRECTION_FACTOR)
    assert preview.usage.output_tokens == 50
    assert preview.raw_input_tokens == 100
    assert preview.correction_factor == CLAUDE_3_CORRECTION_FACTOR
    assert preview.estimated_cost_usd == pytest.approx(0.000212)


def test_preview_text_usage_and_cost_for_models_supports_overrides() -> None:
    previews = preview_text_usage_and_cost_for_models(
        "hello",
        models=["gpt-4o", "gpt-4o-mini"],
        provider="openai",
        expected_output_tokens=10,
        pricing={
            "openai:gpt-4o": ModelPricing(input_per_million=1.0, output_per_million=1.0),
            "openai:gpt-4o-mini": ModelPricing(
                input_per_million=2.0,
                output_per_million=2.0,
            ),
        },
        correction_config=TokenCorrectionConfig(model_overrides={"gpt-4o-mini": 2.0}),
        raw_token_counter=lambda _text, _model: 10,
    )
    assert len(previews) == 2
    assert previews[0].model == "gpt-4o"
    assert previews[0].provider == "openai"
    assert previews[0].usage.input_tokens == 10
    assert previews[1].model == "gpt-4o-mini"
    assert previews[1].usage.input_tokens == 20
    assert previews[1].estimated_cost_usd == pytest.approx(0.00006)


def test_preview_conversation_usage_and_cost_for_models_supports_provider_map() -> None:
    previews = preview_conversation_usage_and_cost_for_models(
        [Message(role="user", content="hello")],
        models=["claude-3-5-sonnet-latest", "gpt-4o"],
        provider="openai",
        providers={"claude-3-5-sonnet-latest": "anthropic"},
        expected_output_tokens=5,
        pricing={
            "anthropic:claude-3-5-sonnet-latest": ModelPricing(
                input_per_million=1.0,
                output_per_million=1.0,
            ),
            "openai:gpt-4o": ModelPricing(input_per_million=1.0, output_per_million=1.0),
        },
        raw_token_counter=lambda _text, _model: 10,
    )
    assert len(previews) == 2
    assert previews[0].provider == "anthropic"
    assert previews[1].provider == "openai"
    assert previews[0].usage.total_tokens > previews[1].usage.total_tokens
