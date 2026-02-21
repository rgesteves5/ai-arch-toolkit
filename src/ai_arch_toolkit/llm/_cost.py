"""Cost tracking middleware and usage/cost preview helpers."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from ai_arch_toolkit.llm._middleware import Request
from ai_arch_toolkit.llm._tokens import (
    RawTokenCounter,
    TokenCorrectionConfig,
    estimate_conversation_tokens_for_model,
    estimate_text_tokens_for_model,
    get_correction_factor,
    raw_tiktoken_count,
)
from ai_arch_toolkit.llm._types import ConversationItem, Response, StreamEvent, Usage

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Pricing for a model in USD per 1M tokens."""

    input_per_million: float
    output_per_million: float
    cache_creation_per_million: float = 0.0
    cache_read_per_million: float = 0.0


DEFAULT_MODEL_PRICING: dict[str, ModelPricing] = {
    "claude-3-5-sonnet-latest": ModelPricing(input_per_million=3.0, output_per_million=15.0),
    "gemini-1.5-flash": ModelPricing(input_per_million=0.075, output_per_million=0.3),
    "gemini-1.5-pro": ModelPricing(input_per_million=1.25, output_per_million=5.0),
    "gpt-4o": ModelPricing(input_per_million=5.0, output_per_million=15.0),
    "gpt-4o-mini": ModelPricing(input_per_million=0.15, output_per_million=0.6),
}


@dataclass(frozen=True, slots=True)
class CostSnapshot:
    """A point-in-time summary of tracked usage and estimated cost."""

    total_cost_usd: float
    request_count: int
    total_usage: Usage
    per_model_cost_usd: dict[str, float] = field(default_factory=dict)
    per_model_usage: dict[str, Usage] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CostPreview:
    """Preview of estimated usage and cost for a prospective model call."""

    model: str
    provider: str | None
    usage: Usage
    estimated_cost_usd: float
    correction_factor: float
    raw_input_tokens: int | None = None
    pricing: ModelPricing | None = None


def _accumulate_usage(total: Usage, delta: Usage) -> Usage:
    return Usage(
        input_tokens=total.input_tokens + delta.input_tokens,
        output_tokens=total.output_tokens + delta.output_tokens,
        total_tokens=total.total_tokens + delta.total_tokens,
        cache_creation_tokens=total.cache_creation_tokens + delta.cache_creation_tokens,
        cache_read_tokens=total.cache_read_tokens + delta.cache_read_tokens,
    )


def _estimate_cost(pricing: ModelPricing, usage: Usage) -> float:
    return (
        usage.input_tokens / 1_000_000 * pricing.input_per_million
        + usage.output_tokens / 1_000_000 * pricing.output_per_million
        + usage.cache_creation_tokens / 1_000_000 * pricing.cache_creation_per_million
        + usage.cache_read_tokens / 1_000_000 * pricing.cache_read_per_million
    )


def resolve_model_pricing(
    model: str,
    *,
    provider: str | None = None,
    pricing: Mapping[str, ModelPricing] | None = None,
) -> ModelPricing | None:
    """Resolve pricing by ``provider:model`` first, then plain ``model``.

    ``pricing`` overrides ``DEFAULT_MODEL_PRICING`` when keys overlap.
    """
    merged = dict(DEFAULT_MODEL_PRICING)
    if pricing is not None:
        merged.update(pricing)

    if provider:
        full_name = f"{provider}:{model}"
        if full_name in merged:
            return merged[full_name]
    return merged.get(model)


def estimate_usage_cost(
    usage: Usage,
    *,
    model: str,
    provider: str | None = None,
    pricing: Mapping[str, ModelPricing] | None = None,
) -> float:
    """Estimate USD cost from usage with optional pricing overrides."""
    resolved = resolve_model_pricing(model, provider=provider, pricing=pricing)
    if resolved is None:
        return 0.0
    return _estimate_cost(resolved, usage)


def preview_text_usage_and_cost(
    text: str,
    *,
    model: str,
    provider: str | None = None,
    expected_output_tokens: int = 0,
    pricing: Mapping[str, ModelPricing] | None = None,
    correction_config: TokenCorrectionConfig | None = None,
    raw_token_counter: RawTokenCounter | None = None,
) -> CostPreview:
    """Preview input/output usage and cost for a text prompt.

    Input tokens are estimated from model-aware token counting with correction
    factors. Output tokens are caller-provided via ``expected_output_tokens``.
    """
    counter = raw_token_counter or raw_tiktoken_count
    raw_input_tokens = counter(text, model) if text else 0
    input_tokens = estimate_text_tokens_for_model(
        text,
        model,
        correction_config=correction_config,
        raw_token_counter=counter,
    )
    usage = Usage(
        input_tokens=input_tokens,
        output_tokens=max(0, expected_output_tokens),
        total_tokens=input_tokens + max(0, expected_output_tokens),
    )
    resolved_pricing = resolve_model_pricing(model, provider=provider, pricing=pricing)
    estimated_cost_usd = _estimate_cost(resolved_pricing, usage) if resolved_pricing else 0.0
    correction_factor = get_correction_factor(model, config=correction_config)
    return CostPreview(
        model=model,
        provider=provider,
        usage=usage,
        estimated_cost_usd=estimated_cost_usd,
        correction_factor=correction_factor,
        raw_input_tokens=raw_input_tokens,
        pricing=resolved_pricing,
    )


def preview_conversation_usage_and_cost(
    items: list[ConversationItem],
    *,
    model: str,
    provider: str | None = None,
    expected_output_tokens: int = 0,
    pricing: Mapping[str, ModelPricing] | None = None,
    correction_config: TokenCorrectionConfig | None = None,
    raw_token_counter: RawTokenCounter | None = None,
) -> CostPreview:
    """Preview usage and cost for a full conversation payload."""
    input_tokens = estimate_conversation_tokens_for_model(
        items,
        model,
        correction_config=correction_config,
        raw_token_counter=raw_token_counter,
    )
    usage = Usage(
        input_tokens=input_tokens,
        output_tokens=max(0, expected_output_tokens),
        total_tokens=input_tokens + max(0, expected_output_tokens),
    )
    resolved_pricing = resolve_model_pricing(model, provider=provider, pricing=pricing)
    estimated_cost_usd = _estimate_cost(resolved_pricing, usage) if resolved_pricing else 0.0
    correction_factor = get_correction_factor(model, config=correction_config)
    return CostPreview(
        model=model,
        provider=provider,
        usage=usage,
        estimated_cost_usd=estimated_cost_usd,
        correction_factor=correction_factor,
        raw_input_tokens=None,
        pricing=resolved_pricing,
    )


def preview_text_usage_and_cost_for_models(
    text: str,
    *,
    models: Sequence[str],
    provider: str | None = None,
    providers: Mapping[str, str] | None = None,
    expected_output_tokens: int = 0,
    pricing: Mapping[str, ModelPricing] | None = None,
    correction_config: TokenCorrectionConfig | None = None,
    raw_token_counter: RawTokenCounter | None = None,
) -> list[CostPreview]:
    """Preview text usage/cost for multiple models.

    ``provider`` is the default provider for all models. Use ``providers`` to
    override provider per model key.
    """
    previews: list[CostPreview] = []
    for model in models:
        resolved_provider = providers.get(model, provider) if providers else provider
        previews.append(
            preview_text_usage_and_cost(
                text,
                model=model,
                provider=resolved_provider,
                expected_output_tokens=expected_output_tokens,
                pricing=pricing,
                correction_config=correction_config,
                raw_token_counter=raw_token_counter,
            )
        )
    return previews


def preview_conversation_usage_and_cost_for_models(
    items: list[ConversationItem],
    *,
    models: Sequence[str],
    provider: str | None = None,
    providers: Mapping[str, str] | None = None,
    expected_output_tokens: int = 0,
    pricing: Mapping[str, ModelPricing] | None = None,
    correction_config: TokenCorrectionConfig | None = None,
    raw_token_counter: RawTokenCounter | None = None,
) -> list[CostPreview]:
    """Preview conversation usage/cost for multiple models."""
    previews: list[CostPreview] = []
    for model in models:
        resolved_provider = providers.get(model, provider) if providers else provider
        previews.append(
            preview_conversation_usage_and_cost(
                items,
                model=model,
                provider=resolved_provider,
                expected_output_tokens=expected_output_tokens,
                pricing=pricing,
                correction_config=correction_config,
                raw_token_counter=raw_token_counter,
            )
        )
    return previews


class CostTracker:
    """Middleware that accumulates usage and estimates USD cost per model."""

    def __init__(self, pricing: Mapping[str, ModelPricing] | None = None) -> None:
        self._pricing = dict(DEFAULT_MODEL_PRICING)
        if pricing is not None:
            self._pricing.update(pricing)
        self._total_usage = Usage()
        self._per_model_usage: dict[str, Usage] = {}
        self._total_cost_usd = 0.0
        self._per_model_cost_usd: dict[str, float] = {}
        self._request_count = 0
        self._warned_missing_prices: set[str] = set()
        self._lock = Lock()

    def before(self, request: Request) -> Request:
        return request

    def after(self, request: Request, result: Any) -> Any:
        if isinstance(result, Response):
            self._record_usage(request, result.usage)
            return result
        if request.operation == "stream_events" and isinstance(result, Iterator):
            return self._wrap_stream_events(request, result)
        return result

    async def abefore(self, request: Request) -> Request:
        return self.before(request)

    async def aafter(self, request: Request, result: Any) -> Any:
        if isinstance(result, Response):
            self._record_usage(request, result.usage)
            return result
        if request.operation == "stream_events" and isinstance(result, AsyncIterator):
            return self._awrap_stream_events(request, result)
        return result

    def snapshot(self) -> CostSnapshot:
        with self._lock:
            return CostSnapshot(
                total_cost_usd=self._total_cost_usd,
                request_count=self._request_count,
                total_usage=self._total_usage,
                per_model_cost_usd=dict(self._per_model_cost_usd),
                per_model_usage=dict(self._per_model_usage),
            )

    def update_pricing(self, pricing: Mapping[str, ModelPricing]) -> None:
        """Update per-model pricing overrides at runtime."""
        with self._lock:
            self._pricing.update(pricing)

    def _resolve_pricing(self, provider: str, model: str) -> ModelPricing | None:
        full_name = f"{provider}:{model}"
        return self._pricing.get(full_name) or self._pricing.get(model)

    def _record_usage(self, request: Request, usage: Usage) -> None:
        model_key = f"{request.provider}:{request.model}"
        pricing = self._resolve_pricing(request.provider, request.model)
        estimated_cost = 0.0
        if pricing is None and model_key not in self._warned_missing_prices:
            logger.warning(
                "No pricing configured for provider=%s model=%s; cost remains zero",
                request.provider,
                request.model,
            )
            self._warned_missing_prices.add(model_key)
        elif pricing is not None:
            estimated_cost = _estimate_cost(pricing, usage)

        with self._lock:
            self._request_count += 1
            self._total_usage = _accumulate_usage(self._total_usage, usage)
            model_usage = self._per_model_usage.get(model_key, Usage())
            self._per_model_usage[model_key] = _accumulate_usage(model_usage, usage)
            self._total_cost_usd += estimated_cost
            self._per_model_cost_usd[model_key] = (
                self._per_model_cost_usd.get(model_key, 0.0) + estimated_cost
            )

    def _wrap_stream_events(
        self, request: Request, stream: Iterator[StreamEvent]
    ) -> Iterator[StreamEvent]:
        for event in stream:
            if event.type == "usage" and event.usage is not None:
                self._record_usage(request, event.usage)
            yield event

    async def _awrap_stream_events(
        self, request: Request, stream: AsyncIterator[StreamEvent]
    ) -> AsyncIterator[StreamEvent]:
        async for event in stream:
            if event.type == "usage" and event.usage is not None:
                self._record_usage(request, event.usage)
            yield event
