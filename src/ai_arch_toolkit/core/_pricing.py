"""Pricing registry — estimates cost from token usage."""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_arch_toolkit.core._metering._cost import Cost
from ai_arch_toolkit.core._metering._money import Money

if TYPE_CHECKING:
    from ai_arch_toolkit.core._metering._operation import OperationRequest
    from ai_arch_toolkit.core._response import Usage

logger = logging.getLogger(__name__)

__all__ = ["ModelPricing", "PricingRegistry", "pricing"]

_MISS = object()  # sentinel: distinguishes "not cached" from a cached None (known-unpriced)


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelPricing:
    """USD per 1M tokens."""

    input: float = 0.0
    output: float = 0.0
    cache_write: float | None = None
    cache_read: float | None = None
    batch_input: float | None = None
    batch_output: float | None = None
    # Long context pricing (exceeding threshold triggers premium rates)
    long_context_threshold: int | None = None
    long_context_input: float | None = None
    long_context_output: float | None = None
    # Fast mode pricing
    fast_input: float | None = None
    fast_output: float | None = None


class PricingRegistry:
    """Registry of model pricing. Ships with defaults, fully overridable.

    Usage::

        from ai_arch_toolkit.core import ModelPricing, pricing

        cost = pricing.estimate_cost("claude-sonnet-5", input_tokens=1000)
        pricing.register("my-model", ModelPricing(input=1.0, output=2.0))
        pricing.load("./my_pricing.toml")
        pricing.reset()
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelPricing] = {}
        # Memoize longest-prefix lookups: get() is on the settle hot path (once per LLM attempt),
        # and a run reuses the same model string thousands of times. Cleared on any mutation.
        self._cache: dict[str, ModelPricing | None] = {}
        self._load_defaults()

    def _load_defaults(self) -> None:
        """Load the shipped default pricing TOML."""
        try:
            default_path = Path(__file__).parent / "_default_pricing.toml"
            if default_path.exists():
                self._load_toml(default_path)
        except (OSError, tomllib.TOMLDecodeError):
            logger.warning("Failed to load default pricing table", exc_info=True)

    # ── Registration ──

    def register(self, model_prefix: str, pricing: ModelPricing) -> None:
        """Register or override pricing for a model prefix."""
        self._models[model_prefix] = pricing
        self._cache.clear()

    def unregister(self, model_prefix: str) -> None:
        """Remove pricing for a model prefix."""
        self._models.pop(model_prefix, None)
        self._cache.clear()

    # ── Query ──

    def get(self, model: str) -> ModelPricing | None:
        """Find pricing by longest prefix match (memoized). Returns None if unknown."""
        cached = self._cache.get(model, _MISS)
        if cached is not _MISS:
            return cached  # type: ignore[return-value]  # _MISS excluded above
        best: ModelPricing | None = None
        best_len = 0
        for prefix, p in self._models.items():
            if model.startswith(prefix) and len(prefix) > best_len:
                best = p
                best_len = len(prefix)
        self._cache[model] = best
        return best

    def has(self, model: str) -> bool:
        """Check if a model has pricing registered."""
        return self.get(model) is not None

    def list_models(self) -> list[str]:
        """List all registered model prefixes."""
        return sorted(self._models.keys())

    # ── Cost Estimation ──

    def estimate_cost(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_write_tokens: int = 0,
        cache_read_tokens: int = 0,
        *,
        is_batch: bool = False,
        is_fast: bool = False,
    ) -> float | None:
        """Estimate cost in USD.

        Priority: ``is_fast`` > ``is_batch`` > long-context > standard.

        Returns:
            Cost in USD, or ``None`` if no pricing data exists for the model.
        """
        p = self.get(model)
        if p is None:
            return None

        per_m = 1_000_000
        total_input = input_tokens + cache_write_tokens + cache_read_tokens

        if is_fast and p.fast_input is not None:
            inp = p.fast_input
            out = p.fast_output if p.fast_output is not None else p.output
        elif is_batch:
            inp = p.batch_input if p.batch_input is not None else p.input
            out = p.batch_output if p.batch_output is not None else p.output
        elif (
            p.long_context_threshold is not None
            and total_input > p.long_context_threshold
            and p.long_context_input is not None
        ):
            inp = p.long_context_input
            out = p.long_context_output if p.long_context_output is not None else p.output
        else:
            inp = p.input
            out = p.output

        total = inp * input_tokens / per_m + out * output_tokens / per_m

        if cache_write_tokens > 0 and p.cache_write is not None:
            total += p.cache_write * cache_write_tokens / per_m
        if cache_read_tokens > 0 and p.cache_read is not None:
            total += p.cache_read * cache_read_tokens / per_m

        return total

    def price(self, request: OperationRequest, usage: Usage) -> Cost:
        """Turn an operation's facts + observed usage into a typed :class:`Cost`.

        The default :class:`~ai_arch_toolkit.core.Pricer`. A missing table entry yields
        :meth:`Cost.unknown` — never a silent ``$0`` — so an unpriced call fails closed.
        Provider-hosted server tools make the whole cost ``unknown`` because their charge is
        not reflected in the token counts.
        """
        if request.kind != "llm":
            return Cost.known(Money.zero())  # non-LLM ops carry no token cost here
        if request.has_server_tools:
            return Cost.unknown("provider-hosted server tools have unmetered cost")
        model = request.model
        if model is None:
            return Cost.unknown("operation has no model to price")
        usd = self.estimate_cost(
            model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            cache_read_tokens=usage.cache_read_tokens,
        )
        if usd is None:
            return Cost.unknown(f"no pricing for model {model!r}")
        return Cost.known(Money.from_usd(usd))

    # ── Load ──

    def load(self, path: str | Path) -> None:
        """Load pricing from a TOML file. Merges with existing — loaded values win."""
        self._load_toml(Path(path))

    def _load_toml(self, path: Path) -> None:
        """Parse a pricing TOML file and register all entries."""
        with open(path, "rb") as f:
            data: dict[str, Any] = tomllib.load(f)

        for prefix, values in data.items():
            if isinstance(values, dict):
                self._models[prefix] = ModelPricing(
                    input=values.get("input", 0.0),
                    output=values.get("output", 0.0),
                    cache_write=values.get("cache_write"),
                    cache_read=values.get("cache_read"),
                    batch_input=values.get("batch_input"),
                    batch_output=values.get("batch_output"),
                    long_context_threshold=values.get("long_context_threshold"),
                    long_context_input=values.get("long_context_input"),
                    long_context_output=values.get("long_context_output"),
                    fast_input=values.get("fast_input"),
                    fast_output=values.get("fast_output"),
                )
        self._cache.clear()

    def reset(self) -> None:
        """Reset to shipped defaults, discarding all custom registrations."""
        self._models.clear()
        self._cache.clear()
        self._load_defaults()


# ── Module-level singleton ──
pricing = PricingRegistry()


def _estimate_response_cost(model: str, usage: Any) -> float | None:
    """Estimate response cost from a ``Usage``-like object."""
    return pricing.estimate_cost(
        model,
        input_tokens=getattr(usage, "input_tokens", 0),
        output_tokens=getattr(usage, "output_tokens", 0),
        cache_write_tokens=getattr(usage, "cache_write_tokens", 0),
        cache_read_tokens=getattr(usage, "cache_read_tokens", 0),
    )


def estimate_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
    *,
    is_batch: bool = False,
    is_fast: bool = False,
) -> float | None:
    """Convenience wrapper around the global pricing registry."""
    return pricing.estimate_cost(
        model,
        input_tokens,
        output_tokens,
        cache_write_tokens,
        cache_read_tokens,
        is_batch=is_batch,
        is_fast=is_fast,
    )
