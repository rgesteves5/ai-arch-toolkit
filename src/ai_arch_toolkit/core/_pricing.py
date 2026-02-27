"""Pricing registry — estimates cost from token usage."""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """USD per 1M tokens."""

    input: float = 0.0
    output: float = 0.0
    cache_write: float | None = None
    cache_read: float | None = None
    batch_input: float | None = None
    batch_output: float | None = None


class PricingRegistry:
    """Registry of model pricing. Ships with defaults, fully overridable.

    Usage::

        from ai_arch_toolkit.core._pricing import pricing

        cost, known = pricing.estimate_cost("claude-sonnet-4-20250514", input_tokens=1000)
        pricing.register("my-model", ModelPricing(input=1.0, output=2.0))
        pricing.load("./my_pricing.toml")
        pricing.reset()
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelPricing] = {}
        self._load_defaults()

    def _load_defaults(self) -> None:
        """Load the shipped default pricing TOML."""
        try:
            default_path = Path(__file__).parent / "_default_pricing.toml"
            if default_path.exists():
                self._load_toml(default_path)
        except Exception:
            logger.warning("Failed to load default pricing table", exc_info=True)

    # ── Registration ──

    def register(self, model_prefix: str, pricing: ModelPricing) -> None:
        """Register or override pricing for a model prefix."""
        self._models[model_prefix] = pricing

    def unregister(self, model_prefix: str) -> None:
        """Remove pricing for a model prefix."""
        self._models.pop(model_prefix, None)

    # ── Query ──

    def get(self, model: str) -> ModelPricing | None:
        """Find pricing by longest prefix match. Returns None if unknown."""
        best: ModelPricing | None = None
        best_len = 0
        for prefix, p in self._models.items():
            if model.startswith(prefix) and len(prefix) > best_len:
                best = p
                best_len = len(prefix)
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
    ) -> float | None:
        """Estimate cost in USD.

        Returns:
            Cost in USD, or ``None`` if no pricing data exists for the model.
        """
        p = self.get(model)
        if p is None:
            return None

        per_m = 1_000_000

        if is_batch:
            inp = p.batch_input if p.batch_input is not None else p.input
            out = p.batch_output if p.batch_output is not None else p.output
        else:
            inp = p.input
            out = p.output

        total = inp * input_tokens / per_m + out * output_tokens / per_m

        if cache_write_tokens > 0 and p.cache_write is not None:
            total += p.cache_write * cache_write_tokens / per_m
        if cache_read_tokens > 0 and p.cache_read is not None:
            total += p.cache_read * cache_read_tokens / per_m

        return total

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
                )

    def reset(self) -> None:
        """Reset to shipped defaults, discarding all custom registrations."""
        self._models.clear()
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
) -> float | None:
    """Convenience wrapper around the global pricing registry."""
    return pricing.estimate_cost(
        model,
        input_tokens,
        output_tokens,
        cache_write_tokens,
        cache_read_tokens,
        is_batch=is_batch,
    )
