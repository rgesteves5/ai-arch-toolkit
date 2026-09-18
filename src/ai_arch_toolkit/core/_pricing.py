"""Pricing registry — estimates cost from token usage."""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ai_arch_toolkit.core._metering._cost import Cost
from ai_arch_toolkit.core._metering._money import Money
from ai_arch_toolkit.core._model_id import lookup

if TYPE_CHECKING:
    from ai_arch_toolkit.core._metering._operation import OperationRequest
    from ai_arch_toolkit.core._response import Usage

logger = logging.getLogger(__name__)

__all__ = ["ModelPricing", "PricingRegistry", "pricing"]

type PriceMatch = Literal["exact", "prefix"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelPricing:
    """USD per 1M tokens."""

    input: float = 0.0
    output: float = 0.0
    cache_write: float | None = None
    cache_read: float | None = None
    batch_input: float | None = None
    batch_output: float | None = None
    batch_cache_write: float | None = None
    batch_cache_read: float | None = None
    # Long-context pricing (threshold boundary is provider-configurable)
    long_context_threshold: int | None = None
    long_context_inclusive: bool = False
    long_context_input: float | None = None
    long_context_output: float | None = None
    long_context_cache_write: float | None = None
    long_context_cache_read: float | None = None
    batch_long_context_input: float | None = None
    batch_long_context_output: float | None = None
    batch_long_context_cache_write: float | None = None
    batch_long_context_cache_read: float | None = None
    # Fast mode pricing
    fast_input: float | None = None
    fast_output: float | None = None
    fast_cache_write: float | None = None
    fast_cache_read: float | None = None
    fast_long_context_input: float | None = None
    fast_long_context_output: float | None = None
    fast_long_context_cache_write: float | None = None
    fast_long_context_cache_read: float | None = None


_PRICE_FIELDS = frozenset(field.name for field in fields(ModelPricing))


def _first_rate(*rates: float | None) -> float | None:
    """Return the first explicitly configured rate, preserving valid zeroes."""
    return next((rate for rate in rates if rate is not None), None)


class PricingRegistry:
    """Registry of model pricing. Ships with defaults, fully overridable.

    A model id finds its price by its own entry or as a dated snapshot of one
    (``claude-haiku-4-5-20251001`` → ``claude-haiku-4-5``); a variant such as ``o3-pro`` never
    inherits ``o3``. A prefix entry is an explicit choice, for a family of local models.

    Usage::

        from ai_arch_toolkit.core import ModelPricing, pricing

        cost = pricing.estimate_cost("claude-sonnet-5", input_tokens=1000)
        pricing.register("my-model", ModelPricing(input=1.0, output=2.0))
        pricing.register("llama3", ModelPricing(), match="prefix")  # local: zero, explicitly
        pricing.load("./my_pricing.toml")
        pricing.reset()
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelPricing] = {}  # exact ids and their aliases
        self._prefixes: dict[str, ModelPricing] = {}  # registered with match="prefix"
        # Memoize lookups: get() is on the settle hot path (once per LLM attempt), and a run
        # reuses the same model string thousands of times. Cleared on any mutation.
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

    def register(self, model: str, pricing: ModelPricing, *, match: PriceMatch = "exact") -> None:
        """Register or override the price of ``model`` and its dated snapshots.

        With ``match="prefix"`` the price covers every id that starts with ``model`` — meant for
        a family of local models. An id's own entry always wins over a prefix.
        """
        self._table(match)[model] = pricing
        self._cache.clear()

    def unregister(self, model: str) -> None:
        """Remove the entry for ``model``, exact or prefix."""
        self._models.pop(model, None)
        self._prefixes.pop(model, None)
        self._cache.clear()

    def _table(self, match: PriceMatch) -> dict[str, ModelPricing]:
        if match == "exact":
            return self._models
        if match == "prefix":
            return self._prefixes
        raise ValueError(f"match must be 'exact' or 'prefix', got {match!r}")

    # ── Query ──

    def get(self, model: str) -> ModelPricing | None:
        """The price of ``model`` (memoized), or ``None`` when it has none."""
        if model not in self._cache:
            found = lookup(model, self._models, self._prefixes)
            self._cache[model] = found.value if found is not None else None
        return self._cache[model]

    def has(self, model: str) -> bool:
        """Check if a model has pricing registered."""
        return self.get(model) is not None

    def list_models(self) -> list[str]:
        """Every registered id, alias and prefix, sorted."""
        return sorted({*self._models, *self._prefixes})

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

        Priority: ``is_fast`` > ``is_batch`` > standard. Long-context variants
        combine with the selected mode when the corresponding rates are configured.

        Returns:
            Cost in USD, or ``None`` if no pricing data exists for the model.
        """
        p = self.get(model)
        if p is None:
            return None

        per_m = 1_000_000
        total_input = input_tokens + cache_write_tokens + cache_read_tokens

        is_long = p.long_context_threshold is not None and (
            total_input >= p.long_context_threshold
            if p.long_context_inclusive
            else total_input > p.long_context_threshold
        )

        if is_fast and p.fast_input is not None:
            inp = _first_rate(
                p.fast_long_context_input if is_long else None,
                p.fast_input,
                p.input,
            )
            out = _first_rate(
                p.fast_long_context_output if is_long else None,
                p.fast_output,
                p.output,
            )
            cache_write = _first_rate(
                p.fast_long_context_cache_write if is_long else None,
                p.fast_cache_write,
                p.cache_write,
            )
            cache_read = _first_rate(
                p.fast_long_context_cache_read if is_long else None,
                p.fast_cache_read,
                p.cache_read,
            )
        elif is_batch and p.batch_input is not None:
            inp = _first_rate(
                p.batch_long_context_input if is_long else None,
                p.batch_input,
                p.input,
            )
            out = _first_rate(
                p.batch_long_context_output if is_long else None,
                p.batch_output,
                p.output,
            )
            cache_write = _first_rate(
                p.batch_long_context_cache_write if is_long else None,
                p.batch_cache_write,
                p.cache_write,
            )
            cache_read = _first_rate(
                p.batch_long_context_cache_read if is_long else None,
                p.batch_cache_read,
                p.cache_read,
            )
        elif is_long and p.long_context_input is not None:
            inp = p.long_context_input
            out = _first_rate(p.long_context_output, p.output)
            cache_write = _first_rate(p.long_context_cache_write, p.cache_write)
            cache_read = _first_rate(p.long_context_cache_read, p.cache_read)
        else:
            inp = p.input
            out = p.output
            cache_write = p.cache_write
            cache_read = p.cache_read

        # Every input/output branch above has a concrete standard-rate fallback.
        assert inp is not None
        assert out is not None

        total = inp * input_tokens / per_m + out * output_tokens / per_m

        if cache_write_tokens > 0 and cache_write is not None:
            total += cache_write * cache_write_tokens / per_m
        if cache_read_tokens > 0 and cache_read is not None:
            total += cache_read * cache_read_tokens / per_m

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
        """Parse a pricing TOML file and register all entries.

        An entry may list ``aliases`` (other ids with the same price) and ``match = "prefix"``.
        """
        with open(path, "rb") as f:
            data: dict[str, Any] = tomllib.load(f)

        for model, values in data.items():
            if isinstance(values, dict):
                price = ModelPricing(**{k: v for k, v in values.items() if k in _PRICE_FIELDS})
                table = self._table(values.get("match", "exact"))
                for name in (model, *values.get("aliases", ())):
                    table[name] = price
        self._cache.clear()

    def reset(self) -> None:
        """Reset to shipped defaults, discarding all custom registrations."""
        self._models.clear()
        self._prefixes.clear()
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
