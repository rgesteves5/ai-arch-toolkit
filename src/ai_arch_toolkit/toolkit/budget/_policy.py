"""``BudgetPolicy`` — user-facing budget caps that compile to neutral core ``ResourceLimits``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ai_arch_toolkit.core._metering._admission import ResourceLimits
from ai_arch_toolkit.core._metering._money import Money

__all__ = ["BudgetPolicy", "Reserve"]

type Reserve = Literal["none", "strict"]
"""Reservation mode. ``none``: measure + settle only (soft, the default). ``strict``: reserve a
worst-case token/cost hold per operation before it runs, failing closed on unpriced models."""


@dataclass(frozen=True, slots=True, kw_only=True)
class BudgetPolicy:
    """Cumulative run-level caps. ``max_cost`` is in USD; the rest are absolute counts/seconds.

    A bare ``BudgetPolicy()`` caps nothing (equivalent to measure-only). Compiles to a core
    :class:`ResourceLimits` (the hard, race-safe caps) via :meth:`to_limits`.
    """

    max_wall_s: float | None = None
    max_llm_calls: int | None = None
    max_tool_calls: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    max_cost: float | None = None
    reserve: Reserve = "none"

    def __post_init__(self) -> None:
        for name in (
            "max_llm_calls",
            "max_tool_calls",
            "max_input_tokens",
            "max_output_tokens",
            "max_total_tokens",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.max_cost is not None and self.max_cost < 0:
            raise ValueError(f"max_cost must be >= 0, got {self.max_cost}")
        if self.max_wall_s is not None and self.max_wall_s <= 0:
            raise ValueError(f"max_wall_s must be > 0, got {self.max_wall_s}")

    @property
    def is_empty(self) -> bool:
        """True when no cap is set — enforcement is a no-op."""
        return all(
            v is None
            for v in (
                self.max_wall_s,
                self.max_llm_calls,
                self.max_tool_calls,
                self.max_input_tokens,
                self.max_output_tokens,
                self.max_total_tokens,
                self.max_cost,
            )
        )

    def to_limits(self) -> ResourceLimits:
        """The hard, re-validatable caps the store enforces under concurrency."""
        return ResourceLimits(
            max_llm_calls=self.max_llm_calls,
            max_tool_calls=self.max_tool_calls,
            max_input_tokens=self.max_input_tokens,
            max_output_tokens=self.max_output_tokens,
            max_total_tokens=self.max_total_tokens,
            max_cost=Money.from_usd(self.max_cost) if self.max_cost is not None else None,
            max_wall_s=self.max_wall_s,
        )
