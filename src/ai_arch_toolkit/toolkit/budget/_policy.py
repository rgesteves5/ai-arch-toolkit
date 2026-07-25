"""``BudgetPolicy`` — user-facing budget caps that compile to neutral core ``ResourceLimits``."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from ai_arch_toolkit.core._metering._admission import ResourceLimits
from ai_arch_toolkit.core._metering._money import Money

__all__ = ["BudgetPolicy", "Reserve", "Unpriced"]

type Reserve = Literal["none", "strict"]
"""Reservation mode. ``none``: measure + settle only (soft, the default). ``strict``: reserve a
worst-case token/cost hold per operation before it runs, failing closed on unpriced models."""

type Unpriced = Literal["fail_closed", "allow"]
"""What a ``max_cost`` cap does when a call's cost is unknown (an unpriced model or a server tool
whose charge isn't in the token counts). ``fail_closed`` (default): once such a call is seen, deny
further operations — an unknown cost can't be bounded. ``allow``: proceed (the cap may undercount).
"""


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
    unpriced: Unpriced = "fail_closed"

    def __post_init__(self) -> None:
        if self.reserve not in ("none", "strict"):
            raise ValueError(f"reserve must be 'none' or 'strict', got {self.reserve!r}")
        if self.unpriced not in ("fail_closed", "allow"):
            raise ValueError(f"unpriced must be 'fail_closed' or 'allow', got {self.unpriced!r}")
        for name in (
            "max_llm_calls",
            "max_tool_calls",
            "max_input_tokens",
            "max_output_tokens",
            "max_total_tokens",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
        if self.max_cost is not None and (
            isinstance(self.max_cost, bool)
            or not isinstance(self.max_cost, (int, float))
            or not math.isfinite(self.max_cost)
            or self.max_cost < 0
        ):
            raise ValueError(f"max_cost must be a finite number >= 0, got {self.max_cost}")
        if self.max_wall_s is not None and (
            isinstance(self.max_wall_s, bool)
            or not isinstance(self.max_wall_s, (int, float))
            or not math.isfinite(self.max_wall_s)
            or self.max_wall_s <= 0
        ):
            raise ValueError(f"max_wall_s must be a finite positive number, got {self.max_wall_s}")

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
