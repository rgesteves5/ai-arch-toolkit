"""``BudgetReport`` — a human-facing projection of a run's meter against its policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.core._metering._admission import MeterSnapshot
from ai_arch_toolkit.toolkit.budget._policy import BudgetPolicy

__all__ = ["BudgetReport"]


def _breached(snapshot: MeterSnapshot, policy: BudgetPolicy) -> tuple[str, ...]:
    """Committed dimensions that have reached or exceeded their cap."""
    committed_input = (
        snapshot.input_tokens + snapshot.cache_read_tokens + snapshot.cache_write_tokens
    )
    rows = (
        ("llm_calls", policy.max_llm_calls, snapshot.llm_calls),
        ("tool_calls", policy.max_tool_calls, snapshot.tool_calls),
        ("input_tokens", policy.max_input_tokens, committed_input),
        ("output_tokens", policy.max_output_tokens, snapshot.output_tokens),
        ("total_tokens", policy.max_total_tokens, snapshot.total_tokens),
        ("cost", policy.max_cost, snapshot.cost.to_float()),
        ("wall_s", policy.max_wall_s, snapshot.elapsed_s),
    )
    return tuple(dim for dim, cap, current in rows if cap is not None and current >= cap)


@dataclass(frozen=True, slots=True, kw_only=True)
class BudgetReport:
    """What a run consumed, and whether it hit its budget. ``cost`` is the known USD total."""

    llm_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    unknown_cost_count: int
    cost_uncertain: bool  # some call(s) couldn't be priced -> `cost` undercounts the real spend
    elapsed_s: float
    over_budget: bool
    breached: tuple[str, ...]

    @classmethod
    def from_snapshot(
        cls, snapshot: MeterSnapshot, policy: BudgetPolicy | None = None
    ) -> BudgetReport:
        """Project a snapshot; when a policy is given, flag the dimensions it reached/exceeded."""
        breached = _breached(snapshot, policy) if policy is not None else ()
        return cls(
            llm_calls=snapshot.llm_calls,
            tool_calls=snapshot.tool_calls,
            input_tokens=snapshot.input_tokens,
            output_tokens=snapshot.output_tokens,
            total_tokens=snapshot.total_tokens,
            cost=snapshot.cost.to_float(),
            unknown_cost_count=snapshot.unknown_cost_count,
            cost_uncertain=snapshot.unknown_cost_count > 0,
            elapsed_s=snapshot.elapsed_s,
            over_budget=bool(breached),
            breached=breached,
        )

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable view for traces/logs."""
        return {
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "unknown_cost_count": self.unknown_cost_count,
            "cost_uncertain": self.cost_uncertain,
            "elapsed_s": self.elapsed_s,
            "over_budget": self.over_budget,
            "breached": list(self.breached),
        }
