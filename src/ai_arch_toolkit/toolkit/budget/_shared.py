"""``SharedBudget`` — one budget several runs spend from at once (G-28, D57)."""

from __future__ import annotations

import math

from ai_arch_toolkit.core._metering._money import Money
from ai_arch_toolkit.core._metering._store import SharedMeter
from ai_arch_toolkit.toolkit.budget._policy import BudgetPolicy
from ai_arch_toolkit.toolkit.budget._report import BudgetReport

__all__ = ["SharedBudget"]


class SharedBudget(SharedMeter):
    """A :class:`~ai_arch_toolkit.core.SharedMeter` built from a :class:`BudgetPolicy`.

    Bind it to every run that spends from it, with ``config=RunConfig(shared=budget)`` (and a
    controller of the run's own, if it has one): their operations are admitted and settled
    against it under one lock, so runs in parallel never pass its ``max_cost`` together. Seed it
    with what was already spent (``spent``, in USD), from an app's ledger.

    Only ``max_cost``, ``max_llm_calls`` and ``max_tool_calls`` are shared; ``unpriced`` decides
    whether a cost no one could bound closes it.
    """

    def __init__(self, policy: BudgetPolicy, *, spent: float = 0.0) -> None:
        if isinstance(spent, bool) or not math.isfinite(spent) or spent < 0:
            raise ValueError(f"spent must be a finite number >= 0, got {spent!r}")
        super().__init__(
            policy.to_limits(),
            spent=Money.from_usd(spent),
            fail_on_unknown=policy.unpriced == "fail_closed",
        )
        self.policy = policy

    def report(self) -> BudgetReport:
        """The shared spend, the seed included, projected against the policy's caps."""
        return BudgetReport.from_snapshot(self.snapshot(), self.policy)
