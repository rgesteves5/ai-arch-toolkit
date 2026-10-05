"""Budget policy for the metering mechanism — the opinion layer over the neutral ``core`` meter.

Wire it in via a :class:`~ai_arch_toolkit.core.RunConfig`::

    from ai_arch_toolkit.core import MeterScope, RunConfig
    from ai_arch_toolkit.toolkit.budget import BudgetPolicy, BudgetController, BudgetReport

    policy = BudgetPolicy(max_llm_calls=5, max_cost=0.50)
    with MeterScope(RunConfig(controller=BudgetController(policy))) as scope:
        ...  # LLM/tool calls are measured AND enforced
    report = BudgetReport.from_snapshot(scope.snapshot(), policy)

Runs in parallel spend from one :class:`SharedBudget` (seeded with what an app already spent)
when each binds it: ``RunConfig(controller=..., shared=budget)`` (D57).
"""

from __future__ import annotations

from ai_arch_toolkit.toolkit.budget._controller import BudgetController
from ai_arch_toolkit.toolkit.budget._estimator import Estimator, HeuristicEstimator
from ai_arch_toolkit.toolkit.budget._exceptions import BudgetExceeded
from ai_arch_toolkit.toolkit.budget._policy import BudgetPolicy, Reserve, Unpriced
from ai_arch_toolkit.toolkit.budget._report import BudgetReport
from ai_arch_toolkit.toolkit.budget._scope import budget_scope
from ai_arch_toolkit.toolkit.budget._shared import SharedBudget

__all__ = [
    "BudgetController",
    "BudgetExceeded",
    "BudgetPolicy",
    "BudgetReport",
    "Estimator",
    "HeuristicEstimator",
    "Reserve",
    "SharedBudget",
    "Unpriced",
    "budget_scope",
]
