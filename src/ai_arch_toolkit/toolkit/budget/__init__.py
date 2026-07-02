"""Budget policy for the metering mechanism — the opinion layer over the neutral ``core`` meter.

Wire it in via a :class:`~ai_arch_toolkit.core.RunConfig`::

    from ai_arch_toolkit.core import MeterScope, RunConfig
    from ai_arch_toolkit.toolkit.budget import BudgetPolicy, BudgetController, BudgetReport

    policy = BudgetPolicy(max_llm_calls=5, max_cost=0.50)
    with MeterScope(RunConfig(controller=BudgetController(policy))) as scope:
        ...  # LLM/tool calls are measured AND enforced
    report = BudgetReport.from_snapshot(scope.snapshot(), policy)
"""

from __future__ import annotations

from ai_arch_toolkit.toolkit.budget._controller import BudgetController
from ai_arch_toolkit.toolkit.budget._estimator import Estimator, HeuristicEstimator
from ai_arch_toolkit.toolkit.budget._exceptions import BudgetExceeded
from ai_arch_toolkit.toolkit.budget._policy import BudgetPolicy, Reserve, Unpriced
from ai_arch_toolkit.toolkit.budget._report import BudgetReport

__all__ = [
    "BudgetController",
    "BudgetExceeded",
    "BudgetPolicy",
    "BudgetReport",
    "Estimator",
    "HeuristicEstimator",
    "Reserve",
    "Unpriced",
]
