"""``budget_scope`` — a one-liner metered scope wiring a :class:`BudgetPolicy` into the meter."""

from __future__ import annotations

from collections.abc import Sequence

from ai_arch_toolkit.core._metering._events import UsageSink
from ai_arch_toolkit.core._metering._scope import MeterScope, Pricer, RunConfig
from ai_arch_toolkit.toolkit.budget._controller import BudgetController
from ai_arch_toolkit.toolkit.budget._estimator import Estimator, HeuristicEstimator
from ai_arch_toolkit.toolkit.budget._policy import BudgetPolicy

__all__ = ["budget_scope"]


def budget_scope(
    policy: BudgetPolicy | None = None,
    *,
    estimator: Estimator | None = None,
    sinks: Sequence[UsageSink] = (),
    pricer: Pricer | None = None,
) -> MeterScope:
    """Build a metered scope enforcing ``policy`` (measure-only when it is ``None`` or empty).

    Use it as a context manager; everything run inside — flows included, since they inherit the
    enclosing scope — is metered under one cumulative budget::

        with budget_scope(BudgetPolicy(max_cost=0.50)) as scope:
            flow.run_sync(state)
        report = BudgetReport.from_snapshot(scope.snapshot())

    Optional ``sinks`` receive redacted usage events for audit, and ``pricer``/``estimator`` swap
    the default pricing/strict-reserve estimation.
    """
    controller = None
    if policy is not None and not policy.is_empty:
        # The default estimator must price with the SAME pricer as the settle (RunConfig.pricer),
        # or a strict reservation and the actual charge would diverge under a custom pricer.
        est = estimator or HeuristicEstimator(pricer=pricer)
        controller = BudgetController(policy, estimator=est)
    return MeterScope(RunConfig(controller=controller, sinks=tuple(sinks), pricer=pricer))
