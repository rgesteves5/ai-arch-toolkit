"""The full metering/budget surface is reachable from the top-level package (and in __all__)."""

from __future__ import annotations

import pytest

import ai_arch_toolkit as ai

# Everything a user should be able to `from ai_arch_toolkit import ...` without reaching into
# .core / .toolkit.budget. A regression here means an export was dropped or renamed.
_METERING_SURFACE = [
    # core mechanism
    "MeterScope",
    "MeterSnapshot",
    "RunConfig",
    "Money",
    "Cost",
    "CostKind",
    "Reservation",
    "OperationRequest",
    "AdmissionController",
    "AdmissionDecision",
    "AdmissionDenied",
    "NotMeteredOperationError",
    "Pricer",
    "UsageEvent",
    "UsageSink",
    # toolkit budget opinion
    "BudgetPolicy",
    "BudgetController",
    "BudgetReport",
    "BudgetExceeded",
    "Estimator",
    "HeuristicEstimator",
    "Reserve",
    "Unpriced",
    "budget_scope",
]


@pytest.mark.parametrize("name", _METERING_SURFACE)
def test_metering_name_is_exported(name: str) -> None:
    assert hasattr(ai, name), f"{name} is not importable from ai_arch_toolkit"
    assert name in ai.__all__, f"{name} is missing from ai_arch_toolkit.__all__"


def test_budget_scope_round_trips_from_the_top_level() -> None:
    # The exported helper builds an enforcing scope whose report is the exported report type.
    with ai.budget_scope(ai.BudgetPolicy(max_cost=1.0)) as scope:
        assert isinstance(scope, ai.MeterScope)
    report = ai.BudgetReport.from_snapshot(scope.snapshot())
    assert isinstance(report, ai.BudgetReport) and report.cost == 0.0
