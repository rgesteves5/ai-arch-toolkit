"""toolkit/budget: policy → limits, controller admit/deny/strict, report, end-to-end enforce."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._metering._admission import (
    AdmissionDenied,
    MeterSnapshot,
    Reservation,
)
from ai_arch_toolkit.core._metering._money import Money
from ai_arch_toolkit.core._metering._operation import OperationRequest
from ai_arch_toolkit.core._metering._scope import MeterScope, RunConfig
from ai_arch_toolkit.toolkit.budget import (
    BudgetController,
    BudgetExceeded,
    BudgetPolicy,
    BudgetReport,
)

MODEL = "claude-sonnet-4-6"  # priced in _default_pricing.toml


def llm_req(**kw) -> OperationRequest:
    return OperationRequest(kind="llm", parent_span_id="run", **kw)


# ── policy ───────────────────────────────────────────────────────────────────


def test_to_limits_maps_cost_to_money():
    lim = BudgetPolicy(max_llm_calls=3, max_cost=0.5).to_limits()
    assert lim.max_llm_calls == 3 and lim.max_cost == Money.from_usd(0.5)


@pytest.mark.parametrize("kwargs", [{"max_llm_calls": -1}, {"max_cost": -0.1}, {"max_wall_s": 0}])
def test_policy_validation(kwargs):
    with pytest.raises(ValueError):
        BudgetPolicy(**kwargs)


def test_is_empty():
    assert BudgetPolicy().is_empty
    assert not BudgetPolicy(max_cost=1.0).is_empty


def test_budget_exceeded_is_an_admission_denied():
    assert issubclass(BudgetExceeded, AdmissionDenied)


# ── controller admit ─────────────────────────────────────────────────────────


def test_admit_under_cap_allows_with_limits():
    d = BudgetController(BudgetPolicy(max_llm_calls=5)).admit(
        MeterSnapshot(llm_calls=2), llm_req()
    )
    assert d.admitted and d.limits is not None and d.limits.max_llm_calls == 5


def test_admit_at_the_call_cap_denies_with_dimension():
    d = BudgetController(BudgetPolicy(max_llm_calls=1)).admit(
        MeterSnapshot(llm_calls=1), llm_req()
    )
    assert not d.admitted
    assert isinstance(d.denial, BudgetExceeded) and d.denial.dimension == "llm_calls"


def test_admit_counts_outstanding_not_only_committed():
    d = BudgetController(BudgetPolicy(max_llm_calls=1)).admit(
        MeterSnapshot(out_llm_calls=1), llm_req()
    )
    assert not d.admitted  # one already in flight fills the cap


def test_admit_denies_when_committed_cost_is_over_cap():
    # Soft reserve holds no cost, so denial happens once committed cost is already over the cap.
    d = BudgetController(BudgetPolicy(max_cost=0.01)).admit(
        MeterSnapshot(cost=Money.from_usd(0.02)), llm_req()
    )
    assert not d.admitted and d.denial is not None and d.denial.dimension == "cost"


def test_admit_at_cost_cap_still_allows_a_soft_op():
    # At the cap with a zero-cost reservation, the op is admitted (cost is enforced post-hoc).
    d = BudgetController(BudgetPolicy(max_cost=0.01)).admit(
        MeterSnapshot(cost=Money.from_usd(0.01)), llm_req()
    )
    assert d.admitted


def test_soft_default_reserves_nothing():
    d = BudgetController(BudgetPolicy(max_llm_calls=5)).admit(
        MeterSnapshot(), llm_req(model=MODEL)
    )
    assert d.admitted and d.reservation == Reservation()


# ── strict reservation ───────────────────────────────────────────────────────


def test_strict_reserves_for_a_priced_model():
    d = BudgetController(BudgetPolicy(reserve="strict")).admit(
        MeterSnapshot(), llm_req(model=MODEL, content_size_hint=400, declared_max_output_tokens=50)
    )
    assert d.admitted
    assert d.reservation.input_tokens == 100 and d.reservation.output_tokens == 50
    assert d.reservation.cost.pico > 0


def test_strict_denies_an_unpriced_model_fail_closed():
    d = BudgetController(BudgetPolicy(reserve="strict")).admit(
        MeterSnapshot(), llm_req(model="totally-made-up-model")
    )
    assert not d.admitted and d.denial is not None and d.denial.dimension == "cost"


# ── unpriced fail-closed under a cost cap ────────────────────────────────────


def test_unpriced_fail_closed_denies_after_an_unknown_cost_settles():
    # A prior call settled with an unknown cost: it never entered committed_cost, so the cap could
    # silently overshoot. Default policy fails closed on the next op.
    d = BudgetController(BudgetPolicy(max_cost=1.0)).admit(
        MeterSnapshot(unknown_cost_count=1), llm_req(model=MODEL)
    )
    assert not d.admitted and d.denial is not None and d.denial.dimension == "cost"


def test_unpriced_fail_closed_denies_a_server_tool_op_before_it_runs():
    # Server tools carry a provider-side charge absent from the token counts -> unbounded.
    d = BudgetController(BudgetPolicy(max_cost=1.0)).admit(
        MeterSnapshot(), llm_req(model=MODEL, has_server_tools=True)
    )
    assert not d.admitted and d.denial is not None and d.denial.dimension == "cost"


def test_unpriced_allow_proceeds_despite_an_unknown_cost():
    d = BudgetController(BudgetPolicy(max_cost=1.0, unpriced="allow")).admit(
        MeterSnapshot(unknown_cost_count=1), llm_req(model=MODEL)
    )
    assert d.admitted


def test_unpriced_allow_proceeds_for_a_server_tool_op():
    d = BudgetController(BudgetPolicy(max_cost=1.0, unpriced="allow")).admit(
        MeterSnapshot(), llm_req(model=MODEL, has_server_tools=True)
    )
    assert d.admitted


def test_unpriced_fail_closed_is_inert_without_a_cost_cap():
    # No max_cost -> the unknown cost can't threaten a (nonexistent) budget; the op proceeds.
    d = BudgetController(BudgetPolicy(max_llm_calls=5)).admit(
        MeterSnapshot(unknown_cost_count=1), llm_req(model=MODEL, has_server_tools=True)
    )
    assert d.admitted


# ── report ───────────────────────────────────────────────────────────────────


def test_report_flags_cost_uncertain_when_a_call_was_unpriced():
    snap = MeterSnapshot(llm_calls=1, cost=Money.from_usd(0.02), unknown_cost_count=1)
    r = BudgetReport.from_snapshot(snap)
    assert r.cost_uncertain and r.unknown_cost_count == 1
    assert r.to_dict()["cost_uncertain"] is True


def test_report_cost_is_certain_when_everything_was_priced():
    r = BudgetReport.from_snapshot(MeterSnapshot(llm_calls=2, cost=Money.from_usd(0.05)))
    assert not r.cost_uncertain


def test_report_flags_a_reached_cap():
    snap = MeterSnapshot(
        llm_calls=3, input_tokens=100, output_tokens=40, cost=Money.from_usd(0.02)
    )
    r = BudgetReport.from_snapshot(snap, BudgetPolicy(max_llm_calls=3))
    assert r.llm_calls == 3 and r.cost == 0.02
    assert r.over_budget and "llm_calls" in r.breached


def test_report_without_a_policy_is_never_over():
    r = BudgetReport.from_snapshot(MeterSnapshot(llm_calls=3))
    assert not r.over_budget and r.breached == ()


# ── end-to-end through the scope + store ─────────────────────────────────────


def test_scope_enforces_the_budget_end_to_end():
    scope = MeterScope(RunConfig(controller=BudgetController(BudgetPolicy(max_llm_calls=1))))
    with scope:
        scope.open(OperationRequest(kind="llm", parent_span_id=scope.run_span_id)).mark_started()
        with pytest.raises(BudgetExceeded):
            scope.open(OperationRequest(kind="llm", parent_span_id=scope.run_span_id))
    assert scope.snapshot().llm_calls == 1
