"""toolkit/budget: policy → limits, controller admit/deny/strict, report, end-to-end enforce."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._metering._admission import (
    AdmissionDenied,
    MeterSnapshot,
    Reservation,
)
from ai_arch_toolkit.core._metering._cost import Cost
from ai_arch_toolkit.core._metering._money import Money
from ai_arch_toolkit.core._metering._operation import OperationRequest
from ai_arch_toolkit.core._metering._scope import MeterScope, RunConfig
from ai_arch_toolkit.toolkit.budget import (
    BudgetController,
    BudgetExceeded,
    BudgetPolicy,
    BudgetReport,
    budget_scope,
)

MODEL = "claude-sonnet-4-6"  # priced in _default_pricing.toml


def llm_req(**kw) -> OperationRequest:
    return OperationRequest(kind="llm", parent_span_id="run", **kw)


# ── policy ───────────────────────────────────────────────────────────────────


def test_to_limits_maps_cost_to_money():
    lim = BudgetPolicy(max_llm_calls=3, max_cost=0.5).to_limits()
    assert lim.max_llm_calls == 3 and lim.max_cost == Money.from_usd(0.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_llm_calls": -1},
        {"max_cost": -0.1},
        {"max_wall_s": 0},
        # non-finite caps must fail loud at construction, never silently disable enforcement
        {"max_cost": float("nan")},
        {"max_cost": float("inf")},
        {"max_wall_s": float("nan")},
        {"max_wall_s": float("inf")},
        {"max_llm_calls": float("nan")},
        {"max_llm_calls": 1.5},
        {"max_llm_calls": True},
        {"max_llm_calls": "3"},
        {"max_cost": True},
        {"max_wall_s": "30"},
    ],
)
def test_policy_validation(kwargs):
    with pytest.raises(ValueError):
        BudgetPolicy(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"reserve": "strcit"}, "reserve must be 'none' or 'strict'"),
        ({"reserve": ["strict"]}, "reserve must be 'none' or 'strict'"),
        ({"unpriced": "fail-close"}, "unpriced must be 'fail_closed' or 'allow'"),
        ({"unpriced": ["allow"]}, "unpriced must be 'fail_closed' or 'allow'"),
    ],
)
def test_policy_rejects_invalid_modes(kwargs, message):
    with pytest.raises(ValueError, match=message):
        BudgetPolicy(**kwargs)


def test_is_empty():
    assert BudgetPolicy().is_empty
    assert not BudgetPolicy(max_cost=1.0).is_empty


def test_budget_exceeded_is_an_admission_denied():
    assert issubclass(BudgetExceeded, AdmissionDenied)


def test_budget_exceeded_preserves_maximum_and_to_dict():
    e = BudgetExceeded(dimension="cost", limit=0.5, current=0.4, attempted=0.2)
    assert e.maximum == 0.5  # back-compat alias for .limit
    d = e.to_dict()
    assert d["error"] == "budget_exceeded" and d["dimension"] == "cost"
    assert d["limit"] == 0.5 and d["maximum"] == 0.5
    assert d["current"] == 0.4 and d["attempted"] == 0.2


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


def test_admit_at_exact_cost_boundary_is_not_a_float_rounding_denial():
    # Regression (N11): committed 0.01 + outstanding 0.05 == 0.06 exactly in Money, but the float
    # sum 0.01 + 0.05 is 0.060000000000000005 > 0.06. The controller must compare in Money (mirror
    # the store) and ADMIT — a float compare would spuriously deny at the exact boundary.
    d = BudgetController(BudgetPolicy(max_cost=0.06)).admit(
        MeterSnapshot(cost=Money.from_usd(0.01), out_cost=Money.from_usd(0.05)), llm_req()
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


def test_controller_wants_request_size_only_for_strict():
    # The charge site consults this to skip stringifying the request when the estimate is unused.
    assert not BudgetController(BudgetPolicy(max_cost=1.0)).wants_request_size()
    assert BudgetController(BudgetPolicy(max_cost=1.0, reserve="strict")).wants_request_size()


def test_strict_denies_an_unpriced_model_fail_closed():
    d = BudgetController(BudgetPolicy(reserve="strict")).admit(
        MeterSnapshot(), llm_req(model="totally-made-up-model")
    )
    assert not d.admitted and d.denial is not None and d.denial.dimension == "cost"


def test_strict_reserve_rounds_input_tokens_up():
    # A reservation is a worst-case hold: ceil(401/4) = 101, not floor 100.
    d = BudgetController(BudgetPolicy(reserve="strict")).admit(
        MeterSnapshot(), llm_req(model=MODEL, content_size_hint=401, declared_max_output_tokens=0)
    )
    assert d.admitted and d.reservation.input_tokens == 101


def test_strict_reserve_adds_a_non_text_allowance():
    # Multimodal parts carry far more tokens than their char-hint placeholder — reserve for them.
    d = BudgetController(BudgetPolicy(reserve="strict")).admit(
        MeterSnapshot(),
        llm_req(
            model=MODEL, content_size_hint=400, non_text_parts=2, declared_max_output_tokens=50
        ),
    )
    assert d.admitted and d.reservation.input_tokens == 100 + 2 * 4000


def test_budget_scope_wires_the_pricer_into_the_default_estimator():
    # #5: estimate (strict reserve) and settle must use the SAME pricer, or they diverge.
    class _Pricer:
        def price(self, request, usage) -> Cost:
            return Cost.known(Money.zero())

    p = _Pricer()
    scope = budget_scope(BudgetPolicy(reserve="strict", max_cost=1.0), pricer=p)
    ctrl = scope.controller
    assert isinstance(ctrl, BudgetController)
    assert ctrl.estimator.pricer is p


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


# ── budget_scope convenience ─────────────────────────────────────────────────


def test_budget_scope_enforces_like_a_hand_built_scope():
    with budget_scope(BudgetPolicy(max_llm_calls=1)) as scope:
        scope.open(OperationRequest(kind="llm", parent_span_id=scope.run_span_id)).mark_started()
        with pytest.raises(BudgetExceeded):
            scope.open(OperationRequest(kind="llm", parent_span_id=scope.run_span_id))
    assert scope.snapshot().llm_calls == 1


def test_budget_scope_empty_policy_is_measure_only():
    with budget_scope(BudgetPolicy()) as scope:  # empty policy -> no controller, never denies
        for _ in range(3):
            op = scope.open(OperationRequest(kind="llm", parent_span_id=scope.run_span_id))
            op.mark_started()
    assert scope.controller is None and scope.snapshot().llm_calls == 3


def test_budget_scope_none_policy_is_measure_only():
    with budget_scope() as scope:
        assert scope.controller is None
