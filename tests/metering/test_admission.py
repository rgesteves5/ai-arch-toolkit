"""Admission value types: snapshot derivations, reservation math, decision validation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ai_arch_toolkit.core._metering._admission import (
    AdmissionDecision,
    AdmissionDenied,
    MeterSnapshot,
    NotMeteredOperationError,
    Reservation,
    ResourceLimits,
)
from ai_arch_toolkit.core._metering._money import Money


def test_admission_denied_is_structured():
    e = AdmissionDenied("over", dimension="llm_calls", limit=5, current=5, attempted=1)
    assert e.dimension == "llm_calls" and e.limit == 5 and e.current == 5 and e.attempted == 1
    assert "over" in str(e)


def test_not_metered_is_an_admission_denied():
    # So the flow executor catching AdmissionDenied also catches the batch guard.
    assert issubclass(NotMeteredOperationError, AdmissionDenied)
    assert isinstance(NotMeteredOperationError("batch"), AdmissionDenied)


def test_reservation_none_and_add():
    assert Reservation.none() == Reservation()
    a = Reservation(input_tokens=10, output_tokens=5, cost=Money.from_usd(0.01))
    b = Reservation(input_tokens=2, cost=Money.from_usd(0.02))
    s = a + b
    assert s.input_tokens == 12 and s.output_tokens == 5 and s.cost == Money.from_usd(0.03)


def test_reservation_frozen():
    with pytest.raises(FrozenInstanceError):
        Reservation().input_tokens = 1


def test_snapshot_derived_totals():
    snap = MeterSnapshot(
        input_tokens=100,
        output_tokens=40,
        cache_read_tokens=30,
        cache_write_tokens=20,
        out_input_tokens=7,
        out_output_tokens=3,
    )
    assert snap.total_tokens == 190
    assert snap.out_total_tokens == 10


def test_snapshot_defaults_are_zero():
    snap = MeterSnapshot()
    assert snap.llm_calls == 0 and snap.cost == Money.zero() and snap.out_cost == Money.zero()
    assert snap.total_tokens == 0 and snap.out_total_tokens == 0


def test_resource_limits_optional():
    lim = ResourceLimits(max_llm_calls=5, max_cost=Money.from_usd(0.10))
    assert lim.max_llm_calls == 5 and lim.max_cost == Money.from_usd(0.10)
    assert lim.max_tool_calls is None and lim.max_wall_s is None


def test_decision_allow():
    d = AdmissionDecision.allow()
    assert d.admitted and d.reservation == Reservation() and d.denial is None
    res = Reservation(cost=Money.from_usd(0.01))
    lim = ResourceLimits(max_cost=Money.from_usd(0.10))
    d2 = AdmissionDecision.allow(res, lim)
    assert d2.reservation == res and d2.limits == lim


def test_decision_deny():
    denial = AdmissionDenied(dimension="max_cost")
    d = AdmissionDecision.deny(denial)
    assert not d.admitted and d.denial is denial


@pytest.mark.parametrize(
    "kwargs",
    [
        {"admitted": True, "denial": AdmissionDenied()},  # admit must not carry denial
        {"admitted": False},  # deny requires a denial
    ],
)
def test_decision_validation(kwargs):
    with pytest.raises(ValueError):
        AdmissionDecision(**kwargs)
