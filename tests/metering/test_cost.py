"""Cost: typed known/estimated/unknown, and the fail-closed-preserving merge."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ai_arch_toolkit.core._metering._cost import Cost
from ai_arch_toolkit.core._metering._money import Money


def test_factories():
    k = Cost.known(Money.from_usd(0.01))
    assert k.kind == "known" and k.is_known and k.amount == Money.from_usd(0.01)

    e = Cost.estimated(Money.from_usd(0.02))
    assert e.kind == "estimated" and not e.is_known

    u = Cost.unknown("model_unpriced")
    assert u.kind == "unknown" and u.reason == "model_unpriced" and u.amount is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "known"},  # missing amount
        {"kind": "estimated"},  # missing amount
        {"kind": "unknown"},  # missing reason
        {"kind": "unknown", "amount": Money.zero(), "reason": "x"},  # amount on unknown
        {"kind": "known", "amount": Money.zero(), "reason": "x"},  # reason on known
    ],
)
def test_validation_rejects_illegal_shapes(kwargs):
    with pytest.raises(ValueError):
        Cost(**kwargs)


def test_merged_known_sum():
    m = Cost.merged(Cost.known(Money.from_usd(0.01)), Cost.known(Money.from_usd(0.02)))
    assert m.kind == "known" and m.amount == Money.from_usd(0.03)


def test_merged_degrades_to_estimated():
    m = Cost.merged(Cost.known(Money.from_usd(0.01)), Cost.estimated(Money.from_usd(0.02)))
    assert m.kind == "estimated" and m.amount == Money.from_usd(0.03)


def test_merged_any_unknown_poisons_the_merge():
    # F6 safety property: an unknown component makes the whole merge unknown, so a
    # cost cap's fail-closed cannot be defeated by hiding an unknown inside a sum.
    m = Cost.merged(Cost.known(Money.from_usd(0.01)), Cost.unknown("model_unpriced"))
    assert m.kind == "unknown" and m.amount is None and "model_unpriced" in (m.reason or "")


def test_merged_empty_is_known_zero():
    m = Cost.merged()
    assert m.kind == "known" and m.amount == Money.zero()


def test_frozen_and_hashable():
    c = Cost.known(Money.from_usd(0.01))
    assert c in {c}
    with pytest.raises(FrozenInstanceError):
        c.kind = "unknown"
