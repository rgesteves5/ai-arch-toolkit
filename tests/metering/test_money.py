"""Money: exact integer pico-USD arithmetic (the oracle's money primitive)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from ai_arch_toolkit.core._metering._money import Money


def test_zero():
    assert Money.zero().to_float() == 0.0
    assert Money.zero() == Money.from_pico(0)
    assert Money() == Money.zero()


def test_from_usd_roundtrip():
    assert Money.from_usd(0.10).to_float() == pytest.approx(0.10)
    assert Money.from_usd(Decimal("0.0021")).to_float() == pytest.approx(0.0021)


def test_from_pico_rounds_a_float_instead_of_truncating():
    # N12: a custom pricer's fractional rate*tokens must round to nearest pico, not truncate to 0.
    assert Money.from_pico(1.9).pico == 2
    assert Money.from_pico(2.4).pico == 2
    assert Money.from_pico(5).pico == 5  # ints pass through unchanged


def test_from_usd_is_decimal_exact():
    # 0.10 USD is EXACTLY ten cents, not the nearest binary float.
    assert Money.from_usd(0.10).pico == 100_000_000_000


def test_exact_accumulation_no_float_drift():
    # The headline guarantee: 0.1 + 0.2 == 0.3 exactly (raw floats fail this).
    assert Money.from_usd(0.1) + Money.from_usd(0.2) == Money.from_usd(0.3)
    assert (0.1 + 0.2) != 0.3  # sanity: raw floats really do drift


def test_from_pico_rate_times_tokens_exact():
    # $3.00 / 1M tokens = 3_000_000 pico-USD per token; x 1500 tokens = $0.0045 exactly.
    cost = Money.from_pico(3_000_000) * 1500
    assert cost == Money.from_usd(Decimal("0.0045"))


def test_arithmetic():
    a, b = Money.from_usd(0.10), Money.from_usd(0.25)
    assert a + b == Money.from_usd(0.35)
    assert b - a == Money.from_usd(0.15)
    assert a * 3 == Money.from_usd(0.30)
    assert 3 * a == Money.from_usd(0.30)
    assert -a == Money.from_pico(-a.pico)


def test_comparison():
    a, b = Money.from_usd(0.10), Money.from_usd(0.25)
    assert a < b and a <= b and b > a and b >= a and a != b


def test_hashable():
    assert len({Money.from_usd(0.10), Money.from_usd(0.10), Money.from_usd(0.20)}) == 2


def test_sum():
    parts = [Money.from_usd(0.10), Money.from_usd(0.20), Money.from_usd(0.30)]
    assert sum(parts, Money.zero()) == Money.from_usd(0.60)
    assert sum(parts) == Money.from_usd(0.60)  # bare sum starts at int 0 (via __radd__)


def test_banker_rounding_at_the_pico_boundary():
    assert Money.from_usd(Decimal("0.0000000000005")).pico == 0  # 0.5 pico -> 0 (even)
    assert Money.from_usd(Decimal("0.0000000000015")).pico == 2  # 1.5 pico -> 2 (even)


def test_equality_is_money_only():
    assert Money.from_usd(0.10) != 0.10
    assert Money.from_usd(0.10) != "0.10"


def test_immutable():
    m = Money.from_usd(0.10)
    with pytest.raises(FrozenInstanceError):
        m._pico = 5
