"""Numbers as the tools write them: every digit, no scientific notation."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.toolkit.tools._numbers import plain_number


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (2.91849e13, "29184900000000"),  # the World Bank's GDP read "2.91849e+13"
        (1e-05, "0.00001"),
        (10578174.0, "10578174"),
        (1.23456789012, "1.23456789012"),  # not rounded to six digits
        (10578174, "10578174"),
        (-0.25, "-0.25"),
        ("+1.750", "1.750"),  # a Wikidata amount keeps the digits it gave
        ("-3", "-3"),
        ("1.0E7", "10000000"),
        ("not a number", "not a number"),
        (float("nan"), "nan"),
    ],
)
def test_a_number_reads_in_plain_digits(value: float | str, text: str) -> None:
    assert plain_number(value) == text
