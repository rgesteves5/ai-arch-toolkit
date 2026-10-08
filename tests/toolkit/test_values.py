"""Tests for toolkit/tools/_values.py: how the tools write the numbers and instants they read
(T00, output point 5)."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.toolkit.tools._values import decimal_text, plain, utc


class TestPlain:
    @pytest.mark.parametrize(
        ("value", "text"),
        [
            (1e-05, "0.00001"),
            (2.91849e13, "29184900000000.0"),  # the World Bank's GDP read "2.91849e+13"
            (1e16, "10000000000000000.0"),
            (10578174.0, "10578174.0"),  # a float keeps the ".0" its JSON number had
            (5.0, "5.0"),
            (-0.0, "0.0"),
            (21.4, "21.4"),
            (-0.25, "-0.25"),
            (1.23456789012, "1.23456789012"),  # not rounded to six digits
            (5, "5"),
            (10578174, "10578174"),
            (float("nan"), "nan"),
            (float("inf"), "inf"),
            (None, ""),
            (True, "true"),  # JSON's word, not Python's
            (False, "false"),
            ("  a  b ", "a b"),
            ("007", "007"),  # text stays text: a code keeps its zeros
            ("1.0E7", "1.0E7"),
        ],
    )
    def test_numbers_come_in_plain_digits_and_other_values_as_one_line(
        self, value: object, text: str
    ) -> None:
        assert plain(value) == text


class TestDecimalText:
    @pytest.mark.parametrize(
        ("text", "shown"),
        [
            ("+1.750", "1.750"),  # a Wikidata amount keeps the digits it gave
            ("-3", "-3"),
            ("1.0E7", "10000000"),  # an xsd:double from the query service
            ("2.5e-3", "0.0025"),
            (" 12 ", "12"),
            ("007", "7"),  # only for numbers sent as text: a code goes through plain
            (".5", "0.5"),
        ],
    )
    def test_a_number_sent_as_text_reads_in_plain_digits(self, text: str, shown: str) -> None:
        assert decimal_text(text) == shown

    @pytest.mark.parametrize(
        "text", ["not a number", "NaN", "Infinity", "1_000", "١٢٣", "0x10", "", "1e"]
    )
    def test_text_that_is_no_decimal_number_comes_back_as_it_was(self, text: str) -> None:
        assert decimal_text(text) == text

    @pytest.mark.parametrize("text", ["1E1000000", "1e-1000000", "-4.2E+999"])
    def test_an_exponent_past_the_bound_keeps_the_sources_text(self, text: str) -> None:
        # One SPARQL cell of 1E1000000 would be a line of a million digits.
        assert decimal_text(text) == text

    def test_an_exponent_within_the_bound_is_written_out(self) -> None:
        assert decimal_text("1E100") == "1" + "0" * 100
        assert decimal_text("1E-100") == "0." + "0" * 99 + "1"


class TestUtc:
    def test_instants_are_iso_8601_utc(self) -> None:
        assert utc(1710000000) == "2024-03-09T16:00:00Z"
        assert utc(1710000000.123) == "2024-03-09T16:00:00.123Z"

    def test_an_instant_out_of_range_is_a_shape_error(self) -> None:
        with pytest.raises(OverflowError):
            utc(1e30)
