"""Tests for toolkit/tools/_first_results.py and _values.py: paging a source that has no offset,
and writing what the geo tools read (T08b)."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.toolkit.tools._first_results import asked, first_results_window
from ai_arch_toolkit.toolkit.tools._values import plain, utc

_LINES = [f"{number}. place {number}" for number in range(1, 41)]


class TestFirstResults:
    def test_it_asks_for_one_more_than_the_page_ends_on_within_the_depth(self) -> None:
        assert asked(0, 10, 40) == 11
        assert asked(20, 10, 40) == 31
        assert asked(35, 10, 40) == 40

    def test_one_more_result_means_a_next_page_with_no_total(self) -> None:
        window = first_results_window(_LINES[:11], offset=0, limit=10, depth=40, narrow="x")

        assert window.text().endswith("[results 1-10 | next: offset=10]")
        assert "11. place 11" not in window.body

    def test_fewer_results_than_asked_is_the_whole_list_with_its_total(self) -> None:
        window = first_results_window(_LINES[:13], offset=10, limit=10, depth=40, narrow="x")

        assert window.body.splitlines() == ["11. place 11", "12. place 12", "13. place 13"]
        assert window.text().endswith("[results 11-13 of 13 | end]")

    def test_at_the_depth_the_page_says_how_to_reach_the_rest(self) -> None:
        window = first_results_window(
            _LINES, offset=30, limit=10, depth=40, narrow="add a country"
        )

        assert window.body.endswith(
            "40. place 40\n(the source returns no more than 40 results; add a country)"
        )
        assert window.next_call is None and window.total is None

    def test_a_short_list_read_whole_has_no_footer(self) -> None:
        window = first_results_window(_LINES[:3], offset=0, limit=10, depth=40, narrow="x")

        assert window.text() == "1. place 1\n2. place 2\n3. place 3"


class TestValues:
    @pytest.mark.parametrize(
        ("value", "text"),
        [
            (1e-05, "0.00001"),
            (2.91849e13, "29184900000000.0"),
            (21.4, "21.4"),
            (5, "5"),
            (None, ""),
            (True, "true"),
            ("  a  b ", "a b"),
        ],
    )
    def test_numbers_come_without_scientific_notation(self, value: object, text: str) -> None:
        assert plain(value) == text

    def test_instants_are_iso_8601_utc(self) -> None:
        assert utc(1710000000) == "2024-03-09T16:00:00Z"
        assert utc(1710000000.123) == "2024-03-09T16:00:00.123Z"

    def test_an_instant_out_of_range_is_a_shape_error(self) -> None:
        with pytest.raises(OverflowError):
            utc(1e30)
