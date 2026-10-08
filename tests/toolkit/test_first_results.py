"""Tests for toolkit/tools/_first_results.py: paging a source that has no offset (T08b). How the
values are written, ``_values.py``, is tested in ``test_values.py``."""

from __future__ import annotations

from ai_arch_toolkit.toolkit.tools._first_results import asked, first_results_window

_LINES = [f"{number}. place {number}" for number in range(1, 41)]


class TestFirstResults:
    def test_it_asks_for_one_more_than_the_page_ends_on_within_the_depth(self) -> None:
        assert asked(0, 10, 40) == 11
        assert asked(20, 10, 40) == 31
        assert asked(35, 10, 40) == 40

    def test_one_more_result_means_a_next_page_with_no_total(self) -> None:
        window = first_results_window(
            _LINES[:11], offset=0, limit=10, requested=11, depth=40, narrow="x"
        )

        assert window.text().endswith("[results 1-10 | next: offset=10]")
        assert "11. place 11" not in window.body

    def test_fewer_results_than_asked_is_the_whole_list_with_its_total(self) -> None:
        window = first_results_window(
            _LINES[:13], offset=10, limit=10, requested=21, depth=40, narrow="x"
        )

        assert window.body.splitlines() == ["11. place 11", "12. place 12", "13. place 13"]
        assert window.text().endswith("[results 11-13 of 13 | end]")

    def test_at_the_depth_the_page_says_how_to_reach_the_rest(self) -> None:
        window = first_results_window(
            _LINES, offset=30, limit=10, requested=40, depth=40, narrow="add a country"
        )

        assert window.body.endswith(
            "40. place 40\n(the source returns no more than 40 results; add a country)"
        )
        assert window.next_call is None and window.total is None

    def test_asked_for_the_whole_depth_the_first_page_has_the_total(self) -> None:
        # A source whose first results change with how many are asked (Nominatim) is asked for
        # all it gives: every page is cut from the same answer, and its total is exact.
        found = _LINES[:12]

        pages = [
            first_results_window(found, offset=offset, limit=5, requested=40, depth=40, narrow="x")
            for offset in (0, 5, 10)
        ]

        assert [page.text().splitlines()[-1] for page in pages] == [
            "[results 1-5 of 12 | next: offset=5]",
            "[results 6-10 of 12 | next: offset=10]",
            "[results 11-12 of 12 | end]",
        ]

    def test_a_short_list_read_whole_has_no_footer(self) -> None:
        window = first_results_window(
            _LINES[:3], offset=0, limit=10, requested=11, depth=40, narrow="x"
        )

        assert window.text() == "1. place 1\n2. place 2\n3. place 3"
