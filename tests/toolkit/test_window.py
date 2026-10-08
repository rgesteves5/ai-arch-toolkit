"""The window every cut goes through: what was shown, the total, and the call that reads on."""

from __future__ import annotations

from itertools import pairwise

from ai_arch_toolkit.core._response import ToolCall
from ai_arch_toolkit.core._tools._decorator import tool
from ai_arch_toolkit.core._tools._executor import execute_tool
from ai_arch_toolkit.core._tools._result import ToolResult
from ai_arch_toolkit.toolkit.tools._window import (
    Window,
    find_window,
    list_window,
    page_window,
    text_window,
)

TEXT = "".join(f"row {i:03d} | value {i * 7}\n" for i in range(300))
TABLE = (
    '|- id="1959"\n| 1959\n| Emilio Segre\n| "For the antiproton."\n'
    '|- id="1960"\n| 1960\n| Donald Glaser\n| "For the bubble chamber."\n'
    '|- id="1961"\n| 1961\n| Robert Hofstadter\n| "For electron scattering."\n'
)


def _follow(first: Window, read_on) -> list[Window]:
    windows = [first]
    while windows[-1].next_call is not None:
        windows.append(read_on(windows[-1].next_call))
    return windows


class TestTextWindow:
    def test_following_the_footers_rebuilds_the_text_exactly(self) -> None:
        windows = _follow(
            text_window(TEXT, limit=1000),
            lambda call: text_window(TEXT, offset=call["offset"], limit=1000),
        )

        assert "".join(window.body for window in windows) == TEXT
        assert len(windows) > 5

    def test_a_window_ends_on_a_line_and_names_the_next_call(self) -> None:
        window = text_window(TEXT, limit=1000)

        assert window.body.endswith("\n") and len(window.body) <= 1000
        assert (
            window.footer()
            == f"[chars 0-{window.last} of {len(TEXT)} | next: offset={window.last}]"
        )

    def test_a_line_longer_than_the_window_is_cut_where_the_limit_falls(self) -> None:
        window = text_window("x" * 50, limit=20)

        assert (window.body, window.next_call) == ("x" * 20, {"offset": 20})

    def test_the_last_window_says_end(self) -> None:
        start = len(TEXT) - 30
        window = text_window(TEXT, offset=start, limit=1000)

        assert window.next_call is None
        assert window.footer() == f"[chars {start}-{len(TEXT)} of {len(TEXT)} | end]"

    def test_a_text_that_fits_has_no_footer(self) -> None:
        window = text_window("short\n", limit=100)

        assert window.footer() == "" and window.text() == "short\n"

    def test_an_offset_past_the_end_shows_nothing_and_says_end(self) -> None:
        window = text_window("abc", offset=10, limit=5)

        assert window.body == "" and window.text() == "[chars 3-3 of 3 | end]"

    def test_the_footer_follows_the_body_on_its_own_line(self) -> None:
        window = text_window("x" * 50, limit=20)

        assert window.text() == "x" * 20 + "\n[chars 0-20 of 50 | next: offset=20]"

    def test_a_rest_no_call_reaches_says_how_to_narrow_instead(self) -> None:
        # A command's output: reading on would run it again (run_command, python_repl).
        cut = Window(body="a\n", unit="chars", first=0, last=2, total=40, rest="run it narrowed")
        dead_end = Window(body="a\n", unit="chars", first=0, last=2, total=40)

        assert cut.text() == "a\n[chars 0-2 of 40 | run it narrowed]"
        assert dead_end.footer() == "[chars 0-2 of 40 | the rest cannot be read here]"
        assert cut.result().metadata["window"]["next_call"] is None


class TestFindWindow:
    def test_a_match_comes_with_the_whole_lines_around_it(self) -> None:
        window = find_window(TABLE, "glaser", limit=500, context=40)

        assert '| 1960\n| Donald Glaser\n| "For the bubble chamber."' in window.body
        assert window.body.startswith("[at char ")
        assert window.footer() == '[matches 1-1 of 1 for "glaser" | end]'

    def test_matching_ignores_case(self) -> None:
        assert find_window(TABLE, "GLASER", limit=500).total == 1

    def test_following_the_footers_reaches_every_match_once_in_order(self) -> None:
        doc = "".join(f"line {i}\n" for i in range(200))
        total = sum(1 for i in range(200) if str(i).startswith("1"))
        windows = _follow(
            find_window(doc, "line 1", limit=60, context=0),
            lambda call: find_window(
                doc, call["find"], offset=call["offset"], limit=60, context=0
            ),
        )

        assert [window.total for window in windows] == [total] * len(windows)
        assert windows[0].first == 1 and windows[-1].last == total
        for before, after in pairwise(windows):
            assert after.first == before.last + 1

    def test_the_next_call_repeats_the_term_and_moves_the_offset(self) -> None:
        doc = "".join(f"line {i}\n" for i in range(200))
        window = find_window(doc, "line 1", limit=60, context=0)

        assert window.next_call == {"find": "line 1", "offset": window.next_call["offset"]}
        assert window.footer().endswith(
            f'next: find="line 1", offset={window.next_call["offset"]}]'
        )

    def test_matches_close_together_share_one_block(self) -> None:
        window = find_window("alpha\nbeta alpha\ngamma\n", "alpha", limit=500, context=10)

        assert window.body.count("[at char ") == 1
        assert (window.first, window.last, window.total) == (1, 2, 2)

    def test_no_match_says_so(self) -> None:
        window = find_window(TABLE, "nobel", limit=500)

        assert window.body == "" and window.total == 0
        assert window.text() == '[no matches for "nobel"]'

    def test_a_frequent_term_stays_within_the_limit_and_counts_every_match_once(self) -> None:
        windows = _follow(
            find_window(TEXT, "row", limit=500),
            lambda call: find_window(TEXT, call["find"], offset=call["offset"], limit=500),
        )

        assert all(len(window.body) <= 600 for window in windows)
        assert windows[0].first == 1 and windows[-1].last == TEXT.count("row")
        for before, after in pairwise(windows):
            assert after.first == before.last + 1

    def test_an_empty_term_matches_nothing(self) -> None:
        window = find_window(TABLE, "", limit=500)

        assert window.total == 0 and window.text() == '[no matches for ""]'

    def test_a_term_with_accents_stays_readable_in_the_footer(self) -> None:
        window = find_window("Émile Borel\n" * 30, "émile", limit=40, context=0)

        assert window.total == 30
        assert 'for "émile"' in window.footer() and 'next: find="émile"' in window.footer()


class TestListWindows:
    def test_a_page_the_source_cut_names_its_place_and_the_next_page(self) -> None:
        window = list_window(["a", "b"], first=21, total=45, next_call={"offset": 23})

        assert window.text() == "a\nb\n[results 21-22 of 45 | next: offset=23]"

    def test_a_cursor_page_without_a_total(self) -> None:
        window = list_window(["a"], next_call={"page_token": "abc"})

        assert window.footer() == '[results 1-1 | next: page_token="abc"]'

    def test_the_last_page_says_end(self) -> None:
        assert (
            list_window(["x", "y"], first=44, total=45).footer() == "[results 44-45 of 45 | end]"
        )

    def test_a_complete_first_page_has_no_footer(self) -> None:
        assert list_window(["a", "b"], total=2).footer() == ""

    def test_a_list_the_tool_holds_is_paged_with_its_total(self) -> None:
        items = [str(i) for i in range(50)]

        middle = page_window(items, offset=20, limit=20)
        last = page_window(items, offset=40, limit=20, param="start")

        assert middle.body == "\n".join(items[20:40])
        assert middle.footer() == "[results 21-40 of 50 | next: offset=40]"
        assert last.footer() == "[results 41-50 of 50 | end]"


class TestResult:
    def test_the_result_carries_the_text_and_the_window(self) -> None:
        window = text_window(TEXT, limit=1000)

        result = window.result()

        assert isinstance(result, ToolResult) and result.ok
        assert result.value == window.text()
        assert result.metadata == {
            "window": {
                "unit": "chars",
                "first": 0,
                "last": window.last,
                "total": len(TEXT),
                "next_call": {"offset": window.last},
            }
        }

    def test_the_executor_hands_the_window_to_the_caller(self) -> None:
        @tool
        def read(offset: int = 0) -> ToolResult:
            """Read the document."""
            return text_window(TEXT, offset=offset, limit=1000).result()

        result = execute_tool(ToolCall(id="c1", name="read", input={"offset": 1000}), [read])

        assert result.ok and str(result.value).startswith(TEXT[1000:1020])
        assert result.metadata["window"]["first"] == 1000
