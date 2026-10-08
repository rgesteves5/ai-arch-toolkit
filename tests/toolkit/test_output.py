"""The output of a run within a limit: its start, its size, and how to narrow it (T09)."""

from __future__ import annotations

from ai_arch_toolkit.toolkit.tools._output import Head, fit

_REST = "run it narrowed"


def _head(text: str, keep: int = 101) -> Head:
    head = Head(keep)
    for start in range(0, len(text), 7):  # as it arrives, a piece at a time
        head.add(text[start : start + 7])
    return head


class TestHead:
    def test_it_keeps_the_start_and_counts_everything(self) -> None:
        head = _head("x" * 10_000, keep=50)

        assert (head.text, head.total) == ("x" * 50, 10_000)

    def test_a_stream_within_keep_is_kept_whole(self) -> None:
        assert _head("abc\ndef\n").text == "abc\ndef\n"


class TestFit:
    def test_parts_that_fit_show_whole(self) -> None:
        assert fit([_head("out\n"), _head("err\n")], limit=100, rest=_REST) == ["out\n", "err\n"]

    def test_a_part_that_does_not_fit_ends_on_a_line_with_its_size_and_how_to_narrow(self) -> None:
        lines = "".join(f"line {n}\n" for n in range(100))

        (shown,) = fit([_head(lines)], limit=40, rest=_REST)

        assert shown == "line 0\nline 1\nline 2\nline 3\nline 4\n" + (
            f"[chars 0-35 of {len(lines)} | {_REST}]"
        )

    def test_a_short_part_shows_whole_and_leaves_its_share_to_the_long_one(self) -> None:
        out, err = fit([_head("y" * 500), _head("failed\n")], limit=100, rest=_REST)

        assert err == "failed\n"
        assert out.startswith("y" * 93 + "\n[chars 0-93 of 500 |")

    def test_two_long_parts_share_the_limit_evenly(self) -> None:
        out, err = fit([_head("a" * 500), _head("b" * 500)], limit=100, rest=_REST)

        assert out.startswith("a" * 50 + "\n[chars 0-50 of 500 |")
        assert err.startswith("b" * 50 + "\n[chars 0-50 of 500 |")
