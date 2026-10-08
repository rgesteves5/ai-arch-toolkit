"""Tests for toolkit/tools/_text.py."""

from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
from typing import Any

import pytest

from ai_arch_toolkit.core import ToolCall, ToolGroup, ToolResult
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools import _text as text_tools
from ai_arch_toolkit.toolkit.tools._text import (
    base64_decode,
    base64_encode,
    regex_search,
    text_stats,
)


def _invalid(call, *args) -> str:
    """The message of the validation_error ``call`` raises."""
    with pytest.raises(ToolFailure) as caught:
        call(*args)
    assert caught.value.error.type == "validation_error"
    return caught.value.error.message


def _text(result: ToolResult) -> str:
    assert result.ok and isinstance(result.value, str), result
    return result.value


def _window(result: ToolResult) -> dict[str, Any]:
    return result.metadata["window"]


class TestRegexSearch:
    def test_basic_match(self):
        result = regex_search("foo123bar", r"\d+")
        assert _text(result) == "1 match(es) for '\\\\d+':\n  [3:6] '123'"

    def test_multiple_matches(self):
        result = _text(regex_search("a1 b2 c3", r"\d"))
        assert "3 match" in result

    def test_no_matches_say_the_pattern(self):
        result = regex_search("hello", r"\d+")
        assert _text(result) == "No matches for '\\\\d+'."

    def test_groups(self):
        result = _text(regex_search("2026-02-27", r"(\d{4})-(\d{2})-(\d{2})"))
        assert "groups=" in result

    def test_invalid_regex(self):
        assert "invalid regex" in _invalid(regex_search, "text", r"[invalid")

    def test_a_page_holds_a_thousand_matches_with_the_total_and_the_next_offset(self):
        result = regex_search("a" * 5000, "a")

        lines = _text(result).splitlines()
        assert lines[0] == "5000 match(es) for 'a':"
        assert lines[1] == "  [0:1] 'a'" and lines[1000] == "  [999:1000] 'a'"
        assert lines[-1] == "[results 1-1000 of 5000 | next: offset=1000]"

    def test_following_the_footers_reaches_every_match_once(self):
        text = "ab" * 2500
        starts: list[int] = []
        call: dict[str, Any] | None = {"offset": 0}
        while call is not None:
            result = regex_search(text, "b", **call)
            starts += [
                int(line[3:].split(":")[0])
                for line in _text(result).splitlines()[1:]
                if line.startswith("  [")
            ]
            call = _window(result)["next_call"]

        assert starts == list(range(1, 5000, 2))

    def test_a_page_stays_within_its_characters_however_long_the_matches(self):
        text = ("x" * 1000 + "\n") * 19  # each match is shown twice: whole, and as its group

        result = regex_search(text, "(x+)")

        assert len(_text(result)) < 22_000
        assert _window(result)["next_call"] is not None
        assert _window(result)["total"] == 19

    def test_the_executor_refuses_a_negative_offset(self):
        call = ToolCall(
            id="c", name="regex_search", input={"text": "a", "pattern": "a", "offset": -1}
        )

        result = ToolGroup(regex_search).execute(call)

        assert result.error is not None and result.error.type == "validation_error"


class TestTextStats:
    def test_basic(self):
        result = text_stats("Hello world.")
        assert "Words: 2" in result
        assert "Characters: 12" in result
        assert "Sentences: 1" in result

    def test_multiline(self):
        result = text_stats("Line 1\nLine 2\nLine 3")
        assert "Lines: 3" in result

    def test_empty(self):
        result = text_stats("")
        assert "Words: 0" in result

    def test_paragraphs(self):
        result = text_stats("Para one.\n\nPara two.\n\nPara three.")
        assert "Paragraphs: 3" in result


class TestBase64:
    def test_encode(self):
        assert base64_encode("Hello") == "SGVsbG8="

    def test_decode(self):
        assert base64_decode("SGVsbG8=") == "Hello"

    def test_roundtrip(self):
        original = "The quick brown fox"
        assert base64_decode(base64_encode(original)) == original

    def test_decode_invalid(self):
        assert "not valid base64" in _invalid(base64_decode, "!!!not-base64!!!")
        assert "not valid base64" in _invalid(base64_decode, "SGVsbG8=é")


class TestRegexGuards:
    @pytest.mark.parametrize(
        "pattern",
        [
            r"(a+)+$",
            r"(a*)*",
            r"(?:a+)+",
            r"(a|aa)+$",
            r"(\w+\s?){2,}",
            r"((ab)*)+",
            r"(.*a){20}",
            r"(\d+)\1",
            r"(?P<x>a)(?P=x)",
        ],
    )
    def test_shapes_that_backtrack_exponentially_are_refused(self, pattern):
        assert _invalid(regex_search, "aaaa!", pattern).startswith("pattern refused:")

    @pytest.mark.parametrize(
        ("pattern", "text", "expected"),
        [
            (r"\d{3}-\d{4}", "call 555-1234", "'555-1234'"),
            (r"(\d{3})-(\d{4})", "555-1234", "groups=('555', '1234')"),
            (r"\w+@\w+\.\w+", "mail a@b.pt", "'a@b.pt'"),
            (r"(?:ab)+", "ababx", "'abab'"),
            (r"(?i)hello", "HeLLo", "'HeLLo'"),
            (r"(foo|bar)?baz", "barbaz", "'barbaz'"),
            (r"[(]+x", "((x", "'((x'"),
            (r"\(a+\)+", "(aa))", "'(aa))'"),
            (r"(?P<year>\d{4})-\d{2}", "2026-09", "groups=('2026',)"),
        ],
    )
    def test_ordinary_patterns_still_match(self, pattern, text, expected):
        assert expected in _text(regex_search(text, pattern))

    def test_a_long_pattern_or_text_is_refused(self):
        assert _invalid(regex_search, "a", "a" * 501).startswith("pattern refused:")
        assert _invalid(regex_search, "a" * 20_001, "a").startswith("text refused:")


class TestPolynomialBacktracking:
    """A match runs in a child process with a time budget: a pattern whose quantifiers backtrack
    polynomially (``a*a*b``: n³ steps) is refused when the budget runs out, and this process,
    whose GIL the engine would hold, runs on meanwhile."""

    def test_a_star_a_star_b_on_20000_characters_is_refused_within_the_budget(self) -> None:
        # In a child with a hard deadline: were the match to run here, it would hold the suite
        # for about 18 minutes.
        code = (
            "import time\n"
            "from ai_arch_toolkit.core import ToolFailure\n"
            "from ai_arch_toolkit.toolkit.tools._text import regex_search\n"
            "started = time.monotonic()\n"
            "try:\n"
            "    regex_search('a' * 20000, 'a*a*b')\n"
            "except ToolFailure as failure:\n"
            "    print(failure.error.type, round(time.monotonic() - started, 1))\n"
            "    print(failure.error.message)\n"
        )
        done = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=30, check=False
        )

        kind, took = done.stdout.split("\n", 1)[0].split()
        assert kind == "validation_error", done.stdout + done.stderr
        assert float(took) < text_tools._MATCH_S + 3
        assert f"took longer than {text_tools._MATCH_S:g}s" in done.stdout

    def test_this_process_runs_on_while_the_match_does(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(text_tools, "_MATCH_S", 0.5)
        ticks: list[float] = []
        stop = threading.Event()

        def tick() -> None:
            while not stop.is_set():
                ticks.append(time.monotonic())
                time.sleep(0.01)

        ticker = threading.Thread(target=tick, daemon=True)
        ticker.start()
        started = time.monotonic()
        try:
            message = _invalid(regex_search, "a" * 3000, "a*a*b")  # about 4 s in the engine
        finally:
            stop.set()
            ticker.join()

        assert time.monotonic() - started < 2
        assert message.startswith("pattern refused:")
        assert sum(1 for at in ticks if at >= started) >= 10

    @pytest.mark.parametrize(
        ("pattern", "text"),
        [
            (r"(a)|(b)", "abc"),
            (r"(?P<word>\w+)(\s)?", "héllo wörld 😀 x"),
            (r"", "ab"),
            (r"(?=(a*))a", "aaa"),
        ],
    )
    def test_the_child_finds_what_re_finds_here(self, pattern: str, text: str) -> None:
        expected = [
            f"  [{m.start()}:{m.end()}] {m.group()!r}"
            + (f" groups={m.groups()}" if m.groups() else "")
            for m in re.finditer(pattern, text)
        ]

        assert _text(regex_search(text, pattern)).split("\n")[1:] == expected

    def test_a_program_with_no_python_to_start_is_an_upstream_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "frozen", True, raising=False)

        with pytest.raises(ToolFailure) as caught:
            regex_search("abc", "b")

        assert caught.value.error.type == "upstream"
        assert "sys.executable" in caught.value.error.message


@pytest.mark.parametrize("pattern", ["a{4294967296}", "(" * 2000 + ")" * 2000])
def test_a_pattern_the_engine_cannot_compile_is_a_validation_error(pattern):
    with pytest.raises(ToolFailure) as caught:
        regex_search("aaa", pattern)

    assert caught.value.error.type == "validation_error"
