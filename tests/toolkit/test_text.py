"""Tests for toolkit/tools/_text.py."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._tools._result import ToolFailure
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


class TestRegexSearch:
    def test_basic_match(self):
        result = regex_search("foo123bar", r"\d+")
        assert "1 match" in result
        assert "'123'" in result

    def test_multiple_matches(self):
        result = regex_search("a1 b2 c3", r"\d")
        assert "3 match" in result

    def test_no_matches(self):
        result = regex_search("hello", r"\d+")
        assert "No matches" in result

    def test_groups(self):
        result = regex_search("2026-02-27", r"(\d{4})-(\d{2})-(\d{2})")
        assert "groups=" in result

    def test_invalid_regex(self):
        assert "invalid regex" in _invalid(regex_search, "text", r"[invalid")


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
        assert expected in regex_search(text, pattern)

    def test_a_long_pattern_or_text_is_refused(self):
        assert _invalid(regex_search, "a", "a" * 501).startswith("pattern refused:")
        assert _invalid(regex_search, "a" * 20_001, "a").startswith("text refused:")

    def test_matches_stop_at_a_thousand(self):
        result = regex_search("a" * 5000, "a")

        assert result.startswith("1000 match(es) shown")
        assert result.count("\n") == 1000  # the header, then one line a match


@pytest.mark.parametrize("pattern", ["a{4294967296}", "(" * 2000 + ")" * 2000])
def test_a_pattern_the_engine_cannot_compile_is_a_validation_error(pattern):
    with pytest.raises(ToolFailure) as caught:
        regex_search("aaa", pattern)

    assert caught.value.error.type == "validation_error"
