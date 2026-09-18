"""Tests for toolkit/tools/_text.py."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.toolkit.tools._text import (
    base64_decode,
    base64_encode,
    regex_search,
    text_stats,
)


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
        result = regex_search("text", r"[invalid")
        assert "Invalid regex" in result


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
        result = base64_decode("!!!not-base64!!!")
        assert "error" in result.lower() or "Error" in result


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
        assert regex_search("aaaa!", pattern).startswith("Pattern refused:")

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
        assert regex_search("a", "a" * 501).startswith("Pattern refused:")
        assert regex_search("a" * 20_001, "a").startswith("Text refused:")

    def test_matches_stop_at_a_thousand(self):
        result = regex_search("a" * 5000, "a")

        assert result.startswith("1000 match(es) shown")
        assert result.count("\n") == 1000  # the header, then one line a match
