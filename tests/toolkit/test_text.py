"""Tests for toolkit/tools/_text.py."""

from __future__ import annotations

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
