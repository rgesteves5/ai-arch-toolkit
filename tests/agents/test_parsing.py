"""Tests for shared parsing helpers."""

from __future__ import annotations

from ai_arch_toolkit.agents._parsing import parse_numbered_items, parse_score


def test_parse_numbered_items_basic():
    """Parses numbered items from standard format."""
    text = "1. First item\n2. Second item\n3. Third item"
    result = parse_numbered_items(text, max_items=3)
    assert result == ["First item", "Second item", "Third item"]


def test_parse_numbered_items_max_items():
    """Respects max_items limit."""
    text = "1. A\n2. B\n3. C\n4. D"
    result = parse_numbered_items(text, max_items=2)
    assert len(result) == 2
    assert result == ["A", "B"]


def test_parse_numbered_items_single_unnumbered():
    """Single unnumbered block is returned as-is."""
    text = "Approach A\nApproach B"
    result = parse_numbered_items(text, max_items=5)
    assert len(result) == 1
    assert "Approach A" in result[0]


def test_parse_numbered_items_empty():
    """Returns empty list for empty or whitespace input."""
    assert parse_numbered_items("", max_items=3) == []
    assert parse_numbered_items("   \n  ", max_items=3) == []


def test_parse_score_basic():
    """Parses a simple score."""
    assert parse_score("0.8") == 0.8
    assert parse_score("The score is 0.65") == 0.65


def test_parse_score_clamping():
    """Clamps values to [0.0, 1.0]."""
    assert parse_score("2.5") == 1.0
    assert parse_score("0.0") == 0.0


def test_parse_score_default():
    """Returns default when no number found."""
    assert parse_score("no number here") == 0.5
    assert parse_score("no number here", default=0.3) == 0.3
