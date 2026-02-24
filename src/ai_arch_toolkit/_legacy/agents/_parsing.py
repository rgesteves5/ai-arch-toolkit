"""Shared parsing helpers for agent implementations."""

from __future__ import annotations

import re


def parse_numbered_items(text: str, max_items: int) -> list[str]:
    """Parse numbered items (1. foo\\n2. bar) from LLM output.

    Falls back to splitting by newlines if no numbered format found.
    """
    parts = re.split(r"\n\d+\.\s+", "\n" + text)
    items = [p.strip() for p in parts if p.strip()]
    if items:
        return items[:max_items]
    # Fallback: split by newlines
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    return lines[:max_items]


def parse_score(text: str, default: float = 0.5) -> float:
    """Parse a float score from LLM output, clamped to [0.0, 1.0]."""
    match = re.search(r"(\d+\.?\d*)", text)
    if match:
        try:
            return min(1.0, max(0.0, float(match.group(1))))
        except ValueError:
            pass
    return default
