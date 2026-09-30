"""Tests for the helpers the flow factories share."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.toolkit.agents.flows._common import parse_score


@pytest.mark.parametrize(
    ("reply", "score"),
    [
        ("0.7", 0.7),
        ("Score (0.0-1.0): 0.8", 0.8),
        ("Step 2 looks strong: 0.9", 0.9),
        ("Score: 1", 1.0),
        ("Score: .25", 0.25),
        ("The step is sound.", 0.5),  # no number: undecided
        ("8/10", 0.5),  # no number in [0, 1]: undecided
    ],
)
def test_a_score_is_the_last_number_in_range_of_the_reply(reply: str, score: float) -> None:
    assert parse_score(reply) == score
