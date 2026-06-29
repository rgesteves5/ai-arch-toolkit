"""OperationFacts: pure facts + the count-vs-kind invariants."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._metering._operation import OperationFacts


def test_llm_facts():
    f = OperationFacts(kind="llm", parent_span_id="run", mode="complete", model="claude-x")
    assert f.kind == "llm" and f.count == 1 and f.mode == "complete"


def test_stream_is_llm_mode():
    f = OperationFacts(kind="llm", parent_span_id="run", mode="stream")
    assert f.kind == "llm" and f.mode == "stream"


def test_custom_has_zero_count():
    f = OperationFacts(kind="custom", parent_span_id="run", count=0)
    assert f.count == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "llm", "parent_span_id": "r", "count": 0},  # llm needs count >= 1
        {"kind": "tool", "parent_span_id": "r", "count": 0},  # tool needs count >= 1
        {"kind": "custom", "parent_span_id": "r", "count": 1},  # custom must be 0
        {"kind": "tool", "parent_span_id": "r", "mode": "complete"},  # mode only for llm
    ],
)
def test_count_and_mode_invariants(kwargs):
    with pytest.raises(ValueError):
        OperationFacts(**kwargs)


def test_metadata_default_is_empty():
    f = OperationFacts(kind="tool", parent_span_id="step:1")
    assert f.metadata == {}
