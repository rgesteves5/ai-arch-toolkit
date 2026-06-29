"""Per-span projection: operations roll up into their span and ancestors; siblings isolated."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._metering._cost import Cost
from ai_arch_toolkit.core._metering._money import Money
from ai_arch_toolkit.core._metering._operation import OperationRequest
from ai_arch_toolkit.core._metering._store import MeterStore
from ai_arch_toolkit.core._response import Usage


def store() -> MeterStore:
    return MeterStore(clock=lambda: 0.0)  # constant clock -> deterministic elapsed for == checks


def llm_in(span: str) -> OperationRequest:
    return OperationRequest(kind="llm", parent_span_id=span)


def settle(op, n_input: int = 10, cost_usd: float = 0.01) -> None:
    op.mark_started()
    op.settle(usage=Usage(input_tokens=n_input), cost=Cost.known(Money.from_usd(cost_usd)))


def test_for_span_root_equals_snapshot():
    s = store()
    settle(s.open(llm_in(s.run_span_id), None))
    assert s.for_span(s.run_span_id) == s.snapshot()


def test_op_rolls_into_its_span_and_every_ancestor():
    s = store()
    step = s.open_span("step")
    settle(s.open(llm_in(step), None), n_input=10, cost_usd=0.01)
    for view in (s.for_span(step), s.snapshot()):  # span itself + ancestor root
        assert view.llm_calls == 1 and view.input_tokens == 10
        assert view.cost == Money.from_usd(0.01)


def test_sibling_spans_are_isolated():
    s = store()
    a = s.open_span("step")
    b = s.open_span("step")
    settle(s.open(llm_in(a), None))
    assert s.for_span(a).llm_calls == 1
    assert s.for_span(b).llm_calls == 0
    assert s.snapshot().llm_calls == 1  # root sees it


def test_nested_spans_roll_up_to_each_ancestor():
    s = store()
    outer = s.open_span("step")
    inner = s.open_span("tool", outer)
    settle(s.open(llm_in(inner), None))
    settle(s.open(llm_in(outer), None))  # sibling of `inner`, under `outer`
    assert s.for_span(inner).llm_calls == 1  # only its own op
    assert s.for_span(outer).llm_calls == 2  # both (inner is a descendant)
    assert s.snapshot().llm_calls == 2


def test_outstanding_rolls_up_and_releases():
    s = store()
    step = s.open_span("step")
    op = s.open(llm_in(step), None)  # reserved, not started
    assert s.for_span(step).out_llm_calls == 1 and s.snapshot().out_llm_calls == 1
    op.abort()
    assert s.for_span(step).out_llm_calls == 0 and s.snapshot().out_llm_calls == 0


def test_open_with_unknown_parent_span_raises():
    s = store()
    with pytest.raises(ValueError, match="unknown parent span"):
        s.open(llm_in("ghost"), None)


def test_open_span_unknown_parent_raises():
    s = store()
    with pytest.raises(ValueError, match="unknown parent span"):
        s.open_span("step", "ghost")


def test_for_span_unknown_raises():
    s = store()
    with pytest.raises(ValueError, match="unknown span"):
        s.for_span("ghost")
