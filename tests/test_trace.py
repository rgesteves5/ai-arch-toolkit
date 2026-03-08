"""Tests for Trace and StepTrace."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._response import Usage
from ai_arch_toolkit.core._trace import StepTrace, Trace


class TestStepTrace:
    def test_defaults(self) -> None:
        st = StepTrace(name="test")
        assert st.name == "test"
        assert st.attempts == 1
        assert not st.skipped
        assert st.error is None
        assert st.children == ()

    def test_roundtrip(self) -> None:
        st = StepTrace(
            name="step1",
            duration=1.5,
            cost=0.01,
            confidence=0.9,
            usage=Usage(input_tokens=100, output_tokens=50),
            attempts=2,
            policy_decisions=("retry",),
            children=(StepTrace(name="child1"),),
        )
        d = st.to_dict()
        st2 = StepTrace.from_dict(d)
        assert st2.name == "step1"
        assert st2.duration == 1.5
        assert st2.attempts == 2
        assert st2.policy_decisions == ("retry",)
        assert len(st2.children) == 1
        assert st2.children[0].name == "child1"

    def test_skipped(self) -> None:
        st = StepTrace(name="skipped_step", skipped=True, skip_reason="condition not met")
        assert st.skipped
        assert st.skip_reason == "condition not met"


class TestTrace:
    def _make_trace(self) -> Trace:
        return Trace(
            flow_name="test_flow",
            steps=(
                StepTrace(
                    name="step1",
                    cost=0.1,
                    confidence=0.9,
                    usage=Usage(input_tokens=100, output_tokens=50),
                ),
                StepTrace(
                    name="step2",
                    cost=0.2,
                    confidence=0.8,
                    usage=Usage(input_tokens=200, output_tokens=100),
                    children=(
                        StepTrace(
                            name="nested",
                            cost=0.05,
                            confidence=0.95,
                            usage=Usage(input_tokens=10),
                        ),
                    ),
                ),
            ),
            duration=5.0,
        )

    def test_step_lookup(self) -> None:
        trace = self._make_trace()
        s = trace.step("step1")
        assert s is not None
        assert s.name == "step1"

    def test_step_lookup_nested(self) -> None:
        trace = self._make_trace()
        s = trace.step("nested")
        assert s is not None
        assert s.name == "nested"

    def test_step_not_found(self) -> None:
        trace = self._make_trace()
        assert trace.step("nonexistent") is None

    def test_flow_lookup(self) -> None:
        trace = self._make_trace()
        f = trace.flow("step2")
        assert f is not None
        assert f.name == "step2"
        assert len(f.children) == 1

    def test_total_cost(self) -> None:
        trace = self._make_trace()
        assert trace.total_cost == pytest.approx(0.35)

    def test_total_duration(self) -> None:
        trace = self._make_trace()
        assert trace.total_duration == 5.0

    def test_confidence_min(self) -> None:
        trace = self._make_trace()
        assert trace.confidence == pytest.approx(0.8)

    def test_confidence_none_when_no_values(self) -> None:
        trace = Trace(flow_name="empty", steps=())
        assert trace.confidence is None

    def test_total_usage(self) -> None:
        trace = self._make_trace()
        usage = trace.total_usage
        assert usage.input_tokens == 310
        assert usage.output_tokens == 150

    def test_roundtrip(self) -> None:
        trace = self._make_trace()
        d = trace.to_dict()
        t2 = Trace.from_dict(d)
        assert t2.flow_name == "test_flow"
        assert len(t2.steps) == 2
        assert t2.duration == 5.0
