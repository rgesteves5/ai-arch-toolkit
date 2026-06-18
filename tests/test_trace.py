"""Tests for Trace and StepTrace."""

from __future__ import annotations

import json

import pytest

from ai_arch_toolkit.core._response import Usage
from ai_arch_toolkit.core._tools._result import ToolResult
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

    def test_to_dict_redacts_sensitive_payloads_by_default(self) -> None:
        st = StepTrace(
            name="secret_step",
            input_state={
                "messages": [
                    {
                        "role": "user",
                        "content": "OPENAI_API_KEY=sk-testsecret1234567890",
                    }
                ],
                "headers": {"authorization": "Bearer abc.def.ghi"},
            },
            output_result={
                "value": "postgresql://user:pass@example.com/db",
                "artifacts": {
                    "tool_results": [
                        ToolResult.failure(
                            "runtime_error",
                            "password=supersecret",
                            details={"api_key": "sk-detailsecret1234567890"},
                        )
                    ]
                },
            },
            error=("-----BEGIN PRIVATE KEY-----\nabc123\n-----END PRIVATE KEY-----"),
        )

        payload = json.dumps(st.to_dict())

        assert "sk-testsecret1234567890" not in payload
        assert "abc.def.ghi" not in payload
        assert "user:pass@example.com" not in payload
        assert "supersecret" not in payload
        assert "sk-detailsecret1234567890" not in payload
        assert "BEGIN PRIVATE KEY" not in payload
        assert "[REDACTED]" in payload

    def test_full_debug_requires_explicit_opt_in(self) -> None:
        st = StepTrace(
            name="debug_step",
            input_state={"api_key": "sk-fullsecret1234567890"},
        )

        default_payload = json.dumps(st.to_dict())
        full_payload = json.dumps(st.to_dict(trace_mode="full_debug"))

        assert "sk-fullsecret1234567890" not in default_payload
        assert "sk-fullsecret1234567890" in full_payload

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

    def test_trace_to_dict_redacts_initial_state_by_default(self) -> None:
        trace = Trace(
            flow_name="secret_flow",
            initial_state={
                "operational": {
                    "env": "ANTHROPIC_API_KEY=sk-ant-secret1234567890",
                    "authorization": "Bearer secret-token",
                }
            },
            steps=(
                StepTrace(
                    name="provider_error",
                    error="Provider failed with token=secret-token",
                ),
            ),
        )

        payload = trace.to_dict()
        serialized = json.dumps(payload)

        assert payload["trace_mode"] == "redacted"
        assert "sk-ant-secret1234567890" not in serialized
        assert "secret-token" not in serialized
        assert "[REDACTED]" in serialized

    def test_metadata_only_omits_payloads(self) -> None:
        trace = Trace(
            flow_name="metadata",
            initial_state={"operational": {"password": "secret"}},
            steps=(StepTrace(name="step", input_state={"token": "secret"}),),
        )

        payload = trace.to_dict(trace_mode="metadata_only")

        assert payload["trace_mode"] == "metadata_only"
        assert payload["initial_state"] == {}
        assert payload["steps"][0]["input_state"] == {}
        assert payload["steps"][0]["output_result"] == {}
