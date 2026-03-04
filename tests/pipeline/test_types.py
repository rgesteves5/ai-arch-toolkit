"""Tests for pipeline types — PipelineContext, PhaseResult, PipelineResult."""

from __future__ import annotations

import warnings

import pytest

from ai_arch_toolkit.toolkit.pipeline import PhaseResult, PipelineContext, PipelineResult

# ---- PhaseResult factory methods ----


class TestPhaseResultFactories:
    def test_ok(self):
        r = PhaseResult.ok(foundation="text", plan="chapters")
        assert r.status == "ok"
        assert r.artifacts == {"foundation": "text", "plan": "chapters"}
        assert r.error is None

    def test_ok_with_token_usage_and_warnings(self):
        r = PhaseResult.ok(
            token_usage={"input": 100, "output": 50},
            warnings=["used fallback model"],
            data="x",
        )
        assert r.token_usage == {"input": 100, "output": 50}
        assert r.warnings == ["used fallback model"]
        assert r.artifacts == {"data": "x"}

    def test_partial(self):
        r = PhaseResult.partial(error="incomplete", data="partial_data")
        assert r.status == "partial"
        assert r.error == "incomplete"
        assert r.artifacts == {"data": "partial_data"}

    def test_failed(self):
        r = PhaseResult.failed("boom")
        assert r.status == "failed"
        assert r.error == "boom"
        assert r.artifacts == {}

    def test_skipped(self):
        r = PhaseResult.skipped(phase="step3", reason="pipeline stopped")
        assert r.status == "skipped"
        assert r.phase == "step3"
        assert r.error == "pipeline stopped"

    def test_skipped_no_reason(self):
        r = PhaseResult.skipped()
        assert r.error is None


class TestPhaseResultProperties:
    def test_is_ok(self):
        assert PhaseResult.ok().is_ok
        assert not PhaseResult.ok().is_failure

    def test_is_partial(self):
        assert PhaseResult.partial().is_partial

    def test_is_failure(self):
        assert PhaseResult.failed("x").is_failure

    def test_is_skipped(self):
        assert PhaseResult.skipped().is_skipped

    def test_has_artifacts(self):
        assert PhaseResult.ok(data="x").has_artifacts
        assert not PhaseResult.ok().has_artifacts


class TestPhaseResultSerialization:
    def test_roundtrip(self):
        r = PhaseResult.ok(
            phase="step1",
            token_usage={"input": 10},
            warnings=["w1"],
            output="data",
        )
        d = r.to_dict()
        r2 = PhaseResult.from_dict(d)
        assert r2.status == r.status
        assert r2.phase == r.phase
        assert r2.artifacts == r.artifacts
        assert r2.token_usage == r.token_usage
        assert r2.warnings == r.warnings


# ---- PipelineContext ----


class TestPhaseResultSerializationSparse:
    def test_to_dict_omits_empty_fields(self):
        r = PhaseResult.ok(phase="s1")
        d = r.to_dict()
        assert "artifacts" not in d
        assert "error" not in d
        assert "token_usage" not in d
        assert "warnings" not in d


class TestPipelineContext:
    def test_dict_access(self):
        ctx = PipelineContext({"a": 1})
        assert ctx["a"] == 1
        ctx["b"] = 2
        assert "b" in ctx
        assert ctx.get("c", 99) == 99

    def test_require_success(self):
        ctx = PipelineContext({"key": "val"})
        assert ctx.require("key") == "val"

    def test_require_missing(self):
        ctx = PipelineContext({"a": 1})
        with pytest.raises(KeyError, match="Required artifact 'missing'"):
            ctx.require("missing")

    def test_merge_with_provenance(self):
        ctx = PipelineContext()
        ctx.merge({"x": 1, "y": 2}, phase="step1")
        assert ctx["x"] == 1
        assert ctx.produced_by("x") == "step1"

    def test_merge_cross_phase_overwrite_warns(self):
        ctx = PipelineContext()
        ctx.merge({"x": 1}, phase="step1")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ctx.merge({"x": 2}, phase="step2")
            assert len(w) == 1
            assert "overwritten" in str(w[0].message)

    def test_merge_same_phase_no_warning(self):
        ctx = PipelineContext()
        ctx.merge({"x": 1}, phase="step1")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ctx.merge({"x": 2}, phase="step1")
            assert len(w) == 0

    def test_metadata_separate_from_data(self):
        ctx = PipelineContext({"a": 1}, metadata={"run_id": "abc"})
        assert ctx.data == {"a": 1}
        assert ctx.metadata == {"run_id": "abc"}

    def test_produced_by_none_for_unknown(self):
        ctx = PipelineContext()
        assert ctx.produced_by("x") is None

    def test_total_token_usage_none_when_empty(self):
        ctx = PipelineContext()
        assert ctx.total_token_usage is None
        # Also None when phases exist but none report token_usage
        ctx._record_phase("s1", PhaseResult.ok(phase="s1"))
        assert ctx.total_token_usage is None

    def test_phase_accumulation(self):
        ctx = PipelineContext()
        r1 = PhaseResult.ok(
            phase="s1",
            token_usage={"input": 10, "output": 5},
            warnings=["w1"],
        )
        r2 = PhaseResult.ok(
            phase="s2",
            token_usage={"input": 20, "output": 10},
            warnings=["w2"],
        )
        ctx._record_phase("s1", r1)
        ctx._record_phase("s2", r2)

        assert ctx.total_token_usage == {"input": 30, "output": 15}
        assert ctx.total_warnings == ["w1", "w2"]

    def test_last_phase_and_result(self):
        ctx = PipelineContext()
        assert ctx.last_phase is None
        assert ctx.last_result is None
        r = PhaseResult.ok(phase="s1")
        ctx._record_phase("s1", r)
        assert ctx.last_phase == "s1"
        assert ctx.last_result is r

    def test_total_duration(self):
        ctx = PipelineContext()
        r1 = PhaseResult(status="ok", phase="s1", duration=1.5)
        r2 = PhaseResult(status="ok", phase="s2", duration=2.5)
        ctx._record_phase("s1", r1)
        ctx._record_phase("s2", r2)
        assert ctx.total_duration == pytest.approx(4.0)

    def test_to_dict_from_dict_roundtrip(self):
        ctx = PipelineContext({"a": 1}, metadata={"run": "x"})
        ctx.merge({"b": 2}, phase="s1")
        ctx._record_phase("s1", PhaseResult.ok(phase="s1", val=2))

        d = ctx.to_dict()
        ctx2 = PipelineContext.from_dict(d)
        assert ctx2["a"] == 1
        assert ctx2["b"] == 2
        assert ctx2.metadata == {"run": "x"}
        assert ctx2.produced_by("b") == "s1"
        assert "s1" in ctx2.phase_results


# ---- PipelineResult ----


class TestPipelineResult:
    def test_status_ranking_ok(self):
        r = PipelineResult(status="ok")
        assert r.is_ok

    def test_failed_phases(self):
        phases = (
            PhaseResult.ok(phase="s1"),
            PhaseResult.failed("err", phase="s2"),
            PhaseResult.ok(phase="s3"),
        )
        r = PipelineResult(status="failed", phases=phases)
        assert r.failed_phases == ["s2"]
        assert r.completed_phases == ["s1", "s3"]

    def test_skipped_phases(self):
        phases = (
            PhaseResult.ok(phase="s1"),
            PhaseResult.skipped(phase="s2"),
        )
        r = PipelineResult(status="ok", phases=phases)
        assert r.skipped_phases == ["s2"]

    def test_all_skipped_status(self):
        """Pipeline with all phases skipped should have status 'skipped'."""
        from ai_arch_toolkit.toolkit.pipeline._pipeline import _aggregate

        ctx = PipelineContext()
        ctx._record_phase("a", PhaseResult.skipped(phase="a"))
        ctx._record_phase("b", PhaseResult.skipped(phase="b"))
        r = _aggregate(ctx, 0.0)
        assert r.status == "skipped"
