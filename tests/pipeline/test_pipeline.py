"""Tests for Pipeline, run_phase, run_phases."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.toolkit.pipeline import (
    LocalExecutor,
    PhaseResult,
    Pipeline,
    PipelineContext,
    run_phase,
    run_phases,
)

# ---- Helpers ----


async def phase_a(ctx: PipelineContext) -> PhaseResult:
    return PhaseResult.ok(output_a="hello")


async def phase_b(ctx: PipelineContext) -> PhaseResult:
    return PhaseResult.ok(
        output_b=ctx["output_a"] + " world",
        token_usage={"input": 10, "output": 5},
        warnings=["w1"],
    )


async def phase_fail(ctx: PipelineContext) -> PhaseResult:
    msg = "intentional error"
    raise ValueError(msg)


async def phase_partial(ctx: PipelineContext) -> PhaseResult:
    return PhaseResult.partial(error="incomplete", partial_data="some")


async def phase_c(ctx: PipelineContext) -> PhaseResult:
    return PhaseResult.ok(output_c="final")


# ---- run_phase ----


class TestRunPhase:
    async def test_success_auto_name_and_timing(self):
        ctx = PipelineContext()
        result = await run_phase(phase_a, ctx)
        assert result.is_ok
        assert result.phase == "phase_a"
        assert result.duration > 0
        assert ctx["output_a"] == "hello"

    async def test_exception_caught(self):
        ctx = PipelineContext()
        result = await run_phase(phase_fail, ctx)
        assert result.is_failure
        assert "intentional error" in result.error

    async def test_base_exception_propagates(self):
        async def kb_phase(ctx: PipelineContext) -> PhaseResult:
            raise KeyboardInterrupt

        ctx = PipelineContext()
        with pytest.raises(KeyboardInterrupt):
            await run_phase(kb_phase, ctx)

    async def test_duration_preserved_if_executor_set_it(self):
        class TimedExecutor:
            async def execute(self, name, fn, ctx):
                result = await fn(ctx)
                from dataclasses import replace

                return replace(result, duration=99.0)

        ctx = PipelineContext()
        result = await run_phase(phase_a, ctx, executor=TimedExecutor())
        assert result.duration == 99.0

    async def test_artifacts_merged_on_ok(self):
        ctx = PipelineContext()
        await run_phase(phase_a, ctx)
        assert ctx["output_a"] == "hello"
        assert ctx.produced_by("output_a") == "phase_a"

    async def test_artifacts_not_merged_on_failed(self):
        ctx = PipelineContext()
        await run_phase(phase_fail, ctx)
        assert "output" not in ctx

    async def test_provenance_tracked(self):
        ctx = PipelineContext()
        await run_phase(phase_a, ctx)
        await run_phase(phase_b, ctx)
        assert ctx.produced_by("output_a") == "phase_a"
        assert ctx.produced_by("output_b") == "phase_b"

    async def test_artifacts_merged_on_partial(self):
        ctx = PipelineContext()
        await run_phase(phase_partial, ctx)
        assert ctx["partial_data"] == "some"
        assert ctx.produced_by("partial_data") == "phase_partial"

    async def test_with_custom_executor(self):
        ctx = PipelineContext()
        result = await run_phase(phase_a, ctx, executor=LocalExecutor())
        assert result.is_ok


# ---- Pipeline.iter() ----


class TestPipelineIter:
    async def test_yields_name_result_per_phase(self):
        pipe = Pipeline(phase_a, phase_b)
        ctx = PipelineContext()
        results = []
        async for name, result in pipe.iter(ctx):
            results.append((name, result))
        assert len(results) == 2
        assert results[0][0] == "phase_a"
        assert results[1][0] == "phase_b"

    async def test_stops_on_failure(self):
        pipe = Pipeline(phase_a, phase_fail, phase_c)
        ctx = PipelineContext()
        results = []
        async for name, _result in pipe.iter(ctx, stop_on_failure=True):
            results.append(name)
        assert results == ["phase_a", "phase_fail"]
        # phase_c should be recorded as skipped
        assert "phase_c" in ctx.phase_results
        assert ctx.phase_results["phase_c"].is_skipped

    async def test_records_remaining_as_skipped(self):
        pipe = Pipeline(phase_fail, phase_a, phase_b)
        ctx = PipelineContext()
        async for _ in pipe.iter(ctx, stop_on_failure=True):
            pass
        assert ctx.phase_results["phase_a"].is_skipped
        assert ctx.phase_results["phase_b"].is_skipped

    async def test_break_early(self):
        pipe = Pipeline(phase_a, phase_b, phase_c)
        ctx = PipelineContext()
        async for name, _result in pipe.iter(ctx):
            if name == "phase_a":
                break
        # Only phase_a should have run
        assert "output_a" in ctx
        assert "output_b" not in ctx

    async def test_stop_on_partial(self):
        pipe = Pipeline(phase_a, phase_partial, phase_c)
        ctx = PipelineContext()
        results = []
        async for name, _result in pipe.iter(ctx, stop_on_partial=True):
            results.append(name)
        assert results == ["phase_a", "phase_partial"]

    async def test_duplicate_names_raises(self):
        pipe = Pipeline(phase_a, phase_a)
        ctx = PipelineContext()
        with pytest.raises(ValueError, match="Duplicate phase names"):
            async for _ in pipe.iter(ctx):
                pass


# ---- Pipeline.run() ----


class TestPipelineRun:
    async def test_aggregates_results(self):
        pipe = Pipeline(phase_a, phase_b)
        result = await pipe.run()
        assert result.is_ok
        assert len(result.phases) == 2
        assert result.context is not None
        assert result.context["output_b"] == "hello world"

    async def test_returns_pipeline_result_with_tokens_and_warnings(self):
        pipe = Pipeline(phase_a, phase_b)
        result = await pipe.run()
        assert result.total_token_usage == {"input": 10, "output": 5}
        assert result.total_warnings == ["w1"]
        assert result.duration > 0

    async def test_creates_context_if_none(self):
        pipe = Pipeline(phase_a)
        result = await pipe.run()
        assert result.context is not None
        assert result.context["output_a"] == "hello"


# ---- Pipeline.run_from() ----


class TestPipelineRunFrom:
    async def test_resumes_from_named_phase(self):
        pipe = Pipeline(phase_a, phase_b, phase_c)
        # Pre-populate ctx as if phase_a already ran
        ctx = PipelineContext({"output_a": "hello"})
        result = await pipe.run_from("phase_b", ctx)
        assert result.context["output_b"] == "hello world"
        assert result.context["output_c"] == "final"

    async def test_records_earlier_as_skipped(self):
        pipe = Pipeline(phase_a, phase_b, phase_c)
        ctx = PipelineContext({"output_a": "hello"})
        await pipe.run_from("phase_b", ctx)
        assert ctx.phase_results["phase_a"].is_skipped

    async def test_unknown_phase_raises(self):
        pipe = Pipeline(phase_a, phase_b)
        ctx = PipelineContext()
        with pytest.raises(ValueError, match="Unknown phase"):
            await pipe.run_from("nonexistent", ctx)

    async def test_phases_ordered_correctly(self):
        """Skipped phases should appear before executed ones in result.phases."""
        pipe = Pipeline(phase_a, phase_b, phase_c)
        ctx = PipelineContext({"output_a": "hello"})
        result = await pipe.run_from("phase_b", ctx)
        phase_names = [p.phase for p in result.phases]
        assert phase_names == ["phase_a", "phase_b", "phase_c"]
        assert result.phases[0].is_skipped


# ---- run_phases() ----


class TestRunPhases:
    async def test_convenience_wrapper(self):
        result = await run_phases([phase_a, phase_b])
        assert result.is_ok
        assert result.context["output_b"] == "hello world"


# ---- Pipeline properties ----


class TestPipelineProperties:
    def test_phase_names(self):
        pipe = Pipeline(phase_a, phase_b)
        assert pipe.phase_names == ["phase_a", "phase_b"]

    def test_len(self):
        pipe = Pipeline(phase_a, phase_b, phase_c)
        assert len(pipe) == 3
