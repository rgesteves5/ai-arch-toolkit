"""Tests for PhaseExecutor Protocol and LocalExecutor."""

from __future__ import annotations

from ai_arch_toolkit.toolkit.pipeline import (
    LocalExecutor,
    PhaseExecutor,
    PhaseResult,
    PipelineContext,
)


class TestLocalExecutor:
    async def test_success(self):
        async def my_phase(ctx: PipelineContext) -> PhaseResult:
            return PhaseResult.ok(answer=42)

        exe = LocalExecutor()
        result = await exe.execute("my_phase", my_phase, PipelineContext())
        assert result.is_ok
        assert result.artifacts["answer"] == 42

    async def test_exception_propagates(self):
        """LocalExecutor does NOT catch exceptions — run_phase does."""

        async def bad_phase(ctx: PipelineContext) -> PhaseResult:
            msg = "boom"
            raise RuntimeError(msg)

        exe = LocalExecutor()
        try:
            await exe.execute("bad_phase", bad_phase, PipelineContext())
            assert False, "should have raised"  # noqa: B011
        except RuntimeError:
            pass

    def test_protocol_check(self):
        assert isinstance(LocalExecutor(), PhaseExecutor)
