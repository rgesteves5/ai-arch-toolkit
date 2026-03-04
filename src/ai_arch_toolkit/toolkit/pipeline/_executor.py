"""PhaseExecutor Protocol and LocalExecutor implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ai_arch_toolkit.toolkit.pipeline._types import PhaseFn, PhaseResult, PipelineContext


@runtime_checkable
class PhaseExecutor(Protocol):
    """Protocol for executing a pipeline phase."""

    async def execute(self, name: str, fn: PhaseFn, ctx: PipelineContext) -> PhaseResult: ...


class LocalExecutor:
    """Executes phases directly in the current process."""

    async def execute(self, name: str, fn: PhaseFn, ctx: PipelineContext) -> PhaseResult:
        return await fn(ctx)
