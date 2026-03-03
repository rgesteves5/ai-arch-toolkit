"""Pipeline — sequential phase execution with context accumulation."""

from __future__ import annotations

from ai_arch_toolkit.toolkit.pipeline._executor import LocalExecutor, PhaseExecutor
from ai_arch_toolkit.toolkit.pipeline._pipeline import Pipeline, run_phase, run_phases
from ai_arch_toolkit.toolkit.pipeline._types import (
    PhaseFn,
    PhaseResult,
    PhaseStatus,
    PipelineContext,
    PipelineResult,
)

__all__ = [
    "LocalExecutor",
    "PhaseExecutor",
    "PhaseFn",
    "PhaseResult",
    "PhaseStatus",
    "Pipeline",
    "PipelineContext",
    "PipelineResult",
    "run_phase",
    "run_phases",
]
