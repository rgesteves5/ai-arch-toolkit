"""Pipeline — sequential phase executor with iter/run/run_from."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from typing import Any

from ai_arch_toolkit.toolkit.pipeline._executor import LocalExecutor, PhaseExecutor
from ai_arch_toolkit.toolkit.pipeline._types import (
    _STATUS_RANK,
    PhaseFn,
    PhaseResult,
    PipelineContext,
    PipelineResult,
)

_default_executor = LocalExecutor()


async def run_phase(
    fn: PhaseFn,
    ctx: PipelineContext,
    *,
    executor: PhaseExecutor | None = None,
) -> PhaseResult:
    """Execute a single phase function, handling errors and provenance."""
    name = fn.__name__
    exe = executor or _default_executor
    t0 = time.monotonic()

    try:
        result = await exe.execute(name, fn, ctx)
    except Exception as exc:
        elapsed = time.monotonic() - t0
        result = PhaseResult.failed(str(exc), phase=name)
        result = replace(result, duration=elapsed)
        ctx._record_phase(name, result)
        return result

    elapsed = time.monotonic() - t0

    # Auto-fill phase name and duration
    updates: dict[str, Any] = {}
    if not result.phase:
        updates["phase"] = name
    if result.duration == 0.0:
        updates["duration"] = elapsed
    if updates:
        result = replace(result, **updates)

    # Merge artifacts on ok/partial
    if result.status in ("ok", "partial"):
        ctx.merge(result.artifacts, phase=name)

    ctx._record_phase(name, result)
    return result


class Pipeline:
    """Sequential phase pipeline with iter/run/run_from."""

    __slots__ = ("_phases", "name")

    def __init__(self, *phases: PhaseFn, name: str = "pipeline") -> None:
        self._phases = phases
        self.name = name

    @property
    def phase_names(self) -> list[str]:
        return [fn.__name__ for fn in self._phases]

    def __len__(self) -> int:
        return len(self._phases)

    async def iter(
        self,
        ctx: PipelineContext,
        *,
        stop_on_failure: bool = True,
        stop_on_partial: bool = False,
        executor: PhaseExecutor | None = None,
    ) -> AsyncIterator[tuple[str, PhaseResult]]:
        """Async generator yielding (phase_name, PhaseResult) per phase."""
        names = self.phase_names
        if len(names) != len(set(names)):
            seen: set[str] = set()
            dupes: list[str] = []
            for n in names:
                if n in seen:
                    dupes.append(n)
                seen.add(n)
            msg = f"Duplicate phase names: {dupes}"
            raise ValueError(msg)

        for i, fn in enumerate(self._phases):
            result = await run_phase(fn, ctx, executor=executor)
            yield (fn.__name__, result)

            should_stop = (stop_on_failure and result.is_failure) or (
                stop_on_partial and result.is_partial
            )
            if should_stop:
                # Record remaining as skipped
                for remaining_fn in self._phases[i + 1 :]:
                    rname = remaining_fn.__name__
                    skipped = PhaseResult.skipped(phase=rname, reason="pipeline stopped")
                    ctx._record_phase(rname, skipped)
                return

    async def run(
        self,
        ctx: PipelineContext | None = None,
        *,
        stop_on_failure: bool = True,
        stop_on_partial: bool = False,
        executor: PhaseExecutor | None = None,
    ) -> PipelineResult:
        """Run all phases, returning an aggregated PipelineResult."""
        if ctx is None:
            ctx = PipelineContext()

        t0 = time.monotonic()

        async for _name, _result in self.iter(
            ctx,
            stop_on_failure=stop_on_failure,
            stop_on_partial=stop_on_partial,
            executor=executor,
        ):
            pass

        elapsed = time.monotonic() - t0
        return _aggregate(ctx, elapsed)

    async def run_from(
        self,
        phase_name: str,
        ctx: PipelineContext,
        **kw: Any,
    ) -> PipelineResult:
        """Resume pipeline from a named phase, skipping earlier ones."""
        names = self.phase_names
        if phase_name not in names:
            msg = f"Unknown phase {phase_name!r}. Available: {names}"
            raise ValueError(msg)

        start_idx = names.index(phase_name)

        # Record earlier phases as skipped
        for fn in self._phases[:start_idx]:
            skipped = PhaseResult.skipped(phase=fn.__name__, reason="resumed past")
            ctx._record_phase(fn.__name__, skipped)

        # Build a sub-pipeline from start_idx onward
        sub = Pipeline(*self._phases[start_idx:], name=self.name)
        return await sub.run(ctx, **kw)


async def run_phases(
    phases: Sequence[PhaseFn],
    ctx: PipelineContext | None = None,
    **kw: Any,
) -> PipelineResult:
    """Convenience: run a sequence of phase functions as a pipeline."""
    return await Pipeline(*phases).run(ctx, **kw)


def _aggregate(ctx: PipelineContext, duration: float) -> PipelineResult:
    """Compute aggregated status from ctx.phase_results (insertion-ordered)."""
    all_results = list(ctx._phase_results.values())

    non_skipped = [r for r in all_results if not r.is_skipped]
    if not non_skipped:
        status = "skipped" if all_results else "ok"
    else:
        worst = max(_STATUS_RANK.get(r.status, 0) for r in non_skipped)
        status = {v: k for k, v in _STATUS_RANK.items()}[worst]

    return PipelineResult(
        status=status,
        phases=tuple(all_results),
        context=ctx,
        duration=duration,
        total_token_usage=ctx.total_token_usage,
        total_warnings=ctx.total_warnings,
    )
