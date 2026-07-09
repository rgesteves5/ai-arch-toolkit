"""Concurrency controls: the global inference cap (`inference_limit`) and the
per-flow fan-out cap (`Flow(max_parallelism=...)`)."""

from __future__ import annotations

import asyncio

import pytest

from ai_arch_toolkit import LLM, inference_limit
from ai_arch_toolkit.core._response import Response, Usage


class _TrackingProvider:
    """Records the peak number of concurrent in-flight complete() calls."""

    def __init__(self, delay: float = 0.02) -> None:
        self._delay = delay
        self.live = 0
        self.peak = 0

    async def complete(self, *a, **k) -> Response:
        self.live += 1
        self.peak = max(self.peak, self.live)
        try:
            await asyncio.sleep(self._delay)
        finally:
            self.live -= 1
        return Response(text="ok", usage=Usage(input_tokens=1))


def _llm() -> tuple[LLM, _TrackingProvider]:
    prov = _TrackingProvider()
    llm = LLM("claude-sonnet-4-6", api_key="test")
    llm._provider = prov  # type: ignore[assignment]
    return llm, prov


# ── B: global inference cap ──────────────────────────────────────────────────


async def test_inference_limit_caps_concurrent_calls():
    llm, prov = _llm()
    with inference_limit(2):
        await asyncio.gather(*[llm.complete("hi") for _ in range(8)])
    assert prov.peak == 2


async def test_no_limit_is_unbounded():
    llm, prov = _llm()
    await asyncio.gather(*[llm.complete("hi") for _ in range(8)])
    assert prov.peak == 8


async def test_nested_inference_limit_innermost_wins():
    llm, prov = _llm()
    with inference_limit(4), inference_limit(1):
        await asyncio.gather(*[llm.complete("hi") for _ in range(5)])
    assert prov.peak == 1


async def test_the_cap_is_global_across_nested_gathers():
    # Two layers of fan-out sharing ONE ambient limit — total in-flight never exceeds it,
    # no matter the nesting (the leaf-level slot cannot deadlock).
    llm, prov = _llm()

    async def branch() -> None:
        await asyncio.gather(*[llm.complete("hi") for _ in range(3)])

    with inference_limit(2):
        await asyncio.gather(*[branch() for _ in range(3)])  # 3 x 3 = 9 calls, cap 2
    assert prov.peak == 2


async def test_inference_limit_rejects_non_positive():
    with pytest.raises(ValueError), inference_limit(0):
        pass


# ── A: per-flow fan-out cap ──────────────────────────────────────────────────

from ai_arch_toolkit.core import Result, State, Step  # noqa: E402
from ai_arch_toolkit.toolkit.flow import Flow, FlowStep  # noqa: E402


def _peak_tracker():
    state = {"live": 0, "peak": 0}

    async def work(_snap) -> Result:
        state["live"] += 1
        state["peak"] = max(state["peak"], state["live"])
        try:
            await asyncio.sleep(0.02)
        finally:
            state["live"] -= 1
        return Result(value="ok")

    return state, work


def _fan_out(work, n: int, *, max_parallelism: int | None) -> Flow:
    # n independent steps (run in parallel) + a join that depends on all of them,
    # which makes the flow a DAG so the parallel branch is exercised.
    async def join(_snap) -> Result:
        return Result(value="done")

    steps = [Step(name=f"p{i}", fn=work) for i in range(n)]
    return Flow(
        *steps,
        FlowStep(step=Step(name="join", fn=join), after=tuple(f"p{i}" for i in range(n))),
        max_parallelism=max_parallelism,
    )


async def test_flow_max_parallelism_caps_the_fan_out():
    tracker, work = _peak_tracker()
    await _fan_out(work, 6, max_parallelism=2).run(State(operational={}))
    assert tracker["peak"] == 2


async def test_flow_without_max_parallelism_is_unbounded():
    tracker, work = _peak_tracker()
    await _fan_out(work, 6, max_parallelism=None).run(State(operational={}))
    assert tracker["peak"] == 6


async def test_nested_flows_with_max_parallelism_do_not_deadlock():
    # A parallel flow whose steps are themselves parallel sub-flows, all capped at 1.
    # Per-flow semaphores are independent, so this completes (a global cap would deadlock).
    _tracker, work = _peak_tracker()

    def sub_flow_step(name: str) -> Step:
        sub = _fan_out(work, 3, max_parallelism=1)
        return Step(name=name, fn=sub.as_step().fn)

    async def join(_snap) -> Result:
        return Result(value="done")

    outer = Flow(
        sub_flow_step("s0"),
        sub_flow_step("s1"),
        FlowStep(step=Step(name="join", fn=join), after=("s0", "s1")),
        max_parallelism=1,
    )
    result = await asyncio.wait_for(outer.run(State(operational={})), timeout=5.0)
    assert result is not None  # completed — no deadlock


async def test_flow_max_parallelism_rejects_non_positive():
    with pytest.raises(ValueError):
        Flow(Step(name="a", fn=lambda s: Result(value=1)), max_parallelism=0)
