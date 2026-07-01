"""Flow executor — DAG, sequential, and cyclic execution modes."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from ai_arch_toolkit.core._budget import BudgetExceeded, BudgetState
from ai_arch_toolkit.core._metering._scope import MeterScope, bind_meter
from ai_arch_toolkit.core._state import State, StateSnapshot
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.core._step_engine import execute_step
from ai_arch_toolkit.core._trace import StepTrace, Trace
from ai_arch_toolkit.toolkit.budget import BudgetReport
from ai_arch_toolkit.toolkit.flow._flow import Flow, FlowEvent, FlowResult, FlowStep
from ai_arch_toolkit.toolkit.flow._scope import apply_scope


async def execute_flow(flow: Flow, state: State) -> FlowResult:
    """Execute a Flow, choosing DAG or sequential mode."""
    t0 = time.monotonic()
    traces: list[StepTrace] = []
    results: dict[str, Result] = {}
    _ensure_budget(flow, state)
    initial_state = state.to_dict()
    scope = _open_meter_scope(state)

    if flow.is_dag:
        await _execute_dag(flow, state, traces, results)
    else:
        await _execute_sequential(flow, state, traces, results)

    scope.close()
    trace = Trace(
        flow_name=flow.name,
        steps=tuple(traces),
        initial_state=initial_state,
        duration=time.monotonic() - t0,
        metadata=_trace_metadata(state),
    )
    return FlowResult(state=state, trace=trace, results=results)


async def iter_flow(flow: Flow, state: State) -> AsyncIterator[FlowEvent]:
    """Stream events during flow execution."""
    t0 = time.monotonic()
    traces: list[StepTrace] = []
    results: dict[str, Result] = {}
    _ensure_budget(flow, state)
    initial_state = state.to_dict()
    scope = _open_meter_scope(state)

    yield FlowEvent(type="flow_start", flow_name=flow.name)

    if flow.is_dag:
        async for event in _iter_dag(flow, state, traces, results):
            yield event
    else:
        async for event in _iter_sequential(flow, state, traces, results):
            yield event

    scope.close()
    trace = Trace(
        flow_name=flow.name,
        steps=tuple(traces),
        initial_state=initial_state,
        duration=time.monotonic() - t0,
        metadata=_trace_metadata(state),
    )
    yield FlowEvent(type="flow_end", flow_name=flow.name, trace=trace)


# --- Sequential / Cyclic Execution ---


async def _execute_sequential(
    flow: Flow,
    state: State,
    traces: list[StepTrace],
    results: dict[str, Result],
) -> None:
    has_conditions = any(fs.when is not None for fs in flow.steps)
    max_iter = flow.max_iterations

    if not has_conditions:
        # Pure sequential — single pass
        for fs in flow.steps:
            if _append_budget_precheck_trace(flow, state, traces, results):
                break
            result, step_trace = await _execute_flow_step(fs, flow, state)
            traces.append(step_trace)
            if result is not None:
                results[fs.step.name] = result
                state.merge(result)
                if _record_budget_after_result(state, result, traces, results):
                    break
                if result.is_error and _should_halt(result, fs):
                    break
    else:
        # Cyclic — iterate until no steps execute or max_iterations reached
        iteration = 0
        while max_iter is None or iteration < max_iter:
            any_executed = False
            for fs in flow.steps:
                if _append_budget_precheck_trace(flow, state, traces, results):
                    return
                snapshot = state.snapshot()
                scoped = _resolve_and_apply_scope(snapshot, fs, flow)

                if fs.when is not None:
                    try:
                        condition_met = fs.when(scoped)
                    except Exception as exc:
                        traces.append(
                            StepTrace(
                                name=fs.step.name,
                                skipped=True,
                                skip_reason=f"condition error: {exc}",
                                started_at=time.monotonic(),
                            )
                        )
                        continue
                    if not condition_met:
                        traces.append(
                            StepTrace(
                                name=fs.step.name,
                                skipped=True,
                                skip_reason="condition not met",
                                started_at=time.monotonic(),
                            )
                        )
                        continue

                result, step_trace = await _run_step_with_scope(fs.step, scoped)
                traces.append(step_trace)
                any_executed = True
                if result is not None:
                    results[fs.step.name] = result
                    state.merge(result)
                    if _record_budget_after_result(state, result, traces, results):
                        return
                    if result.is_error and _should_halt(result, fs):
                        return

            if not any_executed:
                break
            iteration += 1


async def _iter_sequential(
    flow: Flow,
    state: State,
    traces: list[StepTrace],
    results: dict[str, Result],
) -> AsyncIterator[FlowEvent]:
    has_conditions = any(fs.when is not None for fs in flow.steps)
    max_iter = flow.max_iterations

    if not has_conditions:
        for fs in flow.steps:
            if _append_budget_precheck_trace(flow, state, traces, results):
                break
            yield FlowEvent(type="step_start", flow_name=flow.name, step_name=fs.step.name)
            result, step_trace = await _execute_flow_step(fs, flow, state)
            traces.append(step_trace)
            if step_trace.skipped:
                yield FlowEvent(
                    type="step_skipped",
                    flow_name=flow.name,
                    step_name=fs.step.name,
                )
            else:
                if result is not None:
                    results[fs.step.name] = result
                    state.merge(result)
                    if _record_budget_after_result(state, result, traces, results):
                        yield FlowEvent(
                            type="policy_decision",
                            flow_name=flow.name,
                            policy_decision="budget_exceeded",
                        )
                        break
                yield FlowEvent(
                    type="step_end",
                    flow_name=flow.name,
                    step_name=fs.step.name,
                    result=result,
                )
                if result is not None and result.is_error and _should_halt(result, fs):
                    break
    else:
        iteration = 0
        while max_iter is None or iteration < max_iter:
            any_executed = False
            for fs in flow.steps:
                if _append_budget_precheck_trace(flow, state, traces, results):
                    yield FlowEvent(
                        type="policy_decision",
                        flow_name=flow.name,
                        policy_decision="budget_exceeded",
                    )
                    return
                snapshot = state.snapshot()
                scoped = _resolve_and_apply_scope(snapshot, fs, flow)

                if fs.when is not None:
                    try:
                        condition_met = fs.when(scoped)
                    except Exception as exc:
                        traces.append(
                            StepTrace(
                                name=fs.step.name,
                                skipped=True,
                                skip_reason=f"condition error: {exc}",
                                started_at=time.monotonic(),
                            )
                        )
                        yield FlowEvent(
                            type="step_skipped",
                            flow_name=flow.name,
                            step_name=fs.step.name,
                        )
                        continue
                    if not condition_met:
                        traces.append(
                            StepTrace(
                                name=fs.step.name,
                                skipped=True,
                                skip_reason="condition not met",
                                started_at=time.monotonic(),
                            )
                        )
                        yield FlowEvent(
                            type="step_skipped",
                            flow_name=flow.name,
                            step_name=fs.step.name,
                        )
                        continue

                yield FlowEvent(type="step_start", flow_name=flow.name, step_name=fs.step.name)
                result, step_trace = await _run_step_with_scope(fs.step, scoped)
                traces.append(step_trace)
                any_executed = True
                if result is not None:
                    results[fs.step.name] = result
                    state.merge(result)
                    if _record_budget_after_result(state, result, traces, results):
                        yield FlowEvent(
                            type="policy_decision",
                            flow_name=flow.name,
                            policy_decision="budget_exceeded",
                        )
                        return
                yield FlowEvent(
                    type="step_end",
                    flow_name=flow.name,
                    step_name=fs.step.name,
                    result=result,
                )
                if result is not None and result.is_error and _should_halt(result, fs):
                    return

            if not any_executed:
                break
            iteration += 1


# --- DAG Execution ---


async def _execute_dag(
    flow: Flow,
    state: State,
    traces: list[StepTrace],
    results: dict[str, Result],
) -> None:
    step_map = {fs.step.name: fs for fs in flow.steps}
    in_degree: dict[str, int] = {fs.step.name: len(fs.after) for fs in flow.steps}
    dependents: dict[str, list[str]] = {fs.step.name: [] for fs in flow.steps}
    for fs in flow.steps:
        for dep in fs.after:
            dependents[dep].append(fs.step.name)

    completed: set[str] = set()
    failed: set[str] = set()
    skipped: set[str] = set()

    while len(completed) + len(failed) + len(skipped) < len(flow.steps):
        ready = [
            name
            for name, deg in in_degree.items()
            if deg == 0 and name not in completed and name not in failed and name not in skipped
        ]
        if not ready:
            break

        # Check skip propagation and when conditions for ready steps
        to_execute: list[str] = []
        for name in ready:
            fs = step_map[name]
            skip_reason = _check_skip_propagation(fs, failed, skipped)
            if not skip_reason and fs.when is not None:
                snapshot = state.snapshot()
                scoped = _resolve_and_apply_scope(snapshot, fs, flow)
                if not fs.when(scoped):
                    skip_reason = "condition not met"
            if skip_reason:
                skipped.add(name)
                traces.append(
                    StepTrace(
                        name=name,
                        skipped=True,
                        skip_reason=skip_reason,
                        started_at=time.monotonic(),
                    )
                )
                for dep_name in dependents[name]:
                    in_degree[dep_name] -= 1
            else:
                to_execute.append(name)

        if not to_execute:
            continue

        if len(to_execute) == 1:
            # Single step — execute directly on state
            if _append_budget_precheck_trace(flow, state, traces, results):
                break
            name = to_execute[0]
            fs = step_map[name]
            result, step_trace = await _execute_flow_step(fs, flow, state)
            traces.append(step_trace)
            if result is not None:
                results[name] = result
                state.merge(result)
                if _record_budget_after_result(state, result, traces, results):
                    break
                if result.is_error:
                    failed.add(name)
                else:
                    completed.add(name)
            else:
                completed.add(name)
            for dep_name in dependents[name]:
                in_degree[dep_name] -= 1
        else:
            # Multiple ready — fork, execute in parallel, merge
            tasks: list[tuple[str, FlowStep, State]] = []
            for name in to_execute:
                fs = step_map[name]
                forked = state.fork()
                tasks.append((name, fs, forked))

            async def _run_parallel(
                name: str, fs: FlowStep, forked_state: State
            ) -> tuple[str, Result | None, StepTrace]:
                r, st = await _execute_flow_step(fs, flow, forked_state)
                return name, r, st

            gathered = await asyncio.gather(
                *[_run_parallel(n, f, s) for n, f, s in tasks],
                return_exceptions=True,
            )

            parallel_results: list[Result] = []
            for item in gathered:
                if isinstance(item, BaseException):
                    # Should not happen since execute_step catches exceptions,
                    # but handle gracefully
                    continue
                name, result, step_trace = item
                traces.append(step_trace)
                if result is not None:
                    results[name] = result
                    parallel_results.append(result)
                    if result.is_error:
                        failed.add(name)
                    else:
                        completed.add(name)
                else:
                    completed.add(name)
                for dep_name in dependents[name]:
                    in_degree[dep_name] -= 1

            if parallel_results:
                state.merge(*parallel_results)
                for result in parallel_results:
                    if _record_budget_after_result(state, result, traces, results):
                        return


async def _iter_dag(
    flow: Flow,
    state: State,
    traces: list[StepTrace],
    results: dict[str, Result],
) -> AsyncIterator[FlowEvent]:
    # For streaming DAG we use the same logic but yield events
    # Simplified: execute DAG and yield step events
    step_map = {fs.step.name: fs for fs in flow.steps}
    in_degree: dict[str, int] = {fs.step.name: len(fs.after) for fs in flow.steps}
    dependents: dict[str, list[str]] = {fs.step.name: [] for fs in flow.steps}
    for fs in flow.steps:
        for dep in fs.after:
            dependents[dep].append(fs.step.name)

    completed: set[str] = set()
    failed: set[str] = set()
    skipped: set[str] = set()

    while len(completed) + len(failed) + len(skipped) < len(flow.steps):
        ready = [
            name
            for name, deg in in_degree.items()
            if deg == 0 and name not in completed and name not in failed and name not in skipped
        ]
        if not ready:
            break

        for name in ready:
            fs = step_map[name]
            skip_reason = _check_skip_propagation(fs, failed, skipped)
            if not skip_reason and fs.when is not None:
                snapshot = state.snapshot()
                scoped = _resolve_and_apply_scope(snapshot, fs, flow)
                if not fs.when(scoped):
                    skip_reason = "condition not met"
            if skip_reason:
                skipped.add(name)
                traces.append(
                    StepTrace(
                        name=name,
                        skipped=True,
                        skip_reason=skip_reason,
                        started_at=time.monotonic(),
                    )
                )
                yield FlowEvent(
                    type="step_skipped",
                    flow_name=flow.name,
                    step_name=name,
                )
                for dep_name in dependents[name]:
                    in_degree[dep_name] -= 1
                continue

            yield FlowEvent(type="step_start", flow_name=flow.name, step_name=name)
            if _append_budget_precheck_trace(flow, state, traces, results):
                yield FlowEvent(
                    type="policy_decision",
                    flow_name=flow.name,
                    policy_decision="budget_exceeded",
                )
                return
            result, step_trace = await _execute_flow_step(fs, flow, state)
            traces.append(step_trace)
            if result is not None:
                results[name] = result
                state.merge(result)
                if _record_budget_after_result(state, result, traces, results):
                    yield FlowEvent(
                        type="policy_decision",
                        flow_name=flow.name,
                        policy_decision="budget_exceeded",
                    )
                    return
                if result.is_error:
                    failed.add(name)
                else:
                    completed.add(name)
            else:
                completed.add(name)
            yield FlowEvent(
                type="step_end",
                flow_name=flow.name,
                step_name=name,
                result=result,
            )
            for dep_name in dependents[name]:
                in_degree[dep_name] -= 1


# --- Helpers ---


async def _execute_flow_step(
    fs: FlowStep, flow: Flow, state: State
) -> tuple[Result | None, StepTrace]:
    """Execute a single FlowStep, applying scope and delegating to step engine."""
    snapshot = state.snapshot()
    scoped = _resolve_and_apply_scope(snapshot, fs, flow)
    # Bind the run's meter around step execution only (never across an event yield), so LLM/tool
    # calls inside the step are metered under the run scope.
    with bind_meter(_meter_scope(state)):
        return await _run_step_with_scope(fs.step, scoped)


async def _run_step_with_scope(
    step: Step, scoped_snapshot: StateSnapshot
) -> tuple[Result, StepTrace]:
    """Run step engine with an already-scoped snapshot."""
    return await execute_step(step, scoped_snapshot)


def _resolve_and_apply_scope(snapshot: Any, fs: FlowStep, flow: Flow) -> Any:
    """Resolve scope: FlowStep > Step > Flow, then apply."""
    scope = fs.scope or fs.step.scope or flow.scope
    return apply_scope(snapshot, scope)


def _check_skip_propagation(fs: FlowStep, failed: set[str], skipped: set[str]) -> str | None:
    """Check if a step should be skipped due to dependency status."""
    if not fs.after:
        return None

    # Any dep failed → skip
    for dep in fs.after:
        if dep in failed:
            return f"dependency {dep!r} failed"

    # All deps skipped → skip
    if all(dep in skipped for dep in fs.after):
        return "all dependencies skipped"

    # Some deps skipped (mixed) → skip (all deps must succeed)
    for dep in fs.after:
        if dep in skipped:
            return f"dependency {dep!r} was skipped"

    return None


def _should_halt(result: Result, fs: FlowStep) -> bool:
    """Check if an error result should halt the flow."""
    policy = fs.step.policy
    if policy is None:
        return True  # Default: halt on error
    return policy.on_exhausted == "halt"


def _open_meter_scope(state: State) -> MeterScope:
    """Create the run's meter (measure-only for now) and stash it for the step binder.

    Stored in the ``world`` layer: it is run-global and, crucially, ``world`` is shared by
    reference across ``State.fork()`` (DAG branches) — the meter holds a ``threading.Lock`` that a
    deep copy of the ``operational`` layer would choke on.
    """
    scope = MeterScope()
    state.set("_meter_scope", scope, layer="world")
    return scope


def _meter_scope(state: State) -> MeterScope | None:
    value = state.get("_meter_scope")
    return value if isinstance(value, MeterScope) else None


def _ensure_budget(flow: Flow, state: State) -> None:
    if flow.budget_policy is None or isinstance(state.get("budget_state"), BudgetState):
        return
    state.set("budget_state", BudgetState.start(flow.budget_policy))


def _budget_state(state: State) -> BudgetState | None:
    value = state.get("budget_state")
    return value if isinstance(value, BudgetState) else None


def _trace_metadata(state: State) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    budget = _budget_state(state)
    if budget is not None:
        metadata["budget"] = budget.to_dict()
    scope = _meter_scope(state)
    if scope is not None:
        metadata["meter"] = BudgetReport.from_snapshot(scope.snapshot()).to_dict()
    return metadata


def _append_budget_precheck_trace(
    flow: Flow,
    state: State,
    traces: list[StepTrace],
    results: dict[str, Result],
) -> bool:
    if flow.budget_policy is None:
        return False
    budget = _budget_state(state)
    if budget is None:
        return False
    try:
        budget.check_wall_time()
    except BudgetExceeded as exc:
        _append_budget_exceeded(exc, budget, state, traces, results)
        return True
    return False


def _record_budget_after_result(
    state: State,
    result: Result,
    traces: list[StepTrace],
    results: dict[str, Result],
) -> bool:
    budget = _budget_state(state)
    if budget is None:
        return False

    authoritative = result.artifacts.get("budget_state")
    if isinstance(authoritative, BudgetState):
        state.set("budget_state", authoritative)
        if authoritative.exceeded is not None:
            _append_budget_exceeded(authoritative.exceeded, authoritative, state, traces, results)
            return True
        return False

    try:
        state.set("budget_state", budget.record_result(result))
    except BudgetExceeded as exc:
        _append_budget_exceeded(exc, budget.with_exceeded(exc), state, traces, results)
        return True
    return False


def _append_budget_exceeded(
    exc: BudgetExceeded,
    budget: BudgetState,
    state: State,
    traces: list[StepTrace],
    results: dict[str, Result],
) -> None:
    exceeded_budget = budget.with_exceeded(exc)
    state.set("budget_state", exceeded_budget)
    result = Result(
        error=str(exc),
        artifacts={
            "budget_exceeded": exc.to_dict(),
            "budget_state": exceeded_budget,
        },
    )
    results["budget_exceeded"] = result
    traces.append(
        StepTrace(
            name="budget_exceeded",
            output_result=result.to_dict(),
            error=str(exc),
            policy_decisions=("budget_exceeded",),
            started_at=time.monotonic(),
        )
    )
