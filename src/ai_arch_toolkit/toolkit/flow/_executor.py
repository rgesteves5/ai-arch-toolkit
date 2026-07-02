"""Flow executor — DAG, sequential, and cyclic execution modes."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from ai_arch_toolkit.core._metering._admission import AdmissionDenied
from ai_arch_toolkit.core._metering._scope import (
    MeterScope,
    RunConfig,
    bind_meter,
    current_meter,
)
from ai_arch_toolkit.core._state import State, StateSnapshot
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.core._step_engine import execute_step
from ai_arch_toolkit.core._trace import StepTrace, Trace
from ai_arch_toolkit.toolkit.budget import BudgetController, BudgetReport
from ai_arch_toolkit.toolkit.flow._flow import Flow, FlowEvent, FlowResult, FlowStep
from ai_arch_toolkit.toolkit.flow._scope import apply_scope


async def execute_flow(flow: Flow, state: State) -> FlowResult:
    """Execute a Flow, choosing DAG or sequential mode."""
    t0 = time.monotonic()
    traces: list[StepTrace] = []
    results: dict[str, Result] = {}
    initial_state = state.to_dict()
    scope, owned = _open_meter_scope(flow, state)

    try:
        if flow.is_dag:
            await _execute_dag(flow, state, traces, results)
        else:
            await _execute_sequential(flow, state, traces, results)
    except AdmissionDenied as exc:
        _append_denial(exc, state, traces, results)  # a hard mid-step budget denial

    if owned:
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
    initial_state = state.to_dict()
    scope, owned = _open_meter_scope(flow, state)

    yield FlowEvent(type="flow_start", flow_name=flow.name)

    try:
        if flow.is_dag:
            async for event in _iter_dag(flow, state, traces, results):
                yield event
        else:
            async for event in _iter_sequential(flow, state, traces, results):
                yield event
    except AdmissionDenied as exc:
        _append_denial(exc, state, traces, results)
        yield FlowEvent(
            type="policy_decision", flow_name=flow.name, policy_decision="budget_exceeded"
        )

    if owned:
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

                result, step_trace = await _run_step_with_scope(
                    fs.step, scoped, _meter_scope(state)
                )
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
                result, step_trace = await _run_step_with_scope(
                    fs.step, scoped, _meter_scope(state)
                )
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
    return await _run_step_with_scope(fs.step, scoped, _meter_scope(state))


async def _run_step_with_scope(
    step: Step, scoped_snapshot: StateSnapshot, scope: MeterScope | None
) -> tuple[Result, StepTrace]:
    """Run the step engine, binding the run's meter around it (every path funnels through here).

    Bound only around the step's execution — never across an ``iter_flow`` event yield — so a
    step's LLM/tool calls are metered under the run scope, without a cross-context token reset.
    """
    with bind_meter(scope):
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


def _open_meter_scope(flow: Flow, state: State) -> tuple[MeterScope, bool]:
    """Bind the run's meter, inheriting an enclosing scope so nested flows share one budget.

    Returns ``(scope, owned)``; only the outermost flow that created the scope closes it. The
    scope enforces when ``budget_policy`` is set (via its controller), else it is measure-only.
    Stored in the ``world`` layer — shared by reference across ``State.fork()`` (DAG branches),
    since the meter holds a ``threading.Lock`` a deep copy would choke on.
    """
    inherited = current_meter()
    if inherited is not None:
        state.set("_meter_scope", inherited, layer="world")
        return inherited, False
    policy = flow.budget_policy
    controller = BudgetController(policy) if policy is not None and not policy.is_empty else None
    scope = MeterScope(RunConfig(controller=controller))
    state.set("_meter_scope", scope, layer="world")
    return scope, True


def _meter_scope(state: State) -> MeterScope | None:
    value = state.get("_meter_scope")
    return value if isinstance(value, MeterScope) else None


def _trace_metadata(state: State) -> dict[str, Any]:
    scope = _meter_scope(state)
    if scope is None:
        return {}
    policy = scope.controller.policy if isinstance(scope.controller, BudgetController) else None
    return {"meter": BudgetReport.from_snapshot(scope.snapshot(), policy).to_dict()}


def _over_budget(state: State) -> BudgetReport | None:
    """Report if the run exceeded WALL-TIME — the one cap not enforced at a charge site.

    Call/token/cost caps are enforced precisely at the charge site: the controller denies the
    operation that would exceed and it surfaces as an :class:`AdmissionDenied` caught at the top.
    """
    scope = _meter_scope(state)
    if scope is None or not isinstance(scope.controller, BudgetController):
        return None
    policy = scope.controller.policy
    if policy.max_wall_s is None:
        return None
    snap = scope.snapshot()
    if snap.elapsed_s <= policy.max_wall_s:
        return None
    return BudgetReport.from_snapshot(snap, policy)


def _append_budget_precheck_trace(
    flow: Flow,
    state: State,
    traces: list[StepTrace],
    results: dict[str, Result],
) -> bool:
    """Halt before a step if the meter has already reached a cap (wall-time / soft overshoot)."""
    report = _over_budget(state)
    if report is None:
        return False
    _append_meter_exceeded(report, traces, results)
    return True


def _record_budget_after_result(
    state: State,
    result: Result,
    traces: list[StepTrace],
    results: dict[str, Result],
) -> bool:
    """Halt after a step whose metered calls pushed the run over a cap (soft overshoot)."""
    report = _over_budget(state)
    if report is None:
        return False
    _append_meter_exceeded(report, traces, results)
    return True


def _append_meter_exceeded(
    report: BudgetReport, traces: list[StepTrace], results: dict[str, Result]
) -> None:
    dimension = report.breached[0] if report.breached else "budget"
    _append_budget_exceeded_trace(
        f"Budget exceeded: {dimension}", report.to_dict(), traces, results
    )


def _append_denial(
    exc: AdmissionDenied, state: State, traces: list[StepTrace], results: dict[str, Result]
) -> None:
    """Record a hard mid-step admission denial (raised at a charge site) as budget_exceeded."""
    info = {
        "dimension": exc.dimension,
        "limit": exc.limit,
        "current": exc.current,
        "attempted": exc.attempted,
    }
    _append_budget_exceeded_trace(str(exc), info, traces, results)


def _append_budget_exceeded_trace(
    message: str, info: dict[str, Any], traces: list[StepTrace], results: dict[str, Result]
) -> None:
    result = Result(error=message, artifacts={"budget_exceeded": info})
    results["budget_exceeded"] = result
    traces.append(
        StepTrace(
            name="budget_exceeded",
            output_result=result.to_dict(),
            error=message,
            policy_decisions=("budget_exceeded",),
            started_at=time.monotonic(),
        )
    )
