"""Flow executor — one engine behind ``run()`` and ``iter()``: sequential, cyclic, and DAG modes.

The engine is a single async generator. Every step runs in its own task, and the events a step
produces while it runs (its start, retries, timeouts, fallbacks) are delivered as they happen.
Between steps the engine only moves on when the consumer asks for the next event, so an abandoned
iteration stops where it was left. ``run()`` drains the generator; ``iter()`` hands it to the
caller. Closing or abandoning an iteration cancels the steps still running and closes the run's
meter. The run's ``timeout`` is enforced with bounded waits, never with a cancel scope held
across a ``yield``.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import AsyncGenerator, Callable, Iterator
from contextvars import ContextVar
from dataclasses import replace
from typing import Any

from ai_arch_toolkit.core._metering._admission import AdmissionDenied
from ai_arch_toolkit.core._metering._scope import (
    MeterScope,
    RunConfig,
    bind_meter,
    current_meter,
    current_span_id,
)
from ai_arch_toolkit.core._state import State, StateSnapshot
from ai_arch_toolkit.core._step import Result
from ai_arch_toolkit.core._step_engine import execute_step
from ai_arch_toolkit.core._sync import _stream_sync
from ai_arch_toolkit.core._trace import (
    PolicyDecision,
    StepTrace,
    Trace,
    TraceCapture,
    capture_result,
    copy_state,
)
from ai_arch_toolkit.toolkit.budget import BudgetController, BudgetPolicy, BudgetReport
from ai_arch_toolkit.toolkit.flow._flow import (
    Flow,
    FlowEvent,
    FlowResult,
    FlowStep,
    _scope_report,
)
from ai_arch_toolkit.toolkit.flow._scope import apply_scope

__all__ = ["FlowExecution", "SyncFlowExecution", "execute_flow", "iter_flow"]

type _StepTask = asyncio.Task[tuple[Result, StepTrace]]

# Where a running step collects the traces of flows run inside it (``as_step()`` wrappers and
# strategies that call ``inner.run()``), so they can be linked as that step's children.
_child_traces: ContextVar[list[StepTrace] | None] = ContextVar(
    "ai_arch_flow_child_traces", default=None
)


# --------------------------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------------------------


async def execute_flow(
    flow: Flow,
    state: State,
    *,
    budget_policy: BudgetPolicy | None = None,
    config: RunConfig | None = None,
) -> FlowResult:
    """Run a flow to completion and return its result (drains the engine)."""
    run = _FlowRun(flow, state, budget_policy=budget_policy, config=config)
    events = run.events()
    try:
        async for _ in events:
            pass
    finally:
        await events.aclose()
    if run.result is None:  # pragma: no cover - the engine always sets it before finishing
        raise RuntimeError(f"flow {flow.name!r} finished without a result")
    return run.result


def iter_flow(
    flow: Flow,
    state: State,
    *,
    budget_policy: BudgetPolicy | None = None,
    config: RunConfig | None = None,
) -> FlowExecution:
    """Start iterating a flow run: events as they happen, then ``.result``."""
    return FlowExecution(_FlowRun(flow, state, budget_policy=budget_policy, config=config))


class FlowExecution:
    """A flow run being iterated: an async iterator of :class:`FlowEvent`, then its result.

    Nothing runs until the first event is requested, and the run only moves past a step when the
    next event is requested. ``result`` is ``None`` until the run has finished.

    A ``break`` does not stop the run by itself: the steps in flight keep running while the
    execution is referenced. :meth:`aclose`, or leaving an ``async with`` block, cancels them and
    closes the run's meter; so does garbage collection once nothing references the execution.
    """

    __slots__ = ("_events", "_run")

    def __init__(self, run: _FlowRun) -> None:
        self._run = run
        self._events = run.events()

    def __aiter__(self) -> FlowExecution:
        return self

    async def __anext__(self) -> FlowEvent:
        return await self._events.__anext__()

    async def aclose(self) -> None:
        """Stop the run: cancel the steps still running and close its meter."""
        await self._events.aclose()

    async def __aenter__(self) -> FlowExecution:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    @property
    def result(self) -> FlowResult | None:
        """The finished run's result, or ``None`` while it is running or if it was abandoned."""
        return self._run.result

    @property
    def state(self) -> State:
        """The state the run reads and writes."""
        return self._run.state

    @property
    def meter_scope(self) -> MeterScope | None:
        """The run's meter scope, once the run has started."""
        return self._run.scope


class SyncFlowExecution:
    """Synchronous counterpart of :class:`FlowExecution`; the run happens on a background loop.

    As with the async execution, a ``break`` leaves the run going on its thread while the object
    is referenced; :meth:`close`, or leaving a ``with`` block, cancels it.
    """

    __slots__ = ("_events", "_execution")

    def __init__(self, start: Callable[[], FlowExecution]) -> None:
        self._execution: FlowExecution | None = None

        def begin() -> FlowExecution:
            self._execution = start()
            return self._execution

        self._events: Iterator[FlowEvent] = _stream_sync(begin)

    def __iter__(self) -> SyncFlowExecution:
        return self

    def __next__(self) -> FlowEvent:
        return next(self._events)

    def close(self) -> None:
        """Stop consuming; the background run is cancelled."""
        close = getattr(self._events, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> SyncFlowExecution:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def result(self) -> FlowResult | None:
        """The finished run's result, or ``None`` while it is running."""
        return self._execution.result if self._execution is not None else None


# --------------------------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------------------------


class _FlowTimeout(Exception):
    """Internal signal that this run's own ``timeout`` elapsed; never escapes the run."""


class _OrchestrationError(Exception):
    """A user-supplied ``when`` or ``Scope`` callable raised while scheduling a step."""

    def __init__(self, kind: str, error: Exception) -> None:
        super().__init__(f"{kind} error: {type(error).__name__}: {error}")


class _FlowRun:
    """One run of a flow: the bookkeeping behind the engine generator."""

    __slots__ = (
        "_events",
        "_running",
        "_signal",
        "budget_policy",
        "config",
        "deadline",
        "flow",
        "owned",
        "result",
        "results",
        "scope",
        "span_id",
        "state",
        "traces",
    )

    def __init__(
        self,
        flow: Flow,
        state: State,
        *,
        budget_policy: BudgetPolicy | None,
        config: RunConfig | None,
    ) -> None:
        self.flow = flow
        self.state = state
        self.budget_policy = budget_policy
        self.config = config
        self.traces: list[StepTrace] = []
        self.results: dict[str, Result] = {}
        self.result: FlowResult | None = None
        self.scope: MeterScope | None = None
        self.owned = False
        self.span_id: str | None = None
        self.deadline: float | None = None
        self._events: deque[FlowEvent] = deque()
        self._signal: asyncio.Event | None = None
        self._running: dict[_StepTask, str] = {}

    # ---------------------------------------------------------------------------- lifecycle

    async def events(self) -> AsyncGenerator[FlowEvent]:
        """The engine: yields the run's events and records its result before ``flow_end``."""
        flow = self.flow
        started = time.monotonic()
        capture = flow.trace_capture
        initial_state = {} if capture == "none" else copy_state(self.state.to_dict())
        parent_children = _child_traces.get()
        scope, self.owned = _open_meter_scope(flow, self.budget_policy, self.config)
        self.scope = scope
        # A nested run shares its owner's scope and stays under the span that is current where it
        # starts (e.g. a parent step's max_cost span), so the parent step sees the nested spend.
        self.span_id = None if self.owned else current_span_id()
        self._signal = asyncio.Event()
        if flow.timeout is not None:
            self.deadline = asyncio.get_running_loop().time() + flow.timeout

        yield FlowEvent(type="flow_start", flow_name=flow.name)
        body = self._run_dag() if flow.is_dag else self._run_sequential()
        try:
            try:
                async for event in body:
                    yield event
            except AdmissionDenied as exc:
                if not self.owned:
                    raise  # nested run: the owning (outermost) run converts it once, at the top
                _append_denial(exc, self.traces, self.results, capture)
                yield FlowEvent(
                    type="policy_decision", flow_name=flow.name, policy_decision="budget_exceeded"
                )
            except _FlowTimeout:
                message = self._record_timeout()
                await self._cancel_running()  # stopped before the consumer hears of the timeout
                yield FlowEvent(type="timeout", flow_name=flow.name, error=message)
        finally:
            # Also runs when the consumer abandons the iteration (GeneratorExit/cancellation):
            # no step keeps running and no started meter operation leaks past the run.
            try:
                await body.aclose()
                await self._cancel_running()
            finally:
                if self.owned:
                    scope.close()  # even if a second cancellation interrupts the cleanup

        trace = Trace(
            flow_name=flow.name,
            steps=tuple(self.traces),
            initial_state=initial_state,
            duration=time.monotonic() - started,
            metadata=_trace_metadata(scope),
        )
        self.result = FlowResult(
            state=self.state, trace=trace, results=self.results, meter_scope=scope
        )
        if parent_children is not None:
            final = self.result.final_result
            parent_children.append(
                StepTrace(
                    name=flow.name,
                    duration=trace.duration,
                    error=final.error if final is not None else None,
                    children=trace.steps,
                    started_at=started,
                )
            )
        yield FlowEvent(type="flow_end", flow_name=flow.name, trace=trace)

    async def _cancel_running(self) -> None:
        running = [task for task in self._running if not task.done()]
        for task in running:
            task.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)

    # ---------------------------------------------------------------------------- modes

    async def _run_sequential(self) -> AsyncGenerator[FlowEvent]:
        flow = self.flow
        cyclic = any(fs.when is not None for fs in flow.steps)
        iteration = 0
        while True:
            if cyclic and flow.max_iterations is not None and iteration >= flow.max_iterations:
                return
            any_executed = False
            for fs in flow.steps:
                if (stop := self._over_budget()) is not None:
                    yield stop
                    return
                self._check_deadline()
                try:
                    scoped = self._scoped(fs, self.state.snapshot())
                    if fs.when is not None and not self._condition(fs, scoped):
                        yield self._record_skip(fs, "condition not met")
                        continue
                except _OrchestrationError as error:
                    for event in self._record_orchestration_error(fs, error):
                        yield event
                    return

                task = self._spawn(fs, scoped)
                async for event in self._wait([task]):
                    yield event
                result, trace = task.result()
                any_executed = True
                self._record(fs, result, trace)
                self.state.merge(result)
                yield _step_end(flow.name, fs.step.name, result)
                if (stop := self._over_budget()) is not None:
                    yield stop
                    return
                if result.is_error and self._should_halt(fs):
                    return

            if not cyclic or not any_executed:
                return
            iteration += 1

    async def _run_dag(self) -> AsyncGenerator[FlowEvent]:
        flow = self.flow
        step_map = {fs.step.name: fs for fs in flow.steps}
        in_degree = {fs.step.name: len(fs.after) for fs in flow.steps}
        dependents: dict[str, list[str]] = {fs.step.name: [] for fs in flow.steps}
        for fs in flow.steps:
            for dep in fs.after:
                dependents[dep].append(fs.step.name)
        completed: set[str] = set()
        failed: set[str] = set()
        skipped: set[str] = set()

        def release(name: str) -> None:
            for dependent in dependents[name]:
                in_degree[dependent] -= 1

        while len(completed) + len(failed) + len(skipped) < len(flow.steps):
            ready = [
                name
                for name, degree in in_degree.items()
                if degree == 0 and name not in completed and name not in failed
                if name not in skipped
            ]
            if not ready:
                break

            wave: list[FlowStep] = []
            for name in ready:
                fs = step_map[name]
                reason = _check_skip_propagation(fs, failed, skipped)
                if reason is None and fs.when is not None:
                    try:
                        if not self._condition(fs, self._scoped(fs, self.state.snapshot())):
                            reason = "condition not met"
                    except _OrchestrationError as error:
                        for event in self._record_orchestration_error(fs, error):
                            yield event
                        return
                if reason is not None:
                    skipped.add(name)
                    yield self._record_skip(fs, reason)
                    release(name)
                    continue
                wave.append(fs)
            if not wave:
                continue

            if (stop := self._over_budget()) is not None:
                yield stop
                return
            self._check_deadline()

            if len(wave) == 1:
                fs = wave[0]
                try:
                    scoped = self._scoped(fs, self.state.snapshot())
                except _OrchestrationError as error:
                    for event in self._record_orchestration_error(fs, error):
                        yield event
                    return
                task = self._spawn(fs, scoped)
                async for event in self._wait([task]):
                    yield event
                result, trace = task.result()
                self._record(fs, result, trace)
                self.state.merge(result)
                (failed if result.is_error else completed).add(fs.step.name)
                yield _step_end(flow.name, fs.step.name, result)
                if (stop := self._over_budget()) is not None:
                    yield stop
                    return
                release(fs.step.name)
                continue

            # A parallel wave: each step reads its own fork, so siblings never see each other's
            # writes; every result is merged once the whole wave has finished.
            launches: list[tuple[FlowStep, StateSnapshot]] = []
            for fs in wave:
                try:
                    launches.append((fs, self._scoped(fs, self.state.fork().snapshot())))
                except _OrchestrationError as error:
                    for event in self._record_orchestration_error(fs, error):
                        yield event
                    return
            semaphore = (
                asyncio.Semaphore(flow.max_parallelism)
                if flow.max_parallelism is not None
                else None
            )
            tasks = [
                self._spawn(fs, scoped, semaphore=semaphore, announce_end=True)
                for fs, scoped in launches
            ]
            try:
                async for event in self._wait(tasks):
                    yield event
            except _FlowTimeout:
                # Like a denial, a timeout keeps the siblings that finished before it.
                on_time: list[Result] = []
                for (fs, _), task in zip(launches, tasks, strict=True):
                    if task.done() and not task.cancelled() and task.exception() is None:
                        result, trace = task.result()
                        self._record(fs, result, trace)
                        on_time.append(result)
                if on_time:
                    self.state.merge(*on_time)
                raise

            denial: AdmissionDenied | None = None
            finished: list[Result] = []
            for (fs, _), task in zip(launches, tasks, strict=True):
                error = task.exception()
                if error is not None:
                    if isinstance(error, AdmissionDenied):
                        denial = denial or error  # terminal, but only after keeping the siblings
                        continue
                    raise error
                result, trace = task.result()
                self._record(fs, result, trace)
                finished.append(result)
                (failed if result.is_error else completed).add(fs.step.name)
                release(fs.step.name)
            if finished:
                self.state.merge(*finished)
            if denial is not None:
                raise denial
            if (stop := self._over_budget()) is not None:
                yield stop
                return

    # ---------------------------------------------------------------------------- steps

    def _spawn(
        self,
        fs: FlowStep,
        scoped: StateSnapshot,
        *,
        semaphore: asyncio.Semaphore | None = None,
        announce_end: bool = False,
    ) -> _StepTask:
        task = asyncio.create_task(self._step(fs, scoped, semaphore, announce_end))
        self._running[task] = fs.step.name
        task.add_done_callback(self._forget)
        return task

    def _forget(self, task: _StepTask) -> None:
        self._running.pop(task, None)

    async def _step(
        self,
        fs: FlowStep,
        scoped: StateSnapshot,
        semaphore: asyncio.Semaphore | None,
        announce_end: bool,
    ) -> tuple[Result, StepTrace]:
        if semaphore is None:
            return await self._execute(fs, scoped, announce_end)
        async with semaphore:
            return await self._execute(fs, scoped, announce_end)

    async def _execute(
        self, fs: FlowStep, scoped: StateSnapshot, announce_end: bool
    ) -> tuple[Result, StepTrace]:
        flow_name = self.flow.name
        name = fs.step.name
        self._emit(FlowEvent(type="step_start", flow_name=flow_name, step_name=name))

        def on_decision(decision: PolicyDecision) -> None:
            self._emit(_decision_event(flow_name, name, decision))

        children: list[StepTrace] = []
        token = _child_traces.set(children)
        try:
            with bind_meter(self.scope, self.span_id):
                result, trace = await execute_step(
                    fs.step,
                    scoped,
                    policy=self.flow.policy,
                    on_decision=on_decision,
                    capture=self.flow.trace_capture,
                )
        finally:
            _child_traces.reset(token)
        if children:
            trace = replace(trace, children=_link_children(name, children))
        if announce_end:
            self._emit(_step_end(flow_name, name, result))
        return result, trace

    def _emit(self, event: FlowEvent) -> None:
        self._events.append(event)
        if self._signal is not None:
            self._signal.set()

    async def _wait(self, tasks: list[_StepTask]) -> AsyncGenerator[FlowEvent]:
        """Yield events as they arrive until every task is done, within the run's deadline."""
        signal = self._signal
        assert signal is not None
        loop = asyncio.get_running_loop()
        pending: set[asyncio.Future[Any]] = set(tasks)
        while True:
            while self._events:
                yield self._events.popleft()
            if not pending:
                return
            if self.deadline is not None and loop.time() >= self.deadline:
                raise _FlowTimeout  # a steady stream of events must not postpone the deadline
            signal.clear()  # no await since the drain above, so no event can slip past
            timeout = None if self.deadline is None else max(0.0, self.deadline - loop.time())
            waiter = asyncio.ensure_future(signal.wait())
            try:
                done, _ = await asyncio.wait(
                    {*pending, waiter}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                waiter.cancel()
            pending.difference_update(done)
            if not done:
                raise _FlowTimeout

    # ---------------------------------------------------------------------------- bookkeeping

    def _scoped(self, fs: FlowStep, snapshot: StateSnapshot) -> StateSnapshot:
        scope = fs.scope or fs.step.scope or self.flow.scope
        try:
            return apply_scope(snapshot, scope)
        except Exception as exc:
            raise _OrchestrationError("scope", exc) from exc

    def _condition(self, fs: FlowStep, scoped: StateSnapshot) -> bool:
        assert fs.when is not None
        try:
            return bool(fs.when(scoped))
        except Exception as exc:
            raise _OrchestrationError("condition", exc) from exc

    def _record(self, fs: FlowStep, result: Result, trace: StepTrace) -> None:
        self.traces.append(trace)
        self.results[fs.step.name] = result

    def _record_skip(self, fs: FlowStep, reason: str) -> FlowEvent:
        self.traces.append(
            StepTrace(
                name=fs.step.name, skipped=True, skip_reason=reason, started_at=time.monotonic()
            )
        )
        return FlowEvent(type="step_skipped", flow_name=self.flow.name, step_name=fs.step.name)

    def _record_orchestration_error(
        self, fs: FlowStep, error: _OrchestrationError
    ) -> tuple[FlowEvent, FlowEvent]:
        """A raising ``when``/``Scope`` callable is a programming error: record it and halt."""
        name = fs.step.name
        message = f"{error} (step {name!r})"
        result = Result(error=message)
        self.results[name] = result
        output_result, output_keys = capture_result(result.to_dict(), self.flow.trace_capture)
        self.traces.append(
            StepTrace(
                name=name,
                output_result=output_result,
                output_keys=output_keys,
                error=message,
                policy_decisions=("halt",),
                started_at=time.monotonic(),
            )
        )
        return (
            FlowEvent(type="step_start", flow_name=self.flow.name, step_name=name),
            _step_end(self.flow.name, name, result),
        )

    def _record_timeout(self) -> str:
        in_flight = sorted(name for task, name in self._running.items() if not task.done())
        suffix = f" (in flight: {', '.join(in_flight)})" if in_flight else ""
        message = f"Flow {self.flow.name!r} timed out after {self.flow.timeout}s{suffix}"
        result = Result(error=message)
        self.results["flow_timeout"] = result
        output_result, output_keys = capture_result(result.to_dict(), self.flow.trace_capture)
        self.traces.append(
            StepTrace(
                name="flow_timeout",
                output_result=output_result,
                output_keys=output_keys,
                error=message,
                policy_decisions=("timeout",),
                started_at=time.monotonic(),
            )
        )
        return message

    def _should_halt(self, fs: FlowStep) -> bool:
        """Whether an error result stops the flow (the step's policy, else the flow's)."""
        policy = fs.step.policy or self.flow.policy
        return policy is None or policy.on_exhausted == "halt"

    def _check_deadline(self) -> None:
        if self.deadline is not None and asyncio.get_running_loop().time() >= self.deadline:
            raise _FlowTimeout

    def _over_budget(self) -> FlowEvent | None:
        """Record and report a breached wall-time budget (the one cap checked between steps)."""
        report = _wall_time_breach(self.scope)
        if report is None:
            return None
        dimension = report.breached[0] if report.breached else "budget"
        _append_budget_exceeded_trace(
            f"Budget exceeded: {dimension}",
            report.to_dict(),
            self.traces,
            self.results,
            self.flow.trace_capture,
        )
        return FlowEvent(
            type="policy_decision", flow_name=self.flow.name, policy_decision="budget_exceeded"
        )


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def _step_end(flow_name: str, step_name: str, result: Result) -> FlowEvent:
    return FlowEvent(
        type="step_end",
        flow_name=flow_name,
        step_name=step_name,
        result=result,
        error=result.error,
    )


def _decision_event(flow_name: str, step_name: str, decision: PolicyDecision) -> FlowEvent:
    """Stream a policy decision the moment the step engine takes it."""
    if decision == "retry":
        return FlowEvent(
            type="retry", flow_name=flow_name, step_name=step_name, policy_decision=decision
        )
    if decision == "fallback":
        return FlowEvent(
            type="fallback", flow_name=flow_name, step_name=step_name, policy_decision=decision
        )
    if decision == "timeout":
        return FlowEvent(
            type="timeout", flow_name=flow_name, step_name=step_name, policy_decision=decision
        )
    return FlowEvent(
        type="policy_decision", flow_name=flow_name, step_name=step_name, policy_decision=decision
    )


def _link_children(step_name: str, children: list[StepTrace]) -> tuple[StepTrace, ...]:
    """The traces of flows run inside a step, as that step's children.

    A step made by ``Flow.as_step()`` runs exactly one nested flow of the same name; its steps are
    linked directly rather than nesting the flow under a step of the same name.
    """
    if len(children) == 1 and children[0].name == step_name:
        return children[0].children
    return tuple(children)


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


def _open_meter_scope(
    flow: Flow,
    override: BudgetPolicy | None = None,
    config: RunConfig | None = None,
) -> tuple[MeterScope, bool]:
    """The run's meter scope, inheriting an enclosing one so nested flows share one budget.

    Returns ``(scope, owned)``; only the outermost run that created the scope closes it. A per-run
    ``config`` (full :class:`RunConfig`) takes precedence over a per-run ``override`` budget, which
    takes precedence over the construction-time ``budget_policy``; all are ignored when a scope is
    inherited. The scope lives on the run (and on its :class:`FlowResult`), never in the ``State``.
    """
    inherited = current_meter()
    if inherited is not None:
        return inherited, False
    if config is not None:
        return MeterScope(config), True
    policy = override if override is not None else flow.budget_policy
    controller = BudgetController(policy) if policy is not None and not policy.is_empty else None
    return MeterScope(RunConfig(controller=controller)), True


def _trace_metadata(scope: MeterScope | None) -> dict[str, Any]:
    report = _scope_report(scope)
    return {"meter": report.to_dict()} if report is not None else {}


def _wall_time_breach(scope: MeterScope | None) -> BudgetReport | None:
    """Report if the run exceeded WALL-TIME — the one cap not enforced at a charge site.

    Call/token/cost caps are enforced precisely at the charge site: the controller denies the
    operation that would exceed and it surfaces as an :class:`AdmissionDenied` caught at the top.
    """
    if scope is None or not isinstance(scope.controller, BudgetController):
        return None
    policy = scope.controller.policy
    if policy.max_wall_s is None:
        return None
    snap = scope.snapshot()
    if snap.elapsed_s <= policy.max_wall_s:
        return None
    return BudgetReport.from_snapshot(snap, policy)


def _append_denial(
    exc: AdmissionDenied,
    traces: list[StepTrace],
    results: dict[str, Result],
    capture: TraceCapture,
) -> None:
    """Record a hard mid-step admission denial (raised at a charge site) as budget_exceeded."""
    info = {
        "dimension": exc.dimension,
        "limit": exc.limit,
        "current": exc.current,
        "attempted": exc.attempted,
    }
    _append_budget_exceeded_trace(str(exc), info, traces, results, capture)


def _append_budget_exceeded_trace(
    message: str,
    info: dict[str, Any],
    traces: list[StepTrace],
    results: dict[str, Result],
    capture: TraceCapture,
) -> None:
    result = Result(error=message, artifacts={"budget_exceeded": info})
    results["budget_exceeded"] = result
    output_result, output_keys = capture_result(result.to_dict(), capture)
    traces.append(
        StepTrace(
            name="budget_exceeded",
            output_result=output_result,
            output_keys=output_keys,
            error=message,
            policy_decisions=("budget_exceeded",),
            started_at=time.monotonic(),
        )
    )
