"""Flow — composable orchestration of Steps."""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from ai_arch_toolkit.core._policy import Policy
from ai_arch_toolkit.core._state import State, StateSnapshot
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.core._sync import _run_sync, _stream_sync
from ai_arch_toolkit.core._trace import Trace
from ai_arch_toolkit.toolkit.flow._scope import Scope

type ConditionFn = Callable[[StateSnapshot], bool]


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowStep:
    """A Step within a Flow, with optional dependencies and conditions."""

    step: Step
    after: tuple[str, ...] = ()
    when: ConditionFn | None = None
    scope: Scope | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowResult:
    """Result of a complete Flow execution."""

    state: State
    trace: Trace
    results: dict[str, Result] = field(default_factory=dict)

    @property
    def final_result(self) -> Result | None:
        """Last non-skipped result."""
        for st in reversed(self.trace.steps):
            if not st.skipped and st.name in self.results:
                return self.results[st.name]
        return None

    @property
    def total_cost(self) -> float:
        return self.trace.total_cost

    @property
    def total_duration(self) -> float:
        return self.trace.total_duration


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowEvent:
    """Streaming event from Flow execution."""

    type: Literal[
        "flow_start",
        "flow_end",
        "step_start",
        "step_end",
        "step_skipped",
        "retry",
        "fallback",
        "timeout",
        "policy_decision",
    ]
    flow_name: str = ""
    step_name: str = ""
    result: Result | None = None
    error: str | None = None
    policy_decision: str | None = None
    trace: Trace | None = None


class Flow:
    """Composable orchestration of Steps with DAG, sequential, and cyclic modes."""

    __slots__ = ("_is_dag", "_max_iterations", "_name", "_policy", "_scope", "_steps")

    def __init__(
        self,
        *steps: FlowStep | Step | Flow,
        name: str = "flow",
        policy: Policy | None = None,
        scope: Scope | None = None,
        max_iterations: int | None = None,
    ) -> None:
        self._name = name
        self._policy = policy
        self._scope = scope
        self._max_iterations = max_iterations

        # Normalize inputs to FlowSteps
        flow_steps: list[FlowStep] = []
        for s in steps:
            if isinstance(s, FlowStep):
                flow_steps.append(s)
            elif isinstance(s, Step):
                flow_steps.append(FlowStep(step=s))
            elif isinstance(s, Flow):
                converted = s.as_step()
                flow_steps.append(FlowStep(step=converted))
            else:
                raise TypeError(f"Expected FlowStep, Step, or Flow, got {type(s).__name__}")

        # Validate unique names
        names = [fs.step.name for fs in flow_steps]
        seen: set[str] = set()
        for n in names:
            if n in seen:
                raise ValueError(f"Duplicate step name: {n!r}")
            seen.add(n)

        # Validate after references
        name_set = set(names)
        for fs in flow_steps:
            for dep in fs.after:
                if dep not in name_set:
                    raise ValueError(f"Step {fs.step.name!r} depends on unknown step {dep!r}")

        self._steps = tuple(flow_steps)

        # Determine if this is a DAG (has after dependencies)
        has_deps = any(fs.after for fs in flow_steps)
        has_conditions = any(fs.when is not None for fs in flow_steps)
        self._is_dag = has_deps

        # Require max_iterations when `when` conditions make a flow cyclic
        if has_conditions and not has_deps and max_iterations is None:
            raise ValueError(
                "Flow has `when` conditions without `after` dependencies, which makes it "
                "cyclic. You must set `max_iterations` to limit the loop."
            )

        # Cycle detection for DAGs
        if self._is_dag:
            self._validate_dag(flow_steps)

    @property
    def name(self) -> str:
        return self._name

    @property
    def steps(self) -> tuple[FlowStep, ...]:
        return self._steps

    @property
    def policy(self) -> Policy | None:
        return self._policy

    @property
    def scope(self) -> Scope | None:
        return self._scope

    @property
    def is_dag(self) -> bool:
        return self._is_dag

    @property
    def step_names(self) -> tuple[str, ...]:
        return tuple(fs.step.name for fs in self._steps)

    @property
    def max_iterations(self) -> int | None:
        return self._max_iterations

    async def run(self, state: State) -> FlowResult:
        """Execute the flow."""
        from ai_arch_toolkit.toolkit.flow._executor import execute_flow

        return await execute_flow(self, state)

    def run_sync(self, state: State) -> FlowResult:
        """Synchronous wrapper for run()."""
        return _run_sync(self.run(state))

    async def iter(self, state: State) -> AsyncIterator[FlowEvent]:
        """Stream events during flow execution."""
        from ai_arch_toolkit.toolkit.flow._executor import iter_flow

        async for event in iter_flow(self, state):
            yield event

    def iter_sync(self, state: State) -> Iterator[FlowEvent]:
        """Synchronous wrapper for iter()."""
        return _stream_sync(lambda: self.iter(state))

    def as_step(self) -> Step:
        """Wrap this Flow as a Step for composition."""
        flow = self

        async def _run_flow(snapshot: StateSnapshot) -> Result:
            # Collect original keys to diff against later
            original_keys = set(snapshot.operational)
            state = State(
                current=dict(snapshot.current),
                operational=dict(snapshot.operational),
                persistent=dict(snapshot.persistent),
                # Share by reference. State stores world as ``dict[str, Any]``
                # but only reads from it, so a MappingProxyType is safe in practice.
                world=cast(dict[str, Any], snapshot.world),
            )
            flow_result = await flow.run(state)
            final = flow_result.final_result

            # Only return new/changed artifacts, not the entire operational layer
            new_artifacts: dict[str, Any] = {}
            for k, v in state._operational.items():
                if k not in original_keys or v is not snapshot.operational.get(k):
                    new_artifacts[k] = v

            return Result(
                value=final.value if final else None,
                artifacts=new_artifacts,
                usage=flow_result.trace.total_usage,
                cost=flow_result.total_cost,
                confidence=flow_result.trace.confidence,
                error=final.error if final else None,
                duration=flow_result.total_duration,
            )

        return Step(name=self._name, fn=_run_flow, policy=self._policy, scope=self._scope)

    @staticmethod
    def _validate_dag(steps: list[FlowStep]) -> None:
        """Topological sort to detect cycles."""
        adj: dict[str, list[str]] = {fs.step.name: [] for fs in steps}
        in_degree: dict[str, int] = {fs.step.name: 0 for fs in steps}
        for fs in steps:
            for dep in fs.after:
                adj[dep].append(fs.step.name)
                in_degree[fs.step.name] += 1

        queue = deque(n for n, d in in_degree.items() if d == 0)
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited != len(steps):
            raise ValueError("Flow contains a cycle in step dependencies")
