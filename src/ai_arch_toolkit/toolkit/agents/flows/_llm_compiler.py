"""LLMCompiler as a Flow — Plan DAG → Parallel execute → Join, with optional replan."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.core._budget import BudgetExceeded, BudgetPolicy, BudgetState
from ai_arch_toolkit.core._content import Content, user
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._policy import Policy
from ai_arch_toolkit.core._state import State, StateSnapshot
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents.flows._react import react_flow, react_initial_state
from ai_arch_toolkit.toolkit.flow._flow import Flow

_DAG_RE = re.compile(r"\$(\d+)\.\s+(.+?)\s+\[deps:\s*(.*?)\]")


@dataclass(slots=True)
class _DAGTask:
    """A task in the compiler DAG."""

    id: int
    description: str
    deps: tuple[int, ...]
    result: str = ""
    done: bool = False
    failed: bool = False


def _ready_tasks(dag: list[_DAGTask]) -> list[_DAGTask]:
    """Find tasks ready to execute (all deps done, not yet done itself)."""
    done_ids = {t.id for t in dag if t.done}
    failed_ids = {t.id for t in dag if t.failed}
    ready: list[_DAGTask] = []

    for t in dag:
        if t.done or t.failed:
            continue
        if any(d in failed_ids for d in t.deps):
            t.failed = True
            continue
        if all(d in done_ids for d in t.deps):
            ready.append(t)

    return ready


def llm_compiler_flow(
    llm: LLM,
    tools: ToolGroup,
    *,
    system: str = "",
    max_replans: int = 2,
    max_react_iterations: int = 3,
    planner_system: str = (
        "You are a planning agent. Break the task into a DAG of subtasks.\n"
        "Format each task as:\n$1. Task description [deps: none]\n"
        "$2. Task description [deps: $1]\n$3. Task description [deps: $1, $2]\n"
        "Use $N references for dependencies."
    ),
    joiner_system: str = (
        "You are a synthesis agent. Given the task and results from all "
        "subtasks, provide the final answer.\n"
        "If the results are insufficient, start your response with REPLAN "
        "followed by what needs to change."
    ),
    timeout: float | None = None,
    policy: Policy | None = None,
    budget_policy: BudgetPolicy | None = None,
    planner_llm: LLM | None = None,
    exec_tools: ToolGroup | None = None,
    joiner_llm: LLM | None = None,
) -> Flow:
    """Create an LLMCompiler Flow — plan DAG, parallel execute, join/replan.

    The plan-execute-join cycle with replanning is handled internally in a
    single step. The flow is sequential: compile → (result with answer).
    """
    plan_llm = planner_llm or llm
    inner_tools = exec_tools or tools
    join_llm = joiner_llm or llm

    async def compile(snap: StateSnapshot) -> Result:
        """Plan DAG, execute in parallel, join — with optional replanning."""
        task: str = snap.require("task")
        total_cost = 0.0
        budget_state = _budget_state_from_snapshot(snap)

        async def _complete(model: LLM, *args: Any, **kwargs: Any):
            nonlocal budget_state
            if budget_state is not None:
                budget_state.check_llm_calls()
            response = await model.complete(*args, **kwargs)
            if budget_state is not None:
                budget_state = budget_state.record_response(response)
            return response

        def _budget_artifacts() -> dict[str, Any]:
            return {"budget_state": budget_state} if budget_state is not None else {}

        def _budget_error(exc: BudgetExceeded) -> Result:
            if budget_state is None:
                return Result(error=str(exc), artifacts={"budget_exceeded": exc.to_dict()})
            exceeded = budget_state.with_exceeded(exc)
            return Result(
                error=str(exc),
                artifacts={
                    "budget_exceeded": exc.to_dict(),
                    "budget_state": exceeded,
                },
            )

        for _replan in range(max_replans + 1):
            # PLAN
            try:
                response = await _complete(plan_llm, [user(task)], system=planner_system)
            except BudgetExceeded as exc:
                return _budget_error(exc)
            total_cost += response.cost or 0.0

            dag: list[_DAGTask] = []
            valid_ids: set[int] = set()
            for match in _DAG_RE.finditer(response.text):
                task_id = int(match.group(1))
                desc = match.group(2)
                deps_str = match.group(3).strip()

                if deps_str.lower() == "none" or not deps_str:
                    deps: tuple[int, ...] = ()
                else:
                    dep_ids = [int(d) for d in re.findall(r"\$(\d+)", deps_str)]
                    deps = tuple(dep_ids)

                valid_ids.add(task_id)
                dag.append(_DAGTask(id=task_id, description=desc, deps=deps))

            for t in dag:
                for d in t.deps:
                    if d not in valid_ids:
                        t.failed = True

            # EXECUTE
            while True:
                ready = _ready_tasks(dag)
                if not ready:
                    break

                async def _run_one(
                    t: _DAGTask,
                    dag_ref: list[_DAGTask] = dag,
                    run_budget_state: BudgetState | None = budget_state,
                ) -> tuple[float, BudgetState | None]:
                    desc = t.description
                    for other in dag_ref:
                        if other.done:
                            desc = desc.replace(f"${other.id}", other.result)

                    inner_system = system
                    if inner_system:
                        inner_system += "\n\n"
                    inner_system += f"Subtask: {desc}"

                    inner = react_flow(
                        llm,
                        inner_tools,
                        system=inner_system,
                        max_iterations=max_react_iterations,
                        budget_policy=(
                            run_budget_state.policy if run_budget_state is not None else None
                        ),
                    )

                    initial = react_initial_state(task)
                    if run_budget_state is not None:
                        initial["budget_state"] = run_budget_state
                    state = State(operational=initial)
                    result = await inner.run(state)
                    next_budget = state.get("budget_state")
                    nonlocal_budget = next_budget if isinstance(next_budget, BudgetState) else None

                    inner_resp = state.get("response")
                    t.result = inner_resp.text if inner_resp else ""
                    t.done = True

                    if result.trace.steps and any(
                        st.error is not None for st in result.trace.steps if not st.skipped
                    ):
                        t.failed = True

                    return result.total_cost, nonlocal_budget

                run_results = await asyncio.gather(*[_run_one(t) for t in ready])
                total_cost += sum(cost for cost, _budget in run_results)
                for _cost, next_budget in run_results:
                    if next_budget is not None:
                        budget_state = next_budget
                        if budget_state.exceeded is not None:
                            return _budget_error(budget_state.exceeded)

            # JOIN
            results_block = "\n".join(
                f"${t.id}. {t.description}: {t.result}" for t in dag if t.done
            )
            try:
                join_response = await _complete(
                    join_llm,
                    [user(f"Task: {task}\n\nResults:\n{results_block}")],
                    system=joiner_system,
                )
            except BudgetExceeded as exc:
                return _budget_error(exc)
            total_cost += join_response.cost or 0.0

            first_line = join_response.text.strip().split("\n")[0]
            if "REPLAN" not in first_line.upper() or _replan >= max_replans:
                return Result(
                    value=join_response.text,
                    artifacts={
                        "answer": join_response.text,
                        "response": join_response,
                        **_budget_artifacts(),
                    },
                    cost=total_cost,
                )

        # Should not reach here, but satisfy type checker
        return Result(
            error="Max replans exhausted",
            cost=total_cost,
            artifacts=_budget_artifacts(),
        )

    flow_policy = policy
    if timeout is not None and flow_policy is None:
        flow_policy = Policy(timeout=timeout)

    return Flow(
        Step(name="compile", fn=compile),
        name="llm_compiler",
        policy=flow_policy,
        budget_policy=budget_policy,
    )


def llm_compiler_initial_state(task: Content) -> dict[str, Any]:
    """Create the initial operational state for a llm_compiler_flow."""
    task_str = task if isinstance(task, str) else str(task)
    return {"task": task_str}


def _budget_state_from_snapshot(snap: StateSnapshot) -> BudgetState | None:
    value = snap.get("budget_state")
    return value if isinstance(value, BudgetState) else None
