"""PlanExecute as a Flow — Plan → per-step ReAct execution → Solve, with replanning."""

from __future__ import annotations

import re
from typing import Any, Unpack

from ai_arch_toolkit.core._content import Content, user
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._state import StateSnapshot
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents.flows._common import FlowOptions, substitute_tools
from ai_arch_toolkit.toolkit.agents.flows._keys import ANSWER, RESPONSE, TASK
from ai_arch_toolkit.toolkit.agents.flows._react import run_react
from ai_arch_toolkit.toolkit.flow._flow import Flow

_STEP_RE = re.compile(r"^\d+\.\s+(.+)", re.MULTILINE)


def plan_execute_flow(
    llm: LLM,
    tools: ToolGroup,
    *,
    system: str = "",
    max_replans: int = 1,
    max_iterations_per_step: int = 3,
    planner_system: str = (
        "You are a planning agent. Break the task into numbered steps.\n"
        "Format: 1. First step\n2. Second step\n...\n\n"
        "Available tools:\n{tools}"
    ),
    solver_system: str = (
        "You are a solving agent. Given the task, plan, and results "
        "from each step, provide the final answer."
    ),
    llm_kwargs: dict[str, Any] | None = None,
    planner_llm: LLM | None = None,
    exec_llm: LLM | None = None,
    exec_tools: ToolGroup | None = None,
    solver_llm: LLM | None = None,
    **options: Unpack[FlowOptions],
) -> Flow:
    """Create a PlanExecute Flow — plan, execute each step via ReAct, solve.

    The plan+execute cycle handles replanning internally. The flow is two
    sequential steps: plan_and_execute → solve. ``llm_kwargs`` apply to every
    phase's LLM calls; ``planner_llm``/``exec_llm``/``exec_tools``/``solver_llm``
    override the default LLM and tools per phase. A ``{tools}`` token in
    ``planner_system`` is replaced with the executor's rendered tool catalog;
    a prompt without the token is never modified.

    ``options`` are the options of the ``Flow`` it builds (``FlowOptions``).
    """
    plan_llm = planner_llm or llm
    inner_llm = exec_llm or llm
    inner_tools = exec_tools if exec_tools is not None else tools
    solve_llm = solver_llm or llm
    extra = llm_kwargs or {}
    # "{tools}" resolves against the executor's tools, so the plan matches what
    # the execution phase can actually call.
    plan_system = substitute_tools(planner_system, inner_tools)

    async def plan_and_execute(snap: StateSnapshot) -> Result:
        """Plan, execute each step, and optionally replan."""
        task: str = snap.require(TASK)
        plan_text = ""
        step_results: list[str] = []

        for _attempt in range(max_replans + 1):
            # PLAN
            response = await plan_llm.complete([user(task)], system=plan_system, **extra)
            plan_text = response.text
            planned_steps = _STEP_RE.findall(plan_text)

            # EXECUTE each step
            step_results = []
            any_failed = False

            for step_desc in planned_steps:
                prev_results = "\n".join(f"Step {j + 1}: {r}" for j, r in enumerate(step_results))

                inner_system = system
                if inner_system:
                    inner_system += "\n\n"
                inner_system += f"Current step: {step_desc}"
                if prev_results:
                    inner_system += f"\n\nPrevious results:\n{prev_results}"

                run = await run_react(
                    inner_llm,
                    inner_tools,
                    step_desc,
                    system=inner_system,
                    max_iterations=max_iterations_per_step,
                    llm_kwargs=llm_kwargs,
                    **options,
                )
                step_results.append(run.answer)
                any_failed = any_failed or run.failed

            # CHECK REPLAN
            if not any_failed or _attempt >= max_replans:
                break

        return Result(
            value=step_results,
            artifacts={
                "plan_text": plan_text,
                "step_results": step_results,
            },
        )

    async def solve(snap: StateSnapshot) -> Result:
        """Synthesize final answer from task, plan, and results."""
        task: str = snap.require(TASK)
        plan_text: str = snap.get("plan_text", "")
        step_results: list[str] = snap.get("step_results", [])

        results_block = "\n".join(f"Step {i + 1}: {r}" for i, r in enumerate(step_results))

        response = await solve_llm.complete(
            [user(f"Task: {task}\n\nPlan:\n{plan_text}\n\nResults:\n{results_block}")],
            system=solver_system,
            **extra,
        )

        return Result(
            value=response.text,
            artifacts={ANSWER: response.text, RESPONSE: response},
        )

    return Flow(
        Step(name="plan_and_execute", fn=plan_and_execute),
        Step(name="solve", fn=solve),
        name="plan_execute",
        **options,
    )


def plan_execute_initial_state(task: Content) -> dict[str, Any]:
    """Create the initial operational state for a plan_execute_flow."""
    task_str = task if isinstance(task, str) else str(task)
    return {TASK: task_str}
