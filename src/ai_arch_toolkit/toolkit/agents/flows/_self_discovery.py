"""SelfDiscovery as a Flow — Select → Adapt → Operationalize → Solve."""

from __future__ import annotations

from typing import Any, Unpack

from ai_arch_toolkit.core._content import Content, user
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._state import StateSnapshot
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents.flows._common import FlowOptions
from ai_arch_toolkit.toolkit.agents.flows._keys import ANSWER, RESPONSE, TASK
from ai_arch_toolkit.toolkit.agents.flows._react import run_react
from ai_arch_toolkit.toolkit.flow._flow import Flow

_DEFAULT_MODULES = (
    "Critical Thinking: Question assumptions and evaluate evidence systematically.",
    "Creative Thinking: Generate novel ideas and unconventional approaches.",
    "Systems Thinking: Understand how components interact within the whole.",
    "Analytical Thinking: Break problems into parts and examine relationships.",
    "Analogical Reasoning: Draw parallels from similar problems or domains.",
    "Causal Reasoning: Identify cause-and-effect relationships.",
    "Probabilistic Thinking: Consider likelihoods and uncertainty.",
    "Inductive Reasoning: Derive general principles from specific observations.",
    "Deductive Reasoning: Apply general principles to specific cases.",
    "Abductive Reasoning: Find the simplest, most likely explanation.",
)


def self_discovery_flow(
    llm: LLM,
    tools: ToolGroup,
    *,
    system: str = "",
    modules: tuple[str, ...] = _DEFAULT_MODULES,
    max_react_iterations: int = 10,
    select_system: str = (
        "Select the reasoning modules most relevant to this task. List only the selected modules."
    ),
    adapt_system: str = (
        "Adapt the selected reasoning modules to be specific to this task. "
        "Rephrase each module in the context of the problem."
    ),
    plan_system: str = (
        "Operationalize the adapted reasoning modules into a step-by-step "
        "reasoning plan. Create a structured plan with clear steps."
    ),
    solve_system: str = (
        "Follow the reasoning plan to solve the task. Apply each step of the plan systematically."
    ),
    llm_kwargs: dict[str, Any] | None = None,
    reasoning_llm: LLM | None = None,
    solver_llm: LLM | None = None,
    solver_tools: ToolGroup | None = None,
    **options: Unpack[FlowOptions],
) -> Flow:
    """Create a SelfDiscovery Flow — select, adapt, plan, then solve via ReAct.

    Args:
        llm: Default language model.
        tools: Tool group for the solver ReAct phase.
        system: Base system prompt.
        modules: Reasoning module descriptions.
        max_react_iterations: Max iterations for the inner ReAct solver.
        select_system: System prompt for module selection.
        adapt_system: System prompt for module adaptation.
        plan_system: System prompt for operationalization.
        solve_system: System prompt for the solve phase.
        llm_kwargs: Additional kwargs passed to every phase's LLM call.
        reasoning_llm: Override LLM for select/adapt/plan phases.
        solver_llm: Override LLM for the solve phase.
        solver_tools: Override tools for the solve phase.
        **options: The options of the ``Flow`` it builds (``FlowOptions``).
    """
    reason_llm = reasoning_llm or llm
    solve_llm = solver_llm or llm
    solve_tools = solver_tools if solver_tools is not None else tools
    extra = llm_kwargs or {}

    modules_text = "\n".join(f"- {m}" for m in modules)

    async def select(snap: StateSnapshot) -> Result:
        """Select relevant reasoning modules."""
        task: str = snap.require(TASK)

        response = await reason_llm.complete(
            [user(f"Task: {task}\n\nAvailable modules:\n{modules_text}")],
            system=select_system,
            **extra,
        )
        return Result(
            value=response.text,
            artifacts={"selected_modules": response.text},
        )

    async def adapt(snap: StateSnapshot) -> Result:
        """Adapt selected modules to the specific task."""
        task: str = snap.require(TASK)
        selected: str = snap.require("selected_modules")

        response = await reason_llm.complete(
            [user(f"Task: {task}\n\nSelected modules:\n{selected}")],
            system=adapt_system,
            **extra,
        )
        return Result(
            value=response.text,
            artifacts={"adapted_modules": response.text},
        )

    async def operationalize(snap: StateSnapshot) -> Result:
        """Create a step-by-step reasoning plan from adapted modules."""
        task: str = snap.require(TASK)
        adapted: str = snap.require("adapted_modules")

        response = await reason_llm.complete(
            [user(f"Task: {task}\n\nAdapted modules:\n{adapted}")],
            system=plan_system,
            **extra,
        )
        return Result(
            value=response.text,
            artifacts={"reasoning_plan": response.text},
        )

    async def solve(snap: StateSnapshot) -> Result:
        """Solve the task using the reasoning plan via inner ReAct."""
        task: str = snap.require(TASK)
        reasoning_plan: str = snap.require("reasoning_plan")

        inner_system = solve_system
        if inner_system:
            inner_system += "\n\n"
        inner_system += f"Reasoning plan:\n{reasoning_plan}"
        if system:
            inner_system = f"{system}\n\n{inner_system}"

        run = await run_react(
            solve_llm,
            solve_tools,
            task,
            system=inner_system,
            max_iterations=max_react_iterations,
            llm_kwargs=llm_kwargs,
            **options,
        )
        return Result(value=run.answer, artifacts={ANSWER: run.answer, RESPONSE: run.response})

    return Flow(
        Step(name="select", fn=select),
        Step(name="adapt", fn=adapt),
        Step(name="operationalize", fn=operationalize),
        Step(name="solve", fn=solve),
        name="self_discovery",
        **options,
    )


def self_discovery_initial_state(task: Content) -> dict[str, Any]:
    """Create the initial operational state for a self_discovery_flow."""
    task_str = task if isinstance(task, str) else str(task)
    return {TASK: task_str}
