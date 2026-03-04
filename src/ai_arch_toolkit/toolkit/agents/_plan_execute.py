"""PlanExecuteAgent — Plan numbered steps → Execute via ReAct → Solve."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.core._content import Content, user
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents._base import (
    AgentConfig,
    AgentEvent,
    BaseAgent,
    PhaseConfig,
    _resolve_llm,
    _resolve_tools,
)
from ai_arch_toolkit.toolkit.agents._react import ReActAgent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_PLANNER_SYSTEM = (
    "You are a planning assistant. Given a task and available tools, produce a "
    "numbered step-by-step plan. Each step should be a single action.\n\n"
    "Format:\n1. First step\n2. Second step\n...\n\n"
    "Available tools:\n"
)

_DEFAULT_SOLVER_SYSTEM = (
    "You are a solving assistant. Given the original task and results from each "
    "step, produce a final answer."
)

_STEP_RE = re.compile(r"^\d+\.\s*(.+)", re.MULTILINE)


@dataclass(frozen=True, slots=True, kw_only=True)
class PlanExecuteConfig:
    """Configuration specific to the PlanExecute agent."""

    planner_system: str = _DEFAULT_PLANNER_SYSTEM
    solver_system: str = _DEFAULT_SOLVER_SYSTEM
    max_replans: int = 1
    planner: PhaseConfig | None = None
    executor: PhaseConfig | None = None
    solver: PhaseConfig | None = None

    def __post_init__(self) -> None:
        if self.max_replans < 0:
            raise ValueError(f"max_replans must be >= 0, got {self.max_replans}")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class PlanExecuteAgent(BaseAgent):
    """Plan-Execute: numbered plan → per-step ReAct execution → final solve.

    Three phases:
    1. **Plan** — LLM generates a numbered step list.
    2. **Execute** — Each step is executed by an inner ReActAgent.
    3. **Solve** — LLM produces a final answer from all step results.

    Optional replanning on failure (``max_replans > 0``).
    """

    __slots__ = ("plan_execute",)

    def __init__(
        self,
        llm: LLM,
        tools: ToolGroup,
        *,
        config: AgentConfig | None = None,
        plan_execute: PlanExecuteConfig | None = None,
    ) -> None:
        super().__init__(llm, tools, config=config)
        self.plan_execute = plan_execute or PlanExecuteConfig()

    async def _run_loop(self, task: Content, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        task_str = task if isinstance(task, str) else str(task)
        step_num = 0
        user_sys = self.config.system

        # Resolve phase overrides once (loop-invariant).
        planner_llm = _resolve_llm(self.plan_execute.planner, self.llm)
        exec_llm = _resolve_llm(self.plan_execute.executor, self.llm)
        exec_tools = _resolve_tools(self.plan_execute.executor, self.tools)
        solver_llm = _resolve_llm(self.plan_execute.solver, self.llm)

        # Augment planner system with tool descriptions.
        # Use the executor's resolved tools so the plan matches what's available.
        tool_lines: list[str] = []
        for defn in exec_tools.definitions:
            desc = defn.get("description", "")
            tool_lines.append(f"- {defn['name']}: {desc}")
        augmented_planner = self.plan_execute.planner_system + "\n".join(tool_lines)
        if user_sys:
            augmented_planner = user_sys + "\n\n" + augmented_planner

        planned_steps: list[str] = []
        results: list[str] = []
        replan_count = 0

        while True:
            # ---------------------------------------------------------------
            # Phase 1: Plan
            # ---------------------------------------------------------------
            step_num += 1
            yield AgentEvent(type="step_start", step=step_num)

            plan_response = await planner_llm.complete(
                [user(task_str)],
                system=augmented_planner,
            )
            yield AgentEvent(type="step_end", step=step_num, response=plan_response)

            planned_steps = _STEP_RE.findall(plan_response.text)

            # ---------------------------------------------------------------
            # Phase 2: Execute each planned step
            # ---------------------------------------------------------------
            results = []
            any_failed = False

            for planned_step in planned_steps:
                step_num += 1
                yield AgentEvent(type="step_start", step=step_num)

                # Build context with prior results
                context_parts: list[str] = []
                if user_sys:
                    context_parts.append(user_sys)
                context_parts.append(f"Current step: {planned_step}")
                if results:
                    context_parts.append(
                        "Previous results:\n"
                        + "\n".join(f"  Step {i + 1}: {r}" for i, r in enumerate(results))
                    )
                inner_system = "\n\n".join(context_parts)

                inner_config = AgentConfig(
                    max_iterations=3,
                    system=inner_system,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout,
                    tool_choice=self.config.tool_choice,
                    parallel_tool_calls=self.config.parallel_tool_calls,
                    llm_kwargs=self.config.llm_kwargs,
                )
                inner = ReActAgent(exec_llm, exec_tools, config=inner_config)

                last_answer = ""
                inner_stop: str | None = None

                async for event in inner._run_loop(planned_step):
                    if event.type == "step_end":
                        if event.stop_reason is not None:
                            inner_stop = event.stop_reason
                        if event.response is not None:
                            last_answer = event.response.text

                    # Re-number inner events but skip terminal step_end
                    if event.type == "step_end" and event.stop_reason is not None:
                        if event.stop_reason in ("timeout", "error", "budget_exhausted"):
                            yield AgentEvent(
                                type=event.type,
                                step=step_num,
                                response=event.response,
                                stop_reason=event.stop_reason,
                                error=event.error,
                            )
                        continue

                    yield AgentEvent(
                        type=event.type,
                        step=step_num,
                        tool_name=event.tool_name,
                        tool_call_id=event.tool_call_id,
                        tool_args=event.tool_args,
                        result=event.result,
                        error=event.error,
                        response=event.response,
                    )

                if inner_stop in ("timeout", "budget_exhausted"):
                    return

                results.append(last_answer)
                if inner_stop == "error":
                    any_failed = True

                # Emit step_end for this planned step
                yield AgentEvent(type="step_end", step=step_num)

            # Check if replanning is needed
            if any_failed and replan_count < self.plan_execute.max_replans:
                replan_count += 1
                continue  # Re-plan
            break

        # ---------------------------------------------------------------
        # Phase 3: Solve
        # ---------------------------------------------------------------
        step_num += 1
        yield AgentEvent(type="step_start", step=step_num)

        plan_block = (
            "\n".join(f"{i + 1}. {s}" for i, s in enumerate(planned_steps))
            if planned_steps
            else "(no steps planned)"
        )
        results_block = (
            "\n".join(f"Step {i + 1}: {r}" for i, r in enumerate(results))
            if results
            else "(no steps executed)"
        )
        solver_msg = (
            f"Task: {task_str}\n\n"
            f"Plan:\n{plan_block}\n\n"
            f"Step results:\n{results_block}\n\n"
            f"Produce a final answer."
        )

        solver_system = self.plan_execute.solver_system
        if user_sys:
            solver_system = user_sys + "\n\n" + solver_system

        solve_response = await solver_llm.complete(
            [user(solver_msg)],
            system=solver_system,
            output_schema=self.config.output_schema,
            **self.config.llm_kwargs,
        )

        yield AgentEvent(
            type="step_end",
            step=step_num,
            response=solve_response,
            stop_reason="completed",
        )
