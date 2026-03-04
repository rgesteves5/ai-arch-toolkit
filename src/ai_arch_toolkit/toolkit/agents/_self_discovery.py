"""SelfDiscoveryAgent — Select → Adapt → Operationalize → Solve."""

from __future__ import annotations

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
# Default reasoning modules (from the Self-Discover paper)
# ---------------------------------------------------------------------------

_DEFAULT_MODULES: tuple[str, ...] = (
    "Critical thinking: Analyze assumptions and evaluate evidence objectively.",
    "Creative thinking: Generate novel ideas and unconventional approaches.",
    "Systems thinking: Understand how components interact within a whole.",
    "Analogical reasoning: Draw parallels from similar problems or domains.",
    "Causal reasoning: Identify cause-and-effect relationships.",
    "Step-by-step decomposition: Break the problem into smaller sub-problems.",
    "Abstraction & generalization: Identify core patterns beyond specifics.",
    "Constraint identification: Recognize limitations and boundaries.",
    "Hypothesis testing: Formulate and test potential explanations.",
    "Risk assessment: Evaluate potential pitfalls and their likelihood.",
)

# ---------------------------------------------------------------------------
# Default system prompts
# ---------------------------------------------------------------------------

_DEFAULT_SELECT_SYSTEM = (
    "You are a reasoning strategist. Given a task, select the most relevant "
    "reasoning modules from the list provided. Return only the selected modules, "
    "one per line."
)

_DEFAULT_ADAPT_SYSTEM = (
    "You are a reasoning strategist. Adapt the selected reasoning modules to "
    "be specific to the given task. For each module, describe how it applies. "
    "Return the adapted modules, one per line."
)

_DEFAULT_PLAN_SYSTEM = (
    "You are a reasoning planner. Given a task and adapted reasoning modules, "
    "create a step-by-step reasoning plan that integrates the modules. The plan "
    "should guide a solver to arrive at the answer."
)

_DEFAULT_SOLVE_SYSTEM = (
    "You are a problem solver. Follow the reasoning plan below to solve the task. "
    "Use the available tools when needed.\n\n"
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class SelfDiscoveryConfig:
    """Configuration specific to the Self-Discovery agent."""

    modules: tuple[str, ...] = _DEFAULT_MODULES
    select_system: str = _DEFAULT_SELECT_SYSTEM
    adapt_system: str = _DEFAULT_ADAPT_SYSTEM
    plan_system: str = _DEFAULT_PLAN_SYSTEM
    solve_system: str = _DEFAULT_SOLVE_SYSTEM
    reasoning: PhaseConfig | None = None
    solver: PhaseConfig | None = None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class SelfDiscoveryAgent(BaseAgent):
    """Self-Discovery: select reasoning modules → adapt → plan → solve.

    Four phases:
    1. **Select** — LLM picks the most relevant reasoning modules.
    2. **Adapt** — LLM adapts selected modules to the specific task.
    3. **Operationalize** — LLM creates a step-by-step reasoning plan.
    4. **Solve** — Inner ReActAgent follows the plan with tool access.
    """

    __slots__ = ("self_discovery",)

    def __init__(
        self,
        llm: LLM,
        tools: ToolGroup,
        *,
        config: AgentConfig | None = None,
        self_discovery: SelfDiscoveryConfig | None = None,
    ) -> None:
        super().__init__(llm, tools, config=config)
        self.self_discovery = self_discovery or SelfDiscoveryConfig()

    async def _run_loop(self, task: Content, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        task_str = task if isinstance(task, str) else str(task)
        sd = self.self_discovery
        step_num = 0
        user_sys = self.config.system

        # ---------------------------------------------------------------
        # Phase 1: SELECT
        # ---------------------------------------------------------------
        step_num += 1
        yield AgentEvent(type="step_start", step=step_num)

        modules_text = "\n".join(f"- {m}" for m in sd.modules)
        select_msg = (
            f"Task: {task_str}\n\n"
            f"Available reasoning modules:\n{modules_text}\n\n"
            f"Select the most relevant modules for this task."
        )
        select_system = (user_sys + "\n\n" + sd.select_system) if user_sys else sd.select_system
        reasoning_llm = _resolve_llm(sd.reasoning, self.llm)
        select_response = await reasoning_llm.complete(
            [user(select_msg)],
            system=select_system,
        )
        yield AgentEvent(type="step_end", step=step_num, response=select_response)
        selected = select_response.text

        # ---------------------------------------------------------------
        # Phase 2: ADAPT
        # ---------------------------------------------------------------
        step_num += 1
        yield AgentEvent(type="step_start", step=step_num)

        adapt_msg = (
            f"Task: {task_str}\n\n"
            f"Selected reasoning modules:\n{selected}\n\n"
            f"Adapt these modules to be specific to the task."
        )
        adapt_system = (user_sys + "\n\n" + sd.adapt_system) if user_sys else sd.adapt_system
        adapt_response = await reasoning_llm.complete(
            [user(adapt_msg)],
            system=adapt_system,
        )
        yield AgentEvent(type="step_end", step=step_num, response=adapt_response)
        adapted = adapt_response.text

        # ---------------------------------------------------------------
        # Phase 3: OPERATIONALIZE
        # ---------------------------------------------------------------
        step_num += 1
        yield AgentEvent(type="step_start", step=step_num)

        plan_msg = (
            f"Task: {task_str}\n\n"
            f"Adapted reasoning modules:\n{adapted}\n\n"
            f"Create a step-by-step reasoning plan that integrates these modules."
        )
        plan_system = (user_sys + "\n\n" + sd.plan_system) if user_sys else sd.plan_system
        plan_response = await reasoning_llm.complete(
            [user(plan_msg)],
            system=plan_system,
        )
        yield AgentEvent(type="step_end", step=step_num, response=plan_response)
        reasoning_plan = plan_response.text

        # ---------------------------------------------------------------
        # Phase 4: SOLVE (inner ReAct)
        # ---------------------------------------------------------------
        step_num += 1
        yield AgentEvent(type="step_start", step=step_num)

        solve_base = (user_sys + "\n\n" + sd.solve_system) if user_sys else sd.solve_system
        solve_system = solve_base + f"Reasoning plan:\n{reasoning_plan}\n\n" + f"Task: {task_str}"
        inner_config = AgentConfig(
            max_iterations=self.config.max_iterations,
            system=solve_system,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
            tool_choice=self.config.tool_choice,
            parallel_tool_calls=self.config.parallel_tool_calls,
            llm_kwargs=self.config.llm_kwargs,
            output_schema=self.config.output_schema,
        )
        solver_llm = _resolve_llm(sd.solver, self.llm)
        solver_tools = _resolve_tools(sd.solver, self.tools)
        inner = ReActAgent(solver_llm, solver_tools, config=inner_config)

        last_response = None
        inner_stop: str | None = None

        async for event in inner._run_loop(task):
            if event.type == "step_end":
                if event.stop_reason is not None:
                    inner_stop = event.stop_reason
                if event.response is not None:
                    last_response = event.response

            # Re-number inner events but skip terminal step_end
            if event.type == "step_end" and event.stop_reason is not None:
                # Propagate hard stops immediately
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

        if inner_stop in ("timeout", "error", "budget_exhausted"):
            return

        # Emit final step_end for solve phase
        yield AgentEvent(
            type="step_end",
            step=step_num,
            response=last_response,
            stop_reason="completed",
        )
