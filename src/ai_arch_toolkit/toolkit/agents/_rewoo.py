"""ReWOOAgent — Plan with placeholders → Execute tools → Solve."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.core._content import Content, user
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._response import ToolCall
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents._base import AgentConfig, AgentEvent, BaseAgent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_PLANNER_SYSTEM = (
    "You are a planning assistant. Given a task and available tools, produce a "
    "step-by-step plan. Each step uses exactly one tool.\n\n"
    "Format each step as:\n"
    "#E{n} = ToolName[argument]\n\n"
    "You may reference previous results with #E{n} in arguments.\n\n"
    "Available tools:\n"
)

_DEFAULT_SOLVER_SYSTEM = (
    "You are a solving assistant. Given the original task and evidence gathered "
    "from tool executions, produce a final answer."
)

_PLAN_RE = re.compile(r"#E(\d+)\s*=\s*(\w+)\[([^\]]*)\]")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReWOOConfig:
    """Configuration specific to the ReWOO agent."""

    planner_system: str = _DEFAULT_PLANNER_SYSTEM
    solver_system: str = _DEFAULT_SOLVER_SYSTEM


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ReWOOAgent(BaseAgent):
    """ReWOO: Reasoning WithOut Observation.

    Three phases:
    1. **Plan** — LLM generates a step-by-step plan with tool calls and
       ``#E{n}`` placeholders for intermediate results.
    2. **Execute** — tools run sequentially, substituting ``#E`` references.
    3. **Solve** — LLM produces a final answer given the task + all evidence.

    Limitation: tool arguments are mapped to the *first parameter* in the
    tool's schema.  Multi-parameter tools need a wrapper or JSON string.
    """

    __slots__ = ("rewoo",)

    def __init__(
        self,
        llm: LLM,
        tools: ToolGroup,
        *,
        config: AgentConfig | None = None,
        rewoo: ReWOOConfig | None = None,
    ) -> None:
        super().__init__(llm, tools, config=config)
        self.rewoo = rewoo or ReWOOConfig()

    async def _run_loop(self, task: Content, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        # NOTE: multimodal Content is converted via str() — lossy for non-str
        # tasks.  ReWOO's text-based plan/solve prompts work best with str.
        task_str = task if isinstance(task, str) else str(task)
        step_num = 0

        # ---------------------------------------------------------------
        # Phase 1: Plan
        # ---------------------------------------------------------------
        step_num += 1
        yield AgentEvent(type="step_start", step=step_num)

        # Augment planner system with tool descriptions
        tool_lines: list[str] = []
        for defn in self.tools.definitions:
            desc = defn.get("description", "")
            tool_lines.append(f"- {defn['name']}: {desc}")
        augmented_system = self.rewoo.planner_system + "\n".join(tool_lines)

        # llm_kwargs (e.g. thinking) are intentionally NOT forwarded to the
        # planner — it only needs to produce a structured plan, not reason
        # deeply.  The solver call does forward llm_kwargs.
        plan_response = await self.llm.complete(
            [user(task_str)],
            system=augmented_system,
        )
        yield AgentEvent(type="step_end", step=step_num, response=plan_response)

        # Parse plan
        plan_steps = _PLAN_RE.findall(plan_response.text)
        # plan_steps: list of (evidence_id, tool_name, raw_args)

        # ---------------------------------------------------------------
        # Phase 2: Execute
        # ---------------------------------------------------------------
        evidence: dict[str, str] = {}

        for eid, tool_name, raw_args in plan_steps:
            step_num += 1
            yield AgentEvent(type="step_start", step=step_num)

            # Substitute #E references in args
            resolved_args = raw_args
            for ref, val in evidence.items():
                resolved_args = resolved_args.replace(ref, val)

            # Map to first parameter of the tool's schema
            first_param = _first_param_name(self.tools, tool_name)
            tc = ToolCall(
                id=f"rewoo_e{eid}",
                name=tool_name,
                input={first_param: resolved_args} if first_param else {},
            )

            yield AgentEvent(
                type="tool_call",
                step=step_num,
                tool_name=tc.name,
                tool_call_id=tc.id,
                tool_args=dict(tc.input),
            )

            try:
                result_str = await self.tools.async_execute(tc)
            except KeyError:
                result_str = f"Error: unknown tool {tool_name!r}"
                yield AgentEvent(
                    type="error",
                    step=step_num,
                    tool_name=tool_name,
                    error=result_str,
                )
            else:
                yield AgentEvent(
                    type="tool_result",
                    step=step_num,
                    tool_name=tc.name,
                    tool_call_id=tc.id,
                    result=result_str,
                )

            evidence[f"#E{eid}"] = result_str
            yield AgentEvent(type="step_end", step=step_num)

        # ---------------------------------------------------------------
        # Phase 3: Solve
        # ---------------------------------------------------------------
        step_num += 1
        yield AgentEvent(type="step_start", step=step_num)

        evidence_block = "\n".join(f"{ref}: {val}" for ref, val in sorted(evidence.items()))
        solver_msg = f"Task: {task_str}\n\nEvidence:\n{evidence_block}\n\nProduce a final answer."

        solve_response = await self.llm.complete(
            [user(solver_msg)],
            system=self.rewoo.solver_system,
            output_schema=self.config.output_schema,
            **self.config.llm_kwargs,
        )

        yield AgentEvent(
            type="step_end",
            step=step_num,
            response=solve_response,
            stop_reason="completed",
        )


def _first_param_name(tools: ToolGroup, tool_name: str) -> str:
    """Return the first parameter name from a tool's schema, or empty string."""
    for defn in tools.definitions:
        if defn["name"] == tool_name:
            props = defn.get("input_schema", {}).get("properties", {})
            if props:
                return next(iter(props))
    return ""
