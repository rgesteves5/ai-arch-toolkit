"""ReWOO (Reasoning WithOut Observation) agent implementation."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.agents._base import (
    AgentResult,
    AgentStep,
    BaseAgent,
    _accumulate_usage,
    _fire_event,
)
from ai_arch_toolkit.llm._types import Message, Response, ToolCall, ToolResult, Usage

_PLAN_PROMPT = (
    "You are a planner. Given the task, create a plan of tool calls. "
    "Use the format: #E<N> = <tool_name>[<input>]\n"
    "You can reference previous results using #E<N>.\n"
    "Available tools: {tools}\n\nTask: {task}"
)

_SOLVE_PROMPT = (
    "Given the task and the results of the executed plan, "
    "provide a final answer.\n\n"
    "Task: {task}\n\n"
    "Results:\n{results}"
)

_PLAN_RE = re.compile(r"#E(\d+)\s*=\s*(\w[\w.]*)\[(.+?)\]", re.DOTALL)


@dataclass
class _PlanStep:
    id: str
    tool_name: str
    tool_input: str
    result: str = ""


class ReWOOAgent(BaseAgent):
    """ReWOO agent that separates planning from execution.

    1. Planner: LLM generates a plan with placeholder refs.
    2. Worker: Execute each step, substituting prior results.
    3. Solver: LLM synthesizes final answer from all results.
    """

    def _build_plan_prompt(self, task: str) -> str:
        tool_names = ", ".join(t.name for t in self.tools.definitions())
        return _PLAN_PROMPT.format(tools=tool_names, task=task)

    def _build_solve_prompt(self, task: str, plan_steps: list[_PlanStep]) -> str:
        results_text = "\n".join(f"#{ps.id}: {ps.result}" for ps in plan_steps)
        return _SOLVE_PROMPT.format(task=task, results=results_text)

    def run(self, task: str, **kwargs: Any) -> AgentResult:
        """Run the ReWOO plan-execute-solve loop."""
        system = self.config.system or None
        total_usage = Usage()
        steps: list[AgentStep] = []
        start = time.monotonic()

        # Phase 1: Plan
        _fire_event(self.config, "step_start", step_number=1)
        plan_resp = self.client.chat(
            [Message(role="user", content=self._build_plan_prompt(task))],
            system=system,
            **kwargs,
        )
        total_usage = _accumulate_usage(total_usage, plan_resp.usage)
        steps.append(AgentStep(step_number=1, response=plan_resp))
        plan_steps = _parse_plan(plan_resp.text)
        _fire_event(
            self.config,
            "plan_created",
            step_number=1,
            result=plan_resp.text,
        )
        _fire_event(self.config, "step_end", step_number=1)

        # Phase 2: Execute
        if self._check_timeout(start):
            return AgentResult(
                answer="[timeout exceeded]",
                steps=tuple(steps),
                total_usage=total_usage,
            )
        _fire_event(self.config, "step_start", step_number=2)
        results_map: dict[str, str] = {}
        tool_results: list[ToolResult] = []
        for ps in plan_steps:
            if self._check_timeout(start):
                return AgentResult(
                    answer="[timeout exceeded]",
                    steps=tuple(steps),
                    total_usage=total_usage,
                )
            # Substitute #E references
            resolved_input = _substitute_refs(ps.tool_input, results_map)
            tc = ToolCall(
                id=ps.id,
                name=ps.tool_name,
                arguments={"input": resolved_input},
            )
            _fire_event(
                self.config,
                "tool_call",
                step_number=2,
                tool_name=ps.tool_name,
                tool_args={"input": resolved_input},
            )
            try:
                result_str = self.tools.execute(tc)
            except Exception as exc:
                result_str = f"Error: {exc}"
                _fire_event(
                    self.config,
                    "error",
                    step_number=2,
                    tool_name=ps.tool_name,
                    error=str(exc),
                )
            ps.result = result_str
            results_map[ps.id] = result_str
            _fire_event(
                self.config,
                "tool_result",
                step_number=2,
                tool_name=ps.tool_name,
                result=result_str,
            )
            tool_results.append(
                ToolResult(
                    tool_call_id=ps.id,
                    name=ps.tool_name,
                    content=result_str,
                )
            )
        steps.append(
            AgentStep(
                step_number=2,
                response=Response(text="", usage=Usage()),
                tool_results=tuple(tool_results),
            )
        )
        _fire_event(self.config, "step_end", step_number=2)

        # Phase 3: Solve
        _fire_event(self.config, "step_start", step_number=3)
        solve_resp = self.client.chat(
            [Message(role="user", content=self._build_solve_prompt(task, plan_steps))],
            system=system,
            **kwargs,
        )
        total_usage = _accumulate_usage(total_usage, solve_resp.usage)
        steps.append(AgentStep(step_number=3, response=solve_resp))
        _fire_event(self.config, "step_end", step_number=3)

        return AgentResult(
            answer=solve_resp.text,
            steps=tuple(steps),
            total_usage=total_usage,
        )

    async def async_run(self, task: str, **kwargs: Any) -> AgentResult:
        """Run the ReWOO plan-execute-solve loop asynchronously."""
        system = self.config.system or None
        total_usage = Usage()
        steps: list[AgentStep] = []
        start = time.monotonic()

        # Phase 1: Plan
        _fire_event(self.config, "step_start", step_number=1)
        plan_resp = await self.client.chat(
            [Message(role="user", content=self._build_plan_prompt(task))],
            system=system,
            **kwargs,
        )
        total_usage = _accumulate_usage(total_usage, plan_resp.usage)
        steps.append(AgentStep(step_number=1, response=plan_resp))
        plan_steps = _parse_plan(plan_resp.text)
        _fire_event(
            self.config,
            "plan_created",
            step_number=1,
            result=plan_resp.text,
        )
        _fire_event(self.config, "step_end", step_number=1)

        # Phase 2: Execute
        if self._check_timeout(start):
            return AgentResult(
                answer="[timeout exceeded]",
                steps=tuple(steps),
                total_usage=total_usage,
            )
        _fire_event(self.config, "step_start", step_number=2)
        results_map: dict[str, str] = {}
        tool_results: list[ToolResult] = []
        for ps in plan_steps:
            if self._check_timeout(start):
                return AgentResult(
                    answer="[timeout exceeded]",
                    steps=tuple(steps),
                    total_usage=total_usage,
                )
            resolved_input = _substitute_refs(ps.tool_input, results_map)
            tc = ToolCall(
                id=ps.id,
                name=ps.tool_name,
                arguments={"input": resolved_input},
            )
            _fire_event(
                self.config,
                "tool_call",
                step_number=2,
                tool_name=ps.tool_name,
                tool_args={"input": resolved_input},
            )
            try:
                result_str = await self.tools.async_execute(tc)
            except Exception as exc:
                result_str = f"Error: {exc}"
                _fire_event(
                    self.config,
                    "error",
                    step_number=2,
                    tool_name=ps.tool_name,
                    error=str(exc),
                )
            ps.result = result_str
            results_map[ps.id] = result_str
            _fire_event(
                self.config,
                "tool_result",
                step_number=2,
                tool_name=ps.tool_name,
                result=result_str,
            )
            tool_results.append(
                ToolResult(
                    tool_call_id=ps.id,
                    name=ps.tool_name,
                    content=result_str,
                )
            )
        steps.append(
            AgentStep(
                step_number=2,
                response=Response(text="", usage=Usage()),
                tool_results=tuple(tool_results),
            )
        )
        _fire_event(self.config, "step_end", step_number=2)

        # Phase 3: Solve
        _fire_event(self.config, "step_start", step_number=3)
        solve_resp = await self.client.chat(
            [Message(role="user", content=self._build_solve_prompt(task, plan_steps))],
            system=system,
            **kwargs,
        )
        total_usage = _accumulate_usage(total_usage, solve_resp.usage)
        steps.append(AgentStep(step_number=3, response=solve_resp))
        _fire_event(self.config, "step_end", step_number=3)

        return AgentResult(
            answer=solve_resp.text,
            steps=tuple(steps),
            total_usage=total_usage,
        )


def _parse_plan(text: str) -> list[_PlanStep]:
    """Parse plan text into steps: #E1 = tool[input]."""
    plan_steps: list[_PlanStep] = []
    for match in _PLAN_RE.finditer(text):
        step_id = f"E{match.group(1)}"
        tool_name = match.group(2)
        tool_input = match.group(3).strip()
        plan_steps.append(
            _PlanStep(
                id=step_id,
                tool_name=tool_name,
                tool_input=tool_input,
            )
        )
    return plan_steps


def _substitute_refs(text: str, results: dict[str, str]) -> str:
    """Replace #E1, #E2 etc. with actual results."""
    for ref_id, result in results.items():
        text = text.replace(f"#{ref_id}", result)
    return text
