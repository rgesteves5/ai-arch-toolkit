"""ReWOO (Reasoning WithOut Observation) agent implementation."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.agents._base import (
    AgentResult,
    AgentStep,
    BaseAgent,
    ReWOOResult,
    _accumulate_usage,
    _fire_event,
)
from ai_arch_toolkit.llm._types import Message, Response, ToolCall, ToolResult, Usage

_PLAN_PROMPT = (
    "You are a planner. Given the task, create a JSON array of tool-call steps. "
    "Each item must be an object with keys: id, tool, input.\n"
    "Example: [{{\"id\":\"E1\",\"tool\":\"search\",\"input\":\"query\"}}].\n"
    "You can reference previous results using #E<N> inside the input.\n"
    "Return ONLY JSON.\n"
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
        stream = kwargs.pop("stream", False)
        cancellation_token = self._resolve_cancellation_token(
            kwargs.pop("cancellation_token", None)
        )
        if stream:
            return self.run_stream(
                task,
                cancellation_token=cancellation_token,
                **kwargs,
            )
        system = self.config.system or None
        total_usage = Usage()
        budget = self._new_budget_manager()
        steps: list[AgentStep] = []
        start = time.monotonic()

        def _record_repair_response(step_number: int, response: Response) -> None:
            nonlocal total_usage
            total_usage = _accumulate_usage(total_usage, response.usage)
            repair_cost = self._observe_response(budget, response, step_number=step_number)
            steps.append(
                AgentStep(
                    step_number=step_number,
                    response=response,
                    usage=response.usage,
                    cost_usd=repair_cost,
                    metadata={"repair_attempt": True},
                )
            )

        # Phase 1: Plan
        _fire_event(self.config, "step_start", step_number=1)
        plan_resp = self.client.chat(
            [Message(role="user", content=self._build_plan_prompt(task))],
            system=system,
            **kwargs,
        )
        total_usage = _accumulate_usage(total_usage, plan_resp.usage)
        plan_cost = self._observe_response(budget, plan_resp, step_number=1)
        steps.append(
            AgentStep(step_number=1, response=plan_resp, usage=plan_resp.usage, cost_usd=plan_cost)
        )
        plan_steps = self._parse_plan_with_fallback(
            plan_resp.text,
            system=system,
            step_number=1,
            on_repair_response=_record_repair_response,
            **kwargs,
        )
        _fire_event(
            self.config,
            "plan_created",
            step_number=1,
            result=plan_resp.text,
        )
        _fire_event(self.config, "step_end", step_number=1)

        # Phase 2: Execute
        if self._is_cancelled(cancellation_token):
            return self._finalize_result(
                ReWOOResult(
                    answer="[cancelled]",
                    steps=tuple(steps),
                    total_usage=total_usage,
                    stop_reason="cancelled",
                ),
                result_type=ReWOOResult,
            )
        if self._check_timeout(start):
            return self._finalize_result(
                ReWOOResult(
                    answer="[timeout exceeded]",
                    steps=tuple(steps),
                    total_usage=total_usage,
                    stop_reason="timeout",
                ),
                result_type=ReWOOResult,
            )
        if budget.exhausted_reason() is not None:
            return self._finalize_result(
                ReWOOResult(
                    answer="[token budget exceeded]",
                    steps=tuple(steps),
                    total_usage=total_usage,
                    stop_reason="budget_exhausted",
                ),
                result_type=ReWOOResult,
            )
        _fire_event(self.config, "step_start", step_number=2)
        results_map: dict[str, str] = {}
        tool_results: list[ToolResult] = []
        for ps in plan_steps:
            if self._is_cancelled(cancellation_token):
                return self._finalize_result(
                    ReWOOResult(
                        answer="[cancelled]",
                        steps=tuple(steps),
                        total_usage=total_usage,
                        stop_reason="cancelled",
                    ),
                    result_type=ReWOOResult,
                )
            if self._check_timeout(start):
                return self._finalize_result(
                    ReWOOResult(
                        answer="[timeout exceeded]",
                        steps=tuple(steps),
                        total_usage=total_usage,
                        stop_reason="timeout",
                    ),
                    result_type=ReWOOResult,
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
        solve_cost = self._observe_response(budget, solve_resp, step_number=3)
        steps.append(
            AgentStep(
                step_number=3,
                response=solve_resp,
                usage=solve_resp.usage,
                cost_usd=solve_cost,
            )
        )
        _fire_event(self.config, "step_end", step_number=3)

        return self._finalize_result(
            ReWOOResult(
                answer=solve_resp.text,
                steps=tuple(steps),
                total_usage=total_usage,
                stop_reason="completed",
                metadata={"plan_size": len(plan_steps)},
            ),
            result_type=ReWOOResult,
        )

    async def async_run(self, task: str, **kwargs: Any) -> AgentResult:
        """Run the ReWOO plan-execute-solve loop asynchronously."""
        stream = kwargs.pop("stream", False)
        cancellation_token = self._resolve_cancellation_token(
            kwargs.pop("cancellation_token", None)
        )
        if stream:
            return self.async_run_stream(
                task,
                cancellation_token=cancellation_token,
                **kwargs,
            )
        system = self.config.system or None
        total_usage = Usage()
        budget = self._new_budget_manager()
        steps: list[AgentStep] = []
        start = time.monotonic()

        def _record_repair_response(step_number: int, response: Response) -> None:
            nonlocal total_usage
            total_usage = _accumulate_usage(total_usage, response.usage)
            repair_cost = self._observe_response(budget, response, step_number=step_number)
            steps.append(
                AgentStep(
                    step_number=step_number,
                    response=response,
                    usage=response.usage,
                    cost_usd=repair_cost,
                    metadata={"repair_attempt": True},
                )
            )

        # Phase 1: Plan
        _fire_event(self.config, "step_start", step_number=1)
        plan_resp = await self.client.chat(
            [Message(role="user", content=self._build_plan_prompt(task))],
            system=system,
            **kwargs,
        )
        total_usage = _accumulate_usage(total_usage, plan_resp.usage)
        plan_cost = self._observe_response(budget, plan_resp, step_number=1)
        steps.append(
            AgentStep(step_number=1, response=plan_resp, usage=plan_resp.usage, cost_usd=plan_cost)
        )
        plan_steps = await self._aparse_plan_with_fallback(
            plan_resp.text,
            system=system,
            step_number=1,
            on_repair_response=_record_repair_response,
            **kwargs,
        )
        _fire_event(
            self.config,
            "plan_created",
            step_number=1,
            result=plan_resp.text,
        )
        _fire_event(self.config, "step_end", step_number=1)

        # Phase 2: Execute
        if self._is_cancelled(cancellation_token):
            return self._finalize_result(
                ReWOOResult(
                    answer="[cancelled]",
                    steps=tuple(steps),
                    total_usage=total_usage,
                    stop_reason="cancelled",
                ),
                result_type=ReWOOResult,
            )
        if self._check_timeout(start):
            return self._finalize_result(
                ReWOOResult(
                    answer="[timeout exceeded]",
                    steps=tuple(steps),
                    total_usage=total_usage,
                    stop_reason="timeout",
                ),
                result_type=ReWOOResult,
            )
        if budget.exhausted_reason() is not None:
            return self._finalize_result(
                ReWOOResult(
                    answer="[token budget exceeded]",
                    steps=tuple(steps),
                    total_usage=total_usage,
                    stop_reason="budget_exhausted",
                ),
                result_type=ReWOOResult,
            )
        _fire_event(self.config, "step_start", step_number=2)
        results_map: dict[str, str] = {}
        tool_results: list[ToolResult] = []
        for ps in plan_steps:
            if self._is_cancelled(cancellation_token):
                return self._finalize_result(
                    ReWOOResult(
                        answer="[cancelled]",
                        steps=tuple(steps),
                        total_usage=total_usage,
                        stop_reason="cancelled",
                    ),
                    result_type=ReWOOResult,
                )
            if self._check_timeout(start):
                return self._finalize_result(
                    ReWOOResult(
                        answer="[timeout exceeded]",
                        steps=tuple(steps),
                        total_usage=total_usage,
                        stop_reason="timeout",
                    ),
                    result_type=ReWOOResult,
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
        solve_cost = self._observe_response(budget, solve_resp, step_number=3)
        steps.append(
            AgentStep(
                step_number=3,
                response=solve_resp,
                usage=solve_resp.usage,
                cost_usd=solve_cost,
            )
        )
        _fire_event(self.config, "step_end", step_number=3)

        return self._finalize_result(
            ReWOOResult(
                answer=solve_resp.text,
                steps=tuple(steps),
                total_usage=total_usage,
                stop_reason="completed",
                metadata={"plan_size": len(plan_steps)},
            ),
            result_type=ReWOOResult,
        )

    def _parse_plan_with_fallback(
        self,
        text: str,
        *,
        system: str | None,
        step_number: int,
        on_repair_response: Callable[[int, Response], None] | None = None,
        **kwargs: Any,
    ) -> list[_PlanStep]:
        plan = _parse_plan_json(text) or _parse_plan_regex(text)
        if plan:
            return plan
        repaired = text
        for _ in range(max(0, self.config.planner_repair_retries)):
            repair_resp = self.client.chat(
                [
                    Message(
                        role="user",
                        content=(
                            "Repair the following into a valid JSON array of "
                            "{\"id\":\"E1\",\"tool\":\"name\",\"input\":\"...\"} entries. "
                            "Return only JSON.\n\n"
                            f"{repaired}"
                        ),
                    )
                ],
                system=system,
                **kwargs,
            )
            if on_repair_response is not None:
                on_repair_response(step_number, repair_resp)
            repaired = repair_resp.text
            plan = _parse_plan_json(repaired) or _parse_plan_regex(repaired)
            if plan:
                _fire_event(
                    self.config,
                    "plan_created",
                    step_number=step_number,
                    result=repaired,
                    metadata={"repair_used": True},
                )
                return plan
        msg = "Unable to parse ReWOO planner output"
        _fire_event(
            self.config,
            "error",
            step_number=step_number,
            error=msg,
            metadata={"raw_plan": text},
        )
        raise ValueError(msg)

    async def _aparse_plan_with_fallback(
        self,
        text: str,
        *,
        system: str | None,
        step_number: int,
        on_repair_response: Callable[[int, Response], None] | None = None,
        **kwargs: Any,
    ) -> list[_PlanStep]:
        plan = _parse_plan_json(text) or _parse_plan_regex(text)
        if plan:
            return plan
        repaired = text
        for _ in range(max(0, self.config.planner_repair_retries)):
            repair_resp = await self.client.chat(
                [
                    Message(
                        role="user",
                        content=(
                            "Repair the following into a valid JSON array of "
                            "{\"id\":\"E1\",\"tool\":\"name\",\"input\":\"...\"} entries. "
                            "Return only JSON.\n\n"
                            f"{repaired}"
                        ),
                    )
                ],
                system=system,
                **kwargs,
            )
            if on_repair_response is not None:
                on_repair_response(step_number, repair_resp)
            repaired = repair_resp.text
            plan = _parse_plan_json(repaired) or _parse_plan_regex(repaired)
            if plan:
                _fire_event(
                    self.config,
                    "plan_created",
                    step_number=step_number,
                    result=repaired,
                    metadata={"repair_used": True},
                )
                return plan
        msg = "Unable to parse ReWOO planner output"
        _fire_event(
            self.config,
            "error",
            step_number=step_number,
            error=msg,
            metadata={"raw_plan": text},
        )
        raise ValueError(msg)


def _parse_plan_regex(text: str) -> list[_PlanStep]:
    """Compatibility parser: #E1 = tool[input]."""
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


def _parse_plan_json(text: str) -> list[_PlanStep]:
    """Primary parser: JSON array of objects with id/tool/input."""
    candidates = [text]
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        candidates.append(match.group())
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, list):
            continue
        parsed: list[_PlanStep] = []
        for idx, item in enumerate(value, start=1):
            if not isinstance(item, dict):
                continue
            step_id = str(item.get("id", f"E{idx}"))
            if not step_id.startswith("E"):
                step_id = f"E{step_id}"
            tool_name = str(item.get("tool", "")).strip()
            tool_input = str(item.get("input", "")).strip()
            if tool_name and tool_input:
                parsed.append(
                    _PlanStep(id=step_id, tool_name=tool_name, tool_input=tool_input)
                )
        if parsed:
            return parsed
    return []


def _substitute_refs(text: str, results: dict[str, str]) -> str:
    """Replace #E1, #E2 etc. with actual results."""
    for ref_id, result in results.items():
        text = text.replace(f"#{ref_id}", result)
    return text
