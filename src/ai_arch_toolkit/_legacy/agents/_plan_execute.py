"""Plan-then-Execute agent implementation."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit._legacy.agents._base import (
    AgentConfig,
    AgentResult,
    AgentStep,
    BaseAgent,
    PlanExecuteResult,
    _accumulate_usage,
    _fire_event,
)
from ai_arch_toolkit._legacy.agents._react import ReActAgent
from ai_arch_toolkit._legacy.llm._types import Message, Response, Usage

_PLAN_PROMPT = (
    "Create a step-by-step plan to accomplish the following task. "
    "Return the plan as a JSON array of strings, where each string "
    "is a step description. Return ONLY the JSON array.\n\n"
    "Task: {task}"
)

_REPLAN_PROMPT = (
    "The original plan needs to be updated. "
    "Here are the steps completed so far:\n{completed}\n\n"
    "The following step failed: {failed_step}\n"
    "Error: {error}\n\n"
    "Create an updated plan (remaining steps only) as a JSON array "
    "of strings. Return ONLY the JSON array.\n\n"
    "Original task: {task}"
)

_SYNTHESIZE_PROMPT = (
    "Synthesize the results of all steps into a final answer.\n\n"
    "Task: {task}\n\n"
    "Step results:\n{results}"
)


@dataclass
class _PlanStep:
    description: str
    result: str = ""
    status: str = "pending"  # "pending" | "completed" | "failed"


class PlanExecuteAgent(BaseAgent):
    """Plan-then-Execute agent.

    1. Plan: LLM generates a list of step descriptions.
    2. Execute: Each step runs via a mini-ReAct agent.
    3. Re-plan: On failure, LLM generates updated plan.
    4. Synthesize: LLM combines all results into final answer.
    """

    def _build_inner_config(self, step_description: str) -> AgentConfig:
        return AgentConfig(
            max_iterations=3,
            system=f"Execute this step: {step_description}",
            max_tokens=self.config.max_tokens,
            planner_repair_retries=self.config.planner_repair_retries,
        )

    def _build_synthesize_prompt(self, task: str, finished_steps: list[_PlanStep]) -> str:
        results_text = "\n".join(
            f"Step {j + 1} ({c.description}): {c.result}" for j, c in enumerate(finished_steps)
        )
        return _SYNTHESIZE_PROMPT.format(task=task, results=results_text)

    def run(self, task: str, **kwargs: Any) -> AgentResult:
        """Run the plan-execute loop."""
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
        all_steps: list[AgentStep] = []
        replans = 0
        start = time.monotonic()

        def _record_repair_response(step_number: int, response: Response) -> None:
            nonlocal total_usage
            total_usage = _accumulate_usage(total_usage, response.usage)
            repair_cost = self._observe_response(budget, response, step_number=step_number)
            all_steps.append(
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
            [Message(role="user", content=_PLAN_PROMPT.format(task=task))],
            system=system,
            **kwargs,
        )
        total_usage = _accumulate_usage(total_usage, plan_resp.usage)
        plan_cost = self._observe_response(budget, plan_resp, step_number=1)
        all_steps.append(
            AgentStep(
                step_number=1,
                response=plan_resp,
                usage=plan_resp.usage,
                cost_usd=plan_cost,
            )
        )
        plan = self._parse_plan_with_fallback(
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
        finished_steps: list[_PlanStep] = []
        step_num = 2
        i = 0
        while i < len(plan):
            if self._is_cancelled(cancellation_token):
                return self._finalize_result(
                    PlanExecuteResult(
                        answer="[cancelled]",
                        steps=tuple(all_steps),
                        total_usage=total_usage,
                        stop_reason="cancelled",
                    ),
                    result_type=PlanExecuteResult,
                )
            if self._check_timeout(start):
                return self._finalize_result(
                    PlanExecuteResult(
                        answer="[timeout exceeded]",
                        steps=tuple(all_steps),
                        total_usage=total_usage,
                        stop_reason="timeout",
                    ),
                    result_type=PlanExecuteResult,
                )
            if budget.exhausted_reason() is not None:
                return self._finalize_result(
                    PlanExecuteResult(
                        answer="[token budget exceeded]",
                        steps=tuple(all_steps),
                        total_usage=total_usage,
                        stop_reason="budget_exhausted",
                    ),
                    result_type=PlanExecuteResult,
                )
            ps = plan[i]
            _fire_event(self.config, "step_start", step_number=step_num)

            inner = ReActAgent(
                self.client,
                self.tools,
                config=self._build_inner_config(ps.description),
            )
            try:
                inner_result = inner.run(
                    ps.description,
                    cancellation_token=cancellation_token,
                    **kwargs,
                )
                ps.result = inner_result.answer
                ps.status = "completed"
                total_usage = _accumulate_usage(total_usage, inner_result.total_usage)
                budget.observe_usage(inner_result.total_usage)
                all_steps.extend(inner_result.steps)
            except Exception as exc:
                ps.result = str(exc)
                ps.status = "failed"

                # Re-plan if allowed
                if replans < self.config.max_iterations:
                    replans += 1
                    completed_text = "\n".join(
                        f"- {c.description}: {c.result}" for c in finished_steps
                    )
                    replan_resp = self.client.chat(
                        [
                            Message(
                                role="user",
                                content=_REPLAN_PROMPT.format(
                                    completed=completed_text or "(none)",
                                    failed_step=ps.description,
                                    error=str(exc),
                                    task=task,
                                ),
                            )
                        ],
                        system=system,
                        **kwargs,
                    )
                    total_usage = _accumulate_usage(total_usage, replan_resp.usage)
                    replan_cost = self._observe_response(budget, replan_resp, step_number=step_num)
                    all_steps.append(
                        AgentStep(
                            step_number=step_num,
                            response=replan_resp,
                            usage=replan_resp.usage,
                            cost_usd=replan_cost,
                        )
                    )
                    new_plan = self._parse_plan_with_fallback(
                        replan_resp.text,
                        system=system,
                        step_number=step_num,
                        on_repair_response=_record_repair_response,
                        **kwargs,
                    )
                    _fire_event(
                        self.config,
                        "plan_created",
                        step_number=step_num,
                        result=replan_resp.text,
                    )
                    plan = [*finished_steps, *new_plan]
                    i = len(finished_steps)
                    _fire_event(
                        self.config,
                        "step_end",
                        step_number=step_num,
                    )
                    step_num += 1
                    continue

            finished_steps.append(ps)
            _fire_event(self.config, "step_end", step_number=step_num)
            step_num += 1
            i += 1

        # Phase 3: Synthesize
        _fire_event(self.config, "step_start", step_number=step_num)
        synth_prompt = self._build_synthesize_prompt(task, finished_steps)
        synth_resp = self.client.chat(
            [Message(role="user", content=synth_prompt)],
            system=system,
            **kwargs,
        )
        total_usage = _accumulate_usage(total_usage, synth_resp.usage)
        synth_cost = self._observe_response(budget, synth_resp, step_number=step_num)
        all_steps.append(
            AgentStep(
                step_number=step_num,
                response=synth_resp,
                usage=synth_resp.usage,
                cost_usd=synth_cost,
            )
        )
        _fire_event(self.config, "step_end", step_number=step_num)

        return self._finalize_result(
            PlanExecuteResult(
                answer=synth_resp.text,
                steps=tuple(all_steps),
                total_usage=total_usage,
                stop_reason="completed",
            ),
            result_type=PlanExecuteResult,
        )

    async def async_run(self, task: str, **kwargs: Any) -> AgentResult:
        """Run the plan-execute loop asynchronously."""
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
        all_steps: list[AgentStep] = []
        replans = 0
        start = time.monotonic()

        def _record_repair_response(step_number: int, response: Response) -> None:
            nonlocal total_usage
            total_usage = _accumulate_usage(total_usage, response.usage)
            repair_cost = self._observe_response(budget, response, step_number=step_number)
            all_steps.append(
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
            [Message(role="user", content=_PLAN_PROMPT.format(task=task))],
            system=system,
            **kwargs,
        )
        total_usage = _accumulate_usage(total_usage, plan_resp.usage)
        plan_cost = self._observe_response(budget, plan_resp, step_number=1)
        all_steps.append(
            AgentStep(
                step_number=1,
                response=plan_resp,
                usage=plan_resp.usage,
                cost_usd=plan_cost,
            )
        )
        plan = await self._aparse_plan_with_fallback(
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
        finished_steps: list[_PlanStep] = []
        step_num = 2
        i = 0
        while i < len(plan):
            if self._is_cancelled(cancellation_token):
                return self._finalize_result(
                    PlanExecuteResult(
                        answer="[cancelled]",
                        steps=tuple(all_steps),
                        total_usage=total_usage,
                        stop_reason="cancelled",
                    ),
                    result_type=PlanExecuteResult,
                )
            if self._check_timeout(start):
                return self._finalize_result(
                    PlanExecuteResult(
                        answer="[timeout exceeded]",
                        steps=tuple(all_steps),
                        total_usage=total_usage,
                        stop_reason="timeout",
                    ),
                    result_type=PlanExecuteResult,
                )
            if budget.exhausted_reason() is not None:
                return self._finalize_result(
                    PlanExecuteResult(
                        answer="[token budget exceeded]",
                        steps=tuple(all_steps),
                        total_usage=total_usage,
                        stop_reason="budget_exhausted",
                    ),
                    result_type=PlanExecuteResult,
                )
            ps = plan[i]
            _fire_event(self.config, "step_start", step_number=step_num)

            inner = ReActAgent(
                self.client,
                self.tools,
                config=self._build_inner_config(ps.description),
            )
            try:
                inner_result = await inner.async_run(
                    ps.description,
                    cancellation_token=cancellation_token,
                    **kwargs,
                )
                ps.result = inner_result.answer
                ps.status = "completed"
                total_usage = _accumulate_usage(total_usage, inner_result.total_usage)
                budget.observe_usage(inner_result.total_usage)
                all_steps.extend(inner_result.steps)
            except Exception as exc:
                ps.result = str(exc)
                ps.status = "failed"

                if replans < self.config.max_iterations:
                    replans += 1
                    completed_text = "\n".join(
                        f"- {c.description}: {c.result}" for c in finished_steps
                    )
                    replan_resp = await self.client.chat(
                        [
                            Message(
                                role="user",
                                content=_REPLAN_PROMPT.format(
                                    completed=completed_text or "(none)",
                                    failed_step=ps.description,
                                    error=str(exc),
                                    task=task,
                                ),
                            )
                        ],
                        system=system,
                        **kwargs,
                    )
                    total_usage = _accumulate_usage(total_usage, replan_resp.usage)
                    replan_cost = self._observe_response(budget, replan_resp, step_number=step_num)
                    all_steps.append(
                        AgentStep(
                            step_number=step_num,
                            response=replan_resp,
                            usage=replan_resp.usage,
                            cost_usd=replan_cost,
                        )
                    )
                    new_plan = await self._aparse_plan_with_fallback(
                        replan_resp.text,
                        system=system,
                        step_number=step_num,
                        on_repair_response=_record_repair_response,
                        **kwargs,
                    )
                    _fire_event(
                        self.config,
                        "plan_created",
                        step_number=step_num,
                        result=replan_resp.text,
                    )
                    plan = [*finished_steps, *new_plan]
                    i = len(finished_steps)
                    _fire_event(
                        self.config,
                        "step_end",
                        step_number=step_num,
                    )
                    step_num += 1
                    continue

            finished_steps.append(ps)
            _fire_event(self.config, "step_end", step_number=step_num)
            step_num += 1
            i += 1

        # Phase 3: Synthesize
        _fire_event(self.config, "step_start", step_number=step_num)
        synth_prompt = self._build_synthesize_prompt(task, finished_steps)
        synth_resp = await self.client.chat(
            [Message(role="user", content=synth_prompt)],
            system=system,
            **kwargs,
        )
        total_usage = _accumulate_usage(total_usage, synth_resp.usage)
        synth_cost = self._observe_response(budget, synth_resp, step_number=step_num)
        all_steps.append(
            AgentStep(
                step_number=step_num,
                response=synth_resp,
                usage=synth_resp.usage,
                cost_usd=synth_cost,
            )
        )
        _fire_event(self.config, "step_end", step_number=step_num)

        return self._finalize_result(
            PlanExecuteResult(
                answer=synth_resp.text,
                steps=tuple(all_steps),
                total_usage=total_usage,
                stop_reason="completed",
            ),
            result_type=PlanExecuteResult,
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
        plan = _parse_plan_structured(text)
        if plan:
            return plan
        repaired = text
        for _ in range(max(0, self.config.planner_repair_retries)):
            repair_resp = self.client.chat(
                [
                    Message(
                        role="user",
                        content=(
                            "Repair the following into a valid JSON array of step strings. "
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
            plan = _parse_plan_structured(repaired)
            if plan:
                _fire_event(
                    self.config,
                    "plan_created",
                    step_number=step_number,
                    result=repaired,
                    metadata={"repair_used": True},
                )
                return plan
        # Last resort backward-compatible parsing from plain line output.
        line_fallback = _parse_plan_lines(text)
        if line_fallback:
            return line_fallback
        msg = "Unable to parse planner output into a valid plan"
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
        plan = _parse_plan_structured(text)
        if plan:
            return plan
        repaired = text
        for _ in range(max(0, self.config.planner_repair_retries)):
            repair_resp = await self.client.chat(
                [
                    Message(
                        role="user",
                        content=(
                            "Repair the following into a valid JSON array of step strings. "
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
            plan = _parse_plan_structured(repaired)
            if plan:
                _fire_event(
                    self.config,
                    "plan_created",
                    step_number=step_number,
                    result=repaired,
                    metadata={"repair_used": True},
                )
                return plan
        line_fallback = _parse_plan_lines(text)
        if line_fallback:
            return line_fallback
        msg = "Unable to parse planner output into a valid plan"
        _fire_event(
            self.config,
            "error",
            step_number=step_number,
            error=msg,
            metadata={"raw_plan": text},
        )
        raise ValueError(msg)


def _parse_plan_structured(text: str) -> list[_PlanStep]:
    """Parse structured JSON plan output."""
    candidates: list[str] = [text]
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        candidates.append(match.group())
    for candidate in candidates:
        try:
            items = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(items, list):
            parsed: list[_PlanStep] = []
            for item in items:
                if isinstance(item, str):
                    parsed.append(_PlanStep(description=item))
                elif isinstance(item, dict) and "description" in item:
                    parsed.append(_PlanStep(description=str(item["description"])))
                else:
                    parsed.append(_PlanStep(description=str(item)))
            if parsed:
                return parsed
    return []


def _parse_plan_lines(text: str) -> list[_PlanStep]:
    """Fallback parser for plain line-based plan output."""
    return [_PlanStep(description=line.strip()) for line in text.splitlines() if line.strip()]
