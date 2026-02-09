"""Plan-then-Execute agent implementation."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.agents._base import (
    AgentConfig,
    AgentResult,
    AgentStep,
    BaseAgent,
    _accumulate_usage,
    _fire_event,
)
from ai_arch_toolkit.agents._react import ReActAgent
from ai_arch_toolkit.llm._types import Message, Usage

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
        )

    def _build_synthesize_prompt(self, task: str, finished_steps: list[_PlanStep]) -> str:
        results_text = "\n".join(
            f"Step {j + 1} ({c.description}): {c.result}" for j, c in enumerate(finished_steps)
        )
        return _SYNTHESIZE_PROMPT.format(task=task, results=results_text)

    def run(self, task: str, **kwargs: Any) -> AgentResult:
        """Run the plan-execute loop."""
        system = self.config.system or None
        total_usage = Usage()
        all_steps: list[AgentStep] = []
        replans = 0
        start = time.monotonic()

        # Phase 1: Plan
        _fire_event(self.config, "step_start", step_number=1)
        plan_resp = self.client.chat(
            [Message(role="user", content=_PLAN_PROMPT.format(task=task))],
            system=system,
            **kwargs,
        )
        total_usage = _accumulate_usage(total_usage, plan_resp.usage)
        all_steps.append(AgentStep(step_number=1, response=plan_resp))
        plan = _parse_plan_json(plan_resp.text)
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
            if self._check_timeout(start):
                return AgentResult(
                    answer="[timeout exceeded]",
                    steps=tuple(all_steps),
                    total_usage=total_usage,
                )
            ps = plan[i]
            _fire_event(self.config, "step_start", step_number=step_num)

            inner = ReActAgent(
                self.client,
                self.tools,
                config=self._build_inner_config(ps.description),
            )
            try:
                inner_result = inner.run(ps.description, **kwargs)
                ps.result = inner_result.answer
                ps.status = "completed"
                total_usage = _accumulate_usage(total_usage, inner_result.total_usage)
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
                    new_plan = _parse_plan_json(replan_resp.text)
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
        all_steps.append(AgentStep(step_number=step_num, response=synth_resp))
        _fire_event(self.config, "step_end", step_number=step_num)

        return AgentResult(
            answer=synth_resp.text,
            steps=tuple(all_steps),
            total_usage=total_usage,
        )

    async def async_run(self, task: str, **kwargs: Any) -> AgentResult:
        """Run the plan-execute loop asynchronously."""
        system = self.config.system or None
        total_usage = Usage()
        all_steps: list[AgentStep] = []
        replans = 0
        start = time.monotonic()

        # Phase 1: Plan
        _fire_event(self.config, "step_start", step_number=1)
        plan_resp = await self.client.chat(
            [Message(role="user", content=_PLAN_PROMPT.format(task=task))],
            system=system,
            **kwargs,
        )
        total_usage = _accumulate_usage(total_usage, plan_resp.usage)
        all_steps.append(AgentStep(step_number=1, response=plan_resp))
        plan = _parse_plan_json(plan_resp.text)
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
            if self._check_timeout(start):
                return AgentResult(
                    answer="[timeout exceeded]",
                    steps=tuple(all_steps),
                    total_usage=total_usage,
                )
            ps = plan[i]
            _fire_event(self.config, "step_start", step_number=step_num)

            inner = ReActAgent(
                self.client,
                self.tools,
                config=self._build_inner_config(ps.description),
            )
            try:
                inner_result = await inner.async_run(ps.description, **kwargs)
                ps.result = inner_result.answer
                ps.status = "completed"
                total_usage = _accumulate_usage(total_usage, inner_result.total_usage)
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
                    new_plan = _parse_plan_json(replan_resp.text)
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
        all_steps.append(AgentStep(step_number=step_num, response=synth_resp))
        _fire_event(self.config, "step_end", step_number=step_num)

        return AgentResult(
            answer=synth_resp.text,
            steps=tuple(all_steps),
            total_usage=total_usage,
        )


def _parse_plan_json(text: str) -> list[_PlanStep]:
    """Parse a JSON array of step descriptions from LLM output."""
    # Try to extract JSON array from the text
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            items = json.loads(match.group())
            if isinstance(items, list):
                return [_PlanStep(description=str(item)) for item in items]
        except json.JSONDecodeError:
            pass
    # Fallback: treat each non-empty line as a step
    return [
        _PlanStep(description=line.strip()) for line in text.strip().split("\n") if line.strip()
    ]
