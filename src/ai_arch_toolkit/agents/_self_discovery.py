"""Self-Discovery agent — four-phase reasoning without tools."""

from __future__ import annotations

import time
from typing import Any

from ai_arch_toolkit.agents._base import (
    AgentResult,
    AgentStep,
    BaseAgent,
    SelfDiscoveryResult,
    _accumulate_usage,
    _fire_event,
)
from ai_arch_toolkit.llm._types import Message, Response, Usage

DEFAULT_REASONING_MODULES: list[str] = [
    "Critical Thinking",
    "Creative Thinking",
    "Systems Thinking",
    "Analytical Reasoning",
    "Deductive Reasoning",
    "Inductive Reasoning",
    "Abductive Reasoning",
    "Analogical Reasoning",
    "Causal Reasoning",
    "Counterfactual Thinking",
]

_SELECT_PROMPT = (
    "Given the task below, select the most relevant reasoning modules "
    "from the following list that would help solve it. "
    "Return ONLY the names of the selected modules, one per line.\n\n"
    "Available modules:\n{modules}\n\nTask: {task}"
)

_ADAPT_PROMPT = (
    "Rephrase and adapt each selected reasoning module to be "
    "specific to the task at hand. Make each module actionable "
    "for this particular problem.\n\n"
    "Selected modules:\n{modules}\n\nTask: {task}"
)

_IMPLEMENT_PROMPT = (
    "Using the following adapted reasoning modules, create a structured "
    "step-by-step reasoning plan as a JSON object. Each key should be a "
    "step name and each value should describe what to do.\n\n"
    "Adapted modules:\n{modules}\n\nTask: {task}"
)

_SOLVE_PROMPT = (
    "Follow this reasoning plan step-by-step to solve the task.\n\nPlan:\n{plan}\n\nTask: {task}"
)


class SelfDiscoveryAgent(BaseAgent):
    """Self-Discovery agent that uses four-phase reasoning.

    Phase 1 (SELECT): Pick relevant reasoning modules.
    Phase 2 (ADAPT): Rephrase modules to be task-specific.
    Phase 3 (IMPLEMENT): Operationalize into a structured JSON plan.
    Phase 4 (SOLVE): Follow the plan to produce a final answer.
    """

    def _chat(self, prompt: str, system: str | None, **kwargs: Any) -> Response:
        messages = [Message(role="user", content=prompt)]
        return self.client.chat(messages, system=system, **kwargs)

    async def _async_chat(self, prompt: str, system: str | None, **kwargs: Any) -> Response:
        messages = [Message(role="user", content=prompt)]
        return await self.client.chat(messages, system=system, **kwargs)

    def run(self, task: str, **kwargs: Any) -> AgentResult:
        """Run the Self-Discovery four-phase process."""
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
        modules = kwargs.pop("reasoning_modules", DEFAULT_REASONING_MODULES)
        system = self.config.system or None
        total_usage = Usage()
        budget = self._new_budget_manager()
        steps: list[AgentStep] = []
        start = time.monotonic()

        # Phase 1: SELECT
        _fire_event(self.config, "step_start", step_number=1)
        modules_str = "\n".join(f"- {m}" for m in modules)
        select_prompt = _SELECT_PROMPT.format(modules=modules_str, task=task)
        resp1 = self._chat(select_prompt, system, **kwargs)
        total_usage = _accumulate_usage(total_usage, resp1.usage)
        cost1 = self._observe_response(budget, resp1, step_number=1)
        steps.append(AgentStep(step_number=1, response=resp1, usage=resp1.usage, cost_usd=cost1))
        _fire_event(self.config, "step_end", step_number=1)

        # Phase 2: ADAPT
        if self._is_cancelled(cancellation_token):
            return self._finalize_result(
                SelfDiscoveryResult(
                    answer="[cancelled]",
                    steps=tuple(steps),
                    total_usage=total_usage,
                    stop_reason="cancelled",
                ),
                result_type=SelfDiscoveryResult,
            )
        if self._check_timeout(start):
            return self._finalize_result(
                SelfDiscoveryResult(
                    answer="[timeout exceeded]",
                    steps=tuple(steps),
                    total_usage=total_usage,
                    stop_reason="timeout",
                ),
                result_type=SelfDiscoveryResult,
            )
        if budget.exhausted_reason() is not None:
            return self._finalize_result(
                SelfDiscoveryResult(
                    answer="[token budget exceeded]",
                    steps=tuple(steps),
                    total_usage=total_usage,
                    stop_reason="budget_exhausted",
                ),
                result_type=SelfDiscoveryResult,
            )
        _fire_event(self.config, "step_start", step_number=2)
        adapt_prompt = _ADAPT_PROMPT.format(modules=resp1.text, task=task)
        resp2 = self._chat(adapt_prompt, system, **kwargs)
        total_usage = _accumulate_usage(total_usage, resp2.usage)
        cost2 = self._observe_response(budget, resp2, step_number=2)
        steps.append(AgentStep(step_number=2, response=resp2, usage=resp2.usage, cost_usd=cost2))
        _fire_event(self.config, "step_end", step_number=2)

        # Phase 3: IMPLEMENT
        if self._is_cancelled(cancellation_token):
            return self._finalize_result(
                SelfDiscoveryResult(
                    answer="[cancelled]",
                    steps=tuple(steps),
                    total_usage=total_usage,
                    stop_reason="cancelled",
                ),
                result_type=SelfDiscoveryResult,
            )
        if self._check_timeout(start):
            return self._finalize_result(
                SelfDiscoveryResult(
                    answer="[timeout exceeded]",
                    steps=tuple(steps),
                    total_usage=total_usage,
                    stop_reason="timeout",
                ),
                result_type=SelfDiscoveryResult,
            )
        if budget.exhausted_reason() is not None:
            return self._finalize_result(
                SelfDiscoveryResult(
                    answer="[token budget exceeded]",
                    steps=tuple(steps),
                    total_usage=total_usage,
                    stop_reason="budget_exhausted",
                ),
                result_type=SelfDiscoveryResult,
            )
        _fire_event(self.config, "step_start", step_number=3)
        implement_prompt = _IMPLEMENT_PROMPT.format(modules=resp2.text, task=task)
        resp3 = self._chat(implement_prompt, system, **kwargs)
        total_usage = _accumulate_usage(total_usage, resp3.usage)
        cost3 = self._observe_response(budget, resp3, step_number=3)
        steps.append(AgentStep(step_number=3, response=resp3, usage=resp3.usage, cost_usd=cost3))
        _fire_event(self.config, "step_end", step_number=3)

        # Phase 4: SOLVE
        if self._is_cancelled(cancellation_token):
            return self._finalize_result(
                SelfDiscoveryResult(
                    answer="[cancelled]",
                    steps=tuple(steps),
                    total_usage=total_usage,
                    stop_reason="cancelled",
                ),
                result_type=SelfDiscoveryResult,
            )
        if self._check_timeout(start):
            return self._finalize_result(
                SelfDiscoveryResult(
                    answer="[timeout exceeded]",
                    steps=tuple(steps),
                    total_usage=total_usage,
                    stop_reason="timeout",
                ),
                result_type=SelfDiscoveryResult,
            )
        if budget.exhausted_reason() is not None:
            return self._finalize_result(
                SelfDiscoveryResult(
                    answer="[token budget exceeded]",
                    steps=tuple(steps),
                    total_usage=total_usage,
                    stop_reason="budget_exhausted",
                ),
                result_type=SelfDiscoveryResult,
            )
        _fire_event(self.config, "step_start", step_number=4)
        solve_prompt = _SOLVE_PROMPT.format(plan=resp3.text, task=task)
        resp4 = self._chat(solve_prompt, system, **kwargs)
        total_usage = _accumulate_usage(total_usage, resp4.usage)
        cost4 = self._observe_response(budget, resp4, step_number=4)
        steps.append(AgentStep(step_number=4, response=resp4, usage=resp4.usage, cost_usd=cost4))
        _fire_event(self.config, "step_end", step_number=4)

        return self._finalize_result(
            SelfDiscoveryResult(
                answer=resp4.text,
                steps=tuple(steps),
                total_usage=total_usage,
                stop_reason="completed",
            ),
            result_type=SelfDiscoveryResult,
        )

    async def async_run(self, task: str, **kwargs: Any) -> AgentResult:
        """Run the Self-Discovery four-phase process asynchronously."""
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
        modules = kwargs.pop("reasoning_modules", DEFAULT_REASONING_MODULES)
        system = self.config.system or None
        total_usage = Usage()
        budget = self._new_budget_manager()
        steps: list[AgentStep] = []
        start = time.monotonic()

        # Phase 1: SELECT
        _fire_event(self.config, "step_start", step_number=1)
        modules_str = "\n".join(f"- {m}" for m in modules)
        select_prompt = _SELECT_PROMPT.format(modules=modules_str, task=task)
        resp1 = await self._async_chat(select_prompt, system, **kwargs)
        total_usage = _accumulate_usage(total_usage, resp1.usage)
        cost1 = self._observe_response(budget, resp1, step_number=1)
        steps.append(AgentStep(step_number=1, response=resp1, usage=resp1.usage, cost_usd=cost1))
        _fire_event(self.config, "step_end", step_number=1)

        # Phase 2: ADAPT
        if self._is_cancelled(cancellation_token):
            return self._finalize_result(
                SelfDiscoveryResult(
                    answer="[cancelled]",
                    steps=tuple(steps),
                    total_usage=total_usage,
                    stop_reason="cancelled",
                ),
                result_type=SelfDiscoveryResult,
            )
        if self._check_timeout(start):
            return self._finalize_result(
                SelfDiscoveryResult(
                    answer="[timeout exceeded]",
                    steps=tuple(steps),
                    total_usage=total_usage,
                    stop_reason="timeout",
                ),
                result_type=SelfDiscoveryResult,
            )
        if budget.exhausted_reason() is not None:
            return self._finalize_result(
                SelfDiscoveryResult(
                    answer="[token budget exceeded]",
                    steps=tuple(steps),
                    total_usage=total_usage,
                    stop_reason="budget_exhausted",
                ),
                result_type=SelfDiscoveryResult,
            )
        _fire_event(self.config, "step_start", step_number=2)
        adapt_prompt = _ADAPT_PROMPT.format(modules=resp1.text, task=task)
        resp2 = await self._async_chat(adapt_prompt, system, **kwargs)
        total_usage = _accumulate_usage(total_usage, resp2.usage)
        cost2 = self._observe_response(budget, resp2, step_number=2)
        steps.append(AgentStep(step_number=2, response=resp2, usage=resp2.usage, cost_usd=cost2))
        _fire_event(self.config, "step_end", step_number=2)

        # Phase 3: IMPLEMENT
        if self._is_cancelled(cancellation_token):
            return self._finalize_result(
                SelfDiscoveryResult(
                    answer="[cancelled]",
                    steps=tuple(steps),
                    total_usage=total_usage,
                    stop_reason="cancelled",
                ),
                result_type=SelfDiscoveryResult,
            )
        if self._check_timeout(start):
            return self._finalize_result(
                SelfDiscoveryResult(
                    answer="[timeout exceeded]",
                    steps=tuple(steps),
                    total_usage=total_usage,
                    stop_reason="timeout",
                ),
                result_type=SelfDiscoveryResult,
            )
        if budget.exhausted_reason() is not None:
            return self._finalize_result(
                SelfDiscoveryResult(
                    answer="[token budget exceeded]",
                    steps=tuple(steps),
                    total_usage=total_usage,
                    stop_reason="budget_exhausted",
                ),
                result_type=SelfDiscoveryResult,
            )
        _fire_event(self.config, "step_start", step_number=3)
        implement_prompt = _IMPLEMENT_PROMPT.format(modules=resp2.text, task=task)
        resp3 = await self._async_chat(implement_prompt, system, **kwargs)
        total_usage = _accumulate_usage(total_usage, resp3.usage)
        cost3 = self._observe_response(budget, resp3, step_number=3)
        steps.append(AgentStep(step_number=3, response=resp3, usage=resp3.usage, cost_usd=cost3))
        _fire_event(self.config, "step_end", step_number=3)

        # Phase 4: SOLVE
        if self._is_cancelled(cancellation_token):
            return self._finalize_result(
                SelfDiscoveryResult(
                    answer="[cancelled]",
                    steps=tuple(steps),
                    total_usage=total_usage,
                    stop_reason="cancelled",
                ),
                result_type=SelfDiscoveryResult,
            )
        if self._check_timeout(start):
            return self._finalize_result(
                SelfDiscoveryResult(
                    answer="[timeout exceeded]",
                    steps=tuple(steps),
                    total_usage=total_usage,
                    stop_reason="timeout",
                ),
                result_type=SelfDiscoveryResult,
            )
        if budget.exhausted_reason() is not None:
            return self._finalize_result(
                SelfDiscoveryResult(
                    answer="[token budget exceeded]",
                    steps=tuple(steps),
                    total_usage=total_usage,
                    stop_reason="budget_exhausted",
                ),
                result_type=SelfDiscoveryResult,
            )
        _fire_event(self.config, "step_start", step_number=4)
        solve_prompt = _SOLVE_PROMPT.format(plan=resp3.text, task=task)
        resp4 = await self._async_chat(solve_prompt, system, **kwargs)
        total_usage = _accumulate_usage(total_usage, resp4.usage)
        cost4 = self._observe_response(budget, resp4, step_number=4)
        steps.append(AgentStep(step_number=4, response=resp4, usage=resp4.usage, cost_usd=cost4))
        _fire_event(self.config, "step_end", step_number=4)

        return self._finalize_result(
            SelfDiscoveryResult(
                answer=resp4.text,
                steps=tuple(steps),
                total_usage=total_usage,
                stop_reason="completed",
            ),
            result_type=SelfDiscoveryResult,
        )
