"""Reflexion agent — retry loop with self-reflection."""

from __future__ import annotations

import time
from typing import Any

from ai_arch_toolkit._legacy.agents._base import (
    AgentConfig,
    AgentResult,
    AgentStep,
    BaseAgent,
    ReflexionResult,
    _accumulate_usage,
    _fire_event,
)
from ai_arch_toolkit._legacy.agents._react import ReActAgent
from ai_arch_toolkit._legacy.llm._types import Message, Usage

_REFLECT_PROMPT = (
    "The previous attempt did not meet the quality threshold. "
    "Analyze what went wrong and suggest improvements.\n\n"
    "Task: {task}\n\n"
    "Previous result: {result}\n\n"
    "Previous reflections:\n{reflections}"
)


class ReflexionAgent(BaseAgent):
    """Reflexion agent that wraps an inner ReActAgent with retry + reflection.

    1. Run inner ReActAgent on the task.
    2. Evaluate result with evaluator function.
    3. If score >= threshold, return result.
    4. Else: LLM generates reflection, append to context, retry.
    """

    def _build_inner_config(self, system: str | None, reflections: list[str]) -> AgentConfig:
        inner_system = system or ""
        if reflections:
            inner_system += "\n\nPrevious reflections:\n" + "\n".join(
                f"- {r}" for r in reflections
            )
        return AgentConfig(
            max_iterations=self.config.max_iterations,
            system=inner_system,
            max_tokens=self.config.max_tokens,
            planner_repair_retries=self.config.planner_repair_retries,
        )

    def _build_reflect_prompt(self, task: str, result: str, reflections: list[str]) -> str:
        reflections_text = "\n".join(reflections) if reflections else "(none)"
        return _REFLECT_PROMPT.format(
            task=task,
            result=result,
            reflections=reflections_text,
        )

    def run(self, task: str, **kwargs: Any) -> AgentResult:
        """Run the Reflexion loop."""
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
        evaluator = kwargs.pop("evaluator", None)
        if evaluator is None:
            msg = "ReflexionAgent requires an 'evaluator' kwarg"
            raise ValueError(msg)
        threshold = kwargs.pop("threshold", 0.8)
        system = self.config.system or None
        total_usage = Usage()
        budget = self._new_budget_manager()
        all_steps: list[AgentStep] = []
        reflections: list[str] = []
        last_inner_answer = ""
        start = time.monotonic()

        for attempt in range(1, self.config.max_iterations + 1):
            if self._is_cancelled(cancellation_token):
                return self._finalize_result(
                    ReflexionResult(
                        answer="[cancelled]",
                        steps=tuple(all_steps),
                        total_usage=total_usage,
                        stop_reason="cancelled",
                    ),
                    result_type=ReflexionResult,
                )
            if self._check_timeout(start):
                return self._finalize_result(
                    ReflexionResult(
                        answer="[timeout exceeded]",
                        steps=tuple(all_steps),
                        total_usage=total_usage,
                        stop_reason="timeout",
                    ),
                    result_type=ReflexionResult,
                )
            if budget.exhausted_reason() is not None:
                return self._finalize_result(
                    ReflexionResult(
                        answer="[token budget exceeded]",
                        steps=tuple(all_steps),
                        total_usage=total_usage,
                        stop_reason="budget_exhausted",
                    ),
                    result_type=ReflexionResult,
                )
            _fire_event(self.config, "step_start", step_number=attempt)

            inner_config = self._build_inner_config(system, reflections)
            inner = ReActAgent(
                self.client,
                self.tools,
                config=inner_config,
            )
            inner_result = inner.run(
                task,
                cancellation_token=cancellation_token,
                **kwargs,
            )
            last_inner_answer = inner_result.answer
            total_usage = _accumulate_usage(total_usage, inner_result.total_usage)
            budget.observe_usage(inner_result.total_usage)
            all_steps.extend(inner_result.steps)

            # Evaluate
            score = evaluator(inner_result.answer)
            _fire_event(self.config, "step_end", step_number=attempt)

            if score >= threshold:
                return self._finalize_result(
                    ReflexionResult(
                        answer=inner_result.answer,
                        steps=tuple(all_steps),
                        total_usage=total_usage,
                        stop_reason="completed",
                        metadata={"reflections": list(reflections)},
                    ),
                    result_type=ReflexionResult,
                )

            # Reflect
            reflect_prompt = self._build_reflect_prompt(task, inner_result.answer, reflections)
            reflect_resp = self.client.chat(
                [Message(role="user", content=reflect_prompt)],
                system=system,
                **kwargs,
            )
            total_usage = _accumulate_usage(total_usage, reflect_resp.usage)
            reflect_cost = self._observe_response(budget, reflect_resp, step_number=attempt)
            all_steps.append(
                AgentStep(
                    step_number=attempt,
                    response=reflect_resp,
                    usage=reflect_resp.usage,
                    cost_usd=reflect_cost,
                )
            )
            reflections.append(reflect_resp.text)
            _fire_event(
                self.config,
                "reflection",
                step_number=attempt,
                result=reflect_resp.text,
            )

        # Max iterations reached — return last result
        return self._finalize_result(
            ReflexionResult(
                answer=last_inner_answer or "[max iterations reached]",
                steps=tuple(all_steps),
                total_usage=total_usage,
                stop_reason="max_iterations",
                metadata={"reflections": list(reflections)},
            ),
            result_type=ReflexionResult,
        )

    async def async_run(self, task: str, **kwargs: Any) -> AgentResult:
        """Run the Reflexion loop asynchronously."""
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
        evaluator = kwargs.pop("evaluator", None)
        if evaluator is None:
            msg = "ReflexionAgent requires an 'evaluator' kwarg"
            raise ValueError(msg)
        threshold = kwargs.pop("threshold", 0.8)
        system = self.config.system or None
        total_usage = Usage()
        budget = self._new_budget_manager()
        all_steps: list[AgentStep] = []
        reflections: list[str] = []
        last_inner_answer = ""
        start = time.monotonic()

        for attempt in range(1, self.config.max_iterations + 1):
            if self._is_cancelled(cancellation_token):
                return self._finalize_result(
                    ReflexionResult(
                        answer="[cancelled]",
                        steps=tuple(all_steps),
                        total_usage=total_usage,
                        stop_reason="cancelled",
                    ),
                    result_type=ReflexionResult,
                )
            if self._check_timeout(start):
                return self._finalize_result(
                    ReflexionResult(
                        answer="[timeout exceeded]",
                        steps=tuple(all_steps),
                        total_usage=total_usage,
                        stop_reason="timeout",
                    ),
                    result_type=ReflexionResult,
                )
            if budget.exhausted_reason() is not None:
                return self._finalize_result(
                    ReflexionResult(
                        answer="[token budget exceeded]",
                        steps=tuple(all_steps),
                        total_usage=total_usage,
                        stop_reason="budget_exhausted",
                    ),
                    result_type=ReflexionResult,
                )
            _fire_event(self.config, "step_start", step_number=attempt)

            inner_system = system or ""
            if reflections:
                inner_system += "\n\nPrevious reflections:\n" + "\n".join(
                    f"- {r}" for r in reflections
                )

            inner_config = AgentConfig(
                max_iterations=self.config.max_iterations,
                system=inner_system,
                max_tokens=self.config.max_tokens,
                planner_repair_retries=self.config.planner_repair_retries,
            )
            inner = ReActAgent(
                self.client,
                self.tools,
                config=inner_config,
            )
            inner_result = await inner.async_run(
                task,
                cancellation_token=cancellation_token,
                **kwargs,
            )
            last_inner_answer = inner_result.answer
            total_usage = _accumulate_usage(total_usage, inner_result.total_usage)
            budget.observe_usage(inner_result.total_usage)
            all_steps.extend(inner_result.steps)

            score = evaluator(inner_result.answer)
            _fire_event(self.config, "step_end", step_number=attempt)

            if score >= threshold:
                return self._finalize_result(
                    ReflexionResult(
                        answer=inner_result.answer,
                        steps=tuple(all_steps),
                        total_usage=total_usage,
                        stop_reason="completed",
                        metadata={"reflections": list(reflections)},
                    ),
                    result_type=ReflexionResult,
                )

            reflect_prompt = self._build_reflect_prompt(task, inner_result.answer, reflections)
            reflect_resp = await self.client.chat(
                [Message(role="user", content=reflect_prompt)],
                system=system,
                **kwargs,
            )
            total_usage = _accumulate_usage(total_usage, reflect_resp.usage)
            reflect_cost = self._observe_response(budget, reflect_resp, step_number=attempt)
            all_steps.append(
                AgentStep(
                    step_number=attempt,
                    response=reflect_resp,
                    usage=reflect_resp.usage,
                    cost_usd=reflect_cost,
                )
            )
            reflections.append(reflect_resp.text)
            _fire_event(
                self.config,
                "reflection",
                step_number=attempt,
                result=reflect_resp.text,
            )

        return self._finalize_result(
            ReflexionResult(
                answer=last_inner_answer or "[max iterations reached]",
                steps=tuple(all_steps),
                total_usage=total_usage,
                stop_reason="max_iterations",
                metadata={"reflections": list(reflections)},
            ),
            result_type=ReflexionResult,
        )
