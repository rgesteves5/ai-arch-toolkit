"""Reflexion agent — retry loop with self-reflection."""

from __future__ import annotations

import time
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
        evaluator = kwargs.pop("evaluator", None)
        if evaluator is None:
            msg = "ReflexionAgent requires an 'evaluator' kwarg"
            raise ValueError(msg)
        threshold = kwargs.pop("threshold", 0.8)
        system = self.config.system or None
        total_usage = Usage()
        all_steps: list[AgentStep] = []
        reflections: list[str] = []
        start = time.monotonic()

        for attempt in range(1, self.config.max_iterations + 1):
            if self._check_timeout(start):
                return AgentResult(
                    answer="[timeout exceeded]",
                    steps=tuple(all_steps),
                    total_usage=total_usage,
                )
            _fire_event(self.config, "step_start", step_number=attempt)

            inner_config = self._build_inner_config(system, reflections)
            inner = ReActAgent(
                self.client,
                self.tools,
                config=inner_config,
            )
            inner_result = inner.run(task, **kwargs)
            total_usage = _accumulate_usage(total_usage, inner_result.total_usage)
            all_steps.extend(inner_result.steps)

            # Evaluate
            score = evaluator(inner_result.answer)
            _fire_event(self.config, "step_end", step_number=attempt)

            if score >= threshold:
                return AgentResult(
                    answer=inner_result.answer,
                    steps=tuple(all_steps),
                    total_usage=total_usage,
                )

            # Reflect
            reflect_prompt = self._build_reflect_prompt(task, inner_result.answer, reflections)
            reflect_resp = self.client.chat(
                [Message(role="user", content=reflect_prompt)],
                system=system,
                **kwargs,
            )
            total_usage = _accumulate_usage(total_usage, reflect_resp.usage)
            reflections.append(reflect_resp.text)
            _fire_event(
                self.config,
                "reflection",
                step_number=attempt,
                result=reflect_resp.text,
            )

        # Max iterations reached — return last result
        last_answer = all_steps[-1].response.text if all_steps else ""
        return AgentResult(
            answer=last_answer or "[max iterations reached]",
            steps=tuple(all_steps),
            total_usage=total_usage,
        )

    async def async_run(self, task: str, **kwargs: Any) -> AgentResult:
        """Run the Reflexion loop asynchronously."""
        evaluator = kwargs.pop("evaluator", None)
        if evaluator is None:
            msg = "ReflexionAgent requires an 'evaluator' kwarg"
            raise ValueError(msg)
        threshold = kwargs.pop("threshold", 0.8)
        system = self.config.system or None
        total_usage = Usage()
        all_steps: list[AgentStep] = []
        reflections: list[str] = []
        start = time.monotonic()

        for attempt in range(1, self.config.max_iterations + 1):
            if self._check_timeout(start):
                return AgentResult(
                    answer="[timeout exceeded]",
                    steps=tuple(all_steps),
                    total_usage=total_usage,
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
            )
            inner = ReActAgent(
                self.client,
                self.tools,
                config=inner_config,
            )
            inner_result = await inner.async_run(task, **kwargs)
            total_usage = _accumulate_usage(total_usage, inner_result.total_usage)
            all_steps.extend(inner_result.steps)

            score = evaluator(inner_result.answer)
            _fire_event(self.config, "step_end", step_number=attempt)

            if score >= threshold:
                return AgentResult(
                    answer=inner_result.answer,
                    steps=tuple(all_steps),
                    total_usage=total_usage,
                )

            reflect_prompt = self._build_reflect_prompt(task, inner_result.answer, reflections)
            reflect_resp = await self.client.chat(
                [Message(role="user", content=reflect_prompt)],
                system=system,
                **kwargs,
            )
            total_usage = _accumulate_usage(total_usage, reflect_resp.usage)
            reflections.append(reflect_resp.text)
            _fire_event(
                self.config,
                "reflection",
                step_number=attempt,
                result=reflect_resp.text,
            )

        last_answer = all_steps[-1].response.text if all_steps else ""
        return AgentResult(
            answer=last_answer or "[max iterations reached]",
            steps=tuple(all_steps),
            total_usage=total_usage,
        )
