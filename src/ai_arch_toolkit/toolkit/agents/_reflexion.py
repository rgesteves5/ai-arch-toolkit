"""ReflexionAgent — ReAct with self-critique retry loop."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
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

_DEFAULT_REFLECT_SYSTEM = (
    "You are a reflection assistant. Analyse the previous attempt at answering "
    "a task. Identify what went wrong and suggest concrete improvements for the "
    "next attempt. Be concise."
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ReflexionConfig:
    """Configuration specific to the Reflexion retry loop."""

    max_retries: int = 3
    threshold: float = 0.7
    evaluator: Callable[[str, str], float]
    reflect_system: str = _DEFAULT_REFLECT_SYSTEM
    executor: PhaseConfig | None = None
    reflector: PhaseConfig | None = None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ReflexionAgent(BaseAgent):
    """Reflexion: ReAct + self-critique retry loop.

    Each attempt runs a full ReAct loop.  After each attempt the evaluator
    scores the answer.  If below threshold, the agent reflects on the failure
    and prepends reflections to the system prompt for the next attempt.
    """

    __slots__ = ("reflexion",)

    def __init__(
        self,
        llm: LLM,
        tools: ToolGroup,
        *,
        config: AgentConfig | None = None,
        reflexion: ReflexionConfig | None = None,
    ) -> None:
        super().__init__(llm, tools, config=config)
        if reflexion is None:
            raise TypeError("ReflexionAgent requires a ReflexionConfig (reflexion=...)")
        self.reflexion = reflexion

    async def _run_loop(self, task: Content, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        step_offset = 0
        reflections: list[str] = []
        # NOTE: multimodal Content (list of parts) is converted via str(),
        # which produces a lossy repr.  Reflexion evaluation and reflection
        # prompts are text-based, so this agent works best with str tasks.
        task_str = task if isinstance(task, str) else str(task)
        exec_llm = _resolve_llm(self.reflexion.executor, self.llm)
        exec_tools = _resolve_tools(self.reflexion.executor, self.tools)
        reflect_llm = _resolve_llm(self.reflexion.reflector, self.llm)

        for _attempt in range(self.reflexion.max_retries):
            # Build inner system prompt with reflections
            system_parts: list[str] = []
            if self.config.system:
                system_parts.append(self.config.system)
            if reflections:
                system_parts.append("Previous reflections:\n" + "\n---\n".join(reflections))
            inner_system = "\n\n".join(system_parts) if system_parts else ""

            # on_event is intentionally omitted — the outer _consume() fires
            # callbacks for the re-numbered events we yield.
            inner_config = AgentConfig(
                max_iterations=self.config.max_iterations,
                system=inner_system,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
                tool_choice=self.config.tool_choice,
                parallel_tool_calls=self.config.parallel_tool_calls,
                llm_kwargs=self.config.llm_kwargs,
                output_schema=self.config.output_schema,
            )
            inner = ReActAgent(exec_llm, exec_tools, config=inner_config)

            # Stream inner events, re-numbering steps
            last_response = None
            inner_step_count = 0
            last_answer = ""
            inner_stop: str | None = None

            async for event in inner._run_loop(task):
                if event.type == "step_end":
                    inner_step_count = event.step
                    if event.stop_reason is not None:
                        inner_stop = event.stop_reason
                    if event.response is not None:
                        last_response = event.response
                        last_answer = event.response.text

                # Inner's terminal step_end: propagate hard stops, skip soft
                # stops (completed/max_iterations) — the outer loop emits its
                # own terminal step_end after evaluation.
                if event.type == "step_end" and event.stop_reason is not None:
                    if event.stop_reason in ("timeout", "error", "budget_exhausted"):
                        yield AgentEvent(
                            type=event.type,
                            step=event.step + step_offset,
                            response=event.response,
                            stop_reason=event.stop_reason,
                            error=event.error,
                        )
                    continue

                yield AgentEvent(
                    type=event.type,
                    step=event.step + step_offset,
                    tool_name=event.tool_name,
                    tool_call_id=event.tool_call_id,
                    tool_args=event.tool_args,
                    result=event.result,
                    error=event.error,
                    response=event.response,
                )

            # If inner agent hit a hard stop, propagate it
            if inner_stop in ("timeout", "error", "budget_exhausted"):
                return

            step_offset += inner_step_count

            # Evaluate
            score = self.reflexion.evaluator(task_str, last_answer)
            if score >= self.reflexion.threshold:
                yield AgentEvent(
                    type="step_end",
                    step=step_offset,
                    response=last_response,
                    stop_reason="completed",
                )
                return

            # Reflect (one LLM call)
            step_offset += 1
            yield AgentEvent(type="step_start", step=step_offset)

            reflect_prompt = (
                f"Task: {task_str}\n\n"
                f"Answer: {last_answer}\n\n"
                f"Score: {score:.2f} (threshold: {self.reflexion.threshold:.2f})\n\n"
                "What went wrong and how should the next attempt improve?"
            )
            response = await reflect_llm.complete(
                [user(reflect_prompt)],
                system=self.reflexion.reflect_system,
            )
            reflections.append(response.text)

            yield AgentEvent(
                type="step_end",
                step=step_offset,
                response=response,
            )

        # All retries exhausted
        yield AgentEvent(
            type="step_end",
            step=step_offset,
            stop_reason="max_iterations",
        )
