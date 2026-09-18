"""Reflexion as a Flow — retry loop with self-critique."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Unpack

from ai_arch_toolkit.core._content import Content, user
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._state import StateSnapshot
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents.flows._common import FlowOptions
from ai_arch_toolkit.toolkit.agents.flows._keys import ANSWER, RESPONSE, TASK
from ai_arch_toolkit.toolkit.agents.flows._react import run_react
from ai_arch_toolkit.toolkit.flow._flow import Flow, FlowStep


def reflexion_flow(
    llm: LLM,
    tools: ToolGroup,
    *,
    evaluator: Callable[[str, str], float],
    threshold: float = 0.7,
    max_retries: int = 3,
    system: str = "",
    max_iterations: int = 10,
    reflect_system: str = (
        "You are a reflection assistant. Analyze the previous attempt "
        "and provide specific, actionable feedback for improvement."
    ),
    llm_kwargs: dict[str, Any] | None = None,
    exec_llm: LLM | None = None,
    exec_tools: ToolGroup | None = None,
    reflect_llm: LLM | None = None,
    **options: Unpack[FlowOptions],
) -> Flow:
    """Create a Reflexion Flow — inner ReAct with evaluate + reflect retry loop.

    Args:
        llm: Default language model.
        tools: Tool group for the inner ReAct agent.
        evaluator: Callable(task_str, answer) → score in [0, 1].
        threshold: Minimum score to accept an answer.
        max_retries: Maximum retry attempts.
        system: Base system prompt for the inner ReAct.
        max_iterations: Max iterations for the inner ReAct per attempt.
        reflect_system: System prompt for the reflector LLM.
        llm_kwargs: Additional kwargs passed to every phase's LLM call.
        exec_llm: Override LLM for the executor (inner ReAct).
        exec_tools: Override tools for the executor.
        reflect_llm: Override LLM for the reflector.
        **options: The options of the ``Flow`` it builds (``FlowOptions``).
    """
    inner_llm = exec_llm or llm
    inner_tools = exec_tools if exec_tools is not None else tools
    reflector_llm = reflect_llm or llm
    extra = llm_kwargs or {}

    async def attempt(snap: StateSnapshot) -> Result:
        """Run inner ReAct and return the answer."""
        task: str = snap.require(TASK)
        reflections: list[str] = snap.get("reflections", [])

        inner_system = system
        if reflections:
            inner_system += "\n\nPrevious reflections:\n" + "\n---\n".join(reflections)

        run = await run_react(
            inner_llm,
            inner_tools,
            task,
            system=inner_system,
            max_iterations=max_iterations,
            llm_kwargs=llm_kwargs,
            **options,
        )
        return Result(
            value=run.answer,
            artifacts={"last_answer": run.answer, "last_response": run.response},
        )

    async def evaluate(snap: StateSnapshot) -> Result:
        """Evaluate the answer against the task."""
        task: str = snap.require(TASK)
        answer: str = snap.get("last_answer", "")

        score = evaluator(task, answer)
        passed = score >= threshold

        # The attempt just scored is the answer so far, passed or not: out of retries, it stands.
        artifacts = {
            "score": score,
            "passed": passed,
            ANSWER: answer,
            RESPONSE: snap.get("last_response"),
        }

        return Result(
            value=score,
            artifacts=artifacts,
            confidence=score,
        )

    async def reflect(snap: StateSnapshot) -> Result:
        """Generate reflection on low-scoring answer."""
        task: str = snap.require(TASK)
        answer: str = snap.get("last_answer", "")
        score: float = snap.get("score", 0.0)
        reflections: list[str] = list(snap.get("reflections", []))

        response = await reflector_llm.complete(
            [
                user(
                    f"Task: {task}\n\nAttempt answer: {answer}\n\n"
                    f"Score: {score:.2f} (threshold: {threshold})\n\n"
                    "Provide specific feedback for improvement."
                )
            ],
            system=reflect_system,
            **extra,
        )

        reflections.append(response.text)

        return Result(
            value=response.text,
            artifacts={"reflections": reflections},
        )

    def not_passed(snap: StateSnapshot) -> bool:
        return not snap.get("passed", False)

    return Flow(
        FlowStep(step=Step(name="attempt", fn=attempt), when=not_passed),
        FlowStep(step=Step(name="evaluate", fn=evaluate), when=not_passed),
        FlowStep(step=Step(name="reflect", fn=reflect), when=not_passed),
        name="reflexion",
        max_iterations=max_retries,
        **options,
    )


def reflexion_initial_state(task: Content) -> dict[str, Any]:
    """Create the initial operational state for a reflexion_flow."""
    task_str = task if isinstance(task, str) else str(task)
    return {
        TASK: task_str,
        "reflections": [],
        "passed": False,
    }
