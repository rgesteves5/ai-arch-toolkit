"""Reflexion as a Flow — retry loop with self-critique."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ai_arch_toolkit.core._content import Content, user
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._policy import Policy
from ai_arch_toolkit.core._state import State, StateSnapshot
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents.flows._react import react_flow, react_initial_state
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
    timeout: float | None = None,
    policy: Policy | None = None,
    exec_llm: LLM | None = None,
    exec_tools: ToolGroup | None = None,
    reflect_llm: LLM | None = None,
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
        timeout: Overall timeout in seconds.
        policy: Optional execution policy.
        exec_llm: Override LLM for the executor (inner ReAct).
        exec_tools: Override tools for the executor.
        reflect_llm: Override LLM for the reflector.
    """
    inner_llm = exec_llm or llm
    inner_tools = exec_tools or tools
    reflector_llm = reflect_llm or llm

    async def attempt(snap: StateSnapshot) -> Result:
        """Run inner ReAct and return the answer."""
        task: str = snap.require("task")
        reflections: list[str] = snap.get("reflections", [])

        inner_system = system
        if reflections:
            inner_system += "\n\nPrevious reflections:\n" + "\n---\n".join(reflections)

        inner = react_flow(
            inner_llm,
            inner_tools,
            system=inner_system,
            max_iterations=max_iterations,
        )

        state = State(operational=react_initial_state(task))
        result = await inner.run(state)

        response = state.get("response")
        answer = response.text if response else ""

        return Result(
            value=answer,
            artifacts={"last_answer": answer, "last_response": response},
            usage=result.trace.total_usage,
            cost=result.total_cost,
        )

    async def evaluate(snap: StateSnapshot) -> Result:
        """Evaluate the answer against the task."""
        task: str = snap.require("task")
        answer: str = snap.get("last_answer", "")

        score = evaluator(task, answer)
        passed = score >= threshold

        artifacts: dict[str, Any] = {"score": score, "passed": passed}
        if passed:
            artifacts["answer"] = answer
            artifacts["response"] = snap.get("last_response")

        return Result(
            value=score,
            artifacts=artifacts,
            confidence=score,
        )

    async def reflect(snap: StateSnapshot) -> Result:
        """Generate reflection on low-scoring answer."""
        task: str = snap.require("task")
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
        )

        reflections.append(response.text)

        return Result(
            value=response.text,
            artifacts={"reflections": reflections},
            usage=response.usage,
            cost=response.cost or 0.0,
        )

    def not_passed(snap: StateSnapshot) -> bool:
        return not snap.get("passed", False)

    flow_policy = policy
    if timeout is not None and flow_policy is None:
        flow_policy = Policy(timeout=timeout)

    return Flow(
        FlowStep(step=Step(name="attempt", fn=attempt), when=not_passed),
        FlowStep(step=Step(name="evaluate", fn=evaluate), when=not_passed),
        FlowStep(step=Step(name="reflect", fn=reflect), when=not_passed),
        name="reflexion",
        policy=flow_policy,
        max_iterations=max_retries,
    )


def reflexion_initial_state(task: Content) -> dict[str, Any]:
    """Create the initial operational state for a reflexion_flow."""
    task_str = task if isinstance(task, str) else str(task)
    return {
        "task": task_str,
        "reflections": [],
        "passed": False,
    }
