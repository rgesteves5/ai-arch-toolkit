"""Generate-Review as a Flow — configurable generate + review loop."""

from __future__ import annotations

from typing import Any

from ai_arch_toolkit.core._content import Content, user
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._policy import Policy
from ai_arch_toolkit.core._state import State, StateSnapshot
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents.flows._react import react_flow, react_initial_state
from ai_arch_toolkit.toolkit.flow._flow import Flow, FlowStep


def generate_review_flow(
    gen_llm: LLM,
    review_llm: LLM,
    *,
    gen_tools: ToolGroup | None = None,
    review_tools: ToolGroup | None = None,
    gen_system: str = "",
    review_system: str = (
        "You are a review assistant. Evaluate the answer for correctness, "
        "completeness, and quality.\n\n"
        "Respond with ACCEPT if the answer is satisfactory.\n"
        "Respond with RETRY followed by detailed feedback for improvement."
    ),
    gen_kwargs: dict[str, Any] | None = None,
    review_kwargs: dict[str, Any] | None = None,
    max_cycles: int = 3,
    max_gen_iterations: int = 5,
    max_review_iterations: int = 5,
    timeout: float | None = None,
    policy: Policy | None = None,
) -> Flow:
    """Create a Generate-Review Flow — configurable generate + review loop.

    Both phases can independently use tools and have their own LLM, system
    prompt, and kwargs (temperature, top_p, etc.).

    Args:
        gen_llm: Language model for generation.
        review_llm: Language model for review.
        gen_tools: Optional tools for the generator (triggers inner ReAct).
        review_tools: Optional tools for the reviewer (triggers inner ReAct).
        gen_system: System prompt for the generator.
        review_system: System prompt for the reviewer.
        gen_kwargs: Additional kwargs passed to gen_llm.complete() (e.g. temperature).
        review_kwargs: Additional kwargs passed to review_llm.complete() (e.g. temperature).
        max_cycles: Maximum generate-review cycles.
        max_gen_iterations: Max iterations for inner ReAct during generation.
        max_review_iterations: Max iterations for inner ReAct during review.
        timeout: Overall timeout in seconds.
        policy: Optional execution policy.
    """
    gen_extra = gen_kwargs or {}
    review_extra = review_kwargs or {}

    async def generate(snap: StateSnapshot) -> Result:
        """Generate an answer, optionally using tools via inner ReAct."""
        task: str = snap.require("task")
        feedback: list[str] = snap.get("feedback", [])

        system = gen_system
        if feedback:
            system += "\n\nPrevious feedback:\n" + "\n---\n".join(feedback)

        if gen_tools is not None:
            inner = react_flow(
                gen_llm,
                gen_tools,
                system=system,
                max_iterations=max_gen_iterations,
                llm_kwargs=gen_extra or None,
            )
            state = State(operational=react_initial_state(task))
            result = await inner.run(state)
            response = state.get("response")
            answer = response.text if response else ""
            return Result(
                value=answer,
                artifacts={"last_answer": answer, "last_response": response},
                cost=result.total_cost,
            )

        response = await gen_llm.complete([user(task)], system=system or None, **gen_extra)
        return Result(
            value=response.text,
            artifacts={"last_answer": response.text, "last_response": response},
            usage=response.usage,
            cost=response.cost or 0.0,
        )

    async def review(snap: StateSnapshot) -> Result:
        """Review and fact-check the answer, optionally using tools."""
        task: str = snap.require("task")
        answer: str = snap.get("last_answer", "")
        feedback: list[str] = list(snap.get("feedback", []))

        review_prompt = f"Task: {task}\n\nProposed answer: {answer}\n\nReview this answer."

        if review_tools is not None:
            inner = react_flow(
                review_llm,
                review_tools,
                system=review_system,
                max_iterations=max_review_iterations,
                llm_kwargs=review_extra or None,
            )
            state = State(operational=react_initial_state(review_prompt))
            result = await inner.run(state)
            response = state.get("response")
            verdict_text = response.text if response else ""
            cost = result.total_cost
        else:
            response = await review_llm.complete(
                [user(review_prompt)], system=review_system, **review_extra
            )
            verdict_text = response.text
            cost = response.cost or 0.0

        first_line = verdict_text.strip().split("\n")[0].lower()
        accepted = "accept" in first_line and "unacceptable" not in first_line

        artifacts: dict[str, Any] = {"accepted": accepted}
        if accepted:
            artifacts["answer"] = answer
            artifacts["response"] = snap.get("last_response")
        else:
            feedback.append(verdict_text)
            artifacts["feedback"] = feedback
            # Keep last_answer accessible as fallback if max_cycles exhausted
            artifacts["answer"] = answer

        return Result(value=verdict_text, artifacts=artifacts, cost=cost)

    def not_accepted(snap: StateSnapshot) -> bool:
        return not snap.get("accepted", False)

    flow_policy = policy
    if timeout is not None and flow_policy is None:
        flow_policy = Policy(timeout=timeout)

    return Flow(
        FlowStep(step=Step(name="generate", fn=generate), when=not_accepted),
        FlowStep(step=Step(name="review", fn=review), when=not_accepted),
        name="generate_review",
        policy=flow_policy,
        max_iterations=max_cycles,
    )


def generate_review_initial_state(task: Content) -> dict[str, Any]:
    """Create the initial operational state for a generate_review_flow."""
    task_str = task if isinstance(task, str) else str(task)
    return {"task": task_str, "feedback": [], "accepted": False}
