"""Generate-Review as a Flow — configurable generate + review loop."""

from __future__ import annotations

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
    **options: Unpack[FlowOptions],
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
        **options: The options of the ``Flow`` it builds (``FlowOptions``).
    """
    gen_extra = gen_kwargs or {}
    review_extra = review_kwargs or {}

    async def generate(snap: StateSnapshot) -> Result:
        """Generate an answer, optionally using tools via inner ReAct."""
        task: str = snap.require(TASK)
        feedback: list[str] = snap.get("feedback", [])

        system = gen_system
        if feedback:
            system += "\n\nPrevious feedback:\n" + "\n---\n".join(feedback)

        if gen_tools is not None:
            run = await run_react(
                gen_llm,
                gen_tools,
                task,
                system=system,
                max_iterations=max_gen_iterations,
                llm_kwargs=gen_extra or None,
                **options,
            )
            return Result(
                value=run.answer,
                artifacts={"last_answer": run.answer, "last_response": run.response},
            )

        response = await gen_llm.complete([user(task)], system=system or None, **gen_extra)
        return Result(
            value=response.text,
            artifacts={"last_answer": response.text, "last_response": response},
        )

    async def review(snap: StateSnapshot) -> Result:
        """Review and fact-check the answer, optionally using tools."""
        task: str = snap.require(TASK)
        answer: str = snap.get("last_answer", "")
        feedback: list[str] = list(snap.get("feedback", []))

        review_prompt = f"Task: {task}\n\nProposed answer: {answer}\n\nReview this answer."

        if review_tools is not None:
            run = await run_react(
                review_llm,
                review_tools,
                review_prompt,
                system=review_system,
                max_iterations=max_review_iterations,
                llm_kwargs=review_extra or None,
                **options,
            )
            verdict_text = run.answer
        else:
            response = await review_llm.complete(
                [user(review_prompt)], system=review_system, **review_extra
            )
            verdict_text = response.text

        first_line = verdict_text.strip().split("\n")[0].lower()
        accepted = "accept" in first_line and "unacceptable" not in first_line

        # The draft just reviewed is the answer so far, accepted or not: out of cycles, it stands.
        artifacts: dict[str, Any] = {
            "accepted": accepted,
            ANSWER: answer,
            RESPONSE: snap.get("last_response"),
        }
        if not accepted:
            feedback.append(verdict_text)
            artifacts["feedback"] = feedback

        return Result(value=verdict_text, artifacts=artifacts)

    def not_accepted(snap: StateSnapshot) -> bool:
        return not snap.get("accepted", False)

    return Flow(
        FlowStep(step=Step(name="generate", fn=generate), when=not_accepted),
        FlowStep(step=Step(name="review", fn=review), when=not_accepted),
        name="generate_review",
        max_iterations=max_cycles,
        **options,
    )


def generate_review_initial_state(task: Content) -> dict[str, Any]:
    """Create the initial operational state for a generate_review_flow."""
    task_str = task if isinstance(task, str) else str(task)
    return {TASK: task_str, "feedback": [], "accepted": False}
