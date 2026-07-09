"""33 — Flow with LLM.

A realistic flow that calls an LLM in each step, accumulates
token usage, and demonstrates how Flow integrates with the
core LLM class.

Requires API key: set OPENAI_API_KEY in environment.
"""

from __future__ import annotations

import asyncio

from ai_arch_toolkit import LLM
from ai_arch_toolkit.core import Result, State, Step
from ai_arch_toolkit.toolkit.flow import Flow, FlowStep


async def research(snap):
    """Generate key points about the topic."""
    llm: LLM = snap["llm"]
    topic = snap["topic"]

    response = await llm.complete(
        f"List 3 key points about: {topic}. Be concise, one sentence each."
    )
    return Result(
        value=response.text,
        artifacts={"key_points": response.text},
    )  # no manual usage/cost — the run's meter captures it automatically


async def draft(snap):
    """Write a short paragraph using the key points."""
    llm: LLM = snap["llm"]
    key_points = snap["key_points"]

    response = await llm.complete(
        f"Write a concise paragraph incorporating these points:\n{key_points}"
    )
    return Result(
        value=response.text,
        artifacts={"draft_text": response.text},
    )  # no manual usage/cost — the run's meter captures it automatically


async def review(snap):
    """Review and score the draft."""
    llm: LLM = snap["llm"]
    draft_text = snap["draft_text"]

    response = await llm.complete(
        f"Rate this text 1-10 for clarity and suggest one improvement:\n\n{draft_text}"
    )
    return Result(
        value=response.text,
        artifacts={"review": response.text},
    )  # no manual usage/cost — the run's meter captures it automatically


async def main():
    llm = LLM("gpt-4.1-nano")
    flow = Flow(
        FlowStep(step=Step(name="research", fn=research)),
        FlowStep(step=Step(name="draft", fn=draft)),
        FlowStep(step=Step(name="review", fn=review)),
        name="content-flow",
    )

    state = State(
        operational={"topic": "why async programming matters", "llm": llm},
    )

    # Stream step-by-step for real-time progress
    print(f"Running flow: {flow.name}")
    print()

    async for event in flow.iter(state):
        if event.type == "step_end":
            print(f"  [{event.step_name}] done")
        elif event.type == "flow_end":
            # The meter is the single source of truth for spend — read it off the trace metadata.
            meter = event.trace.metadata["meter"]
            print(f"\n  Total cost: ${meter['cost']:.4f}  ({meter['total_tokens']} tokens)")
            print(f"  Total duration: {event.trace.total_duration:.2f}s")

    # Final summary
    print()
    print("--- Key Points ---")
    print(state["key_points"])
    print()
    print("--- Draft ---")
    print(state["draft_text"])
    print()
    print("--- Review ---")
    print(state["review"])


asyncio.run(main())
