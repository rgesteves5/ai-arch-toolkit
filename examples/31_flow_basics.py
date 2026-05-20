"""31 — Flow Basics.

Define Steps, compose them into a Flow, run it, and inspect the
FlowResult — trace, cost, duration, and state artifacts.

No API keys needed.
"""

from __future__ import annotations

import asyncio

from ai_arch_toolkit.core import Result, State, Step
from ai_arch_toolkit.toolkit.flow import Flow, FlowStep

# --- 1. Define step functions ---
# Each step receives a StateSnapshot and returns a Result.
# Use Result(artifacts=...) to pass data to downstream steps.


async def gather_requirements(snap):
    topic = snap["topic"]
    return Result(
        value=f"Requirements for {topic}",
        artifacts={
            "requirements": f"Write a short essay about {topic}",
            "audience": "general",
        },
    )


async def create_outline(snap):
    _ = snap["requirements"]  # ensure dependency exists
    return Result(
        value="Outline created",
        artifacts={
            "outline": ["Introduction", "Main argument", "Conclusion"],
        },
    )


async def draft_content(snap):
    outline = snap["outline"]
    audience = snap["audience"]
    draft = f"Draft for {audience}: " + " -> ".join(outline)
    return Result(value=draft, artifacts={"draft": draft})


# --- 2. Build and run the flow ---


async def main():
    flow = Flow(
        FlowStep(step=Step(name="gather_requirements", fn=gather_requirements)),
        FlowStep(step=Step(name="create_outline", fn=create_outline)),
        FlowStep(step=Step(name="draft_content", fn=draft_content)),
        name="essay",
    )

    # Pre-populate state with initial data
    state = State(operational={"topic": "async programming"})

    result = await flow.run(state)

    # --- 3. Inspect the result ---
    print(f"Flow: {flow.name}")
    print(f"Duration: {result.total_duration:.4f}s")
    print(f"Cost: ${result.total_cost:.6f}")
    print(f"Steps: {len(result.trace.steps)}")
    print()

    # Step-level details
    for step_trace in result.trace.steps:
        print(f"  [{step_trace.name}] duration={step_trace.duration:.4f}s")

    # State holds all accumulated artifacts
    print()
    print(f"Final draft: {state['draft']}")
    print()

    # Results keyed by step name
    for name, step_result in result.results.items():
        print(f"  '{name}' value: {step_result.value}")


asyncio.run(main())
