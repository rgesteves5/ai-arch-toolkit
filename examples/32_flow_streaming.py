"""32 — Flow Streaming.

Use Flow.iter() / iter_sync() for step-by-step streaming control.
Also demonstrates conditional steps with ``when`` and early observation
of flow events.

No API keys needed.
"""

from __future__ import annotations

import asyncio

from ai_arch_toolkit.core import Result, State, Step
from ai_arch_toolkit.toolkit.flow import Flow, FlowStep


async def phase_fetch(snap):
    return Result(
        value="fetched",
        artifacts={"raw_data": "fetched content from source"},
    )


async def phase_parse(snap):
    return Result(
        value="parsed",
        artifacts={"parsed": snap["raw_data"].upper()},
    )


async def phase_validate(snap):
    if len(snap["parsed"]) < 10:
        return Result(error="parsed content too short")
    return Result(value="validated", artifacts={"validated": True})


async def phase_store(snap):
    return Result(value="stored", artifacts={"stored": True})


async def main():
    flow = Flow(
        FlowStep(step=Step(name="fetch", fn=phase_fetch)),
        FlowStep(step=Step(name="parse", fn=phase_parse)),
        FlowStep(step=Step(name="validate", fn=phase_validate)),
        FlowStep(step=Step(name="store", fn=phase_store)),
        name="etl",
    )

    # --- 1. iter() for step-by-step streaming ---
    print("=== Streaming with iter() ===")
    state = State(operational={})
    async for event in flow.iter(state):
        if event.type == "step_end":
            status = "OK" if event.result and event.result.is_ok else "ERROR"
            print(f"  Completed: {event.step_name} [{status}]")
        elif event.type == "flow_end":
            print(f"  Flow done — {len(event.trace.steps)} steps")
    print()

    # --- 2. Sync streaming ---
    print("=== Sync streaming with iter_sync() ===")
    state2 = State(operational={})
    for event in flow.iter_sync(state2):
        if event.type == "step_end":
            print(f"  Completed: {event.step_name}")
        elif event.type == "flow_end":
            print(f"  Total duration: {event.trace.total_duration:.4f}s")
    print()

    # --- 3. Non-streaming run ---
    print("=== Non-streaming run() ===")
    state3 = State(operational={})
    result = await flow.run(state3)
    print(f"  Status: {'ok' if result.final_result and result.final_result.is_ok else 'error'}")
    print(f"  Steps completed: {len(result.trace.steps)}")
    print(f"  stored={state3.get('stored')}")


asyncio.run(main())
