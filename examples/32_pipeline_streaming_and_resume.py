"""32 — Pipeline Streaming & Resume.

Use Pipeline.iter() for phase-by-phase streaming control and
Pipeline.run_from() to resume a pipeline from an intermediate phase.

Also demonstrates stop_on_failure, early break, and partial results.

No API keys needed.
"""

from __future__ import annotations

import asyncio

from ai_arch_toolkit.toolkit.pipeline import PhaseResult, Pipeline, PipelineContext


async def phase_fetch(ctx: PipelineContext) -> PhaseResult:
    return PhaseResult.ok(raw_data="fetched content from source")


async def phase_parse(ctx: PipelineContext) -> PhaseResult:
    return PhaseResult.ok(
        parsed=ctx["raw_data"].upper(),
        token_usage={"input": 50, "output": 30},
    )


async def phase_validate(ctx: PipelineContext) -> PhaseResult:
    if len(ctx["parsed"]) < 10:
        return PhaseResult.failed("parsed content too short")
    return PhaseResult.ok(validated=True)


async def phase_store(ctx: PipelineContext) -> PhaseResult:
    return PhaseResult.ok(stored=True, token_usage={"input": 10, "output": 5})


async def main():
    pipe = Pipeline(phase_fetch, phase_parse, phase_validate, phase_store)

    # --- 1. iter() for phase-by-phase streaming ---
    print("=== Streaming with iter() ===")
    ctx = PipelineContext()
    async for name, result in pipe.iter(ctx):
        print(f"  Completed: {name} [{result.status}]")
    print(f"  Token totals: {ctx.total_token_usage}")
    print()

    # --- 2. iter() with early break ---
    print("=== Early break after first phase ===")
    ctx2 = PipelineContext()
    async for name, _result in pipe.iter(ctx2):
        print(f"  Got: {name}")
        if name == "phase_fetch":
            print("  Breaking early — only needed the fetch.")
            break
    print(f"  Context has raw_data: {'raw_data' in ctx2}")
    print(f"  Context has parsed: {'parsed' in ctx2}")
    print()

    # --- 3. stop_on_failure skips remaining phases ---
    print("=== stop_on_failure with a failing phase ===")

    async def phase_boom(ctx: PipelineContext) -> PhaseResult:
        msg = "network error"
        raise ConnectionError(msg)

    fail_pipe = Pipeline(phase_fetch, phase_boom, phase_store)
    result = await fail_pipe.run(stop_on_failure=True)

    print(f"  Status: {result.status}")
    for p in result.phases:
        print(f"  [{p.status:>7}] {p.phase}")
    print(f"  Failed: {result.failed_phases}")
    print(f"  Skipped: {result.skipped_phases}")
    print()

    # --- 4. run_from() to resume from a checkpoint ---
    print("=== Resume with run_from() ===")

    # Simulate: phase_fetch and phase_parse already ran in a previous execution.
    # We have their artifacts saved and want to resume from phase_validate.
    saved_ctx = PipelineContext(
        {
            "raw_data": "fetched content from source",
            "parsed": "FETCHED CONTENT FROM SOURCE",
        }
    )

    result = await pipe.run_from("phase_validate", saved_ctx)
    print(f"  Status: {result.status}")
    for p in result.phases:
        print(f"  [{p.status:>7}] {p.phase}")
    print(f"  Stored: {result.context['stored']}")

    # --- 5. Partial results ---
    print()
    print("=== Partial results ===")

    async def phase_partial_work(ctx: PipelineContext) -> PhaseResult:
        return PhaseResult.partial(
            error="only got 3 of 5 items",
            items=["a", "b", "c"],
            warnings=["incomplete fetch"],
        )

    partial_pipe = Pipeline(phase_partial_work, phase_store)
    result = await partial_pipe.run(stop_on_partial=True)
    print(f"  Status: {result.status}")
    print(f"  Partial artifacts merged: {result.context.get('items')}")
    print(f"  Skipped: {result.skipped_phases}")


asyncio.run(main())
