"""31 — Pipeline Basics.

Define async phase functions, run them as a Pipeline, and inspect the
PipelineResult — status, artifacts, provenance, duration, and warnings.

No API keys needed.
"""

from __future__ import annotations

import asyncio

from ai_arch_toolkit.toolkit.pipeline import PhaseResult, Pipeline, PipelineContext

# --- 1. Define phase functions ---
# Each phase receives a PipelineContext and returns a PhaseResult.
# Use PhaseResult.ok(**artifacts) to pass data to downstream phases.


async def gather_requirements(ctx: PipelineContext) -> PhaseResult:
    topic = ctx.require("topic")
    return PhaseResult.ok(
        requirements=f"Write a short essay about {topic}",
        audience="general",
    )


async def create_outline(ctx: PipelineContext) -> PhaseResult:
    _ = ctx["requirements"]  # ensure dependency exists
    return PhaseResult.ok(
        outline=["Introduction", "Main argument", "Conclusion"],
        warnings=["outline is simplified"],
    )


async def draft_content(ctx: PipelineContext) -> PhaseResult:
    outline = ctx["outline"]
    audience = ctx["audience"]
    draft = f"Draft for {audience}: " + " -> ".join(outline)
    return PhaseResult.ok(draft=draft)


# --- 2. Build and run the pipeline ---


async def main():
    pipe = Pipeline(gather_requirements, create_outline, draft_content, name="essay")

    # Pre-populate context with initial data
    ctx = PipelineContext({"topic": "async programming"}, metadata={"run_id": "demo-001"})

    result = await pipe.run(ctx)

    # --- 3. Inspect the result ---
    print(f"Pipeline: {pipe.name}")
    print(f"Status:   {result.status}")
    print(f"Duration: {result.duration:.4f}s")
    print(f"Phases:   {len(result.phases)}")
    print()

    # Phase-level details
    for phase in result.phases:
        print(f"  [{phase.status}] {phase.phase} ({phase.duration:.4f}s)")
        if phase.warnings:
            print(f"         warnings: {phase.warnings}")

    # Context holds all accumulated artifacts
    print()
    print(f"Final draft: {result.context['draft']}")
    print()

    # Provenance: which phase produced each artifact
    for key in ["requirements", "outline", "draft"]:
        print(f"  '{key}' produced by: {result.context.produced_by(key)}")

    # Metadata is preserved separately from data
    print()
    print(f"Metadata: {result.context.metadata}")

    # Aggregated warnings from all phases
    print(f"All warnings: {result.total_warnings}")


asyncio.run(main())
