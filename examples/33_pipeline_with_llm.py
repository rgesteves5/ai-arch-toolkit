"""33 — Pipeline with LLM.

A realistic pipeline that calls an LLM in each phase, accumulates
token usage, and demonstrates how Pipeline integrates with the
core LLM class.

Requires API key: set OPENAI_API_KEY in environment.
"""

from __future__ import annotations

import asyncio

from ai_arch_toolkit import LLM
from ai_arch_toolkit.toolkit.pipeline import PhaseResult, Pipeline, PipelineContext


async def research(ctx: PipelineContext) -> PhaseResult:
    """Generate key points about the topic."""
    llm: LLM = ctx.require("llm")
    topic = ctx.require("topic")

    response = await llm.complete(
        f"List 3 key points about: {topic}. Be concise, one sentence each."
    )
    return PhaseResult.ok(
        key_points=response.text,
        token_usage={
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
        },
    )


async def draft(ctx: PipelineContext) -> PhaseResult:
    """Write a short paragraph using the key points."""
    llm: LLM = ctx.require("llm")
    key_points = ctx.require("key_points")

    response = await llm.complete(
        f"Write a concise paragraph incorporating these points:\n{key_points}"
    )
    return PhaseResult.ok(
        draft_text=response.text,
        token_usage={
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
        },
    )


async def review(ctx: PipelineContext) -> PhaseResult:
    """Review and score the draft."""
    llm: LLM = ctx.require("llm")
    draft_text = ctx.require("draft_text")

    response = await llm.complete(
        f"Rate this text 1-10 for clarity and suggest one improvement:\n\n{draft_text}"
    )
    return PhaseResult.ok(
        review=response.text,
        token_usage={
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
        },
    )


async def main():
    llm = LLM("gpt-4.1-nano")
    pipe = Pipeline(research, draft, review, name="content-pipeline")

    ctx = PipelineContext(
        {"topic": "why async programming matters", "llm": llm},
        metadata={"model": "gpt-4.1-nano"},
    )

    # Stream phase-by-phase for real-time progress
    print(f"Running pipeline: {pipe.name}")
    print(f"Phases: {pipe.phase_names}")
    print()

    async for name, result in pipe.iter(ctx):
        status = "OK" if result.is_ok else result.status.upper()
        tokens = result.token_usage or {}
        print(f"  [{status}] {name} — {tokens.get('input', 0)}+{tokens.get('output', 0)} tokens")

    # Final summary
    print()
    print(f"Total tokens: {ctx.total_token_usage}")
    print(f"Total duration: {ctx.total_duration:.2f}s")
    print()
    print("--- Key Points ---")
    print(ctx["key_points"])
    print()
    print("--- Draft ---")
    print(ctx["draft_text"])
    print()
    print("--- Review ---")
    print(ctx["review"])


asyncio.run(main())
