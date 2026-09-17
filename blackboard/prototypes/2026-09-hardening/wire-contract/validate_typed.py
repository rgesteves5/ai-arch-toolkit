"""Validate the captured kwargs of the three TypedDict-based SDK surfaces, strictly."""
from __future__ import annotations

import asyncio
import statistics
import sys
import time
from typing import Any

from capture import capture
from strict import check, strict_validator


def targets(adapter: str) -> dict[str, Any]:
    if adapter == "anthropic":
        from anthropic.types.message_create_params import (
            MessageCreateParamsNonStreaming,
            MessageCreateParamsStreaming,
        )

        # messages.stream(**kw) takes the same kwargs as create() without ``stream``.
        return {
            "messages.create": MessageCreateParamsNonStreaming,
            "messages.stream": MessageCreateParamsNonStreaming,
            "_streaming": MessageCreateParamsStreaming,
        }
    if adapter == "openai":
        from openai.types.chat.completion_create_params import (
            CompletionCreateParamsNonStreaming,
            CompletionCreateParamsStreaming,
        )

        return {
            "chat.completions.create": CompletionCreateParamsNonStreaming,
            "_streaming": CompletionCreateParamsStreaming,
        }
    from openai.types.responses.response_create_params import (
        ResponseCreateParamsNonStreaming,
        ResponseCreateParamsStreaming,
    )

    return {
        "responses.create": ResponseCreateParamsNonStreaming,
        "_streaming": ResponseCreateParamsStreaming,
    }


async def main(adapter: str, model: str | None) -> None:
    types = targets(adapter)
    validators: dict[Any, Any] = {}
    for tp in set(types.values()):
        t0 = time.perf_counter()
        validators[tp] = strict_validator(tp)
        print(f"build {tp.__name__}: {(time.perf_counter() - t0) * 1000:.0f} ms")

    timings: list[float] = []
    for cap in await capture(adapter, model=model):
        if cap.kwargs is None:
            print(f"  skip  {cap.scenario:24s} adapter raised: {cap.error}")
            continue
        tp = types["_streaming"] if cap.kwargs.get("stream") else types[cap.method]
        t0 = time.perf_counter()
        errs = check(validators[tp], cap.kwargs)
        timings.append((time.perf_counter() - t0) * 1000)
        print(f"  {'FAIL' if errs else 'pass'}  {cap.scenario:24s} [{tp.__name__}]")
        for e in errs[:8]:
            print("           ", e[:300])
        if len(errs) > 8:
            print(f"            ... +{len(errs) - 8} more")
    print(
        f"per-validation: median {statistics.median(timings):.2f} ms, max {max(timings):.2f} ms "
        f"(n={len(timings)})"
    )


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
