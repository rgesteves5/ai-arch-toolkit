"""Per-step Policy(max_cost=) after a transient provider error that the LLM retried successfully."""
from __future__ import annotations

import asyncio

import ai_arch_toolkit.core._retry as retry_mod
from ai_arch_toolkit.core import LLM, Policy, Response, Result, Step, Usage
from ai_arch_toolkit.core._exceptions import APIError
from ai_arch_toolkit.core._retry import RetryConfig
from ai_arch_toolkit.toolkit.flow import Flow
from ai_arch_toolkit.core import State

MODEL = "claude-sonnet-4-6"


class Flaky:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, *, system=None, tools=None, **kwargs) -> Response:
        self.calls += 1
        if self.calls == 1:
            raise APIError(503, "overloaded")
        return Response(text="ok", usage=Usage(input_tokens=100, output_tokens=20), model=MODEL)


async def _no_sleep(_s: float) -> None:
    return None


async def main() -> None:
    retry_mod.asyncio.sleep = _no_sleep  # type: ignore[attr-defined]
    prov = Flaky()
    llm = LLM(MODEL, api_key="test")
    llm._provider = prov  # type: ignore[assignment]
    llm._retry = RetryConfig(max_retries=2, base_delay=0.01)

    async def ask(snapshot) -> Result:
        out = await llm.complete("hi")
        return Result(value=out.text)

    flow = Flow(Step(name="ask", fn=ask, policy=Policy(max_cost=1.0)), name="f")
    result = await flow.run(State())
    final = result.final_result
    print("provider calls:", prov.calls, "| step error:", final.error if final else None)
    print("meter:", result.meter)


asyncio.run(main())
