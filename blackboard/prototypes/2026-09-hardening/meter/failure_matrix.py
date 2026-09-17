"""Failure classes under a cost cap: what happens to the NEXT admission?"""
from __future__ import annotations

import asyncio

import ai_arch_toolkit.core._retry as retry_mod
from ai_arch_toolkit.core import LLM, Response, Usage
from ai_arch_toolkit.core._exceptions import APIError
from ai_arch_toolkit.core._retry import RetryConfig
from ai_arch_toolkit.toolkit.budget import BudgetPolicy, budget_scope

MODEL = "claude-sonnet-4-6"
OK = Response(text="ok", usage=Usage(input_tokens=100, output_tokens=20), model=MODEL)
_real_sleep = asyncio.sleep


async def _no_sleep(_s: float) -> None:
    return None


class Scripted:
    """Provider whose i-th call runs script[i] (an exception to raise, a delay, or 'ok')."""

    def __init__(self, *script: object) -> None:
        self.script, self.calls = list(script), 0

    async def complete(self, messages, *, system=None, tools=None, **kwargs) -> Response:
        step = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, float):
            await _real_sleep(step)
        return OK


def llm_with(provider: Scripted, *, retry: int = 0, fallback: LLM | None = None) -> LLM:
    llm = LLM(MODEL, api_key="test", fallback=[fallback] if fallback else None)
    llm._provider = provider  # type: ignore[assignment]
    if retry:
        llm._retry = RetryConfig(max_retries=retry, base_delay=0.01)
    return llm


async def run(name: str, body) -> None:
    with budget_scope(BudgetPolicy(max_cost=5.0)) as scope:
        try:
            outcome = await body()
        except BaseException as exc:  # noqa: BLE001
            outcome = f"{type(exc).__name__}: {exc}"
        snap = scope.snapshot()
    print(f"{name:<36} -> {outcome}  [llm_calls={snap.llm_calls} unknown={snap.unknown_cost_count}]")


async def main() -> None:
    async def fallback_case():
        fb = llm_with(Scripted("ok"))
        primary = llm_with(Scripted(APIError(503, "overloaded")), fallback=fb)
        return (await primary.complete("hi")).text

    await run("primary 503 -> fallback", fallback_case)

    async def network_case():
        retry_mod.asyncio.sleep = _no_sleep  # type: ignore[attr-defined]
        try:
            llm = llm_with(Scripted(ConnectionError("reset"), "ok"), retry=2)
            return (await llm.complete("hi")).text
        finally:
            retry_mod.asyncio.sleep = _real_sleep  # type: ignore[attr-defined]

    await run("ConnectionError -> retry", network_case)

    async def timeout_case():
        llm = llm_with(Scripted(0.2, "ok"))
        try:
            await asyncio.wait_for(llm.complete("hi"), timeout=0.02)
        except TimeoutError:
            pass
        return (await llm.complete("again")).text

    await run("caller timeout -> next call", timeout_case)

    async def validation_case():
        llm = llm_with(Scripted(ValueError("tool_choice not supported"), "ok"))
        try:
            await llm.complete("hi")
        except ValueError:
            pass
        return (await llm.complete("again")).text

    await run("adapter ValueError -> next call", validation_case)


asyncio.run(main())
