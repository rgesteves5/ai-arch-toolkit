"""Does a transient provider error under a cost cap block the retry / the rest of the run?"""

from __future__ import annotations

import asyncio

import ai_arch_toolkit.core._retry as retry_mod
from ai_arch_toolkit.core import LLM, Response, Usage
from ai_arch_toolkit.core._exceptions import APIError
from ai_arch_toolkit.core._retry import RetryConfig
from ai_arch_toolkit.toolkit.budget import BudgetPolicy, budget_scope

MODEL = "claude-sonnet-4-6"


class Flaky:
    def __init__(self, failures: int, status: int = 503) -> None:
        self.failures, self.status, self.calls = failures, status, 0

    async def complete(self, messages, *, system=None, tools=None, **kwargs) -> Response:
        self.calls += 1
        if self.calls <= self.failures:
            raise APIError(self.status, "overloaded")
        return Response(text="ok", usage=Usage(input_tokens=100, output_tokens=20), model=MODEL)


async def _no_sleep(_s: float) -> None:
    return None


retry_mod.asyncio.sleep = _no_sleep  # type: ignore[attr-defined]


async def scenario(
    name: str,
    policy: BudgetPolicy | None,
    *,
    failures: int,
    status: int = 503,
    retry: bool = True,
    second_call: bool = False,
) -> None:
    prov = Flaky(failures, status)
    llm = LLM(MODEL, api_key="test")
    llm._provider = prov  # type: ignore[assignment]
    if retry:
        llm._retry = RetryConfig(max_retries=2, base_delay=0.01)
    with budget_scope(policy) as scope:
        try:
            out = await llm.complete("hi")
            first = f"ok text={out.text!r}"
        except Exception as exc:  # noqa: BLE001
            first = f"{type(exc).__name__}: {exc}"
        second = ""
        if second_call:
            try:
                await llm.complete("again")
                second = " | 2nd call ok"
            except Exception as exc:  # noqa: BLE001
                second = f" | 2nd call {type(exc).__name__}: {exc}"
        snap = scope.snapshot()
    print(
        f"{name:<44} provider calls={prov.calls} -> {first}{second} "
        f"[llm_calls={snap.llm_calls} unknown={snap.unknown_cost_count}]"
    )


async def main() -> None:
    cap = BudgetPolicy(max_cost=5.0)
    await scenario("measure-only, 503 then ok, retry=2", None, failures=1)
    await scenario("max_cost=5, 503 then ok, retry=2", cap, failures=1)
    await scenario("max_cost=5, 429 then ok, retry=2", cap, failures=1, status=429)
    await scenario(
        "max_cost=5, unpriced=allow, 503 then ok",
        BudgetPolicy(max_cost=5.0, unpriced="allow"),
        failures=1,
    )
    await scenario(
        "max_cost=5, no retry, 400 then next call",
        cap,
        failures=1,
        status=400,
        retry=False,
        second_call=True,
    )
    await scenario("max_llm_calls=10 only, 503 then ok", BudgetPolicy(max_llm_calls=10), failures=1)


asyncio.run(main())
