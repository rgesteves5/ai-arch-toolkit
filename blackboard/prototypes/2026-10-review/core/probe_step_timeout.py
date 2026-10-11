"""A provider timeout raised inside a step is taken for the step's own timeout (no network)."""

from __future__ import annotations

import asyncio

from ai_arch_toolkit.core import Policy, RetryConfig
from ai_arch_toolkit.core._exceptions import APIError, ProviderTimeout
from ai_arch_toolkit.core._state import State
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.core._step_engine import execute_step

calls = {"timeout": 0, "api": 0}


async def flaky_timeout(_snapshot) -> Result:
    calls["timeout"] += 1
    if calls["timeout"] == 1:
        raise ProviderTimeout("read timed out")  # what LLM.complete raises on an I/O timeout
    return Result(value="ok")


async def flaky_api(_snapshot) -> Result:
    calls["api"] += 1
    if calls["api"] == 1:
        raise APIError(503, "overloaded")
    return Result(value="ok")


async def main() -> None:
    policy = Policy(retry=RetryConfig(max_retries=2, base_delay=0.01))  # no step timeout set
    for fn in (flaky_api, flaky_timeout):
        result, trace = await execute_step(Step(name=fn.__name__, fn=fn, policy=policy), State().snapshot())
        print(
            f"{fn.__name__:14} value={result.value!r} error={result.error!r} "
            f"attempts={trace.attempts} decisions={trace.policy_decisions}"
        )


asyncio.run(main())
