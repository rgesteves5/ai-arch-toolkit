"""A stream that is created and dropped, never iterated nor closed, keeps its reservation."""

from __future__ import annotations

import pathlib
import asyncio
import gc
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))  # the repo root, 4 levels up

from ai_arch_toolkit.core import MeterScope, RunConfig
from ai_arch_toolkit.toolkit.budget import BudgetController, BudgetPolicy
from tests.fake_provider import fake_llm


async def main() -> None:
    llm, provider = fake_llm()
    policy = BudgetPolicy(max_llm_calls=1)
    with MeterScope(RunConfig(controller=BudgetController(policy))) as scope:
        llm.stream("never read")  # dropped
        gc.collect()
        print("after dropping the stream:", scope.snapshot().out_llm_calls, "outstanding llm call(s)")
        try:
            await llm.complete("hi")
            print("complete: ok")
        except Exception as exc:  # noqa: BLE001
            print("complete:", type(exc).__name__, exc, "| provider calls:", provider.calls)


asyncio.run(main())
