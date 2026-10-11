"""A successful call whose provider reports no usage is an unbounded unknown cost (no network)."""

from __future__ import annotations

import pathlib
import asyncio
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))  # the repo root, 4 levels up

from ai_arch_toolkit.core import MeterScope, ModelPricing, RunConfig, pricing
from ai_arch_toolkit.core._exceptions import APIError
from ai_arch_toolkit.toolkit.budget import BudgetController, BudgetPolicy
from tests.fake_provider import Reply, fake_llm

pricing.register("local-llama", ModelPricing())  # a local model, priced zero explicitly (D16)


async def main() -> None:
    llm, _ = fake_llm(Reply(usage_reported=False), model="local-llama", base_url="http://localhost:1/v1")
    with MeterScope() as scope:
        await llm.complete("hi")
        s = scope.snapshot()
        print("success without usage:", "unknown", s.unknown_cost_count, "uncertain", s.uncertain_cost)

    failing, _ = fake_llm(APIError(500, "boom"), model="local-llama", base_url="http://localhost:1/v1")
    with MeterScope() as scope:
        try:
            await failing.complete("hi")
        except APIError:
            pass
        s = scope.snapshot()
        print("failure (500)       :", "unknown", s.unknown_cost_count, "uncertain", s.uncertain_cost)

    llm, _ = fake_llm(Reply(usage_reported=False), model="local-llama", base_url="http://localhost:1/v1")
    policy = BudgetPolicy(max_cost=5.0)
    with MeterScope(RunConfig(controller=BudgetController(policy))):
        await llm.complete("first")
        try:
            await llm.complete("second")
            print("second call: ok")
        except Exception as exc:  # noqa: BLE001
            print("second call:", type(exc).__name__, exc)


asyncio.run(main())
