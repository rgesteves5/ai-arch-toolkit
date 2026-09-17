"""End-to-end: which failures poison a cost-capped MeterScope today (loopback only, dummy keys)."""

from __future__ import annotations

import asyncio
import sys
import warnings

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import fakeserver as fs  # noqa: E402

from ai_arch_toolkit.core import LLM, MeterScope, RunConfig  # noqa: E402
from ai_arch_toolkit.toolkit.budget import BudgetController, BudgetPolicy  # noqa: E402

warnings.simplefilter("ignore")


def snap(scope: MeterScope) -> str:
    s = scope.snapshot()
    return f"llm_calls={s.llm_calls} unknown_cost_count={s.unknown_cost_count} cost={s.cost}"


async def case(title: str, make_llm, first, port_stats=None) -> None:
    print(f"\n=== {title}")
    policy = BudgetPolicy(max_cost=1.0)
    with MeterScope(RunConfig(controller=BudgetController(policy))) as scope:
        llm = make_llm()
        try:
            await first(llm)
            print("  first call: no exception")
        except Exception as exc:  # noqa: BLE001
            print(f"  first call raised {type(exc).__name__}: {str(exc)[:90]}")
        if port_stats is not None:
            print(f"  server saw requests={port_stats.requests}")
        print(f"  after first: {snap(scope)}")
        try:
            await llm.complete("hi")
            print("  second call: no exception")
        except Exception as exc:  # noqa: BLE001
            print(f"  second call raised {type(exc).__name__}: {str(exc)[:110]}")
        await llm.close()


async def main() -> None:
    # (a) build-phase ValueError in the Meta adapter (tool_choice='required'): nothing is sent
    srv, port, st = await fs.start("status", status=200, body={"unexpected": True})
    await case(
        "Meta build error via LLM.complete (tool_choice='required')",
        lambda: LLM("muse-spark-1", api_key="dummy", base_url=f"http://127.0.0.1:{port}/v1"),
        lambda llm: llm.complete("hi", tool_choice="required"),
        st,
    )
    srv.close()

    # (a2) same through LLM.stream
    srv, port, st = await fs.start("status", status=200, body={"unexpected": True})

    async def _stream(llm: LLM) -> None:
        async for _ in llm.stream("hi", tool_choice="required"):
            pass

    await case(
        "Meta build error via LLM.stream (tool_choice='required')",
        lambda: LLM("muse-spark-1", api_key="dummy", base_url=f"http://127.0.0.1:{port}/v1"),
        _stream,
        st,
    )
    srv.close()

    # (b) HTTP 429 on Anthropic (not billed)
    body = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}}
    srv, port, st = await fs.start("status", status=429, body=body)
    await case(
        "Anthropic HTTP 429 via LLM.complete (no retry configured)",
        lambda: LLM("claude-sonnet-4-5", api_key="dummy", base_url=f"http://127.0.0.1:{port}"),
        lambda llm: llm.complete("hi"),
        st,
    )
    srv.close()

    # (c) never connected
    closed = fs.free_port()
    await case(
        "OpenAI connect refused via LLM.complete",
        lambda: LLM("gpt-4o-mini", api_key="dummy", base_url=f"http://127.0.0.1:{closed}/v1"),
        lambda llm: llm.complete("hi"),
    )

    # (d) a validation error raised by LLM itself BEFORE the op opens (json_mode + output_schema)
    await case(
        "LLM-level ValueError before the op opens (json_mode + output_schema)",
        lambda: LLM("gpt-4o-mini", api_key="dummy", base_url=f"http://127.0.0.1:{closed}/v1"),
        lambda llm: llm.complete("hi", json_mode=True, output_schema={"name": "x", "schema": {"type": "object"}}),
    )


asyncio.run(main())
