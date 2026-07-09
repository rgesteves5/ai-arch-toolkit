"""37 — Budgets & metering.

Every Flow / Agent run opens a meter. By default it just MEASURES — afterwards
you read what the run cost from ``result.meter`` (never by summing anything
yourself; the meter is the single source of truth). Attach a ``BudgetPolicy`` to
also ENFORCE caps: the meter denies the call that would breach a cap *before* it
runs, so nothing is over-spent.

Five scenarios:
  1. measure-only (the default)
  2. enforce a budget and hit it
  3. per-run budgets (one flow, a different cap each run)
  4. budget_scope — a budget around raw LLM calls, outside any Flow
  5. audit — retain every usage event

Requires an API key for the model below (e.g. ``OPENAI_API_KEY``). Uses a cheap
model and tiny budgets, so it costs a fraction of a cent to run.
"""

from __future__ import annotations

import asyncio

from ai_arch_toolkit import (
    LLM,
    BudgetExceeded,
    BudgetPolicy,
    BudgetReport,
    Flow,
    MeterScope,
    Result,
    RunConfig,
    State,
    Step,
    budget_scope,
)

MODEL = "gpt-4.1-nano"  # cheap + priced in the registry; swap for any model you have a key for


def _ask(llm: LLM, prompt: str):
    """A trivial step that makes one LLM call. Note: no manual cost/usage bookkeeping —
    the meter captures it automatically at the charge site."""

    async def step(_snap) -> Result:
        await llm.complete(prompt)
        return Result(value="ok")

    return step


async def scenario_measure_only(llm: LLM) -> None:
    print("\n=== 1. Measure-only (the default — no budget attached) ===")
    flow = Flow(
        Step(name="a", fn=_ask(llm, "Name a colour in one word.")),
        Step(name="b", fn=_ask(llm, "Name an animal in one word.")),
        name="measure",
    )
    result = await flow.run(State())

    report: BudgetReport = result.meter  # the authoritative view of what the run consumed
    print(f"  llm_calls = {report.llm_calls}   total_tokens = {report.total_tokens}")
    print(f"  cost = ${result.total_cost:.6f}   usage = {result.usage}")
    print(f"  cost_uncertain = {report.cost_uncertain}  (True if a call couldn't be priced)")


async def scenario_enforce(llm: LLM) -> None:
    print("\n=== 2. Enforce a budget and hit it ===")
    flow = Flow(
        Step(name="a", fn=_ask(llm, "Name a colour.")),
        Step(name="b", fn=_ask(llm, "Name an animal.")),
        Step(name="c", fn=_ask(llm, "Name a country.")),  # this 3rd call is denied
        name="capped",
        budget_policy=BudgetPolicy(max_llm_calls=2),
    )
    result = await flow.run(State())  # returns normally — a denial does NOT raise inside a flow

    report = result.meter
    print(f"  ran {report.llm_calls} call(s), then the cap stopped the run")
    print(f"  over_budget = {report.over_budget}   breached = {report.breached}")
    print(f"  'budget_exceeded' in result.results = {'budget_exceeded' in result.results}")


async def scenario_per_run(llm: LLM) -> None:
    print("\n=== 3. Per-run budgets (reuse one flow, cap each run differently) ===")
    flow = Flow(Step(name="a", fn=_ask(llm, "Say hi.")), name="reusable")

    tight = await flow.run(State(), budget_policy=BudgetPolicy(max_llm_calls=0))
    print(f"  max_llm_calls=0 -> budget_exceeded: {'budget_exceeded' in tight.results}")

    loose = await flow.run(State(), budget_policy=BudgetPolicy(max_llm_calls=5))
    print(f"  max_llm_calls=5 -> ran {loose.meter.llm_calls} call(s)")


async def scenario_budget_scope(llm: LLM) -> None:
    print("\n=== 4. A budget around raw calls (no Flow): budget_scope ===")
    # Outside a Flow there is nothing to convert a denial into a result, so it RAISES.
    with budget_scope(BudgetPolicy(max_llm_calls=1)) as scope:
        try:
            await llm.complete("Write one word.")
            await llm.complete("Write another word.")  # 2nd call exceeds max_llm_calls=1 -> denied
        except BudgetExceeded as exc:
            print(f"  denied: dimension={exc.dimension}, cap={exc.maximum}")

    report = BudgetReport.from_snapshot(scope.snapshot())
    print(f"  spent ${report.cost:.6f} across {report.llm_calls} admitted call(s)")


async def scenario_audit(llm: LLM) -> None:
    print("\n=== 5. Audit: retain every usage event ===")
    with MeterScope(RunConfig(retain_meter_events=True)) as scope:
        await llm.complete("Say hi.")
    for event in scope.events():
        print(
            f"  event op={event.op_id} model={event.model} status={event.status} "
            f"in={event.usage.input_tokens} out={event.usage.output_tokens}"
        )


async def main() -> None:
    llm = LLM(MODEL)
    await scenario_measure_only(llm)
    await scenario_enforce(llm)
    await scenario_per_run(llm)
    await scenario_budget_scope(llm)
    await scenario_audit(llm)


asyncio.run(main())
