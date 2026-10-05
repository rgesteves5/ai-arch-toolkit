# Cost Control & Budgets

Every LLM call and every paid tool call has a price, and every `Flow` and `Agent` run has a meter
that records what it spent, run by run and step by step. A budget is enforced by that meter before
each call is sent, not reported after it.

A token counter tells you what a run spent once it has spent it. That is enough for a report, but
not for a limit: by the time a counter sees a run cross $1, the call that crossed it is already
paid, and so is every call running beside it. Here the meter sits where the money is spent (the
LLM call and the tool executor), and it admits each call before the call is sent.

## What you get

- **The cost of everything, with no bookkeeping.** `response.cost` for a call, `result.cost` and
  `result.report` for a run, `StepTrace.metered` for each step. Retries, fallbacks, tools and
  nested agents are included. Prices cover cache reads and writes, long-context tiers, batch and
  fast modes, and image tokens.
- **No silent zero.** Inside a run, a call to a model with no price is refused before anything is
  sent (`UnpricedModelError`). A cost that can't be known, such as a provider-hosted tool's, is
  reported as unknown (`cost_uncertain`) rather than as $0.
- **Caps that stop a run cleanly.** A `BudgetPolicy` caps USD, tokens, LLM calls, tool calls and
  wall time. The meter checks each call against the caps before sending it, and refuses it once a
  cap is reached. The run then ends normally, and the report says which cap stopped it.
- **A hard cost ceiling when you need one.** With `reserve="strict"`, each call holds its worst
  case before it is sent, so calls running in parallel cannot pass `max_cost` together.
- **Failed calls count.** A timeout or a dropped stream that the provider may have billed counts at
  its worst case, not at zero. A 429 costs nothing. Each adapter follows its provider's billing
  documentation.
- **Paid tools on the same meter.** Brave Search and Tavily, and your own paid APIs, are priced in
  the same table as the models, and charged the units the service billed.
- **One ceiling for many runs.** Parallel runs can spend from one `SharedBudget`, seeded with what
  your app already spent. Every call holds its worst case there, so the runs cannot race past it
  together.
- **Prices with an end date.** A promotional price can name its last day and the price that follows
  it. The meter switches on that day by itself.
- **An audit trail.** A `UsageSink` receives one redacted event per call: model, tokens, cost,
  outcome, and the step it ran in.

## Three modes

| Mode | When | What happens |
|---|---|---|
| Unmetered | An `LLM` call outside any run | `response.cost` is computed and nothing is checked. An unpriced model gives `None`. |
| Measure | Every `Flow` and `Agent` run, by default | Every call is recorded. An unpriced model is refused before sending. |
| Enforce | A `budget_policy=`, a `budget_scope(...)`, or a `RunConfig(controller=...)` | Everything Measure does, plus every call is admitted against the caps first. |

Nested flows and sub-agents run under the enclosing run's meter, so one budget covers them all.

## See what things cost

One call:

```python
from ai_arch_toolkit import LLM

llm = LLM("gpt-4.1-mini")
response = llm.complete_sync("Name three primes.")
print(response.cost)  # USD from the bundled price table; None when the model has no price
```

A run, and each of its steps:

```python
from ai_arch_toolkit import LLM, ToolGroup
from ai_arch_toolkit.toolkit.agents import Agent, ReasoningSpec
from ai_arch_toolkit.toolkit.tools import geocode, get_weather

agent = Agent(ReasoningSpec(strategy="react"), LLM("gpt-4.1-mini"), ToolGroup(get_weather, geocode))
result = agent.run_sync("Weather and coordinates of Tokyo?")

report = result.report                  # a BudgetReport: the run's meter
print(report.cost, report.llm_calls, report.tool_calls, report.total_tokens)

for step in result.flow_result.trace.steps:
    if step.metered is not None:        # what the meter measured in this step's own span
        print(step.name, step.metered.cost.to_float(), step.metered.llm_calls)
```

A `Flow` run gives the same report as `result.meter`, with `result.total_cost` and `result.usage`
as shortcuts. Always read spend from the meter. `Result(cost=...)` annotations and
`trace.total_cost` only add up what steps reported about themselves.

`report.cost` is known spend. `report.cost_at_most` adds the bounds of failed calls that may have
been billed, and is `None` when some cost has no bound. `report.cost_uncertain` tells you whether
either kind of uncertainty is present. To measure a block of code that is not a flow step, open a
span of your own ([Flow Architecture → Trace](flow-architecture.md#trace)).

## Cap a run

```python
from ai_arch_toolkit import BudgetPolicy

result = agent.run_sync(
    "Weather and coordinates of Tokyo?",
    budget_policy=BudgetPolicy(max_cost=0.05, max_llm_calls=10),
)
if result.report.over_budget:
    print("stopped by", result.report.breached)  # e.g. ('llm_calls',)
```

The budget is set when you run the agent, not in its spec, so one agent can run under different
caps. `Flow(budget_policy=...)` and `flow.run(state, budget_policy=...)` work the same way, and an
agent manifest can declare it under `limits` ([Configuring Agents](configuring-agents.md)).
When the run owns its budget, a denial does not raise: the run returns its result, and the trace
records `budget_exceeded`.

How tightly each cap holds:

| Cap | Checked | How tight |
|---|---|---|
| `max_llm_calls`, `max_tool_calls` | Before each call, under the meter's lock | Exact, also with parallel steps |
| `max_cost` and the token caps, default `reserve="none"` | Before each call, against what has settled | Soft: the call that crosses the cap still completes (as do any calls running beside it), and the next one is refused |
| The same caps, `reserve="strict"` | Before each call, with that call's worst case held | Hard, as far as the hold is right |
| `max_wall_s` | Between steps, and whenever a call is admitted | A call already running is not interrupted. Use `Policy(timeout=...)` or `Flow(timeout=...)` for that |

For example, take a $0.05 cap and calls that cost $0.03 each. Under the default, the second call is
admitted (only $0.03 has settled) and the run ends at $0.06. Under `reserve="strict"`, the second
call's hold does not fit, so the run ends at $0.03.

### Strict reservations

```python
BudgetPolicy(max_cost=0.05, reserve="strict")
```

A call's hold is its worst case at the run's prices. The input side is the request's size at four
characters per token, plus an allowance for each image or document. The output side is the call's
`max_tokens` (the `LLM` default is 4096), plus, for each image requested, the tokens its provider
publishes for that model, quality and size (or a generous allowance). The hold is released and
replaced by the real cost when the call settles.

- **The hold follows `max_tokens`.** With `gpt-4.1-mini` ($1.60 per 1M output tokens), a call with
  the default 4096 holds about $0.0066. A call whose hold does not fit is refused, even if its
  real cost would have fit, so set `max_tokens` to what your calls need.
- **The input side is an estimate.** Text that splits into more tokens than one per four
  characters (digits, code, non-Latin scripts) can use more input than its hold. The call is
  still charged in full.
- **A call that cannot be priced is refused.**

### Cap a single step

`Policy(max_cost=...)` caps what one step spends, including its retries, its fallback and the flows
it runs:

```python
from ai_arch_toolkit import Policy, Step

step = Step(name="research", fn=research, policy=Policy(max_cost=0.10))
```

It is checked when the step ends. A step that spent more returns an error instead of its result,
with `cost_exceeded` in the trace, and is not retried. To stop the spending itself, use a run
budget.

## Budgets outside a flow

`budget_scope` puts a budget around any LLM or tool calls. There is no run here to absorb a denial,
so a denial raises:

```python
from ai_arch_toolkit import BudgetExceeded, BudgetPolicy, BudgetReport, budget_scope

policy = BudgetPolicy(max_cost=0.10)
with budget_scope(policy) as scope:
    try:
        for chunk in chunks:
            await llm.complete(f"Summarise: {chunk}")
    except BudgetExceeded as exc:
        print("stopped at", exc.dimension, exc.maximum)

report = BudgetReport.from_snapshot(scope.snapshot(), policy)
```

Flows and agents that run inside the scope spend from it too. In that case a denial propagates out
of `flow.run()`, because the scope owns the budget, not the flow.

## One ceiling for many runs

A `BudgetPolicy` caps one run. To let runs in parallel spend from one ceiling, bind each of them to
the same `SharedBudget`, seeded with what was already spent:

```python
import asyncio

from ai_arch_toolkit import BudgetPolicy, RunConfig
from ai_arch_toolkit.toolkit.budget import SharedBudget

today = SharedBudget(BudgetPolicy(max_cost=5.0), spent=ledger.spent_today())  # your app's ledger
config = RunConfig(shared=today)
await asyncio.gather(*(agent.run(task, config=config) for task in tasks))

print(today.report().cost)  # the seed plus what every run spent
```

Inside the shared ceiling, every call holds its worst case, also in a run with no budget of its
own, so runs in parallel cannot pass `max_cost` together. Near the ceiling, a call whose worst case
does not fit is refused. Only `max_cost`, `max_llm_calls` and `max_tool_calls` are shared. A run can
also keep a budget of its own: `RunConfig(controller=BudgetController(policy), shared=today)`.
Details: [A budget several runs share](safety.md#a-budget-several-runs-share).

## Failed calls

A failed call may or may not have been billed. Each adapter classifies the failure by its
provider's documentation:

- **Not sent** (refused before sending): costs nothing.
- **Unbilled** (a 429; Anthropic's error responses; Gemini's 400 and 500): costs nothing.
- **Indeterminate** (other providers' error responses, a timeout or a lost connection after
  sending, a dropped stream, an error inside a stream): counted at its worst case.

An indeterminate failure does not add to `report.cost`, because nothing is known to have been
charged. Its bound goes into `report.cost_at_most` and counts against `max_cost`. A response that
fails but reports its usage (Meta's `response.failed`) is charged at that usage. The full matrix is
in [Failed LLM calls](safety.md#failed-llm-calls).

## Paid tools

A tool that calls a paid API is priced in the same table as the models, by the tool's name, per
unit its service bills. The meter holds one unit before the call and charges the units the
service reports when it ends. A request the service refused costs nothing. `brave_search` and
`tavily_search` come priced. A tool of your own declares its API with `Api(..., billed_as=...)`
([Pricing → Paid tools](pricing.md#paid-tools)).

## Audit and export

A sink receives one event per finished call: `model`, `usage`, `cost`, `status`, `delivery`, and
`span_id` (the step it ran in). Its metadata is redacted. By default, a sink that raises is logged
and does not break the run. Write the events to your ledger, and seed tomorrow's `SharedBudget`
from it.

```python
from ai_arch_toolkit import BudgetController, BudgetPolicy, RunConfig, UsageEvent


class PrintSink:
    def emit(self, event: UsageEvent) -> None:
        print(event.span_id, event.model, event.status, event.cost)


policy = BudgetPolicy(max_cost=0.50)
config = RunConfig(controller=BudgetController(policy), sinks=[PrintSink()])
result = await agent.run(task, config=config)
```

`RunConfig(retain_meter_events=True)` keeps the events in memory instead (`scope.events()`).
`BudgetReport.to_dict()` and `MeterSnapshot.to_dict()` give JSON-ready views. The
`TracingMiddleware` adds tokens and cost to an OpenTelemetry span per LLM call
([Middleware](middleware.md#tracingmiddleware)).

## Limits

- **No calendar windows.** For a daily budget, create a `SharedBudget` per day, seeded from your
  ledger.
- **One process.** A `SharedBudget` is an object in memory. It does not coordinate several
  processes or machines.
- **Provider-hosted tools are not priced.** A provider-run tool, such as its web search or code
  execution, is billed outside the token counts, so its cost is unknown. Under a `max_cost`, the
  default `unpriced="fail_closed"` stops the run after such a call, and a strict budget refuses it
  before sending. `unpriced="allow"` lets the run go on with a cost that undercounts.
- **The Batch API is not metered.** Under an enforcing budget, `batch_*` calls are refused unless
  `RunConfig(allow_unmetered_batch=True)`.
- **The price table is maintained by hand.** It ships in the package (`_default_pricing.toml`).
  Register your own rates with `pricing.register()`, or load a file with `pricing.load()`.
  When the provider reports a call's cost itself, as xAI does, that amount is used.

## Read more

- [Pricing & Cost Tracking](pricing.md): the price table, how a model id finds its price,
  promotional prices, paid tools.
- [Cumulative budgets](safety.md#cumulative-budgets): the full budget contract, `reserve` and
  `unpriced`.
- [Agent & ReasoningSpec → Budgets](agents.md#budgets) and [Flow Architecture → Policy](flow-architecture.md#policy).
- [Examples](examples.md): `37_budgets_and_metering.py` runs measure-only, an enforced budget,
  per-run budgets, `budget_scope`, and audit events.
