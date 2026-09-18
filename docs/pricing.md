# Pricing & Cost Tracking

Every response carries an estimated cost, computed from a built-in pricing registry — so you can track and cap spend without wiring up your own price table.

## Automatic cost estimation

```python
response = await llm.complete("Hello")
print(f"${response.cost:.6f}")  # e.g. $0.000342
```

Costs come from a bundled pricing registry (`_default_pricing.toml`) covering the supported models.
When the provider reports the call's cost itself (xAI does), `response.cost` is that amount. It is
`None` when the model has no price, or when the provider reported no usage (an
OpenAI-compatible server that sends no usage chunk, for example): an unknown cost is never zero.

## Pricing registry

```python
from ai_arch_toolkit import pricing

# Check if a model has pricing
pricing.has("claude-sonnet-5")  # True

# Get pricing details
p = pricing.get("claude-sonnet-5")
p.input   # USD per 1M input tokens
p.output  # USD per 1M output tokens

# Estimate cost (None means the model has no registered pricing)
cost = pricing.estimate_cost(
    "claude-sonnet-5",
    input_tokens=1000,
    output_tokens=500,
)

# Register custom pricing through the public API
from ai_arch_toolkit import ModelPricing
pricing.register("my-model", ModelPricing(input=1.0, output=3.0))

# A family of local models, priced at zero explicitly
pricing.register("llama3", ModelPricing(), match="prefix")

# List all priced models
pricing.list_models()
```

### How a model id finds its price

A model id is priced by its own entry, or as a dated snapshot of an entry:
`claude-haiku-4-5-20251001`, `gpt-4o-2024-08-06`, `grok-4-0709`, and `-latest` ids take the price of
the id they snapshot, unless they have an entry of their own (`gpt-4o-2024-05-13` does). A variant
never inherits the entry its name starts with: `o3-pro`, `gpt-4o-audio-preview`, and
`gemini-2.5-flash-image` have no price from `o3`, `gpt-4o`, or `gemini-2.5-flash`.
`register()` is exact by default and also covers the model's dated snapshots;
`match="prefix"` prices every id that starts with the given one, for a family of local models, and
an id's own entry always wins over a prefix. A TOML file loaded with `pricing.load(path)` takes the
same options per entry: `aliases = [...]` for other ids billed at the entry's rates, and
`match = "prefix"`.

`ModelPricing` also supports cache, batch, long-context, and fast-mode rates. Mode-specific
cache and long-context fields (`batch_cache_*`, `batch_long_context_*`, `fast_cache_*`, and
`fast_long_context_*`) allow these tariffs to combine correctly; omitted fields fall back to
the selected mode's base rates, then standard rates. `long_context_inclusive=True` selects the
premium tier at the threshold itself, as required by xAI.

## Unpriced models under a meter

A metered call must be priceable before it is sent. Inside a `MeterScope` (every `Flow` and
`Agent` run opens one, with or without a budget), a call to a model the scope's pricer cannot price
raises `UnpricedModelError` before anything is sent. It is a `RequestError`, so it is neither
retried nor passed to a fallback, and its message names the `pricing.register(...)` call that fixes
it. A local model registers an explicit zero:

```python
pricing.register("llama3.2", ModelPricing())
```

Without a scope nothing is checked: the call runs and `Response.cost` is `None`. Server tools
(priced per use) are the other unknown cost; `BudgetPolicy(unpriced=...)` governs them.

## Cost tracking across flows

Every `Flow`/`Agent` run opens a meter (measure-only unless you attach a budget). The meter is the **single source of truth** for what the run consumed — read it off the result rather than summing anything yourself:

```python
result = await flow.run(state)

report = result.meter                # BudgetReport (None only if unmetered)
print(f"Total cost: ${result.total_cost:.4f}")        # == report.cost
print(f"{report.llm_calls} LLM calls, {report.total_tokens} tokens")
if report.cost_uncertain:            # some call couldn't be priced -> cost is a lower bound
    print(f"Cost bound: {report.cost_at_most!r}")  # None if any cost is unbounded

# Per-step timing is still on the trace (per-step cost lives in the meter, not the trace):
for st in result.trace.steps:
    print(f"  {st.name}: {st.duration:.1f}s")
```

`Agent` results expose the same via `agent_result.report` / `.cost` / `.usage`.

`report.cost` is known spend. `report.cost_at_most` adds the retained bounds of indeterminate failed
attempts; it is `None` if any cost is unbounded. `report.cost_uncertain` is true for either kind of
uncertainty. Retries and fallback calls each consume a call count. An unbilled 429 contributes zero
cost; a failed response, transport timeout or abandoned stream can consume a bounded allowance
without becoming reported spend. A failed response that reports its usage (Meta's
`response.failed`) is settled at that usage's price instead. Each adapter's `delivery` follows its
provider's documented billing ([LLM Facade → Provider errors](llm.md#provider-errors)).

Bounded uncertainty enters `max_cost` and per-step cost checks. `unpriced="fail_closed"` applies
only to unknowns with no bound. Strict budgets retain the failed operation's own worst-case hold;
soft budgets estimate the bound at failure. Measure-only runs have no controller to supply a bound.
Strict reservations also cover tools priced by a custom `Pricer`. Use
`budget_scope(policy, pricer=...)` to wire the same pricer into reservation and settlement.

## Run-wide budgets

To **cap** a flow's spend (and calls, tokens, and wall-time) rather than just measure it, attach a `BudgetPolicy`:

```python
from ai_arch_toolkit import Flow, BudgetPolicy

flow = Flow(*steps, budget_policy=BudgetPolicy(max_cost=0.50, max_llm_calls=20))
```

See [Tool Governance & Safety → Cumulative budgets](safety.md#cumulative-budgets) for the full budget model and enforcement behavior.

---

See also: [LLM Facade](llm.md) · [Middleware](middleware.md) for a cost-guard middleware pattern.
