# Pricing & Cost Tracking

Every response carries an estimated cost, computed from a built-in pricing registry — so you can track and cap spend without wiring up your own price table.

## Automatic cost estimation

```python
response = await llm.complete("Hello")
print(f"${response.cost:.6f}")  # e.g. $0.000342
```

Costs come from a bundled pricing registry (`_default_pricing.toml`) covering the supported models.

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

# List all priced models
pricing.list_models()
```

`ModelPricing` also supports cache, batch, long-context, and fast-mode rates. Mode-specific
cache and long-context fields (`batch_cache_*`, `batch_long_context_*`, `fast_cache_*`, and
`fast_long_context_*`) allow these tariffs to combine correctly; omitted fields fall back to
the selected mode's base rates, then standard rates. `long_context_inclusive=True` selects the
premium tier at the threshold itself, as required by xAI.

## Cost tracking across flows

Every `Flow`/`Agent` run opens a meter (measure-only unless you attach a budget). The meter is the **single source of truth** for what the run consumed — read it off the result rather than summing anything yourself:

```python
result = await flow.run(state)

report = result.meter                # BudgetReport (None only if unmetered)
print(f"Total cost: ${result.total_cost:.4f}")        # == report.cost
print(f"{report.llm_calls} LLM calls, {report.total_tokens} tokens")
if report.cost_uncertain:            # some call couldn't be priced -> cost is a lower bound
    print("(cost undercounts: an unpriced model or server tool was used)")

# Per-step timing is still on the trace (per-step cost lives in the meter, not the trace):
for st in result.trace.steps:
    print(f"  {st.name}: {st.duration:.1f}s")
```

`Agent` results expose the same via `agent_result.report` / `.cost` / `.usage`.

## Run-wide budgets

To **cap** a flow's spend (and calls, tokens, and wall-time) rather than just measure it, attach a `BudgetPolicy`:

```python
from ai_arch_toolkit import Flow, BudgetPolicy

flow = Flow(*steps, budget_policy=BudgetPolicy(max_cost=0.50, max_llm_calls=20))
```

See [Tool Governance & Safety → Cumulative budgets](safety.md#cumulative-budgets) for the full budget model and enforcement behavior.

---

See also: [LLM Facade](llm.md) · [Middleware](middleware.md) for a cost-guard middleware pattern.
