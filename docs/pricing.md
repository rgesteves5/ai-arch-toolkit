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
pricing.has("claude-sonnet-4-20250514")  # True

# Get pricing details
p = pricing.get("claude-sonnet-4-20250514")
p.input   # USD per 1M input tokens
p.output  # USD per 1M output tokens

# Estimate cost
cost, known = pricing.estimate_cost(
    "claude-sonnet-4-20250514",
    input_tokens=1000,
    output_tokens=500,
)

# Register custom pricing
from ai_arch_toolkit.core._pricing import ModelPricing
pricing.register("my-model", ModelPricing(input=1.0, output=3.0))

# List all priced models
pricing.list_models()
```

`ModelPricing` also supports cache, batch, long-context, and fast-mode rates (`cache_write`, `cache_read`, `batch_input`/`batch_output`, `long_context_*`, `fast_input`/`fast_output`).

## Cost tracking across flows

Flow results accumulate cost across all steps:

```python
result = await flow.run(state)
print(f"Total cost: ${result.total_cost:.4f}")

# Per-step breakdown
for st in result.trace.steps:
    print(f"  {st.name}: ${st.cost:.4f}, {st.duration:.1f}s")
```

## Run-wide budgets

To **cap** a flow's spend (and calls, tokens, and wall-time) rather than just measure it, attach a `BudgetPolicy`:

```python
from ai_arch_toolkit import Flow, BudgetPolicy

flow = Flow(*steps, budget_policy=BudgetPolicy(max_cost=0.50, max_llm_calls=20))
```

See [Tool Governance & Safety → Cumulative budgets](safety.md#cumulative-budgets) for the full budget model and enforcement behavior.

---

See also: [LLM Facade](llm.md) · [Middleware](middleware.md) for a cost-guard middleware pattern.
