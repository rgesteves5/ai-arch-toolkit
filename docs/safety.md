# Tool Governance & Safety

When an LLM can call tools, you need control over *which* tools run, *whether* a human signs off first, *what* leaks into traces, and *how much* a run is allowed to spend. The toolkit handles these with four cooperating mechanisms:

- **Risk metadata** on each tool (`@tool(risk_level=..., requires_approval=...)`).
- **Gates** that run before execution — block dangerous tools, require approval, or dry-run.
- **Structured results** (`ToolResult` / `ToolError`) so failures are data, not exceptions, and error text is redacted.
- **Budgets** (`BudgetPolicy`) that cap a flow's calls, tokens, cost, and wall-time.

The execution pipeline for every tool call is: **resolve → gates (in order) → call-count budget → execute → redact & structure the result**.

---

## Risk metadata

Tag a tool with its risk profile at definition time. This attaches a `ToolRuntimePolicy` the gates read later.

```python
from ai_arch_toolkit import tool

@tool(risk_level="high", requires_approval=True, approval_reason="Deletes data")
def delete_table(name: str) -> str:
    """Drop a database table."""
    return f"dropped {name}"
```

`ToolRuntimePolicy` fields: `capability` (str label), `risk_level` (`"low" | "medium" | "high" | "critical"`), `requires_approval` (bool), `approval_reason` (str). Risk metadata travels with the tool but stays out of the schema sent to the provider — only gates see it.

---

## Structured results

`ToolGroup.execute()` / `async_execute()` and the standalone `execute_tool()` / `async_execute_tool()` never raise on a tool failure. They return a **`ToolResult`**:

```python
result = await group.async_execute(tool_call)

result.ok          # bool — did it succeed?
result.value       # the return value (when ok)
result.error       # a ToolError (when not ok)
result.metadata    # audit info, e.g. gate decisions under metadata["audit"]

result.to_model_text()   # LLM-safe string to send back as a tool_result
result.to_dict()         # JSON-serializable form
```

A **`ToolError`** is structured so an agent (or your retry logic) can reason about it:

| Field | Meaning |
|-------|---------|
| `type` | error code, e.g. `"dangerous_tool_blocked"`, `"approval_denied"` |
| `message` | human-readable message (already redacted) |
| `retryable` | whether retrying might succeed |
| `safe_to_show` | if `False`, `to_model_text()` hides the message from the model |
| `details` | structured extra context |

Construct results directly when writing custom executors:

```python
from ai_arch_toolkit import ToolResult

ToolResult.success(value, metadata=None)
ToolResult.failure("network_error", "backend unreachable", retryable=True)
```

> Exceptions raised inside a tool are caught and wrapped into a `ToolResult.failure(...)` with the message **redacted** (not hidden): the agent sees `"connection failed"`, never `connection_string=postgres://user:pw@host`.

---

## Gates

A gate runs *before* a tool executes and can **pass** (return `None`), **block**, **modify the arguments**, or **dry-run**. Gates implement the `ToolGate` protocol (`check` / `check_sync`). You wire them into a `ToolGroup` two ways:

- `gates=[...]` — explicit gates (e.g. `DangerousToolGate`, `DryRunGate`), run in order.
- `approval_handler=...` — the group manages an `ApprovalGate` for you and **always runs it last**.

> Don't pass your own `ApprovalGate` in `gates=`; use `approval_handler=`. The group appends its own approval gate, so a hand-placed one would be double-gated and denied.

### Dangerous tools

`DangerousToolGate` blocks tools by name unless explicitly allowed. The filesystem/shell/Python/web tools in `ai_arch_toolkit.toolkit.tools.dangerous` (`run_command`, `read_file`, `python_repl`, `http_get`, `scrape_text`, `list_directory`, `search_files`) execute real side effects — gate them.

```python
from ai_arch_toolkit import ToolGroup, DangerousToolGate
from ai_arch_toolkit.toolkit.tools.dangerous import run_command

group = ToolGroup(run_command, gates=[DangerousToolGate(blocked=["run_command"])])

result = await group.async_execute(call)   # call -> run_command
result.ok            # False
result.error.type    # "dangerous_tool_blocked"
```

`DangerousToolGate(*, blocked, allow=False)` — names in `blocked` are refused; set `allow=True` to turn the gate into a no-op (e.g. flip it per environment).

### Human approval

For tools marked `requires_approval=True`, supply an `approval_handler`. The handler receives an **`ApprovalRequest`** (with the *real, unredacted* arguments so it can decide) and returns an **`ApprovalDecision`**. With no handler, approval-required tools are **denied by default**.

```python
from ai_arch_toolkit import ToolGroup, ApprovalDecision

def approve_handler(request):
    # request.tool_name, request.arguments, request.risk_level, request.reason, request.preview
    if request.tool_name == "delete_table" and request.arguments["name"].endswith("_tmp"):
        return ApprovalDecision.approve(reviewer="ci-bot")
    return ApprovalDecision.deny(reason="manual review required")

group = ToolGroup(delete_table, approval_handler=approve_handler)
result = await group.async_execute(call)   # approved -> result.ok is True
```

`ApprovalDecision` factories:

```python
ApprovalDecision.approve(*, modified_args=None, reviewer=None, reason="", metadata=None)
ApprovalDecision.deny(*, reviewer=None, reason="", metadata=None)
```

Returning `modified_args` from `approve(...)` runs the tool with **substituted arguments** — useful for narrowing a request (e.g. forcing a safe target) before letting it through. The full request/decision is recorded under `result.metadata["audit"]["approval"]`. The handler may be sync or async.

### Dry run

`DryRunGate(dry_run=True)` short-circuits execution and reports what *would* have run, without side effects — useful for previews and tests.

```python
from ai_arch_toolkit import DryRunGate
group = ToolGroup(run_command, gates=[DryRunGate(dry_run=True)])
```

### Executing a single tool call

When you're not using a `ToolGroup`, run one call against a plain list of functions:

```python
from ai_arch_toolkit import execute_tool, async_execute_tool

result = execute_tool(tool_call, [get_weather, delete_table], approval_handler=approve_handler)
result = await async_execute_tool(tool_call, [get_weather], approval_handler=None)
```

---

## Trace redaction

Traces and tool results can carry secrets (API keys, tokens, connection strings). The redactor strips them by **key name** (`api_key`, `password`, `token`, `secret`, `authorization`, `bearer`, …) and by **value pattern** (Bearer tokens, PEM blocks, `sk-…` keys, database URLs).

```python
from ai_arch_toolkit import redact, redact_text, RedactionPolicy

redact({"api_key": "sk-abc123", "city": "Lisbon"})
# -> {"api_key": "[REDACTED]", "city": "Lisbon"}

redact_text("Authorization: Bearer sk-secret-token here")
# -> "Authorization: Bearer [REDACTED] here"
```

Control behavior with a `RedactionPolicy`:

```python
RedactionPolicy(
    trace_mode="redacted",        # "metadata_only" | "redacted" | "full_debug"
    replacement="[REDACTED]",     # substitution string
)

redact(payload, RedactionPolicy(trace_mode="full_debug"))   # pass-through, no redaction
redact(payload, RedactionPolicy(replacement="***"))         # custom marker
```

| `trace_mode` | Effect |
|--------------|--------|
| `metadata_only` | keep only metadata, drop payloads |
| `redacted` | **default** — keep payloads but mask secrets |
| `full_debug` | no redaction (local debugging only) |

`Redactor(policy)` is the reusable object behind `redact()`; `redact()` / `redact_text()` are the one-shot helpers (a `None` policy uses the safe default).

---

## Cumulative budgets

A `BudgetPolicy` caps an entire flow run. Attach it to a `Flow` via `budget_policy=`; the run's meter accumulates across every step (and every nested agent flow — they share one cumulative budget).

```python
from ai_arch_toolkit import Flow, BudgetPolicy

flow = Flow(
    step_a, step_b, step_c,
    budget_policy=BudgetPolicy(
        max_llm_calls=20,
        max_total_tokens=100_000,
        max_cost=0.50,     # USD, priced via the registry
        max_wall_s=60.0,   # seconds
    ),
)
result = flow.run_sync(state)
```

`BudgetPolicy` caps (all optional, `None` = unlimited): `max_llm_calls`, `max_tool_calls`, `max_input_tokens`, `max_output_tokens`, `max_total_tokens`, `max_cost` (USD), `max_wall_s`. Two knobs shape the cost cap:

- `reserve` (`"none"` default | `"strict"`) — `"strict"` reserves a worst-case token/cost hold *before* each call, failing closed on unpriced models. `"none"` measures and settles only (a soft cost cap can overshoot by at most the one in-flight call).
- `unpriced` (`"fail_closed"` default | `"allow"`) — under a `max_cost` cap, `"fail_closed"` denies further work once a call's cost can't be known (an unpriced model, or a provider-hosted server tool whose charge isn't in the token counts); `"allow"` proceeds (the cap may undercount).

Enforcement is **hard, at the charge site**: the meter *denies the operation that would exceed a cap before it runs* — the call never happens and nothing is charged. Count and token caps are exact even under concurrency. The denial (`BudgetExceeded`, a neutral `AdmissionDenied`) is terminal; the owning (outermost) flow converts it to `policy_decision="budget_exceeded"` in the trace, so `flow.run()` returns a normal `FlowResult` rather than raising. Wall-time is checked between steps.

The **meter is the single source of truth** for what a run consumed — read it off the result, never by summing anything yourself:

```python
report = result.meter            # BudgetReport | None (None only if unmetered)
report.cost                      # known USD spend
report.cost_uncertain            # True if some call couldn't be priced (cost undercounts)
report.over_budget, report.breached   # which caps were reached
result.total_cost, result.usage  # convenience: meter cost / token usage
```

The same `budget_policy=` works per run — `flow.run_sync(state, budget_policy=...)` overrides the construction-time one (both are ignored when the flow runs nested under an enclosing scope). `Agent.run(task, budget_policy=...)` behaves the same. Outside a Flow, wrap any LLM/tool calls in `budget_scope(BudgetPolicy(...))` (a context manager) and read `scope.snapshot()`.

---

## Step-level policy callbacks

Separate from run-wide budgets, a `Policy` on a `Step` or `Flow` decides what happens at each step's boundaries. The declarative callbacks:

| Field | Options | When |
|-------|---------|------|
| `on_timeout` | `"halt"` \| `"fallback"` | step exceeds `timeout` |
| `on_low_confidence` | `"retry"` \| `"escalate"` \| `"fallback"` | step result below `confidence_threshold` |
| `on_exhausted` | `"halt"` \| `"continue"` \| `"fallback"` | retries used up |

```python
from ai_arch_toolkit import Policy, Step

policy = Policy(
    timeout=10.0,
    on_timeout="fallback",
    confidence_threshold=0.7,
    on_low_confidence="retry",
    fallback=fallback_step,
)
step = Step(name="risky", fn=do_work, policy=policy)
```

See [Flow Architecture](flow-architecture.md#policy) for the full `Policy` model and how decisions feed the trace.

---

See also: [Tools](tools.md) for defining tools and the pre-built catalog · [Flow Architecture](flow-architecture.md) for budgets and policy inside agent flows.
