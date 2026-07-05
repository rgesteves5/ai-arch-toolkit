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
| `type` | error code — one of a closed set (see below), so retry/branch logic can switch on it |
| `message` | human-readable message (already redacted) |
| `retryable` | whether retrying might succeed |
| `safe_to_show` | if `False`, `to_model_text()` hides the message from the model |
| `details` | structured extra context |

The `type` is drawn from a fixed set:

- **Governance blocks** — `"dangerous_tool_blocked"`, `"approval_denied"`, `"max_calls_exceeded"`, `"budget_exceeded"`.
- **Resolution / execution** — `"unknown_tool"` (no matching function), `"validation_error"` (argument mismatch — a `TypeError` raised by the call), `"runtime_error"` (any other exception, `retryable=True`).

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

Beyond gates, `ToolGroup(max_calls=N)` caps how many tools the group runs in one pass — the call past the cap is blocked with `max_calls_exceeded`. The counter is enforced atomically at the pipeline's commit step (not as a gate), so it holds exactly even under concurrent execution; call `group.reset()` to reuse the group for a fresh run.

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

Returning `modified_args` from `approve(...)` runs the tool with **substituted arguments** — useful for narrowing a request (e.g. forcing a safe target) before letting it through. The full request/decision is recorded under `result.metadata["audit"]["approval"]`.

The handler may be sync or async — with one caveat: on the **synchronous** execution path (`group.execute()`, `execute_tool()`) an async handler cannot be awaited, so it is **auto-denied** with reason `"Synchronous execution cannot await approval handler"`. Use the async path (`async_execute()` / `async_execute_tool()`) whenever your handler is a coroutine.

### Dry run

`DryRunGate(dry_run=True)` short-circuits execution and reports what *would* have run, without side effects — useful for previews and tests.

```python
from ai_arch_toolkit import DryRunGate
group = ToolGroup(run_command, gates=[DryRunGate(dry_run=True)])
```

A dry-run result is `ok=True` with `value="[dry-run] would call <tool>"`, carries `metadata["governance"] == {"outcome": "dry_run", "executed": False}`, and records the arguments that *would* have run under `metadata["audit"]["arguments"]`.

### Executing a single tool call

When you're not using a `ToolGroup`, run one call against a plain list of functions:

```python
from ai_arch_toolkit import execute_tool, async_execute_tool

result = execute_tool(tool_call, [get_weather, delete_table], approval_handler=approve_handler)
result = await async_execute_tool(tool_call, [get_weather], approval_handler=None)
```

---

## Trace redaction

Traces and tool results can carry secrets. The redactor walks a payload recursively — through dicts, lists, tuples, and dataclasses — masking them by **key name** and by **value pattern**:

- **Sensitive key fragments** (case-insensitive, `-`/`_` normalized — any key *containing* one is masked wholesale): `api_key`/`apikey`, `authorization`, `bearer`, `client_secret`, `connection_string`, `database_url`, `password`, `private_key`, `secret`, `token`.
- **Value patterns**: PEM private-key blocks, `Bearer <token>`, `sk-…` keys, database URLs (`postgres`/`postgresql`/`mysql`/`mongodb`/`redis://…`), uppercase env-style assignments (`…API_KEY=`, `…TOKEN=`, `…SECRET=`, `…PASSWORD=`, `…PRIVATE_KEY=`), and inline `key: value` / `key=value` pairs for api-key/token/secret/password/private-key.

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

`trace_mode` accepts the string literals above or the equivalent `RedactionMode` enum (`RedactionMode.REDACTED`, `.METADATA_ONLY`, `.FULL_DEBUG`). `Redactor(policy)` is the reusable object behind `redact()`; `redact()` / `redact_text()` are the one-shot helpers (a `None` policy uses the safe default).

Type handling to know about: `bytes` values are replaced wholesale; dataclasses are converted (`asdict`) then redacted; nested containers are walked element-by-element. One caveat — plain non-string scalars (`int`, `float`, `bool`, `None`) pass through **unredacted**, so a numeric secret is only caught when it sits under a sensitive *key* (e.g. `{"token": 12345}` is masked; a bare `12345` is not).

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

Enforcement happens **at the charge site**: the meter denies the operation that would breach a cap, the call never happens, and nothing is charged. The denial (`BudgetExceeded`, a neutral `AdmissionDenied`) is terminal; the owning (outermost) flow converts it to `policy_decision="budget_exceeded"` in the trace, so `flow.run()` returns a normal `FlowResult` rather than raising.

How *tight* the cap is depends on the dimension:

- **Call caps are hard** — `max_llm_calls` / `max_tool_calls` are checked against committed + outstanding *counts* under the meter's lock, so they are exact even under concurrent (parallel-DAG) execution: a run can never overshoot them.
- **Token and cost caps under `reserve="none"` (the default) are soft** — a call is admitted while its token usage / cost is still unknown and only denied *after* it settles, so the total can overshoot `max_input_tokens` / `max_output_tokens` / `max_total_tokens` / `max_cost` by at most the single in-flight call. Use `reserve="strict"` to reserve a worst-case token/cost hold up front and make them hard (it fails closed on unpriced models). An unbounded (unknown) cost fails closed regardless — see `unpriced` above.
- **Wall-time is checked between steps**, so a single long-running step is not interrupted mid-flight (use `Policy(timeout=...)` for that).

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
