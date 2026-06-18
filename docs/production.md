# Production Guide

This guide is a practical runbook for deploying ai-arch-toolkit in production
services, batch jobs, internal tools, or agent workflows.

The project is pre-1.0, so production use should pin versions or commits,
prefer stable import paths, and treat dangerous capabilities as explicit
opt-ins.

## Installation

Install only the extras needed by the deployment:

```bash
uv add "git+https://github.com/rgesteves5/ai-arch-toolkit.git#egg=ai-arch-toolkit[openai]"
```

Common extras:

- `openai`, `anthropic`, `gemini`, `xai`: provider SDKs.
- `graph`: graph and memory backend support through NetworkX.
- `tokens`: local token counting.
- `yaml`: YAML knowledge loaders.
- `all`: every provider and optional feature.

For reproducible deploys, pin a commit SHA:

```bash
uv add "git+https://github.com/rgesteves5/ai-arch-toolkit.git@<commit>#egg=ai-arch-toolkit[openai,graph]"
```

## Configuration

Configure provider credentials through environment variables:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `XAI_API_KEY`

Use `.env.example` for local development only. In production, load secrets from
the platform secret manager and avoid printing the effective environment.

Recommended process defaults:

- Set explicit model names instead of relying on app-level defaults.
- Set per-request timeouts and retry limits.
- Configure run-level budgets for agent workflows.
- Keep tracing redacted unless a controlled break-glass debug path is active.
- Store graph and memory persistence files on durable storage.

## Provider Keys

Provider SDKs are optional dependencies. A minimal install should not import
provider SDKs eagerly. Install the matching extra for each deployed provider.

Example:

```python
from ai_arch_toolkit import LLM

llm = LLM("gpt-4.1-nano")
response = await llm.complete("Health check: reply ok")
```

Do not validate production keys by sending user data. Use small health-check
prompts or provider-native account checks.

## Stable Imports

Use stable public imports in production code:

```python
from ai_arch_toolkit import LLM, State, ToolGroup, tool
from ai_arch_toolkit.core import BudgetPolicy, RedactionPolicy
from ai_arch_toolkit.toolkit.agents import react_flow, react_initial_state
```

See [API Stability](api-stability.md) before importing experimental, internal,
or dangerous modules.

## Safe And Dangerous Tools

Safe tools should be pure or low-impact: text transforms, math helpers,
formatting, date/time helpers, or read-only lookups with bounded inputs.

Dangerous tools can execute code, run shell commands, read local files, access
the network, mutate memory, or persist data. Examples include:

- `run_command`
- `python_repl`
- filesystem tools
- arbitrary web/API tools
- destructive memory or graph operations

Do not expose dangerous tools to model-controlled flows without runtime
approval, tight allowlists, budgets, and redacted traces.

## Tool Permissions

Prefer explicit tool groups per workflow:

```python
tools = ToolGroup(read_only_lookup, calculate)
```

Avoid "all tools" profiles in production unless the caller is trusted and every
tool has clear governance. Keep tool inputs narrow and validate high-impact
arguments outside the model.

For dangerous tools, declare metadata:

```python
from ai_arch_toolkit.core import tool


@tool(
    capability="shell",
    risk_level="critical",
    requires_approval=True,
    approval_reason="Shell commands can mutate host state.",
)
def run_safe_command(command: str) -> str:
    ...
```

## Human Approval

Approval happens in the runtime/tool layer, not in the model prompt. If a tool
requires approval and no handler is configured, execution is denied by default.

```python
from ai_arch_toolkit.core import ApprovalDecision, ApprovalRequest, ToolGroup


def approve(request: ApprovalRequest) -> ApprovalDecision:
    if request.tool_name == "run_safe_command" and request.arguments["command"] == "uptime":
        return ApprovalDecision.approve(reviewer="ops")
    return ApprovalDecision.deny(reviewer="ops", reason="Command not allowlisted")


tools = ToolGroup(run_safe_command, approval_handler=approve)
```

Store approval decisions as audit data, but rely on trace redaction before
serializing or shipping traces to logs.

## Runtime Budgets

Use cumulative budgets for agent flows. Per-step cost limits are not enough for
cyclic or search-style agents.

```python
from ai_arch_toolkit.core import BudgetPolicy, State, ToolGroup
from ai_arch_toolkit.toolkit.agents import react_flow, react_initial_state

flow = react_flow(
    llm,
    ToolGroup(search),
    max_iterations=6,
    budget_policy=BudgetPolicy(
        max_wall_time=30,
        max_llm_calls=8,
        max_tool_calls=4,
        max_total_tokens=20_000,
        max_cost=0.25,
    ),
)
state = State(operational=react_initial_state("Summarize the incident"))
result = await flow.run(state)
```

Budget-exceeded exits are distinct from normal model/tool output and are
recorded in trace metadata.

## Rate Limits And Timeouts

Set timeouts at the flow or policy layer for every production agent. Use retry
policies for transient provider errors, but keep retry count and backoff bounded.

Operational defaults:

- Short timeout for single LLM calls.
- Longer but bounded timeout for multi-step flows.
- Provider-specific rate limiting at the service boundary.
- Separate budgets per request, user, tenant, or job.

## Tracing, Logging, And Redaction

Trace serialization is redacted by default. Use `full_debug` only for controlled
debugging, and avoid sending full-debug traces to shared logs.

```python
trace_payload = result.trace.to_dict()  # redacted by default
metadata_only = result.trace.to_dict(trace_mode="metadata_only")
```

Redaction covers common API keys, bearer tokens, private keys, `.env`-style
snippets, passwords, connection strings, provider errors, and tool payloads.

## Memory And Graph Persistence

Graph and memory persistence use versioned JSON payloads and atomic writes.
Production guidance:

- Store persistence files on durable storage.
- Back up before migrations or schema changes.
- Treat missing `schema_version` as legacy v0.
- Reject unknown future schema versions until a migration is available.
- Validate restore jobs before replacing live state.

For critical deployments, keep regular snapshots and test restore workflows.

## CI Checklist

Before shipping:

- `uv run ruff check src tests examples`
- `uv run ruff format --check src tests examples`
- `uv run pyright src`
- `uv run pytest`
- `uv build`
- Wheel smoke install outside editable/dev mode
- Optional-extra smoke checks for every provider/feature used in production

The CI workflow includes a package smoke job that builds artifacts and tests
minimal plus optional-extra installs.

## Deployment Checklist

Go/no-go checklist:

- Provider keys are loaded from the secret manager.
- The package version or commit is pinned.
- Only required extras are installed.
- Stable imports are used in application code.
- Dangerous tools are absent or approval-gated.
- Approval handlers deny by default and audit decisions.
- Budgets are configured for every agent flow.
- Timeouts and retries are bounded.
- Redacted trace serialization is used by default.
- Persistence files are on durable storage and backed up.
- Health checks use non-sensitive prompts.
- Rollback version and data restore path are known.

## Rollback And Backup

Keep rollback simple:

- Pin deploys to a package version or commit.
- Keep the previous deploy artifact available.
- Snapshot graph and memory files before migrations.
- Restore persistence from backup before restarting older code if schema
  compatibility is uncertain.
- Retain redacted traces for failed deploy analysis.

## Known Risks

Current production risks to account for:

- The project is pre-1.0 and some APIs are experimental.
- Provider behavior and model tool-calling formats can change.
- Dangerous tools require runtime governance, not only prompting.
- Full-debug traces can expose secrets if misused.
- Search-style agents can consume budget quickly without cumulative limits.
- Optional provider SDKs should be installed and smoke-tested explicitly.

If these risks are unacceptable for a workload, keep the deployment read-only,
disable dangerous tools, use metadata-only tracing, and run with conservative
budgets until the relevant area is hardened further.
