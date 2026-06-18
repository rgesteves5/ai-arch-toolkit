# API Stability

ai-arch-toolkit is pre-1.0. Public APIs are intended to be usable, but the
project still reserves room to improve names, safety defaults, and runtime
contracts before a stable 1.0 release.

This guide defines which import paths are stable public API, which are
experimental, which are internal, and which expose dangerous capabilities.

## Stability Levels

### Stable

Stable APIs are the recommended import paths for application code. They should
avoid breaking changes unless there is a clear security, correctness, or
pre-1.0 design reason.

Recommended stable imports:

```python
from ai_arch_toolkit import LLM, Response, ToolGroup, tool
from ai_arch_toolkit.core import LLM, State, Step, Result, ToolGroup, tool
from ai_arch_toolkit.toolkit.agents import react_flow, react_initial_state
```

Use these paths for:

- LLM calls, streaming, tool calls, structured output, and provider routing.
- Core content/message helpers.
- Core state, step, policy, trace, retry, pricing, budget, redaction, and
  tool-result types.
- Public agent flow factories and their `*_initial_state()` helpers.
- Public graph types exposed from `ai_arch_toolkit.core.graph`.

### Experimental

Experimental APIs are useful and tested, but their exact shape can still change
without a long deprecation window while the project is pre-1.0.

Treat these as experimental:

- `ai_arch_toolkit.toolkit.flow`
- `ai_arch_toolkit.toolkit.memory`
- `ai_arch_toolkit.toolkit.knowledge`
- `ai_arch_toolkit.toolkit.moderation`
- Provider-specific modules under `ai_arch_toolkit.core._providers`
- Nano projects under `ai_arch_toolkit.nanope`

Prefer stable top-level or `core` imports when they exist. Import experimental
modules directly only when you are comfortable tracking changes.

### Internal

Internal APIs are implementation details. They are not compatibility promises
and can change or disappear without deprecation.

Internal paths include:

- Any module or object whose name starts with `_`.
- Provider implementation modules such as
  `ai_arch_toolkit.core._providers._openai`.
- Test helpers, scripts, examples, generated docs, and files under
  `docs/internal/`.

Internal imports are acceptable inside the project itself, but downstream users
should avoid them unless there is no public equivalent yet.

### Dangerous

Dangerous APIs can execute commands, run code, read local data, make network
requests, or otherwise perform actions outside pure in-memory reasoning.

Dangerous surfaces include:

- `ai_arch_toolkit.toolkit.tools.run_command`
- `ai_arch_toolkit.toolkit.tools.python_repl`
- Filesystem tools such as `read_file`, `list_directory`, and `search_files`
- Web/network tools such as page fetching and external API lookup helpers
- Memory or graph operations that delete, overwrite, or persist user data

When dangerous tools are exposed to model-controlled agents, use runtime
approval, explicit allowlists, budgets, timeouts, and redacted tracing. Do not
depend on the model to decide whether a dangerous action is safe.

## Recommended Imports

Prefer public package or layer exports:

```python
from ai_arch_toolkit import LLM, ToolGroup, tool
from ai_arch_toolkit.core import BudgetPolicy, Redactor, ToolResult
from ai_arch_toolkit.toolkit.agents import react_flow
```

Avoid importing from private modules in application code:

```python
# Avoid in downstream application code
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._tools._executor import execute_tool
```

If a needed capability is only available through an internal path, treat that
import as temporary and pin the project version or commit.

## Breaking Changes Before 1.0

Before 1.0, the project may make breaking changes when they improve safety,
correctness, or API clarity. The intended policy is:

- Stable APIs should receive migration notes when practical.
- Experimental APIs may change with shorter notice.
- Internal APIs may change without notice.
- Dangerous APIs may become stricter by default for safety reasons.

The most likely pre-1.0 changes are safety defaults, provider-specific
normalization, tool governance, tracing, memory persistence, and agent-flow
runtime policy.

## Documentation Rule

New public APIs should be exported from a stable module and documented as
stable, experimental, internal, or dangerous. If a feature is not documented in
one of those categories, assume it is experimental or internal until clarified.
