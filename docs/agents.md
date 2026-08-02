# Agent & ReasoningSpec

`Agent` and `ReasoningSpec` are the high-level way to run an agent. A
`ReasoningSpec` *describes* how an agent reasons (a named strategy plus limits);
an `Agent` *binds* that spec to a runtime `LLM` and `ToolGroup` and runs it. The
spec is declarative and serializable; the LLM and tools are not.

For the guided end-to-end path — code-first specs, per-phase configuration,
prompts, budgets, and manifests — start with
[Configuring Agents](configuring-agents.md); this page is the reference.

This is the recommended entry point. The underlying `*_flow()` factories
([Flow Architecture](flow-architecture.md)) remain available when you want to
build and wire a `Flow` by hand — `Agent` compiles down to exactly one of them.

```
ReasoningSpec        →  build_flow()  →   Flow   →   Agent.run()  →  AgentResult
(strategy + limits)      (compile)        (run)
```

## Quick start

```python
from ai_arch_toolkit import LLM, ToolGroup
from ai_arch_toolkit.toolkit.agents import Agent, ReasoningSpec
from ai_arch_toolkit.toolkit.tools import wikipedia_search, datetime_now

llm = LLM("claude-sonnet-5")
tools = ToolGroup(wikipedia_search, datetime_now)

spec = ReasoningSpec(strategy="react", system="You are a concise research assistant.")
agent = Agent(spec, llm, tools)

result = agent.run_sync("Who wrote Dune, and what year was it published?")
print(result.text)
print(f"cost: ${result.cost:.4f}  tokens: {result.usage.total_tokens}")
```

`Agent` compiles the `Flow` **once** in its constructor, so a single agent can be
run on many tasks without recompiling.

## ReasoningSpec

A frozen, keyword-only dataclass. Every field has a default — `ReasoningSpec()`
is a valid ReAct spec.

| Field | Type | Default | Purpose |
|---|---|---|---|
| `strategy` | `str` | `"react"` | Which registered strategy to compile (see table below) |
| `system` | `str` | `""` | System prompt |
| `max_iterations` | `int` | `10` | Reasoning-loop cap (meaning is per-strategy) |
| `knobs` | `Mapping[str, Any]` | `{}` | Strategy-specific, **serializable** options |
| `policy` | `Policy \| None` | `None` | Per-step retry/timeout/confidence ([Flow Architecture](flow-architecture.md)) |
| `timeout` | `float \| None` | `None` | Wall-clock timeout (seconds) |
| `llm_kwargs` | `Mapping[str, Any]` | `{}` | Extra kwargs forwarded to the LLM (e.g. `temperature`) |
| `output_schema` | `OutputSchema \| type \| None` | `None` | Structured output; accepts a schema or supported model class (see table) |

`knobs` vs the dedicated fields: a field is dedicated when every strategy uses it
the same way (`system`, `timeout`, `policy`). Anything strategy-specific —
`n_candidates` for `tot`, `threshold` for `reflexion`, `max_cycles` for
`generate_review` — lives in `knobs`, so a spec stays a flat, config-friendly
object.

> **Budgets are not on the spec.** A `ReasoningSpec` carries no `budget_policy`.
> Attach a run-level [budget](#budgets) at call time (`agent.run(task,
> budget_policy=…)`) or on the `Flow`/factory — see below.

## Strategies

`strategy` names a builder in the registry. Built-ins:

| `strategy` | Maps to | `output_schema`? | Key `knobs` |
|---|---|---|---|
| `react` *(default)* | `react_flow` | ✅ | `parallel_tool_calls`, `final_answer_hint`, `strip_tools_on_final`, `show_turn_counter` |
| `completion` | single LLM call (no tools loop) | ✅ | — |
| `plan_execute` | `plan_execute_flow` | — | `max_replans`, `max_iterations_per_step` |
| `rewoo` | `rewoo_flow` | — | — |
| `reflexion` | `reflexion_flow` | — | `threshold`, `max_retries` |
| `generate_review` | `generate_review_flow` | ✅ (generator only) | `max_cycles`, `max_review_iterations`, `reviewer_kwargs` |
| `self_discovery` | `self_discovery_flow` | — | `modules` |
| `llm_compiler` | `llm_compiler_flow` | — | `max_replans` |
| `tot` | `tot_flow` | — | `n_candidates`, `max_depth`, `search_strategy` |
| `lats` | `lats_flow` | — | `n_candidates`, `max_rollouts`, `exploration_weight` |

Multi-phase strategies additionally accept per-phase prompt knobs
(`planner_system`, `evaluator_system`, …) — see
[Per-phase configuration](#per-phase-configuration).

`completion` is the one strategy with no flow factory of its own — a plain
single-shot LLM call, handy as a baseline or for non-agentic steps in a larger
composition. `react`, `completion`, and `generate_review` support `output_schema`;
for `generate_review`, the schema is sent only to the generator, while the reviewer
keeps its plain-text `ACCEPT` / `RETRY` control protocol. Setting it on any other
strategy raises `ValueError` at `build_flow` time, not silently at runtime.

The registered names are discoverable at runtime:

```python
from ai_arch_toolkit.toolkit.agents import strategy_names
strategy_names()   # ('completion', 'generate_review', 'lats', 'llm_compiler', ...)
```

## Runtime dependencies: `deps`

Some strategies need a runtime object that **cannot** live in a config file — an
evaluator callable, a per-phase LLM or ToolGroup. These go in `deps`, separate
from the serializable `knobs`:

```python
spec = ReasoningSpec(strategy="reflexion", knobs={"threshold": 0.8})
agent = Agent(spec, llm, tools, deps={"evaluator": my_scorer})
```

Built-in strategies validate `deps` the same way they validate `knobs`: an
unknown key or a wrongly-typed value raises `ValueError` at build time, so a
typo like `evalutor` cannot be silently ignored. Custom strategies registered
without `allowed_deps` keep the old accept-anything behavior.

Callable deps: `reflexion` → `evaluator`; `lats` → `evaluator_fn`. The
per-phase LLM/tool deps are listed in the next section.

## Per-phase configuration

Multi-phase strategies accept per-phase overrides through the two existing
buckets: **runtime objects** (`<phase>_llm`, `<phase>_tools`) go in `deps`, and
**prompts** (`<phase>_system`) go in `knobs`. Anything not overridden falls back
to the agent's default `llm`/`tools` and the strategy's built-in prompts.

```python
spec = ReasoningSpec(
    strategy="plan_execute",
    knobs={"planner_system": "Plan in at most three numbered steps."},
)
agent = Agent(
    spec, llm, tools,
    deps={"planner_llm": haiku, "executor_tools": narrow_tools},
)
```

| Strategy | Phase | `deps` keys | Prompt knob |
|---|---|---|---|
| `plan_execute` | planner | `planner_llm` | `planner_system` |
| | executor | `executor_llm`, `executor_tools` | — |
| | solver | `solver_llm` | `solver_system` |
| `rewoo` | planner | `planner_llm` | `planner_system` |
| | solver | `solver_llm` | `solver_system` |
| `reflexion` | executor | `executor_llm`, `executor_tools` | — |
| | reflector | `reflector_llm` | `reflector_system` |
| `generate_review` | generator | `generator_llm`, `generator_tools` | *(the spec's `system`/`llm_kwargs`)* |
| | reviewer | `reviewer_llm`, `reviewer_tools` | `reviewer_system`, `reviewer_kwargs` |
| `self_discovery` | reasoning | `reasoning_llm` | `select_system`, `adapt_system`, `plan_system` |
| | solver | `solver_llm`, `solver_tools` | `solver_system` |
| `llm_compiler` | planner | `planner_llm` | `planner_system` |
| | executor | `executor_llm`, `executor_tools` | — |
| | joiner | `joiner_llm` | `joiner_system` |
| `tot` | generator | `generator_llm` | — |
| | evaluator | `evaluator_llm` | `evaluator_system` |
| | solver | `solver_llm` | — |
| `lats` | rollout | `rollout_llm`, `rollout_tools` | — |
| | evaluator | `evaluator_llm` | `evaluator_system` |
| | solver | `solver_llm` | — |
| | reflector | `reflector_llm` | `reflector_system` |

Notes:

- Planner prompts may contain a literal `{tools}` token — the **only**
  substitution the framework performs. At build time it is replaced with the
  phase's resolved tool catalog (`- name: description` lines, `(none)` when
  empty); a prompt without the token is never modified. The default planner
  prompts of `plan_execute`, `rewoo`, and `llm_compiler` carry the token, so
  tool awareness is visible in the declared text instead of appended silently.
  (`rewoo`'s planner needs the tool names to emit `#E1 = ToolName[arg]` steps —
  omit the token there only if your prompt manages tool naming itself.)
- `self_discovery`'s reasoning phase has no single system prompt — its three
  sub-prompts (`select_system`, `adapt_system`, `plan_system`) are plain knobs.
- `generate_review` still accepts the legacy `review_llm` / `review_tools` dep
  keys as aliases of `reviewer_llm` / `reviewer_tools`; passing both is an
  error.
- The spec-level `llm_kwargs` apply to **every** phase's LLM calls; a phase
  override changes who is called, not what is passed. `reviewer_kwargs` merges
  on top of the global `llm_kwargs`, winning per key.
- `react` and `completion` have no phases and reject any `deps`.

## AgentResult

`run`/`run_sync` return an `AgentResult` (frozen dataclass):

| Attribute | Type | Meaning |
|---|---|---|
| `text` | `str` | The final answer, extracted from the flow's state |
| `response` | `Response \| None` | The last full `Response` (usage, citations, tool calls), if any |
| `flow_result` | `FlowResult` | The complete flow trace — step results, timing, metadata |
| `usage` | `Usage` | Cumulative token usage across the run |
| `cost` | `float` | Cumulative USD cost |
| `report` | `BudgetReport \| None` | The run's meter projection — `None` if the run was unmetered ([Budgets](#budgets)) |
| `errors` | `tuple[str, ...]` | Error strings from any failed steps |

The meter is the single source of truth for spend — read totals off the result,
never by summing yourself. `result.cost` / `result.usage` are conveniences over
the same meter that populates `result.report`.

## Streaming

`Agent.iter(task)` yields `FlowEvent`s as the run progresses (async):

```python
async for event in agent.iter("Summarise the latest news on X."):
    print(event)
```

## Budgets

A run-level [`BudgetPolicy`](safety.md#cumulative-budgets) caps LLM calls, tool
calls, tokens, USD cost, and wall-time across the whole run — every step and every
nested sub-agent share one cumulative budget. Because the spec carries no budget,
you attach it at **call time** (it applies to any strategy):

```python
from ai_arch_toolkit import BudgetPolicy

spec = ReasoningSpec(strategy="plan_execute")
result = Agent(spec, llm, tools).run_sync(
    "Research X and summarise.",
    budget_policy=BudgetPolicy(max_llm_calls=10, max_cost=0.25),
)

if result.report and result.report.over_budget:
    print("halted on:", result.report.breached)   # the caps that were reached
```

`BudgetReport` exposes `llm_calls`, `tool_calls`, token tallies, `cost`,
`cost_uncertain` (a call couldn't be priced, so `cost` undercounts), `elapsed_s`,
`over_budget`, and `breached`. Enforcement is **hard at the charge site**: the
call that would exceed a cap never runs. See [Run-Level Budgets in the safety
guide](safety.md#cumulative-budgets) for hard-vs-soft caps, `reserve` /
`unpriced`, and the full model.

## From config: `ReasoningSpec.from_mapping`

Build a spec from a plain mapping (parsed JSON/YAML/dict) — handy for declarative
agent definitions:

```python
spec = ReasoningSpec.from_mapping({
    "strategy": "react",
    "system": "You are helpful.",
    "max_iterations": 6,
    "knobs": {"parallel_tool_calls": False},
})
```

`policy` is passed through only if it is already a `Policy` instance (it is not
coerced from a mapping). `output_schema` accepts a `{"name", "schema", "strict"}`
mapping, an `OutputSchema`, or a supported model class such as a Pydantic model.

## File-backed agent manifests

Use `load_agent_manifest()` when configuration lives in versioned files. The
public loader supports `.agent.yaml`, `.agent.yml`, `.agent.json`, and
`.agent.toml`; YAML requires the `yaml` extra. It strictly rejects unknown
fields and resolves, in order, inherited parents, the child, an optional named
profile, and governed dotted-path overrides.

```yaml
version: 1
id: support.answer
extends: ../profiles/default.agent.yaml

strategy:
  name: react
  max_iterations: 6
  parallel_tool_calls: false

prompts:
  system_manifest: ../prompts/support.prompt.yaml

limits:
  timeout_seconds: 30
  max_llm_calls: 6
  max_cost: 0.25
  reserve: strict

override_policy:
  allow: [model.model, model.temperature, limits.timeout_seconds]
  deny: [strategy, prompts, tools]
```

```python
from ai_arch_toolkit.toolkit.agents import Agent, load_agent_manifest

manifest = load_agent_manifest(
    "agents/support.agent.yaml",
    profile="production",
    overrides={"model.temperature": 0.2},
    allowed_roots=(project_config_root,),
)
spec = manifest.reasoning_spec(system=rendered_system, output_schema=Answer)
result = await Agent(spec, llm, tools).run(
    rendered_user,
    budget_policy=manifest.budget_policy(),
)
```

Relative prompt/tool-manifest paths are resolved against the file that declares
them, including paths inside embedded profiles, and cannot leave `allowed_roots`.
A relative path supplied by a runtime override resolves against the entry
manifest's directory. `manifest.fingerprint` is machine-path independent and
includes the selected profile, fully resolved configuration, and referenced
prompt/tool content; source and referenced fingerprints remain available
separately for provenance. With multiple allowed roots, source provenance keys
use `root[N]:relative/path` so equal relative paths cannot overwrite one another.

Manifest files and override values accept only canonical JSON-like data: null,
strings, booleans, integers, finite floats, arrays, and objects with string keys.
Executable or process-local objects are deliberately rejected; applications
resolve schema, prompt, tool, and adapter ids through their own allowlisted
registries. Secret-like fields are rejected in every source/profile before
selection and again after overrides. Override `deny` paths protect both their
ancestors and descendants, so replacing an allowed parent object cannot bypass a
denied child.

### Per-phase prompts and models: `strategy.phases`

`strategy.phases` declares per-phase prompts and model choices declaratively —
the per-phase counterpart of the established top-level split, where
`strategy.system` flows into the spec and `model` is data the application
resolves:

```yaml
strategy:
  name: plan_execute
  phases:
    planner:
      system_file: ../prompts/planner.md      # or an inline `system:`
      model: { provider: anthropic, model: claude-haiku-4-5, temperature: 0 }
    solver:
      system: Solve tersely from the step results.
```

Prompts fold into the spec automatically: `manifest.reasoning_spec()` reads
`system` — or the content of `system_file`, at bridge time, re-verified against
the load-time hash so a file that changed since loading raises instead of
running unaudited content — into the canonical `<phase>_system` knobs.
`system_file` is **verbatim text**: no variables, no rendering (pointing it at
a `.prompt.*`/`.agent.*` manifest is rejected); to use the prompts system,
render in the application and declare the result. Declared prompt text may
carry the `{tools}` token like any other phase prompt; the fingerprint pins the
declared form, and the token resolves against the runtime tools at build time.
Declaring both `strategy.phases.<name>.system` and
`strategy.knobs.<name>_system` is a load error: one declaration site per value.
`system_file` paths obey `allowed_roots` and join `referenced_fingerprints`, so
editing the prompt file changes the manifest fingerprint without touching the
YAML; dotted override governance covers `strategy.phases.*` like any other
subtree, and secret scanning walks phase model configs.

Phase `model` configs carry the same contract as the top-level `model` section:
validated data, never constructed by the loader. Read them with
`manifest.phase_models()` and resolve them yourself, or let
`agent_from_manifest` do the wiring:

```python
from ai_arch_toolkit.toolkit.agents import agent_from_manifest

agent = agent_from_manifest(
    manifest, llm, tools,
    llm_factory=lambda phase, cfg: LLM(cfg["model"]),   # app-owned resolution
)
```

`llm_factory(phase, model_config)` runs once per declared phase model and binds
the result as the canonical `<phase>_llm` dep. An explicit `deps` entry for a
phase wins over the factory; a declared model with neither is an error — a
manifest's model choice must not be silently ignored.

Validate manifests statically in CI — registry-aware checks (strategy name,
phase names, knob names and values) on top of the loader's shape checks:

```bash
ai-arch agent validate agents/support.agent.yaml --allowed-root .
ai-arch agent inspect agents/support.agent.yaml --allowed-root .
```

`--allowed-root` (repeatable) mirrors the loader's `allowed_roots` — without it
only the manifest's own directory may be referenced, so layouts that keep
profiles or prompt files in sibling directories need it. `inspect` prints the
resolved config and fingerprint.

## Escape hatch: `Agent.from_flow`

When you have built a `Flow` by hand (any composition of `Step`s) and want the
same `run`/`iter`/`AgentResult` surface, wrap it directly — no `ReasoningSpec`
needed:

```python
agent = Agent.from_flow(my_flow, init_state=lambda task: {"messages": [user(task)]})
```

`init_state` builds the per-task operational state: a callable mapping the task
to a dict, a fixed dict (task ignored), or `None` for the default
`{"messages": [user(task)]}`.

## Composition: `Agent.as_step`

`agent.as_step()` returns the agent's `Flow` wrapped as a `Step`, so an agent can
become one node inside a larger `Flow` — multi-agent pipelines without leaving
the Flow model. `agent.flow` exposes the compiled `Flow` directly.

## Custom strategies

A strategy is a named recipe implementing the `StrategyBuilder` protocol
(`build(ctx) -> Flow`, `init_state(task) -> dict`, `supports_output_schema`).
The simplest path is `FlowStrategy`, which adapts a flow factory plus an
initial-state function; register it under a stable name:

```python
from ai_arch_toolkit.toolkit.agents import (
    BuildContext, FlowStrategy, register_strategy,
)

def build_my_flow(ctx: BuildContext):
    s = ctx.spec
    return my_flow(ctx.llm, ctx.tools, system=s.system, depth=s.knobs.get("depth", 2))

register_strategy("my_strategy", FlowStrategy(build_my_flow, my_initial_state))

# Now usable by name:
agent = Agent(ReasoningSpec(strategy="my_strategy", knobs={"depth": 3}), llm, tools)
```

`BuildContext` carries `spec`, `llm`, `tools`, and `deps` — read serializable
config from `spec`/`spec.knobs` and runtime objects from `deps`. `FlowStrategy`
also accepts `allowed_knobs`/`knob_validators` and, symmetrically,
`phases`/`allowed_deps`/`dep_validators`; leave `allowed_deps` as `None` to skip
dep validation, or declare a set so typos fail at build time like the built-ins.

## Lower-level functions

`Agent` is a thin facade over three functions, exposed for when you want the
pieces:

- `build_flow(spec, llm, tools, *, deps=None) -> Flow` — compile a spec to a
  task-independent `Flow`.
- `initial_state(spec, task) -> dict` — build the per-task operational state for
  the spec's strategy.
- `extract_text(state, flow_result) -> str` — pull a single answer string out of
  a finished run.
