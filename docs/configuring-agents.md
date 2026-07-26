# Configuring Agents — the end-to-end guide

This is the canonical path for wiring agents with ai-arch-toolkit, from a
five-line script to a team-owned, audited manifest. It is written to be handed
directly to a new team member **or to a coding agent** working on a downstream
project: every example is complete, uses current model ids, and states where
each piece of configuration is expected to live.

**The one rule everything follows from:** configuration splits into two kinds,
and they never mix.

| Kind | Examples | Lives in |
|---|---|---|
| **Serializable config** | strategy name, prompts, knobs, iteration caps, timeouts | `ReasoningSpec` / a `*.agent.yaml` manifest |
| **Runtime objects** | `LLM` instances, `ToolGroup`s, evaluator callables | `Agent(...)` arguments and `deps` |

Three layers build on that rule. Use the shallowest layer that fits:

```
Layer 1  code-first      Agent(ReasoningSpec(...), llm, tools)      ← default
Layer 2  declarative     *.agent.yaml manifest → agent_from_manifest
Layer 3  escape hatches  flow factories · Agent.from_flow · register_strategy
```

---

## Layer 1 — code-first: `Agent` + `ReasoningSpec`

A `ReasoningSpec` *describes* how the agent reasons; an `Agent` *binds* it to
runtime objects and compiles once to a `Flow` (reusable across many tasks):

```python
from ai_arch_toolkit import LLM, ToolGroup, tool
from ai_arch_toolkit.toolkit.agents import Agent, ReasoningSpec

@tool
def search(query: str) -> str:
    """Search for information on a topic.

    Args:
        query: The search query.
    """
    return f"Result for '{query}'"

llm = LLM("claude-sonnet-5")
tools = ToolGroup(search)

spec = ReasoningSpec(
    strategy="react",                 # see the strategy table below
    system="You are a concise research assistant.",
    max_iterations=6,
    knobs={"parallel_tool_calls": True},
    timeout=60.0,
)
agent = Agent(spec, llm, tools)

result = agent.run_sync("Who wrote Dune?")
print(result.text)                    # final answer
print(result.cost, result.usage)      # meter-derived spend (single source of truth)
```

`agent.run(...)` is the async form; `agent.iter(...)` streams `FlowEvent`s;
`agent.as_step()` embeds the agent inside a larger `Flow`.

### Strategies

`strategy` names a builder in the registry. Built-ins:

| `strategy` | Shape | `output_schema`? |
|---|---|---|
| `react` *(default)* | tool loop: think → call tool → observe | ✅ |
| `completion` | single LLM call, no tools | ✅ |
| `plan_execute` | plan → per-step ReAct → solve | — |
| `rewoo` | plan with evidence slots → execute → solve | — |
| `reflexion` | attempt → evaluate → reflect retry loop | — |
| `generate_review` | generate → review → retry loop | — |
| `self_discovery` | select/adapt/plan reasoning modules → solve | — |
| `llm_compiler` | plan a DAG → parallel execute → join | — |
| `tot` | tree-of-thoughts search | — |
| `lats` | MCTS with ReAct rollouts | — |

### Knobs vs deps — the boundary that keeps configs portable

- **`knobs`** — serializable, strategy-specific options (`threshold` for
  `reflexion`, `n_candidates` for `tot`, per-phase prompts — below). Validated
  at build time: an unknown knob or invalid value raises `ValueError` before a
  single token is spent.
- **`deps`** — runtime objects that cannot live in a config file. Also
  validated per strategy: a typo like `evalutor` fails loudly instead of being
  silently ignored.

```python
spec = ReasoningSpec(strategy="reflexion", knobs={"threshold": 0.8, "max_retries": 2})
agent = Agent(spec, llm, tools, deps={"evaluator": my_scorer})   # callable dep
```

---

## Per-phase configuration (models, tools, and prompts per role)

Multi-phase strategies accept per-phase overrides through the same two
buckets: **runtime objects** as deps (`<phase>_llm`, `<phase>_tools`) and
**prompts** as knobs (`<phase>_system`). Anything not overridden falls back to
the agent's default `llm`/`tools` and the strategy's built-in prompts.

```python
haiku = LLM("claude-haiku-4-5")       # cheap planner, capable solver

spec = ReasoningSpec(
    strategy="plan_execute",
    knobs={
        "planner_system": (
            "Plan in at most three numbered steps (1. ...). "
            "Output only the plan — do not call tools yourself.\n"
            "The executor has these tools:\n{tools}"
        ),
        "max_replans": 1,
    },
)
agent = Agent(spec, llm, tools, deps={"planner_llm": haiku})
```

**The `{tools}` token** is the *only* substitution the framework performs on a
prompt: at build time it is replaced with the phase's resolved tool catalog
(`- name: description` lines, `(none)` when empty). A prompt without the token
is never modified — no silent appends. The default planner prompts of
`plan_execute`, `rewoo`, and `llm_compiler` carry the token.

Full phase map:

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

The spec-level `llm_kwargs` reach **every** phase's LLM calls; a phase override
changes *who* is called, not what is passed. `react` and `completion` have no
phases and reject any `deps`.

---

## Prompts: where they come from

Three levels, in order of sophistication:

1. **Inline strings** — `system=` and the `<phase>_system` knobs, as above.
2. **The prompts system** (`toolkit.prompts`) — file-backed, versioned
   templates with variables, layouts, and fingerprints. The contract is:
   **the application renders; the spec receives finished text.** The framework
   never renders templates inside an agent.

   ```python
   from ai_arch_toolkit.toolkit.prompts import load_prompt

   template = load_prompt("prompts/support.prompt.yaml")
   rendered = template.render(product="Acme", tone="formal")
   spec = ReasoningSpec(strategy="react", system=rendered.text)
   ```

3. **`system_file` in a manifest** (next section) — verbatim text from a file,
   pinned by the manifest fingerprint. Deliberately *without* variables:
   pointing it at a `.prompt.*` manifest is rejected at load. For templates,
   render app-side and pass the result.

---

## Budgets and structured output

A budget is an execution decision, **not** part of the spec — attach it per
run. Structured output lives on the spec, but only `react` and `completion`
support it (other strategies reject it at build time):

```python
from ai_arch_toolkit import BudgetPolicy
from pydantic import BaseModel

class Answer(BaseModel):
    author: str
    year: int

spec = ReasoningSpec(strategy="react", output_schema=Answer)
result = agent.run_sync(
    "Who wrote Dune and when?",
    budget_policy=BudgetPolicy(max_llm_calls=8, max_cost=0.25),
)
if result.report and result.report.over_budget:
    print("halted on:", result.report.breached)
```

Caps are enforced hard at the charge site — the call that would exceed a cap
never runs. Nested agents share one cumulative budget.

---

## Layer 2 — declarative: `*.agent.yaml` manifests

When configuration is owned by a team (reviewed, versioned, audited), move the
serializable half into a manifest. The loader supports `.agent.yaml/.yml/.json/
.toml`, multi-parent inheritance, embedded profiles, governed dotted-path
overrides, and a deterministic fingerprint that covers the resolved config
**and** the content of referenced prompt files.

```yaml
# agents/support.agent.yaml
version: 1
id: support.answer
extends: ../profiles/base.agent.yaml

strategy:
  name: plan_execute
  max_iterations: 6
  knobs: { max_replans: 1 }
  phases:                                   # per-phase prompts and models
    planner:
      system_file: ../prompts/planner.md    # verbatim text, fingerprint-pinned
      model: { provider: anthropic, model: claude-haiku-4-5 }
    solver:
      system: Answer tersely from the step results.

limits:                                     # becomes a BudgetPolicy
  max_llm_calls: 10
  max_cost: 0.50

override_policy:                            # governs runtime overrides
  allow: [strategy.phases.planner.model.model, limits.max_cost]
  deny: [strategy.knobs]
```

Assembling an agent from it is one call. The application supplies only the
runtime half — the default `LLM` and an `llm_factory` that turns per-phase
model configs into `LLM` instances (the loader never constructs clients or
touches API keys):

```python
from ai_arch_toolkit import LLM
from ai_arch_toolkit.toolkit.agents import agent_from_manifest, load_agent_manifest

manifest = load_agent_manifest(
    "agents/support.agent.yaml",
    profile="production",                                # optional
    overrides={"limits.max_cost": 1.0},                  # must pass override_policy
    allowed_roots=(project_root,),
)
print(manifest.fingerprint)                              # audit primitive

agent = agent_from_manifest(
    manifest,
    LLM("claude-sonnet-5"),                              # default for unbound phases
    tools,
    llm_factory=lambda phase, cfg: LLM(cfg["model"]),    # app-owned model resolution
)
result = agent.run_sync(task, budget_policy=manifest.budget_policy())
```

What the manifest layer guarantees:

- **Phase prompts fold automatically** — `strategy.phases.*.system` (or the
  content of `system_file`) becomes the canonical `<phase>_system` knobs via
  `manifest.reasoning_spec()`. Declaring the same prompt in both `knobs` and
  `phases` is a load error: one declaration site per value.
- **Drift fails closed** — a `system_file` edited after the manifest was
  loaded raises at spec-build time instead of running unaudited content.
- **Models are data** — `manifest.phase_models()` returns the validated
  per-phase model configs; `agent_from_manifest` binds each factory result as
  the `<phase>_llm` dep (an explicit `deps` entry wins; a declared model with
  neither is an error).
- **Secrets are rejected** — `api_key`-like fields anywhere in a manifest fail
  the load.
- **Static validation for CI** — registry-aware checks (strategy name, phase
  names, knob values) without executing anything:

  ```bash
  ai-arch agent validate agents/support.agent.yaml --allowed-root .
  ai-arch agent inspect  agents/support.agent.yaml --allowed-root .   # resolved config + fingerprint
  ```

---

## Layer 3 — escape hatches

**Flow factories** — every strategy's factory is public and exposes the full
per-phase surface as kwargs when you want to wire a `Flow` by hand:

```python
from ai_arch_toolkit.toolkit.agents import plan_execute_flow

flow = plan_execute_flow(llm, tools, planner_llm=haiku, max_replans=0)
```

**`Agent.from_flow`** — wrap any hand-built composition of `Step`s with the
same `run`/`iter`/`AgentResult` surface:

```python
agent = Agent.from_flow(my_flow, init_state=lambda task: {"messages": [user(task)]})
```

**Custom strategies** — register a builder under a stable name and it becomes
usable from specs *and* manifests. Declare `allowed_knobs`/`allowed_deps` so
typos fail at build time, like the built-ins:

```python
from ai_arch_toolkit.toolkit.agents import BuildContext, FlowStrategy, register_strategy

def build(ctx: BuildContext):
    return my_flow(ctx.llm, ctx.tools, depth=ctx.spec.knobs.get("depth", 2))

register_strategy("my_strategy", FlowStrategy(
    build, my_initial_state,
    allowed_knobs=frozenset({"depth"}),
    allowed_deps=frozenset(),
))
```

---

## Migrating an existing project to this shape

If your project already uses the framework but predates this configuration
model, these are the moves — each is independent:

1. **Factory calls → `Agent(ReasoningSpec(...))`.** Direct `*_flow()` calls
   keep working, but the spec adds knob/dep validation and manifest
   compatibility for free. Map factory kwargs to knobs (`planner_system`, …)
   and per-phase objects to canonical deps (`planner_llm`, `executor_tools`).
2. **Hand-injected tool lists in prompts → the `{tools}` token.** Place the
   token where you want the catalog; delete any code that concatenates tool
   descriptions into prompt strings.
3. **`review_llm`/`review_tools` deps → `reviewer_llm`/`reviewer_tools`.**
   The legacy keys still work as aliases; passing both is an error.
4. **Hardcoded per-phase models → manifest `strategy.phases.*.model` +
   `llm_factory`.** Model choice becomes reviewable data; the app keeps key
   handling.
5. **Prompt files read ad hoc → `system_file` (static) or `toolkit.prompts`
   (templated).** Either way the effective prompt is exactly what was
   declared — the framework performs no substitution beyond `{tools}`.
6. **Add `ai-arch agent validate` to CI** for every manifest, so a typo'd
   phase or knob fails the build, not the runtime.
7. **Budgets at call time.** Replace any bespoke token counting with
   `BudgetPolicy` per run (or `limits` in the manifest); read spend from
   `result.report` — never sum costs yourself.

---

## Quick reference — where each thing lives

| Concern | Home |
|---|---|
| Strategy, prompts, knobs, iteration caps, timeout | `ReasoningSpec` / manifest (serializable) |
| LLMs, ToolGroups, evaluator callables | `Agent(...)` args + `deps` (runtime) |
| Per-phase models, declaratively | `strategy.phases.*.model` → app `llm_factory` |
| Budget caps | per run (`run(budget_policy=…)`) or manifest `limits` |
| Template rendering | the application (`toolkit.prompts`) — never the framework |
| Structured output | `spec.output_schema` (`react`/`completion` only) |
| Dangerous tools | `toolkit.tools.dangerous` + approval gates on `ToolGroup` |
| Audit | `manifest.fingerprint` + `ai-arch agent validate` in CI |

Deeper reference: [Agent & ReasoningSpec](agents.md) (field-by-field),
[Prompts](prompts.md), [Tool Governance & Safety](safety.md), and runnable
examples `examples/09`–`27` and `examples/47_per_phase_agents.py`.
