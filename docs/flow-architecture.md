# Flow Architecture

The Flow system is a composable orchestration framework built from 6 primitives: **State**, **Step**, **Result**, **Policy**, **Trace**, and **Scope**. A **Flow** composes Steps. A Flow can become a Step. Everything composes recursively.

```
core/              Flat primitives — no opinions about orchestration
  _state.py          State, StateSnapshot, MergeStrategy
  _step.py           Step, StepFn, Result
  _policy.py         Policy
  _trace.py          Trace, StepTrace, PolicyDecision
  _step_engine.py    execute_step() — single-step execution with policy

toolkit/flow/      Opinionated orchestration — built on core/
  _scope.py          Scope, apply_scope()
  _flow.py           Flow, FlowStep, FlowResult, FlowEvent
  _executor.py       execute_flow(), iter_flow()
```

---

## State

State is a mutable container with **4 named layers**. When a Step reads a key, it searches top to bottom:

```
1. current      ← highest priority, ephemeral context
2. operational  ← where Step artifacts land (default write target)
3. persistent   ← survives across runs
4. world        ← shared environment, read-mostly
```

```python
from ai_arch_toolkit.core import State

state = State(
    operational={"task": "Summarize this article"},
    world={"api_key": "sk-..."},
)

state["task"]              # → "Summarize this article" (from operational)
state["api_key"]           # → "sk-..." (from world)
state["task"] = "New task" # writes to operational
state.set("user_id", 42, layer="persistent")
```

### StateSnapshot

Steps never see the mutable State. They receive a **StateSnapshot** — a read-only view of every
layer as it was when the step's wave started:

```python
snapshot = state.snapshot()  # the layers are copied (MappingProxyType per layer)
snapshot["task"]             # reads work
snapshot.operational["task"] = "x"  # TypeError — read-only
```

Steps return Results. The executor merges Result artifacts back into State.

The layers are copied, the values in them are not: a list or dict a step reads is the State's own.
A step must not mutate it in place (append to it, set a key on it); it returns what changes as
`Result.artifacts`. A value mutated in place changes the State for every step after it — siblings
in the same parallel wave included — in every mode. Copying every value for every step would make
long runs quadratic, so the engine does not.

### Merge

Every step of a wave — one step in a sequential flow, every ready step in DAG mode — reads the
snapshot taken when the wave starts, so parallel siblings never see each other's artifacts. When
the wave's last step finishes, the wave's Results merge back into the State in the order the steps
were declared:

```python
state.merge(result_a, result_b, strategy="last_wins")
```

`State.fork()` still gives an independent deep copy (current, operational and persistent copied,
world shared by reference) when you want one — to run a flow on a copy of the state, say. The
engine does not fork per step: deep copies on every step would make long runs quadratic.

| Strategy | Behavior |
|----------|----------|
| `"last_wins"` | Last value wins on conflict (default) |
| `"collect"` | Conflicting values collected into a list |
| `"raise"` | Raises `MergeConflictError` on conflict |

**The contract**: parallel steps that write the same key will conflict. Use `after=` to serialize dependent steps, or write to different keys.

---

## Step and Result

A Step is a named async function with optional policy, scope, and fallback:

```python
from ai_arch_toolkit.core import Step, Result, StateSnapshot

async def summarize(snap: StateSnapshot) -> Result:
    task = snap.require("task")  # helpful KeyError if missing
    # ... do work ...
    return Result(
        value="Summary here",
        artifacts={"summary": "Summary here"},  # merged into State
        cost=0.003,
        confidence=0.95,
    )

step = Step(name="summarize", fn=summarize)
```

### Result

The output of every Step:

```python
Result(
    value=...,          # primary output (any type)
    artifacts={...},    # dict merged into State.operational
    usage=Usage(...),   # token counters
    cost=0.0,           # dollar cost
    confidence=None,    # 0.0–1.0, used by Policy
    error=None,         # None = success, str = failure
    duration=0.0,       # seconds
)
```

`result.is_ok` / `result.is_error` for quick checks.

The `artifacts` dict is the key mechanism: whatever a Step puts in artifacts gets merged into State for the next Step to read.

---

## Policy

Execution constraints attached to a Step. Controls retry, timeout, confidence thresholds, cost limits, and what happens when things fail:

```python
from ai_arch_toolkit.core import Policy
from ai_arch_toolkit.core._retry import RetryConfig

policy = Policy(
    retry=RetryConfig(max_retries=3, base_delay=1.0),
    timeout=30.0,
    confidence_threshold=0.8,
    max_cost=0.50,
    on_exhausted="fallback",      # "halt" | "continue" | "fallback"
    on_low_confidence="retry",    # "retry" | "escalate" | "fallback"
    on_timeout="halt",            # "halt" | "fallback"
)

step = Step(name="critical_step", fn=my_fn, policy=policy)
```

### Decision flow

```
Step runs → success?
  Yes → confidence >= threshold?
    Yes → cost <= max_cost? → done
    No  → retry / escalate / fallback (per on_low_confidence)
  No (error) → retries left?
    Yes → exponential backoff → retry
    No  → halt / continue / fallback (per on_exhausted)

Timeout? → halt / fallback (per on_timeout)
```

All decisions are recorded in the Trace. A step timeout is never retried, even when `retry` is
configured: `on_timeout` chooses between `halt` and `fallback`.

### Policy on a Flow

`Flow(policy=...)` is the default for every step of that flow that has no policy of its own; a
step's own `policy` always wins. A nested flow applies its own policy when it runs, so the step that
wraps it (`as_step()`) carries none and nothing is applied twice.

`Flow(timeout=...)` bounds a whole run, in seconds. When it elapses, the steps in flight are
cancelled, nothing else starts, and the trace ends with a `flow_timeout` step (`iter()` emits a
`timeout` event). A nested flow's timeout is handled by that flow; an outer timeout also cancels a
nested flow that is still running.

```python
flow = Flow(
    Step(name="draft", fn=draft),
    Step(name="review", fn=review),
    policy=Policy(retry=RetryConfig(max_retries=2)),  # each step retries up to twice
    timeout=60.0,                                     # the whole run stops after 60 s
)
```

### Fallback

A fallback is just another Step. It runs when the primary Step exhausts retries or times out:

```python
fallback_step = Step(name="cheap_model", fn=cheap_fn)

step = Step(
    name="expensive_model",
    fn=expensive_fn,
    policy=Policy(
        timeout=10.0,
        on_timeout="fallback",
        fallback=fallback_step,
    ),
)
```

---

## Trace

Every execution produces a Trace — a complete record of what happened:

```python
result = await flow.run(state)
trace = result.trace

# Navigation
trace.step("summarize")        # find by name (recursive)
trace.flow("inner_react")      # find nested flow

# Aggregates
trace.total_duration           # wall clock
trace.confidence               # min across non-skipped steps
trace.total_cost               # raw sum of per-step Result.cost annotations (0 for metered flows)
trace.total_usage              # raw sum of per-step Result.usage annotations

# Per-step detail
for st in trace.steps:
    print(st.name, st.duration)
    print(st.policy_decisions)  # ("retry", "fallback", ...)
    print(st.error)             # None or error string
    print(st.skipped)           # True if condition not met
```

For **spend**, read the run's meter — the single source of truth — not the trace: `result.meter`
(a `BudgetReport`), or the `result.total_cost` / `result.usage` shortcuts. `trace.total_cost` and
`st.cost` only reflect costs a custom step annotated manually via `Result(cost=...)`.

### StepTrace fields

```
name, duration, cost, confidence, usage, attempts
policy_decisions: ("retry", "timeout", "fallback", ...)
error, skipped, skip_reason
children: the steps of flows run inside this step (nested flows, inner agent loops)
input_keys: the keys the step read, per state layer
output_keys: the artifact keys the step returned
input_state, output_result: the values, as trace_capture allows
```

### What a trace captures

`Flow(trace_capture=...)` sets how much of the state and of each result the trace keeps. A long
agent loop carries its whole message history in state, so recording every value at every step
makes a trace grow with the square of the steps.

| `trace_capture` | `input_state` | `output_result` | `input_keys` / `output_keys` |
|---|---|---|---|
| `"keys"` (default) | empty | value, usage, cost, confidence, error, duration; no artifacts | recorded |
| `"full"` | deep copy of every layer, taken before the step runs | everything, artifacts deep-copied | recorded |
| `"none"` | empty | empty | empty |

- **`"full"` is a faithful history.** Values are copied when the step starts and when it ends, so
  a step that mutates a value in place (a queue it pops, a tree it expands) cannot rewrite an
  earlier record. `world` holds shared resources and stays by reference, as in `State.fork()`,
  and so does any value `copy.deepcopy` refuses. Memory and serialization cost grow with state
  size times step count.
- **`initial_state` is kept once per run.** It is copied the same way as `"full"`, except with
  `"none"`, where it is empty.
- **Agents set it on the spec.** Use `ReasoningSpec(trace_capture=...)` or `strategy.trace_capture`
  in a manifest. Strategies that run an inner ReAct loop pass it to that loop.
- **Capture is separate from redaction.** `trace_capture` decides what is recorded during the run;
  `trace.to_dict(trace_mode=...)` decides what a serialized trace shows (see
  [Trace redaction](safety.md#trace-redaction)).

`Trace.from_dict` reads traces saved before `input_keys` and `output_keys` existed.

---

## Scope

Scope controls what a Step can see in the State. It filters each layer independently:

```python
from ai_arch_toolkit.toolkit.flow import Scope

# Only show these keys
scope = Scope(include=frozenset({"task", "context"}))

# Hide these keys
scope = Scope(exclude=frozenset({"api_key", "internal_state"}))

# Transform values before the Step sees them
scope = Scope(transform={"messages": lambda msgs: msgs[-5:]})  # last 5 messages

# Inject computed values into the current layer
scope = Scope(enrich={"word_count": lambda snap: len(snap["text"].split())})
```

Scope resolution: **FlowStep.scope > Step.scope > Flow.scope** (first non-None wins).

Scope preserves layer structure — a Step still knows which layer data came from.

`transform` runs once for each layer that holds the key. `enrich` reads the snapshot the Step will
see — already filtered and transformed — so it can never read a key the scope hides, nor another
enricher's output.

---

## Flow

A Flow composes Steps into an execution graph. Three modes, auto-detected:

### Sequential

Steps run one after another, in order. Each sees the state left by the previous:

```python
from ai_arch_toolkit.core import Step, State
from ai_arch_toolkit.toolkit.flow import Flow

flow = Flow(
    Step(name="plan", fn=plan),
    Step(name="execute", fn=execute),
    Step(name="solve", fn=solve),
    name="my_pipeline",
)

state = State(operational={"task": "Write a report"})
result = await flow.run(state)
# or synchronously:
result = flow.run_sync(state)
```

### Cyclic

Steps with `when` conditions loop until no step fires or `max_iterations` is reached. You **must** set `max_iterations` explicitly — omitting it raises `ValueError` at construction time. This prevents accidental infinite loops:

```python
from ai_arch_toolkit.toolkit.flow import Flow, FlowStep

def needs_work(snap):
    return not snap.get("done", False)

flow = Flow(
    FlowStep(step=Step(name="process", fn=process), when=needs_work),
    FlowStep(step=Step(name="check", fn=check), when=needs_work),
    name="loop",
    max_iterations=10,  # required when using `when`
)
```

Each iteration goes through all steps, evaluates `when`, runs or skips. The loop stops when:
- No step's `when` condition returns True in a full pass
- `max_iterations` is reached

### DAG

Steps with `after` dependencies run in parallel when possible:

```python
flow = Flow(
    FlowStep(step=Step(name="fetch_weather", fn=weather)),
    FlowStep(step=Step(name="fetch_news", fn=news)),
    FlowStep(
        step=Step(name="summarize", fn=summarize),
        after=("fetch_weather", "fetch_news"),
    ),
    name="parallel_fetch",
)
```

The executor computes the execution order from dependencies:

```
fetch_weather ──┐
                ├──▶ summarize
fetch_news   ──┘
```

`fetch_weather` and `fetch_news` have no deps — they run **concurrently** via `asyncio.gather`. `summarize` waits for both.

#### Bounding the fan-out width

By default all ready steps run at once. Pass `Flow(..., max_parallelism=n)` to cap how many of *this flow's* steps run concurrently — useful to throttle parallel non-LLM work (tool/HTTP calls). It is **per-flow** (a nested flow has its own limit, so it never deadlocks), which is a different axis from the global, run-wide `inference_limit(n)` that caps concurrent LLM calls. See [Concurrency & Throttling](concurrency.md).

```python
flow = Flow(*many_steps, summarize, max_parallelism=5)  # ≤ 5 steps live at once
```

#### How parallel state works

1. Independent steps read the **snapshot** taken when their wave starts
2. They run concurrently — **they cannot see each other's artifacts**
3. Each reports `step_end` as it finishes; when the last one does, the wave's Results are
   **merged** back into State (in declaration order), before that last `step_end`

```
State ──snapshot──▶ fetch_weather ──Result A──┐
      ╲                                        ├──▶ State.merge(result_A, result_B)
       ─snapshot──▶ fetch_news    ──Result B──┘
```

The snapshot is a read-only view whose values are shared with the State (see
[StateSnapshot](#statesnapshot)): a step that mutates a list or dict it read, instead of returning
an artifact, changes the State for every step after it and for its siblings too.

**The rule**: if step B needs what step A produces, use `after=("A",)`. If they're truly independent, DAG parallel is safe. The executor enforces `after` deps, but cannot detect implicit State dependencies you forgot to declare.

#### Skip propagation

In DAG mode, failures cascade:
- Any dependency failed → step is skipped
- All dependencies skipped → step is skipped
- Dependency skipped → step is skipped (all deps must succeed)

### Streaming

`flow.iter(state)` runs the flow on the same engine as `run()` — parallel waves, isolation between
siblings, policies, and timeouts behave identically — and yields events as they happen:

```python
async with flow.iter(state) as execution:
    async for event in execution:
        match event.type:
            case "step_start":   print(f"  Running {event.step_name}")
            case "retry" | "timeout" | "fallback":
                print(f"  {event.type} in {event.step_name}")
            case "policy_decision":
                print(f"  {event.policy_decision} in {event.step_name or event.flow_name}")
            case "step_end":     print(f"  Done: {event.error or event.result.value}")
            case "step_skipped": print(f"  Skipped: {event.step_name}")
            case "flow_end":     print(f"Cost: ${event.trace.metadata['meter']['cost']:.4f}")

result = execution.result   # the FlowResult, once the loop has finished

# Or synchronously (the run happens on a background loop):
with flow.iter_sync(state) as sync_execution:
    for event in sync_execution:
        ...
```

| Event | Emitted when |
|---|---|
| `flow_start`, `flow_end` | The run starts; the run has finished (`flow_end.trace` is the complete trace). |
| `step_start`, `step_end` | A step starts; a step finishes (`step_end.result` and `step_end.error` carry the outcome). |
| `step_skipped` | A `when` condition was false, or a DAG dependency failed or was skipped. |
| `retry`, `timeout`, `fallback` | The step engine takes that decision — while the step is still running. |
| `policy_decision` | Any other decision: `low_confidence`, `escalate`, `halt`, `cost_exceeded`, `budget_exceeded`. |

The run only moves past a step when you ask for the next event. A `break` does not stop it by
itself: while you still hold the execution, the step in flight keeps running. Leaving the
`async with` block, or `await execution.aclose()`, cancels the steps still running and closes the
run's meter (for `iter_sync`, the `with` block or `close()`). When a run times out, the steps in
flight are cancelled before the `timeout` event is delivered. In a parallel wave, each sibling
reports `step_end` as it finishes, and the wave's artifacts are merged into the state once every
sibling has finished; if the run times out mid-wave, the siblings that already finished are kept. A step whose `when` or `Scope` callable raises is recorded with the
error, reported by `step_end`, and the flow stops.

### Composition — Flow as Step

A Flow can become a Step via `as_step()`. This is how agents compose:

```python
inner = Flow(
    Step(name="think", fn=think),
    Step(name="act", fn=act),
    name="inner_loop",
    max_iterations=5,
)

outer = Flow(
    Step(name="plan", fn=plan),
    inner,  # auto-converted via as_step()
    Step(name="summarize", fn=summarize),
    name="outer",
)
```

When a Flow runs as a Step:
- It gets a copy of each State layer (values shared by reference)
- Only new/changed artifacts are returned to the parent
- Its spend is metered under the parent's run — one shared meter, read from `result.meter` — and a
  per-step `Policy(max_cost=...)` on the wrapping step counts it
- Its steps appear in the trace as the wrapping step's `children`, as do the steps of any flow a
  step runs itself (e.g. an agent's inner ReAct loop); confidence propagates as the minimum

---

## Agent Flows

The package exposes these built-in agent flow factories:

```python
from ai_arch_toolkit.toolkit.agents.flows import (
    react_flow, react_initial_state,
    reflexion_flow, reflexion_initial_state,
    rewoo_flow, rewoo_initial_state,
    plan_execute_flow, plan_execute_initial_state,
    tot_flow, tot_initial_state,
    lats_flow, lats_initial_state,
    self_discovery_flow, self_discovery_initial_state,
    llm_compiler_flow, llm_compiler_initial_state,
    generate_review_flow, generate_review_initial_state,
)
```

Each flow factory has a companion `*_initial_state(task)` helper that creates the initial operational dict for `State(operational=...)`. This dict contains the task string and any agent-specific keys the flow steps expect to read and write.

### Usage pattern

Every flow factory follows the same pattern:

```python
from ai_arch_toolkit.core import LLM, State, ToolGroup
from ai_arch_toolkit.toolkit.agents.flows import react_flow, react_initial_state

llm = LLM("claude-sonnet-5")
tools = ToolGroup(my_tool_a, my_tool_b)

# Create the flow
flow = react_flow(llm, tools, system="You are a helpful assistant.")

# Create initial state
state = State(operational=react_initial_state("What's the weather in Paris?"))

# Run
result = await flow.run(state)

# response may be None if the flow halted early (check result.trace for errors)
response = state.get("response")
answer = response.text if response else result.trace.steps[-1].error
print(f"Cost: ${result.total_cost:.4f}")
```

### Per-phase LLM/tools override

Most flow factories accept override parameters for different phases:

```python
flow = plan_execute_flow(
    llm,
    tools,
    planner_llm=LLM("claude-sonnet-5"),    # cheap model for planning
    exec_llm=LLM("claude-sonnet-5"),        # same for execution
    solver_llm=LLM("claude-opus-5"),      # expensive model for final answer
)
```

### Flow options

Every factory takes the four options of the `Flow` it builds as keyword arguments (`timeout`,
`trace_capture`, `policy`, `budget_policy`) and hands them to the `Flow` unchanged. They are
declared once, as `FlowOptions`, so a wrapper forwards them with their types:

```python
from typing import Unpack

from ai_arch_toolkit.toolkit.agents.flows import FlowOptions, react_flow


def brief_react(llm: LLM, tools: ToolGroup, **options: Unpack[FlowOptions]) -> Flow:
    return react_flow(llm, tools, system="Answer in one sentence.", **options)
```

A strategy that runs a ReAct loop inside one of its steps passes that loop only its
`trace_capture`, so the inner steps are recorded like the outer ones. The outer run's deadline and
budget already cover the inner loop, and the outer `policy` applies to the outer steps only.

### ReAct

Cyclic flow — LLM reasoning + tool execution loop.

```python
flow = react_flow(llm, tools, system="...", max_iterations=10)
state = State(operational=react_initial_state("Find the capital of France"))
```

Steps: `llm_call` (when: needs_llm_call) → `execute_tools` (when: has_tool_calls) → loop

### Reflexion

Cyclic flow — inner ReAct with evaluate + reflect retry loop.

```python
def my_evaluator(task: str, answer: str) -> float:
    return 1.0 if "correct" in answer else 0.3

flow = reflexion_flow(llm, tools, evaluator=my_evaluator, threshold=0.7, max_retries=3)
state = State(operational=reflexion_initial_state("Solve this math problem"))
```

Steps: `attempt` → `evaluate` → `reflect` → loop (all gated by `when: not passed`)

### ReWOO

Sequential flow — plan with evidence placeholders, execute, solve.

```python
flow = rewoo_flow(llm, tools)
state = State(operational=rewoo_initial_state("Research topic X"))
```

Steps: `plan` → `execute` → `solve`

The planner generates `#E1 = ToolName[args]` steps. The executor runs tools sequentially, substituting `#E{n}` references. The solver synthesizes the final answer.

### Plan-Execute

Sequential flow — numbered plan, per-step ReAct execution, solve.

```python
flow = plan_execute_flow(llm, tools, max_replans=1, max_iterations_per_step=3)
state = State(operational=plan_execute_initial_state("Build a report on climate change"))
```

Steps: `plan_and_execute` → `solve`

The plan_and_execute step internally: plans numbered steps, runs each via inner ReAct, replans on failure.

### Tree of Thoughts

Cyclic flow — DFS/BFS search over reasoning paths.

```python
flow = tot_flow(llm, tools, strategy="dfs", n_candidates=3, max_depth=3, max_iterations=10)
state = State(operational=tot_initial_state("Solve this puzzle"))
```

Steps: `search_step` (when: search_not_done) → loop

Each iteration: select from frontier, generate candidates, evaluate, expand or solve.

### LATS

Cyclic flow — Monte Carlo Tree Search with ReAct rollouts.

```python
flow = lats_flow(llm, tools, n_candidates=5, max_rollouts=10, exploration_weight=1.41)
state = State(operational=lats_initial_state("Complex reasoning task"))
```

Steps: `mcts_rollout` (when: search_not_done) → loop

Each rollout: UCT selection, ReAct expansion, evaluation, backpropagation, optional reflection.
UCT picks the most promising node that still has fewer than `n_candidates` children, one ReAct
attempt from its state becomes a new child, and a low score adds a reflection for the attempts
expanded from it. A node thus gets up to `n_candidates` sibling attempts before the search goes
below it; `max_rollouts` caps the total number of attempts. Each attempt re-runs its tools from
scratch — there is no environment reset — so use `lats` only with read-only, idempotent or
sandboxed tools.

### Self-Discovery

Sequential flow — select reasoning modules, adapt, operationalize, solve via ReAct.

```python
flow = self_discovery_flow(llm, tools, max_react_iterations=10)
state = State(operational=self_discovery_initial_state("Analyze this problem"))
```

Steps: `select` → `adapt` → `operationalize` → `solve`

10 default reasoning modules (critical thinking, analogical reasoning, etc.) are selected and adapted to the task before solving.

### LLM Compiler

Sequential flow — plan DAG, parallel execute, join/replan.

```python
flow = llm_compiler_flow(llm, tools, max_replans=2, max_react_iterations=3)
state = State(operational=llm_compiler_initial_state("Multi-step research task"))
```

Steps: `compile` (internally: plan → parallel execute → join → optional replan)

The planner generates `$N. task [deps: $1, $2]` format. Independent tasks run concurrently via `asyncio.gather`.

### Generate-Review

Cyclic flow — a generator and a reviewer cooperate until the reviewer accepts the answer or the retry budget is exhausted.

```python
flow = generate_review_flow(gen_llm, review_llm, max_cycles=3)
state = State(operational=generate_review_initial_state("Draft a release note"))
```

Steps: `generate` → `review` → loop while not accepted

Useful when you want an explicit critique pass, optional tool use in both phases, and accumulated reviewer feedback injected into later generation attempts. Both phases take their own LLM (`gen_llm` / `review_llm`), optional tools (`gen_tools` / `review_tools`), and iteration caps.

---

## Step Engine

`execute_step()` lives in `core/` — it has **zero toolkit imports**. It receives an already-scoped StateSnapshot and runs a single Step with full policy enforcement:

```python
from ai_arch_toolkit.core._step_engine import execute_step

result, step_trace = await execute_step(step, scoped_snapshot)
```

The Flow executor handles scoping before calling the engine:

```python
scoped = apply_scope(state.snapshot(), scope)
result, trace = await execute_step(step, scoped)
state.merge(result)
```

This separation means:
- `core/` knows nothing about Flows, Scope, or orchestration
- `toolkit/flow/` handles the "how to compose" concerns
- The boundary is clean — you can use `execute_step()` directly for one-off step execution

---

## Complete Example

A custom flow that plans in parallel, then synthesizes:

```python
from ai_arch_toolkit.core import LLM, State, Step, Result, Policy, StateSnapshot
from ai_arch_toolkit.toolkit.flow import Flow, FlowStep, Scope

llm = LLM("claude-sonnet-5")

async def research_tech(snap: StateSnapshot) -> Result:
    task = snap.require("task")
    response = await llm.complete(f"Research technical aspects of: {task}")
    return Result(
        value=response.text,
        artifacts={"tech_research": response.text},
    )  # no manual cost: the run's meter captures LLM spend automatically

async def research_market(snap: StateSnapshot) -> Result:
    task = snap.require("task")
    response = await llm.complete(f"Research market aspects of: {task}")
    return Result(
        value=response.text,
        artifacts={"market_research": response.text},
    )  # no manual cost: the run's meter captures LLM spend automatically

async def synthesize(snap: StateSnapshot) -> Result:
    tech = snap.require("tech_research")
    market = snap.require("market_research")
    response = await llm.complete(
        f"Synthesize:\nTechnical: {tech}\nMarket: {market}"
    )
    return Result(
        value=response.text,
        artifacts={"report": response.text},
    )  # no manual cost: the run's meter captures LLM spend automatically

flow = Flow(
    FlowStep(step=Step(name="research_tech", fn=research_tech)),
    FlowStep(step=Step(name="research_market", fn=research_market)),
    FlowStep(
        step=Step(name="synthesize", fn=synthesize),
        after=("research_tech", "research_market"),  # waits for both
    ),
    name="parallel_research",
)

state = State(operational={"task": "Electric vehicle batteries"})
result = await flow.run(state)

print(state["report"])
print(f"Total cost: ${result.total_cost:.4f}")   # from the meter (single source of truth)
print(f"Duration: {result.total_duration:.1f}s")

# Per-step timing (per-step cost lives in the meter, not the trace):
for st in result.trace.steps:
    print(f"  {st.name}: {st.duration:.1f}s")
```

This runs `research_tech` and `research_market` concurrently (DAG mode detects they're independent), then `synthesize` after both complete.
