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

Steps never see the mutable State. They receive a **StateSnapshot** — a frozen, immutable view:

```python
snapshot = state.snapshot()  # MappingProxyType per layer
snapshot["task"]             # reads work
snapshot["task"] = "x"       # TypeError — immutable
```

Steps return Results. The executor merges Result artifacts back into State.

### Fork and Merge

When parallel steps run (DAG mode), each gets an isolated copy:

```python
forked = state.fork()
# Deep copies: current, operational, persistent
# Shared by reference: world
```

After parallel steps finish, their Results merge back:

```python
state.merge(result_a, result_b, strategy="last_wins")
```

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

All decisions are recorded in the Trace.

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
children: nested StepTraces (from sub-flows)
input_state, output_result: serialized snapshots
```

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

#### How parallel state works

1. Independent steps get **forked** State (deep copy of current/operational/persistent, world shared)
2. They run concurrently — **they cannot see each other's writes**
3. After all finish, Results are **merged** back into State

```
State ──fork──▶ State_A (fetch_weather writes here)
       ╲
        fork──▶ State_B (fetch_news writes here)

After gather: State.merge(result_A, result_B)
```

**The rule**: if step B needs what step A produces, use `after=("A",)`. If they're truly independent, DAG parallel is safe. The executor enforces `after` deps, but cannot detect implicit State dependencies you forgot to declare.

#### Skip propagation

In DAG mode, failures cascade:
- Any dependency failed → step is skipped
- All dependencies skipped → step is skipped
- Dependency skipped → step is skipped (all deps must succeed)

### Streaming

Every flow supports event streaming:

```python
async for event in flow.iter(state):
    match event.type:
        case "flow_start":  print(f"Starting {event.flow_name}")
        case "step_start":  print(f"  Running {event.step_name}")
        case "step_end":    print(f"  Done: {event.result.value}")
        case "step_skipped": print(f"  Skipped: {event.step_name}")
        case "flow_end":    print(f"Cost: ${event.trace.metadata['meter']['cost']:.4f}")  # meter

# Or synchronously:
for event in flow.iter_sync(state):
    ...
```

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
- It gets a forked State (world shared by reference)
- Only new/changed artifacts are returned to the parent
- Cost, usage, confidence propagate up automatically

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

llm = LLM("claude-sonnet-4-20250514")
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
    planner_llm=LLM("claude-sonnet-4-20250514"),    # cheap model for planning
    exec_llm=LLM("claude-sonnet-4-20250514"),        # same for execution
    solver_llm=LLM("claude-opus-4-0-20250514"),      # expensive model for final answer
)
```

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
flow = lats_flow(llm, tools, max_rollouts=10, exploration_weight=1.41)
state = State(operational=lats_initial_state("Complex reasoning task"))
```

Steps: `mcts_rollout` (when: search_not_done) → loop

Each rollout: UCT selection, ReAct expansion, evaluation, backpropagation, optional reflection.

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

llm = LLM("claude-sonnet-4-20250514")

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
