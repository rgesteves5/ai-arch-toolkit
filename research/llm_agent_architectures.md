# LLM Agent Architectures

> A practical reference for building your own agent using model API endpoints.

---

## The Core Loop

Every single-agent architecture is a variation of one idea: call the LLM, check if it wants to use a tool, execute the tool, feed the result back, and repeat until it gives a final answer. The differences lie in **when you plan**, **how you handle failures**, and **whether you explore alternatives**.

---

## 1. ReAct (Reason + Act)

*The default. Sequential think–act–observe loop.*

The LLM interleaves reasoning ("I need to search for X") with action (tool call) and observation (tool result). Each step depends on the previous one. This is what most frameworks use under the hood.

```
while not done:
    response = llm(messages)
    if response.has_tool_call:
        result = execute(tool_call)
        messages.append(result)
    else:
        return response.text
```

**Tradeoffs:** Simple and adaptive, but sequential — each step waits for the previous one. Can get stuck in loops on hard problems.

---

## 2. ReWOO (Reasoning Without Observation)

*Plan everything upfront, execute without re-reading observations.*

The LLM generates a complete plan with placeholder variables (#E1, #E2) before any tool is called. A worker executes each step, substituting real results for placeholders. A solver then synthesizes the final answer. The LLM is only called twice (plan + solve), saving significant token cost.

```
plan = planner_llm(task)                              # generates steps with #E1, #E2 refs
for step in plan:
    results[step.id] = execute(step, substituting prior results)
answer = solver_llm(task, all_results)
```

**Tradeoffs:** Much cheaper and faster than ReAct (fewer LLM calls, smaller context). But if the plan is wrong or a tool fails unexpectedly, the agent can't adapt mid-execution.

---

## 3. LLMCompiler (Parallel DAG Execution)

*Plan a dependency graph, execute independent tasks in parallel.*

Inspired by compiler optimization. The LLM generates a DAG (directed acyclic graph) of tasks with explicit dependencies ($1 = output of task 1). Independent tasks run concurrently. A Joiner evaluates results and can trigger re-planning if needed. Achieves 2–5× cost/latency reduction over ReAct.

```
dag = planner_llm(task)                  # generates tasks + dependency graph
while dag has unfinished tasks:
    ready = tasks with all dependencies met
    results = parallel_execute(ready)
    update(dag, results)
answer = joiner_llm(all_results)         # or trigger re-plan
```

**Tradeoffs:** Excellent for tasks with parallelizable sub-steps. More complex to implement. Re-planning adds robustness but also complexity.

---

## 4. Reflexion (Self-Critique with Memory)

*Run, evaluate, reflect on failure, retry with the reflection in context.*

The agent attempts the task, then an evaluator scores the output. If it fails, the LLM writes a natural-language reflection ("I failed because I didn't handle empty lists"). This reflection is stored and included in the prompt for the next attempt. This is "verbal reinforcement learning" — learning from mistakes using language, not gradient updates.

```
reflections = []
for attempt in range(max_retries):
    result = agent.run(task, reflections)
    score = evaluate(result)
    if score >= threshold: return result
    reflection = llm("What went wrong? " + result)
    reflections.append(reflection)
```

**Tradeoffs:** Improves accuracy by 15–20% on coding benchmarks vs. base ReAct. Requires an evaluation function. Cost scales linearly with retries.

---

## 5. LATS (Language Agent Tree Search)

*Monte Carlo Tree Search adapted for LLM agents. Explore, evaluate, backtrack.*

Instead of committing to a single chain of actions, LATS builds a search tree of possible trajectories. It uses selection (UCT), expansion (generate N candidate actions), evaluation (LLM scores states), simulation (run to completion), and backpropagation (update node values). Failed trajectories produce reflections stored for future attempts. Doubles ReAct's performance on multi-hop QA.

```
tree = initialize(root=task)
while budget remains:
    node = select(tree)                  # UCT formula
    children = expand(node, n=5)         # generate candidate actions
    scores = evaluate(children)          # LLM-as-judge
    best = simulate(top_child)           # run to terminal state
    backpropagate(best.score)
    if best.is_success: return best
    reflect(best)                        # store failure analysis
```

**Tradeoffs:** Most powerful architecture for complex reasoning. Very expensive — 10–50× more API calls than ReAct. Best for high-value tasks.

---

## 6. Tree of Thoughts (ToT)

*Generate multiple candidate thoughts at each step, score them, prune.*

A lighter alternative to LATS. At each reasoning step, the LLM generates K candidate "thoughts" (partial solutions). Each is scored by the LLM or a heuristic. Only the top candidates are pursued. Can use BFS or DFS traversal. Works well for problems with clear intermediate evaluation criteria. This is less an "agent with tools" pattern and more a meta-reasoning strategy that can be combined with any other architecture.

```
def solve(state, depth):
    if is_terminal(state): return state
    thoughts = [llm.generate(state) for _ in range(k)]
    scores = [llm.evaluate(t) for t in thoughts]
    for t in top_n(thoughts, scores):
        result = solve(t, depth + 1)
        if result: return result
```

**Tradeoffs:** Flexible and composable. Cost is K × depth LLM calls. Requires a way to score partial solutions.

---

## 7. Plan-then-Execute (with Re-planning)

*Generate a full plan first, execute step by step, re-plan on failure.*

A powerful planning LLM creates a structured multi-step plan. A simpler/cheaper executor carries out each step. If a step fails or produces unexpected results, the planner is re-invoked. The plan can include conditional branching ("if Tool A fails, try Tool B"). The executor can be a smaller model, a ReAct sub-agent, or even deterministic code.

```
plan = planner_llm(task)                 # structured step list
for step in plan:
    result = executor(step)              # can be cheaper model or code
    if result.failed:
        plan = planner_llm(task, context=results_so_far)
    context.append(result)
answer = planner_llm("Synthesize: " + all_results)
```

**Tradeoffs:** Great for complex, well-structured tasks. The planner is expensive but called sparingly. More rigid than ReAct but more efficient.

---

## 8. Self-Discovery

*Select relevant reasoning strategies, structure them into a plan, then execute.*

Before solving the problem, the LLM selects which reasoning modules are relevant from a library (e.g., "break into subtasks", "think step by step", "use analogies", "consider edge cases"). It then composes them into a structured reasoning plan tailored to the specific problem. Finally, it executes that plan.

```
modules = ["decompose", "analogize", "step-by-step", "verify", ...]
selected = llm("Which modules help with: " + task, modules)
plan = llm("Structure a reasoning plan using: " + selected)
answer = llm("Execute this plan on: " + task, plan)
```

**Tradeoffs:** Adaptive to problem type. Relatively cheap (3 LLM calls). Less explored for tool-heavy tasks.

---

## Quick Decision Guide

| Architecture | Best For | LLM Calls | Parallel? | Adaptive? | Complexity |
|---|---|---|---|---|---|
| **ReAct** | General tasks | Many | No | High | ~50 lines |
| **ReWOO** | Cost-sensitive | 2–3 | No | Low | ~100 lines |
| **LLMCompiler** | Parallel tools | 2–3+ | Yes | Medium | ~200 lines |
| **Reflexion** | Reliability | 2–5× | No | High | ~80 lines |
| **LATS** | Hard reasoning | 10–50× | No | Very High | ~300 lines |
| **ToT** | Creative/math | K×depth | No | Medium | ~150 lines |
| **Plan+Execute** | Structured work | Few | Optional | Medium | ~120 lines |
| **Self-Discovery** | Novel problems | 3 | No | Medium | ~60 lines |

---

## How to Pick — Decision Tree

```
Do you need to use external tools/APIs?
│
├─ YES
│   ├─ Do you need parallel tool execution?
│   │   ├─ YES ──────────────────────────── LLMCompiler
│   │   └─ NO
│   │       ├─ Do you need cost efficiency?
│   │       │   ├─ YES ──────────────────── ReWOO
│   │       │   └─ NO ───────────────────── ReAct
│   │       └─ Do you need high reliability?
│   │           └─ YES ──────────────────── ReAct + Reflexion
│   │
│   └─ Is it a complex multi-step task?
│       ├─ YES ──────────────────────────── Plan-then-Execute
│       └─ NO ───────────────────────────── ReAct
│
└─ NO (pure reasoning)
    ├─ Can you break it into clear steps?
    │   ├─ YES ──────────────────────────── Plan & Solve
    │   └─ NO
    │       ├─ Is it a novel/complex problem?
    │       │   ├─ YES (need exploration) ── LATS
    │       │   ├─ YES (need adaptation) ── Self-Discovery
    │       │   └─ NO (need to learn) ───── Reflexion
    │       └─ Do you need multiple perspectives?
    │           └─ YES ──────────────────── Tree of Thoughts
    │
    └─ Default ──────────────────────────── ReAct
```

---

## Practical Advice

**Start with ReAct.** It's ~50 lines of code and handles 90% of use cases. The entire implementation is: call the LLM, check for tool calls, execute them, append results, repeat until `stop_reason == "end_turn"`.

**Add Reflexion when you need reliability.** Wrap your ReAct loop in a retry with self-critique. This is the highest ROI upgrade you can make.

**Move to LLMCompiler when latency matters.** If your agent calls 3+ independent tools per task, parallel execution can cut latency by 2–5×.

**Use LATS/ToT only for high-value tasks.** The 10–50× API cost increase is only justified when accuracy is critical and the task is genuinely hard.

**All you need to build any of these:** an LLM API wrapper, a tool registry (name → function), a message history manager, and a loop controller (max iterations, stop conditions). Every framework is just an opinionated wrapper around these four primitives.
