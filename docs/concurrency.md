# Concurrency & throttling

Two independent knobs control *how much runs at once*. They sit on different
axes — reach for the one that matches what you're protecting.

| | **`inference_limit(n)`** | **`Flow(max_parallelism=n)`** |
|---|---|---|
| Caps | total concurrent **LLM calls**, globally | **steps of one flow** that run at once |
| Scope | the whole run — every nested flow, agent, fallback | that one flow's fan-out |
| Protects | a shared resource (local GPU, rate-limited endpoint, connection pool) | orchestration width (memory, forked state, non-LLM work) |
| Default | unlimited | unlimited |

Both are **opt-in** — set neither and behaviour is exactly as before. They
**compose**: use them together when you want to bound both the resource and the
orchestration (see [Using both](#using-both)).

---

## `inference_limit` — global inference cap

A run-scoped ceiling on how many `LLM.complete()` calls are in flight at any
instant, shared across **everything** in the run — parallel steps, nested agents,
and fallbacks all draw from the same pool.

```python
from ai_arch_toolkit import inference_limit

# A local GPU that serves 2 inferences comfortably:
with inference_limit(2):
    result = agent.run_sync(task)     # never more than 2 concurrent calls to the model,
                                      # no matter how deeply the agents nest
```

**Behaviour & logic**

- **Model-agnostic.** Applies to cloud and local models alike. Most critical for
  local models (a GPU/CPU has a hard concurrency limit — exceed it and you get
  OOM or a latency collapse); useful for cloud too (stay under a provider's
  concurrency/RPM limit and avoid `429`s proactively, rather than reacting to
  them with [retries](llm.md)).
- **Global, via the ambient context.** The limit is carried on a `ContextVar`, so
  every task spawned under the scope shares one semaphore — the cap holds across
  arbitrary flow nesting.
- **Leaf-level, so it cannot deadlock.** The slot is acquired only around the
  single provider call, never held while a nested flow runs. (This is why it caps
  *LLM calls* and not *steps* — a step-level global cap would deadlock when a step
  is itself a nested flow waiting on the same pool.)
- **Nesting:** the innermost `inference_limit` wins (its own independent
  semaphore); the outer resumes when it exits.
- **Scope:** governs `complete()` / `complete_sync()` (the path every agent flow
  uses). Streaming calls are **not** throttled — they are rarely fanned out.
- `n` must be `>= 1` (else `ValueError`).

**Use it when** you must not exceed a hard concurrency limit on a shared
resource, *regardless of how the run is structured*.

---

## `Flow(max_parallelism)` — per-flow fan-out cap

Bounds how many of a single flow's **ready steps** execute at once. Each flow has
its own independent limit.

```python
from ai_arch_toolkit.toolkit.flow import Flow

# A DAG that fans out 100 scrape steps — run them 5 at a time (be polite):
flow = Flow(*scrape_steps, synthesize, max_parallelism=5)
```

**Behaviour & logic**

- **Per-flow, not global.** A nested flow gets its *own* semaphore, so this never
  deadlocks — but it also means total concurrency across nesting can reach
  `n × depth`. For a hard global ceiling, use `inference_limit`.
- **Bounds step *starts*.** Unlike `inference_limit` (where all steps start and
  then block at the LLM call), `max_parallelism` limits how many steps become
  *live* — so it bounds forked-state **memory**, **non-LLM** parallel work (tool
  calls, HTTP, DB), and gives a controlled rollout.
- **Only affects genuine fan-out** — the parallel branch of a DAG where several
  steps are ready simultaneously. Sequential and single-ready-step execution is
  unaffected.
- `n` must be `>= 1` (else `ValueError` at construction).

**Use it when** you want to shape one orchestration's width — bound memory, throttle
parallel tool/HTTP work, or roll out a wide fan-out gradually.

---

## Using both

They are orthogonal, so combine them for the common local-model swarm case:

```python
from ai_arch_toolkit import inference_limit
from ai_arch_toolkit.toolkit.flow import Flow

swarm = Flow(*agent_steps, join, max_parallelism=5)   # ≤ 5 agents live at once (memory)

with inference_limit(2):                               # ≤ 2 inferences hit the GPU at once
    result = swarm.run_sync(state)
```

`max_parallelism` keeps only 5 agents *alive* (bounding context/memory); `inference_limit`
keeps only 2 of their calls *hitting the model* at any instant. Neither substitutes for
the other.

## Choosing

- Protecting a resource with a hard concurrency limit (GPU, endpoint, pool) →
  **`inference_limit`**.
- Shaping the width of one orchestration (memory, non-LLM fan-out, rollout) →
  **`Flow(max_parallelism=...)`**.
- Both at once → the swarm pattern above.
- Neither set → unlimited, exactly as before.
