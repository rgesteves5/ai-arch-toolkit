# Cognitive Agent Architecture

A unified agent architecture that combines the cognitive science research from `human-memory/` and
`thinking-systems-frameworks/` into a concrete implementation built on the toolkit's Flow, Step,
State, LLM, ToolGroup, and Middleware primitives.

---

## System Overview

```
                           ┌─────────────────┐
                           │  Problem Input   │
                           │  (Content)       │
                           └────────┬────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     DUAL-PROCESS GATE         │
                    │     (Python heuristic, 0 LLM) │
                    │                               │
                    │  signal scan → complexity     │
                    │  score → route decision       │
                    └───┬───────────┬───────────┬───┘
                        │           │           │
               ─────────┘           │           └──────────
              │                     │                      │
              ▼                     ▼                      ▼
     ┌────────────────┐  ┌──────────────────┐  ┌────────────────────┐
     │   SYSTEM 1     │  │  SYSTEM 2 LIGHT  │  │   SYSTEM 2 DEEP   │
     │   (1 LLM call) │  │  (3-5 LLM calls) │  │  (8-18 LLM calls) │
     │                │  │                  │  │                    │
     │  direct_flow() │  │  light_flow()    │  │  deep_flow()       │
     │                │  │                  │  │                    │
     │  Kernel prompt │  │  Kernel prompt   │  │  Kernel prompt     │
     │  No tools      │  │  2-4 tools       │  │  Full playbook     │
     │  No memory     │  │  Working memory  │  │  Tiered memory     │
     └───────┬────────┘  └────────┬─────────┘  └─────────┬──────────┘
             │                    │                       │
             └────────────────────┼───────────────────────┘
                                  │
                                  ▼
                        ┌──────────────────┐
                        │   Final Answer   │
                        │   (Response)     │
                        └──────────────────┘
```

The architecture has **three operating layers** running simultaneously:

| Layer | Role | Implementation | Lifetime |
|-------|------|----------------|----------|
| **Kernel** | Always-on monitoring: metacognition, bias detection, budget governor | `Middleware` on every `LLM` call | Every call |
| **OS** | Situation classification, playbook selection, OODA rhythm | Python code: heuristic classifier + config lookup | Per problem |
| **Application** | Individual thinking systems | `@tool` functions in a `ToolGroup` | Per phase |

---

## Layer 1: The Kernel (Middleware)

The Kernel is a `Middleware` instance injected into every `LLM` used by the cognitive agent. It
never turns off. It modifies requests before they reach the provider and inspects responses after.

### What it does

| Hook | Behavior |
|------|----------|
| `before` | Injects metacognitive system prompt. Enforces token budget. Tracks cumulative cost. Logs effort spent. |
| `after` | Reads confidence from structured output. Detects contradiction with prior semantic memory. Flags low confidence or bias signals. Updates budget remaining. |

### Metacognitive system prompt (injected into every call)

```
You are operating within a cognitive reasoning system.

MONITORING (do this silently, always):
- Track your confidence in your output (0.0-1.0).
- Notice if you are on autopilot (System 1) when the problem needs deliberation (System 2).
- Flag if your answer contradicts any prior finding listed in context.
- Notice if you are anchoring to the first idea instead of exploring alternatives.

BUDGET:
- You have used {used_calls}/{max_calls} reasoning steps and ${cost_spent:.4f}/${cost_limit:.4f}.
- If you are close to budget, prioritize synthesis over further exploration.

CONTEXT:
{semantic_memory_summary}
```

### Budget governor

The Kernel tracks cumulative cost and call count across the entire cognitive flow. When budget
thresholds are reached:

| Threshold | Action |
|-----------|--------|
| 70% budget used | Inject "begin converging" hint into next prompt |
| 90% budget used | Force synthesis step, skip remaining tools |
| 100% budget used | Halt with best answer so far |

### Implementation shape

```python
class CognitiveKernel:
    """Middleware — monitors every LLM call."""

    def before(self, request: Request) -> Request:
        # Inject metacognitive prompt
        # Enforce token budget on messages
        # Log call count
        ...

    def after(self, request: Request, response: Response) -> Response:
        # Extract confidence from structured output
        # Check for contradictions against semantic memory
        # Update cumulative cost tracking
        # Flag bias signals
        ...
```

Maps to: `core/_middleware.py` → `Middleware` protocol.

---

## Layer 2: The OS (Classification + Routing)

The OS layer runs **before any LLM call**. It classifies the problem, selects a playbook, and
configures the Flow that will execute.

### Component 1: Dual-Process Router

A pure Python function (zero LLM calls, <1ms) that scores problem complexity and routes to one of
three paths.

```python
COMPLEXITY_SIGNALS: dict[str, list[str]] = {
    "high": [
        "strategy", "tradeoff", "consequences", "stakeholders",
        "novel", "unprecedented", "crisis", "dilemma",
    ],
    "medium": [
        "compare", "analyze", "evaluate", "design", "plan",
        "optimize", "why", "recommend", "architecture",
    ],
    "low": [
        "what is", "define", "list", "summarize", "explain",
        "how to", "steps", "convert", "calculate",
    ],
}

@dataclass(frozen=True, slots=True)
class Route:
    path: Literal["system_1", "system_2_light", "system_2_deep"]
    complexity_score: float
    signals_found: tuple[str, ...]

def route(problem: str, user_override: str | None = None) -> Route:
    """Score complexity signals, return routing decision."""
    ...
```

Expected distribution: ~65% System 1, ~25% System 2 Light, ~10% System 2 Deep.

### Component 2: Cynefin Classifier

Runs only for System 2 Deep. Two-tier: fast heuristic first, LLM fallback if confidence < 0.55.

```python
DOMAIN_SIGNALS: dict[str, dict] = {
    "clear": {
        "signals": ["how to", "best practice", "standard", "procedure", "recipe"],
        "anti": ["uncertain", "complex", "novel", "emergent"],
    },
    "complicated": {
        "signals": ["optimize", "debug", "architecture", "expert", "root cause"],
        "anti": ["unprecedented", "chaotic", "impossible"],
    },
    "complex": {
        "signals": ["stakeholder", "culture", "strategy", "ecosystem", "emergent"],
        "anti": ["simple", "straightforward", "obvious"],
    },
    "chaotic": {
        "signals": ["crisis", "emergency", "urgent", "immediately", "fire"],
        "anti": ["long-term", "plan", "gradually"],
    },
    "novel": {
        "signals": ["never been done", "no precedent", "invent", "paradigm", "first ever"],
        "anti": ["improve", "optimize", "standard"],
    },
}

@dataclass(frozen=True, slots=True)
class Classification:
    domain: Literal["clear", "complicated", "complex", "chaotic", "novel"]
    confidence: float
    method: Literal["heuristic", "llm"]
```

### Component 3: Playbook Selector

Given a Cynefin domain, returns a deterministic sequence of thinking system tools to invoke.

```python
PLAYBOOKS: dict[str, list[str]] = {
    "clear":       ["critical_thinking"],
    "complicated": ["systems_thinking", "first_principles", "critical_thinking",
                    "bayesian_update", "second_order"],
    "complex":     ["first_principles", "divergent_thinking", "dialectical_synthesis",
                    "bayesian_update", "pre_mortem", "sensemaking"],
    "chaotic":     ["inversion", "critical_thinking", "sensemaking"],
    "novel":       ["first_principles", "inversion", "lateral_thinking",
                    "divergent_thinking", "convergent_evaluation", "pre_mortem"],
}
```

Eight situation-specific playbooks are also available (crisis response, breakthrough innovation,
high-stakes decision, complex system understanding, human-centered design, competitive strategy,
deep learning, group facilitation). These override the domain default when the user specifies a
scenario or the classifier detects one.

---

## Layer 3: Application (Thinking System Tools)

Each thinking system is a `@tool`-decorated function that takes a problem/context string and returns
structured JSON output. All tools share a **common output envelope**:

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class ThinkingOutput:
    """Common envelope returned by every thinking system tool."""
    confidence: float               # 0.0-1.0
    insights: list[str]             # Key findings
    metacog_assessment: str         # Self-assessment of reasoning quality
    suggested_next: list[str]       # Which tools to run next (advisory)
    should_loop_back: bool          # Whether to revisit an earlier phase
    loop_target: str | None         # Which phase to revisit
    system_specific: dict           # Tool-specific structured output
```

### Tool tiers

#### Tier 1 — Core (build first, universal value)

| Tool | What it does | System-specific output |
|------|-------------|----------------------|
| `first_principles` | Decompose to bedrock truths, rebuild | `assumptions`, `bedrock_truths`, `reframed_problem` |
| `inversion` | "How to guarantee failure?" | `failure_guarantees`, `inverted_insights`, `avoided_traps` |
| `pre_mortem` | Imagine the solution failed, explain why | `failure_modes[{description, probability, severity, mitigation}]`, `revised_confidence` |
| `bayesian_update` | Explicit prior → evidence → posterior | `prior`, `likelihood`, `posterior`, `evidence_impact`, `key_uncertainties` |
| `critical_thinking` | Evaluate claims against evidence standards | `claims_evaluated[{claim, evidence, strength, verdict}]`, `logical_fallacies`, `missing_evidence` |
| `divergent_thinking` | Generate maximum options, no judgment | `ideas[]`, `categories`, `wildcard_ideas[]` |
| `convergent_evaluation` | Filter and rank options against criteria | `criteria`, `ranked_options[{option, scores, total}]`, `eliminated[]` |

#### Tier 2 — High value (build second)

| Tool | What it does | System-specific output |
|------|-------------|----------------------|
| `systems_thinking` | Map variables, feedback loops, leverage points | `variables`, `causal_loops[{loop, type, effect}]`, `leverage_points` |
| `socratic_questioning` | Recursive probing questions to surface assumptions | `questions_asked`, `assumptions_surfaced`, `refined_understanding` |
| `sensemaking` | Construct plausible narrative from observations | `observations`, `narrative`, `gaps`, `alternative_narratives` |
| `second_order` | "And then what?" — trace consequences recursively | `consequences[{order, effect, probability}]`, `terminal_states` |
| `dialectical_synthesis` | Thesis + antithesis → synthesis | `thesis`, `antithesis`, `tensions`, `synthesis`, `residual_tensions` |

#### Tier 3 — Specialized (on demand)

| Tool | What it does |
|------|-------------|
| `lateral_thinking` | Random entry provocation, pattern breaking |
| `theory_of_constraints` | Find the one bottleneck |
| `probabilistic_thinking` | Assign distributions, expected values |

### Tool registration

All tools are collected into a single `ToolGroup`. The playbook controls which subset is passed
to the LLM on each call (via `ToolGroup` filtering or by constructing a subset `ToolGroup`).

```python
# Full registry
all_thinking_tools = ToolGroup(
    first_principles, inversion, pre_mortem, bayesian_update,
    critical_thinking, divergent_thinking, convergent_evaluation,
    systems_thinking, socratic_questioning, sensemaking,
    second_order, dialectical_synthesis, lateral_thinking,
    theory_of_constraints, probabilistic_thinking,
)

# Playbook-scoped subset for a complicated problem
playbook_tools = ToolGroup(
    systems_thinking, first_principles, critical_thinking,
    bayesian_update, second_order,
)
```

---

## Memory Architecture

Memory maps to the toolkit's `State` layers. The four-layer State
(`current` / `operational` / `persistent` / `world`) aligns with cognitive memory tiers:

| State Layer | Cognitive Tier | Content | Lifetime |
|-------------|---------------|---------|----------|
| `current` | **Working Memory** | Raw outputs from current phase, active context | Cleared each phase |
| `operational` | **Semantic Memory** | Extracted facts, decisions, compressed findings | Grows across session, pruned |
| `persistent` | **Episodic Log** | Process narrative, phase summaries | Append-only, session-scoped |
| `world` | **World Knowledge** | Problem input, playbook config, tool registry | Immutable reference |

### Token budgets

| Tier | Max tokens | Max items | Pruning strategy |
|------|-----------|-----------|-----------------|
| Working (current) | 2048 | ~8-10 | Oldest evicted on overflow |
| Semantic (operational) | 1024 | ~30 facts | Lowest-confidence pruned |
| Episodic (persistent) | 512 | ~20 entries | Oldest pruned |
| Kernel prompt | ~1000 | Fixed | N/A |
| Tool prompt | ~500 | Per call | N/A |
| **Total per call** | **~5000** | | **Bounded** |

### Consolidation

At each phase transition, a **consolidation step** runs:

1. **Extract** — LLM call (cheap, ~200 tokens) summarizes working memory into semantic facts.
2. **Compress** — Append one-line summary to episodic log.
3. **Clear** — Reset `current` layer for next phase.
4. **Prune** — If semantic memory exceeds 30 items, drop lowest-confidence entries.

This keeps context bounded regardless of how many phases execute.

### Memory in the Flow

```python
def consolidate(snapshot: StateSnapshot) -> Result:
    """Phase transition: working → semantic, log to episodic."""
    working_items = snapshot.current.get("working_outputs", [])
    semantic_facts = snapshot.operational.get("semantic_facts", [])
    episodic_log = snapshot.persistent.get("episodic_log", [])

    # LLM compression call
    extracted = llm.complete_sync(
        messages=[user(f"Extract key facts from:\n{working_items}")],
        system="Return a JSON list of facts with confidence scores.",
    )

    new_facts = extracted.parsed  # list[{fact, confidence}]
    updated_semantic = prune(semantic_facts + new_facts, max_items=30)

    phase_name = snapshot.operational.get("current_phase", "unknown")
    log_entry = f"Phase {phase_name}: {len(new_facts)} facts extracted"

    return Result(
        value="consolidated",
        artifacts={
            "working_outputs": [],           # clear working
            "semantic_facts": updated_semantic,
            "episodic_log": episodic_log + [log_entry],
        },
    )
```

### Forgetting (Active Pruning)

Forgetting is not a failure — it is optimization. Three mechanisms:

| Mechanism | Trigger | Action |
|-----------|---------|--------|
| **Displacement** | Working memory full | Oldest item evicted |
| **Decay** | Semantic fact unused for N phases | Confidence reduced by decay rate |
| **Pruning** | Semantic memory exceeds max items | Lowest-confidence facts removed |

Confidence decay formula: `new_confidence = confidence * (decay_rate ** phases_since_access)`

---

## Attention System

Attention determines which information enters working memory. Implemented as a `Scope` on each
`FlowStep`, controlling what the step can see.

### Salience scoring

When multiple candidate items compete for limited working memory slots:

```python
def salience(item: dict, goal: str, prediction_error: float) -> float:
    """Score item relevance for attention selection."""
    return (
        w_novelty * item.get("novelty", 0.5)
        + w_relevance * semantic_similarity(item["content"], goal)
        + w_emotion * item.get("emotional_significance", 0.0)
        + w_error * prediction_error
        + w_recency * recency_weight(item["timestamp"])
    )
```

### Scope filtering per step

Each FlowStep receives a `Scope` that controls visibility:

```python
# The LLM call step only sees semantic memory + current working items
llm_scope = Scope(
    include=frozenset({"messages", "semantic_facts", "working_outputs",
                       "current_phase", "playbook"}),
    exclude=frozenset({"episodic_log", "raw_responses"}),
)

# The reflect step sees everything including episodic log
reflect_scope = Scope(
    include=frozenset({"messages", "semantic_facts", "episodic_log",
                       "working_outputs", "total_usage"}),
)
```

---

## The Cognitive Loop (Master Sequence as a Flow)

The 7-phase Master Sequence maps to a cyclic `Flow` with conditional `FlowStep`s.

### Phase mapping

| Phase | Step name | What happens | Tools available | Condition |
|-------|-----------|-------------|----------------|-----------|
| 0. Orient | `classify` | Cynefin classification → playbook selection | None (heuristic + optional LLM) | Always runs first |
| 1. Perceive | `perceive` | Gather observations, map the situation | `systems_thinking` | S2 Deep only |
| 2. Frame | `frame` | Define the real problem | `first_principles`, `socratic_questioning`, `inversion` | S2 Light + Deep |
| 3. Generate | `generate` | Produce candidate solutions | `divergent_thinking`, `lateral_thinking`, `dialectical_synthesis` | S2 Deep only |
| 4. Evaluate | `evaluate` | Filter and stress-test candidates | `convergent_evaluation`, `critical_thinking`, `bayesian_update`, `pre_mortem`, `second_order` | S2 Light + Deep |
| 5. Decide | `decide` | Select and commit | `theory_of_constraints` | Always |
| 6. Reflect | `reflect` | Assess process, update beliefs | `sensemaking`, `bayesian_update` | S2 Deep only |
| — | `consolidate` | Working → semantic, log to episodic | None (compression LLM call) | After each phase |
| — | `synthesize` | Produce final answer from accumulated semantic memory | None (synthesis LLM call) | Terminal |

### Phase execution within a step

Each phase step internally runs a **ReAct-style loop** where the LLM calls thinking system tools
from the playbook. The loop continues until the LLM stops calling tools or the phase iteration
limit is reached.

```
Phase step (e.g., "frame"):
  1. Build messages: system(kernel_prompt) + user(phase_prompt + semantic_context)
  2. LLM.complete(messages, tools=playbook_tools_for_phase)
  3. If tool_calls → execute tools → append results → goto 2
  4. If no tool_calls → extract output → store in working memory → done
  5. Run consolidation (working → semantic)
```

This means each phase is itself a mini-flow (a nested `react_flow` scoped to phase-specific tools).

---

## Flow Composition

### Top-level: `cognitive_flow()`

The entry point. A factory function returning a `Flow`.

```
cognitive_flow(llm, thinking_tools, config) -> Flow
```

```
┌─────────────────────────────────────────────────────────────────────────┐
│  cognitive_flow                                                         │
│                                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────────────┐   │
│  │  route    │───▶│  classify    │───▶│  execute_path                │   │
│  │          │    │  (S2 Deep   │    │                              │   │
│  │  0 LLM   │    │   only)     │    │  One of:                    │   │
│  │  calls   │    │  0-1 LLM    │    │  • system_1_flow   (1 call) │   │
│  │          │    │  calls      │    │  • light_flow    (3-5 calls) │   │
│  │          │    │             │    │  • deep_flow    (8-18 calls) │   │
│  └──────────┘    └──────────────┘    └──────────────────────────────┘   │
│                                                                         │
│  ┌──────────────┐                                                       │
│  │  synthesize   │  ◀── always runs last                                │
│  │  1 LLM call  │                                                       │
│  └──────────────┘                                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

### System 1 path: `direct_flow()`

```
┌──────────────────────────────────────────────┐
│  direct_flow (sequential, 1 step)            │
│                                              │
│  ┌────────────────────────────────────────┐  │
│  │  direct_answer                         │  │
│  │                                        │  │
│  │  LLM.complete(                         │  │
│  │    messages=[user(problem)],            │  │
│  │    system=kernel_prompt,                │  │
│  │  )                                     │  │
│  │                                        │  │
│  │  No tools. No memory tiers.            │  │
│  │  Kernel prompt still active.           │  │
│  └────────────────────────────────────────┘  │
└──────────────────────────────────────────────┘
```

**LLM calls: 1**

### System 2 Light path: `light_flow()`

```
┌────────────────────────────────────────────────────────────┐
│  light_flow (sequential, 2-3 steps)                        │
│                                                            │
│  ┌─────────────────────────┐                               │
│  │  frame                  │  ReAct loop (max 2 iters)     │
│  │  tools: [first_principles, inversion]                   │
│  │  LLM calls: 1-2        │                               │
│  └────────────┬────────────┘                               │
│               │                                            │
│               ▼                                            │
│  ┌─────────────────────────┐                               │
│  │  evaluate               │  ReAct loop (max 2 iters)     │
│  │  tools: [critical_thinking, bayesian_update]            │
│  │  LLM calls: 1-2        │                               │
│  └────────────┬────────────┘                               │
│               │                                            │
│               ▼                                            │
│  ┌─────────────────────────┐                               │
│  │  consolidate + decide   │  1 LLM call                   │
│  └─────────────────────────┘                               │
└────────────────────────────────────────────────────────────┘
```

**LLM calls: 3-5.** Working memory only (no semantic/episodic tiers needed for this short path).

### System 2 Deep path: `deep_flow()`

The full Master Sequence. Each phase is a nested `react_flow` with phase-scoped tools.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  deep_flow (sequential with consolidation between phases)              │
│                                                                        │
│  ┌──────────────────────────────────────────┐                          │
│  │  Phase 1: perceive                       │                          │
│  │  Nested react_flow (max 2 iterations)    │                          │
│  │  Tools: [systems_thinking]               │                          │
│  │  LLM calls: 1-2                          │                          │
│  └─────────────────┬────────────────────────┘                          │
│                    │                                                    │
│        ┌───────────▼───────────┐                                       │
│        │  consolidate          │  1 LLM call (compression)             │
│        └───────────┬───────────┘                                       │
│                    │                                                    │
│  ┌──────────────────▼───────────────────────┐                          │
│  │  Phase 2: frame                          │                          │
│  │  Nested react_flow (max 3 iterations)    │                          │
│  │  Tools: [first_principles, socratic,     │                          │
│  │          inversion]                      │                          │
│  │  LLM calls: 1-3                          │                          │
│  └─────────────────┬────────────────────────┘                          │
│                    │                                                    │
│        ┌───────────▼───────────┐                                       │
│        │  consolidate          │  1 LLM call                           │
│        └───────────┬───────────┘                                       │
│                    │                                                    │
│  ┌──────────────────▼───────────────────────┐                          │
│  │  Phase 3: generate                       │                          │
│  │  Nested react_flow (max 3 iterations)    │                          │
│  │  Tools: [divergent_thinking, lateral,    │                          │
│  │          dialectical_synthesis]           │                          │
│  │  LLM calls: 1-3                          │                          │
│  └─────────────────┬────────────────────────┘                          │
│                    │                                                    │
│        ┌───────────▼───────────┐                                       │
│        │  consolidate          │  1 LLM call                           │
│        └───────────┬───────────┘                                       │
│                    │                                                    │
│  ┌──────────────────▼───────────────────────┐                          │
│  │  Phase 4: evaluate                       │                          │
│  │  Nested react_flow (max 3 iterations)    │                          │
│  │  Tools: [convergent_evaluation,          │                          │
│  │          critical_thinking,              │                          │
│  │          bayesian_update, pre_mortem,     │                          │
│  │          second_order]                   │                          │
│  │  LLM calls: 1-3                          │                          │
│  └─────────────────┬────────────────────────┘                          │
│                    │                                                    │
│        ┌───────────▼───────────┐                                       │
│        │  consolidate          │  1 LLM call                           │
│        └───────────┬───────────┘                                       │
│                    │                                                    │
│  ┌──────────────────▼───────────────────────┐                          │
│  │  Phase 5: decide                         │                          │
│  │  1 LLM call — select from evaluated      │                          │
│  │  candidates using semantic memory         │                          │
│  └─────────────────┬────────────────────────┘                          │
│                    │                                                    │
│  ┌──────────────────▼───────────────────────┐                          │
│  │  Phase 6: reflect                        │                          │
│  │  Nested react_flow (max 2 iterations)    │                          │
│  │  Tools: [sensemaking, bayesian_update]   │                          │
│  │  Scope: includes episodic_log            │                          │
│  │  LLM calls: 1-2                          │                          │
│  └──────────────────────────────────────────┘                          │
│                                                                        │
│  Total LLM calls: 8-18 (phases) + 4 (consolidation) = 12-22           │
│  Budget governor caps at 18 total.                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

### Phase step factory

Each phase is built by the same factory — a `react_flow` scoped to phase-specific tools, wrapped
with consolidation.

```python
def phase_step(
    name: str,
    llm: LLM,
    tools: ToolGroup,
    phase_prompt: str,
    max_iterations: int = 3,
    scope: Scope | None = None,
) -> tuple[FlowStep, FlowStep]:
    """Create a phase step + its consolidation step.

    Returns two FlowSteps: the phase ReAct loop and the consolidation.
    """
    inner = react_flow(
        llm, tools,
        system=phase_prompt,
        max_iterations=max_iterations,
    )

    phase = FlowStep(
        step=Step(name=name, fn=_make_phase_fn(inner, name)),
        after=(f"{prev_phase}_consolidate",) if prev_phase else (),
        scope=scope,
    )

    consolidation = FlowStep(
        step=Step(name=f"{name}_consolidate", fn=consolidate),
        after=(name,),
    )

    return phase, consolidation
```

### State initialization

```python
def cognitive_initial_state(problem: Content, config: CognitiveConfig) -> dict:
    return {
        # World layer (immutable reference)
        "problem": problem,
        "config": config,

        # Operational layer (semantic memory — grows)
        "semantic_facts": [],
        "current_phase": "orient",
        "playbook": [],
        "route": None,
        "classification": None,

        # Persistent layer (episodic log — append-only)
        "episodic_log": [],

        # Current layer (working memory — cleared each phase)
        "working_outputs": [],
        "messages": [user(problem)],
        "has_tool_calls": False,
        "total_usage": Usage(),
    }
```

---

## Thinking System Tool Design

Each tool follows the same pattern. The LLM calls the tool, the tool makes its own LLM call
with a specialized system prompt, and returns structured output.

### Option A: Tool as prompt template (no inner LLM call)

The tool is a prompt injection — it reshapes the problem for the LLM to reason about in a specific
way. The LLM's own response IS the tool output.

```python
@tool
def first_principles(problem: str, context: str = "") -> str:
    """Decompose a problem to its fundamental truths and rebuild from there.

    Args:
        problem: The problem to decompose.
        context: Prior findings to build on.
    """
    return json.dumps({
        "instruction": "Apply First Principles thinking to this problem.",
        "steps": [
            "1. List every assumption embedded in the problem statement.",
            "2. Classify each as: bedrock_truth, convention, habit, or uncertain.",
            "3. Discard conventions and habits. Keep only bedrock truths.",
            "4. Restate the problem using only bedrock truths.",
            "5. Build a solution from bedrock truths alone.",
        ],
        "problem": problem,
        "prior_context": context,
        "output_format": {
            "assumptions": [{"assumption": "...", "classification": "..."}],
            "bedrock_truths": ["..."],
            "reframed_problem": "...",
            "reconstructed_solution": "...",
        },
    })
```

This approach costs **zero additional LLM calls** — the tool output is injected back into the
conversation and the LLM processes it in its next response. This is the recommended approach for
most thinking systems.

### Option B: Tool with inner LLM call (dedicated model/temperature)

For tools that benefit from different parameters (e.g., high temperature for divergent thinking,
separate model for evaluation).

```python
@tool
async def divergent_thinking(problem: str, context: str = "") -> str:
    """Generate maximum options without judgment. Quantity over quality.

    Args:
        problem: The problem to brainstorm solutions for.
        context: Prior findings to build on.
    """
    creative_llm = LLM("claude-haiku-4-5-20251001", temperature=1.0)
    response = await creative_llm.complete(
        messages=[user(f"Generate 15+ ideas for: {problem}\nContext: {context}")],
        system="You are a divergent thinker. Generate as many ideas as possible...",
        output_schema=DivergentOutput,
    )
    return response.text
```

This costs **1 additional LLM call per invocation**, but allows tuning model/temperature/system
prompt independently.

### Recommended approach per tool

| Tool | Approach | Rationale |
|------|----------|-----------|
| `first_principles` | A (prompt) | Structured decomposition works in-context |
| `inversion` | A (prompt) | Single reframe, works in-context |
| `pre_mortem` | A (prompt) | Structured failure imagination, works in-context |
| `bayesian_update` | A (prompt) | Prior/posterior tracking, works in-context |
| `critical_thinking` | A (prompt) | Evaluation against standards, works in-context |
| `divergent_thinking` | B (inner call) | Benefits from high temperature |
| `convergent_evaluation` | A (prompt) | Ranking/filtering, works in-context |
| `systems_thinking` | A (prompt) | Causal loop mapping, works in-context |
| `socratic_questioning` | B (inner call) | Multi-turn questioning benefits from isolation |
| `dialectical_synthesis` | A (prompt) | Thesis/antithesis/synthesis, works in-context |
| `sensemaking` | A (prompt) | Narrative construction, works in-context |
| `second_order` | A (prompt) | Recursive "and then what?", works in-context |
| `lateral_thinking` | B (inner call) | Benefits from high temperature + random seed |

---

## Structured Output

Every LLM call within the deep flow uses `output_schema` to enforce structure. This prevents
parsing failures and guarantees the Kernel middleware can extract confidence scores.

### Phase output schema

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class PhaseOutput:
    """Schema for every phase LLM call."""
    response: str                          # Natural language answer
    confidence: float                      # 0.0-1.0
    key_findings: list[str]                # Facts to consolidate
    tool_calls_useful: bool                # Were the tools helpful?
    suggested_phase: str | None            # Override next phase if needed
    should_stop: bool                      # Confident enough to stop early
```

### Tool output schema (common envelope)

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class ToolOutput:
    """Schema for every thinking system tool response."""
    confidence: float
    insights: list[str]
    metacog_assessment: str
    suggested_next: list[str]
    should_loop_back: bool
    loop_target: str | None
    system_specific: dict
```

---

## Emotional Modulation

Emotional state modulates encoding strength, retrieval bias, and decision evaluation.
Represented as a 2D vector (valence, arousal) that decays toward neutral.

### Where it applies

| Component | Effect of high arousal | Effect of negative valence |
|-----------|----------------------|---------------------------|
| Encoding | Stronger storage (higher confidence on new facts) | Bias toward threat-related info |
| Retrieval | Broader activation (more associations surfaced) | Bias toward risk/failure memories |
| Attention | Lower threshold for salience (more gets through) | More weight on `w_emotion` term |
| Decision | More decisive (lower satisficing threshold) | More cautious (higher evidence bar) |
| Learning rate | Faster Hebbian strengthening | Faster aversion learning |

### Implementation

Emotional state is tracked in `operational` state and read by the Kernel middleware:

```python
@dataclass(frozen=True, slots=True)
class EmotionalState:
    valence: float = 0.0    # -1.0 (negative) to 1.0 (positive)
    arousal: float = 0.0    # 0.0 (calm) to 1.0 (activated)

    def decay(self, rate: float = 0.1) -> EmotionalState:
        """Decay toward neutral."""
        return EmotionalState(
            valence=self.valence * (1 - rate),
            arousal=self.arousal * (1 - rate),
        )
```

Emotional state is inferred from the problem (crisis → high arousal, negative valence) and
updated based on tool outputs (pre_mortem → increase arousal, decrease valence).

---

## Predictive Coding Layer

The brain is fundamentally a prediction machine. This maps to an optional always-on layer that:

1. **Predicts** — Before each phase, generates a brief prediction of what the phase will find.
2. **Compares** — After the phase, computes prediction error (how wrong was the prediction?).
3. **Updates** — High prediction error increases attention and learning rate for that phase's
   outputs. Low prediction error allows faster consolidation (less to learn).

### Implementation

```python
async def predict_phase(snapshot: StateSnapshot) -> Result:
    """Generate prediction for upcoming phase."""
    phase = snapshot["current_phase"]
    semantic = snapshot.get("semantic_facts", [])

    response = await llm.complete(
        messages=[user(f"Based on what we know so far:\n{semantic}\n\n"
                       f"Briefly predict what phase '{phase}' will find.")],
        system="One sentence prediction. Be specific.",
    )

    return Result(
        value=response.text,
        artifacts={"phase_prediction": response.text},
    )

async def compute_prediction_error(snapshot: StateSnapshot) -> Result:
    """Compare prediction to actual phase output."""
    prediction = snapshot.get("phase_prediction", "")
    actual = snapshot.get("working_outputs", [])

    response = await llm.complete(
        messages=[user(f"Prediction: {prediction}\nActual: {actual}\n\n"
                       f"How wrong was the prediction? Score 0.0-1.0")],
        output_schema=PredictionError,
    )

    error = response.parsed.error_score

    return Result(
        value=error,
        artifacts={
            "prediction_error": error,
            # High error → increase attention and learning rate
            "attention_boost": error > 0.5,
            "learning_rate_multiplier": 1.0 + error,
        },
    )
```

**Cost**: 2 additional LLM calls per phase (predict + compare). Optional — disabled in System 2
Light and System 1 paths.

---

## Procedural Memory (Automatization)

Over multiple runs, the agent can learn which playbooks work best for which problem types. This
maps to the cognitive progression: controlled → automatic.

### Mechanism

After each full run, if the Reflect phase rates the process as successful (confidence > 0.8):

1. Record: `{problem_signals, route_taken, playbook_used, outcome_confidence, cost}`
2. Over time, build a lookup table: `signal_pattern → best_playbook`
3. The Cynefin classifier checks this table before falling back to heuristic/LLM classification

This converts System 2 Deep decisions into System 1 cached responses — the definition of
expertise and automatization.

### Storage

Procedural memory lives outside the per-problem `State`. It is a persistent `GraphStore`
(or simple JSON file) that accumulates across conversations:

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class ProceduralEntry:
    """A learned pattern → playbook mapping."""
    signal_pattern: tuple[str, ...]   # Signals that triggered this
    domain: str                       # Cynefin domain
    playbook: tuple[str, ...]         # Tool sequence used
    outcome_confidence: float         # How well it worked
    cost: float                       # How much it cost
    count: int                        # How many times this pattern seen
```

---

## OODA Rhythm (Multi-Cycle Orchestration)

For complex problems, the Master Sequence may need to run multiple cycles. The OODA conductor
manages this:

```
Cycle 1: Orient → Perceive → Frame → Generate → Evaluate → Decide → Reflect
                                                                     │
         ┌───────────────────────────────────────────────────────────┘
         │  Reflect phase says: "confidence 0.5, loop back to Frame"
         │
         ▼
Cycle 2: Frame (refined) → Generate (new candidates) → Evaluate → Decide → Reflect
                                                                            │
         ┌──────────────────────────────────────────────────────────────────┘
         │  Reflect phase says: "confidence 0.85, stop"
         │
         ▼
         Synthesize final answer
```

### Implementation

The deep flow's `should_loop_back` flag in the structured output determines whether to cycle.
The flow is cyclic with `max_iterations` set by the budget governor (typically 2-3 cycles).

```python
deep = Flow(
    *perceive_steps,
    *frame_steps,
    *generate_steps,
    *evaluate_steps,
    decide_step,
    reflect_step,
    FlowStep(
        step=Step(name="loop_gate", fn=check_loop_back),
        when=lambda snap: snap.get("should_loop_back", False),
    ),
    name="deep_flow",
    max_iterations=3,  # Max OODA cycles
)
```

---

## Integration Rules (Enforced by Kernel)

These rules from the research are enforced programmatically:

| Rule | Enforcement |
|------|-------------|
| **Classify before you choose** | Route step always runs first (DAG dependency) |
| **Diverge before you converge** | Generate phase is sequenced before Evaluate |
| **Metacognition runs always** | Kernel middleware on every LLM call |
| **Inversion bookends everything** | Inversion tool in both Frame and Evaluate playbooks |
| **Depth proportional to stakes** | Budget governor scales with route (S1/S2L/S2D) |
| **Update, don't anchor** | Bayesian update tool available in every playbook |
| **Cycle, don't waterfall** | OODA rhythm allows loop-back from Reflect |

---

## Anti-Pattern Detection (Kernel Monitors)

The Kernel middleware watches for these anti-patterns:

| Anti-Pattern | Detection Signal | Kernel Response |
|-------------|-----------------|-----------------|
| **Premature convergence** | Evaluation tools called before divergent | Inject "have you explored enough options?" |
| **Analysis paralysis** | >5 tool calls in one phase without decision | Force consolidation and move to Decide |
| **Tool fetishism** | Same tool called >3 times consecutively | Inject "consider a different perspective" |
| **Anchoring** | First idea from Generate appears unchanged in Decide | Inject "are you anchoring to your first idea?" |
| **Overconfidence** | Confidence >0.9 with <3 tools consulted | Inject "have you stress-tested this?" |

---

## Cost Profile

| Path | LLM Calls | Estimated Latency | Estimated Cost | Frequency |
|------|-----------|-------------------|----------------|-----------|
| System 1 | 1 | ~2s | ~$0.003 | ~65% |
| System 2 Light | 3-5 | ~10s | ~$0.015 | ~25% |
| System 2 Deep | 12-22 | ~45-90s | ~$0.05-0.08 | ~10% |
| **Weighted average** | | | **~$0.009** | |

Compared to always running System 2 Deep (~$0.06): **~6.5x cheaper**.

---

## Configuration

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class CognitiveConfig:
    """Configuration for the cognitive agent."""

    # Models
    main_model: str = "claude-sonnet-4-6"
    creative_model: str = "claude-haiku-4-5-20251001"  # For divergent tools
    compression_model: str = "claude-haiku-4-5-20251001"  # For consolidation

    # Budget
    max_calls: int = 20
    max_cost: float = 0.10
    budget_converge_threshold: float = 0.7   # Start converging at 70% budget
    budget_force_threshold: float = 0.9      # Force synthesis at 90% budget

    # Memory
    working_memory_max_items: int = 10
    semantic_memory_max_items: int = 30
    episodic_log_max_entries: int = 20
    confidence_decay_rate: float = 0.05

    # Routing
    system_1_threshold: float = 2.0          # Complexity score below → S1
    system_2_light_threshold: float = 5.0    # Below → S2L, above → S2D
    cynefin_confidence_threshold: float = 0.55

    # Phases
    max_ooda_cycles: int = 3
    phase_max_iterations: int = 3
    early_stop_confidence: float = 0.85

    # Features
    enable_predictive_coding: bool = False
    enable_emotional_modulation: bool = False
    enable_procedural_learning: bool = False
```

---

## Build Sequence

### Week 1: MVP (Variation D — Tool-Use Agent)

1. Implement 6 Tier-1 thinking tools as `@tool` functions (Option A: prompt-only).
2. Build `cognitive_flow()` with System 1 path only.
3. Add Kernel as basic system prompt (no middleware yet).
4. Validate: does the agent call the right tools?

### Week 2: Dual-Process + Light Path

1. Implement Dual-Process Router (pure Python heuristic).
2. Build System 2 Light flow (frame + evaluate, 2 phases).
3. Add working memory (current layer management).
4. Validate: does routing work? Does S2 Light produce better answers?

### Week 3: Deep Path + Memory

1. Build System 2 Deep flow (all 7 phases).
2. Implement Kernel as `Middleware` with budget governor.
3. Add consolidation (working → semantic extraction).
4. Add Cynefin classifier (heuristic + LLM fallback).
5. Validate: does the full pipeline produce significantly better answers?

### Week 4: Optimization + Tier 2 Tools

1. Add Tier-2 tools (systems_thinking, socratic, sensemaking, second_order, dialectical).
2. Implement playbook selector (domain → tool sequence).
3. Add anti-pattern detection to Kernel.
4. Tune routing thresholds on test problems.
5. Validate: cost profile matches expected distribution.

### Week 5+: Advanced Features

1. Predictive coding layer (optional).
2. Emotional modulation (optional).
3. Procedural memory / automatization (optional).
4. Situation-specific playbooks (crisis, innovation, etc.).
5. Multi-agent ensemble for Tier-2 creative tools (Variation B components).

---

## File Structure

```
toolkit/
└── cognitive/
    ├── __init__.py              # Public API: cognitive_flow, CognitiveConfig
    ├── _router.py               # DualProcessRouter, Route
    ├── _classifier.py           # CynefinClassifier, Classification
    ├── _playbooks.py            # PLAYBOOKS dict, playbook_for_domain()
    ├── _kernel.py               # CognitiveKernel (Middleware)
    ├── _memory.py               # Consolidation, pruning, forgetting
    ├── _attention.py            # Salience scoring, scope factories
    ├── _emotion.py              # EmotionalState (optional)
    ├── _prediction.py           # Predictive coding layer (optional)
    ├── _procedural.py           # ProceduralEntry, learning (optional)
    ├── _schemas.py              # PhaseOutput, ToolOutput, ThinkingOutput
    ├── _flow.py                 # cognitive_flow(), direct_flow(), light_flow(), deep_flow()
    └── tools/
        ├── __init__.py          # all_thinking_tools ToolGroup
        ├── _tier1.py            # first_principles, inversion, pre_mortem, etc.
        ├── _tier2.py            # systems_thinking, socratic, sensemaking, etc.
        └── _tier3.py            # lateral_thinking, theory_of_constraints, etc.
```
