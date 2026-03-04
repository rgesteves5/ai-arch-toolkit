# Ultimate Agents — Brainstorm

Ideas from combining two research documents:
- **Doc 11**: Agent Architecture — Programming Thinking Systems (orchestration, 24 thinking frameworks, Cynefin/OODA dispatch)
- **Cognitive Architecture**: Full System Design (memory systems, learning, consolidation, emotional modulation, attention/salience)

---

This is the right question. Here's how I'd think about it.

---

## The Core Tension

Doc 11 has **brilliant orchestration but no memory** — it chains thinking frameworks through an LLM but treats each call as stateless. The Cognitive Architecture has **brilliant memory but no LLM** — it builds a full mind from data structures but reinvents reasoning that an LLM already does natively.

The merge point: **use the Cognitive Architecture's state layer as the substrate that the Thinking Systems Agent reasons over.**

---

## Architecture Alpha: The Hybrid — "LLM as Cortex"

This is the most natural merge. The LLM replaces the reasoning/inference/planning modules from the Cognitive Architecture. Everything else stays.

```
┌──────────────────────────────────────────────────────────────┐
│                    HYBRID COGNITIVE AGENT                     │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  KERNEL (always on, runs every turn)                   │  │
│  │                                                        │  │
│  │  Metacognitive Monitor ← from Doc 11                   │  │
│  │  Attention / Salience  ← from Cog Arch                 │  │
│  │  Emotional State       ← from Cog Arch                 │  │
│  │  Bounded Rationality   ← from Doc 11                   │  │
│  │  Dual-Process Router   ← NEW (merges both)             │  │
│  └──────────────┬─────────────────────────────────────────┘  │
│                 │                                            │
│  ┌──────────────▼─────────────────────────────────────────┐  │
│  │  MEMORY SUBSTRATE (persistent, structured state)       │  │
│  │                                                        │  │
│  │  Working Memory ─── 4-slot focus + goal stack          │  │
│  │  Episodic Memory ── timestamped event log + embeddings │  │
│  │  Semantic Memory ── weighted concept graph             │  │
│  │  Procedural Memory ─ compiled fast-paths (no LLM)      │  │
│  │  Emotional Memory ── valence tags on all of the above  │  │
│  └──────────────┬─────────────────────────────────────────┘  │
│                 │                                            │
│  ┌──────────────▼─────────────────────────────────────────┐  │
│  │  OS LAYER (classifies + routes)                        │  │
│  │                                                        │  │
│  │  Cynefin Classifier ── domain dispatch                 │  │
│  │  OODA Conductor ────── observe-orient-decide-act cycle │  │
│  │  Latticework Selector ─ cross-domain model selection   │  │
│  └──────────────┬─────────────────────────────────────────┘  │
│                 │                                            │
│  ┌──────────────▼─────────────────────────────────────────┐  │
│  │  LLM REASONING LAYER (replaces Cog Arch inference)     │  │
│  │                                                        │  │
│  │  Thinking System modules (Doc 11's 24 systems)         │  │
│  │  Each module = a prompt template + structured output    │  │
│  │  LLM does: inference, planning, analogy, generation    │  │
│  └──────────────┬─────────────────────────────────────────┘  │
│                 │                                            │
│  ┌──────────────▼─────────────────────────────────────────┐  │
│  │  LEARNING + CONSOLIDATION (background)                 │  │
│  │                                                        │  │
│  │  Hebbian: strengthen co-occurring concept edges        │  │
│  │  RL: update action values from outcomes                │  │
│  │  Schema: extract patterns from episodic clusters       │  │
│  │  Consolidation: prune, compress, rebuild indices       │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### How a request flows through Alpha:

1. **Input arrives** → sensory buffer
2. **Attention** computes salience (novelty, goal-relevance, emotional weight, prediction error)
3. **Dual-Process Router** decides: is this a System 1 or System 2 problem?
   - **System 1**: Check procedural memory for compiled fast-paths. If found, execute without LLM call. Done.
   - **System 2**: Continue to full pipeline.
4. **Working Memory** loads: the input, retrieved episodic context ("have I seen this before?"), retrieved semantic context ("what do I know about this?"), current goal stack
5. **Cynefin Classifier** classifies the domain (this can be a fast heuristic first, LLM call if ambiguous)
6. **OS dispatches** the right playbook → sequence of thinking system modules
7. **Each thinking system module** gets a prompt assembled from:
   - Its own template (from Doc 11)
   - Working memory contents (from Cog Arch)
   - Relevant episodic memories (cue-based retrieval)
   - Relevant semantic knowledge (spreading activation)
   - Metacognitive wrapper (confidence, alerts, budget)
8. **LLM executes** the prompt, returns structured output
9. **Post-processing**: update working memory, write to episodic log, update emotional state, Hebbian strengthening of co-active concepts, RL value update if outcome signal exists
10. **Consolidation** runs periodically: replay, schema extraction, pruning

### What the LLM replaces from the Cognitive Architecture:
- Inference engine (deduction, induction, abduction, analogy) → LLM does this natively
- Planner (goal decomposition, search) → LLM + thinking system prompts
- Predictive model (at the abstract level) → LLM's world knowledge

### What the LLM does NOT replace:
- Memory systems → LLM has no persistent memory; you must build it
- Attention/salience → LLM doesn't decide what's important; you must gate
- Learning → LLM doesn't update from experience; you must track
- Consolidation → LLM doesn't compress or reorganize; you must do it offline
- Emotional modulation → LLM doesn't weight memories by affect; you must tag
- Procedural fast-paths → LLM is too slow/expensive for compiled habits

### Advantages:
- Best of both: structured memory + powerful reasoning
- The LLM handles what it's good at (language, inference, planning, analogy)
- The cognitive substrate handles what the LLM can't do (persistence, learning, attention, consolidation)
- Thinking system modules give the LLM *structured approaches* rather than free-form reasoning
- Dual-Process Router saves cost — most routine queries never hit the full pipeline

### Disadvantages:
- Complex to build. This is the most ambitious option.
- Many LLM calls per complex problem (memory retrieval + classification + N thinking systems + reflection)
- Memory systems need their own storage backend (graph DB, vector store, event log)
- Tuning the interaction between cognitive state and LLM prompts is non-trivial
- Risk of the cognitive substrate and the LLM "disagreeing" (e.g., attention says X is irrelevant but the LLM would have found it useful)

---

## Architecture Beta: "Cognitive Memory + Simple Reasoning"

Strip out Doc 11's full thinking system orchestration. Keep the memory substrate from the Cognitive Architecture. Use the LLM with a single well-crafted system prompt that encodes the kernel principles (metacognition, bounded rationality, dual-process awareness). No Cynefin dispatch, no 24-module registry.

```
Input → Attention/Salience → Memory Retrieval → 
  LLM (single call with rich context from memory) → 
    Output → Learning Updates → Periodic Consolidation
```

The LLM gets a system prompt that says: "You have access to these memories. Here's what's in your working memory. Here's your current confidence. Reason carefully." But it decides *how* to reason on its own — no framework dispatch.

### Advantages:
- Much simpler to build. The memory layer is the hard part, and you're building that anyway.
- Fewer LLM calls (typically 1 per turn, occasionally 2 for reflection).
- The LLM's native reasoning is often good enough — it doesn't need to be told "use First Principles" to reason from first principles.
- Easier to debug. One call with clear context vs. a 10-step orchestration.
- Lower latency, lower cost.

### Disadvantages:
- You lose the structured thinking diversity. The LLM will default to its habitual reasoning patterns. Without explicit framework dispatch, you don't get the forced perspective shifts that Doc 11 provides.
- No Cynefin routing means the agent handles a crisis the same way it handles a routine question.
- No diverge-then-converge discipline. The LLM will generate and evaluate simultaneously (premature convergence).
- The "kernel" is just a system prompt, not an active monitoring loop. It can't interrupt mid-reasoning.

### When this is the right choice:
- When you're building a practical assistant, not a research agent
- When latency and cost matter more than reasoning depth
- When the problems are mostly complicated (not complex/chaotic/novel)
- As the foundation you ship first, with Alpha as the upgrade path

---

## Architecture Gamma: "Multi-Agent Cognitive Society"

Each cognitive module is a separate agent with its own LLM instance. Working memory is a shared blackboard. Agents communicate through message passing.

```
┌──────────────────────────────────────────────────┐
│              SHARED BLACKBOARD                    │
│         (Working Memory + Goal Stack)             │
└──┬───┬───┬───┬───┬───┬───┬───┬───┬───┬──────────┘
   │   │   │   │   │   │   │   │   │   │
   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼
 ┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌────┐
 │Att││Sem││Epi││Pro││Cyn││1st││Inv││Pre││Bay││Meta│
 │ent││Mem││Mem││Mem││efn││Pri││ers││Mor││esi││Cog │
 │ion││Agt││Agt││Agt││Agt││Agt││Agt││Agt││Agt││Agt │
 └───┘└───┘└───┘└───┘└───┘└───┘└───┘└───┘└───┘└────┘
```

- **Attention Agent**: Reads sensory input, scores salience, writes high-salience items to blackboard
- **Semantic Memory Agent**: Manages the concept graph, responds to retrieval queries
- **Episodic Memory Agent**: Manages the event log, responds to "have I seen this before?"
- **Cynefin Agent**: Reads the blackboard, classifies the domain, posts the playbook
- **Thinking System Agents** (First Principles, Inversion, Pre-mortem, etc.): Each reads the blackboard, applies its framework, writes insights back
- **Metacognitive Agent**: Monitors the blackboard continuously, posts alerts, can halt or redirect other agents

The orchestrator reads the Cynefin classification and activates the appropriate subset of thinking system agents. They can run in parallel (Six Hats style) or sequential (pipeline style).

### Advantages:
- True diversity of perspective. Each agent has its own system prompt, temperature, and potentially even model. The First Principles agent can be low-temperature and analytical. The Lateral Thinking agent can be high-temperature and creative.
- Natural parallelism. Multiple agents can reason simultaneously.
- Modularity is real, not just architectural — you can add/remove agents without touching others.
- The Metacognitive Agent can actually monitor other agents' outputs in real-time and intervene.
- Maps cleanly to the "society of mind" philosophy from the Cognitive Architecture doc.

### Disadvantages:
- **Expensive.** N agents × M turns = many LLM calls. A single complex problem could cost 20-50 calls.
- **Context synchronization is hard.** The blackboard helps, but agents can step on each other. You need conflict resolution (which is another agent, or a locking mechanism).
- **Latency.** Even with parallelism, the sequential dependencies (classify first, then reason, then evaluate) create a long critical path.
- **Integration quality.** Getting 8 agents to produce a *coherent* final answer is harder than getting 1 agent to reason in 8 ways. You need a Synthesis Agent, and that synthesis itself is non-trivial.
- **Debugging nightmare.** When the system produces a bad answer, which agent failed? The interaction effects make root-cause analysis very hard.

### When this is the right choice:
- High-stakes decisions where quality justifies cost
- Research contexts where you want to study how different reasoning frameworks interact
- When you genuinely need diverse perspectives (not just thorough analysis)

---

## Architecture Delta: "Layered Escalation"

The simplest practical merge. Three tiers of escalating sophistication. Most requests are handled cheaply. Only hard problems get the full treatment.

```
Tier 1 — PROCEDURAL (no LLM)
  Compiled fast-paths, cached responses, pattern matching.
  Handles: greetings, FAQ, repeated questions, simple lookups.
  Cost: ~0. Latency: milliseconds.
       │
       │ (no match or low confidence)
       ▼
Tier 2 — SINGLE LLM CALL (memory-augmented)
  Retrieve episodic + semantic context.
  One LLM call with rich context + metacognitive system prompt.
  Handles: most real questions, analysis, coding, writing.
  Cost: 1 LLM call. Latency: seconds.
       │
       │ (confidence < threshold OR stakes > threshold)
       ▼
Tier 3 — FULL THINKING PIPELINE (Doc 11)
  Cynefin classify → playbook dispatch → N thinking systems.
  Memory-augmented at each step.
  Consolidation after completion.
  Handles: novel problems, high-stakes decisions, complex analysis.
  Cost: 5-15 LLM calls. Latency: 30-120 seconds.
```

The Dual-Process Router from our earlier brainstorm IS the tier selector. System 1 = Tier 1. System 2 light = Tier 2. System 2 deep = Tier 3.

### Advantages:
- **Cost-efficient by default.** 80-90% of requests handled at Tier 1-2.
- **Graceful scaling.** Hard problems automatically get more resources.
- **Incrementally buildable.** Ship Tier 2 first, add Tier 1 caching and Tier 3 depth later.
- **The bounded rationality principle is built into the architecture** — effort is proportional to stakes.
- **Clear debugging.** If something fails, you know which tier handled it.

### Disadvantages:
- The tier boundaries are hard to calibrate. What confidence threshold triggers escalation? What counts as "high stakes"? Getting this wrong means either wasting money on easy problems or under-serving hard ones.
- Tier 1 needs careful curation. Bad fast-paths give wrong answers confidently. You need a mechanism to invalidate cached procedural responses when the world changes.
- Tier 3 is still the full Alpha complexity — you've just gated access to it, not simplified it.
- The jump between tiers can feel jarring. A user might get a fast Tier 1 answer on one turn and wait 60 seconds for Tier 3 on the next.

### When this is the right choice:
- Production systems where cost and latency matter
- When you serve a mix of easy and hard queries
- As the deployment architecture for any of the other designs

---

## Summary Comparison

| | Alpha (Hybrid) | Beta (Simple) | Gamma (Multi-Agent) | Delta (Escalation) |
|---|---|---|---|---|
| **Complexity** | Very High | Medium | Very High | Medium-High |
| **Cost per hard query** | High (5-15 calls) | Low (1-2 calls) | Very High (20-50 calls) | High (gated) |
| **Cost per easy query** | High (unless optimized) | Low | Very High | Very Low |
| **Reasoning quality** | Excellent | Good | Excellent (diverse) | Excellent (when escalated) |
| **Memory sophistication** | Full cognitive substrate | Full cognitive substrate | Full + distributed | Full cognitive substrate |
| **Build effort** | 3-6 months | 1-2 months | 6-12 months | 2-4 months |
| **Best for** | Research, ambitious products | Practical assistants | High-stakes analysis | Production systems |

---

## My Recommendation

**Build Delta with Alpha inside Tier 3.**

- Start with Tier 2 (memory-augmented single LLM call) — this is Beta, and it ships fast
- Add Tier 1 (procedural fast-paths) as usage patterns emerge — compile frequent successful Tier 2 responses into cached fast-paths (this IS the procedural memory automatization from the Cognitive Architecture)
- Add Tier 3 (full thinking pipeline) for when you need it — this is Alpha's orchestration

The memory substrate (working memory, episodic, semantic, consolidation) runs across all tiers. The escalation router is the Dual-Process Router. The learning systems gradually improve the router's calibration and compile successful patterns into Tier 1.

This gives you the cognitive architecture's memory and learning as the persistent foundation, Doc 11's thinking systems as the peak reasoning capability, and bounded rationality as the governor that keeps it practical.

---

That's the right move. It's also what makes this a *toolkit* rather than a monolithic agent framework.

The insight is: nobody needs all 24 thinking systems, all 5 memory tiers, and full consolidation for every use case. Someone building a customer support bot needs procedural memory + episodic memory + maybe Tier 1-2 reasoning. Someone building a research assistant needs semantic memory + episodic memory + the full thinking pipeline. Someone doing competitive analysis needs OODA + Pre-mortem + Second-Order + Bayesian, and nothing else.

The architecture becomes a **parts catalog**, not a blueprint. You compose what you need.

## What Modularity Actually Means Here

Every component has a protocol (interface). You pick implementations. You wire them together through a standard bus.

```python
# You build YOUR agent from parts you choose

agent = CognitiveAgent(
    # Pick your memory systems
    memory=[
        WorkingMemory(capacity=4),
        EpisodicMemory(backend="sqlite", embedding_model="..."),
        SemanticMemory(backend="networkx"),
        # No procedural memory — I don't need fast-paths yet
    ],

    # Pick your kernel monitors
    kernel=[
        MetacognitiveMonitor(),
        BoundedRationalityGovernor(stakes="medium"),
        # No emotional modulation — not relevant for my use case
    ],

    # Pick your reasoning systems
    reasoning=[
        FirstPrinciplesSystem(),
        InversionSystem(),
        PreMortemSystem(),
        BayesianUpdater(),
        # Just 4 out of 24 — that's all I need
    ],

    # Pick your router (or don't — default to single-tier)
    router=DualProcessRouter(
        tier1=ProceduralLookup(),       # optional
        tier2=SingleLLMCall(),           # default
        tier3=ThinkingPipeline(),        # optional
    ),

    # Pick your learning systems
    learning=[
        HebbianLearning(rate=0.01),
        # No RL, no schema learning — keep it simple
    ],

    # Pick your LLM backend
    llm=LLM(model="claude-sonnet-4-5-20250929"),
)
```

Or even simpler for someone who just wants memory-augmented chat:

```python
agent = CognitiveAgent(
    memory=[EpisodicMemory()],
    llm=LLM(model="claude-sonnet-4-5-20250929"),
)
# That's it. Everything else has sensible defaults or is absent.
```

## The Protocol Layer

This is what makes it work. Every module type has a protocol. You can implement the protocol however you want — including wrapping third-party tools.

```python
# Memory protocol — all memory systems implement this
class MemoryStore(Protocol):
    def store(self, item: Representation) -> None: ...
    def retrieve(self, cue: Representation, k: int = 5) -> list[Representation]: ...
    def tick(self) -> None: ...  # time-based maintenance (decay, expiry)

# Reasoning system protocol — all thinking systems implement this
class ReasoningSystem(Protocol):
    name: str
    phases: list[Phase]
    domains: list[Domain]

    def execute(self, problem: str, context: Context) -> SystemOutput: ...

# Kernel monitor protocol — all monitors implement this
class KernelMonitor(Protocol):
    def pre_check(self, context: Context) -> list[Alert]: ...
    def post_check(self, output: SystemOutput, context: Context) -> list[Signal]: ...

# Router protocol — decides which tier handles a request
class Router(Protocol):
    def route(self, input: str, context: Context) -> Tier: ...

# Learning protocol — all learning systems implement this
class LearningSystem(Protocol):
    def update(self, experience: Experience) -> None: ...

# Consolidation protocol
class Consolidator(Protocol):
    def should_run(self, tick: int) -> bool: ...
    def run(self, memories: list[MemoryStore]) -> None: ...
```

Then anybody can create a custom module:

```python
# Custom thinking system — took 10 minutes to write
class RubberDuckSystem:
    name = "rubber_duck"
    phases = [Phase.FRAME]
    domains = [Domain.COMPLICATED, Domain.COMPLEX]

    def execute(self, problem, context):
        return SystemOutput(
            insights=["Explain the problem out loud, step by step, "
                      "as if teaching it to someone who knows nothing."],
            prompt=f"""Explain this problem step by step as if 
            teaching a complete beginner. Often the act of 
            explaining reveals where your understanding breaks down.
            
            PROBLEM: {problem}
            
            Walk through it from the very beginning. Where does 
            the explanation get hard? That's where the real 
            problem is.""",
            confidence=0.6,
        )

# Custom memory backend — wraps an existing vector DB
class PineconeEpisodicMemory:
    """Implements MemoryStore protocol using Pinecone."""
    
    def store(self, item):
        self.index.upsert([(item.id, item.embedding, item.metadata)])
    
    def retrieve(self, cue, k=5):
        results = self.index.query(cue.embedding, top_k=k)
        return [self._to_representation(r) for r in results.matches]

# Custom kernel monitor
class CostMonitor:
    """Tracks spend and halts if budget exceeded."""
    
    def __init__(self, max_usd: float):
        self.max_usd = max_usd
        self.spent = 0.0
    
    def post_check(self, output, context):
        self.spent += output.cost
        if self.spent > self.max_usd:
            return [Signal(type="halt", reason=f"Budget exceeded: ${self.spent:.2f}")]
        return []
```

## What This Gives You

**For the toolkit (`toolkit/`):** The protocols and base implementations. `WorkingMemory`, `EpisodicMemory`, `SemanticMemory`, `MetacognitiveMonitor`, `DualProcessRouter`, `HebbianLearning`, `Consolidator`. These are the building blocks. They work with `LLM` from core.

**For nanope cookbooks (`nanope/`):** Pre-composed agents that demonstrate specific combinations:

| Cookbook | Components Used |
|---|---|
| Research Assistant | Episodic + Semantic memory, First Principles + Bayesian + Socratic, full consolidation |
| Decision Advisor | Working memory, Pre-mortem + Inversion + Second-Order + Probabilistic, emotional modulation |
| Study Buddy | Episodic memory, Spaced Repetition (custom learning system), Socratic |
| Code Reviewer | Procedural memory (compiled patterns), Critical Thinking + Inversion, DualProcess router |
| Strategy Analyst | Full memory stack, OODA + Cynefin + Systems Thinking + Pre-mortem, multi-tier |
| Debate Partner | Working memory, Dialectical + Socratic + Bayesian, no persistence |

**For users:** They compose what they need. They're not locked into "the agent." They pick modules like Lego. They create custom thinking systems in 20 lines. They bring their own storage backend. They share modules as packages.

## What's Actually Hard

Three things:

1. **The `Context` object.** This is the universal representation that flows between all modules. Get this wrong and nothing composes. It needs to be rich enough that any module can read what it needs, but not so coupled that adding a module requires changing the context. The Cognitive Architecture doc's universal representation format is a good starting point but needs simplification for the LLM use case — not everything needs activation levels and emotional arousal.

2. **The orchestration loop.** Even with modular parts, something has to run the cycle: perceive → retrieve → route → reason → learn. That orchestrator needs to be generic enough to work with any combination of modules but smart enough to skip steps when modules are absent. If you have no episodic memory, the retrieval step is a no-op. If you have no router, everything goes to Tier 2.

3. **Memory ↔ LLM interface.** How do you inject retrieved memories into an LLM prompt? How do you extract structured outputs that the learning system can consume? This translation layer between the cognitive substrate and the LLM is where most of the practical difficulty lives. The representation format, the prompt assembly, the output parsing — this needs to be clean.

## What I'd Build, In Order

1. **The protocols** — `MemoryStore`, `ReasoningSystem`, `KernelMonitor`, `Router`, `LearningSystem`, `Consolidator`. Just the interfaces.
2. **The `Context` / `Representation` types** — the data that flows between modules.
3. **The orchestration loop** — the `CognitiveAgent` that wires modules together and runs the cycle.
4. **`EpisodicMemory`** — the single most useful memory system for LLM agents. Gives you conversation history + semantic search over past interactions.
5. **`WorkingMemory`** — the 4-slot focus + goal stack. Forces the agent to be selective about context.
6. **`MetacognitiveMonitor` + `BoundedRationalityGovernor`** — the kernel that makes everything self-aware.
7. **3-5 thinking systems** (First Principles, Inversion, Pre-mortem, Bayesian, Critical Thinking) — enough to demonstrate the value.
8. **`DualProcessRouter`** — the cost/quality optimizer.
9. **Everything else** — remaining memory systems, remaining thinking systems, consolidation, learning, custom backends.

Steps 1-3 are the framework. Steps 4-6 are the first useful agent. Steps 7-8 make it smart. Step 9 is the long tail.

This is also exactly the kind of thing that fits the toolkit's existing pattern — `core/` has the primitives, `toolkit/` has the composed utilities, `nanope/` has the cookbooks. The cognitive modules would be a new layer that builds on both.

---

The ultimate agent is Delta architecture (layered escalation) with every module slot filled, all three tiers active, full learning loop, and consolidation running — a system that gets smarter over time, knows when to think fast vs. slow, remembers everything it needs and forgets what it doesn't, and can reason from 24 different angles when the problem demands it.

---

## The Ultimate Agent

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         THE ULTIMATE AGENT                              │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  INPUT GATE                                                       │  │
│  │                                                                   │  │
│  │  Sensory Buffer ← raw input (text, images, data, tool results)   │  │
│  │       │                                                           │  │
│  │       ▼                                                           │  │
│  │  Attention / Salience Scorer                                      │  │
│  │    ├── Novelty: how different from recent inputs?                 │  │
│  │    ├── Goal relevance: does this relate to active goals?          │  │
│  │    ├── Emotional significance: does this match known triggers?    │  │
│  │    ├── Prediction error: did I expect this?                       │  │
│  │    └── Urgency: is there a time constraint?                       │  │
│  │       │                                                           │  │
│  │       ▼                                                           │  │
│  │  Salience > threshold? ──NO──► discard (never reaches cognition)  │  │
│  │       │YES                                                        │  │
│  │       ▼                                                           │  │
│  │  Working Memory: attempt to load                                  │  │
│  │    ├── Slot available → load                                      │  │
│  │    └── Full → displace lowest-activation item                     │  │
│  │              └── displaced item → episodic trace                  │  │
│  └───────────────────────────┬───────────────────────────────────────┘  │
│                              │                                          │
│  ┌───────────────────────────▼───────────────────────────────────────┐  │
│  │  KERNEL (runs every turn, cannot be disabled)                     │  │
│  │                                                                   │  │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │  │
│  │  │ Metacognitive    │  │ Dual-Process     │  │ Bounded         │  │  │
│  │  │ Monitor          │  │ Classifier       │  │ Rationality     │  │  │
│  │  │                  │  │                  │  │ Governor        │  │  │
│  │  │ • Am I stuck?    │  │ • System 1 or 2? │  │ • Stakes level? │  │  │
│  │  │ • Confidence?    │  │ • Familiar?      │  │ • Budget left?  │  │  │
│  │  │ • Bias risk?     │  │ • Routine?       │  │ • Good enough?  │  │  │
│  │  │ • Strategy ok?   │  │ • Automatic?     │  │ • Stop or go?   │  │  │
│  │  └────────┬────────┘  └────────┬─────────┘  └────────┬────────┘  │  │
│  │           │                    │                      │           │  │
│  │  ┌────────▼────────┐  ┌───────▼──────────┐  ┌───────▼────────┐  │  │
│  │  │ Emotional State │  │ Prediction       │  │ Confidence     │  │  │
│  │  │ Tracker         │  │ Engine           │  │ Calibrator     │  │  │
│  │  │                 │  │                  │  │                │  │  │
│  │  │ valence ±1.0    │  │ "What do I       │  │ Tracks         │  │  │
│  │  │ arousal 0–1.0   │  │  expect to       │  │ prediction vs  │  │  │
│  │  │ modulates       │  │  happen next?"   │  │ outcome over   │  │  │
│  │  │ encoding +      │  │                  │  │ time. Am I     │  │  │
│  │  │ retrieval +     │  │ Prediction error │  │ overconfident?  │  │  │
│  │  │ attention       │  │ drives learning  │  │ Underconfident?│  │  │
│  │  └─────────────────┘  └──────────────────┘  └────────────────┘  │  │
│  │                                                                   │  │
│  │  KERNEL OUTPUT: tier_decision, confidence, alerts, emotional_state│  │
│  └───────────────────────────────┬───────────────────────────────────┘  │
│                                  │                                      │
│  ┌───────────────────────────────▼───────────────────────────────────┐  │
│  │  DUAL-PROCESS ROUTER                                              │  │
│  │                                                                   │  │
│  │  Based on: familiarity, stakes, confidence, complexity, novelty   │  │
│  │                                                                   │  │
│  │  ┌─────────────┐   ┌──────────────┐   ┌────────────────────┐     │  │
│  │  │  TIER 1     │   │  TIER 2      │   │  TIER 3            │     │  │
│  │  │  System 1   │   │  System 2    │   │  System 2 Deep     │     │  │
│  │  │  Fast       │   │  Standard    │   │  Full Pipeline     │     │  │
│  │  │             │   │              │   │                    │     │  │
│  │  │  No LLM     │   │  1 LLM call  │   │  5-15 LLM calls   │     │  │
│  │  │  ~0 cost    │   │  ~$0.01-0.05 │   │  ~$0.10-1.00      │     │  │
│  │  │  <100ms     │   │  1-5 sec     │   │  30-120 sec        │     │  │
│  │  └──────┬──────┘   └──────┬───────┘   └────────┬───────────┘     │  │
│  └─────────┼─────────────────┼────────────────────┼─────────────────┘  │
│            │                 │                     │                    │
│            ▼                 ▼                     ▼                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                 │   │
│  │                    TIER 1: PROCEDURAL                            │   │
│  │                                                                 │   │
│  │  Compiled fast-paths. No LLM involved.                          │   │
│  │                                                                 │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │ Pattern Match against Procedural Memory                  │   │   │
│  │  │   ├── Cached responses (exact or fuzzy match)            │   │   │
│  │  │   ├── Condition-action rules (if X then Y)               │   │   │
│  │  │   ├── Compiled FSMs (multi-step automatic procedures)    │   │   │
│  │  │   └── Lookup tables (instant, zero-thought responses)    │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  │                                                                 │   │
│  │  Confidence check: is the match good enough?                    │   │
│  │    ├── YES → return response. Log to episodic. Update stats.   │   │
│  │    └── NO  → escalate to Tier 2.                                │   │
│  │                                                                 │   │
│  └──────────────────────────────┬──────────────────────────────────┘   │
│                                 │ (escalate)                           │
│  ┌──────────────────────────────▼──────────────────────────────────┐   │
│  │                                                                 │   │
│  │                    TIER 2: AUGMENTED LLM                        │   │
│  │                                                                 │   │
│  │  Single LLM call with rich memory-assembled context.            │   │
│  │                                                                 │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │ CONTEXT ASSEMBLY                                         │   │   │
│  │  │                                                          │   │   │
│  │  │ 1. Working Memory contents (current focus + goal stack)  │   │   │
│  │  │ 2. Episodic retrieval: "Have I seen this before?"        │   │   │
│  │  │    └── top-k similar past interactions + outcomes        │   │   │
│  │  │ 3. Semantic retrieval: "What do I know about this?"      │   │   │
│  │  │    └── spreading activation → primed concepts            │   │   │
│  │  │    └── relevant facts, relationships, schemas            │   │   │
│  │  │ 4. Emotional context: valence/arousal of related items   │   │   │
│  │  │ 5. Metacognitive header: confidence, alerts, budget      │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  │                          │                                      │   │
│  │                          ▼                                      │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │ SYSTEM PROMPT (kernel baked in)                          │   │   │
│  │  │                                                          │   │   │
│  │  │ "You are a cognitive agent with access to your memories. │   │   │
│  │  │  Your working memory, relevant past experiences, and     │   │   │
│  │  │  known facts are provided below. Monitor your own        │   │   │
│  │  │  reasoning quality. Flag when uncertain. If this problem │   │   │
│  │  │  needs deeper analysis than a single response, say so."  │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  │                          │                                      │   │
│  │                          ▼                                      │   │
│  │                    [ LLM CALL ]                                  │   │
│  │                          │                                      │   │
│  │                          ▼                                      │   │
│  │  Confidence check: does the LLM's self-assessed confidence      │   │
│  │  meet threshold? Does it request escalation?                    │   │
│  │    ├── YES → return response. Learn. Log.                       │   │
│  │    └── NO  → escalate to Tier 3.                                │   │
│  │                                                                 │   │
│  └──────────────────────────────┬──────────────────────────────────┘   │
│                                 │ (escalate)                           │
│  ┌──────────────────────────────▼──────────────────────────────────┐   │
│  │                                                                 │   │
│  │              TIER 3: FULL THINKING PIPELINE                     │   │
│  │                                                                 │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │ PHASE 0 — ORIENT                                         │   │   │
│  │  │                                                          │   │   │
│  │  │ Cynefin Classifier → domain (clear/complicated/          │   │   │
│  │  │                       complex/chaotic/novel)              │   │   │
│  │  │ Latticework Selector → relevant cross-domain models      │   │   │
│  │  │ OODA Conductor → initialize observation cycle            │   │   │
│  │  │                                                          │   │   │
│  │  │ OUTPUT: domain, playbook, mental_models, initial orient  │   │   │
│  │  └──────────────────────────┬───────────────────────────────┘   │   │
│  │                             │                                   │   │
│  │  ┌──────────────────────────▼───────────────────────────────┐   │   │
│  │  │ PHASE 1 — PERCEIVE                                       │   │   │
│  │  │                                                          │   │   │
│  │  │ Deep memory retrieval:                                   │   │   │
│  │  │   Episodic: all relevant past experiences                │   │   │
│  │  │   Semantic: full subgraph of related knowledge           │   │   │
│  │  │   Procedural: any existing procedures for this domain    │   │   │
│  │  │ Systems Thinking: map the key variables and connections  │   │   │
│  │  │                                                          │   │   │
│  │  │ OUTPUT: enriched context, system map, knowledge gaps     │   │   │
│  │  └──────────────────────────┬───────────────────────────────┘   │   │
│  │                             │                                   │   │
│  │  ┌──────────────────────────▼───────────────────────────────┐   │   │
│  │  │ PHASE 2 — FRAME                                          │   │   │
│  │  │                                                          │   │   │
│  │  │ Selected by Cynefin domain:                              │   │   │
│  │  │   Novel → First Principles + Socratic Method             │   │   │
│  │  │   Complex → Design Thinking + Sensemaking                │   │   │
│  │  │   Complicated → Systems Thinking + ToC                   │   │   │
│  │  │   Chaotic → (skip framing, go to DECIDE)                 │   │   │
│  │  │                                                          │   │   │
│  │  │ Inversion bookend: "What would make this WORSE?"         │   │   │
│  │  │                                                          │   │   │
│  │  │ OUTPUT: reframed problem, assumptions identified,        │   │   │
│  │  │         bedrock truths, inverted failure modes            │   │   │
│  │  └──────────────────────────┬───────────────────────────────┘   │   │
│  │                             │                                   │   │
│  │  ┌──────────────────────────▼───────────────────────────────┐   │   │
│  │  │ PHASE 3 — GENERATE                                       │   │   │
│  │  │                                                          │   │   │
│  │  │ Divergent Thinking: quantity over quality, no judgment    │   │   │
│  │  │ Lateral Thinking: random provocation, pattern breaking   │   │   │
│  │  │ Latticework: "What would biology/physics/econ suggest?"  │   │   │
│  │  │ Dialectical: can opposing approaches be synthesized?     │   │   │
│  │  │ First Principles: build novel solutions from truths      │   │   │
│  │  │                                                          │   │   │
│  │  │ Can run in PARALLEL — multiple LLM calls simultaneously  │   │   │
│  │  │                                                          │   │   │
│  │  │ OUTPUT: 10-20+ candidate solutions from diverse angles   │   │   │
│  │  └──────────────────────────┬───────────────────────────────┘   │   │
│  │                             │                                   │   │
│  │  ┌──────────────────────────▼───────────────────────────────┐   │   │
│  │  │ PHASE 4 — EVALUATE                                       │   │   │
│  │  │                                                          │   │   │
│  │  │ Convergent Thinking: filter against criteria              │   │   │
│  │  │ Critical Thinking: evidence? assumptions? fallacies?      │   │   │
│  │  │ Bayesian Updater: update confidence given evidence        │   │   │
│  │  │ Second-Order Thinking: "and then what?" ×3               │   │   │
│  │  │ Pre-mortem: "This failed. Why?"                          │   │   │
│  │  │ Inversion bookend: "How could this solution backfire?"   │   │   │
│  │  │                                                          │   │   │
│  │  │ Dialectical synthesis if systems conflict                 │   │   │
│  │  │                                                          │   │   │
│  │  │ OUTPUT: ranked solutions, confidence, failure modes,      │   │   │
│  │  │         mitigations, residual uncertainty                 │   │   │
│  │  └──────────────────────────┬───────────────────────────────┘   │   │
│  │                             │                                   │   │
│  │  ┌──────────────────────────▼───────────────────────────────┐   │   │
│  │  │ PHASE 5 — DECIDE                                         │   │   │
│  │  │                                                          │   │   │
│  │  │ Bounded Rationality check: good enough to act?            │   │   │
│  │  │ Theory of Constraints: where's the bottleneck?            │   │   │
│  │  │ OODA: commit and execute. Treat as hypothesis, not truth.│   │   │
│  │  │                                                          │   │   │
│  │  │ If confidence still too low AND budget remains:           │   │   │
│  │  │   → LOOP BACK to Phase 2 or 3 (cycle, don't waterfall)  │   │   │
│  │  │                                                          │   │   │
│  │  │ OUTPUT: selected solution, rationale, action plan         │   │   │
│  │  └──────────────────────────┬───────────────────────────────┘   │   │
│  │                             │                                   │   │
│  │  ┌──────────────────────────▼───────────────────────────────┐   │   │
│  │  │ PHASE 6 — REFLECT                                        │   │   │
│  │  │                                                          │   │   │
│  │  │ Metacognitive reflection:                                │   │   │
│  │  │   What was the most valuable thinking system used?       │   │   │
│  │  │   Where should the approach have shifted?                │   │   │
│  │  │   What remains uncertain?                                │   │   │
│  │  │   What should be explored next?                          │   │   │
│  │  │                                                          │   │   │
│  │  │ Bayesian update on the agent's own strategy priors       │   │   │
│  │  │ OODA: feed results back as observations for next cycle   │   │   │
│  │  │                                                          │   │   │
│  │  │ OUTPUT: reflection, updated confidence, strategy notes    │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  OUTPUT GATE                                                      │  │
│  │                                                                   │  │
│  │  Assemble final response from whichever tier handled it.          │  │
│  │  Include: answer, confidence, reasoning trace (if requested),     │  │
│  │           open questions, suggested next steps.                   │  │
│  │                                                                   │  │
│  │  Quality check: does the response actually answer the question?   │  │
│  │  Coherence check: is the response internally consistent?          │  │
│  │  If either fails → retry at same or higher tier.                  │  │
│  └───────────────────────────────┬───────────────────────────────────┘  │
│                                  │                                      │
│  ┌───────────────────────────────▼───────────────────────────────────┐  │
│  │  LEARNING (runs after every interaction)                          │  │
│  │                                                                   │  │
│  │  ┌────────────────────────────────────────────────────────────┐   │  │
│  │  │ FAST LEARNING (every turn)                                 │   │  │
│  │  │                                                            │   │  │
│  │  │ Episodic: log the full interaction                         │   │  │
│  │  │   (input, tier used, systems invoked, output, outcome,     │   │  │
│  │  │    confidence, emotional valence, user feedback if any)    │   │  │
│  │  │                                                            │   │  │
│  │  │ Hebbian: strengthen edges between concepts that were       │   │  │
│  │  │   co-active during this interaction                        │   │  │
│  │  │                                                            │   │  │
│  │  │ RL: update value estimates                                 │   │  │
│  │  │   if user feedback → reward signal                         │   │  │
│  │  │   if goal achieved → positive reward                       │   │  │
│  │  │   if confidence was well-calibrated → positive reward      │   │  │
│  │  │   reward updates: router quality, system selection,        │   │  │
│  │  │                    tier selection                           │   │  │
│  │  │                                                            │   │  │
│  │  │ Procedural compilation check:                              │   │  │
│  │  │   Has this exact pattern been handled 5+ times at Tier 2?  │   │  │
│  │  │   Are the responses consistent?                            │   │  │
│  │  │   → Compile into Tier 1 fast-path.                         │   │  │
│  │  │   (This is automatization — deliberate → automatic)        │   │  │
│  │  └────────────────────────────────────────────────────────────┘   │  │
│  │                                                                   │  │
│  │  ┌────────────────────────────────────────────────────────────┐   │  │
│  │  │ SLOW LEARNING (periodic — every N interactions, "sleep")   │   │  │
│  │  │                                                            │   │  │
│  │  │ Episodic replay:                                           │   │  │
│  │  │   Select important recent memories (high reward,           │   │  │
│  │  │   high prediction error, high emotional arousal)           │   │  │
│  │  │   Replay through predictive model → update predictions     │   │  │
│  │  │   Strengthen associations activated during replay          │   │  │
│  │  │                                                            │   │  │
│  │  │ Schema extraction:                                         │   │  │
│  │  │   Cluster similar episodic memories                        │   │  │
│  │  │   Extract common patterns → new schemas in semantic memory │   │  │
│  │  │   "When users ask about X, they usually need Y"            │   │  │
│  │  │   "Problems in domain D usually benefit from system S"     │   │  │
│  │  │                                                            │   │  │
│  │  │ Memory pruning:                                            │   │  │
│  │  │   Decay low-access, low-value episodic memories            │   │  │
│  │  │   Compress old episodes into summaries                     │   │  │
│  │  │   Weaken unused semantic edges                             │   │  │
│  │  │   Remove procedural rules with low success rates           │   │  │
│  │  │                                                            │   │  │
│  │  │ Index rebuilding:                                          │   │  │
│  │  │   Recompute embeddings for changed concepts                │   │  │
│  │  │   Rebuild ANN indices for similarity search                │   │  │
│  │  │   Recompute graph centrality for semantic memory           │   │  │
│  │  │                                                            │   │  │
│  │  │ Router calibration:                                        │   │  │
│  │  │   Was Tier 1 accurate? (check against Tier 2 spot-checks) │   │  │
│  │  │   Was Tier 2 sufficient? (check escalation patterns)       │   │  │
│  │  │   Was Tier 3 worth it? (compare quality vs cost)           │   │  │
│  │  │   Adjust routing thresholds                                │   │  │
│  │  │                                                            │   │  │
│  │  │ Meta-learning:                                             │   │  │
│  │  │   Which thinking systems produced the best outcomes?       │   │  │
│  │  │   Which combinations worked well together?                 │   │  │
│  │  │   Update playbook weights — the agent learns which         │   │  │
│  │  │   frameworks work for which domains over time              │   │  │
│  │  └────────────────────────────────────────────────────────────┘   │  │
│  │                                                                   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  MEMORY SUBSTRATE (persistent across all interactions)            │  │
│  │                                                                   │  │
│  │  ┌─────────────┐                                                  │  │
│  │  │  WORKING     │  4-slot focus (priority queue)                  │  │
│  │  │  MEMORY      │  Goal stack (LIFO, depth-limited)               │  │
│  │  │              │  Scratchpad (intermediate reasoning artifacts)   │  │
│  │  │              │  Cleared each turn, rebuilt from retrieval       │  │
│  │  └─────────────┘                                                  │  │
│  │                                                                   │  │
│  │  ┌─────────────┐                                                  │  │
│  │  │  EPISODIC    │  Every interaction logged with full context      │  │
│  │  │  MEMORY      │  Vector embeddings for similarity search        │  │
│  │  │              │  Temporal index for time-range queries           │  │
│  │  │              │  Emotional valence tags for mood-congruent recall│  │
│  │  │              │  Reconsolidation: retrieved memories can update  │  │
│  │  └─────────────┘                                                  │  │
│  │                                                                   │  │
│  │  ┌─────────────┐                                                  │  │
│  │  │  SEMANTIC    │  Weighted directed graph of concepts             │  │
│  │  │  MEMORY      │  Spreading activation for associative retrieval │  │
│  │  │              │  Embeddings for similarity fallback              │  │
│  │  │              │  Schemas (frames with slots + defaults)          │  │
│  │  │              │  Taxonomies (is-a hierarchies)                   │  │
│  │  │              │  Causal models (causes/enables/prevents edges)   │  │
│  │  └─────────────┘                                                  │  │
│  │                                                                   │  │
│  │  ┌─────────────┐                                                  │  │
│  │  │  PROCEDURAL  │  Compiled fast-paths (Tier 1 cache)             │  │
│  │  │  MEMORY      │  Condition-action rules (if X then Y)           │  │
│  │  │              │  FSMs for multi-step procedures                  │  │
│  │  │              │  Success rates tracked per rule                  │  │
│  │  │              │  Automatization: Tier 2 patterns → Tier 1 rules │  │
│  │  └─────────────┘                                                  │  │
│  │                                                                   │  │
│  │  ┌─────────────┐                                                  │  │
│  │  │  EMOTIONAL   │  Valence-arousal state (continuous 2D vector)   │  │
│  │  │  MEMORY      │  Somatic markers (context → gut feeling)        │  │
│  │  │              │  Modulates: encoding depth, retrieval bias,     │  │
│  │  │              │    attention priority, decision weighting        │  │
│  │  └─────────────┘                                                  │  │
│  │                                                                   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  TOOL INTERFACE                                                   │  │
│  │                                                                   │  │
│  │  The agent can use external tools at any tier:                    │  │
│  │    Tier 1: tools with cached/compiled execution patterns          │  │
│  │    Tier 2: LLM decides which tools to call                       │  │
│  │    Tier 3: thinking systems can invoke tools as part of their     │  │
│  │            reasoning (e.g., search during Perceive phase,         │  │
│  │            calculate during Evaluate phase)                       │  │
│  │                                                                   │  │
│  │  Tool results flow back through the Input Gate as new percepts.   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## What Makes This "Ultimate"

It's not the size — it's the **seven properties that no single-layer agent has**:

**1. It gets smarter over time.**
Every interaction updates episodic memory, strengthens semantic associations, calibrates the router, and occasionally compiles new fast-paths. The agent on day 100 is measurably better than the agent on day 1 — not because the LLM improved, but because the cognitive substrate learned.

**2. It knows when to think fast and when to think slow.**
The Dual-Process Router means 80-90% of requests are handled at Tier 1-2 (fast, cheap). Only genuinely hard problems trigger the full pipeline. And the router itself improves over time through RL — it learns which problems actually need deep thinking.

**3. It remembers selectively.**
Not everything is stored equally. High-emotional-arousal events get richer encoding. Frequently-accessed knowledge gets stronger connections. Rarely-used memories decay and eventually get pruned. The agent doesn't drown in its own history — it curates.

**4. It reasons from multiple angles, but only when it matters.**
Tier 3 deploys different thinking systems for different problem types, runs them in the right sequence, synthesizes conflicts dialectically, and stress-tests solutions before committing. But it only does this when the kernel says the stakes warrant it.

**5. It monitors its own reasoning.**
The metacognitive kernel catches: autopilot responses on hard problems, overconfidence, analysis paralysis, premature convergence, budget exhaustion. It can interrupt, redirect, or halt at any point.

**6. It compiles habits.**
Successful patterns at Tier 2 gradually become Tier 1 fast-paths. This is the automatization pathway from the Cognitive Architecture — deliberate reasoning becomes automatic skill. The agent literally builds its own System 1 from repeated System 2 experience.

**7. It consolidates offline.**
Periodic "sleep" cycles replay important memories, extract schemas, prune waste, rebuild indices, and calibrate the router. This is where the deep integration between episodes happens — where individual experiences become general knowledge.

---

## The One-Sentence Version

An agent with **a brain that learns** (memory substrate + learning systems), **a mind that monitors itself** (kernel), **eyes that focus** (attention + salience), **intuition for routine and deliberation for novelty** (dual-process router), **24 specialized reasoning tools it deploys strategically** (thinking systems), and **a sleep cycle that consolidates everything** (consolidation) — all modular, all replaceable, all optional except the parts you need.