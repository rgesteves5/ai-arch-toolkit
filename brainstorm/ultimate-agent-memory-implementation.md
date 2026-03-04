# Ultimate Agent — Memory Implementation

Design options for the memory subsystem of the cognitive agent architecture. Companion to `ultimate_agents.md`.

---

## The Six Memory Systems

| # | Memory | Structure | Role |
|---|--------|-----------|------|
| 1 | **Sensory Buffer** | Circular buffer | Raw input queue, expires in 2-3 ticks |
| 2 | **Working Memory** | Priority queue (4 slots) + goal stack + scratchpad | Active focus, rebuilt each turn with carry-over |
| 3 | **Semantic Memory** | Weighted directed graph | Concepts, relations, schemas, taxonomies |
| 4 | **Episodic Memory** | Event log + vector store + temporal index | Timestamped interactions with embeddings |
| 5 | **Procedural Memory** | Condition-action rules + FSMs + decision trees | Compiled fast-paths, automatic behaviors |
| 6 | **Emotional Memory** | Overlay (valence-arousal annotations) | Metadata on nodes/edges of other memories |

Of the six: **four have a graph dimension** (semantic, episodic, procedural, emotional as overlay). One is linear/temporal (sensory buffer). One is composed of bounded structures (working memory).

---

## Graph Representations by Memory Type

### Semantic Memory — The Primary Graph

The direct, unambiguous mapping. Semantic memory IS a graph.

- **Nodes** = concepts
- **Edges** = directed, weighted relations

```
dog --is_a--> animal           (taxonomy)
dog --has_property--> loyal    (property)
fire --causes--> smoke         (causality)
hammer --used_for--> nailing   (function)
Paris --located_in--> France   (spatial)
king --opposite_of--> subject  (opposition)
atom --analogous_to--> solar_system  (analogy)
```

Edge weights = association strength. Strengthened by Hebbian learning (co-activation).

**Key operations:**
- Spreading activation (weighted BFS — thinking about one concept activates related concepts)
- Taxonomic reasoning (property inheritance along `is_a` edges)
- Indirect relation discovery (shortest path between distant concepts reveals non-obvious connections)

**For an LLM agent:** The LLM already has implicit semantic memory in its weights — but it's static and opaque. The explicit graph stores what the agent learned during operation: user preferences, project-specific facts, discovered relationships, knowledge the LLM doesn't have. It's the agent's *personal* semantic memory, not the generic one.

### Episodic Memory — Temporal Graph

Primarily a temporal log, but with an important graph dimension.

**Intra-episodic edges:**
```
episode_42 --followed_by--> episode_43      (temporal sequence)
episode_42 --similar_to--> episode_17       (similarity)
episode_42 --caused--> episode_45           (causality)
episode_42 --contradicts--> episode_30      (contradiction)
```

**Cross-layer edges (to semantic graph):**
```
episode_42 --involves--> concept_dog        (concept reference)
episode_42 --occurred_at--> location_park   (context)
episode_42 --used_framework--> first_principles  (reasoning trace)
```

This creates a **bipartite graph** between episodes and concepts. Embedding-based retrieval finds *similar* episodes. Graph-based retrieval finds *related* episodes — causally or temporally linked but not necessarily semantically similar. The two retrieval modes are complementary.

**Implementation:** Event log + vector store for primary retrieval, graph edges for inter-episode and episode-to-concept links.

### Procedural Memory — State Graphs (FSMs)

Finite state machines are directed graphs:

- **Nodes** = states
- **Edges** = transitions (with conditions and actions)

```
state: receive_request
  --[has_receipt]--> state: verify_purchase
  --[no_receipt]--> state: request_receipt

state: verify_purchase
  --[within_30_days]--> state: approve_refund
  --[over_30_days]--> state: offer_store_credit
```

Decision trees are also graphs (acyclic). Condition-action rules form a bipartite graph between conditions and actions.

**Inter-procedural edges:**
```
procedure_A --requires--> procedure_B       (dependency)
procedure_C --alternative_to--> procedure_D (alternatives)
```

This creates a **skill graph** that enables task decomposition by the planner.

### Emotional Memory — Overlay on All Graphs

Not a standalone graph. Metadata on nodes and edges of other memories.

**On semantic nodes:**
```
concept_spider → {valence: -0.7, arousal: 0.8}   (negative, high intensity)
concept_beach  → {valence: +0.6, arousal: 0.3}   (positive, relaxing)
```

**On episodic nodes:**
```
episode_42 → {valence: -0.9, arousal: 0.95}      (very negative, very intense)
```

**Somatic markers (special edges):**
```
decision_context_hash --somatic_marker {valence: -0.6}-->
  (gut feeling: "this won't go well")
```

Emotions don't exist in isolation — they exist associated with concepts and experiences.

### Working Memory — Not a Graph

The exception. Priority queue (4 focus slots) + stack (goal stack) + scratchpad (temporaries). Linear structures, not graphs.

But working memory constantly reads from and writes to graphs. Its contents come from the semantic graph (via spreading activation) and episodic graph (via cue-based retrieval). What's in working memory determines which parts of the graphs are activated.

### The Unified View: Multi-Layer Graph

```
LAYER 1: Semantic Graph (concepts + relations)
    │
    │  cross-layer edges: concept ←→ episode
    │
LAYER 2: Episodic Graph (episodes + sequences + similarities)
    │
    │  cross-layer edges: episode ←→ procedure
    │
LAYER 3: Procedural Graph (skills + FSMs + dependencies)

OVERLAY: Emotional annotations on all layers
```

**Cross-layer edges:**
- Concepts ↔ Episodes: "this concept appeared in this episode"
- Episodes ↔ Procedures: "this procedure was used in this episode"
- Concepts ↔ Procedures: "this procedure applies when this concept is present"

**Example queries this enables:**
- "What procedure should I use when concept X is present?" → concept→procedure edges
- "When did I last use procedure Y and what was the outcome?" → procedure→episode edges
- "Which concepts are associated with negative outcomes?" → filter semantic nodes by emotional overlay
- "Which episodes involve concept X and had positive results?" → concept→episode edges + reward filter

---

## Implementation Approaches

### Approach 1 — Unified Graph

All four graph-dimension memories in a single graph. Nodes have a `type` field (`concept`, `episode`, `procedure`, `emotion_tag`). Edges have a `type` field (`is_a`, `causes`, `followed_by`, `involves`, etc.).

```python
memories = {
    "sensory":   CircularBuffer(capacity=20),
    "working":   WorkingMemory(capacity=4),
    "knowledge": UnifiedGraph(),  # semantic + episodic + procedural + emotional
}
```

**Advantages:**
- Cross-layer queries are trivial — single graph traversal, no joins
- One index, one query language, one backend
- Spreading activation crosses layers naturally — activating a concept can activate related episodes and procedures in one operation
- Consolidation operates on one structure — schema extraction, pruning, PageRank, all in one place
- Emotional overlay is node metadata — no separate store
- Simplest to implement initially

**Disadvantages:**
- Graph grows fast — every turn generates at least 1 episode, concepts accumulate, procedures multiply
- Different node types have very different access patterns — episodes by timestamp and similarity, concepts by relation and activation, procedures by pattern matching. One index can't optimize for all
- Pruning is dangerous — removing an episodic node can break edges linking to semantic concepts
- Mixes information of different "speeds" — semantic concepts are stable (change slowly), episodes are fast (one per turn), procedures are medium. Different lifetimes in the same structure complicates garbage collection
- Scaling: when the graph passes thousands to millions of nodes, operations like PageRank and community detection become expensive over the entire graph when you only wanted to recalculate the semantic portion
- Backups and migration are all-or-nothing

**Best for:** Prototypes, small/medium agents, early development, when cross-layer queries are the primary use case.

**Backing store:** NetworkX (in-memory) or Neo4j (persistent).

---

### Approach 2 — Federated Graphs (Separate Stores with Cross-References)

Each memory is an independent graph/store. Cross-references maintained by IDs — an episode stores `involved_concept_ids = ["c_42", "c_17"]`, a concept stores `appeared_in_episodes = ["e_103", "e_207"]`.

```python
memories = {
    "sensory":    CircularBuffer(capacity=20),
    "working":    WorkingMemory(capacity=4),
    "semantic":   SemanticGraph(),         # graph of concepts
    "episodic":   EpisodicGraph(),         # graph of episodes + vector store
    "procedural": ProceduralGraph(),       # graph of FSMs + rules
    "emotional":  EmotionalOverlay(),      # overlay with refs to the other 3
}
```

**Advantages:**
- Each graph optimized for its access pattern — semantic uses spreading activation and PageRank, episodic uses temporal index + ANN similarity, procedural uses pattern matching
- Different backends per memory — semantic in NetworkX (in-memory, fast for medium graphs), episodic in SQLite + FAISS (persistent, good for similarity search), procedural in pure Python structures (fast pattern matching)
- Pruning is safe — deleting an episode doesn't affect the semantic graph (only removes cross-references)
- Independent scaling — if episodic memory grows too large, migrate only that store to a more robust backend
- Different lifecycles — consolidation on episodic (frequent, many episodes) without touching semantic (rare, stable structure)
- Granular backups — backup semantic memory without including thousands of episodes
- Easier to test — each memory is independently testable

**Disadvantages:**
- Cross-layer queries require manual joins — "what procedure works for concept X with positive outcome?" needs: query semantic for X → fetch cross-refs to episodes → filter by outcome → fetch cross-refs to procedures. More code, slower
- Cross-reference consistency — deleting an episode requires cleaning references in concepts and procedures. Without care, you get orphan references
- Spreading activation doesn't cross layers naturally — need separate intra-graph and inter-graph implementations
- More infrastructure complexity — multiple stores, connections, indices to maintain
- Emotional overlay needs to know where to fetch nodes — has references to all other graphs

**Best for:** Production systems, agents that will grow significantly, when different memories have very different performance requirements, teams where different people maintain different memories.

**Backing store:** Mix of NetworkX + SQLite + FAISS + Python structures.

---

### Approach 3 — Partitioned Property Graph

Single graph database backend with logical partitions. Each node has a partition label. Queries can operate within a partition (fast, limited scope) or cross partitions (slower, full scope). Emotional overlay = properties on nodes in any partition.

Physically together, logically separate.

```python
memories = {
    "sensory":   CircularBuffer(capacity=20),
    "working":   WorkingMemory(capacity=4),
    "knowledge": PartitionedPropertyGraph(
                     partitions=["semantic", "episodic", "procedural"],
                     overlays=["emotional"],
                 ),
}
```

**Advantages:**
- Cross-layer queries are possible and relatively easy (single graph), but scope can be limited when not needed
- Partition-scoped operations — "recalculate PageRank only on semantic partition" is natural
- Single backend with flexibility to treat each partition differently
- Partition-specific indexes — temporal index only on episodic, embedding index on all, full-text only on semantic
- Consolidation can operate per-partition or cross-partition as needed
- Neo4j and modern graph databases support labels natively — aligns with existing tools
- Emotional overlay = node properties (`valence: float`, `arousal: float`) — no separate store

**Disadvantages:**
- Requires a real graph database (Neo4j, Memgraph, etc.) — not trivial with pure NetworkX for large graphs
- Logical separation can "leak" — an operation meant to be semantic-only accidentally sweeps episodes if partition filter isn't applied
- Performance depends on partitioning quality — poorly defined partitions lose the advantages
- More complex to maintain than Approach 1, less modular than Approach 2
- Graph database dependency — more infrastructure to manage than pure Python

**Best for:** Medium-to-large scale agents, when you want the best of both worlds, when already using or planning to use a graph database, when cross-layer queries are frequent but intra-layer performance also matters.

**Backing store:** Neo4j or Memgraph.

---

### Approach 4 — Hub-and-Spoke (Semantic Hub + Specialized Stores)

Semantic memory is the central graph (hub). Others are specialized stores (spokes) optimized for their access patterns, with references to the semantic hub. Episodic and procedural are NOT necessarily graphs — they're stores optimized for their specific needs.

Cross-references always pass through the semantic hub.

```python
memories = {
    "sensory":    CircularBuffer(capacity=20),
    "working":    WorkingMemory(capacity=4),
    "semantic":   SemanticGraph(),             # THE HUB — central graph
    "episodic":   EventLog() + VectorStore(),  # SPOKE — optimized for temporal + similarity
    "procedural": RuleEngine() + FSMStore(),   # SPOKE — optimized for pattern matching
    "emotional":  EmotionalOverlay(),          # Overlay on the hub
}
```

**Advantages:**
- Each store uses the optimal data structure for its use case — graphs for conceptual relations, vector stores for similarity, event logs for temporality, rule engines for pattern matching
- Semantic graph as hub gives natural cross-layer queries — everything links through concepts
- Best possible performance for each query type — not forcing similarity search on a graph, nor graph traversal on a vector store
- Modular — can swap any spoke's backend without affecting others
- The most common pattern (concept lookup → related episodes/procedures) is efficient: one hub query, then direct spoke lookups
- Mirrors how the brain actually organizes memory — neocortex (semantic) is the central hub linking hippocampus (episodic), basal ganglia (procedural), and amygdala (emotional)

**Disadvantages:**
- Cross-layer queries that don't pass through the hub are inefficient — "what procedures were used in recent episodes?" requires: episodic → semantic (via involved concepts) → procedural. If no concepts in common, the query fails
- The semantic hub can become a bottleneck — everything passes through it
- More integration complexity — 4 different backends to maintain, each with its own API
- Consistency depends on cross-references being maintained — adding an episode without linking to semantic concepts leaves it "orphaned" from the hub
- Requires more orchestration code — the CognitiveAgent needs to know how to combine results from multiple backends

**Best for:** Production systems needing optimized performance, agents with high episode volume (many interactions), when the primary access pattern is "given a concept, find everything I know about it," architectures where each component can be independent.

**Backing store:** NetworkX + SQLite + FAISS + Python structures.

---

## Comparison Matrix

| Dimension | 1. Unified | 2. Federated | 3. Partitioned | 4. Hub-and-Spoke |
|-----------|-----------|-------------|---------------|-----------------|
| **Initial complexity** | Low | Medium | Medium-High | High |
| **Cross-layer queries** | Trivial | Hard (manual joins) | Easy (with filters) | Easy (via hub) |
| **Intra-layer performance** | Medium (not optimized) | Excellent (each store optimized) | Good (partition-scoped) | Excellent |
| **Scaling** | Poor (everything together) | Excellent (independent) | Good (partition-aware) | Good (spokes independent) |
| **Maintenance** | Simple | Complex (N stores) | Medium (1 store, N partitions) | Complex (N stores + hub) |
| **Consistency** | Guaranteed (1 store) | Hard (manual cross-refs) | Good (1 store) | Medium (refs via hub) |
| **Safe pruning** | Dangerous | Safe (isolated) | Moderate | Moderate |
| **Testability** | Low (everything together) | Excellent (isolated) | Medium | Good |
| **Backing store** | NetworkX / Neo4j | Mix (NX + SQLite + FAISS + Python) | Neo4j / Memgraph | NX + SQLite + FAISS + Python |
| **Cognitive fidelity** | Medium | Low | Medium | High (mirrors brain) |
| **Best phase** | Prototype | Mature production | Medium production | Advanced production |

---

## Recommended Migration Path

```
Approach 1 (Unified Graph)
    │
    │  system grows, bottlenecks appear
    │  (episodic dominates size, need dedicated vector search)
    │
    ▼
Approach 4 (Hub-and-Spoke)
```

**Start with Unified** because it's fastest to implement and cross-layer queries — where the real value lives — are trivial.

**Migrate to Hub-and-Spoke** when the system grows and you hit bottlenecks: the graph gets slow, episodic memory dominates size, you need dedicated vector search. Hub-and-Spoke best mirrors how the brain organizes memory, and each spoke can be independently optimized.

**Alternative paths:**
- `1 → 3 → 4` — if you adopt Neo4j early, Partitioned is a natural stepping stone
- `1 → 2` — if the team is large and different people maintain different memories
- Stay at `1` — if the agent stays small/medium and cross-layer queries are the primary value

---

## Architectural Refinements

### 1. The Context Object (Two-Level Design)

The universal representation flowing between all modules. The most important data design decision in the entire architecture.

**Core Context (mandatory, always present):**
```python
@dataclass(frozen=True, slots=True)
class CoreContext:
    input: str                          # what arrived
    working_memory_contents: tuple      # what's in focus (max 4 items)
    current_goal: str                   # what the agent is trying to do
    confidence: float                   # how certain the agent is
    tick: int                           # internal time (turn number)
```

**Extended Context (optional, added by modules):**
```python
@dataclass(slots=True)
class ExtendedContext:
    retrieved_episodes: list = field(default_factory=list)    # from episodic memory
    primed_concepts: list = field(default_factory=list)       # from semantic memory (spreading activation)
    emotional_state: EmotionalState | None = None             # from emotional memory
    matched_procedures: list = field(default_factory=list)    # from procedural memory
    prediction: str | None = None                             # from predictive model
    prediction_error: float | None = None                     # after comparing with reality
    metacognitive_alerts: list = field(default_factory=list)  # from kernel
```

Each module reads core + what it needs from extended. Each module writes to extended what it produced. No module needs to know all fields — only what it consumes and produces. Extensible without modifying core.

### 2. Working Memory Carry-Over

Working memory should NOT be "cleared each turn." Three behaviors:

**Intra-turn:** 4 slots, priority queue, displacement. Each reasoning step reads and writes. Chain-of-Thought is literally a sequence of working memory states.

**Inter-turn carry-over:** At end of each turn, highest-activation items are marked "carry-over." At start of next turn, before any input, these items are pre-loaded into focus. Gives the agent continuity of thought — "I was thinking about X" persists.

**Decay between turns:** If many turns pass without an item being reactivated, it drops from carry-over. The agent "forgets" what it was thinking about if it's no longer relevant.

**For the LLM:** carry-over translates to key concepts included in the system prompt or beginning of the context window for the next turn. Not the entire prior conversation — the 3-4 most activated items that define "what the agent is focused on now."

### 3. Reasoning-Conditioned Encoding

How the agent reasoned should be stored alongside what happened. When the agent uses First Principles in Tier 3, the episodic trace should contain the decomposition into fundamental truths and the reconstruction. When it uses Pre-mortem, the identified failure modes. When it uses Inversion, the inverted problem and insights.

The `Experience` recorded in episodic memory should include:

```python
@dataclass(frozen=True, slots=True)
class ReasoningTrace:
    thinking_systems_used: tuple[str, ...]        # which frameworks were invoked
    reasoning_structure: str                       # linear, tree, dialectical, etc.
    key_insights: tuple[str, ...]                  # most valuable insights per framework
    framework_outcome: dict[str, float]            # which framework contributed most to the solution
```

This enables meta-learning about thinking processes — the agent learns not just "what works" but "how to think about what works."

### 4. LLM-Powered Consolidation

Consolidation should use the LLM, not just mechanical operations.

**Schema extraction:** Pass a batch of similar episodes to the LLM: "These 8 episodes share a pattern. What's the general principle? What rule can I extract?" The LLM excels at finding abstract patterns in concrete examples. One consolidation LLM call producing a high-quality schema is worth more than 100 runtime calls.

**Procedural rule extraction:** Ask the LLM to analyze WHY a pattern works and under what conditions it wouldn't. Produces more robust rules with more precise conditions.

**Contradiction resolution:** When contradictory memories exist, the LLM can analyze and decide which is more likely correct based on context.

**Cost:** A few LLM calls per consolidation cycle (which runs rarely — every N turns). Benefit: permanent improvement to the entire knowledge system.

### 5. Emotional Signal Sources

For an LLM agent, concrete sources of emotional signals:

| Source | Signal Type | Example |
|--------|------------|---------|
| **Explicit user feedback** | Direct reward | Thumbs up/down, "this was helpful" |
| **Implicit user feedback** | Inferred reward | Reformulated question (negative), followed recommendation (positive), abandoned conversation (ambiguous), follow-up question (positive — engagement) |
| **Self-assessment** | Confidence signal | LLM's confidence in its own response |
| **Historical calibration** | Accuracy signal | When agent said "90% confident" — did it get it right? |
| **Prediction errors** | Arousal driver | Difference between predicted and actual outcome. Large errors → high arousal → stronger encoding |

These aren't "simulated emotions" — they're empirical quality indicators organized as rapid assessments.

### 6. Router Learning Loop

After each interaction, record: `(input_features, tier_chosen, outcome_quality, cost)`.

Over time the router accumulates data to learn: "inputs with these characteristics were well-served at Tier 1" or "inputs with these characteristics needed Tier 3."

**Implementations (simple → sophisticated):**
- Logistic classifier trained on input features
- Multi-armed bandit balancing exploration (test if an input can be answered at a lower tier) vs exploitation (use the tier that historically works)
- Small neural network on feature embeddings

Result: the router self-calibrates — routes with more precision over time, wastes less on easy problems, escalates more aggressively for hard ones.

### 7. Memory-Informed Framework Selection

The Cynefin Classifier in Tier 3 selects frameworks based on problem domain. But the agent's memory should influence this decision.

Example: the agent faces a product design problem. Cynefin says "complex" → default playbook. But episodic memory has 15 past product design episodes, and in 12 of them, Design Thinking + First Principles + Pre-mortem produced the best results. The playbook should be personalized by experience, not fixed by domain.

This is the feedback loop between episodic memory and the Tier 3 pipeline that transforms the agent from a generic playbook executor into an expert that knows, from experience, which frameworks work for which problems.

### 8. Pattern Separation for Similar Memories

When episodic memory accumulates many similar episodes, they interfere with each other during retrieval — the agent gets 5 nearly identical results instead of 5 diverse ones.

**During consolidation:** very similar episodes should be either merged into one (compression) or have their embeddings artificially pushed apart (to preserve distinction). Without this, episodic memory degrades as it grows.

Inspired by the dentate gyrus in the brain, which performs pattern separation to ensure new memories are stored distinctly from existing similar ones.

---

## Build Order

1. **Protocols** — `MemoryStore`, `ReasoningSystem`, `KernelMonitor`, `Router`, `LearningSystem`, `Consolidator`
2. **Context / Representation types** — `CoreContext`, `ExtendedContext`, `ReasoningTrace`, `Experience`
3. **Orchestration loop** — `CognitiveAgent` that wires modules and runs the cycle
4. **Episodic Memory** — most useful memory for LLM agents (conversation history + semantic search)
5. **Working Memory** — 4-slot focus + goal stack + carry-over mechanism
6. **Semantic Memory** — concept graph with spreading activation (start with NetworkX)
7. **Kernel** — `MetacognitiveMonitor` + `BoundedRationalityGovernor`
8. **3-5 Thinking Systems** — First Principles, Inversion, Pre-mortem, Bayesian, Critical Thinking
9. **Dual-Process Router** — tier selection with learning loop
10. **Procedural Memory** — compiled fast-paths from Tier 2 patterns
11. **Consolidation** — LLM-powered schema extraction, pruning, pattern separation
12. **Emotional Memory** — overlay with concrete signal sources
13. **Everything else** — remaining thinking systems, custom backends, advanced learning
