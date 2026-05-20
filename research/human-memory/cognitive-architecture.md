# Cognitive Architecture: Full System Design

## Table of Contents

1. [Philosophy and Design Principles](#1-philosophy-and-design-principles)
2. [System Overview](#2-system-overview)
3. [The Cognitive Loop](#3-the-cognitive-loop)
4. [Module Architecture](#4-module-architecture)
5. [Memory Systems (Detailed)](#5-memory-systems)
6. [Reasoning Systems (Detailed)](#6-reasoning-systems)
7. [Learning Systems (Detailed)](#7-learning-systems)
8. [Decision Systems (Detailed)](#8-decision-systems)
9. [Inter-Module Communication](#9-inter-module-communication)
10. [Data Flow Diagrams](#10-data-flow-diagrams)
11. [Consolidation and Maintenance](#11-consolidation-and-maintenance)
12. [Configuration and Constraints](#12-configuration-and-constraints)
13. [Extension Points](#13-extension-points)
14. [Relationship to Cognitive Science](#14-relationship-to-cognitive-science)

---

## 1. Philosophy and Design Principles

### Core Philosophy

This architecture treats the mind not as a single monolithic system but as a **society of specialized subsystems** that cooperate and compete to produce coherent behavior. Each subsystem has its own data structure, its own algorithms, and its own timescale of operation — but they are all wired together through shared buses, message passing, and a central coordination mechanism.

The architecture is **not** trying to simulate the brain at the neural level. Instead, it operates at the **cognitive level** — implementing the functional systems described by cognitive psychology (working memory, episodic memory, semantic memory, procedural memory, attention, etc.) using the data structures and algorithms that best capture each system's computational properties.

### Design Principles

**1. Biological Plausibility over Engineering Convenience**
Where there's a choice between the "clean" engineering solution and the solution that mirrors how human cognition actually works, we choose the latter. This means:
- Working memory has a hard capacity limit (not unlimited)
- Retrieval is competitive (memories interfere with each other)
- Forgetting is an active, beneficial process (not a bug)
- Learning is slow and incremental (not instant)
- Consolidation requires offline processing ("sleep")

**2. Separation of Storage and Process**
Each memory system is a data store with a well-defined interface. The processes that operate on memory (attention, retrieval, consolidation, reasoning) are separate modules. This mirrors how the brain separates "where things are stored" from "how they are accessed and manipulated."

**3. Everything is a Representation**
Every piece of information in the system — a concept, a memory, a goal, a percept, a skill — is a structured representation with a consistent format. This allows different subsystems to exchange information without translation layers.

**4. Competition and Cooperation**
Subsystems don't just cooperate — they compete for limited resources (working memory slots, processing cycles, attention). This competition produces emergent prioritization without requiring a central "homunculus" that makes all decisions.

**5. Graceful Degradation**
The system should degrade gracefully under load, missing information, or partial failure — just as the brain does. If semantic memory can't find an exact match, it should return the closest approximation. If working memory is full, the least relevant item should be displaced, not the system crash.

**6. Modularity and Extensibility**
Each subsystem should be independently testable, replaceable, and extensible. You should be able to swap in a more sophisticated semantic memory without touching procedural memory.

---

## 2. System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     COGNITIVE AGENT                      │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │              CENTRAL EXECUTIVE                   │    │
│  │         (Attention + Coordination)               │    │
│  └──────────┬──────────┬──────────┬────────────────┘    │
│             │          │          │                      │
│  ┌──────────▼──┐ ┌─────▼─────┐ ┌─▼────────────┐        │
│  │  WORKING    │ │ REASONING │ │  DECISION     │        │
│  │  MEMORY     │ │  ENGINE   │ │  ENGINE       │        │
│  │             │ │           │ │               │        │
│  │ • Focus     │ │ • Predict │ │ • Evaluate    │        │
│  │ • Goal Stack│ │ • Plan    │ │ • Select      │        │
│  │ • Scratchpad│ │ • Infer   │ │ • Commit      │        │
│  └──┬───┬───┬──┘ └───────────┘ └───────────────┘        │
│     │   │   │                                           │
│  ┌──▼───▼───▼──────────────────────────────────────┐    │
│  │            LONG-TERM MEMORY SYSTEMS              │    │
│  │                                                  │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────┐     │    │
│  │  │ SEMANTIC │ │ EPISODIC │ │  PROCEDURAL  │     │    │
│  │  │ MEMORY   │ │ MEMORY   │ │  MEMORY      │     │    │
│  │  │          │ │          │ │              │     │    │
│  │  │ Weighted │ │ Event    │ │ FSMs +       │     │    │
│  │  │ Graph    │ │ Log +    │ │ Decision     │     │    │
│  │  │          │ │ Vectors  │ │ Trees +      │     │    │
│  │  │          │ │          │ │ Rules        │     │    │
│  │  └──────────┘ └──────────┘ └──────────────┘     │    │
│  └──────────────────────────────────────────────────┘    │
│                                                         │
│  ┌──────────────────────────────────────────────────┐    │
│  │              LEARNING SYSTEMS                     │    │
│  │                                                  │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────┐     │    │
│  │  │ HEBBIAN  │ │ REWARD   │ │  SCHEMA      │     │    │
│  │  │ LEARNING │ │ LEARNING │ │  LEARNING    │     │    │
│  │  │          │ │ (RL)     │ │              │     │    │
│  │  │ Assoc.   │ │ TD/Q     │ │ Assimilate/  │     │    │
│  │  │ strength │ │ learning │ │ Accommodate  │     │    │
│  │  └──────────┘ └──────────┘ └──────────────┘     │    │
│  └──────────────────────────────────────────────────┘    │
│                                                         │
│  ┌──────────────────────────────────────────────────┐    │
│  │           MAINTENANCE / CONSOLIDATION             │    │
│  │                                                  │    │
│  │  • Memory replay    • Synaptic pruning           │    │
│  │  • Schema update    • Index rebuilding           │    │
│  │  • Compression      • Interference resolution    │    │
│  └──────────────────────────────────────────────────┘    │
│                                                         │
│  ┌──────────────────────────────────────────────────┐    │
│  │              INPUT / OUTPUT BUS                   │    │
│  │                                                  │    │
│  │  Sensory Buffer ←── Perception                   │    │
│  │  Action Buffer  ──► Effectors                    │    │
│  └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### Directory Structure

```
cognitive_agent/
│
├── core/
│   ├── agent.py                    # Main cognitive loop orchestrator
│   ├── central_executive.py        # Attention control + coordination
│   ├── clock.py                    # Internal time / tick management
│   └── representations.py          # Universal representation formats
│
├── memory/
│   ├── sensory_buffer.py           # Circular buffer — raw input holding
│   ├── working_memory.py           # Priority queue + goal stack + scratchpad
│   ├── semantic_memory.py          # Weighted directed graph
│   ├── episodic_memory.py          # Time-indexed event log + vector similarity
│   ├── procedural_memory.py        # FSMs + condition-action rules + decision trees
│   └── emotional_memory.py         # Valence tags + arousal modulation
│
├── reasoning/
│   ├── attention.py                # Salience computation + filtering
│   ├── predictive_model.py         # World model + prediction error
│   ├── planner.py                  # Goal decomposition + search (A*, MCTS)
│   └── inference.py                # Deduction, induction, analogy, abduction
│
├── learning/
│   ├── hebbian.py                  # Association strengthening/weakening
│   ├── reinforcement.py            # TD learning / Q-values / reward signals
│   ├── schema_learning.py          # Assimilation + accommodation + equilibration
│   └── consolidation.py            # Offline replay, pruning, compression
│
├── decision/
│   ├── evaluation.py               # State/action value estimation
│   ├── policy.py                   # Action selection (explore vs exploit)
│   └── goal_management.py          # Goal creation, prioritization, completion
│
├── io/
│   ├── perception.py               # Input processing pipeline
│   └── action.py                   # Output / effector interface
│
└── config/
    └── parameters.py               # All tunable system parameters
```

---

## 3. The Cognitive Loop

### The Main Cycle

The agent runs a continuous loop (each iteration = one "cognitive tick"). Not every subsystem runs every tick — some are fast (attention, working memory), some are medium (reasoning, retrieval), and some are slow (consolidation, schema learning).

```
EVERY TICK:
│
├── 1. PERCEIVE
│   │   Read sensory buffer
│   │   Compute salience of each item
│   │   Gate high-salience items into working memory
│   │
├── 2. ORIENT
│   │   Check current goal stack
│   │   Query episodic memory: "Have I been in this situation before?"
│   │   Query semantic memory: "What do I know about what I'm perceiving?"
│   │   Query procedural memory: "Do I have automatic responses for this?"
│   │   Compute prediction: "What do I expect to happen next?"
│   │
├── 3. REASON
│   │   Compare predictions to reality (prediction error)
│   │   If prediction error is low → continue current plan
│   │   If prediction error is high → re-evaluate, possibly replan
│   │   If no current plan → generate one (goal decomposition + search)
│   │   Run inference if needed (deduction, analogy, etc.)
│   │
├── 4. DECIDE
│   │   Evaluate candidate actions using value estimates
│   │   Modulate by emotional valence
│   │   Select action (policy: exploit best-known vs. explore)
│   │
├── 5. ACT
│   │   Execute selected action through effector interface
│   │   Push action + context to episodic memory log
│   │
├── 6. LEARN (fast learning — happens every tick)
│   │   Compute reward/outcome signal
│   │   Update TD/Q values (reinforcement learning)
│   │   Strengthen associations that were co-active (Hebbian)
│   │   Update prediction model based on prediction error
│   │
└── 7. MAINTAIN (slow maintenance — happens periodically, not every tick)
        If consolidation_interval reached:
            Replay recent episodic memories
            Compress and abstract repeated patterns
            Prune weak associations
            Update semantic memory structure
            Rebuild indices
```

### Tick Timing and Subsystem Frequencies

Not all subsystems operate at the same speed. This mirrors the brain, where sensory processing is fast (~50ms) and consolidation is slow (hours/days).

| Subsystem | Frequency | Rationale |
|---|---|---|
| Sensory buffer read | Every tick | Continuous perception |
| Attention filtering | Every tick | Real-time gating |
| Working memory update | Every tick | Active processing |
| Procedural memory check | Every tick | Automatic responses are fast |
| Semantic memory query | Every 1–3 ticks | Retrieval takes time |
| Episodic memory query | Every 1–3 ticks | Search takes time |
| Prediction + error | Every tick | Continuous prediction |
| Planning / search | On demand | Only when needed |
| Hebbian learning | Every tick | Continuous strengthening |
| RL value update | Every tick | Continuous evaluation |
| Schema learning | Every 10–50 ticks | Slow abstraction |
| Consolidation / pruning | Every N ticks ("sleep") | Offline maintenance |

---

## 4. Module Architecture

### Universal Representation Format

Every piece of information flowing through the system uses a common representation format. This is the "lingua franca" of the architecture — all modules produce and consume these.

**Concept Representation:**
- `id` — Unique identifier
- `type` — Category (entity, event, relation, property, action, goal, emotion)
- `content` — The actual information (flexible structure depending on type)
- `embedding` — Dense vector representation (for similarity computation)
- `activation` — Current activation level (0.0 to 1.0) — decays over time
- `emotional_valence` — Positive/negative/neutral emotional tag (-1.0 to 1.0)
- `emotional_arousal` — Intensity of emotional significance (0.0 to 1.0)
- `confidence` — How certain the system is about this item (0.0 to 1.0)
- `source` — Where this came from (perception, retrieval, inference, imagination)
- `timestamp` — When this was created or last accessed
- `access_count` — How many times this has been retrieved (for LFU-style forgetting)
- `associations` — Links to other concepts (with weights)

**Event Representation (for episodic memory):**
- All fields from Concept, plus:
- `context` — What was in working memory at the time
- `preceding_event` — Link to what happened before
- `following_event` — Link to what happened after
- `outcome` — What resulted from this event
- `reward` — Reward signal associated with this event

**Rule Representation (for procedural memory):**
- `condition` — Pattern that must be matched in working memory
- `action` — What to do when the condition matches
- `strength` — How often this rule has been successful (affects priority)
- `specificity` — How specific the condition is (more specific rules take priority)

**Goal Representation:**
- `desired_state` — What the agent wants to achieve
- `priority` — How important (affects position in goal stack)
- `deadline` — Optional time constraint
- `subgoals` — Decomposition into smaller goals
- `parent_goal` — The higher-level goal this serves
- `status` — Active / achieved / failed / suspended
- `plan` — Current sequence of actions to achieve this goal

---

## 5. Memory Systems

### 5.1 Sensory Buffer

**Purpose:** Briefly holds raw input before attention filtering. Prevents information loss during processing.

**Data Structure:** Circular buffer (ring buffer) with fixed capacity.

**Properties:**
- Capacity: Configurable (default ~20 items)
- Duration: Items expire after a configurable number of ticks (default 2–3)
- Write policy: New items overwrite the oldest (circular)
- Read policy: Non-destructive read; items remain until they expire
- No processing: Items are stored as raw representations, uninterpreted

**Interface:**
- `write(item)` — Add a new percept
- `read_all()` — Get all non-expired items
- `read_by_salience(threshold)` — Get items above a salience threshold
- `tick()` — Advance time, expire old items

**Cognitive Correspondence:** Iconic memory (visual), echoic memory (auditory), haptic memory (touch).

---

### 5.2 Working Memory

**Purpose:** The agent's active workspace. Holds the items currently being processed, the current goal stack, and a scratchpad for intermediate computations.

**Data Structure:** Composite of three sub-components:

**a) Focus of Attention (Priority Queue)**
- Fixed capacity: 4–7 slots (configurable, default 4)
- Each slot holds one Concept representation with an activation level
- When full, the lowest-activation item is displaced (evicted)
- Activation decays every tick unless refreshed
- Items compete for slots — high-activation items displace low-activation ones
- Implements the central bottleneck of human cognition

**b) Goal Stack (Stack)**
- Holds the hierarchy of current goals and subgoals
- LIFO: Current subgoal is on top; completing it pops back to parent
- Interruptions push new goals onto the stack
- Stack depth limit (configurable, default ~5) — exceeding causes goal shedding (dropping lowest-priority goals)
- Each goal tracks its status, plan, and priority

**c) Scratchpad (Temporary Store)**
- Unstructured workspace for intermediate computations
- Used by the reasoning engine for partial results, hypotheses, comparisons
- Cleared after use or after a timeout
- Not capacity-limited in the same way as Focus, but items decay

**Interface:**
- `add_to_focus(item)` — Attempt to add an item; may displace existing item
- `get_focus_contents()` — Current items in focus
- `refresh(item_id)` — Reset decay timer (rehearsal)
- `push_goal(goal)` — Push a new goal onto the stack
- `pop_goal()` — Complete/abandon current goal, return to parent
- `peek_goal()` — Check current goal without removing
- `write_scratchpad(key, value)` — Store intermediate result
- `read_scratchpad(key)` — Retrieve intermediate result
- `tick()` — Decay all activations; evict items below threshold

**Displacement Mechanics:**
When a new item tries to enter Focus but all slots are full:
1. Compare new item's activation to the minimum activation in Focus
2. If new item's activation > minimum: displace the minimum item
3. Displaced item gets a "displaced" event written to episodic memory (it was briefly in consciousness)
4. If new item's activation ≤ minimum: new item is rejected (doesn't enter consciousness)

**Cognitive Correspondence:** Baddeley's working memory model — the Focus maps to the central executive's attentional focus, the Goal Stack to executive control, and the Scratchpad to the episodic buffer.

---

### 5.3 Semantic Memory

**Purpose:** Stores the agent's long-term conceptual knowledge — facts, categories, relationships, and general knowledge about the world. The agent's "encyclopedia."

**Data Structure:** Weighted directed graph, implemented as an adjacency list.

**Node Structure:**
- Each node is a Concept representation
- Nodes have a base activation level (reflects how frequently/recently accessed)
- Nodes belong to one or more clusters (knowledge domains)

**Edge Structure:**
- Directed and weighted
- Edge types: `is_a`, `has_property`, `part_of`, `causes`, `related_to`, `opposite_of`, `instance_of`, `used_for`, `located_in`, and custom types
- Weight represents association strength (0.0 to 1.0)
- Weights change over time through Hebbian learning

**Key Operations and Their Algorithms:**

**a) Spreading Activation (Primary retrieval mechanism)**
- Triggered when a concept enters working memory or attention
- Algorithm: Modified BFS from the activated node(s)
- Activation spreads along edges, attenuated by edge weight and distance
- Spread formula: activation(neighbor) += activation(source) × edge_weight × decay_factor
- Spread depth is limited (configurable, default 3 hops)
- All nodes whose activation exceeds a threshold are returned as "primed" concepts
- Multiple simultaneous sources of activation can converge on a concept, boosting it (intersection = relevance)

**b) Similarity Search**
- Each concept also has an embedding vector
- Cosine similarity between embeddings enables finding semantically similar concepts even without direct edges
- Used when spreading activation doesn't find a match — fall back to vector similarity

**c) Subgraph Retrieval**
- Given a query concept, retrieve the local neighborhood (ego graph)
- Used when the agent needs context about a concept — "tell me everything you know about X"

**d) Taxonomic Reasoning**
- Follow `is_a` edges upward for generalization ("a dog is a mammal is an animal")
- Follow `is_a` edges downward for specialization ("animals include mammals, which include dogs")
- Property inheritance: properties of parent categories apply to children unless overridden

**e) Analogy**
- Find structural correspondences between subgraphs
- "A is to B as C is to ?" — find node D such that the relationship pattern A→B mirrors C→D
- Uses graph isomorphism / structural mapping

**Indexing:**
- Hash index on concept ID for O(1) direct lookup
- Inverted index from properties/types to concepts (for queries like "find all animals that can fly")
- Embedding index (ANN structure like HNSW) for fast similarity search

**Maintenance:**
- Edge weights decay slowly over time (forgetting unused associations)
- Edge weights strengthen when both endpoints are co-active (Hebbian learning)
- Periodic community detection (Louvain) to identify and label knowledge domains
- PageRank computed periodically to identify hub concepts

**Cognitive Correspondence:** Collins & Quillian's semantic network model, augmented with distributed representations (embeddings) and schema-like clustering.

---

### 5.4 Episodic Memory

**Purpose:** Stores the agent's personal history — timestamped records of what happened, in what context, with what outcome and emotional coloring. The agent's "autobiography."

**Data Structure:** Composite of:
- **Event Log** — Append-only, time-indexed sequence of Event representations
- **Embedding Store** — Each event also stored as an embedding vector for similarity search
- **Temporal Index** — B-tree index on timestamps for efficient time-range queries
- **Context Index** — Inverted index from context elements to events (for cue-based retrieval)

**Storage:**
Each event record contains:
- What happened (the action/event)
- Working memory contents at the time (context)
- Current goal at the time
- Emotional valence and arousal
- Outcome / reward
- Links to preceding and following events
- Embedding vector (dense representation of the entire event)

**Key Operations and Their Algorithms:**

**a) Record (Encoding)**
- Every tick, a snapshot of working memory + action + outcome is written to the log
- Encoding depth varies: high emotional arousal → richer encoding (more context captured); low arousal → sparser encoding
- The emotional memory module modulates encoding strength

**b) Cue-Based Retrieval**
- Given a cue (a concept, a context, an emotional state), find matching past events
- Step 1: Compute embedding of the cue
- Step 2: ANN search (HNSW or LSH) on event embeddings for top-k similar events
- Step 3: Rank results by similarity × recency × emotional significance
- Returns the best-matching events, which are then loaded into working memory

**c) Temporal Retrieval**
- "What happened between tick 100 and tick 200?"
- B-tree range query on the temporal index
- Returns events in chronological order

**d) Pattern Retrieval**
- "Has this situation happened before?"
- Encode current working memory state as a vector
- Search episodic embeddings for similar past states
- Returns episodes where the agent was in a similar situation — enables learning from experience

**e) Sequential Retrieval (Replay)**
- Given an event, follow forward/backward links to replay the sequence
- Used during consolidation ("mental replay")
- Also used for planning ("last time I was in this situation, what happened next?")

**Reconsolidation:**
- When a memory is retrieved, it becomes temporarily modifiable
- The retrieved event can be updated with new information, new emotional tags, or corrections
- This models the reconsolidation window in neuroscience

**Forgetting:**
- Events have an access_count and last_access_timestamp
- Events that are never retrieved gradually become harder to retrieve (lower activation)
- During consolidation, very low-activation events may be compressed (reduced to a summary) or pruned entirely
- Emotional events resist forgetting (higher base activation)

**Cognitive Correspondence:** Tulving's episodic memory; the event log mirrors autobiographical memory; embedding-based retrieval mirrors content-addressable, cue-dependent recall.

---

### 5.5 Procedural Memory

**Purpose:** Stores the agent's skills, habits, and automatic behavioral patterns — "knowing how" rather than "knowing that."

**Data Structure:** Three-layer system:

**Layer 1: Condition-Action Rules (Production Rules)**
- If [pattern in working memory] → then [action]
- Each rule has a strength (success rate) and specificity (how detailed the condition is)
- When multiple rules match, conflict resolution selects the winner:
  1. Most specific match wins
  2. Among equally specific, highest strength wins
  3. Among equal strength, most recently successful wins
- This is directly inspired by the ACT-R production system

**Layer 2: Finite State Machines**
- For sequential, multi-step procedures
- Each FSM represents a skill (e.g., "make coffee" = sequence of states and transitions)
- Current state is tracked; input triggers transitions
- Well-practiced FSMs execute automatically (fast, low working memory load)
- New FSMs require working memory supervision (slow, effortful)

**Layer 3: Decision Trees**
- For complex conditional procedures
- Each internal node is a condition; each leaf is an action
- Trees are learned from experience (see Learning Systems)
- More expressive than simple rules for multi-factor decisions

**Automatization:**
- New procedures start as explicit rule sequences (Layer 1) requiring working memory
- With successful repetition, they compile into FSMs (Layer 2) that run automatically
- Eventually, frequently-used pathways become lookup tables (effectively Layer 3 leaves with no branching — direct stimulus-response)
- This progression models the cognitive → associative → autonomous stages of skill learning

**Interface:**
- `match(working_memory_state)` — Find all rules/FSMs that match the current state
- `execute(rule_or_fsm)` — Execute the matched procedure
- `learn_rule(condition, action, outcome)` — Create or strengthen a rule
- `compile_to_fsm(rule_sequence)` — Convert a practiced sequence into an FSM
- `get_active_fsms()` — Which FSMs are currently running

**Cognitive Correspondence:** Anderson's ACT-R production system; Fitts & Posner's three stages of motor learning.

---

### 5.6 Emotional Memory

**Purpose:** Not a separate storage system, but a **modulatory system** that tags all other memories with emotional significance and influences processing priority.

**Data Structure:**
- Valence-arousal state: A continuous 2D vector representing current emotional state
  - Valence: negative (-1.0) to positive (+1.0)
  - Arousal: calm (0.0) to intense (1.0)
- Emotional associations: Mapping from concepts/events to learned emotional responses
- Somatic markers: Mapping from decision contexts to gut-feeling evaluations

**Functions:**
- **Encoding modulation**: High arousal → stronger encoding in episodic memory, more context captured
- **Retrieval modulation**: Current emotional state biases which memories are retrieved (mood-congruent recall)
- **Attention modulation**: Emotionally significant items get higher salience in the attention filter
- **Decision modulation**: Somatic markers provide fast evaluative signals — "this option feels good/bad" before full deliberation
- **Learning modulation**: Reward/punishment signals from emotional outcomes drive reinforcement learning

**Emotional State Update:**
- Perception of emotionally tagged concepts shifts the valence-arousal state
- Reward signals shift toward positive valence
- Punishment/failure shifts toward negative valence
- Emotional state decays toward neutral over time (emotional regulation)
- Extreme emotional states narrow attention (tunnel vision) and bias retrieval (only emotionally congruent memories surface)

**Cognitive Correspondence:** Damasio's somatic marker hypothesis; LeDoux's emotional processing; amygdala modulation of memory consolidation.

---

## 6. Reasoning Systems

### 6.1 Attention Module

**Purpose:** The gatekeeper between the world and working memory. Computes salience and decides what gets through.

**Salience Computation:**
For each item in the sensory buffer, compute a salience score based on:
- **Novelty**: How different is this from recent percepts? (High novelty = high salience)
- **Relevance**: How related is this to the current goal? (Measured by embedding similarity to goal)
- **Emotional significance**: Does this match known emotional triggers?
- **Expectation violation**: How much does this deviate from the predictive model's expectation? (Prediction error = salience)
- **Intensity**: Raw signal strength

**Formula:** `salience = w1×novelty + w2×relevance + w3×emotional_sig + w4×prediction_error + w5×intensity`

**Filtering:** Items above the salience threshold are candidates for working memory. They compete with existing working memory contents for slots.

**Attention Modes:**
- **Bottom-up (stimulus-driven)**: Salience from novelty, intensity, emotional significance — automatic, reflexive
- **Top-down (goal-driven)**: Salience from relevance to current goal — voluntary, effortful
- The balance shifts based on context: in routine situations, top-down dominates; when something unexpected happens, bottom-up captures attention

**Cognitive Correspondence:** Posner's attention model; Corbetta & Shulman's dorsal (top-down) and ventral (bottom-up) attention networks.

---

### 6.2 Predictive Model

**Purpose:** Maintains an internal model of how the world works. Generates predictions about what will happen next. Computes prediction errors that drive attention and learning.

**Data Structure:** Can be implemented at multiple levels of sophistication:
- **Simple**: Transition probability table — given state S and action A, what's the probability of each next state S'?
- **Medium**: Learned association weights forming a predictive graph
- **Complex**: Small neural network that takes (state, action) and outputs (predicted_next_state, confidence)

**Key Operations:**

**a) Predict**
- Input: Current state (working memory contents) + planned action
- Output: Expected next state + confidence
- Uses the internal model to generate the prediction

**b) Compute Prediction Error**
- Input: Predicted state vs. actual observed state
- Output: Prediction error vector (which dimensions were wrong, by how much)
- Large prediction errors → high salience (attention) + strong learning signal

**c) Update Model**
- Input: Prediction error
- Output: Updated model parameters
- The model adjusts to reduce future prediction errors
- Learning rate modulated by confidence and emotional arousal

**Hierarchical Prediction:**
- Level 0: Raw sensory predictions ("I expect to see X")
- Level 1: Event predictions ("I expect action A to cause outcome B")
- Level 2: Abstract predictions ("In situations like this, things usually go well")
- Each level predicts the activity of the level below
- Errors propagate upward — only unexplained errors at one level get passed to the next

**Cognitive Correspondence:** Friston's predictive coding / free energy principle; Rao & Ballard's hierarchical predictive coding.

---

### 6.3 Planner

**Purpose:** Generates plans to achieve goals by decomposing them into subgoals and searching for action sequences.

**Algorithms (selected based on problem characteristics):**

**a) Goal Decomposition**
- Given a goal, query semantic and procedural memory for known sub-goals
- "To achieve G, you typically need to achieve G1, then G2, then G3"
- Creates a goal tree that is pushed onto the goal stack

**b) Forward Search (BFS / DFS)**
- From current state, enumerate possible actions, predict outcomes, search toward goal state
- BFS for short-horizon problems (find any solution quickly)
- DFS for deep problems (explore one path fully before backtracking)

**c) A* Search**
- Best-first search with heuristic
- Heuristic: estimated distance from predicted state to goal state (embedding distance)
- Optimal when heuristic is admissible

**d) Means-Ends Analysis**
- Compare current state to goal state
- Identify the biggest difference
- Find an action (operator) that reduces that difference
- If the operator has preconditions not met, recursively plan to meet them
- Classic AI planning; maps to Newell & Simon's GPS (General Problem Solver)

**e) Analogical Planning**
- Query episodic memory: "When I was in a similar situation before, what plan worked?"
- Retrieve past plan; adapt it to current situation
- Much faster than planning from scratch when good analogies exist

**f) Monte Carlo Tree Search (MCTS)**
- For complex problems with many branching possibilities
- Repeatedly simulate random rollouts from current state
- Build a search tree weighted by simulation results
- Balance exploration of untried actions with exploitation of promising ones
- Used when the branching factor is too high for exhaustive search

**Plan Monitoring:**
- Once a plan is committed, the predictive model monitors execution
- If actual outcomes deviate from planned outcomes → replanning triggered
- Replanning can be local (adjust one step) or global (generate a new plan)

**Cognitive Correspondence:** Newell & Simon's problem space theory; case-based reasoning (analogical planning); mental simulation.

---

### 6.4 Inference Engine

**Purpose:** Draws conclusions from available knowledge using multiple reasoning strategies.

**Inference Types:**

**a) Deduction (certain conclusions from general rules)**
- If "all birds have feathers" and "Tweety is a bird" → "Tweety has feathers"
- Implementation: Rule matching + forward/backward chaining on semantic memory
- Certainty: conclusion inherits the minimum confidence of the premises

**b) Induction (general rules from specific instances)**
- Observed: "Swan 1 is white, Swan 2 is white, Swan 3 is white" → Tentative: "All swans are white"
- Implementation: Pattern detection across episodic memory; create new semantic memory edges with low initial confidence
- Confidence increases with more confirming instances

**c) Abduction (best explanation for observations)**
- Observed: "The grass is wet" → Explanation candidates: "It rained" / "The sprinklers ran" / "Heavy dew"
- Implementation: Search semantic memory for known causes of the observation; rank by prior probability and consistency with other observations
- Generates hypotheses rather than certainties

**d) Analogy (structural mapping between domains)**
- "The atom is like a solar system" — electrons orbit the nucleus like planets orbit the sun
- Implementation: Structure Mapping Engine (Gentner) — find correspondences between relational structures in two domains
- Powerful for creative insight and transfer learning

**Cognitive Correspondence:** Johnson-Laird's mental models; Gentner's structure mapping theory; Bayesian reasoning.

---

## 7. Learning Systems

### 7.1 Hebbian Learning

**Purpose:** Strengthens associations between concepts/representations that are co-active. The most basic and continuous form of learning.

**Rule:** When two concepts are simultaneously active in working memory, strengthen the edge between them in semantic memory.

**Update:** `edge_weight(A, B) += learning_rate × activation(A) × activation(B)`

**Complementary weakening:** Edges that are NOT co-active decay: `edge_weight(A, B) -= decay_rate × edge_weight(A, B)`

**Effect:** Over time, concepts that frequently co-occur become strongly associated. Concepts that rarely co-occur drift apart. This sculpts the semantic network to reflect the statistical structure of the agent's experience.

**Constraints:**
- Weights are bounded [0.0, 1.0]
- Learning rate is small (associations build gradually)
- Decay rate is very small (forgetting is slow)
- Emotional arousal multiplies the learning rate (emotional events form stronger associations)

**Cognitive Correspondence:** Hebb's rule; long-term potentiation (LTP) and long-term depression (LTD); statistical learning.

---

### 7.2 Reinforcement Learning

**Purpose:** Learns the value of states and actions from reward/punishment signals. Drives adaptive decision-making.

**Components:**

**a) Reward Signal**
- Generated by the emotional memory system and goal management
- Goal achievement → positive reward
- Goal failure → negative reward
- Progress toward goal → small positive reward
- Unexpected positive events → positive reward
- Unexpected negative events → negative reward

**b) Value Function**
- Maps states (working memory configurations) to expected future reward
- Implemented as a table (for discrete states) or function approximator (for continuous states)
- Updated via Temporal Difference learning

**c) TD Learning Update**
- After each tick: `V(s) ← V(s) + α[r + γV(s') − V(s)]`
- Where α = learning rate, γ = discount factor, r = immediate reward, s' = next state
- The TD error `δ = r + γV(s') − V(s)` is the crucial learning signal
- Positive δ = things went better than expected → strengthen current behavior
- Negative δ = things went worse than expected → weaken current behavior

**d) Q-Learning (for action selection)**
- Maps state-action pairs to expected value
- `Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') − Q(s,a)]`
- The decision engine uses Q-values to select actions

**Cognitive Correspondence:** Dopamine signaling in the basal ganglia; Schultz's reward prediction error neurons; habit formation through the striatum.

---

### 7.3 Schema Learning

**Purpose:** Builds and refines the agent's mental models (schemas) — organized knowledge structures that represent typical patterns, situations, and causal relationships.

**Processes:**

**a) Assimilation**
- New information fits an existing schema
- Effect: The schema's confidence and detail increase; the new information is "filed" under the schema
- Implementation: When episodic memory stores a new event that matches an existing schema pattern, increment the schema's exemplar count and optionally add new details

**b) Accommodation**
- New information contradicts an existing schema
- Effect: The schema must be modified, split, or a new schema created
- Trigger: Prediction error that cannot be resolved within the current schema
- Implementation: If a schema's prediction error exceeds a threshold, fork the schema into two variants, or modify the existing schema's conditions/expectations
- This is cognitively costly (takes working memory resources) but necessary for accurate models

**c) Schema Abstraction**
- After many similar episodes, extract the common structure as a new schema
- Implementation: Periodic comparison of clusters of similar episodic memories; extract shared features as schema slots with default values; identify variable features as schema slots without defaults
- The new schema is stored in semantic memory as a frame-like structure

**d) Equilibration**
- Ongoing tension between assimilation (fitting into existing models) and accommodation (changing models)
- The system seeks a balance — not so rigid that it ignores contradictions, not so flexible that it restructures constantly
- Regulated by a conservatism parameter that determines how much evidence is needed to trigger accommodation

**Cognitive Correspondence:** Piaget's assimilation, accommodation, and equilibration; Bartlett's schema theory; Rumelhart & Norman's accretion, tuning, and restructuring.

---

## 8. Decision Systems

### 8.1 Evaluation Module

**Purpose:** Estimates the value of candidate actions and states to guide decision-making.

**Inputs:**
- Q-values from reinforcement learning (learned action values)
- Predictive model outputs (predicted outcomes of each action)
- Somatic markers from emotional memory (gut feelings about options)
- Goal relevance scores (does this action advance the current goal?)

**Integration:**
`action_value(a) = w1×Q(s,a) + w2×predicted_reward(s,a) + w3×somatic_marker(s,a) + w4×goal_relevance(s,a)`

The weights can be adaptive — in familiar situations, Q-values dominate (habitual); in novel situations, predictive model and deliberation dominate.

---

### 8.2 Policy Module

**Purpose:** Selects actions given evaluations.

**Strategies:**

**a) Greedy (Exploitation)**
- Always choose the action with the highest evaluated value
- Good when knowledge is reliable; bad when exploration is needed

**b) ε-Greedy (Exploration)**
- With probability ε, choose a random action; otherwise choose greedy
- ε decreases over time as the agent becomes more confident
- Simple but effective

**c) Softmax (Boltzmann)**
- Choose actions with probability proportional to their evaluated value
- Temperature parameter controls randomness: high temperature → more random; low temperature → more greedy
- Better than ε-greedy because it avoids choosing clearly bad actions during exploration

**d) Thompson Sampling**
- Maintain uncertainty estimates for each action's value
- Sample from the uncertainty distribution and choose the action with the highest sample
- Naturally balances exploration (uncertain options get explored) and exploitation (known-good options get chosen)

**Cognitive Correspondence:** Exploration-exploitation trade-off in foraging theory; the interplay between habitual (basal ganglia) and goal-directed (prefrontal cortex) decision-making.

---

### 8.3 Goal Management

**Purpose:** Creates, prioritizes, tracks, and resolves goals.

**Goal Lifecycle:**
1. **Creation**: Goals arise from needs (internal drives), opportunities (detected by attention), or subgoal decomposition (by the planner)
2. **Prioritization**: Goals are ranked by urgency, importance, feasibility, and emotional significance
3. **Commitment**: Top-priority goal is pushed onto the goal stack; a plan is generated
4. **Pursuit**: Plan is executed tick by tick; progress is monitored
5. **Completion/Failure**: Goal is achieved (pop stack, generate positive reward) or abandoned (pop stack, generate negative reward, possibly adjust priorities)
6. **Interruption**: A higher-priority goal can interrupt the current one (push onto stack); the interrupted goal is suspended and resumed later

**Goal Conflict Resolution:**
When multiple goals compete:
- Higher priority wins
- If equal priority, more urgent (closer deadline) wins
- If still tied, more feasible (closer to completion) wins
- Conflicting goals (pursuing one prevents the other) must be resolved by abandoning one or finding a compromise plan

**Cognitive Correspondence:** BDI architecture (Bratman's Beliefs-Desires-Intentions); Maslow's hierarchy of needs (for goal prioritization); Lewin's goal-tension theory.

---

## 9. Inter-Module Communication

### Communication Architecture

Modules communicate through three mechanisms:

**a) Shared Working Memory**
- The primary communication channel
- Modules read from and write to working memory
- Semantic memory places retrieved concepts in working memory
- The reasoning engine reads working memory to generate inferences
- The decision engine reads working memory to evaluate options
- This mirrors the "global workspace" theory of consciousness

**b) Event Bus**
- For asynchronous signals that don't need to occupy working memory
- Events: `perception_event`, `prediction_error_event`, `goal_achieved_event`, `goal_failed_event`, `emotional_shift_event`, `consolidation_trigger_event`
- Modules subscribe to relevant events
- Example: Emotional memory subscribes to `prediction_error_event` to modulate arousal

**c) Direct Queries**
- Some modules directly query others
- Working memory queries semantic memory ("what do you know about X?")
- The planner queries procedural memory ("do you have a procedure for X?")
- The inference engine queries episodic memory ("have we seen X before?")
- These are synchronous request-response interactions

### Information Flow Summary

```
Sensory Buffer → Attention → Working Memory → {All Systems}
                                ↕
                     ┌──────────┼──────────┐
                     ↓          ↓          ↓
              Semantic Mem  Episodic Mem  Procedural Mem
                     ↓          ↓          ↓
                     └──────────┼──────────┘
                                ↓
                        Reasoning Engine
                     (Prediction + Planning
                      + Inference)
                                ↓
                        Decision Engine
                     (Evaluate + Select)
                                ↓
                          Action Output
                                ↓
                         Learning Systems
                     (Hebbian + RL + Schema)
                                ↓
                     Update All Memory Systems
                                ↓
                     [Periodic: Consolidation]
```

**Cognitive Correspondence:** Baars' Global Workspace Theory — working memory as the "global workspace" that broadcasts information to specialized, otherwise unconscious processing modules.

---

## 10. Data Flow Diagrams

### Flow 1: Processing a Novel Percept

```
New Percept arrives
    │
    ▼
Sensory Buffer [circular buffer write]
    │
    ▼
Attention Module computes salience
    ├── Query Predictive Model: "Was this expected?" → Prediction error
    ├── Query Emotional Memory: "Is this emotionally tagged?" → Emotional significance
    ├── Compare to current Goal: "Is this relevant?" → Goal relevance
    │
    ▼
Salience > threshold?
    ├── NO → Percept decays in sensory buffer, never reaches consciousness
    │
    ├── YES ▼
    │   Working Memory: attempt to add
    │       ├── Slot available → Add directly
    │       ├── Full → Displace lowest-activation item
    │       │           └── Displaced item → brief episodic record
    │       │
    │       ▼
    │   Item is now in conscious processing
    │       │
    │       ├── Spreading Activation in Semantic Memory
    │       │   └── Related concepts become primed
    │       │       └── Highest-activation primed concepts may enter Working Memory
    │       │
    │       ├── Cue-Based Retrieval from Episodic Memory
    │       │   └── "When did I last encounter this?"
    │       │       └── Relevant past episodes may enter Working Memory
    │       │
    │       ├── Pattern Matching in Procedural Memory
    │       │   └── "Do I have an automatic response for this?"
    │       │       ├── YES → Execute procedural response (fast, automatic)
    │       │       └── NO → Proceed to deliberative reasoning
    │       │
    │       ▼
    │   Reasoning / Planning / Decision (if needed)
    │       │
    │       ▼
    │   Action Output
    │       │
    │       ▼
    │   Learning Updates
    │       ├── Episodic: Record what happened
    │       ├── Hebbian: Strengthen co-active associations
    │       ├── RL: Update value estimates based on outcome
    │       └── Schema: Assimilate or accommodate
```

### Flow 2: Goal-Directed Behavior

```
Goal activated (pushed to goal stack)
    │
    ▼
Planner: Do I have a known plan?
    ├── Check Procedural Memory for matching procedure
    ├── Check Episodic Memory for past success with similar goal
    │
    ├── Known plan found → Execute step by step
    │       │
    │       ├── Each step: Predict outcome → Act → Compare → Learn
    │       │
    │       ├── Outcome matches prediction → Continue plan
    │       └── Outcome doesn't match → Replan
    │
    └── No known plan → Generate plan
            │
            ├── Goal Decomposition (semantic memory: subgoal relationships)
            ├── Search (A* / MCTS from current state toward goal state)
            ├── Analogical planning (episodic memory: past similar situations)
            │
            ▼
        Plan generated → Execute step by step (same as above)
```

### Flow 3: Consolidation ("Sleep")

```
Consolidation triggered (every N ticks)
    │
    ├── 1. Episodic Replay
    │       Select recent episodic memories
    │       Replay them through the predictive model
    │       Update predictions based on replay
    │       Strengthen associations activated during replay (Hebbian)
    │
    ├── 2. Schema Extraction
    │       Cluster similar episodic memories
    │       Extract common patterns as new schemas
    │       Store schemas in semantic memory
    │
    ├── 3. Memory Pruning
    │       Identify low-activation, rarely-accessed items in all stores
    │       Episodic: Compress old, low-value events into summaries
    │       Semantic: Weaken low-weight edges; remove edges below threshold
    │       Procedural: Remove rules with very low success rates
    │
    ├── 4. Index Rebuilding
    │       Recompute embeddings for changed concepts
    │       Rebuild ANN index for episodic similarity search
    │       Recompute PageRank for semantic memory
    │       Recompute community detection for knowledge domains
    │
    └── 5. Interference Resolution
            Identify competing memories (similar representations, different content)
            Increase pattern separation (make their embeddings more distinct)
```

---

## 11. Consolidation and Maintenance

### Why Consolidation Matters

The system cannot learn effectively through online learning alone. Several problems require periodic offline processing:

**a) The Stability-Plasticity Dilemma**
Online learning updates can overwrite previously learned knowledge (catastrophic forgetting). Consolidation gradually integrates new knowledge with old, preserving both.

**b) Memory Compression**
Without compression, episodic memory grows without bound. Consolidation extracts the useful patterns and discards redundant detail.

**c) Schema Formation**
Individual episodes need to be abstracted into general knowledge. This abstraction requires comparing many episodes, which is computationally expensive and done offline.

**d) Index Maintenance**
As the content of memory stores changes, indices (embeddings, ANN structures, inverted indices) become stale. Periodic rebuilding keeps retrieval efficient.

### Consolidation Schedule

- **Micro-consolidation**: Every 10–50 ticks. Quick: strengthen recent associations, update indices.
- **Macro-consolidation**: Every 500–1000 ticks ("sleep"). Full: replay, schema extraction, pruning, index rebuild.
- **Emergency consolidation**: Triggered by memory pressure (stores approaching capacity limits). Aggressive pruning.

---

## 12. Configuration and Constraints

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `working_memory_capacity` | 4 | Max items in Focus |
| `goal_stack_depth` | 5 | Max nested goals before shedding |
| `sensory_buffer_size` | 20 | Max items in sensory buffer |
| `sensory_buffer_decay_ticks` | 3 | Ticks before sensory items expire |
| `activation_decay_rate` | 0.05 | How fast activation decays per tick |
| `activation_threshold` | 0.1 | Below this, items are evicted from working memory |
| `salience_threshold` | 0.3 | Below this, percepts don't enter working memory |
| `spreading_activation_depth` | 3 | Max hops for spreading activation |
| `spreading_activation_decay` | 0.5 | Activation multiplier per hop |
| `hebbian_learning_rate` | 0.01 | How fast associations strengthen |
| `hebbian_decay_rate` | 0.001 | How fast unused associations weaken |
| `rl_learning_rate` | 0.1 | TD/Q-learning alpha |
| `rl_discount_factor` | 0.95 | Gamma — how much future rewards are discounted |
| `exploration_rate` | 0.1 | Epsilon for ε-greedy; decreases over time |
| `emotional_arousal_multiplier` | 2.0 | How much emotion amplifies learning rate |
| `consolidation_interval` | 500 | Ticks between full consolidation cycles |
| `pruning_threshold` | 0.01 | Edge weight below which edges are pruned |
| `schema_extraction_min_exemplars` | 5 | Minimum similar episodes before abstracting a schema |
| `prediction_error_attention_weight` | 0.3 | How much prediction errors drive salience |

All parameters should be tunable and accessible from a central configuration.

---

## 13. Extension Points

The architecture is designed to be extended in several directions:

### Environment Interface
- The `io/` layer is currently a stub (no environment)
- Adding an environment requires implementing `perception.py` (how raw input maps to representations) and `action.py` (how decisions map to outputs)
- Any environment can be plugged in: text worlds, grid worlds, API interactions, sensory data streams

### Multi-Modal Representations
- Current representations are unimodal (symbolic + embedding)
- Extension: Add visual representations (image embeddings), auditory representations, etc.
- The architecture supports this — each modality would have its own sensory buffer and encoding pipeline, converging into the universal representation format

### Social Cognition
- Add a Theory of Mind module that maintains models of other agents
- Each other agent would be a node in semantic memory with its own predicted beliefs, desires, and intentions
- Enables social reasoning, cooperation, deception detection

### Language
- Add a language module that can encode working memory contents as natural language and decode natural language input into representations
- Language would be both an input channel (perception) and an output channel (action)
- Inner speech: Using the language module to verbally encode thoughts in working memory, providing an additional rehearsal loop (like the phonological loop)

### Meta-Cognition
- Add a meta-cognitive monitor that observes the agent's own cognitive processes
- Tracks: What's in working memory? How confident is the agent? Are goals being achieved? Is learning progressing?
- Can adjust parameters dynamically (e.g., increase exploration when confidence is low)
- Enables self-awareness and strategic self-regulation

### Creativity
- Add a divergent thinking module that can:
  - Combine concepts from different semantic memory clusters (bisociation)
  - Relax constraints in the planner to find novel solutions
  - Generate hypothetical scenarios by perturbing the predictive model
  - Use analogy across distant domains

---

## 14. Relationship to Cognitive Science

### How This Architecture Maps to Known Cognitive Architectures

| Our Module | ACT-R Equivalent | SOAR Equivalent | Global Workspace Theory |
|---|---|---|---|
| Working Memory | Buffers (goal, retrieval, visual, etc.) | Working Memory | Global Workspace (conscious access) |
| Semantic Memory | Declarative Memory (chunks) | Semantic Memory | Long-term store (unconscious) |
| Episodic Memory | (Partial — through declarative memory) | Episodic Memory | Long-term store (unconscious) |
| Procedural Memory | Production Memory | Production Memory (rules) | Specialized processors |
| Central Executive | Procedural Module (conflict resolution) | Decision Procedure | Attention / Broadcasting |
| Attention | Utility-based selection | Elaboration/Decision | Spotlight mechanism |
| Learning (Hebbian) | Base-level learning | Chunking | — |
| Learning (RL) | Utility learning | Reinforcement learning | — |
| Consolidation | (Not modeled) | (Not modeled) | (Not modeled) |

### Key Differences from Existing Architectures

**vs. ACT-R:**
- We add explicit episodic memory with embedding-based retrieval (ACT-R's declarative memory doesn't cleanly separate episodic from semantic)
- We add a predictive model (ACT-R doesn't have one)
- We add consolidation (ACT-R doesn't model offline learning)
- We use continuous activations and embeddings alongside symbolic representations (ACT-R is primarily symbolic)

**vs. SOAR:**
- We have richer memory systems (SOAR's memories are simpler)
- We add emotional modulation (SOAR has no emotional system)
- We use hybrid representations (symbolic + vector) rather than purely symbolic
- We add consolidation and forgetting

**vs. Global Workspace Theory:**
- We implement GWT's broadcasting mechanism (working memory as global workspace)
- We add specific algorithms for each unconscious specialist module
- We make the architecture concrete and implementable rather than theoretical

### Open Questions This Architecture Helps Explore

1. **How do the capacity limits of working memory shape reasoning and learning?** — Change `working_memory_capacity` and observe effects.
2. **What's the optimal balance between habitual (procedural) and deliberative (planning) decision-making?** — Tune the automatization thresholds.
3. **How does emotional modulation affect memory and decision quality?** — Vary `emotional_arousal_multiplier`.
4. **What consolidation schedule produces the best long-term learning?** — Experiment with `consolidation_interval` and replay strategies.
5. **How does the structure of semantic memory affect reasoning and inference?** — Compare different graph structures and their effect on spreading activation.

---

## Summary

This architecture integrates:
- **5 memory systems** (sensory, working, semantic, episodic, procedural) + emotional modulation
- **4 reasoning capabilities** (attention, prediction, planning, inference)
- **3 learning mechanisms** (Hebbian, reinforcement, schema)
- **3 decision components** (evaluation, policy, goal management)
- **1 consolidation system** (replay, pruning, compression, abstraction)

All connected through a global workspace (working memory) and event bus, running a continuous perceive-orient-reason-decide-act-learn loop with periodic offline consolidation.

Every component maps to a specific cognitive science theory, a specific set of data structures, and a specific set of algorithms — making the bridge from neuroscience to implementation explicit and traceable.
