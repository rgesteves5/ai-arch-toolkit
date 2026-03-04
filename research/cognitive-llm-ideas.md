# Cognitive Science → LLM Enhancement Ideas

Ideas derived from `human-memory/` and `thinking-systems-frameworks/` research. To be built after the core refactoring completes.

---

## For `toolkit/` — Primitives

### 1. Cognitive Router (Cynefin + OODA → Strategy Pattern)
Classify task domain (clear/complicated/complex/chaotic), dispatch the right prompting strategy automatically. Clear → direct completion. Complicated → structured CoT. Complex → multi-hypothesis probing. Chaotic → act-first, refine later.

### 2. Tiered Memory System (Cache Hierarchy → Consolidation)
Replace simple sliding window with cognitive-inspired tiers:
- **Working Memory** — circular buffer (~4 items), aggressively summarized
- **Episodic Memory** — timestamped log, content-addressable retrieval
- **Semantic Memory** — extracted facts/concepts as embeddings, decoupled from episodes
- **Consolidation** — background extraction of semantic knowledge from episodic logs

### 3. Metacognitive Middleware (Profiler/Debugger analog)
Always-on monitoring layer: detect low-confidence/contradictory responses, trigger re-prompting with different strategies, track strategy effectiveness. Implements the "kernel interrupt" concept.

### 4. Spreading Activation Retrieval (Semantic Networks → BFS)
Graph-based RAG: retrieved chunks activate related chunks via BFS with decay. Richer than flat vector similarity.

### 5. Schema-Based Prompts (Frames → Objects with slots/defaults)
Prompts as schemas with named slots, defaults, and inheritance. Extends `_templates.py`.

---

## For `nanope/` — Cookbooks

### 6. Six Hats Deliberation Engine
Six parallel LLM calls (White=facts, Red=intuition, Black=risks, Yellow=benefits, Green=creative, Blue=synthesis), then merge. Maps to MapReduce.

### 7. Dialectical Synthesis
Generate thesis + antithesis, then synthesize. Multi-round dialectical refinement. Maps to merge conflict resolution / CRDTs.

### 8. Pre-mortem Agent
Wrapper around any agent that imagines failure modes before executing. Maps to chaos engineering / fuzzing.

### 9. First Principles Decomposer
Strip problem to fundamental truths, rebuild solution. Implements Innovation Triad: First Principles → Lateral Thinking → Convergent Thinking.

### 10. Bayesian Belief Updater
Maintains explicit probability estimates, updates with new info, explains reasoning. Decision-support agent.

### 11. Spaced Repetition Quiz Generator
Extract key facts from documents, generate quizzes at increasing intervals. Implements forgetting curve + testing effect.

### 12. Enhanced ToT/LATS with Metacognitive Kernel
Existing agents enhanced with self-monitoring, dynamic beam width, second-order evaluation.

### 13. Dual-Process Router
Fast System 1 path (small model, cached, heuristic) + slow System 2 path (large model, full CoT). Router selects based on complexity. Practical cost/latency optimization.

### 14. Sensemaking Pipeline
Cluster → extract themes → construct narrative → identify gaps. For log analysis, research synthesis, incident review.

---

## Architecture Mapping

| Research Layer | Toolkit Layer | Role |
|---|---|---|
| Kernel (Metacognition, Dual-Process) | `toolkit/` middleware | Always-on monitoring |
| OS (Cynefin, OODA, Systems Thinking) | `toolkit/` routers | Task classification, strategy selection |
| Application (Socratic, First Principles, etc.) | `nanope/` cookbooks | Specific thinking patterns as composable chains |

## Build Priority (when ready)
1. Dual-Process Router (nanope) — practical, saves cost
2. Cognitive Memory System (toolkit) — biggest gap
3. Six Hats Deliberation (nanope) — impressive demo
4. Metacognitive Middleware (toolkit) — makes everything better
5. Pre-mortem Agent (nanope) — wraps any existing agent
