# Research

Reference guides and deep-dive explorations spanning LLM APIs, agent architectures, Python practices, graph algorithms, cognitive science, and thinking frameworks.

---

## Top-Level Guides

| File | Description |
|------|-------------|
| `llm_api_complete_guide_2026.md` | Comprehensive guide to LLM provider APIs (Anthropic, OpenAI, Google, xAI) — auth, messaging, streaming, tools, multimodal, batch |
| `llm_agent_architectures.md` | Early practical survey of agent architectures — superseded by the evidence review in `agent-strategies/` |
| `python_best_practices.md` | Modern Python patterns, idioms, and conventions |
| `modern_python_2015_16.md` | Python 3.12–3.16 features and migration notes |
| `networkx_guide.md` | NetworkX library reference for graph construction, algorithms, and visualization |
| `graph_algorithms_overview.md` | Core graph algorithms — traversal, shortest path, MST, centrality, community detection |

---

## Agent Strategies

`agent-strategies/` — An evidence review (September 2026) of the ten strategies behind `ReasoningSpec(strategy=…)`: origin papers, measured results, strengths, failure modes, what changed with 2024–2026 reasoning models, router signals, and how each of our flows compares with its paper.

| # | File | Topic |
|---|------|-------|
| 00 | `00-index.md` | Overview, headline findings, decision guide, state of our implementations |
| 01 | `01-completion.md` | Single call — chain of thought, reasoning models, test-time compute |
| 02 | `02-react.md` | ReAct and native tool loops |
| 03 | `03-plan-execute.md` | Plan-and-execute, replanning, planning benchmarks |
| 04 | `04-rewoo.md` | ReWOO and plan-then-execute security |
| 05 | `05-llm-compiler.md` | LLMCompiler and parallel function calling |
| 06 | `06-reflexion.md` | Reflexion, self-correction, verifier-gated retry |
| 07 | `07-generate-review.md` | Generator–critic loops and LLM-as-judge biases |
| 08 | `08-self-discovery.md` | SELF-DISCOVER and meta-reasoning prompts |
| 09 | `09-tot.md` | Tree of Thoughts and test-time search |
| 10 | `10-lats.md` | LATS and tree search for agents |
| 11 | `11-cross-cutting.md` | Cost-aware comparisons, task properties, reasoning models vs scaffolds, per-query strategy routing |

**Quick start:** read `00` for the verdicts and the decision guide, then the page of the strategy you are about to use.

---

## Human Memory & Cognition

`human-memory/` — A six-part exploration of how the brain stores, organizes, and retrieves knowledge, culminating in a complete mapping to computer science data structures and algorithms.

| # | File | Topic |
|---|------|-------|
| 00 | `00-index.md` | Overview and reading guide |
| 01 | `01-types-of-memory.md` | Sensory, short-term/working memory (Baddeley's model), long-term memory (episodic, semantic, procedural, emotional, autobiographical), brain region mapping |
| 02 | `02-memory-and-learning.md` | Encoding/storage/retrieval cycle, levels of processing, consolidation and sleep, testing effect, spaced repetition, interleaving, elaboration, dual coding, forgetting curve |
| 03 | `03-information-to-knowledge.md` | DIKW hierarchy, schema assimilation/accommodation, chunking, knowledge categories by source, accessibility, structure, and domain |
| 04 | `04-knowledge-representation-in-brain.md` | Hebbian learning, LTP/LTD, localized vs. distributed representation, five theories (semantic networks, schemas, connectionism, embodied cognition, predictive coding) |
| 05 | `05-data-structures-cognitive-parallels.md` | Cognitive systems mapped to CS data structures — graphs, trees, hash maps, stacks/buffers, FSMs, tensors, generative models, cache hierarchies, garbage collection |
| 06 | `06-algorithms-complete-reference.md` | Algorithms catalog for each data structure — graph traversal, tree operations, neural network training, associative memory, decision trees, compression, caching, embeddings, RL |

**Reading order:** Sequential (01 through 06). Each document builds on the previous.

---

## Thinking Systems & Frameworks

`thinking-systems-frameworks/` — A ten-part compilation covering 24 thinking systems, their taxonomy, integration patterns, and computer science analogs.

| # | File | Topic |
|---|------|-------|
| 00 | `00-INDEX.md` | Document structure and reading guide |
| 01 | `01-THINKING-SYSTEMS-OVERVIEW.md` | All 24 systems in depth — origin, mechanics, key figures, practical applications |
| 02 | `02-TAXONOMY-AND-TAGS.md` | 12-dimensional tagging (nature, phase, locus, problem type, temporal, cognitive mode, domain, scale, load, output, epistemic stance, learnability) |
| 03 | `03-CATEGORIZATION-AXES.md` | Five orthogonal classification axes — descriptive/prescriptive, phase, locus, problem type, temporal orientation |
| 04 | `04-INTEGRATION-MASTER-SEQUENCE.md` | Universal 7-phase thinking flow and three operating layers (Kernel / OS / Application) |
| 05 | `05-INTEGRATION-POWER-PAIRS.md` | 15 synergistic pairs and 5 power triads with rationale |
| 06 | `06-INTEGRATION-PLAYBOOKS.md` | 8 situation-based playbooks — crisis, innovation, high-stakes decisions, complex systems, human-centered, competitive strategy, deep learning, group facilitation |
| 07 | `07-INTEGRATION-RULES-AND-ANTIPATTERNS.md` | 10 integration rules, 8 anti-patterns, 5-level maturity model |
| 08 | `08-CS-MAPPINGS-SYSTEMS.md` | All 24 systems mapped to algorithms, data structures, and paradigms in CS |
| 09 | `09-CS-MAPPINGS-INTEGRATION.md` | Integration concepts mapped to CS analogs — pipelines, OS layers, function composition, design patterns, SOLID principles |
| 10 | `10-REFERENCE-TABLES.md` | Quick-reference tables, key figures directory, recommended reading, staged learning sequence |

**The 24 systems:** Dual-Process Theory, Critical Thinking, Design Thinking, Systems Thinking, Lateral Thinking, First Principles, Dialectical Thinking, Deductive/Inductive/Abductive Reasoning, Metacognition, Bayesian Thinking, OODA Loop, Cynefin Framework, Convergent/Divergent Thinking, Munger's Latticework, Socratic Method, Sensemaking, Bounded Rationality, Theory of Constraints, Inversion, Probabilistic Thinking, Embodied Cognition, Distributed Cognition, Pre-mortem, Second-Order Thinking.

**Quick start:** Read `01` for the systems, then `04` for how they fit together. Software engineers should start with `08` and `09`.
