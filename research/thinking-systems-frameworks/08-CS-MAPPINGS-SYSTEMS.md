# 08 — Computer Science Mappings: Thinking Systems

Every thinking system has a structural analog in computer science — not as metaphor, but because both domains solve the same fundamental information-processing problems.

---

## 1. Dual-Process Theory → Cache Hierarchy + Branch Prediction

**CS Concepts:** L1/L2/L3 cache, branch prediction, speculative execution

System 1 is the CPU cache: fast, local, based on recently accessed and frequently used patterns. It returns answers in nanoseconds using heuristics built from experience. System 2 is main memory (RAM): slower, more capacious, and capable of arbitrary computation but expensive to access.

Branch prediction is an even deeper analog: modern CPUs guess which branch of an if/else statement will execute and speculatively run that code. When the prediction is right (most of the time), you get massive speed gains. When it's wrong, you pay a pipeline flush penalty. System 1 biases are exactly this — mispredicted branches in human cognition.

The key insight: caches aren't bugs — they're essential for performance. The problem is cache invalidation (knowing when your fast, heuristic answer is stale or wrong). That's what Metacognition provides.

---

## 2. Critical Thinking → Static Analysis + Formal Verification

**CS Concepts:** linting, static analysis, type checking, formal verification, property-based testing

Critical Thinking examines claims for logical validity, evidence quality, and hidden assumptions — without "running" the claim in the real world. This is exactly what static analysis does to code: it finds bugs, type errors, unreachable code, and logical contradictions by examining the source without executing it.

Formal verification goes further — mathematically proving that a program satisfies its specification. This maps to rigorous critical thinking that doesn't just find flaws but proves correctness. The intellectual standards (clarity, accuracy, relevance, logic) are the type system and linting rules of thought.

---

## 3. Design Thinking → Agile + TDD + User Stories

**CS Concepts:** Agile methodology, test-driven development, user stories, MVP, sprint cycles, CI/CD

Design Thinking's iterative empathize-define-ideate-prototype-test cycle maps directly to Agile's sprint cycle with user stories. Empathize = user research and story writing. Define = sprint planning. Ideate = design sessions. Prototype = MVP/spike. Test = user acceptance testing. The emphasis on "fail fast, learn fast" mirrors the Agile principle of working software over comprehensive documentation.

TDD is particularly apt: write the test (define the need) before writing the code (building the solution). The test IS the empathy — it encodes what the user actually needs.

---

## 4. Systems Thinking → Graph Theory + Feedback Control Systems

**CS Concepts:** directed graphs, adjacency matrices, cycle detection, control theory, PID controllers, signal flow graphs

A system IS a directed graph: nodes are elements, edges are causal relationships. Feedback loops are cycles in the graph. Reinforcing loops are positive feedback in control theory. Balancing loops are negative feedback (the basis of all stable control systems like PID controllers). Emergence is the behavior of graph algorithms that cannot be predicted from individual node properties — it's a property of topology.

Stock-and-flow diagrams are finite-state machines with continuous variables. Leverage points map to graph centrality measures: nodes with high betweenness centrality are where interventions have the most impact.

---

## 5. Lateral Thinking → Simulated Annealing + Genetic Algorithm Mutation

**CS Concepts:** simulated annealing, genetic algorithm mutation operator, random restarts, stochastic gradient descent with noise

Lateral thinking deliberately introduces randomness to escape local optima. Simulated annealing does exactly this: it accepts worse solutions with a probability that decreases over time ("temperature"), allowing the search to explore broadly early on and converge later. The mutation operator in genetic algorithms randomly perturbs solutions to maintain diversity and prevent premature convergence.

de Bono's "random entry" technique IS a random restart. His "provocation" technique IS a large mutation. The cooling schedule in simulated annealing mirrors how creative processes start wild and gradually refine.

---

## 6. First Principles → Bootstrapping + Axiomatic Systems + Compiler Design

**CS Concepts:** bootstrapping, axioms in formal systems, compiler bootstrapping, bare-metal programming, reduction to primitives

First Principles thinking strips away abstractions to reach bedrock truths, then builds up from scratch. In CS, bootstrapping is building a system from nothing — writing a compiler in machine code, then using that compiler to compile itself. Axiomatic systems (like Peano arithmetic or ZFC set theory) define a minimal set of self-evident truths and derive everything else through formal rules.

Compiler design is particularly apt: a compiler reduces high-level abstractions to primitive operations (machine instructions), then constructs optimized solutions from those primitives. First Principles thinking IS compilation — reducing complex problems to primitive truths, then generating novel solutions.

---

## 7. Dialectical Thinking → Merge Conflict Resolution + Consensus Algorithms

**CS Concepts:** git merge conflict resolution, CRDTs (Conflict-free Replicated Data Types), Paxos/Raft consensus, three-way merge

Thesis and antithesis are divergent branches in a repository. Synthesis is the merge commit that integrates both. When the branches conflict (contradiction), you can't just pick one side — you need a three-way merge that understands the common ancestor and creates something new.

CRDTs are a deeper analog: data structures designed so that concurrent, divergent modifications automatically converge to a consistent state. They embody the dialectical principle that contradictions can be resolved structurally, not just by choosing a winner.

---

## 8. Deductive / Inductive / Abductive → Type Inference + ML + Diagnostics

**CS Concepts:** type inference and type checking (deductive), machine learning and statistical inference (inductive), diagnostic systems and model-based reasoning (abductive)

Deductive reasoning maps to type inference: given the type rules (premises) and an expression (situation), derive the type (conclusion) with certainty. If the types don't unify, you have a guaranteed error. Inductive reasoning maps to machine learning: observe many examples, infer a general pattern (model), accept that the model is probabilistic and could fail on new data. Abductive reasoning maps to diagnostic systems: given symptoms (observations), infer the most likely cause (diagnosis) — inference to the best explanation.

---

## 9. Metacognition → Profiler + Debugger + Reflection API

**CS Concepts:** profilers, debuggers, logging/observability, reflection APIs, runtime introspection, OpenTelemetry

Metacognition is a program monitoring its own execution. A profiler tracks which functions consume the most time (metacognitive awareness of cognitive effort). A debugger steps through execution to find where things go wrong (monitoring strategy effectiveness). Reflection APIs let a program inspect and modify its own structure at runtime — the computational equivalent of thinking about your own thinking process.

The modern observability stack (metrics, logs, traces) is metacognition for distributed systems: what's happening? Where's it slow? Where's it failing? Without it, you're flying blind.

---

## 10. Bayesian Thinking → Bayesian Networks + Probabilistic Programming

**CS Concepts:** Bayesian networks, probabilistic programming languages (Stan, Pyro, Gen), MCMC sampling, belief propagation

This is the most direct mapping. Bayesian networks are literal computational implementations of Bayesian reasoning: directed acyclic graphs where nodes are random variables, edges are conditional dependencies, and inference updates posterior probabilities given evidence. Probabilistic programming languages (Stan, Pyro, Gen) let you write generative models and automatically perform Bayesian inference.

Belief propagation algorithms pass messages through the network to update beliefs — which is exactly what Bayesian thinking tells humans to do.

---

## 11. OODA Loop → Event Loop + Reactive Programming + Game Loop

**CS Concepts:** event loops (Node.js), reactive programming (RxJS, ReactiveX), game loop (update-render cycle), actor model

The OODA Loop is a continuous sense-process-act cycle, which is exactly what an event loop does: observe events from the environment, process them (orient), dispatch handlers (decide), execute side effects (act), and loop. Reactive programming extends this: streams of observations flow through transformation pipelines (orientation) that trigger actions.

The game loop is an even closer match: every frame, the game reads input (Observe), updates the world model (Orient), makes AI decisions (Decide), and renders/executes (Act). Boyd's emphasis on speed maps to frame rate — faster loops mean smoother, more responsive systems.

---

## 12. Cynefin Framework → Pattern Matching + Strategy Pattern

**CS Concepts:** pattern matching (Rust, Haskell), Strategy pattern (GoF), polymorphic dispatch, match/case statements

Cynefin classifies a situation into a domain, then dispatches to the appropriate handling method. This is exactly what pattern matching does:

```
match situation {
    Clear => sense_categorize_respond(),
    Complicated => sense_analyze_respond(),
    Complex => probe_sense_respond(),
    Chaotic => act_sense_respond(),
    Confused => decompose_and_reclassify(),
}
```

The Strategy pattern in OOP is the same idea: select the appropriate algorithm at runtime based on context. The insight is that the dispatch mechanism itself is the contribution — not any individual handler.

---

## 13. Convergent / Divergent Thinking → BFS / DFS + Beam Search

**CS Concepts:** breadth-first search, depth-first search, beam search, multi-armed bandit, explore-exploit tradeoff

Divergent thinking is BFS: explore the solution space broadly, visiting many nodes at the same depth before going deeper. It maximizes coverage. Convergent thinking is pruning: eliminate branches that don't meet criteria, narrowing to the best path.

Beam search combines both: explore breadth-first but keep only the top-k most promising candidates at each level. The multi-armed bandit problem formalizes the explore-exploit tradeoff: how much time should you spend trying new options (diverging) versus exploiting the best-known option (converging)?

---

## 14. Munger's Latticework → Plugin Architecture + Unix Pipes

**CS Concepts:** plugin/module architecture, Unix philosophy (pipes and filters), polyglot programming, microservices, adapter pattern

Each mental model is an independent module with its own domain logic. The latticework itself is the plugin bus. The Unix philosophy ("do one thing well, compose via pipes") is the operational principle:

```
echo problem | biology_model | economics_model | psychology_model | synthesize
```

Each tool processes the input through its own lens and passes enriched output to the next. The adapter pattern translates between models from different domains so they can interoperate.

---

## 15. Socratic Method → REPL + Interactive Debugger

**CS Concepts:** REPL (Read-Eval-Print Loop), interactive debugger, recursive descent parsing, LISP evaluation model

The Socratic Method is a conversational REPL: read a claim, evaluate it through questioning, print the result (refined understanding or exposed contradiction), loop. The interactive debugger extends the metaphor: set breakpoints on beliefs, inspect the values of assumptions, step through the reasoning line by line.

Recursive descent parsing is a deeper structural match: break a complex expression into sub-expressions, resolve each one, and build up the parsed tree.

---

## 16. Sensemaking → Clustering + Topic Modeling + Log Aggregation

**CS Concepts:** k-means clustering, LDA topic modeling, ELK/Splunk log aggregation, dimensionality reduction, NLP

Sensemaking takes unstructured, ambiguous data and imposes structure — extracting patterns, categories, and narratives. Clustering algorithms do the same: take high-dimensional, unlabeled data and find natural groupings. Topic modeling extracts coherent themes from unstructured text.

Log aggregation (ELK stack, Splunk) is sensemaking for systems: collect vast streams of unstructured events, index them, extract patterns, and construct narratives about what happened and why. Weick's insight that sensemaking is retrospective maps perfectly to post-hoc log analysis.

---

## 17. Bounded Rationality → Greedy Algorithms + Approximation Algorithms

**CS Concepts:** greedy algorithms, approximation algorithms, Bloom filters, lossy compression, satisficing search

Satisficing IS a greedy algorithm: at each step, take the first option that meets the threshold. Don't backtrack, don't explore further. Greedy algorithms are often "good enough" — and in many cases, provably close to optimal.

Bloom filters are a beautiful analog: they trade perfect accuracy for massive speed and space savings — "definitely not in the set" or "probably in the set" but never "certainly in the set." This is bounded rationality: trade certainty for efficiency, and most of the time it works.

---

## 18. Theory of Constraints → Critical Path Method + Amdahl's Law

**CS Concepts:** critical path method (CPM/PERT), Amdahl's Law, profiler hotspot analysis, load balancing, queueing theory

Amdahl's Law is Theory of Constraints in a formula: the speedup of a system is limited by the fraction that cannot be parallelized (the constraint). Improving non-bottleneck components yields diminishing returns — exactly Goldratt's insight.

Critical Path Method identifies the longest sequence of dependent tasks — the constraint on project completion. Profiler hotspot analysis identifies the functions consuming the most CPU time. TOC's five focusing steps are profiler-guided optimization: identify the hotspot, optimize it, check if it's still the hotspot, repeat.

---

## 19. Inversion → Contrapositive + Reverse Engineering + Threat Modeling

**CS Concepts:** logical contrapositive, reverse engineering, STRIDE threat modeling, backward chaining, adversarial testing

The logical contrapositive (P→Q is equivalent to ¬Q→¬P) is inversion in its purest form. Reverse engineering works backward from the output to understand the mechanism. Threat modeling (especially STRIDE) inverts the security question: instead of "how do we make this secure?", ask "how would an attacker break this?"

---

## 20. Probabilistic Thinking → Monte Carlo Methods + Probabilistic Data Structures

**CS Concepts:** Monte Carlo simulation, MCMC, HyperLogLog, Count-Min Sketch, stochastic simulation, randomized algorithms

Probabilistic thinking models the world as distributions rather than point estimates. Monte Carlo simulation does the same: instead of calculating an exact answer, simulate thousands of random scenarios and measure the distribution of outcomes.

Probabilistic data structures (HyperLogLog, Count-Min Sketch, Bloom filters) trade exactness for efficiency — giving probabilistic answers that are provably close to correct. They embody the principle that a calibrated probability is more useful than a false certainty.

---

## 21. Embodied Cognition → Hardware-Software Co-Design + Edge Computing

**CS Concepts:** hardware-software co-design, ASICs/FPGAs, edge computing, physical computing (Arduino/IoT), neuromorphic chips

Embodied cognition says the hardware shapes the computation. In CS, hardware-software co-design is the recognition that the best systems are designed with hardware and software influencing each other. An ASIC IS embodied computation — the algorithm is literally baked into the physical substrate.

Edge computing pushes intelligence to where the physical interaction happens, rather than centralizing everything in the cloud. This mirrors embodied cognition's argument that intelligence should be close to the body and environment.

---

## 22. Distributed Cognition → MapReduce + CAP Theorem + Consensus Protocols

**CS Concepts:** MapReduce, CAP theorem, Paxos/Raft consensus, distributed hash tables, eventual consistency

MapReduce splits a problem across many nodes (Map — distribute the cognitive labor), then combines the results (Reduce — synthesize the distributed thinking).

The CAP theorem captures a fundamental distributed cognition constraint: you can't simultaneously have Consistency (everyone agrees), Availability (everyone can participate), and Partition tolerance (the system works even when communication breaks down). Human teams face exactly this tradeoff. Consensus protocols (Paxos, Raft) are algorithms for achieving agreement despite unreliable nodes.

---

## 23. Pre-mortem → Chaos Engineering + Fault Injection + Fuzzing

**CS Concepts:** Netflix's Chaos Monkey, fault injection testing, fuzzing, failure mode analysis, game day exercises

A pre-mortem imagines failure before it happens. Chaos engineering does the same to production systems: deliberately inject failures (kill servers, corrupt data, introduce latency) and observe how the system responds. Netflix's Chaos Monkey randomly terminates production instances to ensure resilience.

Fuzzing feeds random, malformed, or unexpected inputs to software to find crash conditions. Game day exercises simulate outages before they happen. All embody the pre-mortem principle: find the failures in your imagination so they don't find you in production.

---

## 24. Second-Order Thinking → Recursion + Higher-Order Functions + Game Trees

**CS Concepts:** recursion, higher-order functions, fixed-point combinators, minimax game trees, indirect effects in causal models

Second-order thinking applies a function to its own output: f(f(x)). This is recursion. "What are the consequences of the consequences?" is applying the consequence-function twice. Higher-order functions take functions as inputs and return functions — thinking about thinking operations rather than just thinking about data.

Minimax game trees are second-order thinking formalized for adversarial settings: I move (first order), they respond (second order), I respond to their response (third order). The depth of the game tree you can evaluate determines strategic sophistication — exactly as in chess engines.
