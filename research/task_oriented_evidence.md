# Task-oriented evidence for agent architectures

> Last reviewed: 2026-09-26. A compact evidence map for architecture routing in
> `ReasoningSpec(strategy=...)`. The figures below are results reported by each method's authors;
> they are not performance guarantees for this toolkit.

## What the evidence supports

There is empirical evidence that some architectures outperform ReAct on particular kinds of task.
There is no proof that any of them is universally superior. The studies use different models,
tools, budgets, and benchmarks; many of the largest gains were measured on 2023 models and have not
been independently reproduced on 2025–2026 reasoning models under matched budgets.

| Architecture | Evidence compared with ReAct | Tasks where it appears stronger |
|---|---|---|
| **LLMCompiler** | Direct comparison: up to **3.74× faster**, **6.73× cheaper**, and about **9 percentage points more accurate** on a parallel-tool benchmark. Moderate evidence from the method's authors using 2023 models. | Independent tool calls that can be represented as a dependency graph and executed concurrently. [Paper](https://proceedings.mlr.press/v235/kim24y.html) |
| **ReWOO** | Direct comparison: across six benchmarks, the paper reports **64% fewer tokens** and **+4.4 accuracy points** over ReAct. This is a single study; direct generation or CoT beat both agents on some benchmarks. | Workflows whose tool plan can mostly be fixed before execution begins. [Paper](https://arxiv.org/abs/2305.18323) |
| **Reflexion** | Direct but not equal-cost: ReAct with Reflexion completed **130 of 134 ALFWorld tasks** and significantly outperformed ReAct alone. It did not significantly improve over ReAct on WebShop. | Repeatable tasks with reliable feedback from tests, compilers, simulators, validators, or the environment. [Paper](https://arxiv.org/abs/2303.11366) |
| **LATS** | Direct but much more computationally expensive: on a HotpotQA subset, **71% versus 32% for ReAct**; on HumanEval with GPT-3.5, **83.8% versus 56.9%**. Some experiments used oracle feedback and up to 50 trajectories. | Difficult, reversible search with a strong evaluator, when quality matters much more than latency or cost. [Paper](https://arxiv.org/abs/2310.04406) |
| **Tree of Thoughts** | Strong evidence against CoT, but **not a clean ReAct comparison**: on Game of 24 it reached **74% versus 4% for CoT**, with substantially more computation. | Small branching search spaces with evaluable intermediate states and backtracking. [Paper](https://papers.nips.cc/paper_files/paper/2023/hash/271db9922b8d1f4dd7aaef84ed5ac703-Abstract-Conference.html) |
| **Self-Discover** | Improvements of up to **32% over CoT**, not ReAct. The evidence mainly concerns models without native reasoning. | Recurring reasoning tasks where the appropriate reasoning structure is not known in advance. [Paper](https://papers.nips.cc/paper_files/paper/2024/hash/e41efb03e20ca3c231940a3c6917ef6f-Abstract-Conference.html) |
| **Generate–Review / Self-Refine** | Improved results across seven tasks. For GPT-4 code optimisation, performance increased from **27.3% to 36.0%**. It was not directly compared with ReAct, and feedback quality is decisive. | Revisable artifacts with a clear rubric: writing, code, structured documents, and constrained generation. [Paper](https://arxiv.org/abs/2303.17651) |
| **Plan–Execute** | No convincing controlled evidence shows that the exact architecture consistently beats ReAct at a matched budget. Evidence for planning and replanning is mixed. | A plausible hypothesis for decomposable deliverables, long horizons, and plans requiring approval; keep experimental until validated on local tasks. |

Figures from different papers must not be compared directly: an architecture may obtain higher
quality merely because it made more calls, received a stronger evaluator, or explored more
trajectories.

## Consequence for an architecture router

A defensible policy, subject to local evaluation, is:

- no tools or interaction: `completion`;
- the next step depends on the previous observation: `react`;
- the tool chain is known in advance: `rewoo`;
- two or more independent read-only calls: `llm_compiler`;
- a reliable external verifier and retry budget exist: `reflexion`;
- the artifact is revisable against a clear rubric: `generate_review`;
- the reasoning structure is unknown but reusable: `self_discovery`;
- branching, reversible search with an exact evaluator: `tot` or `lats`;
- `plan_execute`: experimental until supported by local evidence.

The router should return its recommendation, the conditions supporting it, expected budget,
confidence, and a safe fallback. Low-confidence choices should be measured in shadow mode before
becoming automatic.

For the full evidence review, implementation divergences, and router caveats, see
[`agent-strategies/00-index.md`](agent-strategies/00-index.md) and the strategy-specific pages.

