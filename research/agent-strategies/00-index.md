# Agent strategies: what the evidence says

> An evidence review of the ten reasoning strategies `ai-arch-toolkit` exposes through `ReasoningSpec(strategy=…)`: what each one is, what the literature measured, where it wins, where it falls short, what changed with 2024–2026 reasoning models, and how our implementation compares with the papers.
> Last reviewed: 2026-09-24 · Supersedes the earlier survey [`../llm_agent_architectures.md`](../llm_agent_architectures.md), several of whose recommendations the evidence here contradicts.

## How this was researched

- Six research passes on 2026-09-24 — one per pair of strategies plus one cross-cutting pass — working from primary sources (arXiv, ACL Anthology, OpenReview, PMLR, NeurIPS proceedings, vendor documentation), then follow-ups, replications and critiques.
- Every number is given with its model, benchmark and setting and comes from a source that was opened. Values recomputed from a paper's own tables, figure-read values and vendor claims are marked as such on each page.
- Our implementations were read in full (`src/ai_arch_toolkit/toolkit/agents/flows/`) and the suspicious behaviours were confirmed by simulation. Each page ends with an **In ai-arch-toolkit** section.

**Evidence strength** used below: **strong** — replicated independently and across model generations; **moderate** — original paper plus some independent support, mostly on 2023 models; **weak** — single source, vendor claims, or not replicated; **contested** — independent studies disagree.

## The ten strategies at a glance

| Page | Strategy | Family | Best for | Avoid when | Typical cost | Evidence |
|---|---|---|---|---|---|---|
| [01](01-completion.md) | `completion` | single call | self-contained inputs; closed-form reasoning (raise the effort, not the loop) | fresh/private data, actions, exact computation, verification | 1 call | strong |
| [02](02-react.md) | `react` | tool loop | observation-dependent tool work with uncertain step counts | no tools needed; fixed fan-out; exhaustive coverage | ~1 call per step, growing context | strong (the prompted "interleaving" claim is contested) |
| [03](03-plan-execute.md) | `plan_execute` | plan first | several deliverables, long horizons, planner/executor split, approval | next steps depend on what tools return; weak models | plan + steps × ReAct + solve | mixed |
| [04](04-rewoo.md) | `rewoo` | plan first | fixed tool chains; untrusted data with side-effecting tools (control-flow integrity) | meaning-dependent next steps; ambiguous entities | 2 calls + tools | weak for accuracy; moderate for security |
| [05](05-llm-compiler.md) | `llm_compiler` | plan first, parallel | ≥2 independent read-only lookups, multi-level dependencies | sequential or side-effecting work; depth-1 fan-out (native parallel calls suffice) | planner + joiner (+ replans) | moderate, dated (2023 models) |
| [06](06-reflexion.md) | `reflexion` | feedback loop | retries against an independent evaluator (tests, compiler, environment) | self-evaluation only; near-ceiling tasks; small budgets | tries × (attempt + evaluation + reflection) | strong that it needs an external evaluator; contested vs plain retry |
| [07](07-generate-review.md) | `generate_review` | feedback loop | artifacts with a rubric; tool-grounded or cross-model review | same-model review of closed-form reasoning | 1 + 2 × cycles | strong both ways: positive when the review adds information, negative when it does not |
| [08](08-self-discovery.md) | `self_discovery` | reasoning structure | recurring knowledge/NLU reasoning tasks on non-reasoning models (structure cached per task) | reasoning models; arithmetic/algorithmic tasks | 3 per task + 1 per instance (4 per request if not cached) | weak–moderate |
| [09](09-tot.md) | `tot` | search | combinatorial puzzles with an exact checker, weaker models | reasoning models within competence; no reliable evaluator; problems expressible as code | 5–100× CoT tokens | moderate on 2023 puzzles; thin beyond |
| [10](10-lats.md) | `lats` | search | resettable, side-effect-free environments with a strong reward | any irreversible tool; LLM-judge-only; interactive requests | 5–55× baselines, minutes of latency | contested |

Cross-strategy comparisons, task-property evidence, reasoning models vs scaffolds, and per-query strategy selection research: [11-cross-cutting](11-cross-cutting.md).

## Headline findings

1. **Most headline wins were measured on 2023 models, by the method's authors, against ReAct, without matched budgets.** Independent cost-controlled reruns shrink or erase them — on HumanEval, retry with rising temperature scored 93.2% for $2.45, LATS 88.0% for $134.50, Reflexion 87.8% for $3.90 [01](01-completion.md), [11](11-cross-cutting.md).
2. **An external signal decides whether feedback strategies help.** Reflexion, generate-review, ToT and LATS gain when tests, execution, tools, ground truth or a stronger reviewer supply information; self-evaluation alone hurts accuracy on reasoning tasks ([06](06-reflexion.md), [07](07-generate-review.md)).
3. **Reasoning models absorbed the prompt-level scaffolds, not the information-bringing ones.** CoT prompts, prescribed reasoning structures and puzzle search add little or hurt on reasoning models; tool loops, execution feedback, verifiers and parallel fan-out still add value ([01](01-completion.md), [08](08-self-discovery.md), [09](09-tot.md), [11](11-cross-cutting.md)).
4. **Task structure predicts the architecture.** Observation-dependent → `react`; fixed tool chain → `rewoo`; independent fan-out → `llm_compiler`; decomposable deliverables → `plan_execute` with feedback-driven replanning; checkable output → verifier-gated retry or `reflexion`; strongly sequential work, or tasks a single agent already does well (>~45%), → stay single-agent ([11](11-cross-cutting.md)).
5. **Native API features eroded two efficiency arguments.** Native parallel tool calls captured ~85% of LLMCompiler's latency savings in its own paper; prompt caching removes much of the token cost ReWOO and plan-first designs were built to avoid ([04](04-rewoo.md), [05](05-llm-compiler.md)).
6. **Security is a new reason for plan-first designs.** Fixing the tool plan before reading untrusted data (ReWOO, CaMeL, tool filters) cut targeted prompt-injection success from 57.69% to 6.84% in AgentDojo — control flow is protected, data flow is not ([03](03-plan-execute.md), [04](04-rewoo.md)).
7. **Tree search has hard preconditions.** ToT needs a discriminator ≥~90% accurate or an exact checker; LATS needs an environment that can be reset. Without them, best-of-N plus a verifier is cheaper and as good ([09](09-tot.md), [10](10-lats.md)).
8. **Replanning only helps with feedback.** Plan quality improved sharply on frontier models; what still decides agent tasks is adapting to what comes back — replanning that sees execution results (DEPS, Plan-and-Act) gains a lot; blind retries do not ([03](03-plan-execute.md)).
9. **Choosing a strategy per query mostly saves cost.** Routers (Adaptive-RAG, MasRouter, MaAS, Route to Reason) learn from labels obtained by running candidate strategies and keeping the cheapest that succeeded; classifier accuracy is the bottleneck (54.5% in Adaptive-RAG, with ~12 F1 left to a perfect router) ([11](11-cross-cutting.md)).
10. **The evidence gap:** no controlled study re-tests ToT, LATS, ReWOO, LLMCompiler or Self-Discover against plain ReAct on 2025–26 reasoning models at matched budgets.

## Decision guide

1. **No tool needed?** → `completion`; set the thinking effort by difficulty (low for lookups and transforms, high for multi-step math, logic, planning).
2. **Tools needed:**
   - the next step depends on what comes back → `react`;
   - the whole tool chain is known up front, or the data is untrusted while side-effecting tools are available → `rewoo`;
   - ≥2 independent read-only lookups, especially with multi-level dependencies → `llm_compiler` (depth-1 fan-out: `react` with parallel calls);
   - several distinct deliverables, a long horizon, or a plan that needs approval → `plan_execute`.
3. **An independent evaluator exists** (tests, compiler, validator, environment success) → verifier-gated retry or `reflexion`. **Only a rubric** → `generate_review`, ideally with a different reviewer model and reviewer tools.
4. **Combinatorial puzzle with an exact checker, non-reasoning model** → `tot`. **Resettable sandbox with a strong reward** → `lats`. Otherwise prefer best-of-N with a verifier.
5. **Recurring knowledge/NLU reasoning task on a non-reasoning executor** → `self_discovery`, with the structure cached per task type.

Each page's **Router signals** section lists the concrete choose/avoid conditions.

## State of our implementations (2026-09-24)

The research describes the papers' methods; several of our flows diverge in ways that change the conclusions. Details and recommendations are in each page's **In ai-arch-toolkit** section.

| Strategy | Divergences that matter | Defects confirmed by simulation | Router readiness |
|---|---|---|---|
| `completion` | — | — | ready |
| `react` | no loop detection, no context compaction; no reasoning replay on OpenAI (Chat Completions only) | — | ready |
| `plan_execute` | replans only on exceptions and with no feedback (same prompt as the first plan); executors never see the original task or the plan; a replan reruns every step | — | usable for decomposable deliverables; replanning is ineffective |
| `rewoo` | arguments go to the tool's first parameter only; unmatched plan lines dropped silently; no extraction worker | `#E1` substitution corrupts `#E10` | usable with single-argument tools and <10 evidence steps |
| `llm_compiler` | every DAG node is a full ReAct sub-agent (the paper's call/cost savings do not carry over); wave scheduling; replans get no feedback; a last-round `REPLAN` is returned as the answer | `$1` substitution corrupts `$10` | usable for fan-out; costlier than the paper |
| `reflexion` | scalar-only evaluator; the reflector never sees the trajectory or evaluator feedback; unbounded reflection memory | default evaluator accepts any non-empty answer (the flow = one ReAct run) | only with a real `evaluator` |
| `generate_review` | the generator never sees its previous draft; returns the last draft, not the best; reviewer defaults to the same model | verdict parser accepts "not acceptable" / "cannot accept" | not until the parser fix; then with a different reviewer and/or reviewer tools |
| `self_discovery` | discovery runs per request with no cache; 10 modules instead of 39; solver is a tool-using ReAct loop | — | non-reasoning executors only; pays discovery every request |
| `tot` | no beam width, no pruning, no backtracking; single LLM value sample; no programmatic evaluator | DFS expands the **worst** candidate first; BFS returns **no answer** with defaults; score parsing takes the first number | do not route until fixed |
| `lats` | trajectory-level (a child is a whole ReAct attempt); tools re-run every rollout (side effects repeat) | same score-parsing bug as `tot`. The tree never branched (`n_candidates` unused) — fixed on 2026-09-24 | read-only, idempotent or sandboxed tools only; after the parsing fix |

Fixes for the confirmed defects are tracked separately.

## Files

| # | Page | Topic |
|---|---|---|
| 00 | `00-index.md` | this overview |
| 01 | [`01-completion.md`](01-completion.md) | single call, CoT, reasoning models, test-time compute |
| 02 | [`02-react.md`](02-react.md) | ReAct and native tool loops |
| 03 | [`03-plan-execute.md`](03-plan-execute.md) | plan-and-execute, replanning, planning benchmarks |
| 04 | [`04-rewoo.md`](04-rewoo.md) | ReWOO, plan-then-execute security |
| 05 | [`05-llm-compiler.md`](05-llm-compiler.md) | LLMCompiler, parallel function calling |
| 06 | [`06-reflexion.md`](06-reflexion.md) | Reflexion, self-correction, verifier-gated retry |
| 07 | [`07-generate-review.md`](07-generate-review.md) | generator–critic loops, LLM-as-judge biases |
| 08 | [`08-self-discovery.md`](08-self-discovery.md) | SELF-DISCOVER and meta-reasoning prompts |
| 09 | [`09-tot.md`](09-tot.md) | Tree of Thoughts and test-time search |
| 10 | [`10-lats.md`](10-lats.md) | LATS and tree search for agents |
| 11 | [`11-cross-cutting.md`](11-cross-cutting.md) | comparisons, task properties, reasoning models, strategy routing |

Each strategy page follows the same outline: at a glance · origin · how it works · settings and cost in the papers · evidence table · strengths · weaknesses and costs · best use cases · where it falls short · variants · 2024–2026 context · router signals · in ai-arch-toolkit · open questions · sources.
