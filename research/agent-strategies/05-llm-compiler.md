# LLMCompiler (`llm_compiler`)

> One planner call turns the request into a DAG of tool calls with `$id` placeholders; a dispatcher runs every task whose inputs are ready, in parallel; a joiner answers or asks for a new plan.
> Evidence: real but narrow and dated — 1.4–3.7× lower latency and 3.4–6.7× lower cost than ReAct on fan-out benchmarks with late-2023 models; native parallel function calling already captured ~85% of the latency savings in the paper's own data.
> Last reviewed: 2026-09-24 · Part of [agent-strategies](00-index.md)

## At a glance

- **What it buys:** parallel I/O and fewer LLM calls on requests that fan out ("compare X and Y", "for each of these N") [1].
- **What native parallel calls already give:** in the paper's Table 1, OpenAI parallel function calling reached 85–87% of LLMCompiler's latency savings and 58–82% of its cost savings (derived) [1]. The remaining structural advantage is planning **several levels of dependent calls in one LLM call**.
- **Accuracy gains vanish with strong models:** 62.00% vs ReAct† 62.47% (gpt-3.5, HotpotQA); 89.38% vs 89.09% (GPT-4-Turbo, ParallelQA) [1]. The 2023 gains came from ReAct failure modes (looping, stopping early) of older models.
- **Headline results never replan** (`max_replans: 1` = one round in the reference configs) [4][5].
- **Avoid for** observation-dependent, sequential or side-effecting work [6][13].
- **In this toolkit:** each DAG node is a full ReAct sub-agent (so the paper's call and cost savings do not carry over), replans get no feedback, and `$1` corrupts `$10` — see [In ai-arch-toolkit](#12-in-ai-arch-toolkit).

## 1. Origin

**Kim, Moon, Tabrizi, Lee, Mahoney, Keutzer, Gholami** (UC Berkeley / ICSI / LBNL), "An LLM Compiler for Parallel Function Calling", ICML 2024, PMLR 235:24370–24391; arXiv 2312.04511 [1][2]. Reference code: SqueezeAILab/LLMCompiler, with LangGraph and LlamaIndex ports [3].

## 2. How it works

- **Planner (one call):** emits one task per line with strictly increasing ids; arguments are constants or `$id` references to earlier outputs (`$1=search(A)`, `$2=search(B)`, `$3=math($1/$2)`). The prompt asks it to maximise parallelism and end with a join action [1].
- **Task Fetching Unit (no LLM):** dispatches every task whose inputs are ready and substitutes placeholders with real outputs.
- **Executor:** runs dispatched tasks concurrently; tools may themselves be LLM agents.
- **Streaming:** the planner streams tasks, so execution starts before planning finishes.
- **Joiner:** one call writes the answer or asks for a new plan; on replan, intermediate results go back to the planner; the reference code caps rounds with `max_replans` and forces a final answer on the last one [1][4]. The LangGraph port keeps the same joiner decision [10].

Versus ReAct: ReAct makes one LLM call per tool call over a growing prompt; LLMCompiler makes ~2 LLM calls per round however many tools run, and dependent calls get inputs by substitution instead of another LLM turn [1].

## 3. Settings and cost in the paper

- Models: gpt-3.5-turbo-1106 (HotpotQA, Movie Recommendation), gpt-4-turbo-1106 (ParallelQA), gpt-4-0613 (Game of 24, WebShop), LLaMA-2 70B on vLLM; T=0; OpenAI results averaged over 3 runs (Nov 2023) [1].
- **Benchmarks chosen for their call patterns:** HotpotQA *comparison* split only (1.5k, 2-way parallel); Movie Recommendation (500, 8-way); ParallelQA — the authors' own 113 questions, restricted to facts in Wikipedia first paragraphs so failed searches were designed out; Game of 24 (100); WebShop (500) [1].
- **Baseline:** ReAct† = ReAct plus prompts against looping and early stopping; latency compared only with ReAct† because the original ReAct's loops made its latency unmeasurable [1].
- **Overheads (Movie):** planner 1.88 s + answer 1.62 s — more than half of total latency; slowest search 1.13 s vs 0.61 s mean (stragglers) [1].
- **Latency model:** speedup approaches N for N-way work when tool time dominates, and 1 when planning dominates [1].

## 4. Evidence

| Benchmark | Model | Baseline | Result | Src |
|---|---|---|---|---|
| HotpotQA comparison | gpt-3.5-turbo-1106 | ReAct† 62.47%, 7.12 s; OpenAI parallel FC 62.05%, 4.42 s (1.61×) | 62.00%, 3.95 s (1.80×) | [1] |
| HotpotQA comparison | LLaMA-2 70B | ReAct† 54.40%, 13.44 s | 57.83%, 9.58 s (1.40×) | [1] |
| Movie Recommendation | gpt-3.5-turbo-1106 | ReAct 68.60%; ReAct† 72.47%, 20.47 s; OpenAI parallel 77.00%, 7.42 s (2.76×) | 77.13%, 5.47 s (3.74×) | [1] |
| Movie Recommendation | LLaMA-2 70B | ReAct† 70.60%, 33.37 s | 77.80%, 11.83 s (2.82×) | [1] |
| ParallelQA | gpt-4-turbo-1106 | ReAct 89.09%, 35.90 s; OpenAI parallel 87.32%, 19.29 s (1.86×) | 89.38%, 16.69 s (2.15×) | [1] |
| Cost, $/1k queries (HotpotQA / Movie / ParallelQA) | same | ReAct 5.00 / 20.46 / 480; OpenAI parallel 2.66 / 6.14 / 260 | 1.47 / 3.04 / 103 | [1] |
| Avg input tokens per query | same | ReAct 2.9k / 20k / 46k | 1.3k / 2.8k / 9.2k | [1] |
| Game of 24 (with ToT, replanning) | gpt-4-0613 | ToT 74.00%, 241.2 s | 75.33%, 83.6 s (2.89×) | [1] |
| WebShop | gpt-3.5-turbo | ReAct 19.8%, 5.98 s; LATS 38.0%, 1066 s | 48.2%, **10.48 s** (slower than ReAct) | [1] |
| WebShop | gpt-4-0613 | ReAct 35.2%, 19.90 s; LASER 50.0%, 72.16 s | 55.6%, 26.73 s | [1] |
| HotpotQA bridge (replanning, appendix) | not stated | ReAct† 23.1%, 6.42 s | 26.3%, 4.70 s | [1] |
| Independent: HotpotQA and WebShop (50 each) | Llama-3.1-8B (vLLM) | ReAct | better on HotpotQA; less efficient on WebShop (unneeded tool calls); plan/execute overlap only 18.2% of latency | [6] |
| Independent: BFCL parallel subsets | Llama-3.2-3B / GPT-4o (cloud figures emulated) | sequential calls | batched parallel 1.3× / 1.7×; AsyncLM 1.6× / 2.1×; multi-step parallel: batched 3.2×, AsyncLM 5.4× | [7] |
| Dependency prediction (TaskBench) | GPT-4 | tool-choice F1 81.54 | dependency F1 54.70 — a ~20-point gap across all models | [11] |

## 5. Strengths

- **Parallel I/O on fan-out queries:** 1.4–3.7× lower latency than ReAct† [1].
- **Fewer LLM calls and tokens** because dependent calls need no LLM turn; also 1.8–2.6× cheaper than OpenAI parallel calling in Nov 2023 [1].
- **Avoided 2023 failure modes:** ~85% of gpt-3.5 ReAct runs on Movie stopped before searching all 8 movies vs 1% for LLMCompiler; ~10% of LLaMA-2 ReAct runs on HotpotQA looped (>4 calls) and scored <10% [1].
- **Isolated context per task** — each tool sees only its inputs [1].
- **Exhaustive exploration where coverage pays** (WebShop 19.8% → 48.2% with gpt-3.5) [1].
- **Model-agnostic**, and usable as a parallel executor for other strategies (ToT on Game of 24) [1].

## 6. Weaknesses and costs

- **Call profile:** ~2 × (1 + replans) LLM calls plus any LLM-backed tools; the joiner prompt carries every observation [1].
- **Fixed overhead on the critical path:** planner + answer >50% of latency on Movie; slower than ReAct on WebShop (10.48 s vs 5.98 s; 26.73 s vs 19.90 s) [1].
- **Commits before seeing any result;** data-dependent branches need a replan (≥2 more calls) [1].
- **Brittle planner:** wrong variable mapping caused 8% of ParallelQA failures; shipped prompts target LLaMA-2 70B [1][3].
- **Over-invocation** — planning the whole DAG launched unnecessary calls on WebShop [6].
- **Stragglers and weak overlap** (slowest task ~2× the mean; 18.2% overlap) [1][6]; **bursty load** hits rate limits [18].

## 7. Best use cases

- Fan-out known from the request text: comparisons, per-entity lookups, "for each of these N", gathering from several sources [1].
- I/O-bound tools that are slow relative to an LLM call (streaming paid off most when a slow tool hid the planner's latency) [1].
- Dependencies ≥2 levels deep whose values pass through verbatim (search → arithmetic) — the one case where it structurally beats native parallel calls [1].
- Weaker or open models that loop or stop early under ReAct [1].
- Coverage-critical selection where ReAct commits too early [1].

## 8. Where it falls short

- **Sequential or observation-dependent work:** left out of MATH and HumanEval as unsuited; less efficient than ReAct on WebShop navigation [6]; bridge questions need replanning [1].
- **Failed or ambiguous lookups:** a static plan cannot retry with another entity name — and ParallelQA was built so searches would not fail [1].
- **No accuracy gain with strong models** (see At a glance) [1].
- **Wrong dependencies:** dependency prediction is ~20 F1 points harder than tool choice, and harder as graphs grow [11][25].
- **Chained calls:** BFCL's authors saw parallel calling regress in some late-2024 models and suggest one call at a time may be faster *and* more accurate for chains; real users ask for single-turn parallel calls less often than for choosing between functions [9].
- **Replanning quality:** 2026 benchmarks show complex topologies trap agents in trial-and-error loops, with recovery rates ~37% lower under implicit tool failures [23]; the best model finishes 94.78% of tasks but scores 55.18 on replanning [24].
- **Parallelism does not fix bad tools or reasoning:** 92% of ParallelQA failures came from the executor or the final answer [1].
- **Side effects:** parallelise only independent, read-only calls; run ordered or side-effecting calls in sequence [13].

## 9. Variants and follow-ups

- **LLM-Tool Compiler** — fuses similar tool operations at runtime: up to 4× more parallel calls, −40% tokens, −12% latency [8].
- **AsyncLM** — interrupts let the model keep generating while calls run; beats batched parallel calling but needs fine-tuning for small models [7].
- **AsyncFC** — execution-layer "futures", no model changes [20]; **LLMOrch** (IEEE TSE 2025) — data- and processor-aware scheduling [19].
- **DTA-Llama** (ACL 2025) — Llama trained on DAG-converted ToolBench traces [16]; **GAP** — SFT+RL for dependency-aware planning [15]; **ParallelSearch** — RL rewarding parallel sub-queries (+12.7% at 69.6% of the calls) [17].
- **Flash-Searcher** — DAG web agent re-planning every few steps; 67.7% on BrowseComp with GPT-5 [18].
- **Parallel-Synthesis** — the joiner reads workers' KV caches; 2.5–11× lower time-to-first-token [21]; **ParaGUI** — parallel planner-worker GUI agents, +12.9 points [22].

## 10. With 2024–2026 models

- **Native parallel calling is the default:** OpenAI returns several calls per turn (`parallel_tool_calls=false` to disable) [12]; Claude 4+ calls in parallel by default, all results must return in one user message, and some newer models make fewer parallel calls in long loops [13].
- **Interleaved thinking** lets a ReAct loop batch independent calls *and* adapt to what comes back, removing much of the case for planning everything up front [14].
- **What still holds:** one-call planning of a multi-level DAG, placeholder pass-through, dispatch-as-ready. AsyncLM shows overlapping generation with execution goes further than batching [7].
- **What changed:** the accuracy argument rested on 2023 ReAct failure modes; no 2025–26 frontier-model replication exists (a gap). Reasoning-model planners spend thinking tokens, pushing speedup toward 1 unless tools are slow (inference from the paper's latency model).
- **Direction:** parallelism is being trained into models (GAP, ParallelSearch, DTA-Llama) or pushed into runtimes (AsyncLM, AsyncFC); DAG planners reappear inside deep-research agents (Flash-Searcher).

## 11. Router signals

**Choose `llm_compiler` when:**
- the request names ≥2 independent lookups ("compare", "each of", explicit lists);
- tools are read-only or idempotent and take about as long as an LLM call or longer;
- dependency depth is ≥2 and values pass through verbatim;
- latency or cost is an explicit goal;
- the model is weak/open or lacks native parallel calls;
- a final synthesis over all results is needed.

**Avoid it (use `react` with native parallel calls) when:**
- there is one tool call, or dependency depth is 1 (native parallel calls got ~85% of the savings);
- the next step depends on what comes back (navigation, debugging, step-by-step math or code);
- tools have side effects, a required order, shared state or tight rate limits;
- lookups are likely to fail or be ambiguous;
- tools are fast, so planner and joiner overhead dominates.

**Router features:** `fan_out`, `dependency_depth`, `tool_latency`, `tools_side_effect_free`, `native_parallel_supported`. Cap replans at 1–2 and force a final answer at the cap.

## 12. In ai-arch-toolkit

Flow: [`_llm_compiler.py`](../../src/ai_arch_toolkit/toolkit/agents/flows/_llm_compiler.py), built by `_build_llm_compiler` in [`_builders.py`](../../src/ai_arch_toolkit/toolkit/agents/_builders.py). State as of 2026-09-24.

| Aspect | Toolkit behaviour |
|---|---|
| Plan | `$n. description [deps: …]` lines — nodes are **natural-language subtasks**, not tool calls with arguments |
| Execution | each node runs a full **inner ReAct sub-agent** (`max_react_iterations` = spec `max_iterations`, 10; 3 via the factory) that receives the whole task as its user message and the subtask in its system prompt. Concurrency keeps the latency benefit, but the paper's savings in LLM calls and cost do **not** carry over — this is closer to a parallel plan-and-execute |
| Scheduling | waves: every ready task is gathered, then the next wave (paper: dispatch as soon as each dependency resolves); no plan streaming |
| Dependencies | results substituted into descriptions with `str.replace` — `$1` **corrupts** `$10` (fix tracked separately); tasks depending on failed or unknown tasks are marked failed |
| Joiner / replan | `REPLAN` on the joiner's first line re-calls the planner with **only the task** — no previous results, no joiner reason — so it can regenerate the same plan; `max_replans` = 2 |
| Last round | if the joiner still says `REPLAN` on the last allowed round, that text is **returned as the answer** (the reference code switches to a final-answer prompt) |
| Phases | `planner_llm`, `executor_llm`, `executor_tools`, `joiner_llm`; knobs `planner_system` (`{tools}` token), `joiner_system`, `max_replans` |

**Recommended changes:** let nodes be direct tool calls (with an optional ReAct node type for open subtasks), feed previous results and the joiner's reason into replans, force a final answer on the last round, and dispatch tasks as their dependencies resolve.

## 13. Open questions

1. Does ReAct + native parallel calls + interleaved thinking close the remaining gap with 2025–26 models? No head-to-head exists.
2. How often do joiners trigger replans, and do replans converge?
3. With reasoning planners, how slow must tools be before whole-DAG planning pays off?
4. How accurate is dependency prediction with current models on real tool catalogs?
5. What should happen when some parallel side-effecting calls fail and others succeed?

## Sources

1. Kim et al., *An LLM Compiler for Parallel Function Calling*, ICML 2024 — https://arxiv.org/abs/2312.04511
2. PMLR version — https://proceedings.mlr.press/v235/kim24y.html
3. Reference code — https://github.com/SqueezeAILab/LLMCompiler
4. Reference implementation (`max_replans` loop) — https://raw.githubusercontent.com/SqueezeAILab/LLMCompiler/main/src/llm_compiler/llm_compiler.py
5. Reference configs — https://raw.githubusercontent.com/SqueezeAILab/LLMCompiler/main/configs/hotpotqa/configs.py
6. Kim, Shin, Chung, Rhu (KAIST), agent systems characterization, 2025 — https://arxiv.org/abs/2506.04301
7. Gim et al., *AsyncLM* — https://arxiv.org/abs/2412.07017
8. Singh et al., *LLM-Tool Compiler* — https://arxiv.org/abs/2405.17438
9. Patil et al., *BFCL*, ICML 2025 — https://proceedings.mlr.press/v267/patil25a.html · https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html
10. LangChain, *Planning Agents* — https://www.langchain.com/blog/planning-agents
11. Shen et al., *TaskBench*, NeurIPS 2024 — https://arxiv.org/abs/2311.18760
12. OpenAI docs, function calling — https://developers.openai.com/api/docs/guides/function-calling
13. Anthropic docs, parallel tool use — https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use
14. Anthropic docs, extended thinking — https://platform.claude.com/docs/en/build-with-claude/extended-thinking
15. Wu et al., *GAP* — https://arxiv.org/abs/2510.25320
16. Zhu et al., *DTA-Llama*, ACL 2025 — https://aclanthology.org/2025.acl-long.1401/
17. Zhao et al., *ParallelSearch* — https://arxiv.org/abs/2508.09303
18. Qin et al., *Flash-Searcher* — https://arxiv.org/abs/2509.25301
19. Liu et al., *LLMOrch*, IEEE TSE 2025 — https://arxiv.org/abs/2504.14872
20. Feng et al., *AsyncFC*, 2026 — https://arxiv.org/abs/2605.15077
21. Liu et al., *Parallel-Synthesis*, 2026 — https://arxiv.org/abs/2606.14672
22. Yu et al., *ParaGUIBench*, 2026 — https://arxiv.org/abs/2607.22689
23. *When Tools Fail: Benchmarking Dynamic Replanning and Anomaly Recovery in LLM Agents* (ToolMaze), 2026 — https://arxiv.org/abs/2606.05806
24. *ParaRecover*, EMNLP 2026 — https://arxiv.org/abs/2609.12345
25. Graph learning for LLM planning, NeurIPS 2024 — https://arxiv.org/abs/2405.19119
