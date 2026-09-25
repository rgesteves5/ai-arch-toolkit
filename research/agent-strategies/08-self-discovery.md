# SELF-DISCOVER (`self_discovery`)

> Before solving, the model composes a reasoning structure for the task: SELECT relevant reasoning modules, ADAPT them to the task, IMPLEMENT them as a step-by-step (key-value) plan; then solve by filling in the plan.
> Evidence: moderate for non-reasoning models in 2023–24 (BBH, T4D, MATH), with headline aggregates larger than the paper's own per-task table; the efficiency claim depends on discovering the structure **once per task** and reusing it; no evaluation on reasoning models.
> Last reviewed: 2026-09-24 · Part of [agent-strategies](00-index.md)

## At a glance

- **Task-level, not request-level.** On BBH and T4D the paper runs discovery *once per task* and reuses the structure for every instance; only MATH (and heterogeneous MMLU in the camera-ready) uses per-instance discovery [1][2]. The "10–40× fewer calls than CoT-self-consistency" claim assumes that amortisation.
- **Gains are real but smaller than advertised:** recomputed from the per-task table, GPT-4 gains +3.7 over CoT on BBH (the paper's aggregate says +6); PaLM 2-L +9.1 [1]. The largest gain (T4D, +33 for GPT-4) is on the authors' own non-public benchmark [1][4].
- **Where it helps:** knowledge- and language-understanding reasoning (sports understanding, ruin names, snarks) — ~+20 points on world-knowledge categories vs ~+5 on algorithmic ones (figure-read, PaLM 2-L) [1].
- **Where it hurts:** arithmetic, algorithmic and state-tracking tasks (GPT-4 multistep arithmetic −22, shuffled-object tracking −12 vs CoT); 74.7% of MATH errors were calculation errors [1].
- **2024–26:** vendors advise against prescribing reasoning steps to reasoning models [12][13][14]; reasoning models largely ignore instructions about their own reasoning trace [17]. What likely survives: a strong model writing a plan that a cheap non-reasoning model executes [1].
- **In this toolkit:** discovery runs per request with no cache, 10 custom modules instead of 39, and the solver is a tool-using ReAct loop (untested hybrid) — see [In ai-arch-toolkit](#12-in-ai-arch-toolkit).

## 1. Origin

**Zhou, Pujara, Ren, Chen, Cheng, Le, Chi, D. Zhou, Mishra, Zheng** (USC, Google DeepMind), "SELF-DISCOVER: Large Language Models Self-Compose Reasoning Structures", arXiv 2402.03620 (Feb 2024) [1]; NeurIPS 2024 — the camera-ready adds ToT/GoT baselines, token counts and an instance-level MMLU study [2]. The 39 reasoning modules come from Promptbreeder [1].

## 2. How it works

- **Stage 1 (three meta-prompts, per task):**
  - SELECT — pick the modules crucial for the task, given all 39 descriptions and a few *unlabeled* task examples;
  - ADAPT — rewrite each selected module to be task-specific;
  - IMPLEMENT — turn the adapted modules into a step-by-step key-value (JSON-like) plan, with one human-written structure for a different task as a demonstration.
- **Stage 2 (per instance):** prepend the structure; the model fills in the values in one call and gives the final answer.
- **Once per task vs per instance:** BBH and T4D — once per task ("we only need to run SELF-DISCOVER once for each task"); MATH — per instance with a one-shot demonstration, because problems vary; camera-ready — task-level for well-defined tasks, instance-level for open-domain ones [1][2]. The number of unlabeled examples is not stated (a human-comparison appendix used 3; the replication chose 10) [1][3].

## 3. Settings in the paper

- **Cost:** 3 calls per task + 1 per instance; structures average 224 tokens (BBH), 183 (T4D), 152 (MATH); inputs and outputs longer than CoT [2].
- **Models:** GPT-4 (gpt-4-turbo-preview), GPT-3.5-turbo (Oct–Dec 2023), instruction-tuned PaLM 2-L; Llama2-70B only as an executor, because its own structures were poor [1].
- **Answer extraction:** heuristics written after inspecting outputs; MATH answers hand-checked [1].

## 4. Evidence

| Benchmark | Model | Baseline | Result | Src |
|---|---|---|---|---|
| BBH, 23 tasks (paper aggregate) | PaLM 2-L / GPT-4 | CoT 60 / 75; Plan-and-Solve 61 / 73; direct 56 / 58 | 67 / 81 | [1] |
| BBH, recomputed from the per-task table | PaLM 2-L / GPT-4 | CoT 58.7 / 78.1 | 67.8 / 81.8 (+9.1 / +3.7) | [1] |
| BBH, worst GPT-4 tasks | GPT-4 | CoT: multistep arithmetic 92, shuffled objects 80, penguins 100, web of lies 80 | 70, 68, 90, 71 | [1] |
| BBH, best PaLM 2-L tasks | PaLM 2-L | CoT: sports understanding 47, ruin names 58, snarks 62 | 89, 90, 86 | [1] |
| T4D (authors' benchmark) | PaLM 2-L / GPT-4 | CoT 40 / 52; Plan-and-Solve 42 / 53 | 69 / 85 | [1] |
| MATH (200, per-instance structures) | PaLM 2-L / GPT-4 | CoT 42 / 71; Plan-and-Solve 49 / 70 | 50.5 / 73 | [1] |
| BBH / T4D / MATH vs zero-shot ToT and GoT | GPT-4 | ToT 76 / 50 / 69; GoT 75 / 52 / 70 | 81 / 85 / 73 | [2] |
| MMLU, 10 subjects × 50 (recomputed) | GPT-4 / PaLM 2-L | CoT 71.5 / 64.9 | task-level 73.1 / 65.8; per-instance 78.7 / 70.0 | [2] |
| Structure transfer | Llama2-70B with GPT-4's structures / GPT-3.5 (3-shot) | CoT 42 / 51 | 52 / 56 | [1] |
| Ablation on T4D (figure-read) | GPT-4 | CoT 50 | SELECT only 65 → +ADAPT 80 → +IMPLEMENT 85 | [1] |
| iSelf-Discover, MATH (per-instance) | Llama-3.1-405B / Mistral-Large | JSON plan 63.5 / 67.5 | free-text plan 75.5 / 76.5 | [3] |
| iSelf-Discover, BBH (27 tasks) | Llama-3.1-405B / Mistral-Large | task-level SELF-DISCOVER 86.59 / 82.04 | per-instance free text 87.27 / 85.57 | [3] |
| iSelf-Discover, T4D (re-created) | Llama-3.1-405B / Mistral-Large | task-level 100.00 / 96.63 | per-instance zero-shot 73–79 / 76–82 | [3] |
| Plan-and-Solve (related), 6 math sets | text-davinci-003 | zero-shot CoT 70.4 | PS 72.9; PS+ 76.7 | [5] |
| Buffer of Thoughts (related), Game of 24 / Checkmate-in-One | GPT-4 | ToT 74.0 / 49.2; Meta-Prompting 67.0 / 57.2 | 82.4 / 86.4 | [6] |
| Explicit CoT prompt on reasoning models, GPQA Diamond | o3-mini / o4-mini / Gemini 2.5 Flash | direct prompt | +2.9 / +3.1 / −3.3 pts; 20–80% longer responses | [15] |

**Claims vs confirmation:** the BBH aggregates do not match the per-task table; the "more than 20% better than CoT-SC at 10–40× fewer calls" comparison covers only 2 BBH tasks with GPT-4, and the values appear only in a chart; T4D is not public; the only independent replication [3] has no plain-CoT baseline, so it neither confirms nor refutes the CoT gains.

## 5. Strengths

- Cheap at scale **when amortised**: 3 calls per task, then 1 per instance [1][2].
- Largest gains on knowledge- and language-understanding reasoning [1].
- Interpretable plans: 87.5% of MATH structures judged correct (an expert could solve the problem by following them) [1].
- Structures transfer between models — PaLM 2-L's structures beat OPRO-optimised prompts on GPT-4 in 3 of 4 tasks; GPT-4's structures lifted Llama2-70B and GPT-3.5 [1].
- Zero-shot and label-free.

## 6. Weaknesses and costs

- **Per-instance mode = 4 sequential calls instead of 1** (plus, here, a ReAct loop).
- Longer prompts and outputs than CoT [2].
- Discovery needs a strong model (Llama2's own structures were poor) [1].
- Structure errors propagate: 12.5% of MATH structures were wrong (missing or extra steps) [1].
- Extra machinery: module list, meta-prompts, answer extraction.

## 7. Best use cases

- Recurring, well-defined task families with many instances — discover once, cache, reuse.
- Reasoning that mixes knowledge, pragmatics and social inference (sports understanding, ruin names, snarks, T4D-style theory of mind).
- Non-reasoning models, or a teacher → student setup where a strong model writes the plan and a cheap model executes it.
- When an auditable plan is itself useful.

## 8. Where it falls short

- **Algorithmic, state-tracking and arithmetic tasks regress vs CoT** — GPT-4 worse on 6 of 23 BBH tasks, PaLM 2-L on 3 [1].
- **Execution, not planning, dominates failures** — 74.7% of MATH errors were calculation errors (a code tool would help) [1].
- **Heterogeneous pools gain little from a task-level structure** (MMLU +1.6 for GPT-4, a −6 subject for PaLM 2-L) [2]; **per-instance discovery loses on homogeneous tasks** (re-created T4D) [3].
- **JSON formatting can hurt** — free-text plans scored +18.9% relative on MATH [3]; format restrictions degrade reasoning broadly [11].
- **Weak models cannot meta-reason** — Meta-Reasoning Prompting on GPT-3.5 fell below its best single methods [8].

## 9. Variants and follow-ups

- **Plan-and-Solve** — fixed "devise a plan, then solve" prompt; the paper's main baseline [5].
- **Buffer of Thoughts** (NeurIPS 2024) — reusable thought templates in a meta-buffer; ~12% of the cost of multi-query methods [6]; **ReasonFlux** — ~500 templates learned with hierarchical RL; 91.2% on MATH [10].
- **Meta-Prompting** — a conductor model coordinating expert instances of itself (+15–17 points with a Python interpreter) [7].
- **Meta-Reasoning Prompting** — picks one reasoning method per input; never best on any benchmark but most robust on average [8].
- **StrategyLLM** (NeurIPS 2024) — induces a task-level strategy from examples [9].
- **iSelf-Discover** — per-instance, free-text variant [3]; **LangGraph tutorial** — a popular reimplementation that runs all stages per query [18].

## 10. With 2024–2026 models

- **Vendor guidance:** OpenAI advises against CoT prompts and recommends zero-shot for reasoning models, positioning them as planners that orchestrate faster models [12]; Anthropic advises general instructions ("think thoroughly") over hand-written step plans, noting its models' reasoning often exceeds what a human would prescribe [13]; DeepSeek reports few-shot prompting degrades R1 [14].
- **Independent evidence:** prompted CoT gives small or negative gains on reasoning models at 20–80% more time [15].
- **Structure through fine-tuning** helped models under 30B, while decomposition-style structure degraded 32B models [16] (training-time evidence).
- **Reasoning models largely ignore instructions about their reasoning trace** (<25% comply), so a prescribed plan may not govern hidden reasoning [17].
- **No published evaluation of SELF-DISCOVER on o-series, R1 or Claude-thinking models.**
- **Likely still holds:** plan transfer from a strong planner to a cheap executor, and gains for non-reasoning models.

## 11. Router signals

**Choose `self_discovery` when:**
- the executing model is non-reasoning, and a ≥GPT-4-class model can do discovery;
- the workload is a recurring task type (discover once and cache);
- the task is knowledge- or NLU-heavy multi-step reasoning;
- a plan artifact is wanted, or a planner model is paired with an executor model;
- per-instance mode only for heterogeneous questions where ~4 sequential calls are acceptable.

**Avoid it when:**
- the model is a reasoning model;
- the task is arithmetic, algorithmic, state-tracking or code-executable → `react` with a code tool;
- the task is agentic and tool-heavy (no evidence either way);
- the discovering model is weak;
- the query is one-off and latency-sensitive.

## 12. In ai-arch-toolkit

Flow: [`_self_discovery.py`](../../src/ai_arch_toolkit/toolkit/agents/flows/_self_discovery.py), built by `_build_self_discovery` in [`_builders.py`](../../src/ai_arch_toolkit/toolkit/agents/_builders.py). State as of 2026-09-24.

| Aspect | Paper | Toolkit |
|---|---|---|
| Discovery scope | once per task (BBH, T4D), per instance for MATH / open-domain | **per request, no cache** — 3 extra sequential calls every time, so the amortised efficiency never applies |
| Modules | 39 (from Promptbreeder) | 10 custom defaults; replaceable with the `modules` knob |
| Inputs to SELECT | module descriptions + a few unlabeled task examples | the request only |
| IMPLEMENT | key-value plan, one demonstration structure | free-text step plan, no demonstration (the replication found free text equal or better) [3] |
| Solve | one call without tools | inner **ReAct with tools** (`max_iterations` 10) — an untested hybrid; the paper does suggest tools for its dominant failure (calculation errors) |
| Phases | — | `reasoning_llm` (select/adapt/operationalize), `solver_llm`, `solver_tools`; knobs `modules`, `select_system`, `adapt_system`, `plan_system`, `solver_system` |

**Recommended changes:** cache discovered structures per task type (e.g. keyed by a router label) so discovery is paid once; use a strong `reasoning_llm` with a cheaper `solver_llm`; consider the paper's 39-module list; route to this strategy only for non-reasoning executors.

## 13. Open questions

- Does a task-level cache pay off on real traffic, and how should task types be keyed?
- Does SELF-DISCOVER help or hurt reasoning models? (No published data.)
- Does the hybrid (discovered plan + tool-using ReAct solver) beat plain ReAct?
- Do 39 modules beat 10? Can the T4D gains be reproduced on public data?

## Sources

1. Zhou et al., *SELF-DISCOVER*, arXiv v1 — https://arxiv.org/abs/2402.03620
2. *SELF-DISCOVER*, NeurIPS 2024 camera-ready — https://proceedings.neurips.cc/paper_files/paper/2024/hash/e41efb03e20ca3c231940a3c6917ef6f-Abstract-Conference.html
3. Gunasekara & Ratnayake, *iSelf-Discover*, 2025 — https://arxiv.org/abs/2507.03347
4. Zhou et al., *How FaR Are Large Language Models From Agents with Theory-of-Mind?* (T4D), 2023 — https://arxiv.org/abs/2310.03051
5. Wang et al., *Plan-and-Solve Prompting*, ACL 2023 — https://arxiv.org/abs/2305.04091
6. Yang et al., *Buffer of Thoughts*, NeurIPS 2024 — https://arxiv.org/abs/2406.04271
7. Suzgun & Kalai, *Meta-Prompting*, 2024 — https://arxiv.org/abs/2401.12954
8. Gao et al., *Meta Reasoning for Large Language Models*, 2024 — https://arxiv.org/abs/2406.11698
9. Gao et al., *StrategyLLM*, NeurIPS 2024 — https://arxiv.org/abs/2311.08803
10. Yang et al., *ReasonFlux*, 2025 — https://arxiv.org/abs/2502.06772
11. Tam et al., *Let Me Speak Freely?*, EMNLP 2024 Industry — https://aclanthology.org/2024.emnlp-industry.91/
12. OpenAI, *Reasoning best practices* — https://developers.openai.com/api/docs/guides/reasoning-best-practices
13. Anthropic, *Prompting best practices* — https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices
14. DeepSeek-AI, *DeepSeek-R1*, 2025 — https://arxiv.org/abs/2501.12948
15. Meincke et al., *The Decreasing Value of Chain of Thought in Prompting*, 2025 — https://arxiv.org/abs/2506.07142
16. Wen et al., *ThinkPatterns-21k*, 2025 — https://arxiv.org/abs/2503.12918
17. Kwon et al., *ReasonIF*, ACL Findings 2026 — https://arxiv.org/abs/2510.15211
18. LangGraph Self-Discover tutorial — https://github.com/langchain-ai/langgraph/blob/23961cff61a42b52525f3b20b4094d8d2fba1744/docs/docs/tutorials/self-discover/self-discover.ipynb
