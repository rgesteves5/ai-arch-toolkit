# Plan-and-Execute (`plan_execute`)

> A planner writes an ordered list of sub-goals; a tool-using executor (a small ReAct agent) carries out each one; a replanner may revise the rest; a solver writes the answer.
> Evidence: mixed. A plan fixed up front loses to ReAct when later steps depend on what tools return; replanning *with execution feedback* gives large gains; lasting value is cost control, model split by role, auditability and security.
> Last reviewed: 2026-09-24 · Part of [agent-strategies](00-index.md)

## At a glance

- **Accuracy vs ReAct:** not reliably better. A fixed plan scored 17 vs ReAct's 32 on WebShop [9]; averaged over 11 current models on MobilityBench (2026), 62.6 vs 64.3, winning for 4 of 11 [17].
- **Where it wins:** sub-goals readable from the request, long horizons where drifting from the goal is the main risk, write-heavy deliverables, and a strong planner paired with a cheap executor [9][16][18][19].
- **Replanning is the active ingredient** — but only when the planner sees what happened: DEPS rose from 28.6 to 79.8 on one Minecraft task group (MT1) as replan rounds went from 0 to unlimited [7]; Plan-and-Act's dynamic replanning added ~10 points on WebArena-Lite [10]. Blind retries helped little [9].
- **Cost:** more LLM calls than ReAct, but each executor's context stays short — often fewer tokens overall (65% and 41% of ReAct's tokens on LegalAgentBench) [16][17].
- **Security:** a plan approved before execution supports human review and least privilege — but only binding each step to its tools *before* reading untrusted data protects control flow [20][21].
- **In this toolkit:** replanning fires only on exceptions and receives no feedback, and executors never see the original task — see [In ai-arch-toolkit](#12-in-ai-arch-toolkit).

## 1. Origin

No single paper defines the pattern. The toolkit's flow — planner, per-step ReAct executor, solver, replan loop — is LangChain's 2023 design [3], which credits BabyAGI [2] and Plan-and-Solve [1]; the LangGraph version replans after every step [4][5].

- **Plan-and-Solve (PS/PS+)**, Wang et al., ACL 2023 — a single-call precursor: plan, then carry the plan out, in one prompt [1].
- **Academic variants:** HuggingGPT (NeurIPS 2023) [6], DEPS (NeurIPS 2023) [7], RestGPT [8], ADaPT (NAACL 2024 Findings) [9], Plan-and-Act (ICML 2025) [10].
- **Critiques and benchmarks:** PlanBench [11], LLM-Modulo (ICML 2024) [12], o1 on PlanBench [13], TravelPlanner (ICML 2024) [14], a planning survey [15].

## 2. How it works

1. A planner call turns the task into ordered natural-language sub-goals (a numbered list here; JSON tasks with dependencies in HuggingGPT).
2. Each sub-goal goes to an executor — a small tool-calling ReAct agent that reads tool outputs and produces a step result. Results carry forward.
3. A replanner may revise the remaining plan. Four schedules appear in the literature: after every step (LangGraph, Plan-and-Act), only on failure (ADaPT, DEPS), when a stall counter crosses a threshold (Magentic-One), or never (the fixed-plan baseline).
4. A solver writes the answer from the task, the plan and the step results.

Unlike ReWOO, only the *step list* is fixed in advance; inside each step the executor still reacts to tool outputs.

## 3. Settings and cost in the papers

| Source | Models per role | Plan / replan settings | Budget and cost |
|---|---|---|---|
| Plan-and-Solve [1] | text-davinci-003, one call | T=0; self-consistency variant N=10 at T=0.7 | — |
| HuggingGPT [6] | gpt-3.5-turbo / text-davinci-003 / gpt-4 controller; HF models execute | one planning call; tasks with dependency ids and `<resource>-task_id` placeholders | many round-trips, token limits and instability listed as limitations |
| DEPS [7] | code-davinci-002 planner | describe, explain, replan after each failure; ~7–8 rounds (context-bound) | 30 runs per task |
| ADaPT [9] | gpt-3.5-turbo-instruct; also GPT-4, LLaMA-2-70B, Lemur-70B | recursion depth 3–4 | executor 15–20 steps; ReAct baseline 45–60 |
| Plan-and-Act [10] | fine-tuned LLaMA-3.3-70B planner and executor | rewrite the plan after every executor step; 10k + 5k synthetic plans | one extra planner call per action |
| LegalAgentBench [16] | 8 LLMs incl. GPT-4o, Claude-3.5-Sonnet | re-assess after each task; ReAct ≤10 iterations | tokens over 300 tasks reported |
| MobilityBench [17] | 11 LLMs (GPT-4.1, GPT-5.2, Claude 4.5, Gemini 3, DeepSeek-V3.2, Qwen3) | Plan-and-Execute vs ReAct, ≤10 steps | ReAct used ~35% more input tokens |
| Magentic-One [18] | GPT-4o (+o1 variant) | task ledger + progress ledger; replan when stalled >2 | — |

## 4. Evidence

| Benchmark | Model | Baseline | Result | Src |
|---|---|---|---|---|
| 6 math datasets, zero-shot | text-davinci-003 | zero-shot CoT 70.4 avg | PS+ 76.7 (8-shot CoT 77.6) | [1] |
| HuggingGPT plan accuracy (human-annotated) | GPT-4 / GPT-3.5 | — | sequential 41.36% / 18.18%; graph 58.33% / 20.83% | [6] |
| Minecraft (DEPS) | Codex planner | Inner Monologue 21.64 | 48.56; replan rounds 0 → unlimited: MT1 28.6 → 79.8, MT3 15.1 → 62.4 | [7] |
| ALFWorld / WebShop / TextCraft | GPT-3.5 | ReAct 43.3 / **32.0** / 19.0 | fixed plan 43.3 / **17.0** / 27.0; ADaPT 71.6 / 44.0 / 52.0 | [9] |
| ALFWorld (dev) | GPT-3.5 planner + LLaMA-2-70B executor | executor alone 20.4 | 43.3 | [9] |
| RestBench (TMDB / Spotify) | text-davinci-003 | ReAct 44.0 / 54.5 | fully offline plan 29.0 / 14.5; online planner (RestGPT) 75.0 / 72.7 | [8] |
| Multi-tool tasks, end to end | ChatGPT | one step at a time 55% | plan everything at once 50% (small test set) | [43] |
| HotpotQA / StrategyQA / GSM8K | LLaMA-2-7B fine-tuned | iterative 45.9 / 65.7 / 47.1 | one-pass planning 39.2 / 60.6 / 50.5 | [44] |
| WebArena-Lite | LLaMA-3.3-70B | executor alone 36.36 | untrained planner **17.16**; trained static planner 43.63 → dynamic replanning **53.94** → + CoT 57.58 | [10] |
| ALFWorld / ScienceWorld / Jericho | GPT-4o, GPT-4o-mini, Llama-3.1-8B | ReAct 56.0 / 36.8 / 22.5 (avg) | plan once, then act 55.0 / 39.7 / 18.0 | [26] |
| LegalAgentBench (300) | GPT-4o / Claude-3.5-Sonnet | ReAct 0.791 / 0.658 | 0.491 at 65% of ReAct's tokens / 0.670 at 41%; writing tasks (GPT-4o) 0.864 vs 0.654 | [16] |
| MobilityBench (2026) | 11 current LLMs | ReAct mean 64.3 | 62.6; wins for 4/11 (GPT-4.1, Claude Sonnet/Opus 4.5, Qwen3-4B) | [17] |
| GAIA (validation) | GPT-4o | Magentic-One without ledgers | removing the ledgers costs 31% (relative) | [18] |
| Internal research eval (vendor) | Opus 4 lead + Sonnet 4 workers | single-agent Opus 4 | +90.2% at ~15× chat tokens | [19] |
| TravelPlanner | GPT-4-Turbo | — | 0.6% (search then plan), 4.4% (all info given); LLM → SMT solver 93.9% | [14][30] |
| PlanBench | o1-preview | earlier LLMs | Blocksworld 97.8%; Mystery 52.8%; 20–40-step problems 23.6%; ~65× GPT-4o's cost | [13] |
| IPC-2023 learning track | Gemini 3.1 Pro / GPT-5 | classical planners | 245 vs 234 (best planner); GPT-5 drops 25.9% with obfuscated names | [34] |
| AgentDojo (security) | GPT-4o | no defence | choosing tools before reading any data: targeted attack success 57.69% → 6.84% | [23] |

## 5. Strengths

- **Keeps long tasks on track:** less task forgetting [15]; removing Magentic-One's ledgers cost 31% on GAIA [18]; Anthropic's lead agent writes its plan to memory before its context fills [19].
- **Splits models by role:** a GPT-3.5 planner roughly doubled a LLaMA-2-70B executor on ALFWorld [9]; Opus lead + Sonnet workers +90.2% over Opus alone (vendor) [19]; CaMeL lost ~1% utility for ~12% lower cost with a cheaper model on untrusted data [20].
- **Often fewer tokens than ReAct** because each executor's context stays short [16][17].
- **Wins when the structure is known up front:** TextCraft recipes [9], writing/deliverable tasks [16], route planning under stated preferences [17]; a fixed three-phase pipeline (Agentless) reached 32.00% on SWE-bench Lite at $0.70 per issue [33].
- **Reviewable before anything runs** — human approval, least privilege [21][22].
- **Replanning with execution feedback** has a natural place to go [7][10].

## 6. Weaknesses and costs

- **More, sequential LLM calls** — LangChain's own post names this as the main downside [3]; each step is a full agent loop [5].
- **Replanning frequency is a trade-off:** planning at every step was expensive and *hurt* long-horizon performance; never planning was also worse; the best frequency was intermediate [24].
- **Reasoning-model planners are expensive:** o1-preview ~65× GPT-4o per PlanBench instance [13]; thinking mode raised success by up to 5.98 points on MobilityBench with markedly more tokens and latency [17].
- **Multi-agent delegation multiplies cost** (~15× chat tokens) [19]; 14 failure modes catalogued, and gains over a single agent are often minimal [25].

## 7. Best use cases

1. Sub-goals and their order can be read off the request and do not depend on tool results [9][16][17].
2. Horizons long enough that goal drift is the main risk [10][18][19].
3. Sub-tasks need different tools or models, or a cheap executor needs a strong planner [9][19].
4. The plan must be approved or audited before execution [21][22].
5. The backbone is GPT-4o-class or better — weaker models do not follow plans well [16][26].
6. The budget allows feedback-driven replanning in dynamic environments [7][10].

## 8. Where it falls short

- **Plans go stale when observations matter:** fixed plan 17 vs ReAct 32 on WebShop [9]; offline plans 29.0 / 14.5 vs ReAct 44.0 / 54.5 on RestBench [8]; static planners could not handle run-time content [10]; ReAct won multi-hop tasks for 7 of 8 models [16].
- **A generic planner can hurt a good executor:** WebArena-Lite 36.36 → 17.16 [10].
- **Plan quality was poor in 2023:** 41% of sequential plans right with GPT-4 [6]; wrong APIs and parameters used before they were obtained [8]; TravelPlanner 0.6% [14].
- **Granularity mismatch:** a fixed step can be far harder than the planner expected — ADaPT's motivating case [9].
- **Weak models:** Llama-3.1-8B 18.0 vs 22.5 for ReAct [26].
- **Errors propagate** unless there is a way to adjust [15].
- **Security:** an executor running ReAct sees untrusted tool output and can call any tool it has — no control-flow integrity unless every step is bound to its tools before any data is read, and even then arguments stay exposed [20][21].

## 9. Variants and follow-ups

- **LangGraph plan-and-execute** — replan after each step; early final answer [5].
- **ADaPT** — execute first, decompose recursively only on failure [9].
- **DEPS** — explain the failure, then replan [7]; **RestGPT** — coarse-to-fine online planning over REST APIs [8].
- **Plan-and-Act** — trained planner, synthetic plans, dynamic replanning [10].
- **Magentic-One** — task and progress ledgers; replan when stalled [18].
- **Orchestrator-workers** — sub-tasks decided at run time, parallel workers [27][19].
- **AdaPlanner** [28]; **Tree-Planner** (ICLR 2024, −92.2% tokens) [29]; **LLM-Modulo** with sound external critics [12]; SMT-solver planning (93.9% TravelPlanner) [30]; **Pre-Act** [31]; **Agentic Plan Caching** (NeurIPS 2025: −50.31% cost, −27.28% latency) [32].
- **Plan-then-execute security pattern and CaMeL** [21][20].

## 10. With 2024–2026 models

- **Writing a plan for a fully specified problem is no longer the bottleneck** for frontier models: o1-preview 97.8% on Blocksworld [13]; the best models match classical planners on IPC tasks, though planners remain far cheaper [34]. The "reasoning collapse" claim is contested [35][36]. Missing information and cost are the bottlenecks now.
- **Adaptivity still decides agent tasks** in 2025–26 data [10][16][17].
- **Reasoning models plan inside every call** (interleaved thinking; tools inside the chain of thought) [37][38]. A separate plan-once prompt roughly matches ReAct on GPT-4o [26]; too much deliberation hurts agents (less overthinking → ~30% better at 43% lower cost on SWE-bench Verified) [39]; Plan-and-Solve-style prompts add little for reasoning models [45].
- **What still justifies an explicit planner:** splitting models by role, audit, long-horizon memory, security, parallel delegation.
- **Prompt caching** (reused prefixes at ~10% of input price) narrows plan-first's token advantage over ReAct's growing context (inference, not measured) [40][41].
- **Vendor guidance:** maximise a single agent before splitting [42]; orchestrator-workers only when sub-tasks cannot be predicted [27].

## 11. Router signals

**Choose `plan_execute` when:**
- the request names several deliverables or ≥3 sub-goals whose identity does not depend on tool results;
- the task is long (>~10 expected tool calls) or write-heavy (reports, multi-part answers);
- a stronger model can plan while a cheaper one executes;
- the plan needs approval;
- the backbone is GPT-4o-class or better.

**Avoid it when:**
- next actions depend on content that comes back (search → pick, multi-hop lookups, "do what the email says") → `react`;
- one or two tool calls suffice → `react`; no tools → `completion`;
- the model is small and no planner has been trained;
- the turn is latency-critical and interactive;
- executors hold write tools and the data is untrusted → a fixed tool plan (`rewoo`, CaMeL-style).

**Escalation:** when a step reports a failed precondition, replan *with the step results* (not yet supported here); when the input already implies a fixed-hop tool chain, prefer `rewoo`.

## 12. In ai-arch-toolkit

Flow: [`_plan_execute.py`](../../src/ai_arch_toolkit/toolkit/agents/flows/_plan_execute.py), built by `_build_plan_execute` in [`_builders.py`](../../src/ai_arch_toolkit/toolkit/agents/_builders.py). State as of 2026-09-24.

| Aspect | Toolkit behaviour |
|---|---|
| Plan format | numbered list parsed by regex; a plan that is not a numbered list yields **zero steps** and the solver answers with no tool results |
| Executor | inner ReAct per step; `max_iterations_per_step` defaults to the spec's `max_iterations` (10) via `Agent` (3 when calling the factory directly) |
| Executor context | user message = the step text only; system prompt = base system + "Current step" + **all** previous step results. It never sees the original task or the full plan (LangGraph and Plan-and-Act pass the plan) |
| Replan trigger | only when a step's ReAct run **raised an error**. Tool error strings (toolkit tools return errors as text) and wrong-but-fluent answers never trigger it. `max_replans` = 1 |
| Replan input | the **same prompt** as the first plan — no failed plan, no step results, no reason — so it can regenerate the same plan; the gains in DEPS and Plan-and-Act come from feeding execution back [7][10] |
| Replan scope | reruns every step, discarding the ones that succeeded (the waste ADaPT criticises) [9] |
| Solver | task + plan + step results |
| LLM calls | ≤ (1 + k·m)(1 + R) + 1 for k steps, m iterations per step, R replans — e.g. k=5, m=3, R=1: ≤33 (≈12 with one tool call per step); with the Agent default m=10: ≤103 |
| Phases | `planner_llm`, `executor_llm`, `executor_tools`, `solver_llm`; knobs `planner_system` (`{tools}` token), `solver_system`, `max_replans`, `max_iterations_per_step` |

**Recommended changes (research-backed):** pass the original task and the plan to each executor; trigger replanning on semantic failure signals (tool error results, empty evidence) and send the planner the previous plan, the step results and the failure; keep completed steps (ADaPT-style "decompose only what failed").

## 13. Open questions

- Is a planner call still worth it on top of a reasoning model that plans internally, at matched budgets?
- Which replanning trigger is best under a budget — every step, on failure, a stall counter, or learned [24]?
- How to detect a step's *semantic* failure cheaply, given that LLM self-verification is unreliable [12]?
- Net token and latency figures once prompt caching is on.

## Sources

1. Wang et al., *Plan-and-Solve Prompting*, ACL 2023 — https://aclanthology.org/2023.acl-long.147/
2. Nakajima, *BabyAGI* — https://yoheinakajima.com/birth-of-babyagi/ · https://github.com/yoheinakajima/babyagi
3. LangChain, *Plan-and-Execute Agents* (2023) — https://www.langchain.com/blog/plan-and-execute-agents
4. LangChain, *Planning Agents* (2024) — https://www.langchain.com/blog/planning-agents
5. LangGraph plan-and-execute tutorial — https://github.com/langchain-ai/langgraph/blob/0.3.0/docs/docs/tutorials/plan-and-execute/plan-and-execute.ipynb
6. Shen et al., *HuggingGPT*, NeurIPS 2023 — https://arxiv.org/abs/2303.17580
7. Wang et al., *DEPS*, NeurIPS 2023 — https://arxiv.org/abs/2302.01560
8. Song et al., *RestGPT* — https://arxiv.org/abs/2306.06624
9. Prasad et al., *ADaPT*, NAACL 2024 Findings — https://aclanthology.org/2024.findings-naacl.264/
10. Erdogan et al., *Plan-and-Act*, ICML 2025 — https://arxiv.org/abs/2503.09572
11. Valmeekam et al., *PlanBench* — https://arxiv.org/abs/2206.10498
12. Kambhampati et al., *LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks*, ICML 2024 — https://arxiv.org/abs/2402.01817
13. Valmeekam et al., o1 on PlanBench — https://arxiv.org/abs/2409.13373
14. Xie et al., *TravelPlanner*, ICML 2024 — https://arxiv.org/abs/2402.01622
15. Huang et al., *Understanding the planning of LLM agents: A survey* — https://arxiv.org/abs/2402.02716
16. *LegalAgentBench*, ACL 2025 — https://aclanthology.org/2025.acl-long.116/
17. *MobilityBench*, 2026 — https://arxiv.org/abs/2602.22638
18. Fourney et al., *Magentic-One* — https://arxiv.org/abs/2411.04468
19. Anthropic, *How we built our multi-agent research system* — https://www.anthropic.com/engineering/multi-agent-research-system
20. Debenedetti et al., *CaMeL* — https://arxiv.org/abs/2503.18813
21. Beurer-Kellner et al., *Design Patterns for Securing LLM Agents against Prompt Injections* — https://arxiv.org/abs/2506.08837
22. Del Rosario et al., secure Plan-then-Execute guide — https://arxiv.org/abs/2509.08646
23. Debenedetti et al., *AgentDojo*, NeurIPS 2024 D&B — https://arxiv.org/abs/2406.13352
24. Paglieri et al., *Learning When to Plan* — https://arxiv.org/abs/2509.03581
25. Cemri et al., *Why Do Multi-Agent LLM Systems Fail?* — https://arxiv.org/abs/2503.13657
26. Kim et al., *ReflAct* — https://arxiv.org/abs/2505.15182
27. Anthropic, *Building effective agents* — https://www.anthropic.com/engineering/building-effective-agents
28. Sun et al., *AdaPlanner* — https://arxiv.org/abs/2305.16653
29. Hu et al., *Tree-Planner*, ICLR 2024 — https://arxiv.org/abs/2310.08582
30. Hao et al., planning with formal verification tools — https://arxiv.org/abs/2404.11891
31. Rawat et al., *Pre-Act* — https://arxiv.org/abs/2505.09970
32. Zhang et al., *Agentic Plan Caching*, NeurIPS 2025 — https://arxiv.org/abs/2506.14852
33. Xia et al., *Agentless* — https://arxiv.org/abs/2407.01489
34. Corrêa et al., *Frontier LLMs Rival State-of-the-Art Planners* — https://arxiv.org/abs/2511.09378
35. Shojaee et al., *The Illusion of Thinking* — https://arxiv.org/abs/2506.06941
36. Lawsen, comment on *The Illusion of Thinking* — https://arxiv.org/abs/2506.09250
37. Anthropic docs, extended and interleaved thinking — https://platform.claude.com/docs/en/build-with-claude/extended-thinking
38. OpenAI, *o3 and o4-mini System Card* — https://cdn.openai.com/pdf/2221c875-02dc-4789-800b-e7758f3722c1/o3-and-o4-mini-system-card.pdf
39. Cuadron et al., *The Danger of Overthinking* — https://arxiv.org/abs/2502.08235
40. Anthropic docs, prompt caching — https://platform.claude.com/docs/en/docs/build-with-claude/prompt-caching
41. OpenAI docs, prompt caching — https://developers.openai.com/api/docs/guides/prompt-caching
42. OpenAI, *A practical guide to building agents* — https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf
43. Ruan et al., *TPTU* — https://arxiv.org/abs/2308.03427
44. Yin et al., *Agent Lumos*, ACL 2024 — https://aclanthology.org/2024.acl-long.670/
45. Meincke et al., *The Decreasing Value of Chain of Thought* — https://arxiv.org/abs/2506.07142
