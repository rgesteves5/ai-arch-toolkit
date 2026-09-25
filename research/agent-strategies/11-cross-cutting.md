# Cross-cutting evidence: comparing and choosing strategies

> What no single-strategy page shows: cost-aware comparisons, the task properties that predict which architecture helps, what reasoning models changed, and research on choosing a strategy per query.
> Last reviewed: 2026-09-24 · Part of [agent-strategies](00-index.md)

## Key findings

1. **Counting cost changes the ranking.** In an independent HumanEval rerun, retrying GPT-4 with rising temperature scored 93.2% for $2.45; LATS 88.0% for $134.50; Reflexion 87.8% for $3.90 [1]. Across 21,730 HAL runs, the most expensive model was among the best accuracy-for-cost options in only 1 of 9 benchmarks [2].
2. **Fixed pipelines with a checker can match open-ended agents.** Agentless solved 32.00% of SWE-bench Lite at $0.70/issue vs SWE-agent's 18.33% at $2.53 — though it still spends compute (40 candidate patches, 40 generated tests) and picks by running tests; agents did better when the issue gave no hint where the bug was [3]. Task-specific scaffolds beat a general one in 9 of 12 and 11 of 12 HAL comparisons [2].
3. **Multi-agent gains are mostly extra compute on the right kind of task.** At equal budgets, the average multi-agent change was −0.3% (95% CI −58.7% to +77.2%), from +80.8% on decomposable finance tasks to −70.0% on sequential planning [20]. An independent re-test found automatically designed multi-agent systems rarely beat 5-sample self-consistency with GPT-5, at up to ~10× the cost [10].
4. **Four task properties have measured effects:** decomposability, sequential dependency, number of tools, and how well a single agent already does (above ~45% single-agent accuracy, adding agents hurts) [20]; vendor and practitioner reports agree [14][22].
5. **Checkability decides whether sampling, search and reflection pay off.** Repeated sampling raised SWE-bench Lite coverage from 15.9% to 56%, but without an automatic checker, voting and reward models plateau after a few hundred samples [39]. Self-correction works with reliable outside feedback, not self-critique [55][56]; Self-Refine averaged 70.7 vs 72.8 for one plain call [9].
6. **Reasoning models make prompt-level reasoning scaffolds mostly unnecessary** — vendors advise against step-by-step or planning prompts [30][31][34]; a CoT prompt moved GPQA by +2.9 / +3.1 / −3.3 points for o3-mini / o4-mini / Gemini 2.5 Flash at 20–80% more time [49]; one GPT-5 CoT call scored 87.14% vs CoT-SC's 87.35% at $7.24 vs $46.39 [10].
7. **…but they do not replace feedback loops.** On o1-mini, a debugger that runs code and iterates still added accuracy (96.3% vs 93.9%); the same method without iteration did not (94.4%) [50]. Reasoning models planned far better than standard models but fabricated tool results in one tool task (16.22% vs GPT-4o's 89.19%) [48]; they overthink in coding-agent loops [43].
8. **More thinking does not reliably help:** higher effort did not improve accuracy in 21 of 36 HAL runs [2]; the benefit of inference compute varies by task and shrinks as problems get harder [40]; longer chains are not consistently more accurate [41].
9. **Difficulty should decide compute:** allocating compute per prompt by difficulty was >4× as efficient as best-of-N [38]; more calls help easy queries and hurt hard ones [26].
10. **Choosing a strategy per query mainly saves cost, with room to improve:** Adaptive-RAG matched multi-step retrieval (F1 50.91 vs 50.87) at 44% of its relative time — with a classifier only 54.5% accurate; a perfect router reached 62.80 [57]. Learned routers report savings against other multi-agent systems, not against a single call [58][59].
11. **The evidence is uneven:** most comparisons come from the authors of one method, on 2023-era models, with unmatched budgets; every independent re-test (AATM, HAL, Kim et al., the multi-agent re-evaluation) shrank the claimed gains [1][2][10][20]. No controlled re-test of ToT, LATS, ReWOO, LLMCompiler or Self-Discover on current reasoning models exists.

## A. Cost-aware comparisons and simple baselines

**AI Agents That Matter** (arXiv July 2024; TMLR 2025) [1] — HumanEval, 164 problems, 5 runs:

| Approach | Accuracy | Total cost |
|---|---|---|
| GPT-4, single call | 89.6% | $1.93 |
| Retry (T=0, up to 5 tries on test failure) | 92.0% | $2.51 |
| Warming (retry with T raised 0 → 0.5) | 93.2% | $2.45 |
| Escalation (Llama-3-8B → larger → GPT-4 on failure) | 85.0% | $0.27 |
| LDB | 91.0% | $2.19 |
| Reflexion (GPT-4) | 87.8% | $3.90 |
| LATS (GPT-4) | 88.0% | $134.50 |

The authors leave open whether planning, reflection or debugging cause published gains; many published scores exceeded the best of their five runs (LATS paper: 92.7%; Reflexion paper: 91.0%) [1][8]. Joint cost/accuracy tuning cut DSPy's HotpotQA cost by 53% (GPT-3.5) and 41% (Llama-3-70B) at similar accuracy [1].

**Minimal loops also work with strong models:** Claude 3.5 Sonnet reached 49% on SWE-bench Verified with only bash and edit tools [4]; mini-swe-agent, a ~100-line bash-only agent, claims >74% (the page does not name the model) [5].

**HAL** [2] — 21,730 runs, 9 models, 9 benchmarks (~$40,000): Gemini 2.0 Flash was a best-value option on 7 of 9 benchmarks; more tokens went with higher accuracy on 6 of 9; task-specific scaffolds beat the generalist one; higher reasoning effort did not help in 21 of 36 runs.

**Same model, same benchmark, several strategies:**
- **Huang et al.** (text-davinci-003) [6] — ALFWorld success / total cost: CoT 0.43 / $98.60, CoT-SC 0.57 / $105.37, ReAct 0.57 / $152.18, Reflexion 0.71 / $220.17; HotpotQA: CoT 0.32 / $5.73, ReAct 0.34 / $66.00, Reflexion 0.39 / $112.49.
- **ReWOO's table** (gpt-3.5-turbo) [16] — HotpotQA: direct 37.8, CoT 41.6, ReAct 40.8 (9,795 tokens), ReWOO 42.4 (1,986 tokens); TriviaQA: direct 80.6 vs ReAct 59.4 and ReWOO 66.6.
- **ADaPT** (GPT-3.5) [7] — ReAct / plan-and-execute / Reflexion / ADaPT: ALFWorld 43.3 / 43.3 / 57.5 / 71.6; WebShop 32.0 / 17.0 / 35.0 / 44.0; TextCraft 19.0 / 27.0 / 32.0 / 52.0.
- **LATS** (GPT-3.5, HotpotQA EM) [8] — ReAct 0.32, best-of-k 0.38, ToT 0.39, Reflexion 0.51, LATS 0.63 (up to 50 trajectories).
- **AFlow's baselines** (gpt-4o-mini, 6 benchmarks) [9] — single call 72.8, CoT 74.7, CoT-SC 76.0, Self-Refine 70.7, ADAS 67.2, AFlow 80.3.
- **Multi-agent re-evaluation** (GPQA-Diamond, GPT-5) [10] — CoT 87.14% ($7.24); CoT-SC 87.35% ($46.39); MaAS 86.95% ($38.50); AFlow 84.13% ($274.60); ADAS 85.23% ($832.10). Seven of 14 AFlow workflows amounted to running one prompt three times and aggregating; ADAS's published results were chosen as the best candidate on the test set.
- **Efficient Agents** (GAIA) [11] — per-step best-of-N scored by a reward model moved accuracy 53.33% → 53.94% as N went 1 → 4 while tokens rose 243K → 325K.
- **Cost-of-Pass** [12] — majority voting and self-refinement rarely justify their cost; better models do more for cost-efficiency.
- **AgentArch** [13] — native function calling generally beat text ReAct; multi-agent ReAct consistently underperformed; succeeding on all 8 repeated trials happened ≤6.34% of the time.
- **LangChain** practitioner tests [14][15] — ReAct degraded as tools and context grew; a single agent fell off sharply with ≥2 distractor domains; fixing the supervisor implementation alone raised multi-agent performance by nearly 50%.

## B. Task properties that predict which architecture helps

**Kim et al., *Towards a Science of Scaling Agent Systems*** (v3) [20] — 260 configurations comparing a single agent with four multi-agent layouts (independent, centralized, decentralized, hybrid) on six benchmarks (Finance-Agent, BrowseComp-Plus, PlanCraft, WorkBench, SWE-bench Verified, Terminal-Bench), nine models, tools/prompts/compute held equal:
- **Decomposable tasks help:** Finance-Agent centralized +80.8%; parallel exploration small (BrowseComp-Plus decentralized +9.2%).
- **Sequential tasks suffer:** every multi-agent layout hurt PlanCraft (−39.1% to −70.0%).
- **Strong single agents leave little room:** above ~45% single-agent accuracy, extra agents give negative returns; SWE-bench Verified lost 2–15%.
- **Tool count:** tool-heavy tasks pay a coordination penalty.
- **Error amplification** (single agent = 1.0): independent 17.2×, decentralized 7.8×, hybrid 5.1×, centralized 4.4× — a central checkpoint contains errors.
- **Overhead:** successes per 1,000 tokens — single agent 67.7, independent 42.4, decentralized 23.9, centralized 21.5, hybrid 13.6.
- **Prediction:** a regression on these properties explains R² = 0.373 and picks the best layout in 87% of held-out cases. (v1 reported R² = 0.513; the 45% threshold and the 87% survived, the fit quality did not.)
- **Scope note:** these are multi-agent *topologies*, not the toolkit's ten reasoning strategies — the properties transfer as features; the mapping to strategies does not.

**Why multi-agent systems fail (MAST)** [21], NeurIPS 2025 — 1,600+ traces, 14 failure modes: specification problems 43.9%, inter-agent misalignment 32.15%, verification 23.95%; repeating steps (15.7%) and reasoning–action mismatch (13.2%) most common; clearer roles +9.4%, an added verification step +15.6%.

**Reconciling vendor and controlled results:** Anthropic's research system beat single-agent Opus 4 by 90.2% on an internal evaluation, with token usage explaining 80% of BrowseComp variance and ~15× chat tokens [22]; Kim et al. held budgets fixed and found −0.3% on average [20]. Multi-agent is best read as a way to spend more compute in parallel on decomposable work. Trying a single agent first and escalating improved accuracy 1.1–12% at up to 20% lower cost [23]; a single well-prompted agent nearly matches the best multi-agent discussion [24].

**Task type:** CoT helps mainly on math and logic (on MMLU only when an equals sign appears) [25]; majority-vote accuracy can rise then fall as calls increase [26]. Surveys offer taxonomies but no quantitative selection rules [6][27][28][29]. OpenAI puts o3/o4-mini "in distribution" below ~100 tools and 20 arguments per tool [31].

## C. Reasoning models vs scaffolds (2024–2026)

**Vendor guidance.**
- **OpenAI** — don't ask reasoning models to think step by step; start zero-shot; use a reasoning model to plan and a GPT model to execute; don't ask o3/o4-mini to plan more before each call; keep reasoning items between tool calls (τ-bench retail 73.9% → 78.2%) [30][31][32].
- **Anthropic** — start simple; use evaluator-optimizer only with clear criteria [33]; adaptive thinking beat fixed budgets internally and thinks between tool calls; prefer general instructions over prescriptive step lists; explicit chaining still helps to inspect intermediate outputs; the newest models may over-verify or overuse subagents [34][35]. Think-tool study (Claude 3.7 Sonnet, τ-bench airline): baseline 0.332, extended thinking 0.412, think tool 0.404, think tool + tuned prompt 0.584; a Dec 2025 update recommends extended thinking instead in most cases [36].
- **Google** — Gemini adjusts thinking automatically; low effort for lookup/classification, high for coding, math and planning [37].

**Re-tests with reasoning models.** CoT prompts on GPQA (results above; non-reasoning models gained more but made more errors on easy questions) [49]; code tasks — GPT-4o 90.4% → 94.5% with LDB, and the o1-mini result in the key findings [50]; the value of prompting techniques fades differently per model family [51]; a think tool lifted GPT-4.1 from 48.5% to 70.8% but barely moved o3-mini (55.8% → 56.7%) [13]; LaRMA — reasoning models peaked after 1–2 reflection rounds vs 4–5 for standard models, and a standard actor with a reasoning reflector scored best (up to 98.18%) [48]; overthinking — keeping the less-overthought runs gave ~30% better results at 43% lower cost on 4,018 SWE-bench trajectories [43].

**Test-time compute.** Per-prompt difficulty-based allocation >4× as efficient as best-of-N; a small model matched a 14× larger one where it could sometimes solve the problem; beam search for hard questions on small budgets, best-of-N for easy ones on large budgets [38]. Across 9 models and 8 tasks, R1 used ≥5× the tokens of Claude 3.7 Sonnet on AIME 2025 at similar accuracy; a perfect checker took GPT-4o from 42% to 95% on easy TSP instances but did not help hard ones [40]. Correct answers are often shorter than incorrect ones [41], and o1-style models overthink easy inputs [42]; mid-reasoning "aha" moments seldom improve accuracy, while uncertainty-triggered external nudges do [54].

**The "Illusion of Thinking" debate.** Three regimes (standard models win on easy puzzles, reasoning models on medium, both collapse on hard) [44]; rebuttals cite output limits and unsolvable instances [45]; a replication confirms Hanoi failures around 8 disks while solvable River Crossing instances with >100 pairs were solved [46]; giving the model tools reverses the collapse [47]. On long tasks small per-step gains compound, models degrade once their own errors are in context, and thinking reduces that effect [52]; more environment interaction is a separate lever from more thinking per step [53].

**Synthesis (inference):** reasoning models replace the "structure your reasoning" scaffolds — CoT prompts, prescriptive plans, Self-Discover-style structures and, probably, puzzle-search ToT. They do not replace the scaffolds that bring outside information in — environment loops, execution feedback, external checkers, parallel fan-out.

## D. Choosing a strategy per query

| Method | Chooses (granularity) | Features and labels | Reported result | Caveat |
|---|---|---|---|---|
| Adaptive-RAG (NAACL 2024) [57] | no / single-step / multi-step retrieval (per query) | query text, T5-Large classifier; label = simplest strategy that answered correctly, dataset type as fallback | see key finding 10 | classifier 54.5% accurate |
| PET-Select [65] | 9 prompting techniques (per query) | code-model embedding; label = best technique by correctness vs tokens | HumanEval, GPT-4o: +1.9% pass@1, −74.8% tokens | code only |
| Meta-Reasoning Prompting [62] | 7 methods incl. CoT, ToT, Self-Refine (per query) | the model scores method descriptions itself | GPT-4 avg 0.772 vs ToT alone 0.725 | failed with GPT-3.5 |
| SMART [64] | CoT, PoT, least-to-most (per query) | RL on its own outcomes | Gemma 7B GSM8K 40.4% → 55.6% | 7–8B, math only |
| Route to Reason (WWW 2026) [63] | model × {direct, CoT, PAL, CoD} (per query) | query + learned embeddings; every pair run on training data | 82.5% at 1,091 tokens vs QwQ-32B 80.0% at 2,745 | open-weight models |
| MasRouter (ACL 2025) [58] | collaboration mode, roles, LLM per agent (per query) | query embedding; RL on accuracy − cost | MBPP 84.00% at $1.039 vs AFlow 82.20% at $1.723 vs one gpt-4o-mini call 72.20% at $0.143 | savings measured against other multi-agent systems |
| MaAS (ICML 2025) [59] | operators (CoT, debate, CoT-SC, Self-Refine, ReAct, early exit) per layer (per query) | embeddings; accuracy − token cost | MATH 51.82% vs AFlow 51.28%; inference $0.42 vs $1.66 | ≈ CoT-SC in the independent re-test [10] |
| AFlow (ICLR 2025) [9] / ADAS (ICLR 2025) [60] | a workflow in code (per task) | search scored on a validation set | AFlow +5.7% avg; ADAS DROP F1 79.4 vs 65.8 | re-test and test-set selection issue [10] |
| FlowReasoner [61] | a generated multi-agent system (per query) | SFT on R1 outputs, then RL | 81.89% vs o1-mini 71.37% on code | no cost reported |
| AdaptThink / Thinkless [66][67] | think vs answer directly (per query) | RL | −53% response length, +2.4% accuracy; 50–90% less long thinking | small models, math |
| Snell et al. [38] | search method, sequential/parallel split (per difficulty bin) | per-model difficulty | >4× as efficient as best-of-N | needs trained reward and revision models |
| GPT-5 router [68] | fast vs thinking model (per request) | conversation type, complexity, tool needs, explicit intent; trained on switches, preferences, measured correctness | not disclosed | opaque |

**Pattern:** features are mostly a query embedding plus a difficulty estimate or the model's own judgement; labels usually come from running every candidate strategy on training data; accuracy moves a few points, while the larger effect is 50–75% fewer tokens against heavy baselines. DAAO (WWW 2026) adds difficulty-aware workflows; its numbers could not be verified [69].

## Comparison table

"(I)" marks inference rather than a sourced fact.

| Strategy | LLM calls per task | Latency | Adapts to observations | Needs tools | Needs outside evaluator | Strongest evidence | Main risk |
|---|---|---|---|---|---|---|---|
| `completion` | 1 | one call | no | no | no | [25][49][50][10] | nothing catches errors on tool or multi-step work (I) |
| `react` | ~1 per step (4.97 steps on HotpotQA [16]) | sequential; parallel calls within a step | every step | yes | no | minimal scaffolds [4]; native FC beats text ReAct [13] | too many tools or too much context [14]; overthinking [43] |
| `plan_execute` | 1 plan + steps × ReAct + solve + replans (I) | sequential (I) | between steps | yes | no | mixed: ADaPT [7]; planner/doer split [30][48] | brittle plans (WebShop 17.0 vs ReAct 32.0) [7] |
| `rewoo` | 2 + tool calls [16] | plan → tools → solve | no | yes | no | ReWOO paper only: 42.4% vs 40.8% at 5× fewer tokens [16] | must enumerate possibilities when the environment is unknown [16] |
| `reflexion` | tries × (attempt + evaluation + reflection) (I); 1.4–2.0× ReAct's cost with one retry [6] | sequential tries | between tries | optional | **yes** [55][56] | beats ReAct [6][7]; simple retry matches it cheaper [1] | without real feedback it reinforces mistakes |
| `generate_review` | 2 × rounds (I) | sequential | reviewer feedback only | optional | an LLM reviewer (weak signal) | criteria-driven loops [33]; Self-Refine below one call [9][12] | shared blind spots; over-verification [34] |
| `self_discovery` | 3 per task (reusable) + 1 per query [19]; 4 per query (I) | 4 sequential calls | no | no | no | paper: up to 32% over CoT [19] | redundant with reasoning models (I); structured plans lost up to 18.9% [70] |
| `llm_compiler` | 1 planner + 1 joiner (+ replans) [17] | parallel along the DAG | at replans (I) | yes | no | up to 3.7× faster, 6.7× cheaper than ReAct [17]; parallel calls cut research time up to 90% [22] | no gain when steps depend on each other; native parallel calls recover most of the speed (I) |
| `tot` | branches × (proposals + evaluations) × depth (I) | parallel within a level (I) | to evaluator scores | no | yes (LLM) | Game of 24: 74% at $0.74 vs CoT best-of-100 49% at $0.47 [18]; [38] | cost; weak evaluator; no evidence with reasoning models |
| `lats` | iterations × (expansions + values + rollouts + reflection) (I) | mostly sequential | yes | yes | yes | LATS paper [8]; independent rerun: $134.50 vs $1.93 for one call [1] | highest cost; needs resettable environments [8] |

## Guidance: task property → strategy

- **Easy query, or a strong reasoning model:** `completion` with the model's own thinking effort rather than prompt scaffolds — strong evidence [25][30][34][49][50][10].
- **Path to the answer unknown, needs tools:** `react` with native function calling and a small tool set; pass reasoning back between calls — strong [13][4][31][32][14].
- **Result can be checked (tests, validators, environment reward):** verifier-gated retry, `reflexion`, or sample-and-pick; 1–2 reflection rounds for reasoning models — strong [1][3][39][50][48].
- **Only an LLM judge available:** `generate_review` only with explicit criteria; expect small or negative gains otherwise — moderate [33][56][9][12].
- **Many independent sub-queries:** `llm_compiler` or parallel fan-out — moderate [17][22][20].
- **Strongly sequential tasks, or single-agent accuracy already >~45%:** stay single-agent; avoid decomposing — moderate to strong [20][23].
- **Predictable tool sequence, token-sensitive, or untrusted data with side-effecting tools:** `rewoo` — thin for accuracy [16], stronger for security (see [04-rewoo](04-rewoo.md)).
- **Planning is the bottleneck:** `plan_execute` with feedback-driven replanning, ideally decomposing only on failure, with a reasoning-model planner — thin and mixed [7][30][48].
- **Search puzzles with a good evaluator, or resettable environments where accuracy dwarfs cost:** `tot` / `lats` — thin, weaker still for reasoning models [18][8][1].
- **`self_discovery`:** mainly for non-reasoning models (I) [19][70].
- **Tight budgets:** route by estimated difficulty, or start cheap and escalate on failure (escalation baseline 85.0% at $0.27) [1][57][63][23].

## Open questions

1. No study compares ToT, LATS, ReWOO, LLMCompiler or Self-Discover with plain ReAct on GPT-5-class, Claude 4.x+ or Gemini 2.5+ models under matched budgets.
2. Does `plan_execute` still add anything once models think between tool calls?
3. Cheap per-query estimates of difficulty and checkability (the 54.5% classifier; the gap to a perfect router).
4. Repeated-trial reliability is rarely reported (≤6.34% over 8 trials in AgentArch) [13].
5. Standardising compute budgets across studies [10][20][22]; stability of Kim et al.'s model across versions [20].

## Sources

1. Kapoor et al., *AI Agents That Matter*, TMLR 2025 — https://arxiv.org/abs/2407.01502 · https://mlanthology.org/tmlr/2025/kapoor2025tmlr-ai/
2. Kapoor et al., *Holistic Agent Leaderboard* — https://arxiv.org/abs/2510.11977
3. Xia et al., *Agentless*, FSE 2025 — https://arxiv.org/abs/2407.01489
4. Anthropic, *SWE-bench Verified with Claude 3.5 Sonnet* — https://www.anthropic.com/engineering/swe-bench-sonnet
5. *mini-swe-agent* — https://github.com/SWE-agent/mini-swe-agent
6. Huang et al., *Understanding the planning of LLM agents: A survey* — https://arxiv.org/abs/2402.02716
7. Prasad et al., *ADaPT*, NAACL 2024 Findings — https://arxiv.org/abs/2311.05772
8. Zhou et al., *LATS*, ICML 2024 — https://arxiv.org/abs/2310.04406
9. Zhang et al., *AFlow*, ICLR 2025 — https://arxiv.org/abs/2410.10762
10. Jwalapuram et al., *The Illusion of Multi-Agent Advantage* — https://arxiv.org/abs/2606.13003
11. Wang et al., *Efficient Agents* — https://arxiv.org/abs/2508.02694
12. Erol et al., *Cost-of-Pass* — https://arxiv.org/abs/2504.13359
13. Bogavelli et al., *AgentArch* — https://arxiv.org/abs/2509.10769
14. LangChain, *Benchmarking Single Agent Performance* — https://www.langchain.com/blog/react-agent-benchmarking
15. LangChain, *Benchmarking Multi-Agent Architectures* — https://www.langchain.com/blog/benchmarking-multi-agent-architectures
16. Xu et al., *ReWOO* — https://arxiv.org/abs/2305.18323
17. Kim et al., *LLMCompiler*, ICML 2024 — https://arxiv.org/abs/2312.04511
18. Yao et al., *Tree of Thoughts* — https://arxiv.org/abs/2305.10601
19. Zhou et al., *Self-Discover* — https://arxiv.org/abs/2402.03620
20. Kim et al., *Towards a Science of Scaling Agent Systems* — https://arxiv.org/abs/2512.08296
21. Cemri et al., *Why Do Multi-Agent LLM Systems Fail?* — https://arxiv.org/abs/2503.13657
22. Anthropic, *How we built our multi-agent research system* — https://www.anthropic.com/engineering/multi-agent-research-system
23. Gao et al., *Single-agent or Multi-agent Systems?* — https://arxiv.org/abs/2505.18286
24. Wang et al., *Rethinking the Bounds of LLM Reasoning*, ACL 2024 — https://aclanthology.org/2024.acl-long.331/
25. Sprague et al., *To CoT or not to CoT?*, ICLR 2025 — https://arxiv.org/abs/2409.12183
26. Chen et al., *Are More LLM Calls All You Need?*, NeurIPS 2024 — https://arxiv.org/abs/2403.02419
27. Masterman et al., *The Landscape of Emerging AI Agent Architectures* — https://arxiv.org/abs/2404.11584
28. Wei et al., *Agentic Reasoning for Large Language Models* (survey) — https://arxiv.org/abs/2601.12538
29. Yue et al., *From Static Templates to Dynamic Runtime Graphs* — https://arxiv.org/abs/2603.22386
30. OpenAI, *Reasoning best practices* — https://developers.openai.com/api/docs/guides/reasoning-best-practices
31. OpenAI, *o3/o4-mini function calling guide* — https://developers.openai.com/cookbook/examples/o-series/o3o4-mini_prompting_guide
32. OpenAI, *GPT-5 prompting guide* — https://developers.openai.com/cookbook/examples/gpt-5/gpt-5_prompting_guide
33. Anthropic, *Building effective agents* — https://www.anthropic.com/engineering/building-effective-agents
34. Anthropic, *Prompting best practices* — https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices
35. Anthropic, *Extended thinking* — https://platform.claude.com/docs/en/build-with-claude/extended-thinking
36. Anthropic, *The "think" tool* — https://www.anthropic.com/engineering/claude-think-tool
37. Google, *Gemini thinking* — https://ai.google.dev/gemini-api/docs/thinking
38. Snell et al., *Scaling LLM Test-Time Compute Optimally*, ICLR 2025 — https://arxiv.org/abs/2408.03314
39. Brown et al., *Large Language Monkeys* — https://arxiv.org/abs/2407.21787
40. Balachandran et al., *Inference-Time Scaling for Complex Tasks* — https://arxiv.org/abs/2504.00294
41. Zeng et al., *Revisiting the Test-Time Scaling of o1-like Models* — https://arxiv.org/abs/2502.12215
42. Chen et al., *Do NOT Think That Much for 2+3=?* — https://arxiv.org/abs/2412.21187
43. Cuadron et al., *The Danger of Overthinking* — https://arxiv.org/abs/2502.08235
44. Shojaee et al., *The Illusion of Thinking*, NeurIPS 2025 — https://arxiv.org/abs/2506.06941
45. Lawsen, *Comment on The Illusion of Thinking* — https://arxiv.org/abs/2506.09250
46. Dellibarda Varela et al., *Rethinking the Illusion of Thinking* — https://arxiv.org/abs/2507.01231
47. Khan et al., *Reframing the Reasoning Cliff as an Agentic Gap* — https://arxiv.org/abs/2506.18957
48. Zhou et al., *Exploring the Necessity of Reasoning in LLM-based Agent Scenarios* (LaRMA) — https://arxiv.org/abs/2503.11074
49. Meincke et al., *The Decreasing Value of Chain of Thought* — https://arxiv.org/abs/2506.07142
50. Wang et al., *Do Advanced LLMs Eliminate the Need for Prompt Engineering in SE?* — https://arxiv.org/abs/2411.02093
51. Rudyk et al., *Aging of Prompt Engineering Techniques*, ICSME 2026 — https://arxiv.org/abs/2608.24641
52. Sinha et al., *The Illusion of Diminishing Returns*, ICLR 2026 — https://arxiv.org/abs/2509.09677
53. Shen et al., *Thinking vs. Doing* — https://arxiv.org/abs/2506.07976
54. d'Aliberti & Horta Ribeiro, *The Illusion of Insight in Reasoning Models* — https://arxiv.org/abs/2601.00514
55. Huang et al., *LLMs Cannot Self-Correct Reasoning Yet*, ICLR 2024 — https://arxiv.org/abs/2310.01798
56. Kamoi et al., *When Can LLMs Actually Correct Their Own Mistakes?*, TACL 2024 — https://arxiv.org/abs/2406.01297
57. Jeong et al., *Adaptive-RAG*, NAACL 2024 — https://arxiv.org/abs/2403.14403
58. Yue et al., *MasRouter*, ACL 2025 — https://aclanthology.org/2025.acl-long.757/
59. Zhang et al., *MaAS*, ICML 2025 — https://arxiv.org/abs/2502.04180
60. Hu et al., *ADAS*, ICLR 2025 — https://arxiv.org/abs/2408.08435
61. Gao et al., *FlowReasoner* — https://arxiv.org/abs/2504.15257
62. Gao et al., *Meta Reasoning for Large Language Models* — https://arxiv.org/abs/2406.11698
63. Pan et al., *Route to Reason*, WWW 2026 — https://arxiv.org/abs/2505.19435
64. Liu et al., *SMART* (strategy selection) — https://arxiv.org/abs/2410.16128
65. Wang et al., *PET-Select* — https://arxiv.org/abs/2409.16416
66. Zhang et al., *AdaptThink* — https://arxiv.org/abs/2505.13417
67. Fang et al., *Thinkless* — https://arxiv.org/abs/2505.13379
68. OpenAI, *GPT-5 System Card* — https://arxiv.org/abs/2601.03267
69. Su et al., *DAAO*, WWW 2026 — https://arxiv.org/abs/2509.11079
70. Gunasekara & Ratnayake, *iSelf-Discover* — https://arxiv.org/abs/2507.03347
