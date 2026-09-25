# Generate–Review (`generate_review`)

> A generator produces a draft; a reviewer — the same model, another model or prompt, possibly with tools — critiques it against criteria and returns accept or revise; the generator revises until accepted or out of cycles. The generator–critic / evaluator–optimizer family (Self-Refine, CRITIC, Constitutional AI's critique→revision).
> Evidence: strongly positive when the review adds information the generator lacked (tools, tests, rubrics, a stronger or different reviewer); negative for same-model review of closed-form reasoning; equal-compute resampling often wins.
> Last reviewed: 2026-09-24 · Part of [agent-strategies](00-index.md)

## At a glance

- **When it works, the review carries new information:** search evidence (+10.1 F1 on HotpotQA with CRITIC; the tool was the active ingredient) [2]; execution results [16]; stronger-model or human feedback (1.58× more programs repaired) [15]; explicit failed-checklist feedback (RefineBench 18.7 → 98.4 for Claude Opus 4.1 — an oracle-derived ceiling) [18].
- **When it fails, the model reviews itself without an external signal:** GPT-4 on GSM8K 95.5 → 89.0 [7]; graph colouring 16% → 2% [13]; unguided self-refinement over 5 turns still adds ~+2 points or less on frontier models [18].
- **Reviewer failure modes:** rubber-stamping (ChatGPT rated 94% of math solutions fine) [1][25][26]; over-criticism and hallucinated critique [3][13][28]; the generator caving to wrong feedback [21] or gaming the reviewer [22][23]; a same-model reviewer favouring its own style [6][27].
- **Quality oscillates:** most gains come in revisions 1–2 [1][2][3][17]; return the best draft, not the last [1].
- **Better reviewers:** checklists or rubric scores [17][36], a different or reasoning-model reviewer [19][31][32], tool-grounded review [2].
- **In this toolkit:** the generator never sees its previous draft, and the verdict parser accepts "not acceptable" — see [In ai-arch-toolkit](#12-in-ai-arch-toolkit).

## 1. Origin

- **Self-Refine** — Madaan et al., NeurIPS 2023: one LLM drafts, gives itself feedback and revises [1].
- **CRITIC** — Gou et al., ICLR 2024: critique grounded in tool output (search, Python, Perspective API) [2].
- **Constitutional AI** — Bai et al., Anthropic 2022: critique→revision used to generate fine-tuning data, not as an inference loop [3].
- **Evaluator-optimizer workflow** — Anthropic, *Building effective agents*, Dec 2024 [4].
- **The critical line of work:** Huang et al. (ICLR 2024) [7], Kamoi et al. (TACL 2024) [8], Stechly et al. (ICLR 2025) [13], judge-bias papers [5][6].

## 2. How it works

- A generator produces a draft; a reviewer returns accept or revise with a critique; on revise, the generator produces a new version conditioned on the feedback; the loop ends on accept or at the cycle cap.
- **Self-Refine's loop** [1]: the feedback step sees the task and the current draft; the refine step sees the task plus the **full history of drafts and feedback**; it stops on a stop signal or at max iterations, and for multi-aspect tasks returns the **best-scoring draft** across iterations.
- **CRITIC** adds a verify step: query a search engine, a Python interpreter or a toxicity classifier, then critique from that evidence [2].
- **Constitutional AI** samples one of 16 principles per step, critiques, then revises [3].
- **Multi-agent debate** is the many-generator relative: agents read each other's answers and update [10].

## 3. Settings in the papers

- **Self-Refine:** ≤4 iterations, task-specific stop condition; FEEDBACK and REFINE are few-shot prompts even for GPT-4; Vicuna-13B could not produce feedback in the required format [1].
- **CRITIC:** QA ≤3 corrections (stop when the answer is unchanged twice); math ≤4 (stop when the executed result is unchanged twice); toxicity ≤4 (stop below 10%); overhead linear in iterations [2].
- **Constitutional AI:** 16 principles, 4 critique–revision pairs per red-team prompt, T=1 [3].
- **Huang et al.:** round 1 of self-correction = 3 calls, round 2 = 5 [7]. **Du et al. debate:** 3 agents × 2 rounds [10].
- **STICK:** the model writes its own checklist; each revision sees the previous response plus checklist verdicts [17].

## 4. Evidence

| Benchmark / setting | Model | Baseline | Result | Src |
|---|---|---|---|---|
| Dialogue response (GPT-4-judged preference, %) | GPT-4 | 25.4 | 74.6 | [1] |
| Constrained generation (concept coverage %) | GPT-4 / ChatGPT | 15.0 / 44.0 | 45.0 / 67.0 | [1] |
| Math reasoning (solve rate) | GPT-4 / ChatGPT / GPT-3.5 | 92.9 / 74.8 / 64.1 | 93.1 / 75.0 / 64.1 | [1] |
| CommonGen-Hard, Self-Refine re-test (7 calls) | gpt-3.5-turbo-0613 | Madaan prompt 53.0; **stronger initial prompt 81.8** | self-correct 61.1; self-correct on the stronger prompt 75.1 | [7] |
| GSM8K, 2 rounds intrinsic self-correction | GPT-4 / GPT-3.5 / Llama-2 | 95.5 / 75.9 / 62.0 | 89.0 / 74.7 / 36.5 | [7] |
| CommonSenseQA, same | GPT-3.5 / GPT-4 | 75.8 / 82.0 | 41.8 / 80.0 | [7] |
| HotpotQA F1 (500) | ChatGPT | CoT 42.8; self-consistency 47.0 | CRITIC 52.9; CRITIC without tools 46.1 | [2] |
| GSM8K via program synthesis | ChatGPT / text-davinci-003 | PoT 72.5 / 70.1 | CRITIC 78.2 / 72.2; without tools 77.0 / 68.3 | [2] |
| Toxicity probability | ChatGPT | 0.192 | CRITIC 0.040; without tools 0.223 | [2] |
| Graph colouring / Game of 24 / Blocksworld (100 each) | GPT-4 | 16% / 5% / 40% | self-critique 2 / 3 / 55%; sound verifier 37–40 / 36–38 / 60–87% | [13] |
| LiveBench overall, one refinement | GPT-4o / Command-R+ | 55.4 / 32.0 | unstructured critique 47.1 / 23.7; **checklist critique 56.2 / 35.8** | [17] |
| RefineBench (1,000 open-ended), turn 1 → 5 | GPT-5 / Gemini 2.5 Pro / Claude Opus 4.1 | 27.5 / 29.5 / 18.7 | self-refine 29.1 / 31.3 / 20.8; guided by failed-checklist feedback 79.0 / 94.7 / 98.4 | [18] |
| RealCritic, mean of 8 tasks, self-critique vs CoT | o1-mini / GPT-4o / LLaMA-3.1-70B | 59.3 / 58.6 / 56.0 | +3.3 / −4.6 / −4.3 | [19] |
| BoolQ after "are you sure?" | o1-preview / GPT-4o / DeepSeek-R1 | — | −4.9 / −4.9 / −1.6 pts (13.2% / 11.3% / 7.9% of correct answers flipped) | [20] |
| APPS code repair | GPT-4 | own feedback 33.3% repaired | human feedback 52.6% | [15] |
| GSM8K, debate vs self-consistency (9 responses each) | gpt-3.5-turbo-0301 | self-consistency 88.2 | debate 83.0 | [7] |

**Claims vs confirmation:** Self-Refine's ~+20% average comes mostly from preference tasks judged by GPT-4 on GPT-4's own outputs, and part of the constrained-generation gain came from a weak initial prompt [1][7][8]. CRITIC's tool-grounded gains agree with independent work [8][13]. Debate gains did not hold at equal compute in three independent tests [7][11][12].

## 5. Strengths

- Big gains when the review adds information the generator lacked: search (+10.1 F1) [2], execution results (up to +12%) [16], stronger-model or human feedback (1.58×) [15], explicit failure lists [18].
- Suits multi-constraint, open-ended artifacts with writable criteria (STICK: LiveBench Instructions +6.2 / +2.9) [17].
- Splits roles: a cheap generator with a stronger critic — stronger models critique weaker ones well [9], and boosted feedback beat both baselines at every budget [15].
- Can beat resampling when the critique carries new evidence (CRITIC beat oracle-selected rejection sampling by 4.5 and 3.3 EM) [2].
- Bounded (cycle cap) and interpretable (readable feedback trail).

## 6. Weaknesses and costs

- **Calls:** 1 + 2k for k cycles (Self-Refine up to 9; Huang et al. 3 or 5) [1][7].
- **Tokens:** the refine prompt carries all prior drafts and feedback; backprompt tokens grow roughly quadratically with iterations, plain resampling does not [1][13]; reasoning-model reviewers add hidden thinking tokens.
- **Latency:** strictly sequential; linear in iterations [2].
- **Equal-compute baselines often win:** self-consistency beats debate [7][11][12]; independent sampling matches or beats self-repair at small budgets (10 samples + 1 repair 1.05× vs 2 samples + 10 repairs 0.97×) [15]; sampling with a sound verifier matches critique loops [13].
- **Engineering surface:** rubric design, verdict parsing, stop rule, which draft to return.

## 7. Best use cases

- A reliable external check exists: tests, compiler, schema validator, search for fact-checking, calculator, classifier [2][13][16].
- Checking is easier than generating: outputs decompose into separately checkable parts (lists, required elements, constraint coverage) [8].
- Open-ended artifacts with articulable criteria — translation, documents, summaries — ideally with a checklist or rubric [4][17][36].
- Cross-model setups: a small generator with a stronger or reasoning reviewer [15][19].
- Policy or harmlessness revision, accepting some loss of helpfulness [3].

## 8. Where it falls short

1. **Intrinsic self-correction of closed-form reasoning degrades accuracy** (GSM8K 95.5 → 89.0; graph colouring 16% → 2%; RealCritic losses up to −28.0 on MMLU-STEM for o1-mini) [7][13][19]. The bottleneck is *finding* the error — GPT-4 located reasoning mistakes only 52.87% of the time, while correction works once the location is given [14]; models are not reliably better at choosing among their drafts than at generating them [37].
2. **Rubber-stamping:** ChatGPT rated 94% of math solutions fine [1]; LLM judges lean lenient [25]; agreement bias in verifiers [26]; LLM-verifier false positives 10.4% (Game of 24) and 18.55% (Blocksworld) [13]; one-token "master keys" fool generative judges including o1 and Claude 4 [24]; RefineBench models stop refining around turns 3–4 without fixing their responses [18].
3. **Over-critical or hallucinated critique causes churn:** Constitutional AI critiques were often inaccurate or overstated [3]; an LLM verifier rejected 95.8% of correct colourings [13]; CriticGPT hallucinates bugs [28]; the feedback prompt itself biases the model toward changing its answer [7].
4. **The generator gives in to wrong feedback:** when challenged, assistants wrongly admitted mistakes on 42% (GPT-4) to 98% (Claude 1.3) of questions [21].
5. **The generator games the reviewer:** LLM-judge scores rose while human scores did not; sharing context between generator and judge made it worse [22]; in-context reward hacking in feedback loops [23].
6. **Quality oscillates or regresses across rounds** [1][3][7][17].
7. **A same-model reviewer favours its own style:** GPT-4 recognised its own summaries 73.5% of the time, and self-recognition correlates with self-preference [6]; judges favour similar models [27]; position and verbosity biases [5].
8. **Weak models fail** (Vicuna-13B could not follow the feedback format; Llama-2 GSM8K 62.0 → 36.5) [1][7].

## 9. Variants and follow-ups

- **CRITIC** — tool-verified critique [2]; **Self-Debugging** — execution feedback plus code explanation [16].
- **TICK / STICK** — model-generated checklists as the critique [17].
- **Self-grounded verification** (ICLR 2026) — the reviewer states expectations before judging; +25 pp failure detection [26]; **PlanJudge** — the judge plans its evaluation first [32].
- **Single-call rubric grader** — 0.0–1.0 scores plus pass/fail; most consistent with human judgement in Anthropic's research system [36].
- **Trained critics and self-correction:** CriticGPT [28]; SCoRe (+15.6% MATH, +9.1% HumanEval) [29].
- **Multi-agent debate** [10] and its equal-compute critiques [7][11][12] (model heterogeneity helps debate [12]).
- **Best-of-N with a verifier** — the alternative when a sound checker exists [13][15].

## 10. With 2024–2026 models

- **Reasoning models verify internally.** Self-verification emerged in DeepSeek-R1's RL training [33]; Anthropic says recent Claude models check their own work unprompted, and explicit verification prompts can cause over-verification — while still listing generate → review → refine as the most common chaining pattern, mainly to inspect intermediate outputs [34].
- **Unguided self-refinement still barely works on frontier models** (GPT-5 +1.7, Gemini 2.5 Pro +1.8, DeepSeek-R1 −0.1 over 5 turns) [18]; "are you sure?" still flips correct answers on o1-class models [20].
- **Precise feedback works almost perfectly** (the oracle-derived ceiling in [18]).
- **Reasoning models are better critics but still biased:** o1-mini was the only model whose self-critique helped on average [19] and the best step-error finder on ProcessBench [30]; large reasoning models judge better, especially on reasoning-heavy tasks, but keep position, bandwagon and superficial-reflection biases [31][32]; OpenAI recommends reasoning models as graders (vendor anecdote: judge F1 0.12 → 0.74) [35].
- **Native tool calling makes CRITIC-style review routine** — a reviewer with tools is a ReAct loop.
- **Net picture:** negative for a same-model reviewer without tools on closed-form reasoning; positive for a tool- or rubric-grounded reviewer, ideally a different (often reasoning) model, with the draft carried forward.

## 11. Router signals

**Choose `generate_review` when:**
- the reviewer can call a verifier (tests, execution, search, schema checks) → enable reviewer tools;
- the output is open-ended with criteria you can list as a checklist or rubric;
- requirements are multiple independent constraints (coverage, compliance);
- a different-family or reasoning-model reviewer is available and the generator is cheaper;
- quality matters more than latency and ~2–3× the calls are affordable.

**Avoid it when:**
- the task is closed-form math, logic or factual QA with no verifier and a same-model reviewer → self-consistency or best-of-N with a checker;
- knowledge-heavy multiple choice (largest self-critique regressions);
- the criteria could simply go into the generator prompt (CommonGen: better prompt 81.8 vs self-correction 75.1) [7];
- the generator is a strong reasoning model and there is no external signal;
- a small model reviews itself;
- the objective is subjective and generator and reviewer share context (reward-hacking risk);
- the request is latency-critical.

## 12. In ai-arch-toolkit

Flow: [`_generate_review.py`](../../src/ai_arch_toolkit/toolkit/agents/flows/_generate_review.py), built by `_build_generate_review` in [`_builders.py`](../../src/ai_arch_toolkit/toolkit/agents/_builders.py). State as of 2026-09-24.

| Aspect | Toolkit behaviour |
|---|---|
| Generator | inner ReAct when it has tools (via `Agent`, `generator_tools` defaults to the agent's tools; up to spec `max_iterations`, 10), else one call; `output_schema` applies to the generator only |
| Revision | the generator **never sees its previous draft**: each cycle regenerates from the task with all accumulated feedback appended to the system prompt. Self-Refine, Constitutional AI and STICK revise the previous response [1][3][17]; models fix what feedback names and little else [18], so regenerating from scratch risks losing good content and leaves draft-specific feedback dangling |
| Reviewer | inner ReAct when it has tools (`reviewer_tools` defaults to the agent's tools → CRITIC-style review is possible; `max_review_iterations` 5); **same LLM as the generator** unless `reviewer_llm` is set (self-preference risk [6]); `reviewer_kwargs` override per key |
| Verdict | binary: the first line counts as acceptance if it contains "accept" and not "unacceptable" — **"RETRY: this is not acceptable" and "I cannot accept this draft" parse as accepted** (confirmed; fix tracked separately). Tests cover only `UNACCEPTABLE` |
| Result | the **last** reviewed draft, accepted or not (Self-Refine returns the best-scoring one) [1] |
| Loop | `max_cycles` 3 — consistent with gains concentrating in revisions 1–2 |
| Calls | without tools ≤ 6; with tools each phase is a ReAct loop (up to 10 + 5 iterations per cycle) → up to ~45 at defaults |
| Phases | `generator_llm`, `generator_tools`, `reviewer_llm`, `reviewer_tools`; knobs `reviewer_system`, `reviewer_kwargs`, `max_cycles`, `max_review_iterations` |

**Recommended changes:** pass the previous draft (and the feedback history) to the generator; strict verdict parsing that defaults to "not accepted"; a rubric/checklist review with a score so the best draft can be returned; default to a different reviewer model when one is configured; keep reviewer tools on for verifiable content.

## 13. Open questions

- At equal cost, how do same-family and cross-family reviewers compare for 2026 models on open-ended work?
- Which stopping rule is best — reviewer ACCEPT, a single fixed revision, or a score threshold with best-draft selection?
- Do reasoning-model reviewers reduce rubber-stamping enough to pay for their tokens?
- When does critique-conditioned revision beat resampling plus a verifier? ([2] and [13] point in opposite directions; the likely answer is "when the critique carries new information" — an inference.)

## Sources

1. Madaan et al., *Self-Refine*, NeurIPS 2023 — https://arxiv.org/abs/2303.17651
2. Gou et al., *CRITIC*, ICLR 2024 — https://arxiv.org/abs/2305.11738
3. Bai et al., *Constitutional AI*, 2022 — https://arxiv.org/abs/2212.08073
4. Anthropic, *Building effective agents*, Dec 2024 — https://www.anthropic.com/engineering/building-effective-agents
5. Zheng et al., *Judging LLM-as-a-Judge*, NeurIPS 2023 D&B — https://arxiv.org/abs/2306.05685
6. Panickssery, Bowman, Feng, *LLM Evaluators Recognize and Favor Their Own Generations*, NeurIPS 2024 — https://arxiv.org/abs/2404.13076
7. Huang et al., *Large Language Models Cannot Self-Correct Reasoning Yet*, ICLR 2024 — https://arxiv.org/abs/2310.01798
8. Kamoi et al., *When Can LLMs Actually Correct Their Own Mistakes?*, TACL 2024 — https://arxiv.org/abs/2406.01297
9. Lin et al., *CriticBench*, ACL Findings 2024 — https://arxiv.org/abs/2402.14809
10. Du et al., *Improving Factuality and Reasoning through Multiagent Debate*, ICML 2024 — https://proceedings.mlr.press/v235/du24e.html
11. Smit et al., *Should we be going MAD?*, ICML 2024 — https://arxiv.org/abs/2311.17371
12. Zhang et al., *Stop Overvaluing Multi-Agent Debate*, 2025 — https://arxiv.org/abs/2502.08788
13. Stechly, Valmeekam, Kambhampati, *On the Self-Verification Limitations of LLMs*, ICLR 2025 — https://arxiv.org/abs/2402.08115
14. Tyen et al., *LLMs cannot find reasoning errors, but can correct them given the error location*, ACL Findings 2024 — https://arxiv.org/abs/2311.08516
15. Olausson et al., *Is Self-Repair a Silver Bullet for Code Generation?*, ICLR 2024 — https://arxiv.org/abs/2306.09896
16. Chen et al., *Teaching Large Language Models to Self-Debug*, ICLR 2024 — https://arxiv.org/abs/2304.05128
17. Cook et al., *TICK / STICK*, 2024 — https://arxiv.org/abs/2410.03608
18. Lee et al., *RefineBench*, 2025 — https://arxiv.org/abs/2511.22173
19. Tang et al., *RealCritic*, 2025 — https://arxiv.org/abs/2501.14492
20. Zhang et al., *Understanding the Dark Side of LLMs' Intrinsic Self-Correction*, 2024 — https://arxiv.org/abs/2412.14959
21. Sharma et al., *Towards Understanding Sycophancy in Language Models*, ICLR 2024 — https://arxiv.org/abs/2310.13548
22. Pan, He, Bowman, Feng, *Spontaneous Reward Hacking in Iterative Self-Refinement*, 2024 — https://arxiv.org/abs/2407.04549
23. Pan et al., *Feedback Loops With Language Models Drive In-Context Reward Hacking*, ICML 2024 — https://arxiv.org/abs/2402.06627
24. Zhao et al., *One Token to Fool LLM-as-a-Judge*, 2025 — https://arxiv.org/abs/2507.08794
25. Thakur et al., *Judging the Judges*, GEM2 2025 — https://arxiv.org/abs/2406.12624
26. Andrade et al., *Self-Grounded Verification*, ICLR 2026 — https://arxiv.org/abs/2507.11662
27. Goel et al., *Great Models Think Alike*, ICML 2025 — https://arxiv.org/abs/2502.04313
28. McAleese et al., *LLM Critics Help Catch LLM Bugs*, 2024 — https://arxiv.org/abs/2407.00215
29. Kumar et al., *SCoRe*, ICLR 2025 — https://arxiv.org/abs/2409.12917
30. Zheng et al., *ProcessBench*, ACL 2025 — https://arxiv.org/abs/2412.06559
31. Wang et al., *Judging Bias in Large Reasoning Models*, 2025 — https://arxiv.org/abs/2504.09946
32. Huang et al., *Reasoning Model Is Superior LLM-Judge*, 2026 — https://arxiv.org/abs/2601.03630
33. DeepSeek-AI, *DeepSeek-R1*, 2025 — https://arxiv.org/abs/2501.12948
34. Anthropic, *Prompting best practices* (accessed Sept 2026) — https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices
35. OpenAI, *Reasoning best practices* — https://developers.openai.com/api/docs/guides/reasoning-best-practices
36. Anthropic, *How we built our multi-agent research system*, June 2025 — https://www.anthropic.com/engineering/multi-agent-research-system
37. Jiang et al., *SELF-[IN]CORRECT*, 2024 — https://arxiv.org/abs/2404.04298
