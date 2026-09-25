# Reflexion (`reflexion`)

> Attempt → evaluate → on failure, write a short verbal lesson → retry with the lessons in context. "Verbal reinforcement learning": no weights change.
> Evidence: works only with an external evaluator. With ground-truth or test feedback it gains a lot on resettable tasks; with self-evaluation it hurts; cost-controlled reruns found a plain verifier-gated retry at least as good and cheaper.
> Last reviewed: 2026-09-24 · Part of [agent-strategies](00-index.md)

## At a glance

- **The evaluator is the method.** The paper's HotpotQA feedback was exact match against the ground-truth answer [1]. Without such labels, prompted self-correction *lowers* accuracy (GPT-3.5 on CommonSenseQA 75.8 → 41.8) [4].
- **Cost-controlled rerun (HumanEval, gpt-4-turbo):** Reflexion 87.8% at $3.90 vs zero-shot 89.6% at $1.93 and retry-with-rising-temperature 93.2% at $2.45 [6].
- **Informative feedback is what helps:** a verifier naming the first error beat a pass/fail verifier on Blocksworld (87% vs 60%) [11]; tests + reflection beat either alone in the paper's Rust ablation [1].
- **2025–26:** reasoning models reflect internally but mostly to confirm (>90% of reflections keep the earlier answer) [17]; self-written tests are fully correct only ~57–59% of the time [12]; test-feedback loops raise test-gaming [16]. The idea performs best today as **memory across tasks** [9][22][23].
- **In this toolkit:** without an `evaluator` the default accepts any non-empty answer (so `reflexion` = one ReAct run), and the reflector sees only the final answer and a score — see [In ai-arch-toolkit](#12-in-ai-arch-toolkit).

## 1. Origin

**Shinn, Cassano, Berman, Gopinath, Narasimhan, Yao**, "Reflexion: Language Agents with Verbal Reinforcement Learning", NeurIPS 2023; arXiv 2303.11366 [1][2]. Code: noahshinn/reflexion [3].

## 2. How it works

Three LLM roles plus a memory [1]:

- **Actor** (CoT or ReAct) produces an attempt (a trajectory).
- **Evaluator** scores it: exact match against the reference, a hand-written heuristic, self-generated unit tests, or an LLM classifier.
- **Self-reflection LLM** runs on failure: it turns the attempt, the score and earlier lessons into a short first-person lesson (what went wrong, what to do next time).
- **Memory** keeps Ω = 1–3 lessons, added to the actor's prompt on the next attempt, after the environment is reset.

For code, the model writes tests with CoT, keeps the syntactically valid ones (up to 6), runs its implementation against them and reflects on failures (memory 1).

## 3. Settings in the paper

- **ALFWorld:** 134 environments, ReAct actor on "GPT-3", 2 few-shot trajectories; evaluator = heuristic (same action/observation >3 cycles or >30 actions) or LLM classifier; memory 3; up to 12 attempts [1].
- **HotpotQA:** 100 questions; pass/fail from **exact match against the ground-truth answer** between attempts; memory 3 [1].
- **Programming:** HumanEval (161 of 164 problems per [6]), MBPP, Rust translations, LeetcodeHardGym (40 problems); up to 6 self-generated tests; the repo's HumanEval script runs `max_iters 2` [1][3][6].
- **Cost:** not reported; the README warns reruns incur significant GPT-4 charges [3].

## 4. Evidence

| Benchmark | Model | Baseline | Result | Src |
|---|---|---|---|---|
| HumanEval Py, pass@1 | GPT-4 | 80.1 | 91.0 | [1] |
| HumanEval-Rust (50 hardest) / MBPP-Rust | GPT-4 | 60.0 / 70.9 | 68.0 / 75.4 | [1] |
| MBPP Py | GPT-4 | 80.1 | **77.1 (worse)** — wrong solutions passed self-tests 16.3% of the time vs 1.4% on HumanEval | [1] |
| LeetcodeHardGym | GPT-4 | 7.5 | 15.0 | [1] |
| HumanEval-Rust ablation | GPT-4 | 0.60 | tests without reflection 0.60; reflection without tests 0.52; both 0.68 | [1] |
| HumanEval Py | starchat-beta | 0.26 | 0.26 (no gain) | [1] |
| ALFWorld (134) | GPT-3 | ReAct plateaus by attempt 6–7 | 130/134 after 12 attempts (heuristic evaluator) | [1] |
| HotpotQA (100), ground-truth EM feedback | ReAct on gpt-4 / gpt-3.5 / text-davinci-003 | 0.39 / 0.26 / 0.30 | 0.51 / 0.38 / 0.55 | [1] |
| WebShop (100) | GPT-3 | — | no improvement after 4 attempts | [1] |
| **HumanEval rerun, all 164, 5 runs** | gpt-4-turbo-2024-04-09 | zero-shot 89.6% ($1.93); retry ×5 92.0% ($2.51); rising-temperature retry 93.2% ($2.45) | **87.8% ($3.90)** | [6] |
| Same, June-2023 GPT-4 | GPT-4 | zero-shot 86.5% ($2.94); rising-temperature retry 90.6% ($3.88) | 80.2% ($8.29) | [6] |
| Closed-book HotpotQA, with vs without ground truth | GPT-4 | 49.0 | stop on ground truth 59.0; self-judged 43.0 after 2 rounds | [4] |
| CommonSenseQA / GSM8K, self-judged | GPT-3.5 / Llama-2-70B | 75.8 / 62.0 | 41.8 / 36.5 | [4] |
| Self-critique vs a sound verifier (100 each: Graph coloring / Game of 24 / Blocksworld) | GPT-4 | 16% / 5% / 40% | self-critique 2 / 3 / 55%; pass/fail verifier 38 / 36 / 60%; verifier naming the first error 37 / 38 / 87%; plain resampling with verifier, 15 samples 40 / 28 / 68% | [11] |
| Self-repair at a fixed budget (APPS) | GPT-4 | pass@20 / pass@22 | 10 samples + 1 repair: 1.05×; 2 samples + 10 repairs: 0.97× | [7] |
| Who writes the feedback (APPS) | GPT-4 | own feedback: 33.3% repaired | human feedback: 52.6% | [7] |
| Self-debugging with self-written vs real tests (HumanEval) | Claude-3.5-Sonnet / GPT-4o | 94.5 / 92.1 | self-written tests 87.2 / 89.0; real tests 97.6 / 95.1; only 56.71% / 59.15% of self-written suites fully correct | [12] |
| Memory across tasks | gpt-3.5-turbo-0613 | Reflexion after 3 retries: HotpotQA 40%, ALFWorld 54% | ExpeL in one attempt 39% / 59%; ExpeL + Reflexion on ALFWorld 64.2% | [9] |
| Trained reflection writer | text-davinci-003 actor, LongChat-7B reflector | Reflexion: HotpotQA 50%, ALFWorld 84.33%, WebShop 35% | Retroformer 54% / 100% / 36% | [10] |
| Systems cost, HotpotQA (50) | Llama-3.1 8B / 70B | one chat turn 4.23 s / 6.40 s | Reflexion 38% at 649 s / 67% at 720 s | [15] |
| Test-feedback loops and gaming (frontier models) | — | one submission | multiple submissions: honest pass 80 → 83%, cheating 33 → 38%; letting the model abort cut GPT-5 cheating 54 → 9% | [16] |

## 5. Strengths

- No fine-tuning; works with black-box APIs; memory is readable text [1].
- Large gains when the evaluator is reliable and the environment can be reset (ALFWorld 130/134; code with trustworthy tests) [1].
- Only failures pay extra — an attempt that passes costs one attempt plus one evaluation.
- Can catch an early mistake in a long trajectory (e.g. the agent believing it held an item it did not) [1].
- Combines well with memory across tasks (ExpeL + Reflexion) [9].
- Informative feedback measurably helps (Blocksworld 87% vs 60%; Rust ablation 0.60 → 0.68; better feedback in [7]) [1][7][11].

## 6. Weaknesses and costs

- **Cost structure:** each failed attempt = full actor run + evaluator (0 calls for tests/EM, 1 for an LLM judge) + 1 reflection; worst case ≈ attempts × one attempt.
- **Measured:** ~2× zero-shot GPT-4's cost for lower accuracy [6]; steep diminishing returns (the next 4 accuracy points cost 31× more time) [15].
- **Evaluator dependence:** false passes end the loop with a wrong answer (why MBPP fell below baseline); false failures push harmful edits [1].
- **The paper's QA setting cannot be deployed** — it grades attempts with the ground-truth answer [4][5].
- **Weak reflection writers get nothing** (starchat-beta 0.26 → 0.26) [1]; frozen-model reflections often restate the failed plan [10] and can contain made-up content [9].
- **Exploration-heavy tasks fail** (WebShop) [1][10]; the environment must be resettable; feedback loops invite test-gaming [16].

## 7. Best use cases

- Code generation or repair against **trusted** tests (hidden, derived from the spec, read-only) or a compiler/type checker with precise errors [1][12][16].
- Episodic environments with a clear success signal and cheap resets [1].
- Offline or batch work with reference checks [1][4].
- Medium-difficulty tasks — near the ceiling a plain retry wins [6]; for weak models reflection adds nothing [1].
- A stronger model (or a human) writes the feedback [7].

## 8. Where it falls short

- **The model grades itself on reasoning or QA:** accuracy falls, and GPT-3.5 changes right answers to wrong more often than the reverse [4]; no fair demonstration of prompted self-correction on general tasks [5]; a critique-and-refine loop underperformed CoT on 3 of 4 tasks [24]; self-critique collapsed Graph Coloring 16% → 2% [11].
- **Near-ceiling benchmarks:** below zero-shot GPT-4 in both HumanEval reruns, while retry baselines were better and cheaper [6].
- **Self-written tests** are often wrong, and self-debugging against them hurt HumanEval scores [12].
- **Small budgets:** the number of independent first attempts matters more than repair depth [7].
- **Degeneration of thought:** once confident, reflection stops producing new ideas [13]; frozen reflections restate the failed plan [10].
- **Test-gaming:** frontier agents special-case or edit tests; GPT-5 cheated on 76% of impossible SWE-bench variants [16].

## 9. Variants and follow-ups

- **Self-Debugging** (ICLR 2024) — execution feedback plus self-explanation; matches baselines sampling >10× more programs [8].
- **CRITIC** (ICLR 2024) — tool-verified critique; external feedback is essential [25].
- **ExpeL** (AAAI 2024) — insights extracted across tasks [9]; **Retroformer** (ICLR 2024) — reflection writer trained with policy gradient [10].
- **MAR** (2025, not peer reviewed) — reflections from a multi-persona debate; HotpotQA 44% → 47%, HumanEval 76.4% → 82.6% at ~3× Reflexion's API calls [14].
- **Caution:** a widely cited 2024 self-reflection study let the reflecting agents see the correct answer — its gains do not transfer to deployment [26].
- **SCoRe** (ICLR 2025) — RL-trained self-correction: +15.6% MATH, +9.1% HumanEval [19]; **Reflect, Retry, Reward** — RL rewards reflections that lead to a successful retry [20].
- **Dynamic Cheatsheet** — test-time memory across tasks; GPT-4o Game of 24 10% → 99% [22]; **ReasoningBank** (ICLR 2026) — memory from successes and failures; +4.6 to +8.3 on WebArena [23].
- **LATS** — Reflexion plus tree search; $134.50 vs $3.90 for Reflexion in [6] (see [10-lats](10-lats.md)).

## 10. With 2024–2026 models

- **External feedback is the consensus requirement:** self-correction works with reliable outside feedback or large-scale training, not prompting alone [4][5][11][25].
- **Reasoning models reflect internally, mostly to confirm:** >90% of reflections keep the earlier answer; wrong-to-right flips mostly <2% [17]; training, not prompting, makes self-correction work [18][19].
- **Verifier + resampling is a strong, cheap baseline:** DeepSeek-Coder-V2 on SWE-bench Lite rose from 15.9% (1 sample) to 56% (250 samples) [21]; see also [6][11].
- **Native tool loops already feed errors back within an attempt.** Reflexion's distinct parts are the reset, the distilled lesson carried to the next attempt, and evaluator gating; the paper showed reflection beating raw memory of the last attempt by 8 points (GPT era), with no 2025 replication [1].
- **Best current form:** memory across tasks [22][23]. **New risk:** models gaming the evaluator [16].

## 11. Router signals

**Choose `reflexion` when:**
- the evaluator is independent of the actor, rarely passes wrong answers and ideally returns diagnostics (hidden/read-only tests, compiler, schema checks, environment success);
- the task can be retried without side effects;
- the latency budget is ≥2–3× one attempt;
- the first attempt fails sometimes but not always;
- a stronger model can write the reflection.

**Avoid it when:**
- the only evaluator is the same model judging reasoning or QA → `completion` or self-consistency;
- tests are self-written and unvalidated, or the agent can edit them;
- the task needs exploration → sampling, `tot` or `lats`;
- the task is near the model's ceiling, or the budget is small (spend it on independent samples + the verifier);
- latency is interactive or actions are irreversible.

**Defaults:** 2–3 attempts; stop as soon as the evaluator passes; stop if two attempts produce the same answer (heuristic); with a pass/fail-only evaluator, try a plain retry first.

## 12. In ai-arch-toolkit

Flow: [`_reflexion.py`](../../src/ai_arch_toolkit/toolkit/agents/flows/_reflexion.py), built by `_build_reflexion` in [`_builders.py`](../../src/ai_arch_toolkit/toolkit/agents/_builders.py). State as of 2026-09-24.

| Aspect | Toolkit behaviour |
|---|---|
| Actor | each attempt is a full inner ReAct run (`max_iterations` = spec `max_iterations`, 10) |
| Evaluator | `evaluator(task, answer) -> float`, synchronous, **scalar only** (no diagnostics). Via `Agent`, the **default evaluator returns 1.0 for any non-empty answer** — without a real evaluator the flow is one ReAct run |
| Reflector | sees the task, the final answer, the score and the threshold — **not** the trajectory (tool calls, observations) and not any evaluator feedback text; the paper's lessons come from the trajectory and the feedback signal (e.g. failing tests) |
| Memory | **all** previous reflections appended to the actor's system prompt (paper: window Ω = 1–3) |
| Loop | `max_retries` = total attempt passes (default 3); `threshold` 0.7; a reflection is also generated after the final failed attempt (unused); the last attempt's answer stands if none passes |
| Phases | `executor_llm`, `executor_tools`, `reflector_llm`, dependency `evaluator`; knobs `threshold`, `max_retries`, `reflector_system` |

With a scalar-only evaluator and a reflector blind to the trajectory, the flow is close to "retry with a generic hint" — the evidence says the informative feedback is what makes reflection work.

**Recommended changes:** let the evaluator return feedback text alongside the score; give the reflector a compact trajectory summary and that feedback; window the memory to the last 1–3 lessons; skip the reflection after the final attempt; stop on repeated answers; document that `reflexion` needs a real evaluator (or fail fast without one).

## 13. Open questions

1. At equal cost, does a written reflection beat a verifier-gated retry on hard agentic tasks with reasoning models [6]?
2. How often do LLM-judge evaluators pass wrong answers or fail right ones on agent tasks, and how should thresholds be calibrated?
3. Retries within a task vs memory across tasks — where should the budget go?
4. How to detect repetition across retries cheaply?

## Sources

1. Shinn et al., *Reflexion*, NeurIPS 2023 — https://arxiv.org/abs/2303.11366
2. NeurIPS proceedings — https://papers.nips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html
3. Reference code — https://github.com/noahshinn/reflexion · https://raw.githubusercontent.com/noahshinn/reflexion/main/programming_runs/run_reflexion.sh
4. Huang et al., *Large Language Models Cannot Self-Correct Reasoning Yet*, ICLR 2024 — https://arxiv.org/abs/2310.01798
5. Kamoi et al., *When Can LLMs Actually Correct Their Own Mistakes?*, TACL 2024 — https://arxiv.org/abs/2406.01297
6. Kapoor et al., *AI Agents That Matter*, TMLR 2025 — https://arxiv.org/abs/2407.01502
7. Olausson et al., *Is Self-Repair a Silver Bullet for Code Generation?*, ICLR 2024 — https://arxiv.org/abs/2306.09896
8. Chen et al., *Teaching Large Language Models to Self-Debug*, ICLR 2024 — https://arxiv.org/abs/2304.05128
9. Zhao et al., *ExpeL*, AAAI 2024 — https://arxiv.org/abs/2308.10144
10. Yao et al., *Retroformer*, ICLR 2024 — https://arxiv.org/abs/2308.02151
11. Stechly et al., *On the Self-Verification Limitations of LLMs*, ICLR 2025 — https://arxiv.org/abs/2402.08115
12. *Revisit Self-Debugging with Self-Generated Tests*, 2025 — https://arxiv.org/abs/2501.12793
13. Liang et al., *Encouraging Divergent Thinking … through Multi-Agent Debate*, EMNLP 2024 — https://arxiv.org/abs/2305.19118
14. *MAR* (multi-agent reflexion), 2025 — https://arxiv.org/abs/2512.20845
15. Kim, Shin, Chung, Rhu (KAIST), agent systems characterization, 2025 — https://arxiv.org/abs/2506.04301
16. *ImpossibleBench*, 2025 — https://arxiv.org/abs/2510.20270
17. *First Try Matters*, 2025 — https://arxiv.org/abs/2510.08308
18. *Self-Correction Bench*, COLM 2026 — https://arxiv.org/abs/2507.02778
19. Kumar et al., *SCoRe*, ICLR 2025 — https://arxiv.org/abs/2409.12917
20. *Reflect, Retry, Reward*, 2025 — https://arxiv.org/abs/2505.24726
21. Brown et al., *Large Language Monkeys*, 2024 — https://arxiv.org/abs/2407.21787
22. Suzgun et al., *Dynamic Cheatsheet*, 2025 — https://arxiv.org/abs/2504.07952
23. *ReasoningBank*, ICLR 2026 — https://arxiv.org/abs/2509.25140
24. Hu et al., *ADAS*, ICLR 2025 — https://arxiv.org/abs/2408.08435
25. Gou et al., *CRITIC*, ICLR 2024 — https://arxiv.org/abs/2305.11738
26. Renze & Guven, self-reflection study (the reflecting agents saw the correct answer), 2024 — https://arxiv.org/abs/2405.06682
