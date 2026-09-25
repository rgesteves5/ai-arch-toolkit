# Single call (`completion`)

> One request, one response, no tools, no loop. All reasoning happens inside the call — prompted chain of thought on older models, native thinking on reasoning models.
> Evidence: strong. It is the control condition of nearly every agent paper, and cost-controlled studies keep finding it (plus cheap retries) competitive with elaborate agents. Its limits are equally well documented: fresh or private data, actions, exact computation, verification.
> Last reviewed: 2026-09-24 · Part of [agent-strategies](00-index.md)

## At a glance

- **Often enough:** one plain GPT-4 call scored 89.6% on HumanEval at $1.93, against LATS's 88.0% at $134.50 [13]; direct answering beat ReAct on TriviaQA (80.6 vs 59.4) [14]; Anthropic recommends starting with the simplest solution, which may mean no agent at all [23].
- **Reasoning is a dial, not a scaffold:** chain of thought helps mainly on math, logic and symbolic tasks [1][4]; on reasoning models CoT prompts are obsolete (small or negative gains at 20–80% more latency) [12][24]. The main cost lever is the effort level [25][26].
- **More thinking is not always better:** overthinking on easy inputs [8], collapse on high-complexity puzzles (contested) [9][10][11], inverse scaling on some tasks [27]; higher effort did not help in 21 of 36 agent runs [22].
- **Hard limits:** knowledge frozen at training time (FreshQA +32.6 / +49.0 with search) [19]; long-tail hallucination even for reasoning models [17][18]; no actions; no external verification [29]; exact computation (a 6.7B model with a calculator beats GPT-3 175B) [21].
- **In this toolkit:** a single `llm.complete` with optional `output_schema`; reasoning depth comes from `thinking` / `thinking_effort` in `llm_kwargs` — see [In ai-arch-toolkit](#12-in-ai-arch-toolkit).

## 1. Origin

No founding paper — `completion` is the baseline. The work that defines what one call can do:

- **Chain-of-Thought prompting** — Wei et al., NeurIPS 2022 [1]; **zero-shot CoT** — Kojima et al., NeurIPS 2022 [2]; **self-consistency** — Wang et al., ICLR 2023 [3].
- **Reasoning models** — OpenAI o1 (Sep 2024) [6]; DeepSeek-R1 (2025; *Nature*) [7]. **Test-time compute** — Snell et al., ICLR 2025 [5].
- **Reassessments** — Sprague et al., ICLR 2025 [4]; Kapoor et al., *AI Agents That Matter*, TMLR 2025 [13]; Agentless, FSE 2025 [15]; Anthropic's *Building effective agents* [23]; OpenAI's reasoning best practices [24].

## 2. How it works

1. **Direct answer.**
2. **Prompted reasoning:** few-shot exemplars with worked rationales (Wei et al. used 8) or a zero-shot trigger ("Let's think step by step") [1][2].
3. **Native thinking:** a reasoning model spends thinking tokens before answering. Controls: Anthropic's adaptive thinking or a manual budget; OpenAI's `reasoning_effort` [25][26].

The parallel extension — self-consistency, or best-of-N with a verifier — draws n independent samples and votes or ranks; it still never observes the environment [3][5].

## 3. Settings and cost in the papers

- **CoT:** 8 exemplars, greedy decoding, PaLM 540B / GPT-3 175B / LaMDA 137B [1].
- **Self-consistency:** 40 paths (5–10 capture most of the gain), T=0.5–0.7; extra compute is the main limitation [3].
- **o1:** evaluated at maximum test-time compute, plus consensus@64 and re-ranking of 1,000 samples [6]. **R1:** ≤32,768 output tokens, T=0.6; pass@1 averaged over 4–64 samples because greedy decoding repeated itself [7].
- **Thinking tokens are billed as output** even when only a summary is returned [25][26].
- **Measured overheads:** CoT prompts added 35–600% latency on non-reasoning models and 20–80% on reasoning models (GPQA) [12].

## 4. Evidence

| Benchmark | Model / setting | Baseline | Result | Src |
|---|---|---|---|---|
| GSM8K | PaLM 540B, 8-shot | standard prompting 17.9 | CoT 56.9 (gains appear only around 100B parameters) | [1] |
| MultiArith / GSM8K | text-davinci-002, zero-shot | 17.7 / 10.4 | zero-shot CoT 78.7 / 40.7 | [2] |
| CommonsenseQA | text-davinci-002 | 68.8 | 64.6 (**CoT hurt**) | [2] |
| GSM8K | PaLM-540B / code-davinci-002 | CoT greedy 56.5 / 60.1 | self-consistency (40 paths) 74.4 / 78.0 | [3] |
| Meta-analysis of 100+ papers + 20 datasets × 14 models | mixed | direct answer | CoT gain: symbolic +14.2, math +12.3, logic +6.9; all other categories 56.8 vs 56.1; ~95% of the MMLU gain from questions containing `=` | [4] |
| MATH | PaLM 2-S* | best-of-N | per-prompt compute allocation >4× as efficient; no method helps much on the hardest level | [5] |
| AIME 2024 | o1 vs GPT-4o (vendor) | GPT-4o 12% | 74% pass@1; 83% cons@64 | [6] |
| AIME 2024 | DeepSeek-R1 | o1-1217 79.2 | 79.8 pass@1; R1-Zero 15.6 → 71.0 through RL | [7] |
| GPQA Diamond, 25 trials × 198 | non-reasoning models | direct answer | CoT raises the mean (up to +13.5) but lowers the 25/25-correct rate on 3 of 5 models | [12] |
| Same setup | o3-mini / o4-mini / Gemini 2.5 Flash | direct answer | CoT prompt +2.9 / +3.1 / −3.3 | [12] |
| Trivial arithmetic ("2+3") | QwQ-32B-Preview, R1 | conventional LLMs | 1,953% more tokens; in >92% of cases the first round was already correct | [8] |
| HumanEval (164, 5 runs, total $) | gpt-4-turbo | **one plain call 89.6% ($1.93)** | LATS 88.0% ($134.50); Reflexion 87.8% ($3.90); retry on tests 92.0% ($2.51); rising-temperature retry 93.2% ($2.45) | [13] |
| TriviaQA / Sports Understanding / GSM8K | gpt-3.5-turbo | ReAct 59.4 / 58.6 / 62.0 | direct 80.6 / 68.0; CoT 67.4 | [14] |
| SWE-bench Lite | GPT-4o, fixed 3-phase pipeline (Agentless) | agent scaffolds | 32.0% at $0.70/issue; OpenAI reports GPT-4o at 33.2% on SWE-bench Verified with Agentless, the best open-source scaffold at the time | [15][16] |
| SimpleQA / PersonQA hallucination (vendor) | o1 / o3 / o4-mini | — | 0.44 / 0.51 / 0.79 and 0.16 / 0.33 / 0.48 | [17] |
| FreshQA | GPT-4 | vanilla | +32.6 (relaxed) / +49.0 (strict) with search results in the prompt | [19] |
| BrowseComp, no tools | GPT-4o / o1 | — | 0.6% / 9.9% vs 51.5% for the Deep Research agent | [20] |
| ASDiv / SVAMP / MAWPS | GPT-J 6.7B + calculator (Toolformer) | GPT-3 175B 14.0 / 10.0 / 19.8 | 40.4 / 29.4 / 44.0 | [21] |

**Claims vs confirmation:** CoT gains on math/symbolic tasks are replicated many times and bounded to those task types by [4]; OpenAI's o1 numbers are vendor claims, reproduced in spirit by open-weight R1 [7]; overthinking [8] and the complexity regimes of [9] are separate findings, the latter disputed [10][11]; Kapoor et al. are independent of the agents they tested and reproduced the result at scale in HAL [22].

## 5. Strengths

- Cheapest and fastest — one round trip [13].
- Bounded failure surface — no tool errors, loops, injected observations or side effects; latency capped by `max_tokens` and effort.
- Internal reasoning works where it counts — multi-step math, logic, symbolic work [1][4][6][7].
- Often enough for many applications, especially with retrieval and in-context examples in the prompt; both major vendors advise maximising the simplest option before adding agents [23][37]; with long context, a single call beat RAG on average when resources allow (RAG stays cheaper) [31].
- Easy to cache, batch and evaluate.

## 6. Weaknesses and costs

- **Frozen knowledge:** reasoning models still hallucinate long-tail facts (o3 33% on PersonQA vs o1 16%) [17]; training and evaluation reward guessing [30].
- **No actions and no external verification;** without outside feedback, models do not reliably correct themselves [29].
- **Thinking costs:** billed as output, adds latency [25][26]; CoT can raise average accuracy while lowering consistency [12]; longer reasoning can hurt [27]; the thinking text is not a faithful audit log (hint usage revealed <20% of the time) [28].
- **Output-size ceiling:** enumerating Tower of Hanoi moves grows as 2^N−1 [10].

## 7. Best use cases

- Self-contained inputs: classify, extract, summarise, translate, rewrite, format, draft, answer over documents that fit in context.
- Closed-form reasoning with a checkable answer — raise the effort instead of adding a loop [4][6][7].
- Stable, well-known knowledge where an occasional error is cheap or checked downstream.
- A cheap check plus cheap retries (retry-on-test beats complex agents per dollar) [13].
- Tight latency or throughput budgets.

## 8. Where it falls short

- **Fresh information** — accuracy is flat across model sizes on fast-changing questions [19].
- **Long-tail facts and hard lookups** — 0.16–0.79 hallucination without search [17][18]; 0.6–9.9% on BrowseComp [20].
- **Private or live data, and actions** — by construction.
- **Exact computation** — CoT loses to a symbolic solver [4]; PAL (code for an interpreter) beat PaLM-540B by 15% on GSM8K [36]; o4-mini reached 99.5% on AIME 2025 *with* a Python interpreter (vendor) [38].
- **Verification** — gains need an external signal: verifiers or revisions [5], test-based selection [6], tests [13][15].
- **The hardest problems** — no test-time method helped much on level-5 MATH [5]; high-complexity puzzles collapse [9].
- **Open-ended search** — agents do better when the issue gives no hint where the bug is [15].

## 9. Variants and follow-ups

- Few-shot CoT [1]; zero-shot CoT [2]; **Plan-and-Solve** — plan, then execute, in one prompt [34].
- **Self-consistency** [3]; **Universal Self-Consistency** for free-form outputs [35]; compute-optimal best-of-N / beam search against a verifier [5].
- **PAL** — generate code, let an interpreter run it (really one tool call) [36].
- **s1 budget forcing** [33]; **NoThinking + parallel sampling** — matched Thinking with up to 9× lower latency [32].
- **Adaptive thinking and effort controls** [25]; **GPT-5's router** between a fast and a thinking model [18]; **Self-Route** — RAG vs long context per query [31].
- **Retry, warming and model-escalation baselines** [13].

## 10. With 2024–2026 models

- **CoT moved inside the model:** OpenAI advises against CoT prompts for reasoning models and says try zero-shot first [24]; few-shot prompting degraded R1 [7]; CoT prompts add ≤~3 points on reasoning models [12]. Sprague's finding still describes where thinking earns its cost [4].
- **Thinking became a dial and a router:** adaptive thinking may skip thinking on easy inputs [25]; GPT-5 routes between a fast and a thinking model by conversation type, complexity, tool needs and explicit intent — this project's router, built into the model layer [18].
- **More thinking is not always better** [8][9][22][27].
- **Reasoning does not fix factuality by itself** [17][18][20].
- **The line between `completion` and `react` blurs:** o3/o4-mini call tools inside their chain of thought [17][38]; Kimi K2 Thinking 23.9 → 44.9 on HLE with tools (vendor) [39]. For routing, the question is whether *the application* must run tools, not whether the model reasons.
- **Changed:** prompted CoT is obsolete on reasoning models; hidden reasoning tokens dominate cost, so the effort level is the main cost lever.

## 11. Router signals

**Choose `completion` when:**
- everything needed is in the prompt or attachments that fit in context;
- the knowledge is stable and common, and errors are cheap or verified downstream;
- the task is closed-form reasoning with no need to run code — raise the effort rather than add a loop;
- no registered tool matches the intent — needless tool use costs money and can lower accuracy (TriviaQA 80.6 direct vs 59.4 ReAct [14]; SMART cut tool use by 24% while improving performance by >37% [40]);
- the latency budget allows one round trip;
- a deterministic check exists (tests, schema) — completion plus retry [13].

**Set effort by input type:** none/low for lookups, short transforms and labelling (long thinking wastes tokens and can hurt [8][9]); high for multi-step math, logic or planning.

**Escalate on low confidence:** the ReAct paper switched from self-consistency to ReAct when the majority answer got fewer than n/2 votes (see [02-react](02-react.md)); Self-Route routes on the model's own judgement [31].

**Avoid it (route to a tool-using strategy) when** the request needs post-cutoff or changing facts, private or stateful data, actions, long-tail facts or citations, exact computation over data, or output beyond output limits [10][17][19][36].

## 12. In ai-arch-toolkit

`_build_completion` in [`_builders.py`](../../src/ai_arch_toolkit/toolkit/agents/_builders.py) builds a one-step flow: a single `llm.complete(messages, system=…, **llm_kwargs)`. State as of 2026-09-24.

- `output_schema` is supported (structured output); no tools are passed.
- Reasoning depth is set through the LLM call: `thinking=True` and `thinking_effort=…` in the spec's `llm_kwargs` (adapter-specific mapping — adaptive thinking and `output_config.effort` on Anthropic; `thinking_effort` only with `thinking=True` on OpenAI; Grok and Muse Spark reason unasked). The router's effort choice belongs here.
- Provider errors end the step with an error; a budget denial is terminal (converted to `budget_exceeded`).
- Self-consistency / best-of-N are not built-in strategies; they can be composed from `completion` runs plus a checker.

## 13. Open questions

- How accurate are adaptive thinking and vendor routers at estimating difficulty? No independent evaluation found [5].
- When can reasoning replace retrieval, given model-specific hallucination trade-offs [17][20]?
- Single-call reliability (consistency across repeats) is rarely reported [12].
- Thinking text is not a faithful log [28]; GSM8K and MATH are saturated/contaminated [9].

## Sources

1. Wei et al., *Chain-of-Thought Prompting*, NeurIPS 2022 — https://arxiv.org/abs/2201.11903
2. Kojima et al., *Large Language Models are Zero-Shot Reasoners*, NeurIPS 2022 — https://arxiv.org/abs/2205.11916
3. Wang et al., *Self-Consistency*, ICLR 2023 — https://arxiv.org/abs/2203.11171
4. Sprague et al., *To CoT or not to CoT?*, ICLR 2025 — https://arxiv.org/abs/2409.12183
5. Snell et al., *Scaling LLM Test-Time Compute Optimally*, ICLR 2025 — https://arxiv.org/abs/2408.03314
6. OpenAI, *Learning to reason with LLMs* — https://openai.com/index/learning-to-reason-with-llms/
7. DeepSeek-AI, *DeepSeek-R1* — https://arxiv.org/abs/2501.12948
8. Chen et al., *Do NOT Think That Much for 2+3=?* — https://arxiv.org/abs/2412.21187
9. Shojaee et al., *The Illusion of Thinking* — https://arxiv.org/abs/2506.06941
10. Lawsen, *Comment on The Illusion of Thinking* — https://arxiv.org/abs/2506.09250
11. Varela et al., *Rethinking the Illusion of Thinking* — https://arxiv.org/abs/2507.01231
12. Meincke et al., *The Decreasing Value of Chain of Thought in Prompting* — https://arxiv.org/abs/2506.07142
13. Kapoor et al., *AI Agents That Matter*, TMLR 2025 — https://arxiv.org/abs/2407.01502
14. Xu et al., *ReWOO* — https://arxiv.org/abs/2305.18323
15. Xia et al., *Agentless*, FSE 2025 — https://arxiv.org/abs/2407.01489
16. OpenAI, *Introducing SWE-bench Verified* — https://openai.com/index/introducing-swe-bench-verified/
17. OpenAI, *o3 and o4-mini System Card* — https://cdn.openai.com/pdf/2221c875-02dc-4789-800b-e7758f3722c1/o3-and-o4-mini-system-card.pdf · *o1 System Card* — https://arxiv.org/abs/2412.16720
18. OpenAI, *GPT-5 System Card* — https://cdn.openai.com/gpt-5-system-card.pdf
19. Vu et al., *FreshLLMs* — https://arxiv.org/abs/2310.03214
20. Wei et al., *BrowseComp* — https://arxiv.org/abs/2504.12516
21. Schick et al., *Toolformer* — https://arxiv.org/abs/2302.04761
22. Kapoor et al., *Holistic Agent Leaderboard* (HAL) — https://arxiv.org/abs/2510.11977
23. Anthropic, *Building effective agents* — https://www.anthropic.com/engineering/building-effective-agents
24. OpenAI, *Reasoning best practices* — https://developers.openai.com/api/docs/guides/reasoning-best-practices
25. Anthropic docs, thinking — https://platform.claude.com/docs/en/build-with-claude/extended-thinking
26. OpenAI docs, reasoning — https://developers.openai.com/api/docs/guides/reasoning
27. Gema et al., *Inverse Scaling in Test-Time Compute* — https://arxiv.org/abs/2507.14417
28. Chen et al., *Reasoning Models Don't Always Say What They Think* — https://arxiv.org/abs/2505.05410
29. Huang et al., *Large Language Models Cannot Self-Correct Reasoning Yet* — https://arxiv.org/abs/2310.01798
30. Kalai et al., *Why Language Models Hallucinate* — https://arxiv.org/abs/2509.04664
31. Li et al., *Retrieval Augmented Generation or Long-Context LLMs?* (Self-Route) — https://arxiv.org/abs/2407.16833
32. Ma et al., *Reasoning Models Can Be Effective Without Thinking* (NoThinking) — https://arxiv.org/abs/2504.09858
33. Muennighoff et al., *s1: Simple test-time scaling* — https://arxiv.org/abs/2501.19393
34. Wang et al., *Plan-and-Solve Prompting* — https://arxiv.org/abs/2305.04091
35. Chen et al., *Universal Self-Consistency* — https://arxiv.org/abs/2311.17311
36. Gao et al., *PAL: Program-aided Language Models* — https://arxiv.org/abs/2211.10435
37. OpenAI, *A practical guide to building agents* — https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf
38. OpenAI, *Introducing o3 and o4-mini* — https://openai.com/index/introducing-o3-and-o4-mini/
39. Moonshot AI, *Kimi K2 Thinking* model card — https://huggingface.co/moonshotai/Kimi-K2-Thinking
40. Qian et al., *SMART: Self-Aware Agent for Tool Overuse Mitigation*, Findings of ACL 2025 — https://arxiv.org/abs/2502.11435
