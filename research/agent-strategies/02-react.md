# ReAct (`react`)

> The default agent loop: think, call a tool, observe the result, repeat until a final answer. Today it runs on native function calling, with the model's own thinking between tool calls.
> Evidence: strong for grounding and adaptivity (tool loops removed CoT's hallucination failures); the *prompted* version's central claim — that interleaved thoughts help — is contested; minimal loops with good tools remain competitive with elaborate scaffolds.
> Last reviewed: 2026-09-24 · Part of [agent-strategies](00-index.md)

## At a glance

- **What it buys:** grounding and adaptivity — hallucination caused 56% of CoT's HotpotQA failures and 0% of ReAct's [1]; each action depends on a real observation, so unknown step counts and exceptions are handled.
- **What it costs:** sequential calls and a growing context — 9,795 tokens per HotpotQA question vs 482 for CoT (gpt-3.5, 2023) [7]; agents "succeed quickly and fail slowly" (failed SWE-agent runs averaged 21 steps and $2.52 vs a median 12 steps and $1.21 for successes) [5].
- **Known failure modes, still present in 2025 studies:** loops (>90% of turn-limit failures repeated near-identical responses) [6], premature stopping (~85% of runs stopped before 8 required searches) [8], hallucinated tool arguments [3][21], context rot [30][31][32], low repeated-trial reliability (pass^8 < 25% in τ-bench retail) [3].
- **What matters most now:** native function calling over text parsing [3][14]; a few well-designed tools (SWE-agent's interface 18.0% vs plain shell 11.0%) [5]; passing the model's reasoning back between tool calls [18][19][26]; context management [30].
- **In this toolkit:** native tool calling, parallel calls, a final-turn hint; thinking is replayed for Anthropic/Gemini/Meta but not for OpenAI (Chat Completions) — see [In ai-arch-toolkit](#12-in-ai-arch-toolkit).

## 1. Origin

**Yao, Zhao, Yu, Du, Shafran, Narasimhan, Cao**, "ReAct: Synergizing Reasoning and Acting in Language Models", ICLR 2023; arXiv 2210.03629; code github.com/ysymyth/ReAct [1]. Antecedents: **MRKL** (AI21, 2022) routes requests to expert modules [23]; **Toolformer** (NeurIPS 2023) teaches itself API calls but cannot chain them interactively [22]. Today the pattern is the default meaning of "agent" — LLMs using tools in a loop driven by environment feedback [24][25].

## 2. How it works

Each step the model emits a thought and either an action (tool + arguments) or a final answer; the runtime executes the action, appends the observation, and calls the model again with the whole trajectory.

- **Original, prompt-based:** few-shot Thought/Action/Observation exemplars; a stop sequence at `\nObservation` so the model cannot write its own observations; ≤7 steps on HotpotQA; T=0. Knowledge tasks get a thought every step; long decision tasks (ALFWorld) get sparse thoughts [1].
- **Native function calling:** tool calls arrive in a structured field and results return as tool messages; the loop exits on no tool call, a final-output tool, an error or the turn limit [25].
- **Reasoning models:** the "thought" is the model's own thinking between tool calls (interleaved thinking), which must be returned to the API unmodified alongside the tool results [26][27].

## 3. Settings in the paper and later work

- **HotpotQA:** 6 exemplars, ≤7 steps; **FEVER:** 3 exemplars, ≤5 steps; Wikipedia API with `search` (first 5 sentences), `lookup`, `finish` [1].
- **Hybrids:** ReAct → CoT-SC fallback when no answer within the step budget; CoT-SC → ReAct when the majority answer gets < n/2 of 21 votes [1].
- **ALFWorld:** 3 annotated trajectories per task type, 134 unseen games; **WebShop:** one-shot, 500 instructions; PaLM-540B main results; fine-tuned PaLM-8B/62B on 3,000 trajectories [1].
- **Later budgets:** τ-bench ≤30 actions, ≥3 trials [3]; SWE-agent $4 per instance, ~30–40 turns [5]; AgentBench 5–35 rounds [6]; CodeAct ≤10 turns [4].

## 4. Evidence

| Benchmark | Model / setting | Baseline(s) | ReAct / result | Src |
|---|---|---|---|---|
| HotpotQA EM | PaLM-540B | standard 28.7; CoT 29.4; CoT-SC 33.4; Act-only 25.7 | **ReAct 27.4**; ReAct→CoT-SC 35.1; CoT-SC→ReAct 34.2 | [1] |
| FEVER accuracy | PaLM-540B | 57.1 / 56.3 / 60.4 / 58.9 (same order) | ReAct 60.9; CoT-SC→ReAct 64.6 | [1] |
| HotpotQA error analysis (200 trajectories) | PaLM-540B | CoT | hallucination: CoT 56% vs ReAct 0% of failures; reasoning errors incl. loops: ReAct 47% vs CoT 16%; bad search results 23% of ReAct failures | [1] |
| ALFWorld (134) | PaLM-540B | Act best-of-6 45; BUTLER 37 | ReAct best-of-6 71, average 57 | [1] |
| WebShop success / score | PaLM-540B | Act 30.1; IL+RL 28.7 | 40.0 / 66.6 (human expert 59.6) | [1] |
| ALFWorld ablations (6 tasks) | GPT-3.5 / GPT-3.5-instruct / GPT-4 / Claude-3-Opus | base ReAct 27.6 / 44.7 / 23.3 / 56.6 | reasoning up front instead of interleaved 46.6 / 61.9 / 43.3 / 50; irrelevant "magic" guidance 30 / 41 / 36.6 / 30; exemplars from another task 1.6 / 5.2 / 0 / 6.6 | [2] |
| τ-bench pass^1 retail / airline | gpt-4o, native FC | — | 61.2 / 35.2; pass^8 < 25% in retail; native FC > text ReAct > text Act | [3] |
| M3ToolEval success (turns) | gpt-4-1106 | JSON actions 52.4 (7.6); text 53.7 (7.7) | code actions (CodeAct) 74.4 (5.5) | [4] |
| SWE-bench Lite | GPT-4 Turbo | RAG 2.67; plain shell loop 11.00 ($1.46) | designed agent-computer interface (SWE-agent) 18.00 ($1.67) | [5] |
| HotpotQA accuracy / tokens | gpt-3.5-turbo | ReWOO 42.4 / 1,986 | ReAct 40.8 / 9,795 (CoT 41.6 / 482) | [7] |
| Movie Recommendation | gpt-3.5-turbo | LLMCompiler 77.13%, 5.47 s | ReAct 68.60%; with anti-looping prompts 72.47%, 20.47 s | [8] |
| HotpotQA EM / WebShop score | GPT-3.5 | LATS 0.63 / 75.9 | ReAct 0.32 / 53.8; best-of-k 0.38 / 59.1 | [9] |
| ToolBench pass rate | ChatGPT | DFSDT (tree search) 63.8 | ReAct 35.3; ReAct repeated to match DFSDT's cost 44.5 | [11] |
| τ-bench airline pass^1 (pass^5) | Claude 3.7 Sonnet | 0.332 (0.100) | think tool + optimised prompt 0.584 (0.340); extended thinking 0.412 (0.160) | [13] |
| SWE-bench Verified, OpenHands | o1, high effort | text actions 29.1% | native function calling 47.7% | [14] |
| SWE-bench Verified | Claude 3.5 Sonnet + prompt + Bash + Edit | prior SOTA 45% | 49% | [15] |
| BrowseComp | GPT-4o with browsing / o1 without tools | — | 1.9% / 9.9%, against 51.5% for the Deep Research agent — browsing alone is not enough | [17] |
| Keeping vs discarding earlier thinking (vendor) | MiniMax-M2 | discard | τ² 64 → 87; BrowseComp 31.4 → 44.0 | [18] |
| τ-bench retail (vendor) | GPT-5 | 73.9% | 78.2% when earlier reasoning items are passed back | [19] |

**Claims vs confirmation:** the interleaving claim is contested for the prompted form — on ALFWorld, gains track exemplar–query similarity rather than interleaving [2]; τ-bench (same group) confirms ReAct > Act for text formats but native FC beats both [3]; wins over ReAct in ReWOO, LLMCompiler, LATS, ToolLLM and Reflexion are reported by the competing methods' authors, and LLMCompiler needed ReAct-specific prompting to make ReAct competitive [8]; cost-controlled comparisons show elaborate loops do not beat simple retry on HumanEval [12]; 2025 interleaved-thinking gains are mostly vendor-reported [18][19].

## 5. Strengths

- **Grounding** — 0% hallucination failures vs 56% for CoT [1].
- **Adaptivity** — suited to open-ended problems where the number of steps cannot be predicted [24].
- **Simplicity that aged well** — a prompt plus two tools reached 49% on SWE-bench Verified [15]; a ~100-line bash-only agent claims >74% [16].
- **Trainable** — fine-tuned PaLM-8B ReAct beat all PaLM-62B prompting [1]; FireAct +77% on HotpotQA for Llama2-7B [34]; Search-R1 +41% over RAG baselines [36].
- **Interpretable** — the trajectory is a log.

## 6. Weaknesses and costs

- **Tokens:** full context re-sent every step; τ-bench gpt-4o agent $0.38/task, 95.9% of it input [3]; Anthropic's SWE-bench runs often took hundreds of turns and >100k tokens [15]. Prompt caching softens this (cache reads ~0.1× input) [38].
- **Latency:** sequential steps (7.12 s vs 3.95 s for a parallel DAG on HotpotQA; 20.47 s vs 5.47 s on Movie) [8].
- **Failures are the expensive runs** [5]; **variance** — reliability falls sharply over repeated trials [3].
- **More tools, more mistakes** — 17 of 20 inspected failures involved the wrong tool [7].
- **Safety** — agents looked up benchmark answers online and misused credit cards in HAL [20].

## 7. Best use cases

- Observation-dependent tasks with uncertain step counts: debugging, repository navigation, investigative search, troubleshooting, conversational transactions [24].
- Informative, reliable environment feedback: tests, compilers, APIs, databases [24].
- Multi-hop, long-tail questions where grounding is worth the tokens [1].
- A few well-designed native tools — interface design is decisive [5].

## 8. Where it falls short

- **Loops:** repeated thoughts and actions are a frequent error [1]; >90% of turn-limit failures repeated near-identical responses in their last 10 rounds [6]; ~10% of HotpotQA runs looped and scored <10% [8].
- **Premature stopping:** ~85% of runs stopped before 8 required searches [8]; 19.4% of gpt-4o τ-bench failures were partially resolved requests [3]; models believe they succeeded while hidden tests fail [15].
- **Error propagation:** a mistaken action traps the model in a faulty loop [11]; an edit succeeds 90.5% of the time overall but only 57.2% after one failed edit [5]; when every tool returned "No evidence found", ReAct dropped to 0 [7].
- **Hallucinated observations and arguments:** the stop sequence exists because prompted models write their own observations [1]; non-existent IDs per task: gpt-4o FC 0.46, gpt-3.5 FC 2.08, gpt-3.5 text 6.34 [3]; MIRAGE-Bench GPT-4o hallucination rate 0.339 [21].
- **Context growth:** accuracy drops mid-context [32] and as input grows [31]; SWE-agent 15.0 with full history vs 18.0 with history processing [5].
- **One path, no backtracking:** 35.3 vs 63.8 for tree search (44.5 at matched cost) [11].
- **Brittle prompted exemplars** [2]; **rule-following** — 25% of gpt-4o τ-bench failures were wrong policy decisions [3]; performance drops sharply when the user also acts on the environment (dual control) [37].
- **Weak on knowledge the model already has** — HotpotQA 27.4 vs CoT 29.4 [1]; TriviaQA 59.4 vs direct 80.6 [7].

## 9. Variants and follow-ups

- Act-only and the ReAct ↔ CoT-SC fallbacks [1]; **IRCoT** — retrieval interleaved with CoT (up to +21 retrieval, +15 QA) [35].
- **Reflexion** [10] and **LATS** [9] (see [06-reflexion](06-reflexion.md), [10-lats](10-lats.md)); **ReWOO** [7] and **LLMCompiler** [8]; **DFSDT** tree search [11].
- **CodeAct** — actions as code [4]; **SWE-agent** — a designed agent-computer interface [5]; **mini-swe-agent** [16].
- **FireAct** [34] and **Search-R1** [36] — trained loops.
- Anthropic's **think tool** [13] and **interleaved thinking** [26].

## 10. With 2024–2026 models

- **Native function calling replaced text parsing:** o1 went from 29.1% to 47.7% on SWE-bench Verified with native calls [14]; BFCL — single-turn calls are strong, memory, dynamic decisions and long horizons remain open [33].
- **The thought moved into the model:** o3/o4-mini call tools inside their chain of thought, RL-trained to decide when [28]; Anthropic's interleaved thinking is automatic with adaptive thinking on recent models [26]; hundreds of consecutive tool calls are claimed (vendor) [29].
- **Pass earlier reasoning back** with tool results [19][26][27] — vendor-reported gains [18][19].
- **Explicit "think" steps flipped:** a think function did not help τ-bench agents in 2024 [3]; the 2025 think tool raised airline pass^1 0.332 → 0.584; by Dec 2025 Anthropic recommended extended thinking instead in most cases [13].
- **The reasoning–action dilemma:** reasoning models overthink more in agent loops (overthinking score 3.51 vs 2.23); picking less-overthought runs gave ~+30% accuracy at 43% lower cost [14]; higher effort did not help in 21 of 36 HAL runs [20]; OpenAI suggests lower effort and tool-call budgets against over-eager tool use [19].
- **Minimal loops hold up, interfaces matter:** HAL's generalist scaffold was cheaper in 20 of 24 comparisons but less accurate than task-specific ones [20].
- **API caveat:** on Anthropic, extended thinking allows only `tool_choice` auto/none, and some newer models reject forced tool use entirely — loops should not depend on forcing a tool [26].

## 11. Router signals

**Choose `react` when:**
- at least one tool is required and at least one step's input depends on an earlier result;
- the step count is uncertain but moderate (~2–30);
- feedback is informative (tests, errors, API responses);
- tools are native, few and well described;
- the model supports interleaved thinking and its reasoning is passed back;
- grounding is worth a few to tens of calls.

**Avoid it and route elsewhere when:**
- no tool is needed → `completion` [7][39];
- the plan is known up front or the steps fan out → `rewoo` / `llm_compiler` (ReAct stopped early on ~85% of 8-item tasks and ran 1.8–3.7× slower) [7][8];
- every item must be covered exhaustively → a planner with an explicit checklist (`plan_execute`);
- backtracking is needed and a reliable evaluator exists → `tot` / `lats`, budget permitting [11][12];
- a cheap verifier plus retries would do [12];
- irreversible or policy-heavy actions need high pass^k → gated workflows [3];
- hundreds of steps without compaction or sub-agents [30].

**Router features:** needs tools (fresh/private data, actions)? step dependency? fan-out? verifier available? risk level and latency budget? native FC and interleaved thinking supported? self-consistency agreement (the n/2 gate) [1]?

## 12. In ai-arch-toolkit

Flow: [`_react.py`](../../src/ai_arch_toolkit/toolkit/agents/flows/_react.py), built by `_build_react` in [`_builders.py`](../../src/ai_arch_toolkit/toolkit/agents/_builders.py). State as of 2026-09-24. It is also the inner loop of `plan_execute`, `reflexion`, `generate_review`, `self_discovery`, `llm_compiler` and `lats` (via `run_react`).

| Aspect | Toolkit behaviour |
|---|---|
| Format | native tool calling (no text Thought/Action parsing, no stop sequences) |
| Budget | `max_iterations` 10 (spec default) |
| Parallel calls | `parallel_tool_calls` (default on): several calls from one turn run concurrently |
| Final turn | `final_answer_hint` (default on) asks for a text answer without tools on the last turn; `strip_tools_on_final` removes the tools entirely; `show_turn_counter` optional |
| Tool errors | exceptions become retryable `ToolResult` failures returned to the model; a budget denial is terminal; every call goes through the governed executor (approval gates, metering, output caps, deadlines) |
| Reasoning replay | each turn appends `response.to_message()`, which keeps the provider's raw content under `_raw`: the Anthropic adapter replays thinking blocks and signatures as received, Gemini keeps thought signatures, Meta replays encrypted reasoning. The OpenAI adapter uses Chat Completions only, so there are no reasoning items to replay — OpenAI's reported gain from passing reasoning back comes from the Responses API [19] |
| Structured output | `output_schema` supported |
| Knobs | `parallel_tool_calls`, `final_answer_hint`, `strip_tools_on_final`, `show_turn_counter` |

**Gaps relative to the evidence:** no loop/repetition detection, no context compaction or old-tool-result clearing for long runs [30], no tool-call budget separate from the turn limit.

## 13. Open questions

- Does reasoning between tool calls help beyond exemplar mimicry on modern reasoning models? Independent ablations are missing [2][18].
- What effort level should each step use [14][20]?
- Principled stopping rules, given that agents succeed quickly and fail slowly [5].
- How to raise pass^k reliability [3]; how compaction affects accuracy [30].

## Sources

1. Yao et al., *ReAct*, ICLR 2023 — https://arxiv.org/abs/2210.03629 · https://github.com/ysymyth/ReAct
2. Verma et al., *On the Brittle Foundations of ReAct Prompting* — https://arxiv.org/abs/2405.13966
3. Yao et al., *τ-bench*, ICLR 2025 — https://arxiv.org/abs/2406.12045
4. Wang et al., *CodeAct*, ICML 2024 — https://arxiv.org/abs/2402.01030
5. Yang et al., *SWE-agent*, NeurIPS 2024 — https://arxiv.org/abs/2405.15793
6. Liu et al., *AgentBench*, ICLR 2024 — https://arxiv.org/abs/2308.03688
7. Xu et al., *ReWOO* — https://arxiv.org/abs/2305.18323
8. Kim et al., *LLMCompiler*, ICML 2024 — https://arxiv.org/abs/2312.04511
9. Zhou et al., *LATS*, ICML 2024 — https://arxiv.org/abs/2310.04406
10. Shinn et al., *Reflexion*, NeurIPS 2023 — https://arxiv.org/abs/2303.11366
11. Qin et al., *ToolLLM*, ICLR 2024 — https://arxiv.org/abs/2307.16789
12. Kapoor et al., *AI Agents That Matter* — https://arxiv.org/abs/2407.01502
13. Anthropic, *The "think" tool* — https://www.anthropic.com/engineering/claude-think-tool
14. Cuadron et al., *The Danger of Overthinking* — https://arxiv.org/abs/2502.08235
15. Anthropic, *SWE-bench with Claude 3.5 Sonnet* — https://www.anthropic.com/research/swe-bench-sonnet
16. *mini-swe-agent* — https://github.com/SWE-agent/mini-swe-agent
17. Wei et al., *BrowseComp* — https://arxiv.org/abs/2504.12516
18. MiniMax, *Why is interleaved thinking important for M2* — https://www.minimax.io/news/why-is-interleaved-thinking-important-for-m2
19. OpenAI, *GPT-5 prompting guide* — https://developers.openai.com/cookbook/examples/gpt-5/gpt-5_prompting_guide
20. Kapoor et al., *Holistic Agent Leaderboard* (HAL), ICLR 2026 — https://arxiv.org/abs/2510.11977
21. *MIRAGE-Bench* — https://arxiv.org/abs/2507.21017
22. Schick et al., *Toolformer*, NeurIPS 2023 — https://arxiv.org/abs/2302.04761
23. Karpas et al., *MRKL Systems* — https://arxiv.org/abs/2205.00445
24. Anthropic, *Building effective agents* — https://www.anthropic.com/engineering/building-effective-agents
25. OpenAI, *A practical guide to building agents* — https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf
26. Anthropic docs, thinking — https://platform.claude.com/docs/en/build-with-claude/extended-thinking
27. OpenAI docs, reasoning — https://developers.openai.com/api/docs/guides/reasoning
28. OpenAI, *Introducing o3 and o4-mini* — https://openai.com/index/introducing-o3-and-o4-mini/
29. Moonshot AI, *Kimi K2 Thinking* — https://huggingface.co/moonshotai/Kimi-K2-Thinking
30. Anthropic, *Effective context engineering for AI agents* — https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
31. Chroma, *Context Rot* — https://www.trychroma.com/research/context-rot
32. Liu et al., *Lost in the Middle* — https://arxiv.org/abs/2307.03172
33. Patil et al., *BFCL*, ICML 2025 — https://proceedings.mlr.press/v267/patil25a.html
34. Chen et al., *FireAct* — https://arxiv.org/abs/2310.05915
35. Trivedi et al., *IRCoT* — https://arxiv.org/abs/2212.10509
36. Jin et al., *Search-R1* — https://arxiv.org/abs/2503.09516
37. Barres et al., *τ²-bench* — https://arxiv.org/abs/2506.07982
38. Prompt caching — https://platform.claude.com/docs/en/build-with-claude/prompt-caching · https://developers.openai.com/api/docs/guides/prompt-caching
39. Qian et al., *SMART* (tool overuse) — https://arxiv.org/abs/2502.11435
