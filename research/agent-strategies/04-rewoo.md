# ReWOO — Reasoning WithOut Observation (`rewoo`)

> The planner writes the whole tool plan up front, with placeholders (`#E1`, `#E2`) that later steps reference; workers execute it without further reasoning; a solver answers from the collected evidence.
> Evidence: token savings real in the 2023 few-shot setting (~5× on HotpotQA); accuracy roughly equal to ReAct and contested in replications; the lasting argument is control-flow integrity against prompt injection.
> Last reviewed: 2026-09-24 · Part of [agent-strategies](00-index.md)

## At a glance

- **What it buys:** 2 LLM calls plus N tool calls and a flat context — 1,986 vs 9,795 tokens per HotpotQA question against ReAct (gpt-3.5, 2023) [1]; no runaway loops (0 of 100 failures hit the token limit vs 18 for ReAct) [1].
- **Accuracy:** about ReAct's. +1.6 points in the paper on HotpotQA (but −1.8 exact match) [1]; −1.7 [3] and −6.7 [5] in controlled comparisons.
- **Its weakness is structural:** it cannot adapt to what tools return — wrong entity, empty evidence or an unexpected format flows straight through [1][5][24].
- **2024–26:** native parallel tool calls and prompt caching erode the efficiency case [6][16][17][18]; the idea survives as **security** (tool plan fixed before any untrusted data is read) [11][12][13] and as code-based tool plans [14][15].
- **In this toolkit:** arguments go only to a tool's first parameter and `#E1` substitution corrupts `#E10` — see [In ai-arch-toolkit](#12-in-ai-arch-toolkit).

## 1. Origin

**Xu, Peng, Lei, Mukherjee, Liu, Xu**, "ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models", arXiv 2305.18323 (v1 only, May 2023, marked "under review"; no peer-reviewed version found). Code, a fine-tuned Planner-7B and data are released [1][2].

## 2. How it works

- **Planner (one call)** writes the blueprint before any tool runs, alternating `Plan:` reasoning lines with `#En = Tool[input]` lines; later inputs refer to earlier evidence (`#E3 = Wikipedia[#E2]`).
- **Workers** run each `#E` in order, substituting evidence already collected. The paper relies on an `LLM[...]` worker to extract entities from raw evidence ("What is the main ingredient…, given context `#E1`").
- **Solver** receives the task plus the paired plans and evidence; telling it to treat plans and evidence cautiously improved results.
- **Token argument:** ReAct re-sends context, exemplars and all previous turns every step (input grows roughly quadratically with steps); ReWOO pays for context and question about twice, exemplars once [1].

## 3. Settings and cost in the paper

- gpt-3.5-turbo for all main results (text-davinci-003 gave +2.6 for ReWOO, +1.7 for ReAct); a GPT-4 grader scored semantic accuracy [1].
- Exemplars: 6 (HotpotQA), 1 (TriviaQA, GSM8K), one out-of-task exemplar elsewhere; plans had 2–3 steps. ReAct got the same exemplars plus a tool description.
- Tools: 2–5 per benchmark from Wikipedia, Google, WolframAlpha, LLM, Calculator, SearchDoc.
- Data: HotpotQA 1,000; TriviaQA 1,000; GSM8K 1,000; StrategyQA 300; PhysicsQA 53; SportsUnderstanding 300; SOTUQA.
- Distillation: text-davinci-003 wrote 4,000 blueprints; ~2,000 correct ones trained LLaMA-7B → Alpaca-7B → Planner-7B with LoRA on one RTX 4090.
- Reported cost (HotpotQA): $3.97 vs $19.59 per 1,000 queries (2023 gpt-3.5 prices).

## 4. Evidence

| Benchmark | Model | Baseline | Result | Src |
|---|---|---|---|---|
| HotpotQA (1,000) | gpt-3.5-turbo, 6-shot | ReAct 40.8 acc / **32.2 EM** | 42.4 acc / **30.4 EM**; 4.9× fewer tokens | [1] |
| TriviaQA / GSM8K / StrategyQA / PhysicsQA / SportsU / SOTUQA | gpt-3.5-turbo | ReAct 59.4 / 62.0 / 64.6 / 64.1 / 58.6 / 64.8 | 66.6 / 62.4 / 66.6 / 66.0 / 61.3 / 70.2 | [1] |
| Same tables | gpt-3.5-turbo | no tools | direct answer 80.6 (TriviaQA) and 68.0 (SportsU); CoT 67.4 (GSM8K) — **tools lost** | [1] |
| Six public benchmarks, pooled | gpt-3.5-turbo | ReAct | 63.7% fewer tokens; mean +2.6 points (the paper's "4.4% absolute" matches the relative gain) | [1] |
| HotpotQA, every tool returns "No evidence found" | gpt-3.5-turbo | ReAct drops to 0 | ReWOO drops to 13.2 (−29.2) | [1] |
| 100 HotpotQA failures each | gpt-3.5-turbo | ReAct | bad reasoning 51 vs 76; token limit 0 vs 18; unhelpful tool result 29 vs 20; wrong answer despite good evidence 11 vs 3 | [1] |
| Independent re-implementation (EM) | Llama-2-70B | ReAct 28.7 / 53.8 / 44.1 | HotpotQA 27.0; GSM8K 51.9; TriviaQA 52.4; CoT-SC beat both on HotpotQA and GSM8K | [3] |
| HotpotQA (F1) | Mistral-7B / Llama-2-13B / Llama-2-70B | ReAct 29.2 / 24.2 / 36.8; Reflexion 34.7 / 36.7 / 43.3 | ReWOO 32.1 / 25.1 / 39.0 | [4] |
| One-pass vs iterative planning | LLaMA-2-7B fine-tuned | iterative 45.9 / 65.7 / 47.1 | one-pass 39.2 / 60.6 / 50.5 (HotpotQA / StrategyQA / GSM8K); 1.8–8.3× faster | [5] |
| Offline plans on real REST APIs | text-davinci-003 | ReAct 44.0 / 54.5 | 29.0 / 14.5 | [24] |
| Chain-of-Abstraction (fine-tuned placeholders) | LLaMA-2-Chat-7B | FireAct 26.20 EM, 2.71 s | 28.22 EM, 1.90 s | [7] |
| Interleaved retrieval (IRCoT) | GPT-3, Flan-T5 | retrieve once up front | up to +21 retrieval and +15 QA points | [8] |
| AgentDojo tool filter (tools chosen up front) | GPT-4o | no defence | targeted attack success 57.69% → 6.84% | [12] |
| CaMeL (plan written as code) | o3-high / Gemini 2.5 Pro | native tool calling 84.5% / 73.2% | 77.3% / 41.2%; ~2.8× input and ~2.7× output tokens | [13] |
| Programmatic tool calling (vendor) | Claude | direct tool calls | 43,588 → 27,297 tokens (−37%); internal benchmarks 25.6 → 28.5 and 46.5 → 51.2 | [14] |

## 5. Strengths

- **Few LLM calls, flat context** — real savings in the 2023 few-shot setting [1]; one-pass planning 1.8–8.3× faster than iterative [5]; Chain-of-Abstraction ~1.4× faster [7].
- **No runaway loops**, and more graceful degradation when tools fail [1].
- **Trainable planner:** can be trained offline without running tools; small planners work [1][5].
- **Control-flow integrity:** which tools run, and in what order, is fixed before any untrusted data is read — the same design as AgentDojo's tool filter and the basis of CaMeL's guarantees [11][12][13].
- **Auditable and predictable:** after the planner call you know exactly which tools will run.
- **Intermediate results stay out of the model's context** — the idea behind 2025 programmatic tool calling [14][15].

## 6. Weaknesses and costs

- **No mid-course correction;** tools run in sequence even when independent [9][10]; each entity extraction costs an extra LLM call; the solver prompt holds all evidence.
- **Exploratory tasks** force the plan to enumerate options — no better than the worst case of iterative reasoning, by the authors' own admission [1].
- **Accuracy trails iterative methods** in controlled comparisons: −6.7 [5], −1.7 [3], −1.8 EM in the paper itself [1].
- **The cost advantage is partly a 2023 artefact:** re-sent few-shot exemplars dominated ReAct's tokens [1]; with native function calling ReAct used ~2,900 input tokens per HotpotQA comparison question [6]; prompt caching bills reused prefixes at ~10% [16][17].

## 7. Best use cases

- The shape of the information need is fixed: look up A, look up B, compare or compute; API chains where step 2's input is a field of step 1's output.
- Deterministic tools with predictable outputs (typed outputs help, e.g. MCP `outputSchema`) [23].
- Batch or cost-sensitive workloads with long system prompts where caching is unavailable.
- Tools return untrusted content while side-effecting tools are available — fix the tool plan first.
- The whole tool plan must be approved before anything runs; a distilled or small planner is used.

## 8. Where it falls short

- **No adaptation:** a one-pass planner that never saw the first search result asked about an unrelated entity next [5]; CaMeL-style planners cannot plan actions that depend on data they may not read [13]; the tool filter fails when the tool list cannot be fixed in advance [12]; the authors exclude environments without prior context (ALFWorld) [1].
- **Wrong expectations about tool outputs** (a plan assumed a Wikipedia page would give a person's age); more unhelpful-tool-result failures (29 vs 20) and more solver mistakes despite good evidence (11 vs 3) [1].
- **Real APIs:** wrong APIs and parameters used before they were obtained [24].
- **Irrelevant tools hurt:** 17 of 20 inspected failures were tool misuse [1].
- **Replications do not show consistent accuracy gains;** prompted ReWOO trailed Reflexion [3][4].
- **Security gap:** `#E` substitution pipes untrusted text into later tool arguments and the solver reads everything — control flow is protected, data flow is not [11][13].

## 9. Variants and follow-ups

- **LLMCompiler** (ICML 2024) — DAG plan, parallel execution, replanning (see [05-llm-compiler](05-llm-compiler.md)) [6].
- **Chain-of-Abstraction** (COLING 2025) — fine-tuned placeholder reasoning chains [7].
- **HuggingGPT** — `<resource>-task_id` placeholders play the same role as `#E` [25].
- **Lumos-O vs Lumos-I** (ACL 2024) — a controlled one-pass vs iterative comparison [5].
- **Tree-Planner** — plans sampled up front, grounded choices along a tree [26].
- **CaMeL and the security patterns** — plan-then-execute, code-then-execute, dual LLM [13][11].
- **Programmatic tool calling / code execution with MCP** (2025) [14][15]; **Agentic Plan Caching** [27].

## 10. With 2024–2026 models

- **Parallel tool calls are native** on the major APIs [18][19]; OpenAI's parallel calling alone gave 1.61× speed and 1.87× cost savings over ReAct on HotpotQA comparison questions [6] — most of ReWOO's latency case for fan-out queries is gone.
- **Prompt caching** removes most of the cost of re-sending the prefix, which the 5× claim rests on (inference) [16][17].
- **The frontier moved toward more interleaving:** tools inside the chain of thought [20], thinking between tool calls [21], RL-trained search agents interleaving search with reasoning (+41% for Qwen2.5-7B over RAG baselines) [22] — echoing IRCoT [8].
- **ReWOO's idea survives in two forms:** security (tool plans or code fixed before untrusted data is read; utility cost depends heavily on the model: −7.2 points for o3, −32.0 for Gemini 2.5 Pro) [11][12][13], and efficiency (code-based plans with loops and conditionals, at the cost of a sandbox) [14][15].
- **Gap:** no 2025–26 study compares plain ReWOO with native tool-calling ReAct.

## 11. Router signals

**Choose `rewoo` when:**
- the whole tool sequence can be written from the question alone (fixed hops, each fed by substitution or a single extraction);
- tools are deterministic lookups or computations;
- the data is untrusted and side-effecting tools are present;
- there are ≥3 tool calls and cost or latency matters;
- the tool plan must be pre-approved.

**Avoid it when:**
- the next tool or argument depends on the *meaning* of a result, on exploration or on recovery → `react`, or `plan_execute` with replanning;
- entities are ambiguous across hops;
- tools are flaky or need structured multi-argument input (in this toolkit);
- accuracy matters more than cost;
- the model already knows the answer → `completion` (it won on TriviaQA and SportsU in ReWOO's own table) [1].

**Fallback:** if any evidence comes back empty or as an error, rerun with `react` (the paper has no fallback; LLMCompiler adds replanning).

## 12. In ai-arch-toolkit

Flow: [`_rewoo.py`](../../src/ai_arch_toolkit/toolkit/agents/flows/_rewoo.py), built by `_build_rewoo` in [`_builders.py`](../../src/ai_arch_toolkit/toolkit/agents/_builders.py). State as of 2026-09-24.

| Aspect | Toolkit behaviour |
|---|---|
| Planner prompt | asks only for `#En = Tool[arg]` lines — no `Plan:` reasoning lines, no exemplars |
| Parsing | regex `#E(\d+)\s*=\s*(\w+)\[([^\]]*)\]`; an argument containing `]` breaks it; lines that do not match are **dropped silently** |
| Arguments | the whole argument string goes to the tool's **first** schema parameter — multi-argument tools cannot be planned correctly |
| Substitution | plain `str.replace` per reference — `#E1` **corrupts** `#E10` (same bug class as LLMCompiler's `$10`) |
| Extraction | no `LLM[...]` worker unless an LLM tool is in the group, so raw evidence is substituted verbatim into the next tool's input |
| Failures | unknown tools and exceptions become error-text evidence; no retry, no replan |
| Execution | sequential; 2 LLM calls (planner, solver) + N tool calls — faithful to the paper's cost profile |
| Phases | `planner_llm`, `solver_llm`; knobs `planner_system` (`{tools}` token), `solver_system` |

**Recommended changes:** structured tool arguments (JSON per step), regex-safe placeholder substitution, an optional extraction worker, and a fallback to `react` when evidence is empty or an error.

## 13. Open questions

- Does accuracy parity with ReAct hold for 2025 models with native tools and caching?
- How much token saving remains after zero-shot tool schemas and caching?
- How vulnerable is `#E` data flow to injected content, measured on plain ReWOO?
- Which plan language is best: text `#E`, a JSON DAG, or code?

## Sources

1. Xu et al., *ReWOO* — https://arxiv.org/abs/2305.18323
2. ReWOO code — https://github.com/billxbf/ReWOO
3. Crouse et al., *Formally Specifying the High-Level Behavior of LLM-Based Agents* — https://arxiv.org/abs/2310.08535
4. Qiao et al., *AutoAct*, ACL 2024 — https://arxiv.org/abs/2401.05268
5. Yin et al., *Agent Lumos*, ACL 2024 — https://aclanthology.org/2024.acl-long.670/
6. Kim et al., *LLMCompiler*, ICML 2024 — https://arxiv.org/abs/2312.04511
7. Gao et al., *Chain-of-Abstraction*, COLING 2025 — https://aclanthology.org/2025.coling-main.185/
8. Trivedi et al., *IRCoT*, ACL 2023 — https://aclanthology.org/2023.acl-long.557/
9. LangChain, *Planning Agents* — https://www.langchain.com/blog/planning-agents
10. LangGraph ReWOO tutorial — https://github.com/langchain-ai/langgraph/blob/0.3.0/docs/docs/tutorials/rewoo/rewoo.ipynb
11. Beurer-Kellner et al., *Design Patterns for Securing LLM Agents against Prompt Injections* — https://arxiv.org/abs/2506.08837
12. Debenedetti et al., *AgentDojo* — https://arxiv.org/abs/2406.13352
13. Debenedetti et al., *CaMeL* — https://arxiv.org/abs/2503.18813
14. Anthropic, *Advanced tool use* (programmatic tool calling) — https://www.anthropic.com/engineering/advanced-tool-use
15. Anthropic, *Code execution with MCP* — https://www.anthropic.com/engineering/code-execution-with-mcp
16. Anthropic docs, prompt caching — https://platform.claude.com/docs/en/docs/build-with-claude/prompt-caching
17. OpenAI docs, prompt caching — https://developers.openai.com/api/docs/guides/prompt-caching
18. OpenAI docs, function calling — https://developers.openai.com/api/docs/guides/function-calling
19. Anthropic docs, tool use — https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
20. OpenAI, *o3 and o4-mini System Card* — https://cdn.openai.com/pdf/2221c875-02dc-4789-800b-e7758f3722c1/o3-and-o4-mini-system-card.pdf
21. Anthropic docs, interleaved thinking — https://platform.claude.com/docs/en/build-with-claude/extended-thinking
22. Jin et al., *Search-R1* — https://arxiv.org/abs/2503.09516
23. MCP specification 2025-06-18, tools — https://modelcontextprotocol.io/specification/2025-06-18/server/tools
24. Song et al., *RestGPT* — https://arxiv.org/abs/2306.06624
25. Shen et al., *HuggingGPT* — https://arxiv.org/abs/2303.17580
26. Hu et al., *Tree-Planner* — https://arxiv.org/abs/2310.08582
27. Zhang et al., *Agentic Plan Caching* — https://arxiv.org/abs/2506.14852
