# AI Agent & LLM Benchmarks Research

> Last updated: 2026-03-08
> Context: Research for evaluating `ai-arch-toolkit` against public benchmarks/leaderboards.

## Table of Contents

- [Safety Classification](#safety-classification)
- [Top Picks for ai-arch-toolkit](#top-picks-for-ai-arch-toolkit)
- [Benchmark Details](#benchmark-details)
  - [BFCL v4 (Berkeley Function Calling Leaderboard)](#1-bfcl-v4)
  - [tau-bench / tau2-bench](#2-tau-bench--tau2-bench)
  - [HotpotQA](#3-hotpotqa)
  - [GPQA Diamond](#4-gpqa-diamond)
  - [Humanity's Last Exam (HLE)](#5-humanitys-last-exam-hle)
  - [BIG-Bench Extra Hard (BBEH)](#6-big-bench-extra-hard-bbeh)
  - [GAIA](#7-gaia)
  - [BrowseComp](#8-browsecomp)
  - [IFEval](#9-ifeval)
  - [SimpleQA](#10-simpleqa)
  - [MMLU-Pro](#11-mmlu-pro)
  - [LiveBench](#12-livebench)
  - [ALFWorld](#13-alfworld)
  - [MATH-500 / GSM8K](#14-math-500--gsm8k-saturated)
  - [ARC-AGI-2](#15-arc-agi-2)
- [Unsafe Benchmarks (for reference)](#unsafe-benchmarks-for-reference)
- [Leaderboard Links](#leaderboard-links)
- [Detailed Setup Guides (Top 5)](#detailed-setup-guides-top-5)
- [Cost Summary](#cost-summary)
- [Sources](#sources)

---

## Safety Classification

| Benchmark | Safe? | Notes |
|-----------|-------|-------|
| **BFCL v4** | SAFE | AST comparison of function calls, no execution |
| **tau-bench** | SAFE | Simulated DB via JSON, no OS ops |
| **HotpotQA** | SAFE | Pure text QA |
| **GPQA Diamond** | SAFE | Multiple-choice QA |
| **HLE** | SAFE | Multiple-choice + short answer QA |
| **BBEH** | SAFE | Text reasoning |
| **IFEval** | SAFE | Text generation + programmatic verification |
| **SimpleQA** | SAFE | Factual Q&A |
| **MMLU-Pro** | SAFE | Multiple-choice QA |
| **LiveBench** | SAFE (mostly) | Coding tasks may need execution |
| **ALFWorld** | SAFE | Sandboxed text game engine |
| **MATH-500 / GSM8K** | SAFE | Pure math reasoning |
| **ARC-AGI-2** | SAFE | Grid puzzles |
| **GAIA** | DEPENDS | Questions safe; agent needs web + possibly code exec |
| **BrowseComp** | DEPENDS | Questions safe; agent needs web browsing |
| SWE-bench | NOT SAFE | Executes code patches + test suites |
| AgentBench | NOT SAFE | Real bash in Docker |
| MINT-Bench | NOT SAFE | Executes arbitrary Python |
| WebArena | NOT SAFE | Real browser actions on hosted sites |
| Terminal-Bench | NOT SAFE | Real shell commands |
| FrontierMath | NOT SAFE | Python execution for verification |

---

## Top Picks for ai-arch-toolkit

### Priority 1: BFCL v4
- **Why**: Directly benchmarks `@tool` decorator and `ToolGroup` against every other framework
- **Framework fit**: Perfect — tests function calling accuracy (the core of our tool system)
- **Effort**: Low-Medium
- **Leaderboard**: gorilla.cs.berkeley.edu/leaderboard.html

### Priority 2: tau-bench / tau2-bench
- **Why**: Tests exactly what `react_flow` does — multi-turn agent + tools + policy
- **Framework fit**: Perfect — `react_flow` with domain-specific tools
- **Effort**: Low (pip install + API keys)
- **Leaderboard**: taubench.com

### Priority 3: HotpotQA
- **Why**: Multi-hop reasoning, great for `react_flow` + `plan_execute_flow` + knowledge tools
- **Framework fit**: High
- **Effort**: Low (download JSON, build simple harness)
- **Leaderboard**: hotpotqa.github.io

### Priority 4: GPQA Diamond + HLE + BBEH
- **Why**: Unsaturated reasoning benchmarks where `reflexion_flow`, `tot_flow`, `self_discovery_flow` can beat raw LLM
- **Framework fit**: Medium-High (proves the framework adds value over raw LLM calls)
- **Effort**: Low-Medium per benchmark
- **Leaderboards**: Scale AI SEAL, HuggingFace

### Priority 5: GAIA
- **Why**: Gold standard agent benchmark — all the big names compete here
- **Framework fit**: Very High (but needs web browsing + file parsing tools)
- **Effort**: Medium-High (needs web tools, possibly code execution)
- **Leaderboard**: huggingface.co/spaces/gaia-benchmark/leaderboard

---

## Benchmark Details

### 1. BFCL v4

**Berkeley Function Calling Leaderboard**

- **What it tests**: Tool/function calling accuracy across 22 task types: simple calls, parallel calls, multi-turn, agentic (web search, memory), hallucination detection. Multiple languages (Python, Java, JavaScript, REST).
- **Tasks**: 2000+ test instances across 22 categories
- **Scoring**: AST-based evaluation. Overall = Agentic (40%) + Multi-Turn (30%) + Live (10%) + Non-Live (10%) + Hallucination (10%)
- **Dataset**: Public on HuggingFace (gorilla-llm/Berkeley-Function-Calling-Leaderboard) and GitHub
- **Top scores (2025-2026)**: Claude Opus 4.1: 70.36%, Claude Sonnet 4: 70.29%, GPT-5: 59.22%
- **Setup**: `pip install bfcl-eval`, requires API keys for the LLM provider
- **Leaderboard**: https://gorilla.cs.berkeley.edu/leaderboard.html
- **GitHub**: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
- **Paper**: https://openreview.net/forum?id=2GmDdhBdDk

**Key for us**: Can we adapt the evaluation to test our `@tool` decorator output format? Need to check if BFCL supports custom agent implementations or only raw LLM API calls.

---

### 2. tau-bench / tau2-bench

**Tool-Agent-User Benchmark (Sierra Research)**

- **What it tests**: Multi-turn customer service conversations with simulated users, domain-specific API tools, and policy compliance.
- **Domains**: tau-retail (~115 tasks), tau-airline (~50 tasks), tau2-telecom (new)
- **Scoring**: Binary pass/fail based on final database state vs annotated goal. Novel metric: `pass^k` = probability ALL k trials succeed (measures reliability).
- **Top scores**: GPT-4o ~50% pass^1 on retail, pass^8 drops to ~25%. Claude 3.7 Sonnet: 49% on telecom.
- **Components**:
  - Simulated user (LLM-powered, follows scenario script)
  - Agent under test (has API tools, must follow policy document)
  - Simulated database (in-memory JSON)
- **Setup**:
  ```bash
  # tau-bench v1
  git clone https://github.com/sierra-research/tau-bench && cd tau-bench
  pip install -e .
  python run.py --agent-strategy tool-calling --env retail --model gpt-4o --model-provider openai --user-model gpt-4o --user-model-provider openai --user-strategy llm

  # tau2-bench (recommended)
  git clone https://github.com/sierra-research/tau2-bench && cd tau2-bench
  pip install -e .
  cp .env.example .env  # edit with API keys
  tau2 run --domain retail --agent-llm gpt-4o --user-llm gpt-4o --num-trials 3
  ```
- **API keys needed**: For BOTH the agent LLM AND the user simulator LLM
- **Cost**: Many multi-turn conversations × multiple trials = significant API costs
- **Leaderboard**: https://taubench.com/
- **GitHub**: https://github.com/sierra-research/tau-bench (v1), https://github.com/sierra-research/tau2-bench (v2)
- **Paper**: https://arxiv.org/abs/2406.12045 (ICLR 2025)
- **Also**: tau2-bench-verified by Amazon AGI fixes task definition issues: https://github.com/amazon-agi/tau2-bench-verified

**Key for us**: Can we plug `react_flow` as a custom agent? Need to check the agent interface/protocol.

---

### 3. HotpotQA

**Multi-Hop Question Answering over Wikipedia**

- **What it tests**: Multi-hop reasoning — questions requiring information from 2+ Wikipedia paragraphs. Bridge reasoning (infer intermediate entities) and comparison questions.
- **Tasks**: 112,779 QA pairs (7,405 dev set)
- **Settings**: Distractor (10 paragraphs: 2 gold + 8 distractors) and Full Wiki (open retrieval)
- **Scoring**: Exact Match (EM) and F1 score. Also evaluates supporting fact identification.
- **Dataset**: Direct download from https://hotpotqa.github.io/ (JSON format)
- **Leaderboard**: https://hotpotqa.github.io/ (still active)
- **No special infrastructure needed** — just questions and context paragraphs
- **Cost**: One LLM call per question (or multiple for agent loops). 7K dev questions × ~$0.01-0.05 each.

**Key for us**: Perfect for `react_flow` + Wikipedia tool. The original ReAct paper (Yao et al.) used HotpotQA as a primary evaluation. Well-established baseline to compare against.

---

### 4. GPQA Diamond

**Graduate-Level Google-Proof Q&A**

- **What it tests**: PhD-level science questions (biology, physics, chemistry) that are "Google-proof" — domain experts can answer them, but non-experts cannot even with internet access.
- **Tasks**: 198 questions (Diamond subset, highest quality). Also: Extended (546), Main (448).
- **Format**: Multiple-choice, 4 options. Random baseline: 25%.
- **Scoring**: Accuracy (% correct)
- **Top scores**: Gemini 3.1 Pro: 94.1%, GPT-5.4: 92.0%. PhD experts: ~65-74%.
- **Dataset**: https://github.com/idavidrein/gpqa, HuggingFace
- **Leaderboard**: https://artificialanalysis.ai/evaluations/gpqa-diamond, Epoch AI tracker

**Key for us**: `reflexion_flow` and `tot_flow` could improve over single-pass LLM. Proves the framework adds reasoning value. Small dataset = cheap to run.

---

### 5. Humanity's Last Exam (HLE)

**Expert-Level Cross-Domain Questions**

- **What it tests**: 2,500 questions across mathematics, sciences, humanities — contributed by subject-matter experts globally. Designed to be at the frontier of human knowledge.
- **Format**: Multiple-choice and short-answer with unambiguous, verifiable answers.
- **Scoring**: Accuracy. Temperature 0.0.
- **Top scores (2026)**: Gemini 3 Pro: 37.5%, Claude Opus 4.6 Thinking Max: 34.4%, GPT-5 Pro: 31.6%. Expert humans (in domain): ~90%.
- **Dataset**: https://huggingface.co/datasets/cais/hle (public test set). Private held-out set exists.
- **Evaluation code**: https://github.com/centerforaisafety/hle
- **Leaderboard**: https://scale.com/leaderboard/humanitys_last_exam (Scale AI SEAL)

**Key for us**: Very far from saturated. Deep reasoning flows could genuinely help. The gap between 35% and 90% is where agent architectures can shine.

---

### 6. BIG-Bench Extra Hard (BBEH)

**Successor to BIG-Bench Hard**

- **What it tests**: 23 capability categories (same as BBH but much harder). Logical deduction, causal reasoning, algorithmic problems, etc.
- **Scoring**: Varies by task. Harmonic mean across tasks used for overall score.
- **Top scores**: Best reasoning model ~54.2% (micro-average), harmonic mean as low as 9.8%. BBH (predecessor) is saturated at >90%.
- **Dataset**: Paper: https://arxiv.org/abs/2502.19187. Dataset released by Google.
- **BBH (predecessor)**: https://github.com/suzgunmirac/BIG-Bench-Hard, https://huggingface.co/datasets/maveriq/bigbenchhard

**Key for us**: `self_discovery_flow` is literally designed for this kind of task — selecting reasoning modules and adapting them to the problem. Perfect showcase.

---

### 7. GAIA

**General AI Assistants Benchmark**

- **What it tests**: Real-world questions requiring reasoning, multi-modality, web browsing, and tool use.
- **Tasks**: 466 total (165 validation with answers, ~300 test hidden)
- **Levels**: 1 (easy, <5 steps), 2 (5-10 steps, multiple tools), 3 (arbitrarily hard)
- **Scoring**: Quasi-exact-match on unambiguous factual answers. Binary correct/incorrect.
- **Top scores**: H2O.ai: ~75% overall. Human baseline: 92%.
- **Dataset**: Gated on HuggingFace — need approval: https://huggingface.co/datasets/gaia-benchmark/GAIA
- **Attached files**: PDFs, Excel, images, audio, video, code files
- **Agent needs**: Web search, web page reading, file parsing, code execution (for best scores), multi-step reasoning
- **Evaluation harness**: Inspect AI (UK AISI): https://github.com/UKGovernmentBEIS/inspect_evals
  ```bash
  inspect eval inspect_evals/gaia --model openai/gpt-4o
  ```
- **Leaderboard submission**: JSONL with `task_id`, `model_answer`, `reasoning_trace` → upload to HuggingFace leaderboard
- **Leaderboard**: https://huggingface.co/spaces/gaia-benchmark/leaderboard
- **Paper**: https://arxiv.org/abs/2311.12983

**Key for us**: The flagship agent benchmark. But requires web browsing + file parsing tools we don't have yet. Priority 5 — build toward this.

---

### 8. BrowseComp

**OpenAI's Web Browsing Benchmark**

- **What it tests**: Ability to locate hard-to-find information on the internet. Answers are NOT discoverable from the first page of 5 different Google searches.
- **Tasks**: 1,266 problems
- **Scoring**: Binary CORRECT/INCORRECT via LLM judge
- **Top scores**: GPT-5: 54.9%, Deep Research (OpenAI): ~51%, Human trainers: 29.2%
- **Dataset**: https://huggingface.co/datasets/smolagents/browse_comp
- **Leaderboard**: https://www.kaggle.com/benchmarks/openai/browsecomp

**Key for us**: Ideal for deep research agent (`react_flow` + `plan_execute_flow` with web search). Needs web tools.

---

### 9. IFEval

**Instruction Following Evaluation**

- **What it tests**: Ability to follow verifiable format constraints (word count, keyword frequency, bullet points, capitalization, etc.)
- **Tasks**: ~500 prompts with 25 types of verifiable instructions
- **Scoring**: Binary pass/fail per instruction, verified by heuristic rules. Prompt-level and instruction-level accuracy.
- **Top scores**: Kimi K2.5: 94.0%, GLM-4.7: 88.0%
- **Dataset**: https://huggingface.co/datasets/google/IFEval
- **Evaluation**: via EleutherAI lm-evaluation-harness

**Key for us**: A `react_flow` with a "constraint checker" tool could genuinely improve scores through self-verification loops.

---

### 10. SimpleQA

**Factual Accuracy Benchmark (OpenAI)**

- **What it tests**: Short-form factual accuracy. 4,326 adversarially collected questions.
- **Scoring**: F1-score (CORRECT / INCORRECT / NOT_ATTEMPTED grading by LLM judge)
- **Top scores**: Gemini 2.5 Pro: 55.6 F1
- **Dataset**: Public CSV from OpenAI. SimpleQA Verified (1,000 prompts) on Kaggle.
- **Note**: Designed for parametric knowledge only — using tools trivializes it.

**Key for us**: Limited value for agent framework (tools defeat the purpose). Good for raw LLM comparison only.

---

### 11. MMLU-Pro

**Graduate-Level Knowledge (10 choices)**

- **What it tests**: 12,000 questions across 14 disciplines with 10 answer choices (vs 4 in MMLU).
- **Scoring**: Multiple-choice accuracy. Random baseline: 10%.
- **Top scores**: Gemini 3 Pro: 90.1%, Claude Opus 4.5: ~90%
- **Dataset**: https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro
- **Leaderboard**: Kaggle, Artificial Analysis

**Key for us**: Reasoning chains and self-consistency voting could help. But mostly tests knowledge, not agent capabilities.

---

### 12. LiveBench

**Monthly-Refreshed Multi-Category Benchmark**

- **What it tests**: 6 categories (math, coding, reasoning, language, instruction following, data analysis), 18 tasks. Refreshed monthly to prevent contamination.
- **Scoring**: Objective ground-truth, no LLM judge. Average across categories.
- **Top scores**: Below 70% for best models
- **Website**: https://livebench.ai/
- **GitHub**: https://github.com/LiveBench/LiveBench

**Key for us**: Multi-strategy evaluation — different agent architectures per category. Monthly refresh means no memorization advantage.

---

### 13. ALFWorld

**Text-Based Embodied Agent Tasks**

- **What it tests**: Household tasks via text commands (e.g., "put a clean mug on the shelf"). 6 task types.
- **Tasks**: 3,500+
- **Scoring**: Binary success rate
- **Top scores**: LLM agents with ReAct: 70-90%+
- **GitHub**: https://github.com/alfworld/alfworld
- **Paper**: https://arxiv.org/abs/2010.03768

**Key for us**: Perfect for `react_flow` in a text game environment. Safe, sandboxed.

---

### 14. MATH-500 / GSM8K (Saturated)

**Competition Math / Grade School Math**

- **MATH-500**: 500 competition math problems. Frontier models: 95-98%. **Saturated.**
- **GSM8K**: 1,319 grade school word problems. Frontier models: 95-99%. **Saturated.**
- **Use only as**: Baseline sanity check, or for comparing reasoning flows vs raw LLM.
- **Datasets**: HuggingFace (HuggingFaceH4/MATH-500, openai/gsm8k)
- **GSM8K-Platinum** (error-corrected): https://huggingface.co/datasets/madrylab/gsm8k-platinum

---

### 15. ARC-AGI-2

**Abstraction and Reasoning Corpus**

- **What it tests**: Novel pattern recognition on visual grid puzzles. Closest benchmark to "general intelligence."
- **Format**: Given 2-3 example input-output grid pairs, predict output for test input.
- **Top scores (2025)**: Human ~60%, Best commercial model (Claude Opus 4.5 Thinking): 37.6%, Best refined solution: 54% ($30/task)
- **Leaderboard**: https://arcprize.org/leaderboard
- **Competition**: https://www.kaggle.com/competitions/arc-prize-2025

---

## Unsafe Benchmarks (for reference)

These require code execution, Docker, or OS-level operations. Listed for completeness but NOT recommended for our use case.

| Benchmark | What it tests | Why unsafe |
|-----------|--------------|------------|
| **SWE-bench** | Resolve real GitHub issues | Executes code patches + test suites in Docker |
| **AgentBench** | 8 environments (OS, DB, web, etc.) | OS env runs real bash; multiple Docker containers |
| **MINT-Bench** | Multi-turn tool interaction | Executes arbitrary Python code |
| **WebArena** | Web browsing on hosted sites | Real browser actions, modifies DB state |
| **Terminal-Bench** | Command-line workflows | Real shell commands |
| **FrontierMath** | Research-level math | Python execution for answer verification |
| **CUB** | Computer use workflows | OS-level actions |

---

## Leaderboard Links

| Benchmark | Leaderboard URL |
|-----------|----------------|
| BFCL v4 | https://gorilla.cs.berkeley.edu/leaderboard.html |
| tau-bench | https://taubench.com/ |
| HotpotQA | https://hotpotqa.github.io/ |
| GPQA Diamond | https://artificialanalysis.ai/evaluations/gpqa-diamond |
| HLE | https://scale.com/leaderboard/humanitys_last_exam |
| GAIA | https://huggingface.co/spaces/gaia-benchmark/leaderboard |
| BrowseComp | https://www.kaggle.com/benchmarks/openai/browsecomp |
| IFEval | https://llm-stats.com/benchmarks/ifeval |
| SimpleQA | https://www.kaggle.com/benchmarks/deepmind/simpleqa-verified |
| MMLU-Pro | https://www.kaggle.com/benchmarks/open-benchmarks/mmlu-pro |
| LiveBench | https://livebench.ai/ |
| ARC-AGI-2 | https://arcprize.org/leaderboard |
| SWE-bench | https://www.swebench.com/ |
| Arena-Hard | https://github.com/lmarena/arena-hard-auto |
| HAL (aggregated) | https://hal.cs.princeton.edu/ |
| AIME 2025 | https://matharena.ai/ |

---

## Evaluation Harnesses

| Harness | Benchmarks Supported | Link |
|---------|---------------------|------|
| **EleutherAI lm-evaluation-harness** | MATH, GSM8K, MMLU, BBH, DROP, ARC, IFEval, 60+ more | https://github.com/EleutherAI/lm-evaluation-harness |
| **Inspect AI (UK AISI)** | GAIA, SWE-bench, others | https://github.com/UKGovernmentBEIS/inspect_evals |
| **BFCL eval** | BFCL v4 | `pip install bfcl-eval` |
| **tau2-bench** | tau-bench domains | `pip install -e .` from repo |
| **DeepEval** | GSM8K, MATH, others | https://deepeval.com/ |
| **OpenAI simple-evals** | SimpleQA, others | https://github.com/openai/simple-evals |

---

## Detailed Setup Guides (Top 5)

### Setup: BFCL v4

**Install** (Python 3.10+):
```bash
conda create -n BFCL python=3.10 && conda activate BFCL
git clone https://github.com/ShishirPatil/gorilla.git
cd gorilla/berkeley-function-call-leaderboard
pip install -e .
# OR: pip install bfcl-eval  (WARNING: package is bfcl-eval, NOT bfcl)
```

**If using PyPI install**, set: `export BFCL_PROJECT_ROOT=/path/to/your/working/directory`

**Dataset**: Auto-downloaded on first `bfcl generate`. NOT gated. ~4,000-5,000 test cases total. Do NOT use `datasets.load_dataset()` — format is incompatible.

**Run**:
```bash
bfcl generate --model gpt-4o-2024-11-20-FC --test-category simple_python --num-threads 1
bfcl evaluate --model gpt-4o-2024-11-20-FC --test-category simple_python
```

**Output format** your agent must produce:
```python
# decode_ast: list of dicts
[{"func1": {"param1": "val1", "param2": "val2"}}]
# decode_execute: list of callable strings
["func1(param1=val1, param2=val2)"]
```

**Custom agent integration**: Write a handler class in `bfcl_eval/model_handler/`, implement `decode_ast()` + `decode_execute()`, register in `model_config.py` + `handler_map.py`. Significant effort — BFCL is designed around raw LLM API calls, not agent frameworks.

**Leaderboard submission**: Open via PR to gorilla repo. Model must be publicly accessible.

**Cost**: ~$5-20 for a full run with GPT-4o-class pricing.

**API keys**: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc. in `.env`. `SERPAPI_API_KEY` needed for v4 web_search category.

**Gotchas**:
- Temperature matters enormously (up to 10% accuracy swing). Use 0.0.
- `-FC` suffix = native function calling mode; without = prompting mode.
- Result files named `BFCL_v3_*` even for v4 categories.
- Dot-to-underscore: some models convert dots in function names. Must be configured.

---

### Setup: tau2-bench

**Install** (Python 3.10+, 25 dependencies):
```bash
git clone https://github.com/sierra-research/tau2-bench && cd tau2-bench
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env  # add API keys
tau2 check-data
```

**Domains**: retail (115 tasks), airline (50 tasks), telecom (114 tasks, dual-control)

**Run**:
```bash
# Smoke test (cheap)
tau2 run --domain airline --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 1 --num-tasks 5

# Full evaluation
tau2 run --domain retail --agent-llm claude-sonnet-4-20250514 --user-llm gpt-4.1 --num-trials 4

# Interactive play mode (you act as agent or user)
tau2 play

# Explore domain tools via ReDoc
tau2 domain airline
```

**Custom agent integration**: Subclass `LocalAgent`, implement `generate_next_message(message, state) -> (AssistantMessage, state)`. Register in `src/tau2/agent/registry.py`. Agent receives tools as JSON Schema + policy as system prompt.

**Critical constraint**: Agent messages must contain EITHER text OR tool calls, never both.

**Tool examples** (retail): `get_user_details`, `get_order_details`, `cancel_pending_order`, `modify_pending_order_items`, `exchange_delivered_order_items`, `think` (scratchpad)

**API keys**: Need keys for BOTH agent + user simulator LLMs. Also needs `OPENAI_API_KEY` for `gpt-4o-mini` (NL assertion evaluator).

**Cost**: ~$0.61/task. Full retail trial: ~$70. Full airline trial: ~$30. Full leaderboard run (3 domains × 4 trials): **$500-800**.

**Leaderboard**: Open via PR. Run all 3 domains with `--task-split-name base`, then `tau2 submit prepare` + `tau2 submit validate`.

**Gotchas**:
- Ground truth errors in multiple tasks. Consider amazon-agi/tau2-bench-verified.
- 8-9 point accuracy swings between identical runs (LLM user simulator nondeterminism).
- `--max-concurrency > 1` can corrupt results (shared user tool state, Issue #154).
- Non-editable install fails silently without `TAU2_DATA_DIR` env var.

---

### Setup: HotpotQA

**Get data**:
```bash
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json  # 44 MB, 7405 questions
wget https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py
```

**No official agent harness** — build your own loop:
```python
import json
with open("hotpot_dev_distractor_v1.json") as f:
    data = json.load(f)

predictions = {}
for example in data[:500]:  # standard: 500-sample subset
    answer = your_agent.run(example["question"])
    predictions[example["_id"]] = answer

with open("predictions.json", "w") as f:
    json.dump({"answer": predictions, "sp": {}}, f)
```

**Evaluate**: `python hotpot_evaluate_v1.py predictions.json hotpot_dev_distractor_v1.json`

**Agent tools** (ReAct-style): `Search[query]` (Wikipedia API), `Lookup[term]` (Ctrl+F on current page), `Finish[answer]` (submit).

**Distractor vs Full Wiki**: Use distractor for agent eval (10 provided paragraphs, no retriever needed). Full Wiki requires Wikipedia dump or live API.

**Cost**: 500 questions × 5-7 LLM calls each ≈ 2.5-7.5M input tokens. **~$12-37 with Claude Sonnet, ~$57-166 with Opus.**

**Reference implementations**: [ysymyth/ReAct](https://github.com/ysymyth/ReAct) (canonical), [AgentLite](https://github.com/SalesforceAIResearch/AgentLite/tree/main/benchmark/hotpotqa) (cleanest harness).

**MAJOR GOTCHA — read this**: [Peng Qi's critique](https://qipeng.me/blog/stop-using-hotpotqa/) — data contamination (2017 Wikipedia in LLM training data), temporal drift (answers changed since 2017), extractive format mismatch. Consider supplementing with FRAMES or FanOutQA. The leaderboard is largely inactive since 2023.

---

### Setup: GPQA Diamond + HLE + BBEH

#### GPQA Diamond (198 questions, cheapest to start)

**Get data**: HuggingFace [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) (gated, click-through). Or GitHub zip with password `deserted-untie-orchid`.

```python
ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
```

**CRITICAL**: Shuffle answer choices yourself! CSV has correct answer labeled explicitly. Shuffle per-question with fixed seed.

**Eval**: 0-shot or 0-shot CoT. Parse letter (A/B/C/D) from response. Simple accuracy.

**Cost**: ~$0.50-5 single pass, ~$3-15 with Reflexion (3 retries).

**Gotchas**: Saturating at 90%+ for frontier models. Password is in the README (contamination risk).

#### Humanity's Last Exam (2,500 questions)

**Get data**: HuggingFace [cais/hle](https://huggingface.co/datasets/cais/hle) (gated, click-through).

```python
ds = load_dataset("cais/hle", split="test")
```

**Format**: ~24% multiple-choice, ~76% short-answer. Some questions include images.

**Eval**: Two-phase — generate predictions, then grade with LLM judge (o3-mini recommended). Official code at [centerforaisafety/hle](https://github.com/centerforaisafety/hle).

**Cost**: $20-80 single pass (frontier model). $100-500+ with reasoning models. Judge pass: $10-30 extra. With agent scaling (15 variants): $300-1200.

**Leaderboard**: Scale AI SEAL. Contact `agibenchmark@safe.ai`.

**Gotchas**: No train/val split (only test). Multimodal questions (filter by `image is not None`). Top models score ~35%.

#### BBEH (4,520 questions full / 460 mini)

**Get data**: GitHub [google-deepmind/bbeh](https://github.com/google-deepmind/bbeh) (public, not gated).

```python
ds = load_dataset("BBEH/bbeh")  # unofficial HuggingFace mirror
```

**Eval**: 0-shot, exact match against `target` field. Aggregate: harmonic mean across 23 tasks.

**Also available via Inspect AI**:
```bash
inspect eval inspect_evals/bbeh_mini  # start with mini (460 examples)
```

**Cost**: $1-4 mini, $10-40 full (single pass).

**Gotchas**: Harmonic mean = 0% if ANY task scores 0%. Task diversity requires task-specific prompting. Best reasoning model ~45%.

**Agent framework approach**: Task-specific flow selection:
- Logic tasks → `react_flow` with code execution tool
- Math tasks → `react_flow` with calculator tool
- Linguistic tasks → `reflexion_flow` with self-evaluation

---

### Setup: GAIA

**Get data**: HuggingFace [gaia-benchmark/GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) — gated but **instant approval** (click-through, no human review).

```python
from huggingface_hub import snapshot_download
data_dir = snapshot_download(repo_id="gaia-benchmark/GAIA", repo_type="dataset")
```

**Splits**: Validation (165 questions with answers), Test (301 questions, hidden). Levels 1/2/3.

**Attached files**: PDFs, Excel, images, audio, Python files, presentations. A few hundred MB total.

**Minimal eval loop** (no harness needed):
```python
for example in dataset:
    question = example["Question"]
    file_path = os.path.join(data_dir, example["file_path"]) if example["file_name"] else None
    answer = my_agent.solve(question, file_path=file_path)
    correct = normalize(answer) == normalize(example["Final answer"])
```

**Inspect AI harness** (requires Docker for code execution sandbox):
```bash
pip install inspect-ai inspect-evals
inspect eval inspect_evals/gaia --model openai/gpt-4o
```

**Leaderboard submission**: JSONL with `task_id` + `model_answer` + `reasoning_trace` → upload at HuggingFace leaderboard page. Open to anyone.

**Without code execution**: Realistic ceiling ~25-35%. Level 1: 30-50%, Level 2: 10-25%, Level 3: 0-10%. Every top system uses sandboxed code execution.

**Cost**: $5-20 validation run (GPT-4o), $10-40 (Claude). Full test: ~2x.

**Current leaders (2026)**: MiroFlow 82.4%, MiroThinker 80.8%, OpenAI Deep Research 67.4%.

**Gotchas**:
- Old `GAIA.py` loading script broken with `datasets >= 4.0.0`. Use `snapshot_download`.
- External data dependencies rot over time (web pages move/change).
- Reasoning models (o3-class) often WORSE than non-reasoning (GPT-4o) — slower, costlier, hit recursion limits.
- Tool quality matters as much as model quality.
- YouTube questions need frame analysis, not just transcripts.

---

## Cost Summary

| Benchmark | Smoke Test | Full Run | Leaderboard Run |
|-----------|-----------|----------|-----------------|
| **BFCL v4** | ~$1 (one category) | $5-20 | Same (submit via PR) |
| **tau2-bench** | ~$3 (5 tasks, 1 trial) | $100-200 (1 domain, 4 trials) | $500-800 (3 domains, 4 trials) |
| **HotpotQA** | ~$2 (50 questions) | $12-37 (500 questions, Sonnet) | N/A (leaderboard inactive) |
| **GPQA Diamond** | ~$1 | $1-5 (single pass) | N/A (report in paper) |
| **HLE** | ~$5 (100 questions) | $20-80 + $10-30 judge | Contact Scale AI |
| **BBEH mini** | ~$1 | $1-4 | Submit PR to GitHub |
| **BBEH full** | N/A | $10-40 | Submit PR to GitHub |
| **GAIA** | ~$3 (20 validation Qs) | $5-20 (165 validation) | $10-40 (301 test) |

*Prices for Claude Sonnet / GPT-4o class models. Reasoning models (o3, Claude thinking) multiply by 3-10x. Agent frameworks with retries/branching multiply by iteration count.*

---

## Sources

### Papers
- GAIA: https://arxiv.org/abs/2311.12983
- tau-bench: https://arxiv.org/abs/2406.12045
- tau2-bench: https://arxiv.org/abs/2506.07982
- BFCL: https://openreview.net/forum?id=2GmDdhBdDk
- HotpotQA: https://arxiv.org/abs/1809.09600
- GPQA: https://arxiv.org/abs/2311.12022
- MATH: https://arxiv.org/abs/2103.03874
- BBEH: https://arxiv.org/abs/2502.19187
- ALFWorld: https://arxiv.org/abs/2010.03768
- WebArena: https://arxiv.org/abs/2307.13854
- AgentBench: https://arxiv.org/abs/2308.03688
- MINT-Bench: https://arxiv.org/abs/2309.10691
- ToolBench: https://arxiv.org/abs/2307.16789
- SWE-bench: https://arxiv.org/abs/2310.06770

### Guides & Analysis
- Best AI Agent Benchmarks 2025: https://o-mega.ai/articles/the-best-ai-agent-evals-and-benchmarks-full-2025-guide
- Agent Benchmark Compendium: https://github.com/philschmid/ai-agent-benchmark-compendium
- Evidently AI Agent Benchmarks: https://www.evidentlyai.com/blog/ai-agent-benchmarks
- Alan Blog (tau-bench critique): https://medium.com/alan/benchmarking-ai-agents-stop-trusting-headline-scores-start-measuring-trade-offs-0fdae3a418cf
- HuggingFace GAIA Guide: https://huggingface.co/blog/beating-gaia
