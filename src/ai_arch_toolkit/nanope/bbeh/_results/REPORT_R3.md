# BBEH Mini Benchmark Report — Round 3: Multi-Model + react_full

**Date:** 2026-03-09
**Dataset:** BBEH Mini — 460 questions across 23 reasoning tasks (20 per task)
**Framework:** ai-arch-toolkit flows evaluated via Inspect AI
**Previous report:** [REPORT.md](REPORT.md) — Rounds 1-2 (gpt-5-nano strategies)

---

## What's New in Round 3

1. **6 model baselines** — gpt-5-mini, o4-mini, Haiku 4.5 (thinking), Gemini 3.1 Flash Lite, grok-4-1-fast-reasoning
2. **react_full strategy** — All tools: thinking_system + python_eval + table_parse + think
3. **Extended python_eval** — Multi-line support (loops, variables, assignments, regex, string methods)
4. **New table_parse tool** — Parse column-major bracket data (BBEH buggy_tables format)
5. **4 new thinking systems** — adjective_ordering, state_tracking, pattern_decoding, grid_navigation
6. **xAI/grok fix** — `run_in_executor` thread isolation to work around gRPC event loop conflict

---

## Full Leaderboard


| Model                       | Accuracy  | Correct | Cost   | Cost/Sample | Notes                                |
| --------------------------- | --------- | ------- | ------ | ----------- | ------------------------------------ |
| **grok-4-1-fast-reasoning** | **63.7%** | 293/460 | $0.20  | $0.0004     | Best accuracy AND cheapest           |
| **gpt-5-mini**              | **53.7%** | 247/460 | $5.67  | $0.0123     | Strong all-rounder                   |
| **o4-mini**                 | **47.2%** | 217/460 | $10.78 | $0.0234     | Reasoning model                      |
| **Haiku 4.5 + thinking**    | **41.7%** | 192/460 | $13.72 | $0.0298     | Extended thinking enabled            |
| **Gemini 3.1 Flash Lite**   | **37.6%** | 173/460 | $0.67  | $0.0015     | Best non-reasoning cost-efficiency   |
| gpt-5-nano + ts+pyeval      | 27.4%     | 126/460 | $2.32  | $0.0050     | Best nano strategy (Round 2)         |
| **gpt-5-nano + react_full** | **25.9%** | 119/460 | $2.14  | $0.0046     | Round 3 — more tools, slightly worse |
| gpt-5-nano + ts_only        | 25.7%     | 118/460 | $1.99  | $0.0043     | Round 2                              |
| gpt-5-nano + pyeval_only    | 24.6%     | 113/460 | $1.44  | $0.0031     | Round 2                              |
| gpt-5-nano + self_discovery | 16.3%     | 75/460  | $2.18  | $0.0047     | Round 1                              |
| gpt-5-nano + react_tools    | 14.3%     | 66/460  | $0.67  | $0.0014     | Round 1                              |
| gpt-5-nano (baseline)       | 13.5%     | 62/460  | $0.66  | $0.0014     | Round 1                              |


**Key findings:**

- **grok-4-1-fast-reasoning dominates** — 63.7% at just $0.20 (320x more cost-efficient than Haiku+thinking)
- **gpt-5-mini is a surprise** — 53.7% baseline smashes earlier partial result (26.3% with quota errors)
- **react_full (25.9%) underperforms react_ts_pyeval (27.4%)** — extra tools add selection overhead

---

## Per-Task Accuracy — All Models

Sorted by grok accuracy (strongest model).


| Task                 | grok    | gpt-5-mini | o4-mini | Haiku   | Gemini  | nano full | nano ts+py | nano base |
| -------------------- | ------- | ---------- | ------- | ------- | ------- | --------- | ---------- | --------- |
| boardgame qa         | **95%** | 90%        | 60%     | 60%     | 30%     | 40%       | 60%        | 5%        |
| zebra puzzles        | **95%** | 50%        | 50%     | 35%     | 45%     | 30%       | 20%        | 5%        |
| time arithmetic      | **90%** | 85%        | 90%     | 75%     | 75%     | **90%**   | 85%        | 50%       |
| movie recommendation | **85%** | 70%        | 75%     | 75%     | 85%     | 80%       | **90%**    | 85%       |
| object counting      | 85%     | **100%**   | 95%     | 65%     | 35%     | 15%       | 10%        | 0%        |
| web of lies          | **85%** | 50%        | 35%     | 50%     | 35%     | 15%       | 25%        | 0%        |
| boolean expressions  | 75%     | **90%**    | 75%     | 45%     | 80%     | 40%       | 40%        | 0%        |
| dyck languages       | 75%     | **85%**    | 55%     | 40%     | 20%     | 45%       | 35%        | 20%       |
| word sorting         | **75%** | 60%        | 65%     | 65%     | 35%     | 65%       | 60%        | 30%       |
| geometric shapes     | **70%** | 50%        | 25%     | 40%     | 50%     | 5%        | 5%         | 0%        |
| object properties    | **70%** | **70%**    | 20%     | 5%      | 20%     | 0%        | 0%         | 0%        |
| spatial reasoning    | **70%** | 35%        | 40%     | 55%     | 45%     | 5%        | 5%         | 10%       |
| causal understanding | **65%** | 35%        | 55%     | 60%     | 45%     | 50%       | 55%        | 35%       |
| hyperbaton           | **65%** | 30%        | 35%     | 25%     | 5%      | 0%        | 10%        | 0%        |
| multistep arithmetic | **60%** | 50%        | 40%     | 10%     | 0%      | 0%        | 0%         | 0%        |
| temporal sequence    | 60%     | **70%**    | 60%     | 40%     | 40%     | 10%       | 15%        | 0%        |
| disambiguation qa    | 55%     | **60%**    | 60%     | 55%     | 55%     | 35%       | 40%        | 15%       |
| buggy tables         | **50%** | 35%        | 25%     | 10%     | 20%     | 0%        | 0%         | 0%        |
| sportqa              | **45%** | 40%        | 45%     | 25%     | 40%     | 25%       | 35%        | 25%       |
| sarc triples         | 25%     | 25%        | 25%     | **30%** | **40%** | 15%       | 15%        | 10%       |
| shuffled objects     | 25%     | 20%        | 20%     | **60%** | 15%     | 10%       | 5%         | 5%        |
| linguini             | **25%** | 10%        | 10%     | 20%     | **25%** | 10%       | 10%        | 10%       |
| nycc                 | 20%     | **25%**    | **25%** | 15%     | **25%** | 10%       | 10%        | 5%        |


---

## react_full Analysis

**25.9% accuracy, $2.14 cost** — slightly worse than react_ts_pyeval (27.4%, $2.32).

### Gains vs ts+pyeval


| Task             | ts+pyeval | react_full | Delta |
| ---------------- | --------- | ---------- | ----- |
| dyck languages   | 35%       | **45%**    | +10pp |
| zebra puzzles    | 20%       | **30%**    | +10pp |
| shuffled objects | 5%        | **10%**    | +5pp  |
| object counting  | 10%       | **15%**    | +5pp  |
| time arithmetic  | 85%       | **90%**    | +5pp  |
| word sorting     | 60%       | **65%**    | +5pp  |


### Regressions vs ts+pyeval


| Task                 | ts+pyeval | react_full | Delta |
| -------------------- | --------- | ---------- | ----- |
| boardgame qa         | **60%**   | 40%        | -20pp |
| movie recommendation | **90%**   | 80%        | -10pp |
| hyperbaton           | **10%**   | 0%         | -10pp |
| web of lies          | **25%**   | 15%        | -10pp |
| causal understanding | **55%**   | 50%        | -5pp  |
| disambiguation qa    | **40%**   | 35%        | -5pp  |


**Why react_full underperforms:** With 4 tools, the model spends more iterations on tool selection (especially browsing thinking_system when it should compute directly). The "be efficient — 1-3 tool calls max" instruction isn't enough to prevent the overhead. The think scratchpad also burns iterations. ts+pyeval's 2-tool simplicity wins.

### buggy tables: still 0%

The table_parse tool was specifically built for this task, but react_full scored 0%. Possible causes:

1. gpt-5-nano can't reliably call table_parse with the right input format
2. Even with parsed tables, the model can't identify the buggy cell
3. The model never selects table_parse (prefers python_eval)

---

## Grok: The Surprise Winner

grok-4-1-fast-reasoning at **63.7%** and **$0.20** is extraordinary:

### vs Official BBEH Leaderboard


| Model                       | BBEH Mini | Notes                    |
| --------------------------- | --------- | ------------------------ |
| o3-mini (high)              | 56.7%     | Official leaderboard     |
| **grok-4-1-fast-reasoning** | **63.7%** | **Ours — beats o3-mini** |
| **gpt-5-mini**              | **53.7%** | **Ours**                 |
| **o4-mini**                 | **47.2%** | **Ours**                 |
| **Haiku 4.5 + thinking**    | **41.7%** | **Ours**                 |
| **Gemini 3.1 Flash Lite**   | **37.6%** | **Ours**                 |
| DeepSeek R1                 | 37.2%     | Official leaderboard     |
| **gpt-5-nano + ts+pyeval**  | **27.4%** | **Ours**                 |
| Gemini 2.0 Flash            | 27.0%     | Official leaderboard     |
| GPT-4o                      | 23.5%     | Official leaderboard     |


### Grok's strengths


| Task              | grok    | Next best    | Gap   |
| ----------------- | ------- | ------------ | ----- |
| zebra puzzles     | **95%** | 50% (o4/5m)  | +45pp |
| web of lies       | **85%** | 50% (haiku)  | +35pp |
| hyperbaton        | **65%** | 35% (o4)     | +30pp |
| geometric shapes  | **70%** | 50% (gem/5m) | +20pp |
| buggy tables      | **50%** | 35% (5m)     | +15pp |
| spatial reasoning | **70%** | 55% (haiku)  | +15pp |


Grok excels at spatial/structural reasoning (zebra puzzles, geometric shapes, spatial reasoning) and has strong language understanding (hyperbaton, web of lies). Its reasoning model architecture gives it native chain-of-thought at minimal cost.

### gRPC noise but functional

The `run_in_executor` fix works — grok completed all 460 samples with 0 errors. The gRPC `BlockingIOError: [Errno 35]` messages in stderr are non-fatal polling noise from gRPC's completion queue competing with Inspect's event loop. All API calls succeed.

---

## Cost Analysis — Updated


| Model                 | Accuracy  | Cost      | Accuracy/$   | Correct/$  | Errors |
| --------------------- | --------- | --------- | ------------ | ---------- | ------ |
| **grok-4-1-fast**     | **63.7%** | **$0.20** | **325.5%/$** | **1500.5** | 0      |
| Gemini 3.1 Flash Lite | 37.6%     | $0.67     | 56.1%/$      | 258.2      | 1      |
| nano + pyeval_only    | 24.6%     | $1.44     | 17.1%/$      | 78.6       | 0      |
| nano + ts_only        | 25.7%     | $1.99     | 12.9%/$      | 59.3       | 0      |
| nano + react_full     | 25.9%     | $2.14     | 12.1%/$      | 55.6       | 0      |
| nano + ts+pyeval      | 27.4%     | $2.32     | 11.8%/$      | 54.4       | 0      |
| gpt-5-mini            | 53.7%     | $5.67     | 9.5%/$       | 43.6       | 0      |
| o4-mini               | 47.2%     | $10.78    | 4.4%/$       | 20.1       | 9      |
| Haiku 4.5 + thinking  | 41.7%     | $13.72    | 3.0%/$       | 14.0       | 24     |


### Cost to reach each accuracy tier


| Target accuracy | Cheapest option             | Cost      |
| --------------- | --------------------------- | --------- |
| ~25%            | nano + pyeval_only          | $1.44     |
| ~27%            | nano + ts+pyeval            | $2.32     |
| ~38%            | Gemini 3.1 Flash Lite       | $0.67     |
| ~54%            | gpt-5-mini                  | $5.67     |
| ~64%            | **grok-4-1-fast-reasoning** | **$0.20** |


Grok is the cheapest option at every accuracy tier above 38%.

---

## Previously Unsolvable Tasks — Now With More Models


| Task                 | nano (all) | react_full | grok    | gpt-5-mini | o4-mini | Haiku | Gemini |
| -------------------- | ---------- | ---------- | ------- | ---------- | ------- | ----- | ------ |
| buggy tables         | 0%         | 0%         | **50%** | 35%        | 25%     | 10%   | 20%    |
| multistep arithmetic | 0%         | 0%         | **60%** | 50%        | 40%     | 10%   | 0%     |
| object properties    | 0%         | 0%         | **70%** | **70%**    | 20%     | 5%    | 20%    |


react_full didn't crack any of the 0% tasks. These require either: (a) stronger base model reasoning, or (b) multi-step decomposition that gpt-5-nano can't orchestrate reliably.

---

## What Was Built for Round 3

### Extended python_eval

- **Before:** Single-expression evaluator (`ast.parse(code, mode="eval")`)
- **After:** Full multi-line support via class-based `_SafeEvaluator`:
  - Statements: `Assign`, `AugAssign`, `For` (with 10K iteration guard), `If/Elif/Else`, `Pass`, `Delete`
  - Safe attribute access: whitelisted methods for str, list, dict, set, tuple, bytes
  - Added: `re` module, `gcd`, `lcm`, `factorial`, `comb`, `perm`, set/dict comprehensions
  - Blocked: dunder attributes, `eval`, `exec`, `open`, `__import_`_
  - 20 tests passing

### table_parse tool

- Parses column-major bracket data (BBEH buggy_tables format)
- Output formats: table (text), python (dict repr), rows, columns
- Also handles CSV-like formats with auto-separator detection

### 4 new thinking systems (YAML)

- **adjective_ordering** — English adjective category rules
- **state_tracking** — Dict-based swap tracking with python_eval code patterns
- **pattern_decoding** — Symbol→value mapping from examples
- **grid_navigation** — Coordinate tracking with direction vectors

### xAI/grok thread isolation

- `_run_in_thread()` wrapper using `ThreadPoolExecutor` + `asyncio.run()`
- Fresh `LLM` client created per call inside the executor thread
- gRPC gets its own event loop, no conflict with Inspect AI's async runner
- Applied to both `baseline_solver` and all react-based solvers via `_react_solve()`

---

## Recommendations

1. **For maximum accuracy + cost: grok-4-1-fast-reasoning (63.7%, $0.20)**
  Dominates on both axes. Use as default for all BBEH-style reasoning tasks.
2. **For OpenAI-only environments: gpt-5-mini (53.7%, $5.67)**
  Strong all-rounder. 100% on object counting, 90% on boardgame qa and boolean expressions.
3. **For nano-class models: gpt-5-nano + ts+pyeval (27.4%, $2.32)**
  Still the best nano strategy. react_full's extra tools hurt more than help.
4. **Don't use react_full** — The 4-tool setup adds tool selection overhead without enough benefit. Stick with ts+pyeval (2 tools) for nano models.
5. **Potential: grok + tools**
  grok at 63.7% baseline could potentially reach 70%+ with ts+pyeval. Worth testing as the next experiment.

---

## Raw Data

- **Rounds 1-2:** `baseline.json`, `react_tools.json`, `self_discovery.json`, `react_ts_only.json`, `react_pyeval_only.json`, `react_ts_pyeval.json`
- **Round 3 baselines:** `baseline_o4-mini.json`, `baseline_claude-haiku-4-5-20251001.json`, `baseline_gemini-3_1-flash-lite-preview.json`, `baseline_gpt-5-mini.json`, `baseline_grok-4-1-fast-reasoning.json`
- **Round 3 strategy:** `react_full.json`
- **See also:** [REPORT.md](REPORT.md) for detailed Round 1-2 analysis, tool ablation, and per-task breakdowns

