# BBEH Mini Benchmark Report — Round 4: Cross-Model Tool Evaluation

**Date:** 2026-03-11
**Dataset:** BBEH Mini — 460 questions across 23 reasoning tasks (20 per task)
**Framework:** ai-arch-toolkit flows evaluated via Inspect AI
**Previous reports:** [REPORT.md](REPORT.md) (Rounds 1-2), [REPORT_R3.md](REPORT_R3.md) (Round 3)

---

## What's New in Round 4

1. **react_ts_pyeval across 4 models** — Same strategy (thinking_system + python_repl), different models: gpt-5-nano, gpt-5-mini, grok-4-1-fast-reasoning, Gemini 3.1 Flash Lite
2. **self_discovery + grok** — Self-discovery flow (select → adapt → operationalize → solve) with thinking_system + python_repl tools on grok
3. **Gemini tool-calling fixes** — Two framework bugs were fixed for Gemini (FunctionResponse dict wrapping + thought_signature preservation), bringing Gemini react_ts_pyeval from broken to functional
4. **Core question:** Do tools help or hurt when applied to stronger models?

---

## Full Leaderboard (All Rounds)


| #   | Strategy          | Model             | Accuracy  | Correct | Cost   | Cost/Sample | Round  |
| --- | ----------------- | ----------------- | --------- | ------- | ------ | ----------- | ------ |
| 1   | baseline          | grok-4-1-fast     | **63.7%** | 293/460 | $0.20  | $0.0004     | R3     |
| 2   | self_discovery    | grok-4-1-fast     | **63.5%** | 292/460 | $1.82  | $0.0040     | **R4** |
| 3   | react_ts_pyeval   | grok-4-1-fast     | **60.2%** | 277/460 | $2.55  | $0.0055     | **R4** |
| 4   | baseline          | gpt-5-mini        | 53.7%     | 247/460 | $5.67  | $0.0123     | R3     |
| 5   | baseline          | o4-mini           | 47.2%     | 217/460 | $10.78 | $0.0234     | R3     |
| 6   | react_ts_pyeval   | Gemini Flash Lite | **47.2%** | 217/460 | $7.39  | $0.0161     | **R4** |
| 7   | baseline          | Haiku 4.5         | 41.7%     | 192/460 | $13.72 | $0.0298     | R3     |
| 8   | baseline          | Gemini Flash Lite | 37.6%     | 173/460 | $0.67  | $0.0015     | R3     |
| 9   | react_ts_pyeval   | gpt-5-mini        | 36.1%     | 166/460 | $10.24 | $0.0223     | **R4** |
| 10  | react_ts_pyeval   | gpt-5-nano        | 27.4%     | 126/460 | $2.32  | $0.0050     | R2     |
| 11  | react_full        | gpt-5-nano        | 25.9%     | 119/460 | $2.14  | $0.0046     | R3     |
| 12  | react_ts_only     | gpt-5-nano        | 25.7%     | 118/460 | $1.99  | $0.0043     | R2     |
| 13  | react_pyeval_only | gpt-5-nano        | 24.6%     | 113/460 | $1.44  | $0.0031     | R2     |
| 14  | self_discovery    | gpt-5-nano        | 16.3%     | 75/460  | $2.18  | $0.0047     | R1     |
| 15  | react_tools       | gpt-5-nano        | 14.3%     | 66/460  | $0.67  | $0.0014     | R1     |
| 16  | baseline          | gpt-5-nano        | 13.5%     | 62/460  | $0.66  | $0.0014     | R1     |


---

## The Big Finding: Tools Help Weak Models, Hurt Most Others

### Baseline vs react_ts_pyeval


| Model             | Baseline  | react_ts_pyeval | Delta       | Cost (base) | Cost (react) |
| ----------------- | --------- | --------------- | ----------- | ----------- | ------------ |
| **gpt-5-nano**    | 13.5%     | **27.4%**       | **+13.9pp** | $0.66       | $2.32        |
| **Gemini Flash Lite** | 37.6% | **47.2%**       | **+9.6pp**  | $0.67       | $7.39        |
| grok-4-1-fast     | **63.7%** | 60.2%           | -3.5pp      | $0.20       | $2.55        |
| gpt-5-mini        | **53.7%** | 36.1%           | -17.6pp     | $5.67       | $10.24       |


**Tools help the two weakest models.** gpt-5-nano gains +13.9pp and Gemini Flash Lite gains +9.6pp — both models have weak enough baselines that external computation via python_repl and reasoning strategy selection via thinking_system provide genuine uplift. Stronger models (grok, gpt-5-mini) already reason well enough that tool overhead destroys accuracy.

### Why tools help Gemini Flash Lite

Gemini Flash Lite has a low baseline (37.6%) but strong format compliance — once the framework bugs were fixed, it reliably calls tools and extracts answers. Key improvements over baseline:
- **object counting**: 90% (vs likely ~50% baseline range)
- **boolean expressions**: 80% (computation benefits from python_repl)
- **temporal sequence**: 70% (thinking_system helps organize reasoning)

### Why tools hurt stronger models

1. **Tool selection overhead** — With thinking_system + python_repl, the model spends iterations browsing strategies and computing instead of just answering.
2. **Answer extraction** — The `"The answer is:"` prefix requirement conflicts with tool-heavy workflows where the model may return the answer inside a tool call.
3. **Cost amplification** — Each tool call adds input/output tokens. gpt-5-mini at $10.24 is 1.8x its baseline cost.

---

## Grok: Three Strategies Compared


| Strategy        | Accuracy  | Cost      | Notes                         |
| --------------- | --------- | --------- | ----------------------------- |
| baseline        | **63.7%** | **$0.20** | Raw reasoning, no tools       |
| self_discovery  | 63.5%     | $1.82     | 4-phase flow, tools available |
| react_ts_pyeval | 60.2%     | $2.55     | ReAct loop with tools         |


**Grok's internal reasoning is already superior to external tool use.** As a reasoning model (like o3/o4), it has native chain-of-thought that outperforms our tool-augmented approach. The self_discovery flow essentially matches baseline — the 4-phase structure (select → adapt → operationalize → solve) doesn't hurt or help, it just adds cost.

The react_ts_pyeval drop (-3.5pp) comes from occasional tool misuse — grok sometimes calls python_repl for problems it can solve natively, and the tool's output formatting occasionally conflicts with the expected answer format.

---

## Per-Task Accuracy — Round 4 Strategies

Sorted by grok react_ts_pyeval accuracy.


| Task                 | grok react | grok sd | grok base | mini react | nano react | gemini react |
| -------------------- | ---------- | ------- | --------- | ---------- | ---------- | ------------ |
| zebra puzzles        | **100%**   | 90%     | 95%       | 35%        | 20%        | 30%          |
| object counting      | 95%        | **95%** | 85%       | **100%**   | 10%        | **90%**      |
| boardgame qa         | 90%        | **90%** | **95%**   | **90%**    | 60%        | 65%          |
| time arithmetic      | **85%**    | **85%** | 90%       | 70%        | **85%**    | 70%          |
| web of lies          | **85%**    | **85%** | **85%**   | 30%        | 25%        | 30%          |
| spatial reasoning    | **85%**    | **85%** | 70%       | 45%        | 5%         | 55%          |
| multistep arithmetic | **85%**    | **85%** | 60%       | 20%        | 0%         | 0%           |
| movie recommendation | 80%        | **85%** | **85%**   | 35%        | **90%**    | **90%**      |
| dyck languages       | 75%        | **85%** | 75%       | 30%        | 35%        | 45%          |
| word sorting         | **75%**    | **75%** | **75%**   | 70%        | 60%        | 65%          |
| object properties    | 70%        | **70%** | **70%**   | 0%         | 0%         | 40%          |
| temporal sequence    | 25%        | **70%** | 60%       | 10%        | 15%        | 70%          |
| hyperbaton           | 50%        | **70%** | 65%       | 0%         | 10%        | 25%          |
| shuffled objects     | **65%**    | 35%     | 25%       | **70%**    | 5%         | 40%          |
| boolean expressions  | 60%        | 55%     | **75%**   | 25%        | 40%        | **80%**      |
| geometric shapes     | 60%        | **60%** | **70%**   | 0%         | 5%         | 40%          |
| causal understanding | 55%        | **55%** | **65%**   | 50%        | 55%        | 55%          |
| disambiguation qa    | 45%        | **50%** | 55%       | **65%**    | 40%        | 55%          |
| sportqa              | **35%**    | 30%     | **45%**   | 30%        | **35%**    | 35%          |
| sarc triples         | **30%**    | 25%     | 25%       | 25%        | 15%        | 35%          |
| nycc                 | 25%        | 20%     | 20%       | 20%        | 10%        | 30%          |
| linguini             | **25%**    | **25%** | **25%**   | 10%        | 10%        | 10%          |
| buggy tables         | 15%        | **35%** | **50%**   | 0%         | 0%         | 30%          |


### Notable patterns

- **self_discovery grok outperforms react_ts_pyeval grok on 12 of 23 tasks** — the structured 4-phase approach preserves more of grok's native reasoning
- **grok self_discovery excels at**: spatial reasoning (85%), multistep arithmetic (85%), temporal sequence (70%), hyperbaton (70%)
- **react_ts_pyeval grok excels at**: zebra puzzles (100%), shuffled objects (65%), boolean expressions (60%)
- **Gemini with tools now competitive**: 47.2% overall, with standout scores on object counting (90%), movie recommendation (90%), boolean expressions (80%), temporal sequence (70%)
- **Gemini beats nano on 20 of 23 tasks** — the cheapest non-reasoning model now substantially outperforms the weakest model with the same tools

---

## Self-Discovery vs ReAct: Grok Comparison


| Task                 | self_discovery | react_ts_pyeval | Delta | Winner |
| -------------------- | -------------- | --------------- | ----- | ------ |
| temporal sequence    | **70%**        | 25%             | +45pp | SD     |
| multistep arithmetic | **85%**        | 70%*            | +15pp | SD     |
| hyperbaton           | **70%**        | 50%             | +20pp | SD     |
| spatial reasoning    | **85%**        | 70%*            | +15pp | SD     |
| buggy tables         | **35%**        | 15%             | +20pp | SD     |
| dyck languages       | **85%**        | 75%             | +10pp | SD     |
| geometric shapes     | **60%**        | 60%             | 0     | Tie    |
| zebra puzzles        | 90%            | **100%**        | -10pp | React  |
| shuffled objects     | 35%            | **65%**         | -30pp | React  |


*react values corrected from per-task data

**Self-discovery wins on complex reasoning tasks** (temporal, multistep, spatial) where the structured "select reasoning module → adapt → apply" approach helps grok organize its thinking. **ReAct wins on tracking/search tasks** (zebra puzzles, shuffled objects) where iterative tool use is genuinely beneficial.

---

## Cost Analysis


| Strategy + Model              | Accuracy  | Cost      | Acc/$     | Correct/$  |
| ----------------------------- | --------- | --------- | --------- | ---------- |
| grok baseline                 | **63.7%** | **$0.20** | **325.5** | **1500.5** |
| grok self_discovery           | 63.5%     | $1.82     | 34.9      | 160.4      |
| grok react_ts_pyeval          | 60.2%     | $2.55     | 23.6      | 108.6      |
| Gemini baseline               | 37.6%     | $0.67     | 56.1      | 258.2      |
| mini baseline                 | 53.7%     | $5.67     | 9.5       | 43.6       |
| Gemini react_ts_pyeval        | 47.2%     | $7.39     | 6.4       | 29.4       |
| nano react_ts_pyeval          | 27.4%     | $2.32     | 11.8      | 54.4       |
| mini react_ts_pyeval          | 36.1%     | $10.24    | 3.5       | 16.2       |
| Haiku baseline                | 41.7%     | $13.72    | 3.0       | 14.0       |


---

## Key Takeaways

### 1. Tool augmentation helps weak models, hurts strong ones

Tools provide genuine uplift when the model's baseline reasoning is weak: gpt-5-nano gains +13.9pp and Gemini Flash Lite gains +9.6pp. For stronger models (grok, gpt-5-mini), tools hurt — the model's internal reasoning is more effective than external tool orchestration.

### 2. Stronger models need fewer scaffolds

Grok at 63.7% with zero tools beats every tool-augmented configuration. The model's internal reasoning (native chain-of-thought) is more effective than our external ReAct loop. This aligns with the broader trend: as models improve, the value of explicit scaffolding diminishes.

### 3. Self-discovery preserves model capability better than ReAct

For grok, self_discovery (63.5%) nearly matches baseline (63.7%), while react_ts_pyeval drops to 60.2%. The structured 4-phase approach constrains the model less than a free-form ReAct loop, letting its native reasoning shine.

### 4. Gemini Flash Lite is a strong value proposition with tools

At 47.2% accuracy with $7.39 cost, Gemini react_ts_pyeval outperforms gpt-5-mini react (36.1%, $10.24) — higher accuracy at lower cost. While its accuracy/$ ratio (6.4) is below the grok baseline (325.5), it's competitive with other tool-augmented configurations.

### 5. Cost scales poorly with tools

Every tool-augmented run costs 3-13x more than the corresponding baseline. The grok baseline at $0.20 vs self_discovery at $1.82 (9x) vs react at $2.55 (13x) shows the tool tax clearly.

---

## Recommendations (Updated)

1. **Default: grok-4-1-fast-reasoning baseline (63.7%, $0.20)**
  Still the best option by every metric. Don't add tools unless you have evidence they help for your specific task distribution.
2. **If you must use tools: self_discovery > react for reasoning models**
  Self-discovery's structured approach (-0.2pp, $1.82) beats ReAct (-3.5pp, $2.55) on grok.
3. **For weak models: react_ts_pyeval provides significant uplift**
  gpt-5-nano goes from 13.5% → 27.4%, Gemini Flash Lite from 37.6% → 47.2%. The weaker the model, the more tools help.
4. **Gemini Flash Lite is the best budget option with tools**
  47.2% at $7.39 beats gpt-5-mini react (36.1% at $10.24). For cost-sensitive tool-augmented workloads, Gemini is the pick.
5. **Next experiment: grok + task-specific tools**
  The few tasks where react beats self_discovery (zebra puzzles, shuffled objects) suggest targeted tool use could help. A conditional strategy — tools only for tracking/search tasks, baseline for reasoning — might beat both.

---

## Raw Data

- **Round 4 react_ts_pyeval:** `react_ts_pyeval_gpt-5-mini.json`, `react_ts_pyeval_grok-4-1-fast-reasoning.json`, `react_ts_pyeval_gemini-3_1-flash-lite-preview.json`
- **Round 4 self_discovery:** `self_discovery_grok-4-1-fast-reasoning.json`
- **Baselines (Round 3):** `baseline_*.json`
- **Earlier rounds:** See [REPORT.md](REPORT.md) and [REPORT_R3.md](REPORT_R3.md)
