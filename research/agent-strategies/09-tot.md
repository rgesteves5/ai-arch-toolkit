# Tree of Thoughts (`tot`)

> Search over reasoning steps: propose several candidate "thoughts", score them, expand the best, backtrack from dead ends.
> Evidence: strong for 2023 GPT-4 puzzles (Game of 24, crosswords); weak or contested beyond them; thin for 2025–26 reasoning models.
> Last reviewed: 2026-09-24 · Part of [agent-strategies](00-index.md)

## At a glance

- **What it buys:** large gains when an early wrong step is fatal and partial states can be checked — Game of 24 went from 4% (CoT) to 74% on GPT-4 [1].
- **What decides success:** the evaluator, not the search algorithm. Tree search beats simple re-ranking only with a discriminator at least ~90% accurate; with LLM discriminators it was 10–20× slower for gains that were not statistically significant [7].
- **Cost:** 5–100× the tokens of CoT; ~109 LLM calls per Game-of-24 problem; ~57× CoT latency on 7B–13B models [1][4][15].
- **2024–26:** reasoning models internalise much of what ToT added in prompts; external step-level search now helps mostly small or weak models [11][13][14].
- **Better alternative for formal problems:** have the model write the search (successor + goal test) and run classical BFS/DFS — 100% on four suites with a handful of calls [8].
- **In this toolkit:** the current flow does not implement the paper's search correctly (DFS expands the worst candidate first; BFS returns no answer with defaults). Do not route to it until fixed — see [In ai-arch-toolkit](#12-in-ai-arch-toolkit).

## 1. Origin

- **Yao, Yu, Zhao, Shafran, Griffiths, Cao, Narasimhan**, "Tree of Thoughts: Deliberate Problem Solving with Large Language Models", NeurIPS 2023, arXiv 2305.10601. Experiments on GPT-4 (chat completion, temperature 0.7), May 2023 [1].
- **Long (Theta Labs)**, "Large Language Model Guided Tree-of-Thought", arXiv 2305.08291 — a concurrent, non-peer-reviewed preprint with a rule-based checker and a backtracking controller [2].

## 2. How it works

A ToT instance fixes four choices [1]:

1. **Decomposition** — what one thought is: an equation line, a paragraph plan, a crossword word.
2. **Generation** of k candidates per state: *sample* independently (rich spaces such as paragraphs) or *propose* several distinct next steps in one prompt (constrained spaces; avoids duplicates).
3. **Evaluation** — *value* each state on its own (1–10, or sure/likely/impossible, sometimes with a short lookahead), or *vote* across states (a step-level self-consistency).
4. **Search** — BFS keeps the b best states per level; DFS expands the most promising child first, prunes states below a value threshold, and backtracks.

IO prompting, CoT, CoT with self-consistency and self-refine are degenerate trees. Long's variant adds a rule-based checker, a memory of the tree and a controller that orders backtracking [2].

## 3. Settings and cost in the papers

| Task | Setting [1] |
|---|---|
| Game of 24 | 100 hard games (4nums.com indices 901–1000); 3 thought steps; propose prompt (1-shot, 3-shot for GPT-3.5); BFS b=5; each candidate valued sure/maybe/impossible, 3 samples |
| Creative writing | depth 2; sample k=5 plans, 5 zero-shot votes, keep b=1; then 5 passages and another vote |
| Mini crosswords | 20 games; DFS with 5 proposal samples per state; prune when any remaining clue is judged impossible; ≤100 DFS steps, 10 thought steps |
| Long 2023 [2] | gpt-3.5-turbo at T=1; backtrack on checker rejection or after >5 children; ≤100 rounds |

**Reported cost** [1]: Game of 24 — ToT 5.5k completion tokens and $0.74 per problem, against 6.7k and $0.47 for best-of-100 CoT and $0.13 for best-of-100 IO. Creative writing — $0.32 vs $0.07 for CoT (~5×). The authors estimate 5–100× more generated tokens than CoT; the main experiments cost ~$106. The paper's 1.4k prompt-token figure for Game of 24 is inconsistent with its other rows; an independent count gives 13.9k prompt tokens and 109.1 LLM queries per game [4].

## 4. Evidence

| Benchmark | Model | Baseline | Result | Src |
|---|---|---|---|---|
| Game of 24 (100 hard) | GPT-4, 2023 | IO 7.3%; CoT 4.0%; CoT-SC (k=100) 9.0%; IO+refine 27%; oracle best-of-100 CoT 49% | ToT b=1 45%; **b=5 74%** | [1] |
| Game of 24 | GPT-3.5-turbo | IO 6%; CoT 3% | ToT 19%; GPT-4 gen + GPT-3.5 eval 64%; GPT-3.5 gen + GPT-4 eval 31% | [1] |
| Game of 24, independent re-run | GPT-4 | — | ToT (b=5) **69%** at 109.1 queries/game; AoT 71% with 1 query | [4] |
| Game of 24 (50 games) | GPT-3.5 | CoT 0.08 | ToT 0.20; RAP 0.40; LATS 0.44 | [21] |
| Creative writing (100 prompts) | GPT-4, coherence judged by GPT-4 (1–10) | IO 6.19; CoT 6.93; IO+refine 7.67 | ToT 7.56; ToT+refine 7.91 | [1] |
| Mini crosswords (20) | GPT-4 | CoT 15.6% words, 1 game | ToT 60% words, 4/20 games; no pruning 41.5%; no backtracking 20% | [1] |
| Mini crosswords, independent re-run | GPT-4 | — | ToT **46.5%** words at >200 queries; AoT 52% with 2 queries | [4] |
| GSM8K / StrategyQA (zero-shot ToT) | GPT-4 | CoT 86 / 82 | ToT 90 / 83 | [1] |
| GSM8K (500), Spider, Bird | CodeLlama-13B + GPT-3.5 discriminator | re-ranking 47.6 (GSM8K) | tree search 51.0, not significant; ≥10–20× slower; Spider 67.5 vs 70.3 re-ranking | [7] |
| 7 QA / fact-check / arithmetic sets | LLaMA2-7B | CoT 36.0%, 37 s/instance | ToT 39.1%, **1,749 s/instance** (~57.5× CoT time across 3 models) | [15] |
| MATH, equal compute | PaLM 2-S* + process reward model | best-of-N (weighted) | beam search wins at low budgets, loses at high budgets; lookahead worst | [9] |
| 24 Game (1,362), Blocksworld (502), PrOntoQA (4,000), crosswords (20) | GPT-4 writes successor + goal-test code | ToT projected at ~102K calls for the 24 Game suite | **100% on all four** with ~2–4 calls per domain | [8] |
| PlanBench Blocksworld (600) | o1-preview, zero-shot | best non-reasoning LLM 62.6% | 97.8%; 23.6% on 6–20-block problems; Fast Downward 100% in 0.27 s | [14] |

## 5. Strengths

- Large gains when early choices decide the outcome and partial states can be checked: ~60% of CoT samples on Game of 24 were already dead after step one [1].
- Beats oracle best-of-100 CoT at a similar number of generated tokens (74% vs 49%) [1].
- Modular: generator, evaluator, search and decomposition can be swapped; a strong generator with a cheaper evaluator reached 64% [1].
- No training; readable intermediate states; knobs (b, k, votes, pruning threshold, early stop) trade cost for accuracy [1].
- Accepts exact checkers — a hybrid system when rules can be coded [2].
- Search trees can be distilled: with CPO, plain CoT matched ToT without its latency [15].

## 6. Weaknesses and costs

- **Calls and tokens:** 5–100× CoT tokens [1]; ~109 calls per Game-of-24 problem, >200 per crossword [4]. Upper bound (before caching): ~T·b generator calls + T·b·k·s evaluator calls (depth T, beam b, candidates k, value samples s).
- **Latency:** sequential generate → evaluate → select; ~57.5× CoT [15]; ≥10–20× re-ranking [7].
- **Per-task engineering:** each task needed its own decomposition and prompts [1].
- **Fragile self-evaluation:** GPT-4's performance collapsed when it critiqued its own work on Game of 24, graph colouring and planning; a sound external verifier gave large gains [16].
- **Unsound and incomplete:** ToT visits a tiny slice of the state space (~1.6% for 24 Game) [8]; the final-answer selection matters (crosswords 4/20 with the paper's heuristic vs 7/20 with oracle selection) [1].

## 7. Best use cases

- Few discrete steps (3–10) with a small combinatorial choice per step, where an early mistake is fatal: arithmetic target puzzles, constraint satisfaction, crosswords, short planning [1][6].
- Partial states can be ruled out cheaply, ideally by code (bounds, rule checks) [1][2].
- The generator sometimes proposes the right step — single-pass success is low but not zero [1].
- The evaluator is stronger than the generator, or an exact checker exists [1][7].
- Offline or batch work where latency does not matter, including generating training data [15].

## 8. Where it falls short

- **Evaluator errors dominate:** GPT-4 pruned correct crossword states it did not recognise [1]; ≥50% of failures came from discrimination errors [7]; plausible wrong chains get high confidence [5]; at large budgets search over-optimises the verifier and loses to best-of-N on easy questions [9].
- **Generation-limited tasks:** GPT-3.5 generator + GPT-4 evaluator reached only 31% [1].
- **Knowledge-limited or already-easy tasks:** +1 on StrategyQA and +4 on GSM8K at many times the cost [1].
- **Replication drift:** 69% vs 74% (Game of 24) and 46.5% vs 60% (crossword words) in an independent GPT-4 re-run [4].
- **Framework matters less than coverage** at equal budgets [12]; MCTS-style variants need dozens of times more tokens [10].
- **Weak for agents:** on web tasks, ToT-BFS gained 28.6% → 30.7% at 3.2× tokens, far below MCTS variants that use environment feedback [22].

## 9. Variants and follow-ups

- **Graph of Thoughts** (AAAI 2024) — arbitrary graphs with aggregation and refinement; sorting error −62% and cost −31% vs ToT [3].
- **Algorithm of Thoughts** (ICML 2024) — puts DFS-style traces in the prompt so one call explores internally; matches ToT with ~1/100 of the queries [4].
- **Self-evaluation guided beam search** (NeurIPS 2023) — step-level beam search scored by LM confidence; needs logits [5].
- **RAP** (EMNLP 2023) — MCTS with the LLM as policy and world model; Blocksworld 1.00/0.88/0.42 on 2/4/6-step plans with LLaMA-33B [6].
- **Cumulative Reasoning** (TMLR) — proposer/verifier/reporter over verified propositions; 98% on Game of 24 with GPT-4 [17].
- **Thought of Search** (NeurIPS 2024) — the LLM writes successor and goal-test code; classical search runs it, sound and complete [8].
- **REBASE** (ICLR 2025) and **compute-optimal test-time scaling** (ICLR 2025) — pick the search per difficulty; >4× more efficient than best-of-N alone [9][10].
- **CPO** (NeurIPS 2024) — distil ToT preferences into CoT [15]; **Stream of Search** — train on linearised search traces [24].
- **Gambit** (COLM 2026) — thought-level beam search for reasoning models scored by hidden-state probes [23].

## 10. With 2024–2026 models

- **Reasoning models internalise ToT's moves.** DeepSeek-R1-Zero learned to re-evaluate and explore alternatives under RL [13]; o1-preview solves 97.8% of Blocksworld zero-shot [14]. DeepSeek abandoned MCTS and process reward models for R1 because the token-level search space explodes and value models are hard to train [13].
- **External search helps weak models more.** Across Qwen2.5 0.5B–72B, step-level search beats best-of-N for small models while best-of-N wins at every difficulty for 72B [11]; beam search wins only at low budgets or medium difficulty [9].
- **Selection over complete answers still pays** on hard math for small reasoning models (R1-Distill-Qwen-7B: AIME24 63.3 → 83.3 with PRM-guided scaling) [11].
- **Modern search is serving machinery, not prompts:** thought-level beam search on open reasoning models used 49% fewer tokens than 256-way self-consistency at higher accuracy, but needs hidden-state probes and KV-cache sharing that hosted APIs do not expose [23].
- **Inference-time scaling pays less as complexity grows;** more tokens do not reliably buy accuracy, while perfect verifiers give large gains [18].
- **Long combinatorial problems still break reasoning models** [19], though the strongest version of that claim is contested (output limits, unsolvable instances) [20]; a classical planner solves Blocksworld in 0.27 s against o1-preview's 40 s [14]. Both point toward "write the search" rather than prompt-level ToT.
- **Gap:** no controlled study runs 2023-style ToT on a frontier reasoning model and compares it with the model alone at equal cost.

## 11. Router signals

**Choose `tot` when most of these hold:**
- the model is not a reasoning model, or is small;
- the task is a combinatorial puzzle, constraint problem or short-horizon plan (≤ ~10 steps);
- a programmatic checker exists, or the evaluator is clearly stronger than the generator;
- single-pass CoT succeeds less than ~50% of the time, but not never;
- minutes of latency and 10–100× cost are acceptable.

**Avoid it when:**
- a reasoning model can handle the task → `completion` with thinking, or best-of-N with a verifier;
- the problem can be written as code → `react` with a code tool, and let the model write the search [8];
- the task is limited by knowledge or retrieval → `react`;
- the task is open-ended writing → `generate_review` (iterative refine matched ToT: 7.67 vs 7.56) [1];
- there is no reliable evaluator (tree search ≈ re-ranking at 10–20× latency) [7];
- the request is interactive, or CoT already does well.

## 12. In ai-arch-toolkit

Flow: [`_tot.py`](../../src/ai_arch_toolkit/toolkit/agents/flows/_tot.py), built by `_build_tot` in [`_builders.py`](../../src/ai_arch_toolkit/toolkit/agents/_builders.py). State as of 2026-09-24.

| Paper concept | Toolkit | Note |
|---|---|---|
| Generation | one *propose* call returning `n_candidates` numbered thoughts (default 3) | no *sample* mode |
| Evaluation | one *value* call per candidate, 0–1 score, single sample | no vote mode, no multi-sample value; LLM only — no `evaluator_fn` for an exact checker (LATS has one) |
| Search | `search_strategy` `"dfs"` (default) or `"bfs"`; `max_depth` 3; passes capped by the spec's `max_iterations` (10) | no separate beam width, no pruning threshold, no backtracking — the first node at `max_depth` ends the search |
| Early stop | a candidate scoring ≥ 0.9 goes straight to the solver | — |
| Phases | `generator_llm`, `evaluator_llm`, `solver_llm`; knob `evaluator_system` | `tools` is accepted but unused |

**Defects confirmed by simulation (fix tracked separately):**
- DFS expands the **worst** candidate first: children are pushed best→worst and `frontier.pop()` takes the last. With scores 0.8/0.5/0.2 the default DFS answers from the 0.2 → 0.2 → 0.2 path.
- BFS keeps no beam; with defaults it makes 40 LLM calls and ends **without an answer** (1+3+9 pops are needed before the first depth-3 node).
- The score regex takes the first number in the reply: "Score (0.0-1.0): 0.8" parses as 0.0.

**Recommendations beyond the bug fixes:** add an `evaluator_fn` (programmatic checker), a beam width for BFS and a pruning threshold for DFS, so the flow can reproduce the paper's settings; always solve from the best state when the budget runs out.

## 13. Open questions

- Does prompt-level ToT add anything over frontier reasoning models at equal cost, on tasks beyond their competence?
- Can reasoning-model judges reach the ~90% discrimination accuracy that tree search needs on open-ended tasks [7]?
- For reasoning models, is a "thought" a step or a whole solution [23]?
- Can Thought of Search work without human fixes to generated code, and beyond formal puzzles [8]?

## Sources

1. Yao et al., *Tree of Thoughts*, NeurIPS 2023 — https://arxiv.org/abs/2305.10601 · https://proceedings.neurips.cc/paper_files/paper/2023/file/271db9922b8d1f4dd7aaef84ed5ac703-Paper-Conference.pdf · code: https://github.com/princeton-nlp/tree-of-thought-llm
2. Long, *Large Language Model Guided Tree-of-Thought*, 2023 — https://arxiv.org/abs/2305.08291
3. Besta et al., *Graph of Thoughts*, AAAI 2024 — https://ojs.aaai.org/index.php/AAAI/article/view/29720 · https://arxiv.org/abs/2308.09687
4. Sel et al., *Algorithm of Thoughts*, ICML 2024 — https://proceedings.mlr.press/v235/sel24a.html · https://arxiv.org/abs/2308.10379
5. Xie et al., *Self-Evaluation Guided Beam Search*, NeurIPS 2023 — https://arxiv.org/abs/2305.00633
6. Hao et al., *Reasoning with Language Model is Planning with World Model* (RAP), EMNLP 2023 — https://aclanthology.org/2023.emnlp-main.507/
7. Chen et al., *When is Tree Search Useful for LLM Planning? It Depends on the Discriminator*, ACL 2024 — https://aclanthology.org/2024.acl-long.738/
8. Katz et al., *Thought of Search*, NeurIPS 2024 — https://arxiv.org/abs/2404.11833
9. Snell et al., *Scaling LLM Test-Time Compute Optimally*, ICLR 2025 — https://arxiv.org/abs/2408.03314
10. Wu et al., *Inference Scaling Laws* (REBASE), ICLR 2025 — https://arxiv.org/abs/2408.00724
11. Liu et al., *Can 1B LLM Surpass 405B LLM? Compute-Optimal Test-Time Scaling*, 2025 — https://arxiv.org/abs/2502.06703
12. Gan et al., ICML 2025 — https://proceedings.mlr.press/v267/gan25a.html
13. DeepSeek-AI, *DeepSeek-R1*, 2025 — https://arxiv.org/abs/2501.12948
14. Valmeekam et al., *LLMs Still Can't Plan; Can LRMs?* (o1 on PlanBench), 2024 — https://arxiv.org/abs/2409.13373
15. Zhang et al., *Chain of Preference Optimization* (CPO), NeurIPS 2024 — https://arxiv.org/abs/2406.09136
16. Stechly et al., *On the Self-Verification Limitations of LLMs*, 2024 — https://arxiv.org/abs/2402.08115
17. Zhang et al., *Cumulative Reasoning*, TMLR — https://arxiv.org/abs/2308.04371
18. Balachandran et al., *Inference-Time Scaling for Complex Tasks*, 2025 — https://arxiv.org/abs/2504.00294
19. Shojaee et al., *The Illusion of Thinking*, NeurIPS 2025 — https://arxiv.org/abs/2506.06941
20. Lawsen, *Comment on The Illusion of Thinking*, 2025 — https://arxiv.org/abs/2506.09250
21. Zhou et al., *Language Agent Tree Search*, ICML 2024 — https://proceedings.mlr.press/v235/zhou24r.html
22. Yu et al., *ExACT / R-MCTS*, ICLR 2025 — https://arxiv.org/abs/2410.02052
23. Yang et al., *Gambit*, COLM 2026 — https://arxiv.org/abs/2608.08020
24. Gandhi et al., *Stream of Search*, 2024 — https://arxiv.org/abs/2404.03683
