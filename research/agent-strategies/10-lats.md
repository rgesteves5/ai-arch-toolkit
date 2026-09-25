# Language Agent Tree Search (`lats`)

> Monte Carlo Tree Search over a tool-using ReAct agent: expand several actions, score the resulting states, back up rewards, reflect on failures, retry from the most promising node.
> Evidence: strong gains in the 2023 paper (GPT-3.5/GPT-4); an independent cost-controlled rerun erased them on HumanEval; later tree-search agents gain on sandboxed web and code benchmarks at 5–50× cost.
> Last reviewed: 2026-09-24 · Part of [agent-strategies](00-index.md)

## At a glance

- **Hard requirement:** an environment that can be reset to an earlier state, or tools with no side effects. Tree search re-executes actions; with irreversible tools (email, payments, writes, deploys) it is not viable [1][3][4][7][8].
- **What decides success:** the value signal. A ground-truth reward beat the best LLM value function by 6.5 points on web tasks [3]; LATS's HotpotQA setup told the agent whether its answer was correct [1].
- **Cost:** ~55× a simple retry baseline on HumanEval ($134.50 vs $2.45) with *lower* accuracy (88.0% vs 93.2%) [2]; 7–14× on web and SWE-bench tasks [5][6]; ~10× wall-clock [7].
- **Where it still pays:** sandboxed, resettable environments with strong terminal signals, medium-difficulty multi-step tasks, cheap models, and offline data generation [3][4][5][6].
- **2024–26:** industry practice favours parallel attempts plus a verifier over tree search [13][16]; search is moving into training [4][6][20].
- **In this toolkit:** a trajectory-level variant — each child is a whole ReAct attempt and a node takes up to `n_candidates` sibling attempts (branching fixed on 2026-09-24) — that re-runs its tools on every rollout. Route to it only with read-only, idempotent or sandboxed tools — see [In ai-arch-toolkit](#12-in-ai-arch-toolkit).

## 1. Origin

**Zhou, Yan, Shlapentokh-Rothman, Wang, Wang** (UIUC, Lapis Labs), "Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models", ICML 2024, PMLR 235:62138–62160; arXiv 2310.04406 (v1 October 2023). Code: github.com/lapisrocks/LanguageAgentTreeSearch [1].

## 2. How it works

A node is a text state: the input plus the actions and observations so far. Each iteration [1]:

1. **Selection** — descend from the root by UCT: V(s) + w·√(ln N(parent) / N(s)).
2. **Expansion** — sample n actions at the chosen node, execute each in the environment, record the observations as n children.
3. **Evaluation** — each child's value = λ·(LM score 1–10 of the trajectory, including observations) + (1−λ)·(self-consistency: how often the action recurs across samples).
4. **Simulation** — follow the highest-value child to a terminal state.
5. **Backpropagation** — push the environment's terminal reward up the path.
6. **Reflection** — after a failure, the LM writes a verbal lesson that is stored and injected into later prompts.

It stops on success or after k trajectories, and assumes the environment can be restored to any node (by replaying the text context or resetting a simulator). For code, each action is a whole program: simulation is skipped and the reward is the fraction of self-generated tests passed.

## 3. Settings and cost in the papers

- **Defaults** [1]: n=5 children, w=1; λ=0.5 (HotpotQA, Game of 24), λ=0.8 (code, WebShop).
- **Budget:** k=50 trajectories (HotpotQA), 30 (WebShop, Game of 24), 8 iterations (code); depth limits 7 / 8 / 15 / 5. Self-generated tests per coding problem: 4 (GPT-4), 6 (GPT-3.5).
- **Evaluation sets are small:** 100 HotpotQA questions, 50 WebShop instructions, 50 Game of 24 games, 397 MBPP, 164 HumanEval.
- **Paper's cost analysis** (HotpotQA, n=5, k=50) counts only successful searches: LATS 173,290 tokens vs ToT-ReAct 210,215 and RAP-ReAct 176,500; 66.65 nodes to success vs 84.05 and 70.60. Failed searches, which spend the whole budget, are excluded. With n=1, LATS ≈ ReAct with retries (0.48 EM at k=50 vs 0.42 for ReAct best-of-250) [1].
- **Independent cost measurement** [2]: HumanEval, 164 problems, 5 runs, gpt-4-turbo-2024-04-09, 8 iterations, expansion 3 — LATS cost **$134.50** vs **$2.45** for retry-with-rising-temperature (~55×). Some GPT-3.5 tasks took >2 hours; one was still running after 5 hours.
- **Projection** [10]: ~$16,000 for LATS across Thought of Search's four planning suites, vs ~$40 for IO/CoT.

## 4. Evidence

| Benchmark | Model | Baseline | LATS / tree search | Src |
|---|---|---|---|---|
| HotpotQA (100; environment says whether the answer is correct) | GPT-3.5 | ReAct 0.32; best-of-k 0.38; Reflexion 0.51; ToT-ReAct 0.39; RAP 0.54 | 0.63; n=10 0.65; CoT+ReAct 0.71 | [1] |
| HumanEval | GPT-3.5 / GPT-4 | Reflexion 68.1 / 91.0; GPT-4 base 80.1 | 83.8 / **92.7** | [1] |
| MBPP (397) | GPT-3.5 | Reflexion 70.0; RAP 71.4 | 81.1 | [1] |
| WebShop (50), score / success | GPT-3.5 | ReAct 53.8/28.0; Reflexion 64.2/35.0; expert 82.1/59.6 | 75.9/38.0 | [1] |
| Game of 24 (50) | GPT-3.5 | CoT 0.08; ToT 0.20; RAP 0.40 | 0.44 | [1] |
| **HumanEval, independent rerun** (5 runs) | gpt-4-turbo | zero-shot 89.6% ($1.93); retry 92.0% ($2.51); **warming 93.2% ($2.45)** | **LATS 88.0% ($134.50)** | [2] |
| VisualWebArena (910) / WebArena (812) | GPT-4o | 18.9% / 15.0% | best-first search in the real environment 26.4% / 19.2%; WebArena non-search AgentOccam 43.1% | [3] |
| VisualWebArena subset, by value function | GPT-4o | no search 24.5% | GPT-4o value 37.0%; **ground-truth reward 43.5%**; no self-consistency 28.5% | [3] |
| WebShop / OpenTable (live) | xLAM / Llama-3-70B | 28.6% / 18.6% | MCTS 48.4%; Agent Q+MCTS 50.5% / 95.4% | [4] |
| SWE-bench Lite (300) | GPT-4o / Qwen2.5-72B | 25.7 / 18.0 | 31.0 / 24.7 (+23% relative mean); GPT-4o cost **$40.86 → $576.00** | [5] |
| VWA Classifieds, success (token multiple) | GPT-4o | ReAct 28.6% (1×) | ToT-BFS 30.7% (3.2×); best-first 33.8% (4.2×); MCTS 37.6% (7.2×); R-MCTS 41.0% (7.4×) | [6] |
| OSWorld | GPT-4o | ReAct 11.2%; Agent-S (no search) 20.6% | R-MCTS 19.4% | [6] |
| VisualWebArena latency | GPT-4o | reactive 17.6%, ~68–88 s/task | tree search 26.2%, **~750–970 s/task** | [7] |
| GAIA validation (165) | GPT-4.1 | 55.8 | best-of-N 63.0; beam search 57.0; DVTS 55.8 | [13] |

## 5. Strengths

- Uses real environment feedback inside the search and combines it with reflection. Ablations on HotpotQA: MCTS → DFS −0.21; no LM value −0.26; no reflection −0.05 [1].
- Anytime: more iterations help, and n is a direct cost/accuracy knob (n=1 ≈ ReAct with retries) [1].
- Lifts weak models the most: Llama-3-70B +119.7% relative on VisualWebArena [3]; open models under SWE-Search approached GPT-4o [5].
- Search trees make good training data (DPO in Agent Q [4]; exploratory learning in ExACT [6]).

## 6. Weaknesses and costs

- **Money:** ~55× retry on HumanEval [2]; 14× (GPT-4o) and 5.3× (GPT-4o-mini) on SWE-bench Lite [5]; 7–10× ReAct's tokens on web tasks [6].
- **Time:** ~10× a reactive agent's wall-clock [7]; up to 20× the LM calls per step at an expansion budget of 20 [3]; multi-hour tails [2].
- **Calls:** roughly 2LT + bLT per task (T rollouts, L depth, b successors) [10].
- **Backtracking is fragile even in simulators:** resets replay the action sequence [3]; page element IDs change and server-side state (e.g. search history) cannot be reset [6].
- **The value function is the ceiling:** 37% (GPT-4o value) vs 43.5% (ground truth) [3]; SWE-Search's value agent picked the correct patch 73% of the time [5].
- **Engineering load:** policy, value and reflection prompts, sometimes per state type; reflections were often generic on WebShop [1][5].

## 7. Best use cases

- Sandboxed, resettable, side-effect-free environments with strong terminal signals: code with tests (repository state in a git-like commit tree [5]), simulators [1][3], stateless retrieval APIs [1].
- Multi-step tasks of medium difficulty (+75% relative on medium vs +24% easy, +47% hard) [3].
- Cheap or mid-tier models whose shortfalls search can make up for [3][5].
- Offline trajectory generation for fine-tuning [4][6].

## 8. Where it falls short

- **Irreversible actions and live systems.** Real websites are full of irreversible actions, so tree search could not even run on live benchmarks [7]; live MCTS involves risky interactions [4]; destructive actions must be screened out [3]; earlier tree-search agents assume every action is reversible [8].
- **Dependence on oracle feedback.** The HotpotQA setup reveals whether an answer is correct [1]; without external feedback, self-correction does not help reasoning [11].
- **Easy benchmarks and evaluation hygiene.** On HumanEval simple retries beat LATS at ~2% of the cost; the rerun also found LATS marking some incorrect tasks as correct and dropping one task [2]. User-filed issues report a success-count bug in the reference code (unverified; no maintainer reply) [17].
- **Search on a weak agent loses to a better agent without search:** 19.2% vs 43.1% on WebArena [3]; 19.4% vs 20.6% on OSWorld [6].
- **Small margins at matched compute:** SWE-Search's single answer vs best-of-5 baseline runs was 21.0 vs 22.0 and 17.7 vs 21.7 [5]; beam search trailed best-of-N on GAIA [13]; with LLM discriminators tree search ≈ re-ranking [12].

## 9. Variants and follow-ups

- **RAP** (EMNLP 2023) — MCTS with an LLM world model, no real environment [9].
- **ToolChain\*** — A* search over tool-call sequences [19].
- **Best-first search in the real environment** (Koh et al., TMLR 2025) — GPT-4o value averaged over 20 samples [3].
- **Agent Q** — MCTS plus an AI critic, then DPO on the trees [4].
- **SWE-Search** (ICLR 2025) — MCTS over repository states, numeric + verbal value, debating discriminator [5].
- **R-MCTS / ExACT** (ICLR 2025) — contrastive reflection, debate-based value; search later distilled into GPT-4o [6].
- **WebDreamer** — simulate each action's outcome with an LLM instead of backtracking for real; 4–5× more efficient than tree search [7].
- **WebOperator** — safety-aware best-first search with verified backtracking; 54.6% on WebArena with GPT-4o [8].
- **Tree-GRPO** (ICLR 2026) — tree search used in RL training, not at inference [20].
- **Reflexion** (NeurIPS 2023) — the tree-less ancestor that retries with verbal reflection [21] (see [06-reflexion](06-reflexion.md)).

## 10. With 2024–2026 models

- Cost-controlled evaluation reframed the headline: on HumanEval, simple baselines match or beat LATS [2].
- Remaining gains are on sandboxed benchmarks with non-reasoning models (GPT-4o, Llama, Qwen2.5) [3][5][6].
- One reasoning-model data point: on Classifieds, o1-mini as a plain ReAct agent scored 23.9% vs GPT-4o-mini's 20.9% at ~20× output tokens; R-MCTS on GPT-4o scored 32.1% at 9.6× tokens [6].
- **Industry uses parallel attempts plus a verifier.** Claude 3.7 Sonnet on SWE-bench Verified went 63.7% → 70.3% by sampling parallel attempts, discarding patches that break regression tests and ranking the rest with a scoring model [16]; best-of-N beat beam search on GAIA [13].
- **"Act more" can beat branching:** prompting a web agent to check again before stopping raised WebArena success from 23% to ≥28%, while longer thinking or best-of-n gave <3% [14].
- **Search is moving into training:** ExACT's distilled GPT-4o recovered ~87% of R-MCTS's performance at 3.6× instead of 9.6× tokens [6]; Agent Q [4]; Tree-GRPO [20].
- In math with process reward models, MCTS-like lookahead was the least compute-efficient test-time search [18]; for general agents, neither sequential nor parallel scaling was effective (limited by context length and verification) [15].
- **Gap:** no controlled study of LATS with native function calling or with 2025–26 reasoning agents.

## 11. Router signals

**Choose `lats` only if all hold:**
- **Hard gate:** every tool is read-only or idempotent, or runs in a resettable sandbox (container, git worktree, simulator).
- A strong external reward exists per trajectory (tests, validators, exact checkers) — wire it in as `evaluator_fn`.
- The task needs ≥3 tool actions and `react` / `reflexion` plateau on it.
- Minutes of latency and 5–50× cost are acceptable, or the work is offline data generation.

**Avoid it when:**
- any tool has side effects (email, payments, writes, deploys, account changes) or state lives on a server you cannot reset;
- only an LLM judge is available and correctness is subtle;
- a single ReAct run with retries already works (HumanEval-class tasks);
- the request is interactive;
- a strong reasoning model with native tools is available → prefer `react` or `reflexion`, and with more budget, parallel best-of-N plus a verifier.

## 12. In ai-arch-toolkit

Flow: [`_lats.py`](../../src/ai_arch_toolkit/toolkit/agents/flows/_lats.py), built by `_build_lats` in [`_builders.py`](../../src/ai_arch_toolkit/toolkit/agents/_builders.py). State as of 2026-09-24.

| Paper concept | Toolkit | Note |
|---|---|---|
| Node | text state: task + previous attempts appended as `Attempt: …` | trajectory-level, not action-level |
| Expansion | one whole inner ReAct episode per rollout (`max_react_iterations` = spec `max_iterations`, 10) → one child | a node takes up to `n_candidates` (5) sibling attempts before the search goes below it; paper: n *action* children per expansion |
| Selection | UCT with `exploration_weight` 1.41 | descends through nodes that already have `n_candidates` children and expands the most promising one that has fewer. Until 2026-09-24 it descended to a childless node, so the tree was a single chain and `n_candidates` was unused |
| Evaluation | `evaluator_fn(task, answer)` if given, else one LLM score 0–1 (0.5 when unparseable) | no self-consistency term |
| Reflection | stored on the child when score < 0.5 | used when that node is selected next |
| Stop | score ≥ 0.9 → solver call; else best answer after `max_rollouts` (default = spec `max_iterations`, 10) | worst case ≈ 10 × (10 ReAct calls + evaluation + reflection) + solver |
| Environment | tools are re-run from scratch every rollout | **side effects repeat**; no reset or sandbox |
| Phases | `rollout_llm`, `rollout_tools`, `evaluator_llm`, `solver_llm`, `reflector_llm`, `evaluator_fn`; knobs `evaluator_system`, `reflector_system`, `n_candidates`, `max_rollouts`, `exploration_weight` | — |

**Net effect:** since the 2026-09-24 fix it builds a real tree — with the defaults, five independent attempts at the task, then five more that UCT spends refining the most promising of them. It stays trajectory-level (no action-level expansion, no self-consistency term in the value), so it sits between the paper's LATS and best-of-N with refinement; the paper's numbers do not transfer directly. The first-number score-parsing bug it shares with `tot` ("Score (0.0-1.0): 0.8" → 0.0) is being fixed separately.

**Router implication:** gate `lats` to side-effect-free tool sets; supply `evaluator_fn` whenever a programmatic check exists.

## 13. Open questions

- Does LATS add value over 2025–26 reasoning agents with native tool calling, at equal cost?
- Can action-safety classifiers or world models make search safe in live systems [7][8]?
- How can value functions be calibrated without ground truth [3][5]?
- Where is the crossover between spending search at inference and distilling it into training [6]?

## Sources

1. Zhou et al., *Language Agent Tree Search*, ICML 2024 — https://proceedings.mlr.press/v235/zhou24r.html · https://arxiv.org/abs/2310.04406 · https://github.com/lapisrocks/LanguageAgentTreeSearch
2. Kapoor et al., *AI Agents That Matter*, TMLR 2025 — https://arxiv.org/abs/2407.01502
3. Koh et al., *Tree Search for Language Model Agents*, TMLR 2025 — https://arxiv.org/abs/2407.01476 · https://openreview.net/forum?id=QF0N3x2XVm
4. Putta et al., *Agent Q*, 2024 — https://arxiv.org/abs/2408.07199
5. Antoniades et al., *SWE-Search*, ICLR 2025 — https://arxiv.org/abs/2410.20285
6. Yu et al., *ExACT / R-MCTS*, ICLR 2025 — https://arxiv.org/abs/2410.02052
7. Gu et al., *WebDreamer* — https://arxiv.org/abs/2411.06559
8. Dihan et al., *WebOperator*, 2025 — https://arxiv.org/abs/2512.12692
9. Hao et al., *RAP*, EMNLP 2023 — https://aclanthology.org/2023.emnlp-main.507/
10. Katz et al., *Thought of Search*, NeurIPS 2024 — https://arxiv.org/abs/2404.11833
11. Huang et al., *Large Language Models Cannot Self-Correct Reasoning Yet*, ICLR 2024 — https://arxiv.org/abs/2310.01798
12. Chen et al., *When is Tree Search Useful for LLM Planning?*, ACL 2024 — https://aclanthology.org/2024.acl-long.738/
13. OPPO Agent Team, 2025 — https://arxiv.org/abs/2506.12928
14. Shen et al., *Thinking vs. Doing*, 2025 — https://arxiv.org/abs/2506.07976
15. Li et al., *General AgentBench*, 2026 — https://arxiv.org/abs/2602.18998
16. Anthropic, *Claude 3.7 Sonnet* announcement — https://www.anthropic.com/news/claude-3-7-sonnet
17. LATS GitHub issues — https://github.com/lapisrocks/LanguageAgentTreeSearch/issues/30 · https://github.com/lapisrocks/LanguageAgentTreeSearch/issues/28
18. Snell et al., *Scaling LLM Test-Time Compute Optimally*, ICLR 2025 — https://arxiv.org/abs/2408.03314
19. Zhuang et al., *ToolChain\** — https://arxiv.org/abs/2310.13227
20. Ji et al., *Tree-GRPO*, ICLR 2026 — https://arxiv.org/abs/2509.21240
21. Shinn et al., *Reflexion*, NeurIPS 2023 — https://arxiv.org/abs/2303.11366
