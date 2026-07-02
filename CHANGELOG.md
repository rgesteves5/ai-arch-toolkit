# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Cost metering & budgets.** A neutral metering mechanism in `core` (`MeterScope` / `RunConfig`, opaque exact `Money`, `Cost`, `MeterSnapshot`, `UsageEvent` / `UsageSink`, the `AdmissionController` protocol) with an opinion layer on top in `toolkit.budget` (`BudgetPolicy`, `BudgetController`, `BudgetReport`, `budget_scope()`, `HeuristicEstimator`). Attach `budget_policy=BudgetPolicy(max_cost=…, max_llm_calls=…, max_total_tokens=…, max_wall_s=…)` to a `Flow` or `Agent`, at construction or per run (`flow.run(state, budget_policy=…)` / `agent.run(task, budget_policy=…)`); all nine agent flows and `Agent` honour it, and nested flows share one cumulative budget. Caps are enforced **hard at the charge site** — the LLM/tool operation that would exceed a cap is denied before it runs (`BudgetExceeded`, surfaced as `policy_decision="budget_exceeded"`), exact even under concurrency. Two knobs shape a cost cap: `reserve="strict"` (reserve a worst-case hold up front) and `unpriced="fail_closed"` (default — deny once a call can't be priced, e.g. an unpriced model or a server tool). Read spend off the run's meter: `result.meter` (a `BudgetReport`), `result.total_cost`, `result.usage`, or `agent_result.report`. All metering types are re-exported at the top level; see [Tool Governance & Safety → Cumulative budgets](docs/safety.md).
- **Anthropic schema-in-prompt structured output.** `structured_output_mode="prompt"` (an `LLM(...)` constructor default or a per-call kwarg) makes the Anthropic adapter inject the JSON schema into the system prompt and parse the reply, instead of the native `output_config`. This handles large analysis/planning-style schemas that exceed Anthropic's native structured-output complexity limit (which otherwise returns a 400 "schema is too complex"). Defaults to `"native"` (unchanged behaviour); the OpenAI, Gemini, and xAI adapters accept and ignore the kwarg.
- **Local OpenAI-compatible servers.** `LLM("gemma4:e4b", base_url="http://localhost:11434/v1")` routes arbitrary model tags to the OpenAI adapter for Ollama, LM Studio, and vLLM. A `provider=` kwarg on `LLM` / `create_provider()` forces a specific adapter regardless of the model name; an unknown model with `base_url=` set falls back to the OpenAI-compatible adapter automatically.
- **Optional API key on localhost.** When `base_url` points at a loopback host (`localhost` / `127.x` / `::1`), the API key is optional (a placeholder is used) and a cloud key from the environment is **not** forwarded to the local server. Remote endpoints (gateways, proxies) still require a key and fail fast when it is missing.
- **Real-time reasoning streaming.** Vendor reasoning deltas (`reasoning_content` / `reasoning`) from OpenAI-compatible servers surface as incremental `thinking` events in `stream_events()` and populate `Response.thinking` in `complete()`, streaming, and batch responses. The reasoning-so-far is preserved even when a stream is abandoned early or errors mid-way.
- **`StreamEvent.partial`** flag distinguishes incremental reasoning fragments (`True`, OpenAI-compatible servers) from complete thinking blocks (`False`, Anthropic). Concatenate consecutive partial `thinking` events for the full trace.
- nanope research_center, agent_swarm scaffold, and advanced configurable agent (work in progress, not part of the public toolkit API).
- `AGENTS.md` project guidance file for Codex.
- `app` optional-dependency extra (`reflex>=0.7` + graph + yaml).
- `pyright>=1.1.390` and `pre-commit>=4.0` in the `dev` extra; `[tool.pyright]` configured in standard mode over `src/`.
- `.pre-commit-config.yaml` with ruff-check / ruff-format and standard hygiene hooks (trailing whitespace, EOF newline, yaml/toml validity, merge-conflict + large-file guards).
- `CONTRIBUTING.md` covering setup, conventions, and how to add a provider / tool / agent flow.

### Changed
- **The meter is the single source of truth for spend.** Agent flows no longer thread `cost` / `usage` into their step results by hand — `FlowResult.total_cost` / `.usage` / `.meter` and `AgentResult.cost` / `.usage` / `.report` all derive from the run's meter, so nested, parallel, and streaming spend can no longer drift from a parallel hand-rolled tally. A per-step `Policy.max_cost` is now enforced against that step's **metered span cost** (previously the step's manually declared `Result.cost`, which flows no longer set). `Trace.total_cost` / `total_usage` remain a raw view of any cost a custom step annotates via `Result(cost=…)`.
- **String fallback routing.** A string `fallback=` is now routed by its own model name: a recognizable model (e.g. `claude-…`) fails over to its own provider and connection, so a local-primary → cloud-fallback chain works; a bare local tag inherits the primary's `base_url` / `provider`, assuming it lives on the same server. Pass `LLM` instances as fallbacks for full per-fallback control.
- README rewritten with badges, copy-paste snippets (completion, streaming, tools, ReAct), a provider × feature matrix, and an agent-architecture table.
- CI split into `lint`, `typecheck` (non-blocking), and `test` (ubuntu + macos with coverage) jobs.
- `[tool.ruff]` and `[tool.pyright]` both exclude `src/ai_arch_toolkit/nanope` (sub-projects have their own idioms).
- Pricing registry refreshed (2026-05-19): Claude 4.7 added; Claude 4.6/4.7 now ship with 1M context at standard rates (long-context tier removed).
- Examples 31–33 updated to the variadic `Flow(*steps)` API and async `flow.run(state)`.
- Docs and example index aligned with the Flow-based architecture; legacy "pipelines" and "8 agent architectures" wording removed.
- Sync timeouts in `core/_sync.py` no longer expose the dead `SYNC_TIMEOUT` / `STREAM_JOIN_TIMEOUT` aliases; use `configure_sync_timeouts()` instead.
- `RateLimitMiddleware` docstring documents the streaming-bypass limitation explicitly (previously a TODO).
- `uv lock --upgrade` brought every transitive dependency to its latest compatible version (pydantic 2.13, urllib3 2.7, requests 2.34, websockets 16, xai-sdk 1.12, ruff 0.15.13, …); resolved the four Dependabot alerts.

### Fixed
- **Anthropic structured output now validates into the Pydantic model.** When `output_schema` carries a `model_class`, the Anthropic adapter coerces the parsed JSON via `model_validate()` — at parity with the OpenAI, Gemini, and xAI adapters (it previously returned a raw `dict`). It also tolerates Markdown-fenced JSON in the reply.
- OpenAI-compatible streaming now flushes accumulated tool calls when a server ends the turn with `finish_reason="stop"` instead of `"tool_calls"` (some Ollama / LM Studio / vLLM builds), so tool-using agents no longer silently see zero tool calls.
- Structured output: parsed JSON is now validated against the Pydantic model before being returned.
- Lint fixes: ternary form in `toolkit/tools/_datetime.py`; unused `pytest` import in `tests/test_python_eval.py`; misc `ruff format` across toolkit tools.

### Removed
- **The legacy `core._budget` module** (`BudgetState`, its cooperative `BudgetPolicy`, `BudgetExceeded.to_dict()`). Budgets now live in `toolkit.budget` and enforce hard at the charge site rather than by cooperative counter-checking as steps record usage. `BudgetPolicy.max_wall_time` is now `max_wall_s`; the `strict_cost` / `allow_unpriced` flags are now the `reserve` / `unpriced` knobs.

## Historical log

Chronological worklog of features and major changes prior to adopting Keep a Changelog.

| Date | Change |
|------------|--------|
| 2026-03-12 | Add nanope BBEH benchmark suite, expand toolkit tools (python eval, dictionary), and add generate-review flow |
| 2026-03-08 | Add content moderation system with Moderator protocol, OpenAI and LLM implementations |
| 2026-03-08 | Replace legacy agents and pipeline with Flow-based architecture |
| 2026-03-05 | Update pricing registry with latest model prices from all providers |
| 2026-03-05 | Add local token counting and extend pricing registry |
| 2026-03-05 | Extract general-purpose graph layer from memory system with new methods |
| 2026-03-04 | Add production readiness: timeouts, validation, rate limiting, observability |
| 2026-03-03 | Add pipeline system, knowledge registry, and LLM fallback chains with attempt tracking |
| 2026-03-02 | Add graph-backed memory system with views, middleware, and agent tools |
| 2026-02-28 | Add SelfDiscovery and LLMCompiler agents, per-phase customization (PhaseConfig) |
| 2026-02-28 | Add PlanExecute, ToT, and LATS agents, rich streaming events, stream fallback |
| 2026-02-28 | Add Reflexion and ReWOO agents, delete legacy layer |
| 2026-02-27 | Add agent architecture implementation with ReAct and multimodal capabilities |
| 2026-02-25 | Rewrite core/ layer, add Gemini and xAI providers, restructure project |
| 2026-02-24 | New standard paradigm, research knowledge, board system, file structure reorganization |
| 2026-02-23 | Add tools layer — schema inference, @tool decorator, execution, and ToolGroup |
| 2026-02-23 | Add client reuse, stream metadata, and OpenAI provider |
| 2026-02-22 | Rewrite LLM layer from first principles |
| 2026-02-21 | Add MkDocs documentation, build commands, API docs, and CI configuration |
| 2026-02-09 | Migrate from Poetry to uv, initial project structure with examples |
| 2026-02-08 | Add LLMs documentation: agents architecture and API guide |
| 2026-02-07 | Initial commit |
