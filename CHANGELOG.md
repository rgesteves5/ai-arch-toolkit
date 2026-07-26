# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **[docs/configuring-agents.md](docs/configuring-agents.md)** — the end-to-end agent
  configuration guide: the serializable-vs-runtime rule, code-first specs, knobs vs
  deps, per-phase configuration, prompt sourcing, budgets, manifests with
  `agent_from_manifest`, escape hatches, and a migration checklist for downstream
  projects. Linked from the README, docs nav, and `AGENTS.md`.
- **Per-phase configuration for `Agent`/`ReasoningSpec` and manifests.** Multi-phase
  strategies accept canonical per-phase overrides through the two existing buckets:
  runtime LLM/tools as deps (`planner_llm`, `executor_tools`, `reviewer_llm`, …) and
  prompts as knobs (`planner_system`, `evaluator_system`, …), validated per strategy —
  `FlowStrategy` gains `phases`, `allowed_deps`, `dep_validators`, and `validate_spec()`.
  Agent manifests gain a `strategy.phases` section: per-phase `system`/`system_file`
  prompts fold into the spec's canonical knobs via `reasoning_spec()` (verbatim text,
  governed by `allowed_roots`/override policy and re-verified against the load-time
  fingerprint — drift raises), and per-phase `model` configs are exposed via
  `ResolvedAgentManifest.phase_models()` and resolved by the application through the
  new `agent_from_manifest(…, llm_factory=…)` helper (spec and strategy validate
  before the factory runs). New CLI subcommands `ai-arch agent validate|inspect`
  (with `--allowed-root`) run registry-aware checks — strategy name, phase names,
  LLM-bindability of phases declaring models, knob values — for CI. Planner prompts (inline, knob, or
  `system_file`) may carry a `{tools}` token — the only substitution the framework
  performs — replaced at build time with the phase's resolved tool catalog. Also new:
  the `lats.exploration_weight` and `self_discovery.modules` knobs. See
  [docs/agents.md](docs/agents.md).
- Hermetic configured-agent system tests now exercise manifest inheritance and profiles,
  prompt rendering, governed tools, agent compilation, metering, and hard budgets as one
  end-to-end path. The separate `live_api` marker keeps paid provider smoke tests explicit.
- `ModelPricing` and `PricingRegistry` are public core/top-level exports, so custom pricing can be
  registered without importing a private module.
- **Public configurable-agent manifests.** `load_agent_manifest()` strictly loads
  versioned YAML/JSON/TOML agent definitions with multi-parent inheritance, embedded
  profiles, relative-path confinement, governed deterministic overrides, provenance,
  content-aware fingerprints, and direct `ReasoningSpec` / `BudgetPolicy` construction.
- **Recursive prompt subsections.** `PromptSection` accepts nested `sections=`, forming a tree: each section renders its own content and then its subsections, and every layout translates depth (Markdown deepens heading levels, XML nests elements; Text/JSON follow suit). Manifests, section spans, provenance, and the `ai-arch prompt` CLI all follow the hierarchy; see `examples/46_prompt_subsections.py`.
- **Complete prompt and resource system.** `toolkit.resources` now provides policy-controlled local/package loading, TXT/Markdown/JSON/YAML/TOML codecs, RFC 6901 and text selectors, deterministic serializers, fingerprints, and provenance. `toolkit.prompts` adds file-backed sections, typed `PromptTemplate` variables, explicit stdlib/Jinja engines, Text/Markdown/XML/JSON layouts with section spans, versioned YAML/JSON/TOML manifests with includes/extends, Knowledge sources, and `ai-arch prompt validate|inspect|render`. Nanope can append or replace its built-in prompt with a toolkit manifest.
- Prompt messages now compose resolved prompts, templates, literal text, and multimodal `Content` into deterministic system/user/assistant conversations. Resources support in-memory snapshots, resolver-scoped custom serializers, media-type policy allowlists, and direct `Prompt.from_resource()` / `PromptSection.from_resource()` conveniences. Knowledge adds deterministic lexical search and Nanope/CLI integrations.
- **Cost metering & budgets.** A neutral metering mechanism in `core` (`MeterScope` / `RunConfig`, opaque exact `Money`, `Cost`, `MeterSnapshot`, `UsageEvent` / `UsageSink`, the `AdmissionController` protocol) with an opinion layer on top in `toolkit.budget` (`BudgetPolicy`, `BudgetController`, `BudgetReport`, `budget_scope()`, `HeuristicEstimator`). Attach `budget_policy=BudgetPolicy(max_cost=…, max_llm_calls=…, max_total_tokens=…, max_wall_s=…)` to a `Flow` or `Agent`, at construction or per run (`flow.run(state, budget_policy=…)` / `agent.run(task, budget_policy=…)`); all nine agent flows and `Agent` honour it, and nested flows share one cumulative budget. Caps are enforced **hard at the charge site** — the LLM/tool operation that would exceed a cap is denied before it runs (`BudgetExceeded`, surfaced as `policy_decision="budget_exceeded"`), exact even under concurrency. Two knobs shape a cost cap: `reserve="strict"` (reserve a worst-case hold up front) and `unpriced="fail_closed"` (default — deny once a call can't be priced, e.g. an unpriced model or a server tool). Read spend off the run's meter: `result.meter` (a `BudgetReport`), `result.total_cost`, `result.usage`, or `agent_result.report`. All metering types are re-exported at the top level; see [Tool Governance & Safety → Cumulative budgets](docs/safety.md).
- **Concurrency controls.** `inference_limit(n)` caps concurrent LLM calls globally — across every nested flow, agent, and fallback — to protect a shared resource (local GPU, rate-limited endpoint, connection pool); `Flow(max_parallelism=n)` bounds how many steps of one flow fan out at once. Both opt-in, independent, and composable; see [docs/concurrency.md](docs/concurrency.md).
- **Anthropic schema-in-prompt structured output.** `structured_output_mode="prompt"` (an `LLM(...)` constructor default or a per-call kwarg) makes the Anthropic adapter inject the JSON schema into the system prompt and parse the reply, instead of the native `output_config`. This handles large analysis/planning-style schemas that exceed Anthropic's native structured-output complexity limit (which otherwise returns a 400 "schema is too complex"). Defaults to `"native"` (unchanged behaviour); the OpenAI, Gemini, and xAI adapters accept and ignore the kwarg.
- **Agent & ReasoningSpec.** A declarative facade over the flow factories in `toolkit.agents`: `ReasoningSpec` is a frozen, serializable description of how an agent reasons (`strategy`, `system`, `max_iterations`, strategy-specific `knobs`, `policy`, `timeout`, `llm_kwargs`, `output_schema`; `from_mapping()` builds one from parsed JSON/YAML), and `Agent` binds it to an `LLM` + `ToolGroup`, compiles the `Flow` once, and exposes `run()` / `run_sync()` / `iter()`, `Agent.from_flow()`, and `as_step()`. `AgentResult` carries `text` / `response` / `flow_result` plus meter-derived `usage` / `cost` / `report`. The strategy registry (`register_strategy()` / `get_strategy()`) ships 10 built-ins — the nine flow factories plus `completion` (a single LLM call, no tool loop); only `react` / `completion` support `output_schema`. See [docs/agents.md](docs/agents.md).
- `@deprecated` decorator helper for the pre-1.0 deprecation policy.
- **Tool execution governance.** One core pipeline governs every tool call: `@tool` carries `capability` / `risk_level` / `requires_approval` metadata, runtime gates (`DangerousToolGate`, `ApprovalGate` + `ApprovalHandler`, `DryRunGate`) run before execution, and results are structured `ToolResult` / `ToolError` values instead of bare strings. Side-effectful tools (shell, filesystem, Python eval, web fetch) moved behind the opt-in `toolkit.tools.dangerous` namespace, and traces/logs pass through secret redaction (`RedactionPolicy` / `Redactor`). See [Tool Governance & Safety](docs/safety.md).
- **Atomic, versioned graph persistence.** `Graph.save()` writes through a temp file + `os.replace` so a crash can't leave a half-written store, and saved payloads are versioned.
- **25+ new toolkit tools** across science, geo, health, and media APIs — arXiv, ClinicalTrials.gov, Crossref, DataCite, GDELT, Internet Archive, and more — bringing the pre-built, stdlib-only catalog to 132 tools ([docs/tools-catalog.md](docs/tools-catalog.md)).
- **Local OpenAI-compatible servers.** `LLM("gemma4:e4b", base_url="http://localhost:11434/v1")` routes arbitrary model tags to the OpenAI adapter for Ollama, LM Studio, and vLLM. A `provider=` kwarg on `LLM` / `create_provider()` forces a specific adapter regardless of the model name; an unknown model with `base_url=` set falls back to the OpenAI-compatible adapter automatically.
- **Optional API key on localhost.** When `base_url` points at a loopback host (`localhost` / `127.x` / `::1`), the API key is optional (a placeholder is used) and a cloud key from the environment is **not** forwarded to the local server. Remote endpoints (gateways, proxies) still require a key and fail fast when it is missing.
- **Real-time reasoning streaming.** Vendor reasoning deltas (`reasoning_content` / `reasoning`) from OpenAI-compatible servers surface as incremental `thinking` events in `stream_events()` and populate `Response.thinking` in `complete()`, streaming, and batch responses. The reasoning-so-far is preserved even when a stream is abandoned early or errors mid-way.
- **`StreamEvent.partial`** flag distinguishes incremental reasoning fragments (`True`, OpenAI-compatible servers) from complete thinking blocks (`False`, Anthropic). Concatenate consecutive partial `thinking` events for the full trace.
- nanope research_center, agent_swarm scaffold, and advanced configurable agent (work in progress, not part of the public toolkit API).
- `AGENTS.md` — the canonical instructions file for all coding agents, rebuilt lean (commands, layer rules, testing patterns, gotchas); `CLAUDE.md` is now an `@AGENTS.md` import stub so Claude Code reads the same file.
- `app` optional-dependency extra (`reflex>=0.7` + graph + yaml).
- `pyright>=1.1.390` and `pre-commit>=4.0` in the `dev` extra; `[tool.pyright]` configured in standard mode over `src/`.
- `.pre-commit-config.yaml` with ruff-check / ruff-format and standard hygiene hooks (trailing whitespace, EOF newline, yaml/toml validity, merge-conflict + large-file guards).
- `CONTRIBUTING.md` covering setup, conventions, and how to add a provider / tool / agent flow.
- `.env.example` documenting every provider API key; the sync-timeout configuration now validates its inputs.

### Changed
- Planner tool awareness is now an explicit `{tools}` token instead of a silent append:
  a prompt containing the token gets the phase's rendered tool catalog substituted at
  build time (`(none)` when empty), and a prompt without it is never modified. `rewoo`
  previously appended the catalog to custom `planner_system` prompts unconditionally;
  declare the token where you want the list — the built-in default planner prompts of
  `plan_execute`, `rewoo`, and `llm_compiler` carry it.
- Built-in agent strategies now validate `deps` the same way they validate knobs:
  unknown keys and wrongly-typed values raise `ValueError` at build time, so a typo
  like `deps={"evalutor": …}` can no longer be silently ignored. Custom strategies
  registered without `allowed_deps` keep the previous accept-anything behavior.
  `generate_review` accepts canonical `generator_llm`/`generator_tools` and
  `reviewer_llm`/`reviewer_tools` dep keys, with `review_llm`/`review_tools` kept as
  legacy aliases (passing both is an error).
- Built-in agent strategies now reject unknown strategy knobs and invalid knob values at
  compile time, before an agent can spend tokens.
- Reflexion and LATS evaluators are runtime dependencies only (`deps["evaluator"]` and
  `deps["evaluator_fn"]`); serializable strategy knobs no longer accept executable callables.
- `ReasoningSpec.output_schema` now accepts supported model classes in addition to `OutputSchema` instances and schema mappings, matching `LLM.complete()`.
- **Knowledge loading now delegates to Resources.** `KnowledgeRegistry.load()` and `.from_directory()` retain parsed data and source fingerprints; the original `load_text()` / `load_json()` / etc. signatures remain compatibility wrappers. Duplicate `register()` keys now require explicit `overwrite=True`.
- **The meter is the single source of truth for spend.** Agent flows no longer thread `cost` / `usage` into their step results by hand — `FlowResult.total_cost` / `.usage` / `.meter` and `AgentResult.cost` / `.usage` / `.report` all derive from the run's meter, so nested, parallel, and streaming spend can no longer drift from a parallel hand-rolled tally. A per-step `Policy.max_cost` is now enforced against that step's **metered span cost** (previously the step's manually declared `Result.cost`, which flows no longer set). `Trace.total_cost` / `total_usage` remain a raw view of any cost a custom step annotates via `Result(cost=…)`.
- **String fallback routing.** A string `fallback=` is now routed by its own model name: a recognizable model (e.g. `claude-…`) fails over to its own provider and connection, so a local-primary → cloud-fallback chain works; a bare local tag inherits the primary's `base_url` / `provider`, assuming it lives on the same server. Pass `LLM` instances as fallbacks for full per-fallback control.
- README rewritten with badges, copy-paste snippets (completion, streaming, tools, ReAct), a provider × feature matrix, and an agent-architecture table.
- CI hardened: `lint` (ruff check + format), `typecheck` (pyright — now blocking, driven to 0 errors), and `test` across ubuntu + macos × Python 3.13/3.14 with coverage; `uv lock --check` enforces lockfile consistency; an examples smoke test catches public-API drift; live-API integration tests run on demand and on a daily cron.
- `[tool.ruff]` and `[tool.pyright]` both exclude `src/ai_arch_toolkit/nanope` (sub-projects have their own idioms).
- Pricing registry refreshed (2026-05-19): Claude 4.7 added; Claude 4.6/4.7 now ship with 1M context at standard rates (long-context tier removed).
- Examples 31–33 updated to the variadic `Flow(*steps)` API and async `flow.run(state)`.
- Docs restructured into per-subsystem pages with coverage gaps closed; added the code-style guide, the Agent & ReasoningSpec page (now surfaced as the recommended entry point across README and docs), and the prompt-system suite (templates, layouts, manifests, messages, migration, extensibility).
- Docs and example index aligned with the Flow-based architecture; legacy "pipelines" and "8 agent architectures" wording removed.
- Memory `Node` reconciled with `core.graph.Node[T]` (zero pyright ignores).
- Public API surface tightened: `__init__` re-exports audited so internals stop leaking.
- Sync timeouts in `core/_sync.py` no longer expose the dead `SYNC_TIMEOUT` / `STREAM_JOIN_TIMEOUT` aliases; use `configure_sync_timeouts()` instead.
- `RateLimitMiddleware` docstring documents the streaming-bypass limitation explicitly (previously a TODO).
- `uv lock --upgrade` brought every transitive dependency to its latest compatible version (pydantic 2.13, urllib3 2.7, requests 2.34, websockets 16, xai-sdk 1.12, ruff 0.15.13, …); resolved the four Dependabot alerts.

### Fixed
- Examples and docs no longer reference retired Anthropic model ids: the retired
  `claude-sonnet-4-20250514` (404 since 2026-06-15) and the malformed
  `claude-opus-4-0-20250514` became `claude-sonnet-5` / `claude-opus-5`, and the dated
  `claude-haiku-4-5-20251001` was normalized to the recommended `claude-haiku-4-5`
  alias.
- **Bundled pricing table refreshed against each provider's live model list**
  (2026-07-26). Added: `claude-fable-5`/`claude-mythos-5`, `claude-opus-5` (with 2x
  fast-mode rates), `claude-sonnet-5`, and `claude-opus-4-8` — the latter previously
  fell through the `claude-opus-4` fallback prefix and was **billed at 3x its real
  price**; OpenAI `gpt-5.6-sol`/`-terra`/`-luna`; `gemini-3.6-flash`,
  `gemini-3.5-flash`, `gemini-3.5-flash-lite`; `grok-4.5` and `grok-build-0.1`, plus
  the 200k-prompt long-context tiers on all current Grok entries. Removed entries for
  models the providers no longer serve: Claude 3.x and 4.0, `o1-mini`/`o3-pro`/
  `o3-deep-research`, Gemini 1.5, and `grok-2`. Stale 6x fast-mode rates were dropped
  from Opus 4.6/4.7 (fast mode was removed on those models).
- A second sync call on the same `LLM` instance no longer fails with
  `APIConnectionError`. The sync wrappers run each call on a fresh `asyncio.run()`
  loop, but every adapter cached its async SDK client, whose connection pool stayed
  bound to the first (closed) loop. Providers now rebuild the client once the loop it
  served has closed (`LoopAwareClientCache`, all four adapters); directly assigned
  clients — e.g. test mocks — are never replaced. Repeated `complete_sync`/`run_sync`
  calls, notebook usage, and per-test event loops all recover; using one instance
  from two concurrently live loops remains unsupported.
- `uv sync --extra dev` now installs `jsonschema` and `jinja2`, so the prompt-template
  tests pass on a fresh dev environment (previously only the `prompts` extra pulled
  them in).
- The Anthropic adapter now drops the client-default `temperature` for every model
  family that rejects sampling parameters — Opus 4.7/4.8/5, Sonnet 5, and Fable/Mythos 5,
  matched by prefix. Previously only `claude-opus-4-7` was covered, so every call to a
  current Anthropic model failed with `400: temperature is deprecated for this model`.
- `ReasoningSpec.llm_kwargs` now reach every phase of all multi-phase strategies
  (`plan_execute`, `rewoo`, `reflexion`, `self_discovery`, `llm_compiler`, `tot`,
  `lats`, and `generate_review`'s reviewer — where `reviewer_kwargs` merges on top,
  winning per key); previously they were silently dropped.
- `llm_compiler_flow` supports an executor LLM override again (`exec_llm`); the inner
  ReAct always used the default LLM even though planner/joiner overrides existed.
- `plan_execute_flow`'s planner sees the executor's tool catalog again — and
  `llm_compiler_flow`'s planner gains it for the first time — via the `{tools}` token
  in their default planner prompts, so plans match what the execution phase can
  actually call.
- Provider adapters now disable hidden Anthropic, OpenAI, Gemini, and xAI SDK
  retry loops. `LLM.retry` is the single retry owner, so every attempt is
  metered and exposed through `Response.attempts`.
- Streaming now retries or enters the fallback chain when provider errors occur lazily
  before the first emitted item. Each physical attempt is metered and recorded; errors
  after observable output are surfaced without replay. Abandoned async streams close
  their provider iterators immediately; all streams remain outside `inference_limit` and
  preserve nested fallback chains configured on fallback `LLM` instances.
- Agent manifests now resolve relative paths inside embedded profiles against the
  declaring file and override-supplied paths against the entry manifest, validate
  inherited `*.agent.*` suffixes, and default a missing `strategy` section to ReAct.
  Their canonical JSON-like data boundary now guarantees deterministic fingerprints;
  secret scanning covers every source/profile; descendant deny rules cannot be bypassed
  through parent overrides; and multi-root source provenance cannot collide.
- `BudgetPolicy` and agent-manifest limits now reject invalid `reserve` / `unpriced`
  modes, malformed count caps, and non-finite numeric caps instead of silently
  weakening enforcement.
- The Gemini extra now requires `google-genai>=1.21.0`, the first supported SDK floor
  with `HttpOptions.retry_options`.
- Structured agent responses with an empty text field now retain their parsed payload in
  `AgentResult.response`.
- `AgentResult.errors` now retains failures from earlier iterations of cyclic flows instead of
  losing them when a later result with the same step name succeeds.
- **Cached input tokens are no longer double-counted.** OpenAI, Gemini, and xAI adapters (including OpenAI batch responses) now normalize provider-reported inclusive input totals into disjoint `Usage.input_tokens` and `cache_read_tokens`, so pricing and budget metering charge cached tokens only at the cache rate.
- **Anthropic structured output now validates into the Pydantic model.** When `output_schema` carries a `model_class`, the Anthropic adapter coerces the parsed JSON via `model_validate()` — at parity with the OpenAI, Gemini, and xAI adapters (it previously returned a raw `dict`). It also tolerates Markdown-fenced JSON in the reply.
- OpenAI-compatible streaming now flushes accumulated tool calls when a server ends the turn with `finish_reason="stop"` instead of `"tool_calls"` (some Ollama / LM Studio / vLLM builds), so tool-using agents no longer silently see zero tool calls.
- Structured output: parsed JSON is now validated against the Pydantic model before being returned.
- Gemini API key resolution: `GOOGLE_API_KEY` now takes precedence over `GEMINI_API_KEY` when both are set.
- Lint fixes: ternary form in `toolkit/tools/_datetime.py`; unused `pytest` import in `tests/test_python_eval.py`; misc `ruff format` across toolkit tools.

### Removed
- **The legacy `core._budget` module** (`BudgetState`, its cooperative `BudgetPolicy`). Budgets now live in `toolkit.budget` and enforce hard at the charge site rather than by cooperative counter-checking as steps record usage. `BudgetPolicy.max_wall_time` is now `max_wall_s`; the `strict_cost` / `allow_unpriced` flags are now the `reserve` / `unpriced` knobs. `BudgetExceeded` keeps its `.limit` / `.maximum` / `.to_dict()` surface.

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
