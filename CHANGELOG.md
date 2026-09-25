# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Upgrade notes

Code written against the `main` before the hardening front (provider errors and metering, tools,
flows, manifests) needs these changes; each one is detailed below.

- **Provider errors.** Catch `ProviderError`, or one of `RequestError`, `TransportError`,
  `ProviderTimeout` and `ResponseError`: SDK and HTTP-library exceptions no longer leave the
  adapters (`APIError` handlers keep working). `fallback_on` defaults to `(ProviderError,)`;
  pass the old tuple to keep falling back on `ConnectionError`, `TimeoutError` or `OSError`.
- **Requests a model does not take.** A thinking effort, `thinking=True`, sampling parameter,
  forced `tool_choice`, server-tool config or message role that the model's documentation does not
  allow raises `RequestError` before anything is sent, instead of reaching the API or being
  dropped.
- **Prices.** A model id is priced by its own entry: register the variants you call
  (`pricing.register("o3-pro", ...)`), or keep a family with `match="prefix"`;
  `PricingRegistry.register()` takes `model` as its first parameter. Under a `MeterScope`, a model
  without a price raises `UnpricedModelError`: register one (`ModelPricing()` for a local model).
  `Response.cost` is `None`, not `0`, when the provider reported no usage.
- **Custom meters.** `MeterOperation.fail(...)` requires a delivery disposition, and
  `unknown_cost_count` counts only unbounded costs.
- **Streams.** A finished stream carries the same `Response` as `complete()`, and
  `stream_events()` emits the tool-call events after the text, in the response's order.
- **Tools.**
  - `csv_read` is in `toolkit.tools.dangerous` and needs approval.
  - Every tool call has a 120-second deadline and a 200,000-character output cap: set
    `timeout_s`/`max_output_chars` (or `None`) on a tool that needs more.
  - On the sync path a synchronous tool runs in a thread of its own: open thread-bound resources
    inside the tool.
  - `ip_lookup` queries ipwho.is over HTTPS; the MediaWiki tools accept only HTTPS Wikimedia
    hosts.
- **Flows and agents.**
  - The steps of a parallel DAG wave read the wave's snapshot: return changes as artifacts, since
    a value mutated in place now changes the state in every mode. The trace records a wave's steps
    as they finish.
  - `ReasoningSpec.knobs` and `llm_kwargs` are read-only: build a new spec with
    `dataclasses.replace`. A spec no longer deep-copies or pickles.
  - An agent's answer is its flow's `"answer"` or, without one, its last step's value: a flow
    wrapped with `Agent.from_flow` writes `"answer"` (and `"response"`) to keep its text.
- **Manifests.** Shape errors changed wording (they name the field's path). These values are now
  refused:
  - a `strategy.system` that is not text;
  - `version: true`;
  - a boolean `order`;
  - a `metadata_attributes` that is not a list;
  - `select` or `serialize_as` on an inline template.

### Added
- `UnpricedModelError`, and prices for `gpt-4o-2024-05-13` and `gpt-3.5-turbo-1106` (snapshots
  with a tariff of their own), `gpt-5.5-cyber`, and `gpt-5.1` (at `gpt-5`'s rates), from
  OpenAI's pricing page on 2026-09-18.
- Provider failures share `ProviderError` and a typed delivery disposition. New `RequestError`, `TransportError`, `ProviderTimeout`, and `ResponseError` preserve the existing builtin exception handlers.
- **Meta provider (Muse Spark).** `LLM("muse-spark-1.3")` routes to a new `MetaProvider` that
  drives the Meta Model API's Responses API through the `openai` SDK (Meta ships no SDK), with the
  key from `MODEL_API_KEY` and a new `meta` extra. Requests are stateless and replay the model's
  encrypted reasoning from `Response.to_message()`, so tool loops and agents keep their chain of
  thought across turns. Streaming, tool calls, structured output, JSON mode, reasoning summaries,
  web search, images, PDFs, and `count_tokens` are supported; pricing covers the standard and
  contributor tiers. Muse Spark always reasons: `thinking_effort` applies on its own and
  `thinking=True` requests summaries. Only `tool_choice="auto"` is available (`"none"` sends no
  tools; forced choices raise `ValueError`). `count_tokens_local` estimates Muse Spark with
  `o200k_base`. See [docs/model-compatibility.md](docs/model-compatibility.md#meta).
- The live probe inventory accepts a per-model `tool_choice` for providers that cannot force a call.
- **`Flow(timeout=...)`** bounds a whole run of a flow, in seconds. When it elapses, the steps in
  flight are cancelled, nothing else starts, and the trace ends with a `flow_timeout` step.
  `ReasoningSpec.timeout`, manifest `strategy.timeout`, and `limits.timeout_seconds` compile to it.
- **`Flow(trace_capture="keys" | "full" | "none")`** sets what each step's trace records. Also new:
  the `TraceCapture` type, `StepTrace.input_keys` / `output_keys`, `execute_step(capture=)`,
  `ReasoningSpec.trace_capture`, and manifest `strategy.trace_capture`. See
  [docs/flow-architecture.md](docs/flow-architecture.md#what-a-trace-captures).
- **Flow and agent executions.** `Flow.iter()` and `iter_flow()` return a `FlowExecution`,
  `Flow.iter_sync()` a `SyncFlowExecution`, and `Agent.iter()` an `AgentExecution`: iterate them as
  before, then read `.result`. `retry`, `timeout`, `fallback`, and `policy_decision` events stream
  while a step runs; `step_end` events carry the step's `error`; `FlowResult.meter_scope` exposes the
  run's scope; `StepTrace.children` holds the steps of flows run inside a step.
- `Agent.run()`, `run_sync()`, and `iter()` accept a per-run `config=RunConfig(...)`.
- **Tool-call arguments are validated and coerced against the tool's schema before any gate runs.**
  Numeric and boolean strings are coerced (`"3"` for an integer), `enum` and `anyOf` are checked,
  missing or unknown arguments are refused, and an approval handler sees the coerced values. See
  [docs/safety.md](docs/safety.md#argument-validation).
- `ToolGate`, `ExecutionContext`, `GateBlock`, `GateModify`, `GateDryRun`, and `GateResult` are
  exported from `ai_arch_toolkit` and `ai_arch_toolkit.core`, so custom gates no longer import
  private modules.
- GPT-6 Astra pricing (including cache, batch, long-context, and fast rates),
  Chat Completions parameter handling, and manual probe inventory. Unsupported
  reasoning efforts and tool calling fail clearly; Astra tools require Responses.
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
- **Agent & ReasoningSpec.** A declarative facade over the flow factories in `toolkit.agents`: `ReasoningSpec` is a frozen, serializable description of how an agent reasons (`strategy`, `system`, `max_iterations`, strategy-specific `knobs`, `policy`, `timeout`, `llm_kwargs`, `output_schema`; `from_mapping()` builds one from parsed JSON/YAML), and `Agent` binds it to an `LLM` + `ToolGroup`, compiles the `Flow` once, and exposes `run()` / `run_sync()` / `iter()`, `Agent.from_flow()`, and `as_step()`. `AgentResult` carries `text` / `response` / `flow_result` plus meter-derived `usage` / `cost` / `report`. The strategy registry (`register_strategy()` / `get_strategy()`) ships 10 built-ins — the nine flow factories plus `completion` (a single LLM call, no tool loop); `react`, `completion`, and `generate_review` support `output_schema`. See [docs/agents.md](docs/agents.md).
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

- **Output and time limits for every tool.** `ToolRuntimePolicy.max_output_chars` (default
  200,000) and `timeout_s` (default 120 seconds), set per tool with
  `@tool(max_output_chars=..., timeout_s=...)` (`None` switches one off).
  `ToolGroup(max_output_chars=..., timeout_s=...)` sets a ceiling for all its tools: the stricter
  value applies. A longer result is cut and marked, in its text and in
  `ToolResult.metadata["truncated"]`; a call past its deadline returns a `timeout` failure. See
  [docs/safety.md](docs/safety.md#output-and-time-limits).
- `FlowOptions` (`ai_arch_toolkit.toolkit.agents.flows`): the four options every flow factory
  hands to its `Flow` (`timeout`, `trace_capture`, `policy`, `budget_policy`), declared once, so a
  wrapper can forward them with their types (`**options: Unpack[FlowOptions]`). See
  [docs/flow-architecture.md](docs/flow-architecture.md#flow-options).
- Agent manifests have a packaged JSON Schema,
  `ai_arch_toolkit/toolkit/agents/schemas/agent-manifest-v1.schema.json`, for editors and other
  tools. Like the prompt manifest's, it is generated from the declaration the loader enforces. See
  [docs/agents.md](docs/agents.md#file-backed-agent-manifests).

### Changed
- **Breaking: prices match a model id exactly.** A model id is priced by its own entry or as a
  dated snapshot of one (`claude-haiku-4-5-20251001`, `gpt-4o-2024-08-06`, `grok-4-0709`,
  `-latest`). A variant no longer inherits the entry its name starts with: `o3-pro`,
  `gpt-4o-audio-preview`, `gpt-5-codex`, and `gemini-2.5-flash-image` used to be priced as `o3`,
  `gpt-4o`, `gpt-5`, and `gemini-2.5-flash`. `PricingRegistry.register()` is exact by default and
  its first parameter is now `model` (was `model_prefix`); `register(model, price, match="prefix")`,
  or `match = "prefix"` in a TOML entry, keeps prefix matching as an explicit choice for a family
  of local models. A TOML entry takes `aliases = [...]` for other ids billed at its rates, and the
  shipped table lists the documented ids that prefix matching used to cover
  (`grok-4-1-fast-reasoning`, `gemini-3.1-pro-preview`, `gpt-5.1`, …).
- **Breaking: a model without a price does not run under a meter.** Inside a `MeterScope`, with or
  without a budget, a call to a model the scope's pricer cannot price raises `UnpricedModelError`
  (a `RequestError`, so neither retried nor passed to a fallback) before anything is sent; the
  message names the `pricing.register(...)` call that fixes it. A local model registers an
  explicit zero: `pricing.register("llama3.2", ModelPricing())`. Without a scope nothing changes:
  the call runs and `Response.cost` is `None`. `BudgetPolicy(unpriced=...)` now governs only the
  other unknown costs (server tools, a pricer that fails when a call settles).
- **Every provider failure says whether the request was sent.** An adapter now knows when the
  SDK handed a request to its transport. A connection that is refused or times out while
  connecting, or a request the SDK itself refuses to build (a value that is not JSON, for
  example), is a failure that was never sent (`delivery="not_sent"`: no cost, and the next call is
  admitted); an HTTP 200 whose body cannot be read is a `ResponseError`, and so is an error event
  inside a stream; a transport failure while reading a stream is a `TransportError` or
  `ProviderTimeout`. None of these leave as the SDK's or the HTTP library's own exception any
  more.
- **xAI: requests follow each Grok model's documented rules, and gRPC errors keep their meaning.**
  `thinking_effort` is sent as `reasoning_effort` where the model documents it (`low` to `xhigh`
  on `grok-4.6`, `grok-4.5`, and any newer model; also `none` on `grok-4.3` and the retired ids it
  serves), and applies without `thinking=True`; it used to be dropped with a warning. A
  `thinking_effort` on a model that reasons without one (`grok-4.20-reasoning`, `grok-build-0.1`)
  raises `RequestError`, and so does `thinking=True` on a model that does not reason
  (`grok-4.20-non-reasoning`). The reasoning models refuse `stop`, `presence_penalty`, and
  `frequency_penalty` before sending, as xAI's API does. `tool_choice="<tool name>"` works (it
  failed inside the SDK); a server tool, or any tool on `grok-4.20-multi-agent`, raises
  `RequestError`, and so does a request the SDK cannot build, before anything is sent. gRPC
  `UNAVAILABLE` is a `TransportError` and `DEADLINE_EXCEEDED` a `ProviderTimeout` (they were
  `APIError` 503 and 504); every other code maps to its `google.rpc` HTTP status
  (`FAILED_PRECONDITION` is 400, `UNIMPLEMENTED` 501), and only `RESOURCE_EXHAUSTED` is unbilled.
- **Gemini: a turn's tool results go back together, and thinking follows each model's rules.**
  The results of a turn's tool calls are sent in one `user` content, as Gemini documents for
  parallel calls, each with its call's `id` when Gemini gave the call one (Gemini 3 maps results
  by id); they used to be split into one content each, without ids. `thinking_effort` is checked
  against the model's documented levels (`minimal` only where the model has it; `xhigh` or `max`
  now raise `RequestError` instead of reaching the API) and applies without `thinking=True`; on
  Gemini 2.5 it becomes a thinking budget, and a `thinking_budget` outside the model's range raises
  `RequestError`. A server tool with a config, or of a type the adapter does not send, raises
  `RequestError` instead of being dropped, and so does a message role Gemini does not have. The
  SDK always sends through `httpx` now: with `aiohttp` installed (the `xai` extra installs it), it
  used to re-send a request on its own after a connection error, unmetered. A failed request with
  HTTP 400 or 500 is unbilled, as Gemini's billing page says, and so is a 429; an error inside a
  stream is indeterminate.
- **Meta: failures keep their code's documented meaning and their usage.** A failure Meta reports
  inside a response or a stream (`response.failed`, an `error` event) gets the HTTP status Meta's
  error table gives its code; a code outside the table, or none (a 400 and a 500 share it), raises
  `ResponseError` instead of an invented 400 or 500. A failed response's usage is kept: the error
  carries it (`ProviderError.usage`), `Response.attempts` records it, and a meter settles the
  failure with its cost instead of an unknown one. `thinking_effort` is checked per model (`max`
  only on standard-tier `muse-spark-1.3`); `logprobs=True`, a message role the Responses API does
  not have, `code_execution`, and a server tool's config raise `RequestError` (the last two were
  dropped with a warning). `MetaProvider(timeout=...)` no longer needs `httpx`, which the `meta`
  extra does not install. A connection refused before sending is `delivery="not_sent"`.
- `MeterOperation.fail()` takes the `usage` and `cost` of a failure the provider reported usage
  for.
- **Anthropic: thinking, effort and tools follow each Claude model's documented rules.**
  `thinking=True` asks the models that think adaptively (the Claude 5 family, Opus 4.8, 4.7 and
  4.6, Sonnet 4.6) for `{type: "adaptive", display: "summarized"}`, so the summary shows on the
  models that hide it by default; `claude-sonnet-4-6` and `claude-opus-4-6` used to get a deprecated
  `budget_tokens`. `thinking_effort` goes in `output_config.effort` (merged with the structured
  output format) and applies without `thinking=True`; an effort the model does not take (`xhigh`
  on the 4.6 models, for example) raises `RequestError`. On the models that only take a budget
  (Haiku 4.5, Sonnet 4.5, Opus 4.5 and older), an effort or a budget turns thinking on, and the
  budget is added to `max_tokens` instead of being taken from it. The results of a turn's tool
  calls go back in one `user` message, and an assistant turn is replayed as Claude sent it,
  thinking signatures and server tool results included, while the message still matches it.
  Forced `tool_choice` on `claude-fable-5-1` and `claude-mythos-5-1`, a `top_p` or `top_k` on a
  model without sampling parameters, a message role the Messages API does not have, and a request
  without `max_tokens` raise `RequestError` before sending. Server tools carry their `name` (the
  API refused them without it), and `code_execution` moves from the legacy
  `code_execution_20250522` to `code_execution_20250825`; a server tool with a config, which used
  to be dropped, raises `RequestError`. A
  failed request is unbilled, as Anthropic documents; an error event inside a stream gets the
  status of its error type (it was an `APIError` with status 200), and a stream whose tool input is
  not valid JSON raises `ResponseError` (the input used to reach the tool as `{"_raw": ...}`).
  Batch bodies are built as a call's.
- **OpenAI: the output limit follows the host, and model rules follow the model's family.** On
  `api.openai.com` every model receives `max_completion_tokens` (`max_tokens` is deprecated and
  refused by o-series models); an OpenAI-compatible server (`base_url=`) receives `max_tokens`
  and no OpenAI model rule. `thinking=True` on a model that does not reason (`gpt-4o`,
  `gpt-4o-mini`, `gpt-4.1*`, `gpt-4-turbo`, `gpt-4`, `gpt-3.5-turbo`) raises `RequestError`
  before sending; so do a `thinking_effort` outside the SDK's values and a server tool (Chat
  Completions takes only function tools; server tools are planned separately). A model the
  adapter does not know gets the current generation's rules. `top_logprobs` is forwarded, a
  `developer` message is sent as one, and another unknown role raises `RequestError`. The batch
  JSONL body is built by the same code as a call.
- **A stream ends with the same `Response` as `complete()`.** The provider adapters now build
  one response from the SDK's final object on both paths (the SDKs accumulate Anthropic, OpenAI,
  xAI, and Meta streams; Gemini chunks are joined). A finished `stream()` or `stream_events()`
  therefore carries `parsed`, `citations`, `response_id`, and `logprobs`, and its text is the
  provider's, as `complete()` returns it; the Gemini stream's `raw` keeps every part of every
  chunk, so a replayed history keeps all its function calls. Tool-call events in
  `stream_events()` come after the text, from the finished response and in its order (Meta used
  to emit them in arrival order). A stream the caller abandons reports the text it consumed and
  the thinking and tool calls it saw, with unknown usage.
- **A response without usage has no cost.** When the provider reports no usage (an
  OpenAI-compatible server that sends no usage chunk, or an xAI stream whose chunks carry none),
  `Response.cost` is `None` and a meter records the call's cost as unknown instead of zero.
- **A request an adapter refuses is never metered.** Building the provider request happens
  before admission: an adapter's refusal (`RequestError`) opens no operation and counts no call. A stream raises it from `llm.stream(...)`
  itself, or on its first iteration when middleware rewrote the request (its reservation is
  released).
- Complete and both stream APIs share one physical-attempt pipeline. Stream lifecycle handles are explicit; public LLM call signatures are unchanged.
- The default `fallback_on` is `(ProviderError,)` instead of `(APIError, ConnectionError, TimeoutError, OSError)`. The built-in adapters raise `TransportError` or `ProviderTimeout` for network failures, so those still fall back; a raw `OSError` (a missing local file, for example) no longer does. Custom providers should raise the normalized errors.
- **Breaking:** `MeterOperation.fail` requires a delivery disposition. `unknown_cost_count` counts only unbounded costs; bounded uncertainty consumes caps separately from known spend. `BudgetReport.cost_at_most` reports the combined bound.
- MediaWiki tools accept only HTTPS Wikimedia domains and subdomains, without credentials or ports.
- **Breaking:** `csv_read` moves to `toolkit.tools.dangerous` and requires approval in governed execution.
- **Breaking: provider SDK majors and dependency floors.** The extras now require the current SDK
  majors, each capped at the next one: `anthropic>=1.0,<2`, `openai>=3.0,<4` (the `openai` and
  `meta` extras), `google-genai>=2.0,<3`, `xai-sdk>=1.18,<2` (was 1.7; 1.18 is the first release
  that takes the `xhigh` effort). `anthropic` 1.x and `openai` 3.x moved
  their HTTP transport to `httpx2`. `anthropic` 1.x also removed `temperature`, `top_p` and `top_k`
  from its signatures; the adapter now sends them in the request body, so they keep working on the
  Claude models that accept them (4.6 and earlier) and are still dropped where the API rejects
  them. Other floors rose to versions that install on Python 3.13:
  `pyyaml>=6.0.2` and `tiktoken>=0.11`. The old floors (`anthropic>=0.40`, `openai>=1.50`,
  `pyyaml>=6.0`) were never tested, and `pyyaml` 6.0 does not build on Python 3.13. A new CI job
  installs the lowest version of every direct dependency and runs the suite, so the declared
  floors are versions the tests have run against. Projects pinned to `anthropic` 0.x or `openai`
  1.x/2.x must upgrade those SDKs to take this release.
- **Breaking: keys from the environment only go to the provider's own API.** `OPENAI_API_KEY`,
  `ANTHROPIC_API_KEY`, and `MODEL_API_KEY` are sent to `api.openai.com`, `api.anthropic.com`, and
  `api.meta.ai`. A remote `base_url` on any other host — a gateway, a proxy, another vendor's
  OpenAI-compatible server — now needs `api_key=` and raises `ValueError` without it; it used to
  receive the environment key (the OpenAI key went to any OpenAI-compatible server).
- **`RetryConfig` retries network failures.** A request that gets no HTTP response (refused or
  dropped connection, timeout) is retried and falls back: the OpenAI, Meta, Anthropic, and Gemini
  adapters raise it as `ConnectionError` or `TimeoutError` instead of the SDK's own exception,
  which `LLM` neither retried nor treated as a provider error.
- The `dev` extra includes `pytest-timeout`, so `@pytest.mark.timeout` limits are enforced.
- **Breaking: step traces record key names, not values, by default.** `StepTrace.input_state` is
  empty and `output_result` drops the artifacts; `input_keys` and `output_keys` list what the step
  read and returned. A long agent loop's trace no longer grows with the square of its steps.
  `Flow(trace_capture="full")` records deep copies of the values instead.
- **Breaking: every tool in `toolkit.tools.dangerous` requires approval.** `read_file`,
  `list_directory`, and `search_files` declare `capability="filesystem"`; `http_get` and
  `scrape_text` declare `capability="network"`; all are `risk_level="high"`. Without an
  `approval_handler`, a call returns `approval_denied`.
- **Breaking: system prompts are merged, never replaced.** `system=` comes first, then the
  `system()` messages in order, separated by a blank line (Anthropic, Gemini, and xAI, batch
  included). The OpenAI adapter sends `system=` as a leading message and keeps each `system()`
  message at its position. `system=""` no longer hides `system()` messages.
- **Breaking: an argument that cannot be coerced to its type is a `validation_error` before the
  tool runs.** It used to reach the function (`"x"` for an `int`); `"3"` is now coerced to `3`.
- **Breaking: `run_tools()` / `run_tools_sync()` refuse `approval_handler=` together with a
  `ToolGroup`** (`ValueError`); set it on the group. A plain list of callables still takes it.
- **Tool parameters typed `Any` or `object` accept any JSON value.** Their schema is now empty
  instead of `{"type": "string"}`, so the model can send objects, lists, and numbers. Unannotated
  parameters and types the generator does not know are still described as strings.
- `Flow.iter()`, `iter_flow()`, and `Agent.iter()` return execution objects that still work with
  `async for`; `Flow.iter_sync()` returns a `SyncFlowExecution`. The run's `MeterScope` is no longer
  stored in `State.world["_meter_scope"]`.
- `Flow.as_step()` no longer copies the flow's policy onto the wrapping step; the nested run applies
  its own policy and timeout.
- A stream's metering operation is reserved when the stream is created and starts with its first
  attempt. A stream that is never iterated, or that middleware rejects, releases the reservation.
- `configure_sync_timeouts(stream_join_timeout=)` (`AI_ARCH_STREAM_JOIN_TIMEOUT`) also bounds the
  wait for the worker thread of an abandoned sync stream or a timed-out sync call, with a warning if
  it is still alive. Closing a sync stream early can block for up to that long (5 s by default).
- **Bundled model catalog and pricing refreshed from all four providers' official catalogs
  (2026-08-29).** OpenAI gains current GPT-5.6 prices and the base/Cyber/Daybreak/
  `chat-latest` aliases; Anthropic gains permanent Sonnet 5 pricing and published Opus 4.8
  fast rates; Gemini gains 3.7 Flash and current 3.6 promotional pricing; xAI gains current
  Grok aliases and correct Build/Batch prices. Retired Claude Opus 4.1 and Gemini 2.0 entries
  were removed. The pricing engine now combines batch/fast service modes with long-context
  and cache tariffs, including xAI's inclusive 200k boundary, instead of applying standard
  cache prices or ignoring the long-context tier in those combinations.
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
- CI hardened: `lint` (ruff check + format), `typecheck` (pyright — now blocking, driven to 0 errors), and `test` across ubuntu + macos × Python 3.13/3.14 with coverage; `uv lock --check` enforces lockfile consistency; an examples smoke test catches public-API drift. CI never calls paid provider APIs: the test job deselects `live_api` tests, and there is no live-API workflow; run `pytest -m live_api` locally.
- `[tool.ruff]` and `[tool.pyright]` both exclude `src/ai_arch_toolkit/nanope` (sub-projects have their own idioms).
- Pricing registry refreshed (2026-05-19): Claude 4.7 added; Claude 4.6/4.7 now ship with 1M context at standard rates (long-context tier removed).
- Examples 31–33 updated to the variadic `Flow(*steps)` API and async `flow.run(state)`.
- Docs restructured into per-subsystem pages with coverage gaps closed; added the code-style guide, the Agent & ReasoningSpec page (now surfaced as the recommended entry point across README and docs), and the prompt-system suite (templates, layouts, manifests, messages, migration, extensibility).
- Docs and example index aligned with the Flow-based architecture; legacy "pipelines" and "8 agent architectures" wording removed.
- Memory `Node` reconciled with `core.graph.Node[T]` (zero pyright ignores).
- Public API surface tightened: `__init__` re-exports audited so internals stop leaking.
- Sync timeouts in `core/_sync.py` no longer expose the dead `SYNC_TIMEOUT` / `STREAM_JOIN_TIMEOUT` aliases; use `configure_sync_timeouts()` instead.
- `uv lock --upgrade` brought every transitive dependency to its latest compatible version (pydantic 2.13, urllib3 2.7, requests 2.34, websockets 16, xai-sdk 1.12, ruff 0.15.13, …); resolved the four Dependabot alerts.

- **Breaking:** tools time out after 120 seconds and their results are cut at 200,000 characters
  by default. Declare `timeout_s=None` or `max_output_chars=None` on a tool that needs more.
- **Breaking:** on the sync path (`ToolGroup.execute`, `execute_tool`, `run_tools_sync`) a
  synchronous tool runs in a worker thread of its own, as on the async path, so its timeout holds.
  A tool that relies on the caller's thread (thread-local state, a SQLite connection opened there)
  must open those resources itself.
- **Breaking:** `ip_lookup` queries ipwho.is over HTTPS instead of ip-api.com over HTTP, whose free
  endpoint has no HTTPS and forbids commercial use. The output lines are the same.
- The network tools go only over HTTPS to their module's hosts, follow a redirect only on the same
  host (never down to HTTP), read at most 10 MB within their deadline, send one User-Agent
  (`ai-arch-toolkit/1.0 (https://github.com/ai-arch-toolkit)`), and report a 429 as "rate limited
  by <API> (HTTP 429). Try again later.". Where a tool built its query by hand, spaces are now sent
  as `+`.
- `http_get` and `scrape_text` refuse URLs with credentials, stop at a redirect to another host
  (the message names the target, so the model can ask for it under a new approval), and clamp
  `max_chars` to 1–100,000; `http_get` reads no more of the body than it can return.
- `regex_search` refuses back-references, groups that repeat while holding a quantifier or an
  alternation, patterns over 500 characters and texts over 20,000, and lists at most 1,000
  matches. `math_eval` refuses expressions over 1,000 characters, and results too large to print
  (`factorial(3000)`, `7**9000`) before computing them.
- The filesystem tools read at most 100,000 characters (`read_file`) or 1,000,000 (per file in
  `search_files`, and `csv_read`), and clamp their size arguments; `search_files` cuts matching
  lines at 300 characters and `list_directory` lists at most 1,000 entries. `run_command` clamps
  `timeout` to 1–600 seconds and `max_output` to 1–100,000.
- Every toolkit tool declares its `capability` (`network` or `compute`; the dangerous tools keep
  theirs).
- `reverse_geocode` shares Nominatim's one-request-per-second clock with the `osm_*` tools; it had
  none.

- **Breaking:** the steps of a parallel DAG wave read the snapshot taken when the wave starts,
  instead of a deep copy of the state each. Siblings still never see each other's artifacts. A step
  that mutates a value it read in place (instead of returning an artifact) now changes the state in
  every mode; in a parallel wave the change used to be lost. Long DAG runs no longer deep-copy the
  state for every parallel step.
- In a parallel wave, the trace records each step when it finishes, in the order of the
  `step_end` events, instead of in declaration order. The wave's artifacts are merged, in
  declaration order, before the last step's `step_end`.

- **Breaking:** `ReasoningSpec.knobs` and `llm_kwargs` are copied and frozen at construction
  (read-only mappings). Writing to them raises `TypeError`, a change to the dict passed in no longer
  reaches the spec, and a spec can no longer be deep-copied or pickled (use
  `dataclasses.replace`).
- `StateSnapshot` is documented as what it is: a read-only view whose layers are copied and whose
  values are shared with the state. A step returns changes as artifacts; one that mutates a value
  it read changes the state for every step after it. See
  [docs/flow-architecture.md](docs/flow-architecture.md#statesnapshot).
- The nine flow factories take `timeout`, `trace_capture`, `policy`, and `budget_policy` as
  `**options: Unpack[FlowOptions]`. Calls are unchanged, and type checkers still check each name
  and type; `inspect.signature` shows `**options` in place of the four parameters.
- **Breaking:** an agent's answer (`AgentResult.text`, `extract_text`) is the text its flow leaves
  under `"answer"`; a flow that leaves none answers with its last step's value, as a nested flow
  does. The answer no longer falls back to `"response"`, `"last_response"` or `"last_answer"`, and
  `AgentResult.response` comes from the same place: `"response"` next to an answer, otherwise the
  last step's value when it is a `Response` (was `None`). Every built-in strategy leaves both keys,
  also when it runs out of turns: ReAct writes `"answer"` each turn, Reflexion writes both after
  every evaluation, and Generate-Review writes `"response"` when it rejects a draft too. An empty
  answer is now the answer: before, a Reflexion that passed with an empty answer answered with its
  score, and a ReAct run that ended on a tool turn answered with the tool results. See
  [docs/agents.md](docs/agents.md#escape-hatch-agentfrom_flow).
- **Breaking:** agent and prompt manifests are checked against one declared shape each, and the
  packaged prompt JSON Schema is generated from it. It is now as strict as the loader: it used to
  leave the fields of `source`, `template`, `layout`, and the selectors open. Shape errors name
  the field's path, and an unknown field gets a suggestion (`strategy.max_iterations must be a
  positive integer`, `unknown fields: 'stratgey' (did you mean 'strategy'?)`), so their wording
  changed. A few values that used to pass are refused:
  - a `strategy.system` that is not text (it became its `str()`);
  - `version: true`;
  - a boolean as a section's `order`;
  - an XML layout's `metadata_attributes` that is not a list;
  - `select` or `serialize_as` on an inline template (they were ignored).

### Fixed
- The bundled Reflex frontend now uses Reflex 0.9.11 and a security-audited dependency lock,
  removing current PostCSS, React Router, browser-tooling, Nano ID, and Socket.IO advisories.
- Filesystem tools preserve missing, denied, wrong-kind, and other OS error distinctions on Python
  3.14, whose `pathlib` status-query methods now suppress every `OSError`.
- Failed calls no longer poison enforcing scopes when their cost can be bounded. Rate-limit failures are unbilled; indeterminate failures consume a separate cap allowance, so retries, fallbacks and later steps can proceed within the remaining budget.
- Cancellation while waiting for an inference slot releases admission without counting or pricing a call. Streams release their slot before yielding to the consumer, and abandonment closes all delegated transport iterators immediately.
- Attempt history retains failed intermediate fallback models. Settlement still precedes async after hooks, and sync stream cleanup remains safe across threads.
- Strict budgets reserve custom-priced tools and deny unknown or raising prices before execution. Soft budgets admit server tools and fail closed on subsequent work if settlement is unpriced.
- Exact Fable/Mythos 5.1 prices use the published cache rate; Gemini 3.8 Flash gains its own promotional token and batch prices.
- xAI forwards the configured timeout to the SDK instead of ignoring it.
- Anthropic document blocks send their label as `title`, matching the SDK request schema.
- Default retry includes Anthropic overload responses (HTTP 529).
- Flow iteration emits `step_end` for completed steps before wall-budget denial in sequential and single-step DAG waves.
- `ip_lookup` rejects empty or invalid addresses before I/O instead of querying the host machine IP.
- **An empty `ToolGroup` is a value, not an absence.** `Agent(spec, llm, ToolGroup())` used to swap
  the empty group for a new one, so tools added to it later (`group.add(...)`) were unknown to the
  agent. An empty per-phase group (`executor_tools`, `solver_tools`, `rollout_tools`) used to fall
  back to the agent's main tools in `plan_execute`, `reflexion`, `llm_compiler`, `self_discovery` and
  `lats` — a phase declared without tools received all of them. Both now keep the group they were
  given.
- **`LLM(fallback=other)` no longer modifies `other`.** Building the parent emptied the nested
  LLM's own fallback chain and took ownership of the fallbacks it had created, so `other` stopped
  falling back when used on its own and could not be shared by two parents. The parent still walks
  one flat chain (a model reachable twice is tried once); each LLM closes only the fallbacks it
  created from strings.
- **`null` is refused for a tool parameter that does not admit `None`.** Argument validation let
  `None` through for every parameter, so `width: int` received `None`. It is now accepted only with a
  `None` default, an annotation that includes `None`, or no usable annotation; otherwise the call
  fails with `validation_error` before any gate or approval.
- **`ReasoningSpec.from_mapping` drops nothing silently.** A `policy` given as a mapping, a
  malformed `output_schema` and unknown keys were ignored. `policy` now builds a `Policy` from a
  mapping of its fields (`retry` from a mapping of `RetryConfig` fields); an unknown key or an
  unusable value raises `ValueError` naming it.
- **OpenAI structured output accepts Pydantic models again.** `output_schema=Model` was sent in
  strict mode as `model_json_schema()` produced it, and OpenAI answered 400 (`'additionalProperties'
  is required to be supplied and to be false`). Strict schemas are now normalized as the SDK's own
  `parse()` helpers do: every object is closed and lists all its properties as required,
  `default: null` is dropped, and a `$ref` with sibling keys is inlined.
- A tool parameter typed `X | None` without a default can be omitted by the model: the schema
  already made it optional, and the call now receives `None` instead of failing with
  `validation_error`.
- `ToolGroup` and `run_tools()` accept a `functools.partial`: it is named and described by the
  function it wraps and takes the arguments the partial leaves open. It used to raise
  `AttributeError`.
- `ApprovalDecision.approve(modified_args={})` runs the tool with no arguments; the empty dict was
  treated as "no change" and the model's arguments ran.
- **`cache()` parts reach OpenAI, Gemini, and xAI as their text.** Those adapters sent the part's
  Python repr (`CachePart(content='…', ttl='ephemeral')`) to the model. xAI also drops document
  parts with a warning instead of sending their repr.
- **A system message may hold text parts.** A list of strings and `cache()` parts is joined into
  the system prompt, and Anthropic keeps the cache marker, so a system prompt can be cached. Such a
  list used to raise `TypeError` in the Anthropic, Gemini, and xAI adapters and failed inside the
  OpenAI SDK. An image or document in a system message raises a `TypeError` that points to a user
  message. On Anthropic, native blocks passed as `system=` no longer drop the text of `system()`
  messages.
- Sync streams (`LLM.stream_sync()`, `LLM.stream_events_sync()`, `Flow.iter_sync()`) deliver an
  exception object that the source yields as a value, instead of raising it.
- nanope (work in progress): `--allow-dangerous-tools` approves the dangerous tools' calls, the
  BBEH solvers approve `python_repl`, and the research manager approves `http_get` and
  `scrape_text`. Every call to those tools was denied, because they require approval.
- **Anthropic streaming no longer double-counts usage.** `message_delta` usage is cumulative and now
  replaces the `message_start` counts instead of adding to them: input and cache tokens were metered
  twice, tripping token and cost budgets early, and a delta without `input_tokens` raised
  `TypeError`. `complete()` and batch results also record `null` token counts as 0.
- **`Flow(policy=...)` applies to every step without a policy of its own**, also when the flow runs
  directly. `ReasoningSpec.policy`, `ReasoningSpec.timeout`, and manifest `limits.timeout_seconds`
  were inert, and the flow factories dropped `timeout` when given a `policy`.
- **Async middleware runs on streams.** `abefore` / `aafter` run for `stream()`, `stream_events()`,
  and their sync wrappers, so moderation, memory, and rate limiting are no longer skipped when
  streaming. Fallback models receive the messages after middleware, in `complete()` and in streams.
- **`run_tools()` / `run_tools_sync()` apply a `ToolGroup`'s governance** (its gates, approval
  handler, and `max_calls`), which was silently dropped. They also check every tool name in the
  response before running any call, so an unknown name no longer leaves earlier side effects
  without results.
- Anthropic `batch_submit` no longer drops `system()` messages, and middleware that sets
  `request.system` (such as `MemoryMiddleware`) no longer erases them.
- `ToolGroup(web_search())` and `group.add(code_execution())` raise a `TypeError` explaining that
  server tools go next to the group (`tools=[group, web_search()]`), instead of an
  `AttributeError`; other non-callables raise `TypeError` too.
- `ToolGroup.execute()`, `execute_tool()`, and `run_tools_sync()` run `async def` tools to
  completion instead of returning an un-awaited coroutine as a success, and both execution paths
  await an awaitable returned by a sync function. Tools with positional-only parameters can be
  called.
- **Flow engine.** `Flow.iter()` runs DAG waves in parallel and isolates siblings like `run()`. A
  `when` condition or `Scope` callable that raises stops the flow with the error recorded on that
  step, instead of escaping `run()` or silently skipping it. A budget denial in a parallel wave keeps
  the finished siblings in the trace and state, and a step's `Policy(max_cost=...)` counts the spend
  of flows that step runs.
- `OperationRequest.provider` and `UsageEvent.provider` name the adapter that served the call,
  fallbacks included.
- Tool schemas: multi-type unions become a required `anyOf` (optional only with `None`), identical
  members collapse, `*args` / `**kwargs` never enter the schema, and PEP 695 type aliases produce
  their target type's schema.
- A `GateModify` from one gate reaches the next gates and the approval request, and a `TypeError`
  raised inside a tool's body is a `runtime_error` instead of a `validation_error`.
- `Scope.enrich` reads the snapshot the step will see, already filtered and transformed, so it can
  no longer read keys the scope excludes.
- Sync calls made while an event loop is running (`LLM.complete_sync()`, `Flow.run_sync()`,
  `Agent.run_sync()`, …) cancel their coroutine when the sync timeout expires, so the call no longer
  completes, and settles its meter, after the caller has moved on. Sync streams buffer at most 256
  items ahead of a slow consumer, and stopping early cancels the pending read.
- With `trace_capture="full"`, and in `Trace.initial_state`, a step that mutates a value in place no
  longer rewrites earlier records.
- The Gemini adapter no longer fails on tools with `tuple` parameters or schemas with `$defs` /
  `$ref`; they are sent as JSON Schema through `parameters_json_schema`.
- **Nested Pydantic models as tool parameters produce a valid schema.** The model's `$defs`
  table was embedded inside the parameter, so its `#/$defs/...` pointers dangled at the tool's
  root; Gemini rejected such tools with `400 reference to undefined schema`. Nested models are
  now inlined, and a recursive model's `$defs` table is hoisted to the root.
- **Flow timeouts:** a timeout during a parallel DAG wave keeps the siblings that already
  finished (results, trace and state), like a budget denial does; a steady stream of policy events
  can no longer postpone the deadline; and the steps in flight are cancelled before the `timeout`
  event is delivered.
- A cyclic flow with `max_iterations=0` runs no pass again, and the run's meter scope is closed
  even if a second cancellation interrupts the run's cleanup.
- `ReasoningSpec` rejects an unknown `trace_capture` when it is built, and retry backoff no longer
  raises `OverflowError` after about a thousand attempts. `SyncFlowExecution` is exported next to
  `FlowExecution` and works as a context manager.
- The docs no longer claim that `break` stops a flow or agent run: a held execution keeps its step
  running until `aclose()` or the end of an `async with` block.
- **A stream is admitted and priced on the request after async middleware.** Its reservation
  was built from the request before `abefore` ran, so a middleware that added a server tool,
  changed `max_tokens` or injected content left the budget admitting and pricing stale facts.
  The reservation is now replaced before the first attempt when those facts change.
- Fallbacks receive the tools and the kwargs that middleware changed, not only the messages and
  system prompt, in `complete()` and in streams; each fallback keeps its own defaults. A stream
  rejected by middleware records no `StreamAbandoned` attempt, since no provider was called.
- Argument validation no longer raises for an integer string longer than Python's conversion
  limit, a `null` schema branch accepts only `null`, and a custom gate that returns non-mapping
  arguments gets a `validation_error`. `Literal[True, False]` and boolean enums are described as
  booleans, so their values are accepted.
- The Gemini adapter returns an empty response, instead of raising `TypeError`, when a candidate
  cut off by `max_tokens` while thinking carries no content parts.
- Provider reasoning usage is now normalized without double-counting: xAI completion plus
  reasoning tokens and Gemini candidate plus thought tokens become inclusive billable
  `output_tokens`, while OpenAI and Anthropic keep their already-inclusive output totals. Gemini
  tool-use prompt tokens are also counted as input. xAI now prefers the exact per-request
  `cost_in_usd_ticks` charge for response and meter costs, falling back to the local pricing
  registry only when the provider omits it. OpenAI-compatible responses also recover separately
  reported generated tokens from `total_tokens` when it exceeds prompt plus completion.
- **`grok-4.6` is priced at its own rates.** It matched the `grok-4` entry and was estimated at
  $1.25/$2.50 per million input/output tokens instead of $2.00/$6.00 ($0.50 cached input; all
  three double at or above 200k prompt tokens).
- `generate_review` now forwards `ReasoningSpec.output_schema` to its generator while keeping the
  reviewer on its plain-text `ACCEPT` / `RETRY` protocol. Its strategy metadata and the Nanope
  configurable-agent validation now advertise the same support, and Nanope no longer injects
  ReAct-only knobs into other strategies.
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

- The network tools no longer raise on a response of an unexpected shape (a JSON array or `null`
  where an object was expected, a `null` field, undecodable bytes): they return "…: could not parse
  API response: …". 93 of them raised before.
- Tools no longer raise on hostile arguments: very long paths in the filesystem tools, dates past
  the year 9999 in `date_add` and `timezone_convert`, deeply nested JSON in `json_extract`, an empty
  or absolute glob in `list_directory`, a null byte in `run_command`, network errors in the
  `youtube_*` tools.
- `math_eval("9**9**9")`, `factorial(10**7)`, `round(5, -10**9)` and `regex_search` with
  `(a+)+$` return an error at once instead of holding the process.
- An argument can no longer climb out of an API's path: `..` in `country_info`, `define_word`,
  `europe_pmc_citations` and every other path segment is refused.
- Rate-limit throttles are safe with parallel tool calls, which could skip the wait.
- A `null` field in an agent manifest means "not set" everywhere, as the loader already let it.
  Before:
  - `strategy: null` failed in `reasoning_spec()`;
  - `strategy.name: null` asked for a strategy named `"None"`;
  - `strategy.max_iterations: null` raised `TypeError`;
  - `strategy.system: null` became the prompt `"None"`;
  - `limits.reserve: null` failed in `budget_policy()`;
  - `override_policy: null` failed at load.
- A list or an object as an agent manifest's `strategy.trace_capture` fails with
  `AgentManifestError` instead of `TypeError`.
- `python_repl` refuses what it used to get wrong in silence:
  - `{**a}` (it evaluated to `{None: a}`);
  - `f(**d)` (`d` was dropped);
  - `del x.attr` (ignored).
- `python_repl`'s `and`/`or` stop at the operand that decides, as Python's do.
- `python_repl`'s `x **= n` has the same exponent limit as `x ** n`.
- **`lats` branches.** Each rollout added one child to a node that had none, so the search tree
  was a single chain of attempts and `n_candidates` was ignored. A node now takes up to
  `n_candidates` sibling attempts before the search goes below it, and UCT chooses among them, as
  in Zhou et al. 2024 (LATS). `max_rollouts` still caps the total number of ReAct attempts. The
  docstring now warns that rollouts re-run their tools, so tool side effects repeat.

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
