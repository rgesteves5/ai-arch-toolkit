# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
uv sync --extra dev                      # Install all dev dependencies
uv run pytest                            # Run full test suite
uv run pytest tests/test_content.py      # Run a single test file
uv run pytest -k "test_chat_basic"       # Run tests matching a pattern
uv run ruff check src tests examples     # Lint
uv run ruff check --fix src tests        # Auto-fix lint issues (import sorting, etc.)
uv run ruff format src tests examples    # Auto-format
uv run python examples/01_hello_world.py # Run an example (needs API keys in .env)

# Documentation
uv sync --extra dev --extra docs         # Install docs dependencies
uv run mkdocs serve                      # Local docs server
uv run pdoc ai_arch_toolkit -o site/api  # Generate API docs
```

Running examples requires API keys. Load them with `set -a && source .env && set +a` or see `docs/uv-guide.md`.

## Architecture

Two layers under `src/ai_arch_toolkit/`, plus the `ai-arch` CLI entry point and the WIP `nanope/` sub-package:

```
ai_arch_toolkit/
├── core/              # Stateless async-first foundation
│   ├── _llm.py        # LLM facade
│   ├── _providers/    # Anthropic, OpenAI, xAI, Gemini
│   ├── _tools/        # @tool decorator, ToolGroup
│   ├── _metering/     # Neutral cost/usage mechanism (Money, Cost, MeterStore, MeterScope)
│   └── graph/         # General-purpose graph: Node[T], Edge, Graph, protocols
├── toolkit/           # Convenience utilities built on core/
│   ├── agents/        # Agent + ReasoningSpec facade; 9 agent architectures as Flow factories
│   │   └── flows/     # react_flow, reflexion_flow, rewoo_flow, etc.
│   ├── flow/          # Flow orchestration (Flow, FlowStep, FlowResult, FlowEvent)
│   ├── budget/        # Budget policy over the meter (BudgetPolicy, BudgetController, BudgetReport)
│   ├── tools/         # ~130 pre-built tools across 25 domains (+ opt-in dangerous/)
│   ├── memory/        # Graph-backed memory for agents (GraphStore, views, search)
│   ├── resources/     # Reusable loaders, codecs, selectors, serializers, policies
│   ├── knowledge/     # Resource-backed registry for reference data
│   ├── prompts/       # Sections, templates, layouts, manifests, rendering
│   └── moderation/    # LLM/OpenAI moderators + ModerationMiddleware
├── nanope/            # WIP sub-projects (Reflex app; excluded from ruff/pyright)
├── _cli.py            # `ai-arch` CLI (prompt validate / inspect / render)
└── __init__.py        # Re-exports from core/ + toolkit/
```

Layer rules: `core/` is the neutral mechanism layer — stateless, zero dependencies, zero opinions; it never imports `toolkit/`. `toolkit/` is the opinionated convenience layer built on top of core. `nanope/` is a scratchpad of WIP sub-projects (e.g. a Reflex app with its own idioms) — not part of the public API and excluded from repo-wide ruff/pyright.

### Core layer (`core/`)

The stateless, async-first foundation. All new code should build on this.

- **`_llm.py`**: `LLM` class — user-facing facade. `complete()` / `stream()` / `stream_events()` (async) with `complete_sync()` / `stream_sync()` / `stream_events_sync()` wrappers. Accepts `Content` (str or multimodal parts). Stream methods support fallback + middleware.
- **`_content.py`**: Message constructors (`user()`, `assistant()`, `system()`, `tool_result()`) and multimodal types (`ImagePart`, `DocumentPart`, `CachePart`). `type Content = str | list[ContentPart]`.
- **`_response.py`**: `Response`, `Usage`, `ToolCall`, `ThinkingBlock`, `Citation`, `OutputSchema`, `StreamResponse`, `SyncStreamResponse`, `StreamEvent`, `RichStreamResponse`, `SyncRichStreamResponse`.
- **`_providers/`**: `BaseProvider` ABC → `AnthropicProvider`, `OpenAIProvider`, `XAIProvider`, `GeminiProvider`. Factory: `create_provider()` routes by model prefix (`claude-` → Anthropic, `gpt-`/`o1-`/`o3-`/`o4-` → OpenAI, `grok-` → xAI, `gemini-` → Gemini). `provider=` forces an adapter (bypasses prefix detection); an unknown model with `base_url=` set falls back to the OpenAI-compatible adapter (Ollama, LM Studio, vLLM). The API key is required unless `base_url` points at a loopback host (localhost/`127.x`/`::1`), where local servers ignore it and a cloud env key is not forwarded; remote endpoints still fail fast on a missing key. `_normalize_fallbacks` routes string fallbacks by their own name — a recognizable model fails over to its own provider, a bare tag inherits the parent's connection.
- **`_tools/`**: `@tool` decorator (auto-generates JSON Schema from type hints + Google-style docstrings; also carries governance metadata — `capability`, `risk_level`, `requires_approval`). `ToolGroup` (collection with execute/async_execute returning a structured `ToolResult`). Governance pipeline: `ToolDefinition`/`ToolSchema`/`ToolRuntimePolicy`/`RiskLevel` (`_definition`), gates `DangerousToolGate`/`ApprovalGate`/`DryRunGate` (`_governance`), `ApprovalRequest`/`ApprovalDecision`/`ApprovalHandler` (`_approval`), `execute_tool`/`async_execute_tool` (`_executor`), `ToolResult`/`ToolError` (`_result`).
- **`_pricing.py`**: `PricingRegistry` with `_default_pricing.toml`. Access via `pricing` singleton; `pricing.price(request, usage) -> Cost` is the default `Pricer` for the meter.
- **`_metering/`**: the neutral cost/usage **mechanism** (0% opinion). `Money` (opaque exact pico-USD int), `Cost` (one class, `kind` known/estimated/unknown — `unknown` ≠ `known($0)`), `OperationRequest` (pure facts), `MeterStore` (single-writer, `threading.Lock`, reserve→settle escrow, per-span accounting, TOCTOU-re-validated admission run OUTSIDE the lock), `MeterScope`/`RunConfig` + `current_meter`/`bind_meter` (ContextVar ambient binding), `UsageEvent`/`UsageSink` (audit, redacted, built-under-lock/emitted-outside), `AdmissionController` Protocol + `AdmissionDenied` (neutral, terminal). Three modes: no scope = unmetered; scope + `controller=None` = measure-only (Flow/Agent default); scope + controller = measure+enforce. Charge sites: `LLM.complete/stream/stream_events` (per provider attempt) + the common tool executor. See `docs/internal/metering-plan.md`.
- **`_policy.py`**: `Policy` (per-step retry/timeout/confidence/cost) with declarative callbacks `on_timeout`/`on_low_confidence`/`on_exhausted`.
- **`_redaction.py`**: `RedactionPolicy`/`RedactionMode`/`Redactor` + `redact()`/`redact_text()` — strips secrets from traces and tool results by key name and value pattern.
- **`_moderation.py`**: `Moderator` Protocol, `ModerationResult`, `ModerationError` (toolkit moderators build on these).
- **`_sync.py`**: `_run_sync()` and `_stream_sync()` helpers used by LLM and agents.
- **`_middleware.py`**: `Middleware` Protocol with `before`/`after` hooks, `Request` dataclass.
- **`_retry.py`**: `RetryConfig` + `with_retry()` for exponential backoff.
- **`_server_tools.py`**: `ServerTool`, `code_execution()`, `web_search()` for provider-hosted tools.
- **`graph/`**: General-purpose graph layer. `Node[T]`, `Edge` (frozen dataclasses), `Graph` facade (async-first with `_sync` wrappers). Protocols: `GraphBackend` (storage), `GraphAlgorithms` (BFS, DFS, shortest path, PageRank, etc.). Default backend: `NetworkXBackend` (import-guarded, requires `[graph]` extra). `Graph` exposes: node/edge CRUD, `has()`, `degree()`, `node_count()`, `edge_count()`, `list_edges()`, `filter_nodes()`, `filter_edges()`, `get_orphan_nodes()`, `get_stats()`, `copy()`, traversals (BFS, DFS, shortest path, find_all_paths, ancestors, descendants, ego_graph, PageRank, centrality, connected components, subgraph), persistence (`save`/`load`/`to_dict`/`from_dict`).

### Toolkit layer (`toolkit/`)

#### Agent & ReasoningSpec (`toolkit/agents/`)

The recommended user-facing entry point (see `docs/agents.md`). `ReasoningSpec` (`_spec.py`): frozen, serializable description of how an agent reasons — `strategy`, `system`, `max_iterations`, `knobs` (strategy-specific options), `policy`, `timeout`, `llm_kwargs`, `output_schema`; `from_mapping()` builds one from parsed JSON/YAML. `Agent` (`_agent.py`): binds a spec to an `LLM` + `ToolGroup`, compiles the `Flow` once via `build_flow()`, and exposes `run()` / `run_sync()` / `iter()` (each accepts a per-run `budget_policy=`), `Agent.from_flow()` (wrap a hand-built Flow), and `as_step()` (compose into a larger Flow). `AgentResult`: `text`, `response`, `flow_result`, meter-derived `usage`/`cost`/`report`, `errors`. Strategy registry (`_builders.py`/`_compile.py`): `register_strategy()` / `get_strategy()`, `FlowStrategy`, `StrategyBuilder`, `BuildContext` — 10 built-in strategies (the 9 flow factories below + `completion`, a single LLM call with no tool loop); only `react`/`completion` support `output_schema`.

#### Agent Flows (`toolkit/agents/flows/`)

Nine agent architectures as **Flow factories** built on core/ primitives (`LLM`, `ToolGroup`, `State`, `Step`, `Result`). Each factory returns a `Flow` and has a companion `*_initial_state(task)` helper.

- **`react_flow()`**: Cyclic LLM → tool execution loop. Params: `system`, `max_iterations`, `parallel_tool_calls`, `timeout`, `policy`, `llm_kwargs`.
- **`reflexion_flow()`**: Inner ReAct with evaluate + reflect retry. Requires `evaluator: Callable[[str, str], float]`, `threshold`, `max_retries`.
- **`rewoo_flow()`**: Plan with `#E{n}` placeholders → Execute tools → Solve. Three-phase.
- **`plan_execute_flow()`**: Numbered plan → per-step ReAct → Solve. `max_replans`.
- **`tot_flow()`**: Tree of Thoughts with DFS/BFS search. `n_candidates`, `max_depth`, `strategy`.
- **`lats_flow()`**: Language Agent Tree Search (MCTS). `n_candidates`, `max_rollouts`, `exploration_weight`.
- **`self_discovery_flow()`**: Select reasoning modules → Adapt → Operationalize → Solve via inner ReAct.
- **`llm_compiler_flow()`**: Plan DAG → Parallel execute → Join. `max_replans`.
- **`generate_review_flow()`**: Generator → reviewer retry loop, with optional tools in both phases.
- Task input accepts `Content` (str or multimodal list) for vision+tools use cases.

Usage pattern — high level (preferred) and the flow-factory equivalent:
```python
agent = Agent(ReasoningSpec(strategy="react"), llm, tools)
result = agent.run_sync("your task")  # result.text / result.cost / result.report

flow = react_flow(llm, tools, max_iterations=5)
state = State(operational=react_initial_state("your task"))
result = flow.run_sync(state)
answer = state["response"].text
```

#### Tools (`toolkit/tools/`)

~130 pre-built tools across ~45 files (25 domains), all using stdlib only (zero pip deps) and the `@tool` decorator from core/. Domains include datetime/math/text/JSON-CSV utilities, weather + air quality, geo + OpenStreetMap, reference (Wikipedia, Wikidata, dictionary, news), scholarly (arXiv, PubMed, Europe PMC, Semantic Scholar, Crossref, ROR, DataCite, Open Library), biomedical/chemistry (UniProt, PDB, ChEMBL, RxNorm/DailyMed, ClinicalTrials), and public data (GBIF, Open Food Facts, openFDA, World Bank, WHO, Eurostat, USGS/EONET, NVD). The default `toolkit.tools` namespace is safe-by-default; filesystem/shell/Python/web-fetch tools are opt-in via `toolkit.tools.dangerous` and should be gated (see `docs/safety.md`). Full per-tool list: `docs/tools-catalog.md`.

#### Memory (`toolkit/memory/`)

Graph-backed memory for LLM agents, built on `core/graph/`. `GraphStore` wraps `Graph` with memory-specific `Node` (adds `timestamp`, `source`, `confidence`, `access_count`, `last_accessed`, `embedding`), keyword/vector search, and access tracking. Views: `TemporalView` (recent, since), `RelationalView` (neighbors, path), `PropertyView` (by_confidence, by_source, most_accessed), `SimilarityView` (vector search). `MemoryMiddleware` auto-injects relevant memories into LLM context. Presets: `conversational()`, `cognitive()`. `memory_tools()` generates `@tool`-decorated functions for agent use.

#### Knowledge (`toolkit/knowledge/`)

Sync registry for prompt-injectable reference data, built on `toolkit.resources`. `load()` / `from_directory()` preserve parsed data, fingerprints, and provenance. Query via `by_category()` / `by_tags()`. Legacy context and loader helpers remain compatible.

#### Resources (`toolkit/resources/`)

`Resource`, `ResourceRef`, `ResourceResolver`, and `ResourcePolicy` provide reusable local/package loading. Codecs cover text, Markdown, JSON, TOML, YAML, and bytes; selectors cover JSON Pointer, Markdown headings, line ranges, and named blocks.

#### Prompts (`toolkit/prompts/`)

Resolved literal `Prompt` / `PromptSection` plus `PromptTemplate`, typed variables, explicit template engines, Resource/Knowledge sources, versioned manifests, Text/Markdown/XML/JSON layouts, section spans, fingerprints, and provenance. `load_prompt()` is the manifest entry point (the `ai-arch prompt validate|inspect|render` CLI works on the same manifests); default rendering remains byte-compatible.

#### Moderation (`toolkit/moderation/`)

Content moderation built on the `core/_moderation.py` `Moderator` protocol. `OpenAIModerator` (free `omni-moderation-latest` endpoint), `LLMModerator` (any `LLM` as a classifier), and `ModerationMiddleware` (wires a moderator into the LLM middleware chain for input/output checks; `on_flagged` is `raise` or `warn`).

#### Budget (`toolkit/budget/`)

The **opinion** layer over the neutral `core/_metering` mechanism. `BudgetPolicy` (user-facing caps: calls/tokens/cost-USD/wall + `reserve` mode) → `to_limits()` compiles to core `ResourceLimits`. `BudgetController` implements the core `AdmissionController` (pure/sync `admit`); `BudgetExceeded(AdmissionDenied)`; `HeuristicEstimator` (strict-reserve, fails closed on unpriced models); `BudgetReport.from_snapshot()` (projection). Wire in via `RunConfig(controller=BudgetController(policy))` — a `Flow` builds this from its `budget_policy=` and enforces at the charge site; nested agent flows inherit the enclosing scope (one cumulative budget).

#### Runner

`_runner.py`: `run_tools()` / `run_tools_sync()` convenience wrappers — route through the common governed + metered tool executor (approval gate + metering), never the raw function.

**Import convention**: New code should import from `ai_arch_toolkit.core` or `ai_arch_toolkit.toolkit.agents`.

## Testing Patterns

- **Config**: pytest-asyncio with `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed.
- **Test layout**: `tests/` (core tests), `tests/agents/flows/` (agent flow tests), `tests/toolkit/` (toolkit tool tests), `tests/graph/` (core graph tests), `tests/memory/` (memory tests), `tests/flow/` (flow tests), `tests/knowledge/` (knowledge tests), `tests/metering/` (core metering primitives + store + scope), `tests/budget/` (toolkit budget policy/controller), `tests/moderation/` (moderators), `tests/prompts/` + `tests/resources/` (prompt/resource system), `tests/integration/` (real-API tests, `integration` marker), `tests/nanope/` (WIP app).
- **Metering tests**: for a metered LLM/tool, use a REAL `LLM` with a fake `_provider` (so the charge site runs) — a mocked `llm.complete` bypasses metering. Bind a scope with `MeterScope(RunConfig(controller=BudgetController(policy)))`.
- **Core test fixtures** (`tests/conftest.py`): `MockResponse` class (mimics `requests.Response`), `mock_post` fixture, `weather_tool` fixture.
- **Agent test fixtures** (`tests/agents/conftest.py`): `make_response()`, `make_tool_call()` factories. Mock `LLM` with `AsyncMock`, set `llm.complete.side_effect` with pre-built `Response` objects.
- **Toolkit tests**: Mock `urllib.request.urlopen` for API tools. Use `tmp_path` for filesystem tools.
- **Patch paths**: Use the module where the symbol is imported, e.g. `@patch("ai_arch_toolkit.core._providers._openai.post_json")`.
- **SSE mocks**: prefix lines with `"data: "`. Gemini NDJSON: plain JSON strings.

## Code Conventions

- Python 3.13+, `from __future__ import annotations` in every file.
- Ruff line length: 99. Always run `ruff format` after edits.
- All dataclasses: `frozen=True, slots=True` (add `kw_only=True` for 3+ fields).
- PEP 695 `type` aliases: `type Content = str | list[ContentPart]`, `type StopReason = Literal[...]`.
- `__all__` in every `__init__.py`.
- Internal modules prefixed with `_`; public API via `__init__.py` re-exports only.
- Google-style docstrings — no type info repeated (type hints suffice).
- Toolkit tools return error strings (never raise) for graceful agent handling.
- Practical code-writing guidance lives in `docs/code-style.md`: linting shape, docstring/comment style, and class vs function decisions.

## Provider-Specific Gotchas

- **Anthropic**: `input_schema` for tools (not `parameters`), `system` is a top-level field (not a message role), supports extended thinking. Native structured output via `output_config` (not tool trick).
- **Gemini**: `contents`/`parts` structure (not `messages`/`content`), uses NDJSON streaming (not SSE).
- **OpenAI**: Chat Completions API only (no Responses API provider). The adapter also serves OpenAI-compatible servers (Ollama, LM Studio, vLLM) via `base_url=`; vendor reasoning deltas (`reasoning_content`/`reasoning`) surface as thinking events.
- **xAI**: Separate provider (not OpenAI-compat), API key via `XAI_API_KEY`.

## Research Docs

`research/` contains standalone Markdown reference guides (LLM API guide, agent architectures, Python best practices, graph algorithms). Separate from the Python package.
