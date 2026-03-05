# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
uv sync --dev                            # Install all dev dependencies
uv run pytest                            # Run full test suite
uv run pytest tests/test_content.py      # Run a single test file
uv run pytest -k "test_chat_basic"       # Run tests matching a pattern
uv run ruff check src tests examples     # Lint
uv run ruff check --fix src tests        # Auto-fix lint issues (import sorting, etc.)
uv run ruff format src tests examples    # Auto-format
uv run python examples/01_hello_world.py # Run an example (needs API keys in .env)

# Documentation
uv sync --group docs                     # Install docs dependencies
uv run mkdocs serve                      # Local docs server
uv run pdoc ai_arch_toolkit -o site/api  # Generate API docs
```

Running examples requires API keys. Load them with `set -a && source .env && set +a` or see `docs/uv-guide.md`.

## Architecture

Two layers under `src/ai_arch_toolkit/`:

```
ai_arch_toolkit/
├── core/              # Stateless async-first foundation
│   ├── _llm.py        # LLM facade
│   ├── _providers/    # Anthropic, OpenAI, xAI, Gemini
│   ├── _tools/        # @tool decorator, ToolGroup
│   └── graph/         # General-purpose graph: Node[T], Edge, Graph, protocols
├── toolkit/           # Convenience utilities built on core/
│   ├── agents/        # 8 agent architectures
│   ├── tools/         # 25 pre-built tools (weather, geo, news, etc.)
│   ├── memory/        # Graph-backed memory for agents (GraphStore, views, search)
│   ├── pipeline/      # Sequential phase execution with context accumulation
│   └── knowledge/     # Sync registry for prompt-injectable reference data
└── __init__.py        # Re-exports from core/ + toolkit/
```

### Core layer (`core/`)

The stateless, async-first foundation. All new code should build on this.

- **`_llm.py`**: `LLM` class — user-facing facade. `complete()` / `stream()` / `stream_events()` (async) with `complete_sync()` / `stream_sync()` / `stream_events_sync()` wrappers. Accepts `Content` (str or multimodal parts). Stream methods support fallback + middleware.
- **`_content.py`**: Message constructors (`user()`, `assistant()`, `system()`, `tool_result()`) and multimodal types (`ImagePart`, `DocumentPart`, `CachePart`). `type Content = str | list[ContentPart]`.
- **`_response.py`**: `Response`, `Usage`, `ToolCall`, `ThinkingBlock`, `Citation`, `OutputSchema`, `StreamResponse`, `SyncStreamResponse`, `StreamEvent`, `RichStreamResponse`, `SyncRichStreamResponse`.
- **`_providers/`**: `BaseProvider` ABC → `AnthropicProvider`, `OpenAIProvider`, `XAIProvider`, `GeminiProvider`. Factory: `create_provider()` routes by model prefix (`claude-` → Anthropic, `gpt-`/`o1-`/`o3-`/`o4-` → OpenAI, `grok-` → xAI, `gemini-` → Gemini).
- **`_tools/`**: `@tool` decorator (auto-generates JSON Schema from type hints + Google-style docstrings), `ToolGroup` (collection with execute/async_execute), `infer_schema()`, `prepare_tools()`.
- **`_pricing.py`**: `PricingRegistry` with `_default_pricing.toml`. Access via `pricing` singleton.
- **`_sync.py`**: `_run_sync()` and `_stream_sync()` helpers used by LLM and agents.
- **`_middleware.py`**: `Middleware` Protocol with `before`/`after` hooks, `Request` dataclass.
- **`_retry.py`**: `RetryConfig` + `with_retry()` for exponential backoff.
- **`_server_tools.py`**: `ServerTool`, `code_execution()`, `web_search()` for provider-hosted tools.
- **`graph/`**: General-purpose graph layer. `Node[T]`, `Edge` (frozen dataclasses), `Graph` facade (async-first with `_sync` wrappers). Protocols: `GraphBackend` (storage), `GraphAlgorithms` (BFS, DFS, shortest path, PageRank, etc.). Default backend: `NetworkXBackend` (import-guarded, requires `[graph]` extra). `Graph` exposes: node/edge CRUD, `has()`, `degree()`, `node_count()`, `edge_count()`, `list_edges()`, `filter_nodes()`, `filter_edges()`, `get_orphan_nodes()`, `get_stats()`, `copy()`, traversals (BFS, DFS, shortest path, find_all_paths, ancestors, descendants, ego_graph, PageRank, centrality, connected components, subgraph), persistence (`save`/`load`/`to_dict`/`from_dict`).

### Toolkit layer (`toolkit/`)

#### Agents (`toolkit/agents/`)

Built on core/ primitives (`LLM`, `Response`, `ToolGroup`, `Usage`, `ToolCall`, `tool_result()`).

- **`_base.py`**: `BaseAgent` ABC, `AgentConfig`, `AgentEvent`, `AgentStep`, `AgentResult`, `StopReason`. Async-first with sync wrappers. `@overload` on `run()` / `run_sync()` for `stream: bool` type narrowing.
- **`_react.py`**: `ReActAgent` — Thought → Action → Observation loop. `_run_loop()` is a pure async generator yielding `AgentEvent`; callbacks fire in `_consume()`.
- **`_reflexion.py`**: `ReflexionAgent` + `ReflexionConfig` — wraps ReActAgent in a retry loop with self-critique. Evaluator callback scores each attempt; below-threshold triggers reflection + retry.
- **`_rewoo.py`**: `ReWOOAgent` + `ReWOOConfig` — Plan with `#E{n}` placeholders → Execute tools → Solve. Three-phase architecture.
- **`_plan_execute.py`**: `PlanExecuteAgent` + `PlanExecuteConfig` — Numbered step plan → per-step ReAct execution → Solve. Optional replanning on failure.
- **`_tot.py`**: `ToTAgent` + `ToTConfig` — Tree of Thoughts with DFS/BFS search. Generate-evaluate-expand loop.
- **`_lats.py`**: `LATSAgent` + `LATSConfig` — Language Agent Tree Search (MCTS). UCT selection, ReAct rollouts, evaluation, backpropagation, reflection.
- **`_self_discovery.py`**: `SelfDiscoveryAgent` + `SelfDiscoveryConfig` — Select reasoning modules → Adapt → Operationalize → Solve via inner ReAct. 10 default reasoning strategies.
- **`_llm_compiler.py`**: `LLMCompilerAgent` + `LLMCompilerConfig` — Plan DAG → Parallel execute via asyncio.gather → Join → Optional replan. Topological task scheduling.
- Task input accepts `Content` (str or multimodal list) for vision+tools use cases.
- Agent-specific configs (`ReflexionConfig`, `ReWOOConfig`, `PlanExecuteConfig`, `ToTConfig`, `LATSConfig`, `SelfDiscoveryConfig`, `LLMCompilerConfig`) are standalone dataclasses — not inheriting from `AgentConfig`. Passed via separate constructor kwarg.

#### Tools (`toolkit/tools/`)

25 pre-built tools across 11 files, all using stdlib only (zero pip deps). Categories: datetime, math, text, filesystem, shell, JSON/CSV, web, weather (Open-Meteo), knowledge (Wikipedia, Free Dictionary), geo (geocoding, IP lookup, country info), news (Hacker News). All use `@tool` decorator from core/.

#### Memory (`toolkit/memory/`)

Graph-backed memory for LLM agents, built on `core/graph/`. `GraphStore` wraps `Graph` with memory-specific `Node` (adds `timestamp`, `source`, `confidence`, `access_count`, `last_accessed`, `embedding`), keyword/vector search, and access tracking. Views: `TemporalView` (recent, since), `RelationalView` (neighbors, path), `PropertyView` (by_confidence, by_source, most_accessed), `SimilarityView` (vector search). `MemoryMiddleware` auto-injects relevant memories into LLM context. Presets: `conversational()`, `cognitive()`. `memory_tools()` generates `@tool`-decorated functions for agent use.

#### Pipeline (`toolkit/pipeline/`)

Sequential phase execution with context accumulation. `Pipeline` takes named phase functions, runs them in order via `run()` or streams via `iter()`. `PipelineContext` accumulates artifacts and provenance across phases. `PhaseResult` supports `ok`/`failed`/`partial`/`skipped` statuses. Features: `stop_on_failure`, `stop_on_partial`, `run_from()` resume, token tracking.

#### Knowledge (`toolkit/knowledge/`)

Sync in-memory registry for prompt-injectable reference data. `KnowledgeRegistry` stores `KnowledgeEntry` items with category/tags/format. Querying via `by_category()`, `by_tags()`. `as_context()` builds prompt strings with separator/transform. Loaders: `load_text()`, `load_json()`, `load_toml()`, `load_yaml()`, `load_markdown()`, `load_directory()` (flat or recursive).

#### Runner

`_runner.py`: `run_tools()` / `run_tools_sync()` convenience wrappers.

**Import convention**: New code should import from `ai_arch_toolkit.core` or `ai_arch_toolkit.toolkit.agents`.

## Testing Patterns

- **Config**: pytest-asyncio with `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed.
- **Test layout**: `tests/` (core tests), `tests/agents/` (agent tests), `tests/toolkit/` (toolkit tool tests), `tests/graph/` (core graph tests), `tests/memory/` (memory tests), `tests/pipeline/` (pipeline tests), `tests/knowledge/` (knowledge tests).
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

## Provider-Specific Gotchas

- **Anthropic**: `input_schema` for tools (not `parameters`), `system` is a top-level field (not a message role), supports extended thinking. Native structured output via `output_config` (not tool trick).
- **Gemini**: `contents`/`parts` structure (not `messages`/`content`), uses NDJSON streaming (not SSE).
- **OpenAI**: Chat Completions and Responses API — both have provider implementations.
- **xAI**: Separate provider (not OpenAI-compat), API key via `XAI_API_KEY`.

## Research Docs

`research/` contains standalone Markdown reference guides (LLM API guide, agent architectures, Python best practices, graph algorithms). Separate from the Python package.
