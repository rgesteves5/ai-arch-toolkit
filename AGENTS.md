# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

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
uv sync --extra dev --extra docs        # Install docs dependencies
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
│   ├── agents/        # Built-in agent flow factories
│   │   └── flows/     # react_flow, reflexion_flow, rewoo_flow, etc.
│   ├── flow/          # Flow orchestration (Flow, FlowStep, FlowResult, FlowEvent)
│   ├── tools/         # Pre-built tools (datetime, geo, shell, Python, web, etc.)
│   ├── memory/        # Graph-backed memory for agents (GraphStore, views, search)
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

#### Agent Flows (`toolkit/agents/flows/`)

Built-in flow factories built on core/ primitives (`LLM`, `ToolGroup`, `State`, `Step`, `Result`). Each factory returns a `Flow` and has a companion `*_initial_state(task)` helper.

- **`react_flow()`**: Cyclic LLM → tool execution loop. Params: `system`, `max_iterations`, `parallel_tool_calls`, `timeout`, `policy`, `llm_kwargs`.
- **`reflexion_flow()`**: Inner ReAct with evaluate + reflect retry. Requires `evaluator: Callable[[str, str], float]`, `threshold`, `max_retries`.
- **`rewoo_flow()`**: Plan with `#E{n}` placeholders → Execute tools → Solve. Three-phase.
- **`plan_execute_flow()`**: Numbered plan → per-step ReAct → Solve. `max_replans`.
- **`tot_flow()`**: Tree of Thoughts with DFS/BFS search. `n_candidates`, `max_depth`, `strategy`.
- **`lats_flow()`**: Language Agent Tree Search (MCTS). `n_candidates`, `max_rollouts`, `exploration_weight`.
- **`self_discovery_flow()`**: Select reasoning modules → Adapt → Operationalize → Solve via inner ReAct.
- **`llm_compiler_flow()`**: Plan DAG → Parallel execute → Join. `max_replans`.
- **`generate_review_flow()`**: Generator-reviewer retry loop with optional tools in both phases.
- Task input accepts `Content` (str or multimodal list) for vision+tools use cases.

Usage pattern:
```python
flow = react_flow(llm, tools, max_iterations=5)
state = State(operational=react_initial_state("your task"))
result = flow.run_sync(state)
answer = state["response"].text
```

#### Tools (`toolkit/tools/`)

Pre-built tools span datetime, math, text, filesystem, shell, Python, JSON/CSV, web, weather (Open-Meteo), Wikipedia, dictionary lookups, geo, and Hacker News. All use `@tool` decorator from core/ and keep stdlib-only implementations.

#### Memory (`toolkit/memory/`)

Graph-backed memory for LLM agents, built on `core/graph/`. `GraphStore` wraps `Graph` with memory-specific `Node` (adds `timestamp`, `source`, `confidence`, `access_count`, `last_accessed`, `embedding`), keyword/vector search, and access tracking. Views: `TemporalView` (recent, since), `RelationalView` (neighbors, path), `PropertyView` (by_confidence, by_source, most_accessed), `SimilarityView` (vector search). `MemoryMiddleware` auto-injects relevant memories into LLM context. Presets: `conversational()`, `cognitive()`. `memory_tools()` generates `@tool`-decorated functions for agent use.

#### Knowledge (`toolkit/knowledge/`)

Sync in-memory registry for prompt-injectable reference data. `KnowledgeRegistry` stores `KnowledgeEntry` items with category/tags/format. Querying via `by_category()`, `by_tags()`. `as_context()` builds prompt strings with separator/transform. Loaders: `load_text()`, `load_json()`, `load_toml()`, `load_yaml()`, `load_markdown()`, `load_directory()` (flat or recursive).

#### Runner

`_runner.py`: `run_tools()` / `run_tools_sync()` convenience wrappers.

**Import convention**: New code should import from `ai_arch_toolkit.core` or `ai_arch_toolkit.toolkit.agents`.

## Testing Patterns

- **Config**: pytest-asyncio with `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed.
- **Test layout**: `tests/` (core tests), `tests/agents/flows/` (agent flow tests), `tests/toolkit/` (toolkit tool tests), `tests/graph/` (core graph tests), `tests/memory/` (memory tests), `tests/flow/` (flow tests), `tests/knowledge/` (knowledge tests).
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
- Practical code-writing guidance lives in `docs/code-style.md`: linting shape,
  docstring/comment style, and class vs function decisions.

## Provider-Specific Gotchas

- **Anthropic**: `input_schema` for tools (not `parameters`), `system` is a top-level field (not a message role), supports extended thinking. Native structured output via `output_config` (not tool trick).
- **Gemini**: `contents`/`parts` structure (not `messages`/`content`), uses NDJSON streaming (not SSE).
- **OpenAI**: Chat Completions and Responses API — both have provider implementations.
- **xAI**: Separate provider (not OpenAI-compat), API key via `XAI_API_KEY`.

## Research Docs

`research/` contains standalone Markdown reference guides (LLM API guide, agent architectures, Python best practices, graph algorithms). Separate from the Python package.
