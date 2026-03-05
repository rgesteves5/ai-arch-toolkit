# API Documentation

## Auto-generate API docs with pdoc

```bash
uv sync --group docs
uv run pdoc ai_arch_toolkit -o site/api
uv run pdoc ai_arch_toolkit --http :8080  # serve locally
```

## Public API Surface

All public types are re-exported from `ai_arch_toolkit` (top-level) or from `ai_arch_toolkit.core` / `ai_arch_toolkit.toolkit`.

### Core — LLM & Providers

| Symbol | Module | Description |
|--------|--------|-------------|
| `LLM` | `_llm.py` | Unified facade. `complete()` / `stream()` / `stream_events()` + `_sync` wrappers |
| `Response` | `_response.py` | LLM response with `text`, `tool_calls`, `usage`, `cost` |
| `Usage`, `ToolCall`, `ThinkingBlock`, `Citation` | `_response.py` | Response components |
| `OutputSchema` | `_response.py` | Structured output constraint |
| `StreamEvent`, `RichStreamResponse` | `_response.py` | Streaming types |
| `Content`, `ContentPart` | `_content.py` | `str | list[ContentPart]` message content |
| `user()`, `assistant()`, `system()`, `tool_result()` | `_content.py` | Message constructors |
| `ImagePart`, `DocumentPart`, `CachePart` | `_content.py` | Multimodal content parts |
| `Middleware`, `Request` | `_middleware.py` | Before/after hooks protocol |
| `RetryConfig` | `_retry.py` | Exponential backoff configuration |
| `pricing` | `_pricing.py` | Per-model pricing registry singleton |
| `ServerTool`, `code_execution()`, `web_search()` | `_server_tools.py` | Provider-hosted tools |
| `BatchRequest`, `BatchResult` | `_batch.py` | Batch API types |
| `APIError`, `RateLimitError` | `_exceptions.py` | Exception types |

### Core — Tools

| Symbol | Description |
|--------|-------------|
| `@tool` | Decorator: auto-generates JSON Schema from type hints + docstrings |
| `ToolGroup` | Collection with `execute()` / `async_execute()` |
| `infer_schema()` | Manual schema inference from a callable |
| `prepare_tools()` | Convert tools to provider-specific format |

### Core — Graph

| Symbol | Description |
|--------|-------------|
| `Graph` | Primary facade — async-first with `_sync` wrappers |
| `Node[T]` (exported as `GraphNode`) | Generic typed node: `id`, `type`, `content: T`, `metadata` |
| `Edge` (exported as `GraphEdge`) | Directed edge: `source`, `target`, `relation`, `weight`, `metadata` |
| `NodeID`, `NodeType`, `Direction` | Type aliases |
| `GraphBackend` | Protocol — storage interface |
| `GraphAlgorithms` | Protocol — optional algorithms |
| `NetworkXBackend` | Default in-memory backend (import from `core.graph._networkx`) |

**Graph facade methods** (all have `_sync` counterparts):

- **Node ops**: `add`, `get`, `update`, `remove`, `list`, `count`, `has`, `degree`, `add_many`, `remove_many`
- **Edge ops**: `connect`, `disconnect`, `edges`, `get_edges_between`, `list_edges`
- **Queries**: `node_count`, `edge_count`, `is_empty`, `filter_nodes`, `filter_edges`, `get_orphan_nodes`, `get_stats`
- **Algorithms** (require `GraphAlgorithms` backend): `bfs`, `dfs`, `shortest_path`, `find_all_paths`, `get_ancestors`, `get_descendants`, `get_subgraph`, `get_ego_graph`, `pagerank`, `centrality`, `connected_components`
- **Persistence**: `save`, `load`, `to_dict`, `from_dict`, `copy`

### Toolkit — Agents

| Agent | Architecture |
|-------|-------------|
| `ReActAgent` | Thought → Action → Observation loop |
| `ReflexionAgent` | ReAct + self-critique retry (`ReflexionConfig`) |
| `ReWOOAgent` | Plan → Execute → Solve (`ReWOOConfig`) |
| `PlanExecuteAgent` | Plan → per-step ReAct → Solve (`PlanExecuteConfig`) |
| `ToTAgent` | Tree of Thoughts — DFS/BFS (`ToTConfig`) |
| `LATSAgent` | MCTS + ReAct rollouts (`LATSConfig`) |
| `SelfDiscoveryAgent` | Reasoning module selection → Solve (`SelfDiscoveryConfig`) |
| `LLMCompilerAgent` | DAG plan → parallel execute → join (`LLMCompilerConfig`) |

Common types: `BaseAgent`, `AgentConfig`, `PhaseConfig`, `AgentEvent`, `AgentStep`, `AgentResult`, `StopReason`.

### Toolkit — Memory

| Symbol | Description |
|--------|-------------|
| `GraphStore` | Graph-backed memory store with search and access tracking |
| `Node` (memory) | Memory node: adds `timestamp`, `source`, `confidence`, `embedding`, `access_count` |
| `TemporalView` | Query by recency (`recent`, `since`) |
| `RelationalView` | Graph traversal (`neighbors`, `path`) |
| `PropertyView` | Filter by `confidence`, `source`, `access_count` |
| `SimilarityView` | Vector similarity search |
| `MemoryMiddleware` | Auto-injects memories into LLM context |
| `MemoryPreset` | Preset configurations |
| `conversational()`, `cognitive()` | Built-in presets |
| `memory_tools()` | Generate `@tool` functions for agent use |

### Toolkit — Pipeline

| Symbol | Description |
|--------|-------------|
| `Pipeline` | Sequential phase execution: `run()`, `iter()`, `run_from()` |
| `PipelineContext` | Accumulates artifacts, provenance, metadata |
| `PhaseResult` | Phase outcome: `ok` / `failed` / `partial` / `skipped` |
| `PipelineResult` | Aggregate result with token tracking |
| `run_phase()`, `run_phases()` | Convenience functions |

### Toolkit — Knowledge

| Symbol | Description |
|--------|-------------|
| `KnowledgeRegistry` | In-memory store for reference data |
| `KnowledgeEntry` | Entry with `key`, `content`, `format`, `category`, `tags` |
| `load_text()`, `load_json()`, `load_toml()`, `load_yaml()`, `load_markdown()` | File loaders |
| `load_directory()` | Bulk loader (flat or recursive) |
