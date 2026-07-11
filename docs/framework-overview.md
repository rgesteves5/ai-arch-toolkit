# Framework Overview

This document summarizes the **ai-arch-toolkit** package: its structure, features, and capabilities.

## What It Is

**ai-arch-toolkit** is a Python 3.13+ library that provides:

1. A **unified async-first LLM facade** across multiple providers
2. **Middleware** with before/after hooks for caching, cost tracking, guardrails
3. A **tool layer** (`@tool` decorator + `ToolGroup`) for LLM function calling
4. A **Flow orchestration system** — composable Steps, Policies, Traces, and Scopes
5. **Built-in agent flow factories** built on the same core primitives
6. **Pre-built safe tools** plus explicit opt-in dangerous tools for files, shell, Python, and arbitrary URL fetching
7. A **general-purpose graph layer** (`Graph`, `Node[T]`, `Edge`, algorithms)
8. **Graph-backed memory** for agents (search, views, middleware, presets)
9. **Knowledge registry** for prompt-injectable reference data
10. **Structured prompts** with deterministic section ordering and exact fingerprints
11. **Moderation helpers** for classifier-style guardrails and middleware

The public API is re-exported from the top level:

```python
from ai_arch_toolkit import LLM, tool, ToolGroup, ...
from ai_arch_toolkit.core import State, Step, Result, Policy
from ai_arch_toolkit.toolkit.flow import Flow, FlowStep, Scope
from ai_arch_toolkit.toolkit.agents.flows import react_flow, react_initial_state, ...
```

---

## 1. Core Layer (`core/`)

Stateless, async-first foundation. All new code should build on this.

### LLM Facade

- **`LLM`** (`_llm.py`) — user-facing facade. Async: `complete()`, `stream()`, `stream_events()`. Sync wrappers: `complete_sync()`, `stream_sync()`, `stream_events_sync()`.
- Accepts `Content` (str or multimodal parts). Stream methods support fallback + middleware.
- Model prefix routes to the correct provider automatically.

### Providers

- **`BaseProvider`** ABC → `AnthropicProvider`, `OpenAIProvider`, `XAIProvider`, `GeminiProvider`
- Factory: `create_provider()` routes by model prefix (`claude-` → Anthropic, `gpt-`/`o1-`/`o3-`/`o4-` → OpenAI, `grok-` → xAI, `gemini-` → Gemini)

```python
from ai_arch_toolkit import LLM

llm = LLM("claude-sonnet-4-20250514")  # Anthropic
llm = LLM("gpt-4o")                    # OpenAI
llm = LLM("gemini-2.0-flash")          # Gemini
llm = LLM("grok-2")                    # xAI
```

### Content & Messages

- **`_content.py`**: Message constructors (`user()`, `assistant()`, `system()`, `tool_result()`) and multimodal types (`ImagePart`, `DocumentPart`, `CachePart`).
- `type Content = str | list[ContentPart]`

### Response Types

- **`_response.py`**: `Response`, `Usage`, `ToolCall`, `ThinkingBlock`, `Citation`, `OutputSchema`, `StreamEvent`, `StreamResponse`, `RichStreamResponse` (and sync variants).

### Tools

- **`@tool`** decorator (`_tools/`): auto-generates JSON Schema from type hints + Google-style docstrings.
- **`ToolGroup`**: collection with `execute()` / `async_execute()`.
- **`infer_schema()`**, **`prepare_tools()`** for manual schema work.

### Flow Primitives

Core primitives for the Flow orchestration system:

| Module | Purpose |
|--------|---------|
| `_state.py` | `State`, `StateSnapshot`, `MergeStrategy` — 4-layer mutable container |
| `_step.py` | `Step`, `StepFn`, `Result` — named async functions with structured output |
| `_policy.py` | `Policy` — retry, timeout, confidence thresholds, cost limits |
| `_trace.py` | `Trace`, `StepTrace`, `PolicyDecision` — full execution records |
| `_step_engine.py` | `execute_step()` — single-step execution with policy enforcement |

### Other Core Modules

| Module | Purpose |
|--------|---------|
| `_middleware.py` | `Middleware` Protocol with `before`/`after` hooks, `Request` dataclass |
| `_moderation.py` | `Moderator`, `ModerationResult`, `ModerationError` |
| `_retry.py` | `RetryConfig` + `with_retry()` for exponential backoff |
| `_pricing.py` | `PricingRegistry` with `_default_pricing.toml`, `pricing` singleton |
| `_server_tools.py` | `ServerTool`, `code_execution()`, `web_search()` for provider-hosted tools |
| `_batch.py` | `BatchRequest`, `BatchResult` for batch API jobs |
| `_sync.py` | `_run_sync()` and `_stream_sync()` helpers used by LLM and agents |
| `_exceptions.py` | `APIError`, `RateLimitError` |

### Graph Layer (`core/graph/`)

General-purpose graph with typed nodes, directed edges, and pluggable backends.

- **`Node[T]`** — generic node (`id`, `type`, `content: T`, `metadata`). Frozen dataclass.
- **`Edge`** — directed edge (`source`, `target`, `relation`, `weight`, `metadata`). Frozen dataclass.
- **`Graph`** — primary facade. Async-first with `_sync` wrappers. Type-indexed node lookup, persistence (`save`/`load`/`to_dict`/`from_dict`).
- **`GraphBackend`** protocol — storage interface (node/edge CRUD, neighbors, clear).
- **`GraphAlgorithms`** protocol — optional algorithms (BFS, DFS, shortest path, centrality, connected components, subgraph, find_all_paths, ancestors, descendants, ego_graph, PageRank).
- **`NetworkXBackend`** — default in-memory implementation (requires `networkx`, import-guarded).

See [Graph Layer](graph.md) for full documentation.

---

## 2. Toolkit Layer (`toolkit/`)

Convenience utilities built on core/ primitives.

### Flow Orchestration (`toolkit/flow/`)

Composable orchestration framework built on core/ primitives. See [Flow Architecture](flow-architecture.md) for details.

- **`Flow`** — composes Steps into sequential, cyclic, or DAG execution graphs.
- **`FlowStep`** — wraps a Step with optional `when` conditions and `after` dependencies.
- **`FlowResult`** — total cost, duration, usage, and full Trace.
- **`FlowEvent`** — streaming events (`flow_start`, `step_start`, `step_end`, `step_skipped`, `flow_end`).
- **`Scope`** — controls what keys a Step can see (include/exclude/transform/enrich).
- **`execute_flow()`** / **`iter_flow()`** — execution and streaming entry points.

Three execution modes (auto-detected):
- **Sequential**: steps run in order
- **Cyclic**: steps loop with `when` conditions (requires `max_iterations`)
- **DAG**: steps with `after` dependencies run in parallel where possible

### Agent Flows (`toolkit/agents/flows/`)

Built-in agent flow factories implemented on top of `Flow`. See [Agents and Capabilities](agents-and-capabilities.md).

| Flow Factory | Architecture | Phases |
|-------------|-------------|--------|
| `react_flow()` | Thought → Action → Observation loop | single-phase |
| `reflexion_flow()` | ReAct with self-critique retry loop | attempt, evaluate, reflect |
| `rewoo_flow()` | Plan with `#E{n}` → Execute → Solve | plan, execute, solve |
| `plan_execute_flow()` | Numbered plan → per-step ReAct → Solve | plan_and_execute, solve |
| `tot_flow()` | Tree of Thoughts (DFS/BFS search) | search_step |
| `lats_flow()` | Language Agent Tree Search (MCTS + ReAct) | mcts_rollout |
| `self_discovery_flow()` | Select reasoning modules → Adapt → Solve | select, adapt, operationalize, solve |
| `llm_compiler_flow()` | Plan DAG → Parallel execute → Join | compile |
| `generate_review_flow()` | Generate → Review → Retry loop | generate, review |

Each factory has a companion `*_initial_state(task)` helper that creates the initial operational dict.

### Structured Prompts (`toolkit/prompts/`)

- **`PromptSection`** — named prompt content with deterministic `order` and
  `static`/`session`/`request` stability.
- **`Prompt`** — immutable collection of sections and a separator.
- **`render_prompt()`** — validates unique names and renders strictly by section order.
- **`validate_cache_layout()`** — optional strict validation for a cache-optimized
  `static → session → request` layout.
- **`RenderedPrompt`** — exact text, ordered sections, SHA-256 fingerprint, and static-prefix
  diagnostics.

Knowledge and prompt composition remain separate: `KnowledgeRegistry` selects reusable
content, while `Prompt` controls its placement and provenance.

### Pre-built Tools (`toolkit/tools/`)

Pre-built tools are organized across the toolkit tool modules and use stdlib-only implementations.
The default public namespace is safe-by-default; shell execution, filesystem
access, arbitrary URL fetching, and Python execution require explicit opt-in via
`ai_arch_toolkit.toolkit.tools.dangerous`.

| Module | Tools |
|--------|-------|
| `_datetime.py` | current time, date math |
| `_math.py` | calculator, unit conversion |
| `_text.py` | word count, text summarization helpers |
| `_json.py` | JSON/CSV parsing |
| `_weather.py` | Open-Meteo forecast |
| `_wikipedia.py` | Wikipedia search and article retrieval |
| `_dictionary.py` | Free Dictionary lookups |
| `_geo.py` | geocoding, IP lookup, country info |
| `_news.py` | Hacker News |
| `tools.dangerous` | filesystem, shell, Python execution, arbitrary URL fetching |

All use `@tool` decorator from core/. All return error strings (never raise) for graceful agent handling.

### Runner

`_runner.py`: `run_tools()` / `run_tools_sync()` convenience wrappers for single-shot tool execution loops.

### Memory (`toolkit/memory/`)

Graph-backed memory for LLM agents, built on `core/graph/`.

- **`GraphStore`** — wraps `Graph` with memory-specific `Node` (adds `timestamp`, `source`, `confidence`, `access_count`, `last_accessed`, `embedding`), keyword/vector search, and access tracking.
- **Views**: `TemporalView` (recent, since), `RelationalView` (neighbors, path), `PropertyView` (by_confidence, by_source, most_accessed), `SimilarityView` (vector search).
- **`MemoryMiddleware`** — auto-injects relevant memories into LLM context.
- **Presets**: `conversational()` and `cognitive()` for common configurations.
- **`memory_tools()`** — generates `@tool`-decorated functions for agent use.

### Knowledge (`toolkit/knowledge/`)

Sync in-memory registry for prompt-injectable reference data.

- **`KnowledgeRegistry`** — stores `KnowledgeEntry` items with category, tags, and format.
- Querying: `by_category()`, `by_tags()` (match_all or match_any).
- **`as_context()`** — builds prompt strings with separator/transform.
- Loaders: `load_text()`, `load_json()`, `load_toml()`, `load_yaml()`, `load_markdown()`, `load_directory()` (flat or recursive).

### Moderation (`toolkit/moderation/`)

Helpers for classifier-style safety checks and middleware integration.

- **`LLMModerator`** — uses an LLM to classify content against moderation instructions.
- **`OpenAIModerator`** — adapter over the OpenAI Moderation API.
- **`ModerationMiddleware`** — enforces moderation checks before or after LLM calls.

---

## 3. Project Layout

```
src/ai_arch_toolkit/
├── __init__.py          # Re-exports from core/ + toolkit/
├── core/                # Stateless async-first foundation
│   ├── _llm.py          # LLM facade
│   ├── _content.py      # Messages, multimodal types
│   ├── _response.py     # Response, Usage, ToolCall, streaming types
│   ├── _state.py        # State, StateSnapshot, MergeStrategy
│   ├── _step.py         # Step, StepFn, Result
│   ├── _policy.py       # Policy (retry, timeout, confidence, cost)
│   ├── _trace.py        # Trace, StepTrace, PolicyDecision
│   ├── _step_engine.py  # execute_step() — policy-enforced execution
│   ├── _providers/      # BaseProvider → Anthropic, OpenAI, xAI, Gemini
│   ├── _tools/          # @tool decorator, ToolGroup, schema inference
│   ├── graph/           # General-purpose graph layer
│   │   ├── _types.py    # Node[T], Edge, NodeID, Direction
│   │   ├── _backends.py # GraphBackend, GraphAlgorithms protocols
│   │   ├── _store.py    # Graph facade
│   │   └── _networkx.py # NetworkXBackend (import-guarded)
│   ├── _middleware.py    # Middleware protocol
│   ├── _moderation.py   # Moderator protocol and result types
│   ├── _retry.py        # RetryConfig, exponential backoff
│   ├── _pricing.py      # Model pricing registry
│   ├── _server_tools.py # Provider-hosted tools
│   ├── _batch.py        # Batch API types
│   ├── _sync.py         # Async-to-sync bridging
│   └── _exceptions.py   # APIError, RateLimitError
└── toolkit/             # Convenience utilities built on core/
    ├── agents/          # Built-in agent flow factories
    │   └── flows/       # Flow-based agent factories
    │       ├── _react.py
    │       ├── _reflexion.py
    │       ├── _rewoo.py
    │       ├── _plan_execute.py
    │       ├── _tot.py
    │       ├── _lats.py
    │       ├── _self_discovery.py
    │       ├── _llm_compiler.py
    │       └── _generate_review.py
    ├── flow/            # Flow orchestration
    │   ├── _scope.py    # Scope, apply_scope()
    │   ├── _flow.py     # Flow, FlowStep, FlowResult, FlowEvent
    │   └── _executor.py # execute_flow(), iter_flow()
    ├── tools/           # Pre-built tools
    ├── memory/          # Graph-backed memory (GraphStore, views, search)
    ├── knowledge/       # Knowledge registry + loaders
    ├── moderation/      # LLM/OpenAI moderation helpers
    └── _runner.py       # run_tools() convenience wrapper
```

Supporting directories:

- **`examples/`** — examples (01–36): hello world, streaming, tools, agent flows, middleware, server tools, memory, flows, knowledge, fallback chains
- **`tests/`** — `tests/` (core), `tests/agents/`, `tests/agents/flows/`, `tests/toolkit/`, `tests/graph/`, `tests/memory/`, `tests/knowledge/`, `tests/flow/`
- **`research/`** — standalone Markdown reference guides (not part of the package)
- **`docs/`** — MkDocs site source

---

## 4. Capabilities Summary

| Area | Capability |
|------|------------|
| **Multi-provider** | One `LLM` class for Anthropic, OpenAI, xAI, Gemini. Model prefix auto-routes. |
| **Async-first** | `complete()` / `stream()` / `stream_events()` with `_sync()` wrappers. |
| **Streaming** | Text chunks (`stream`), typed events (`stream_events`), rich events with fallback. |
| **Structured output** | `OutputSchema` for constrained JSON, Pydantic model support. |
| **Multimodal** | `ImagePart`, `DocumentPart`, `CachePart` in message content. |
| **Extended thinking** | `ThinkingBlock` for reasoning traces. |
| **Tools** | `@tool` decorator + `ToolGroup` for schema and execution, with built-in utilities across multiple domains. |
| **Middleware** | `before`/`after` hooks on every LLM call. |
| **Retry** | `RetryConfig` with exponential backoff. |
| **Pricing** | Per-model pricing registry with cost tracking on `Response`. |
| **Batch** | `LLM.batch_submit()` / `batch_status()` / `batch_results()` plus `BatchRequest` / `BatchResult` types. |
| **Flows** | Composable Step orchestration — sequential, cyclic, DAG modes with Policy and Trace. |
| **Agent flows** | Built-in flow factories for ReAct-style loops, planners, tree search, and generate-review workflows. |
| **Graph** | `Graph` facade with `Node[T]`/`Edge`, algorithms (BFS, DFS, PageRank, etc.), persistence. |
| **Memory** | `GraphStore` with keyword/vector search, temporal/relational/property views, middleware. |
| **Knowledge** | `KnowledgeRegistry` with category/tag filtering, context building, file loaders. |
| **Moderation** | `Moderator` protocol with `LLMModerator`, `OpenAIModerator`, and `ModerationMiddleware`. |
| **Server tools** | `code_execution()`, `web_search()` for provider-hosted capabilities. |

---

## Quick Links

- [Getting Started](getting-started.md)
- [Flow Architecture](flow-architecture.md)
- [Agents and Capabilities](agents-and-capabilities.md)
- [Graph Layer](graph.md)
- [Examples](examples.md)
- [API Docs](api.md)
- [UV development guide](uv-guide.md)
