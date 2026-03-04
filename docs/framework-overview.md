# Framework Overview

This document summarizes the **ai-arch-toolkit** package: its structure, features, and capabilities.

## What It Is

**ai-arch-toolkit** is a Python 3.13+ library that provides:

1. A **unified async-first LLM facade** across multiple providers
2. **Middleware** with before/after hooks for caching, cost tracking, guardrails
3. A **tool layer** (`@tool` decorator + `ToolGroup`) for LLM function calling
4. **Eight agent architectures** built on the same core primitives
5. **25 pre-built tools** (weather, geo, news, knowledge, filesystem, etc.)

The public API is re-exported from the top level:

```python
from ai_arch_toolkit import LLM, ReActAgent, tool, ToolGroup, AgentConfig, ...
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

### Other Core Modules

| Module | Purpose |
|--------|---------|
| `_middleware.py` | `Middleware` Protocol with `before`/`after` hooks, `Request` dataclass |
| `_retry.py` | `RetryConfig` + `with_retry()` for exponential backoff |
| `_pricing.py` | `PricingRegistry` with `_default_pricing.toml`, `pricing` singleton |
| `_server_tools.py` | `ServerTool`, `code_execution()`, `web_search()` for provider-hosted tools |
| `_batch.py` | `BatchRequest`, `BatchResult` for batch API jobs |
| `_sync.py` | `_run_sync()` and `_stream_sync()` helpers used by LLM and agents |
| `_exceptions.py` | `APIError`, `RateLimitError` |

---

## 2. Toolkit Layer (`toolkit/`)

Convenience utilities built on core/ primitives.

### Agents (`toolkit/agents/`)

All agents inherit **`BaseAgent`** (`_base.py`). They take an **`LLM`**, a **`ToolGroup`**, and an optional **`AgentConfig`**. Async-first with sync wrappers (`run()` / `run_sync()`). Streaming via `run(stream=True)`.

**Common types**: `AgentConfig`, `AgentEvent`, `AgentStep`, `AgentResult`, `StopReason`.

**Per-phase customization**: `PhaseConfig` allows overriding the LLM and/or tools for individual phases of multi-phase agents. All fields default to `None` (falls back to the agent-level default).

| Agent | Architecture | Phases |
|-------|-------------|--------|
| **ReActAgent** | Thought → Action → Observation loop | single-phase |
| **ReflexionAgent** | ReAct with self-critique retry loop | executor, reflector |
| **ReWOOAgent** | Plan with `#E{n}` placeholders → Execute tools → Solve | planner, solver |
| **PlanExecuteAgent** | Numbered plan → per-step ReAct → Solve | planner, executor, solver |
| **ToTAgent** | Tree of Thoughts (DFS/BFS search) | generator, evaluator, solver |
| **LATSAgent** | Language Agent Tree Search (MCTS + ReAct rollouts) | rollout, evaluator, solver, reflector |
| **SelfDiscoveryAgent** | Select reasoning modules → Adapt → Operationalize → Solve | reasoning, solver |
| **LLMCompilerAgent** | Plan DAG → Parallel execute → Join → Optional replan | planner, executor, joiner |

Agent-specific configs (`ReflexionConfig`, `ReWOOConfig`, `PlanExecuteConfig`, `ToTConfig`, `LATSConfig`, `SelfDiscoveryConfig`, `LLMCompilerConfig`) are standalone frozen dataclasses passed via separate constructor kwargs.

All agents return `AgentResult` with `answer`, `parsed`, `steps`, `total_usage`, `total_cost`, `stop_reason`.

### Pre-built Tools (`toolkit/tools/`)

25 tools across 11 modules, all using stdlib only (zero pip deps):

| Module | Tools |
|--------|-------|
| `_datetime.py` | current time, date math |
| `_math.py` | calculator, unit conversion |
| `_text.py` | word count, text summarization helpers |
| `_filesystem.py` | read/write/list files |
| `_shell.py` | run shell commands |
| `_json.py` | JSON/CSV parsing |
| `_web.py` | URL fetching |
| `_weather.py` | Open-Meteo forecast |
| `_knowledge.py` | Wikipedia, Free Dictionary |
| `_geo.py` | geocoding, IP lookup, country info |
| `_news.py` | Hacker News |

All use `@tool` decorator from core/. All return error strings (never raise) for graceful agent handling.

### Runner

`_runner.py`: `run_tools()` / `run_tools_sync()` convenience wrappers for single-shot tool execution loops.

---

## 3. Project Layout

```
src/ai_arch_toolkit/
├── __init__.py          # Re-exports from core/ + toolkit/
├── core/                # Stateless async-first foundation
│   ├── _llm.py          # LLM facade
│   ├── _content.py      # Messages, multimodal types
│   ├── _response.py     # Response, Usage, ToolCall, streaming types
│   ├── _providers/      # BaseProvider → Anthropic, OpenAI, xAI, Gemini
│   ├── _tools/          # @tool decorator, ToolGroup, schema inference
│   ├── _middleware.py    # Middleware protocol
│   ├── _retry.py        # RetryConfig, exponential backoff
│   ├── _pricing.py      # Model pricing registry
│   ├── _server_tools.py # Provider-hosted tools
│   ├── _batch.py        # Batch API types
│   ├── _sync.py         # Async-to-sync bridging
│   └── _exceptions.py   # APIError, RateLimitError
└── toolkit/             # Convenience utilities built on core/
    ├── agents/          # 8 agent architectures
    │   ├── _base.py     # BaseAgent, AgentConfig, PhaseConfig, AgentEvent/Step/Result
    │   ├── _react.py    # ReActAgent
    │   ├── _reflexion.py
    │   ├── _rewoo.py
    │   ├── _plan_execute.py
    │   ├── _tot.py
    │   ├── _lats.py
    │   ├── _self_discovery.py
    │   └── _llm_compiler.py
    ├── tools/           # 25 pre-built tools (11 modules)
    └── _runner.py       # run_tools() convenience wrapper
```

Supporting directories:

- **`examples/`** — 27 examples (01–27): hello world, streaming, tools, each agent, middleware, server tools, etc.
- **`tests/`** — core tests, `tests/agents/` (agent tests), `tests/toolkit/` (toolkit tool tests)
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
| **Tools** | `@tool` decorator + `ToolGroup` for schema and execution. 25 pre-built tools. |
| **Middleware** | `before`/`after` hooks on every LLM call. |
| **Retry** | `RetryConfig` with exponential backoff. |
| **Pricing** | Per-model pricing registry with cost tracking on `Response`. |
| **Batch** | `BatchRequest` / `BatchResult` for batch API jobs. |
| **Agents** | 8 architectures with shared base, event streaming, per-phase customization. |
| **Server tools** | `code_execution()`, `web_search()` for provider-hosted capabilities. |

---

## Quick Links

- [Getting Started](getting-started.md)
- [API Docs](api.md)
- [UV development guide](uv-guide.md)
- [Examples](../examples/)
