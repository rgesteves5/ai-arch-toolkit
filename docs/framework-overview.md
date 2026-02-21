# Framework Overview

This document summarizes the **ai-arch-toolkit** package: its structure, features, and capabilities.

## What It Is

**ai-arch-toolkit** is a Python 3.12+ library that provides:

1. A **unified client API** across multiple LLM providers
2. **Middleware** for caching, cost tracking, guardrails, and tracing
3. A **tool layer** (registry + `@tool` decorator) for LLM function calling
4. **Eight agent architectures** built on top of the same client

The public API is re-exported from the top level so you can do:

```python
from ai_arch_toolkit import Client, Tool, ReActAgent, tool, ...
```

---

## 1. LLM Layer

### Providers (Unified Behind One API)

- **OpenAI-compatible** (one implementation, different configs): **OpenAI**, **xAI**, **Mistral**, **Groq**
- **Anthropic** (native)
- **Gemini** (native; NDJSON streaming)
- **OpenAI Responses API** (`openai-responses`)
- **xAI Responses API** (`xai-responses`)

You choose by name, e.g. `Client("openai", model="gpt-4o")` or `Client("anthropic", model="claude-3-5-sonnet-20241022")`. API keys come from the constructor or env (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).

### Client API

- **Sync**: `Client` — `chat()`, `stream()`, `stream_events()`
- **Async**: `AsyncClient` — same operations, async
- Input: `str` (single user message) or `Sequence[Message | ToolResult]`
- Optional: `system`, `tools`, `json_schema`, `timeout`, provider-specific `**kwargs`
- **Middleware**: every request goes through a pipeline; each middleware has `before` / `after` (and async `abefore` / `aafter`). A middleware can short-circuit and return a cached result.

### Types (`llm/_types.py`)

- Frozen, slotted dataclasses: `Message`, `Response`, `Tool`, `ToolCall`, `ToolResult`, `Usage`
- **Content**: `str` or `tuple[ContentPart, ...]` with `TextPart`, `ImagePart`, `AudioPart`, `DocumentPart`
- **Thinking**: `ThinkingConfig`, `ThinkingBlock`
- **Streaming**: `StreamEvent` (e.g. `text`, `tool_call`, `thinking`, `usage`, `done`)
- **Server tools**: `ServerTool` for provider-built-ins (e.g. web search, code interpreter)

### Built-in Middleware

- **ResponseCache** + **CacheBackend** (e.g. `InMemoryCacheBackend`) — cache responses
- **CostTracker** — track token usage and cost
- **GuardrailMiddleware** — block or flag requests (e.g. PII, forbidden topics)
- **TracingMiddleware** — plug in OpenTelemetry-style tracers

### Utilities

- **Templates**: `PromptTemplate`, `ChatTemplate`
- **Parsing**: `parse_json`, `parse_json_as`, `extract_list`, `extract_code_block`
- **Tokens**: token estimation (with model-specific correction factors), `estimate_usage_cost`, `preview_*` for cost
- **Memory**: `ConversationMemory`, `SlidingWindowMemory`
- **Fallback**: `FallbackClient` — try one provider, then another on failure
- **HTTP**: `RetryConfig`, `post_json`, `stream_sse`, `stream_ndjson` (sync: `requests`, async: `httpx`)

### Batch API

- **BatchClient** / **AsyncBatchClient** — submit batch jobs, poll status, fetch results (OpenAI and Anthropic batch APIs)
- Types: `BatchRequest`, `BatchResult`, `BatchJob`

### Exceptions

- `APIError`, `RateLimitError` from `llm/_exceptions.py`

---

## 2. Tools Layer

- **`@tool`** (`tools/_decorator.py`): decorates a function; builds a `Tool` (name, description, JSON Schema `parameters`) from type hints and Google-style docstrings. Supports primitives, `Literal`, enums, list/dict, dataclasses, TypedDict, Pydantic. Attaches `__tool__` for use with the registry.
- **ToolRegistry** (`tools/_registry.py`): register callables with their `Tool` definition; **execute** / **async_execute** a `ToolCall` (parse args, validate, run, return string result). Can disable tools by name. **ValidationError** on invalid arguments.

Agents receive `Tool` definitions from the registry and use the same registry to execute `ToolCall`s.

---

## 3. Agents Layer

All agents inherit **BaseAgent**: they take a **Client**, a **ToolRegistry** (or None), and **AgentConfig**. They implement **run**, **async_run**, and usually **run_stream** (yields events/steps).

Common config: **AgentConfig** — `max_iterations`, `system`, `max_tokens`, `tool_choice`, `parallel_tool_execution`, `timeout`, `on_event` callback. Shared types: **AgentStep**, **AgentResult** (or subclasses), **AgentEvent**, **BaseResult**, **CheckpointState** (stub for resume).

| Agent | Role |
|-------|------|
| **ReActAgent** | Thought → Action → Observation loop; interleaves reasoning and tool calls until no more tools or max iterations. |
| **PlanExecuteAgent** | Plan (LLM list of steps) then execute each step with a mini-ReAct-style executor. |
| **TreeOfThoughtsAgent** | Tree search (BFS/DFS); at each node generate K thoughts, score with LLM, expand. |
| **LATSAgent** | Monte Carlo Tree Search: SELECT (UCT), EXPAND (LLM), SIMULATE, BACKUP. |
| **ReflexionAgent** | Wraps ReAct: run → evaluate → if bad, reflect and retry with refined system prompt. |
| **ReWOOAgent** | Planner produces a plan with placeholders; worker runs steps and substitutes results; solver produces final answer. |
| **LLMCompilerAgent** | Planner outputs a DAG of tasks; scheduler runs ready tasks; executor runs tools; results feed dependent steps. |
| **SelfDiscoveryAgent** | Four-phase: SELECT modules, ADAPT to task, APPLY, REFLECT. |

All return a result type (e.g. **ReActResult**, **PlanExecuteResult**) with `answer`, `steps`, `total_usage`, `stop_reason`, and optional `metadata`.

---

## 4. Project Layout (High Level)

- **`src/ai_arch_toolkit/`**: `__init__.py` (re-exports), `_logging.py`, **`llm/`**, **`tools/`**, **`agents/`**
- **`llm/`**: types, providers (with `_providers/`), client, async_client, middleware, cache, cost, guardrails, tracing, templates, output_parsing, tokens, memory, fallback, batch, async_batch, http, exceptions
- **`tools/`**: `_decorator.py` (`@tool`), `_registry.py`
- **`agents/`**: `_base.py`, `_parsing.py`, and one module per architecture (`_react`, `_plan_execute`, `_tot`, `_lats`, `_reflexion`, `_rewoo`, `_compiler`, `_self_discovery`)
- **`examples/`**: 01–31 demonstrate hello world, multi-turn, streaming, tools, each agent, middleware, batch, async, templates, tokens, guardrails, tracing, etc.
- **`docs/`**: index, API pointer, uv guide; MkDocs + pdoc for serving/generating docs
- **`research/`**: standalone markdown (LLM APIs, agent designs, etc.), not part of the package

---

## 5. Capabilities Summary

| Area | Capability |
|------|------------|
| **Multi-provider** | One API for OpenAI, Anthropic, xAI, Gemini, Mistral, Groq (and Responses APIs where applicable). |
| **Sync + async** | `Client` / `AsyncClient`; async batch client. |
| **Streaming** | `stream()` (chunks) and `stream_events()` (typed events: text, tool_call, usage, done). |
| **Structured output** | `json_schema` for constrained JSON. |
| **Multimodal** | `ImagePart`, `AudioPart`, `DocumentPart` in message content. |
| **Thinking** | `ThinkingConfig` / `ThinkingBlock` for extended reasoning. |
| **Tools** | `Tool` / `ToolCall` / `ToolResult`; `@tool` + `ToolRegistry` for schema and execution. |
| **Middleware** | Cache, cost, guardrails, tracing, custom middleware; short-circuit supported. |
| **Cost & tokens** | Model pricing, token estimation with correction factors, cost preview/snapshot. |
| **Batch** | Submit, poll, and retrieve batch results (OpenAI/Anthropic). |
| **Agents** | Eight architectures (ReAct, Plan-Execute, ToT, LATS, Reflexion, ReWOO, LLM Compiler, Self-Discovery) with a common base and event hooks. |

---

## Quick Links

- [API Docs](api.md)
- [UV development guide](uv-guide.md)
- [Examples](../examples/README.md)
