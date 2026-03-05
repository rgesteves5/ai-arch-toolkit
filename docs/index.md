# ai-arch-toolkit

A lightweight, unified LLM client and agent architecture toolkit for Python.

## Features

- **Multi-provider LLM client** — Anthropic, OpenAI, Gemini, xAI via a single `LLM` class
- **Async-first** with convenient sync wrappers
- **Tool system** — `@tool` decorator with auto-generated JSON Schema, 25 pre-built tools
- **Agent architectures** — ReAct, Reflexion, ReWOO, PlanExecute, Tree of Thoughts, LATS, Self-Discovery, LLM Compiler
- **Graph layer** — `Graph` facade with typed nodes, directed edges, algorithms (BFS, DFS, PageRank, etc.), persistence
- **Memory system** — graph-backed agent memory with search, temporal/relational/property views, middleware
- **Pipeline** — sequential phase execution with context accumulation, streaming, resume
- **Knowledge registry** — prompt-injectable reference data with category/tag filtering and file loaders
- **Structured output** — native JSON mode + Pydantic model support
- **Streaming** — text chunks and rich structured events (thinking, tool calls)
- **Middleware** — before/after hooks for caching, cost tracking, guardrails
- **Fallback** — automatic provider failover on errors
- **Batch API** — submit and retrieve batch requests

## Quick install

```bash
uv add ai-arch-toolkit
```

## Quick start

```python
from ai_arch_toolkit import LLM

llm = LLM("claude-sonnet-4-20250514")
response = llm.complete_sync("Hello!")
print(response.text)
```

## Links

- [Getting Started](getting-started.md) — installation, first steps, examples
- [API Docs](api.md) — detailed API reference
- [UV Guide](uv-guide.md) — development setup

## Build docs locally

```bash
uv sync --group docs
uv run mkdocs serve
```

