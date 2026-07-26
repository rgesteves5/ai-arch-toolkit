# ai-arch-toolkit

A lightweight, unified LLM client and agent architecture toolkit for Python.

## Features

- **Multi-provider LLM client** — Anthropic, OpenAI, Gemini, xAI via a single `LLM` class
- **Async-first** with convenient sync wrappers
- **Tool system** — `@tool` decorator with auto-generated JSON Schema plus built-in tools for files, web, weather, geo, shell, Python, and knowledge lookups
- **Agent flows** — ReAct, Reflexion, ReWOO, PlanExecute, Tree of Thoughts, LATS, Self-Discovery, LLM Compiler, and Generate-Review
- **Configurable agents** — `Agent`/`ReasoningSpec` with per-phase models, tools, and prompts; file-backed agent manifests with inheritance, fingerprints, override governance, and `ai-arch agent validate|inspect` for CI
- **Graph layer** — `Graph` facade with typed nodes, directed edges, algorithms (BFS, DFS, PageRank, etc.), persistence
- **Memory system** — graph-backed agent memory with search, temporal/relational/property views, middleware
- **Flow orchestration** — sequential, cyclic, and DAG execution over `State`, `Step`, `Policy`, and `Trace`
- **Resources** — reusable TXT/Markdown/JSON/YAML/TOML loading, selection, and provenance
- **Knowledge registry** — categorized and tagged reusable reference data built on Resources
- **Structured prompts** — files, typed templates, manifests, layouts, deterministic rendering, and fingerprints
- **Moderation** — protocol-level moderation types plus LLM/OpenAI moderation helpers
- **Structured output** — native JSON mode + Pydantic model support
- **Streaming** — text chunks and rich structured events (thinking, tool calls)
- **Middleware** — before/after hooks for caching, cost tracking, guardrails
- **Fallback** — automatic provider failover on errors
- **Batch operations** — `LLM.batch_submit()` / `batch_status()` / `batch_results()` with batch request/result types

## Quick install

The package is not on PyPI yet — install from the repo:

```bash
uv add "git+https://github.com/rgesteves5/ai-arch-toolkit.git#egg=ai-arch-toolkit[openai]"
# or substitute another extra: [anthropic], [gemini], [xai], or [all]
```

## Quick start

```python
from ai_arch_toolkit import LLM

llm = LLM("gpt-4.1-nano")
response = llm.complete_sync("Hello!")
print(response.text)
```

## Links

- [Getting Started](getting-started.md) — installation, first steps, examples
- [Configuring Agents](configuring-agents.md) — the end-to-end guide: specs, per-phase config, prompts, manifests
- [Prompts](prompts.md) — literal prompts, files, templates, manifests, and layouts
- [Context Model](context-model.md) — Content vs Resources vs Knowledge vs Memory
- [Model Compatibility](model-compatibility.md) — live-probed model feature matrix
- [Code Style](code-style.md) — linting, docstrings, comments, and class/function choices
- [API Docs](api.md) — detailed API reference
- [UV Guide](uv-guide.md) — development setup

## Build docs locally

```bash
uv sync --extra dev --extra docs
uv run mkdocs serve
```
