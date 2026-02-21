# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
uv sync --dev                          # Install all dev dependencies
uv run pytest                          # Run full test suite
uv run pytest tests/llm/test_client.py # Run a single test file
uv run pytest -k "test_chat_basic"     # Run tests matching a pattern
uv run ruff check src tests examples   # Lint
uv run ruff check --fix src tests      # Auto-fix lint issues (import sorting, etc.)
uv run ruff format src tests           # Auto-format
uv run python examples/01_hello_world.py  # Run an example

# Documentation
uv sync --group docs                   # Install docs dependencies
uv run mkdocs serve                    # Local docs server
uv run pdoc ai_arch_toolkit -o site/api  # Generate API docs
```

## Architecture

**Three subpackages** under `src/ai_arch_toolkit/`: `llm/`, `tools/`, `agents/`. The top-level `__init__.py` re-exports all public symbols for flat imports (`from ai_arch_toolkit import Client, Tool, ReActAgent`).

### LLM layer

- **Types** (`llm/_types.py`): All data types are frozen dataclasses — `Message`, `Response`, `Tool`, `ToolCall`, `Usage`, multimodal parts (`ImagePart`, `AudioPart`, `DocumentPart`), `StreamEvent`, `ThinkingConfig`/`ThinkingBlock`. The `Content` type alias is `str | tuple[ContentPart, ...]`.
- **Providers** (`llm/_providers/`): `BaseProvider` ABC defines `complete`, `stream`, `stream_events` + async variants. Implementations: `_anthropic.py`, `_openai_compat.py` (covers openai/xai/mistral via `OPENAI_COMPAT_PROVIDERS` dict), `_gemini.py`, `_openai_responses.py`, `_xai_responses.py`. Factory is `create_provider()` in `_providers/__init__.py`.
- **Client** (`llm/_client.py`, `llm/_async_client.py`): User-facing facades wrapping providers. Accept `str` or `Sequence[Message | ToolResult]`. Middleware pipeline runs `before`/`after` hooks on every request.
- **HTTP** (`llm/_http.py`, `llm/_async_http.py`): `post_json`, `stream_sse`, `stream_ndjson` helpers with `RetryConfig`. Sync uses `requests`, async uses `httpx`.
- **Middleware** (`llm/_middleware.py`): `Middleware` Protocol with `before`/`after`/`abefore`/`aafter`. `Request` dataclass carries operation context. Implementations: `_tracing.py`, `_guardrails.py`, `_cache.py`, `_cost.py`.
- **Utilities**: `_templates.py` (prompt templates), `_output_parsing.py` (JSON/list extraction), `_tokens.py` (token estimation), `_memory.py` (conversation memory), `_fallback.py` (fallback client).

### Tools layer

- `tools/_registry.py`: `ToolRegistry` — register/execute/async_execute functions, produces `Tool` definitions for LLM APIs.
- `tools/_decorator.py`: `@tool` decorator auto-generates `Tool` JSON Schema from type hints + Google-style docstrings. Attaches `__tool__` attribute.

### Agents layer

- `agents/_base.py`: `BaseAgent` ABC with `run()`, `async_run()`, `run_stream()`. Common types: `AgentConfig`, `AgentStep`, `AgentResult`, `AgentEvent`.
- Eight implementations: `_react.py`, `_rewoo.py`, `_reflexion.py`, `_plan_execute.py`, `_compiler.py`, `_tot.py`, `_lats.py`, `_self_discovery.py`.
- `agents/_parsing.py`: Shared `parse_numbered_items` + `parse_score` used by ToT and LATS.

## Testing Patterns

- `tests/conftest.py`: `MockResponse` (mimics `requests.Response` with `json()`, `iter_lines()`, context manager), `mock_post` fixture (monkeypatches `requests.post`), `weather_tool` fixture.
- Client/provider tests use `@patch("ai_arch_toolkit.llm._client.create_provider")` — note the `.llm.` in the path.
- SSE test data: prefix lines with `"data: "`. Gemini NDJSON tests: plain JSON strings.
- Agent tests: mock `Client` with `MagicMock`, set `client.chat.side_effect` with pre-built `Response` objects.
- pytest-asyncio with `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed.

## Code Conventions

- Python 3.12+, `from __future__ import annotations` in every file.
- Ruff line length: 99. Run `ruff format` after edits — it reformats dict literals, multi-line calls, etc.
- All dataclasses use `frozen=True, slots=True`.
- `type` aliases (PEP 695 style) for union types: `type Content = str | tuple[ContentPart, ...]`.
- Internal modules prefixed with `_` (e.g., `llm/_types.py`); public API is via top-level re-exports only.

## Research Docs

`research/` contains standalone Markdown reference guides (LLM API guide, agent architectures, Python best practices, graph algorithms). These are separate from the Python package.

## Provider-Specific Gotchas

- Anthropic: `input_schema` for tools (not `parameters`), `system` is a top-level field (not a message role).
- Gemini: `contents`/`parts` structure (not `messages`/`content`), uses NDJSON streaming (not SSE).
- OpenAI has two API surfaces: Chat Completions and Responses API — both have provider implementations.
