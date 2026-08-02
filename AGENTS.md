# AGENTS.md

<!--
Canonical instructions for all coding agents. CLAUDE.md is just an
`@AGENTS.md` import stub so Claude Code and other agents read this one
file — edit here, never duplicate content into CLAUDE.md.
This comment is stripped from Claude's context and costs no tokens.
-->

ai-arch-toolkit is a Python library with zero required dependencies: a unified LLM client
(Anthropic, OpenAI, Gemini, xAI, and OpenAI-compatible local servers) plus Flow
orchestration, nine agent architectures, budgets/metering, ~130 stdlib-only tools,
graph-backed memory, and a file-backed prompt/resource system.

## Commands

```bash
uv sync --extra dev                          # install dev deps (all providers, lint, pyright)
uv run pytest                                # full test suite
uv run pytest tests/test_llm.py              # single file
uv run pytest -k "pattern"                   # tests matching a pattern
uv run ruff check --fix src tests examples   # lint + auto-fix
uv run ruff format src tests examples        # format — run after every edit
uv run pyright src                           # type-check (CI runs exactly this)
uv run python examples/01_hello_world.py     # run an example (needs API keys)
```

- After editing dependencies in `pyproject.toml`, run `uv lock` — CI fails on `uv lock --check` if the lockfile is stale.
- Examples and `pytest -m live_api` need API keys: `set -a && source .env && set +a` (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY` — wins over `GEMINI_API_KEY` — `XAI_API_KEY`). Hermetic system tests use `pytest -m "integration and not live_api"`.
- Docs: `uv sync --extra dev --extra docs`, then `uv run mkdocs serve`. API reference: `uv run pdoc ai_arch_toolkit -o site/api`.
- One-time setup: `uv run pre-commit install` (ruff hooks on commit).

## Architecture

Two layers under `src/ai_arch_toolkit/`, plus the `ai-arch` CLI (`_cli.py`: `prompt validate|inspect|render`, `agent validate|inspect`) and `nanope/` (WIP Reflex app — not public API, excluded from ruff and pyright):

- **`core/`** — stateless, async-first mechanism layer: `LLM` facade, provider adapters, `@tool`/`ToolGroup` + governance gates, metering, pricing, retry/middleware/policy, general-purpose graph. Zero opinions; **never imports `toolkit/`**.
- **`toolkit/`** — opinionated convenience layer on core: `Agent`/`ReasoningSpec`, agent flow factories, budget, pre-built tools, memory, prompts/resources/knowledge, moderation.

Cross-cutting rules that no single file shows:

- Providers are thin adapters over the official SDKs (`anthropic`, `openai`, `google-genai`, gRPC `xai-sdk`), import-guarded by `require_sdk()` so core stays dependency-free — users install per-provider extras. Don't hand-roll provider HTTP calls.
- `create_provider()` routes by model prefix (`claude-`/`gpt-`/`o1-`/`o3-`/`o4-`/`grok-`/`gemini-`). An unknown model with `base_url=` falls back to the OpenAI-compatible adapter (Ollama, LM Studio, vLLM); a loopback `base_url` needs no API key.
- Everything is async-first; every public coroutine gets a `_sync` wrapper (helpers in `core/_sync.py`).
- The recommended entry point is `Agent(ReasoningSpec(strategy=...), llm, tools)`, which compiles once to a `Flow`. The nine flow factories (`react`, `reflexion`, `rewoo`, `plan_execute`, `tot`, `lats`, `self_discovery`, `llm_compiler`, `generate_review`) are the level below; `completion` (single call, no tool loop) is the tenth strategy. `react`, `completion`, and `generate_review` support `output_schema` (for `generate_review`, it applies only to the generator; enforced in `agents/_compile.py`). Multi-phase strategies take per-phase overrides — LLM/tools as canonical deps (`planner_llm`, `executor_tools`, …), prompts as knobs (`planner_system`, …; a `{tools}` token is the only prompt substitution) — validated per strategy; agent manifests declare them under `strategy.phases` (see `docs/agents.md`).
- Metering vs budget: `core/_metering` is the neutral mechanism, `toolkit/budget` the opinion layer on top. Charges happen only at `LLM.complete/stream/stream_events` and the common tool executor. Three modes: no scope = unmetered; `MeterScope` alone = measure-only; scope + controller = enforce. Nested agent flows share the enclosing scope (one cumulative budget).
- Tool execution goes through the governed executor (`run_tools()`/`run_tools_sync()` or `ToolGroup` execute) — never the raw function; approval gates and metering live there.
- `toolkit/tools` is safe-by-default and stdlib-only. Side-effectful tools (shell, filesystem, Python eval, web fetch) live in `toolkit.tools.dangerous` and must be gated.
- Public API only via `__init__.py` re-exports (`ai_arch_toolkit.core`, `ai_arch_toolkit.toolkit.*`); internal modules are `_`-prefixed.

Deeper reading, in `docs/`: `framework-overview.md` (layer tour), `configuring-agents.md` (end-to-end agent configuration guide), `agents.md` (Agent/ReasoningSpec), `flow-architecture.md` (Flow/State/Step engine), `code-style.md` (practical style calls), `tools-catalog.md` (per-tool list), `safety.md` (governance gates), `internal/metering-plan.md` (metering design). `research/` holds standalone reference guides, separate from the package.

## Conventions

- Python 3.13+ (CI also runs 3.14); `from __future__ import annotations` in every file.
- Ruff, line length 99.
- Dataclasses: `frozen=True, slots=True`; add `kw_only=True` at 3+ fields.
- PEP 695 `type` aliases; `__all__` in every `__init__.py`.
- Google-style docstrings; never repeat types already in hints.
- Toolkit tools return error strings instead of raising, so agents can keep going.
- User-visible changes get an `[Unreleased]` entry in `CHANGELOG.md` (Added/Changed/Fixed).

## Testing

- pytest-asyncio with `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed.
- The test tree mirrors the package (`tests/agents/`, `tests/flow/`, `tests/metering/`, `tests/budget/`, `tests/prompts/`, …). Cross-component system tests live in `tests/integration/` behind `integration`; tests that call real providers also carry `live_api`.
- Provider tests: build fake SDK objects with `SimpleNamespace` and inject a mocked client via `provider._client = AsyncMock()` (Anthropic `messages.create`, OpenAI `chat.completions.create`; streams are async iterators of fake chunk objects). Patch the SDK module where it's imported, e.g. `@patch("ai_arch_toolkit.core._providers._anthropic.anthropic")`. Helper factories: `tests/test_openai_provider.py`, `tests/test_anthropic_provider.py`.
- Agent tests: `make_response()`/`make_tool_call()` factories in `tests/agents/conftest.py`; mock the `LLM` with `AsyncMock` and feed `complete.side_effect` prebuilt `Response` objects.
- Metering/budget tests: use a real `LLM` with a fake `_provider` so the charge site runs — mocking `llm.complete` bypasses metering entirely.
- Toolkit tool tests: mock `urllib.request.urlopen`; use `tmp_path` for filesystem tools.

## Provider gotchas

- **Anthropic**: tools use `input_schema` (not `parameters`); `system` is a top-level param (not a message role); structured output uses native `output_config`, with an automatic prompt-based fallback when the schema exceeds Anthropic's complexity limit.
- **OpenAI**: Chat Completions API only (no Responses API). The same adapter serves OpenAI-compatible servers via `base_url=`; vendor reasoning deltas (`reasoning_content`/`reasoning`) surface as thinking events.
- **Gemini**: `contents`/`parts` request shape (not `messages`/`content`).
- **xAI**: separate gRPC `xai-sdk` adapter (not OpenAI-compat); key from `XAI_API_KEY`.
