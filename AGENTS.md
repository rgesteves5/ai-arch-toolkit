# AGENTS.md

<!--
Canonical instructions for all coding agents. CLAUDE.md is just an
`@AGENTS.md` import stub so Claude Code and other agents read this one
file — edit here, never duplicate content into CLAUDE.md.
This comment is stripped from Claude's context and costs no tokens.
-->

ai-arch-toolkit is a Python library with zero required dependencies: a unified LLM client
(Anthropic, OpenAI, Gemini, xAI, Meta, and OpenAI-compatible local servers) plus Flow
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
- Examples and `pytest -m live_api` need API keys: `set -a && source .env && set +a` (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY` — wins over `GEMINI_API_KEY` — `XAI_API_KEY`, `MODEL_API_KEY` for Meta). Hermetic system tests use `pytest -m "integration and not live_api"`.
- Docs: `uv sync --extra dev --extra docs`, then `uv run mkdocs serve`. API reference: `uv run pdoc ai_arch_toolkit -o site/api`.
- One-time setup: `uv run pre-commit install` (ruff hooks on commit).

## Coordination

Work that spans several agents or sessions is coordinated in `blackboard/`: read `blackboard/README.md` (protocol) and `blackboard/BOARD.md` before picking up a task. `board/` is a frozen historical board.

## Architecture

Two layers under `src/ai_arch_toolkit/`, plus the `ai-arch` CLI (`_cli.py`: `prompt validate|inspect|render`, `agent validate|inspect`) and `nanope/` (WIP Reflex app — not public API, excluded from ruff and pyright):

- **`core/`** — stateless, async-first mechanism layer: `LLM` facade, provider adapters, `@tool`/`ToolGroup` + governance gates, metering, pricing, retry/middleware/policy, general-purpose graph. Zero opinions; **never imports `toolkit/`**.
- **`toolkit/`** — opinionated convenience layer on core: `Agent`/`ReasoningSpec`, agent flow factories, budget, pre-built tools, memory, prompts/resources/knowledge, moderation.

Cross-cutting rules that no single file shows:

- Providers are thin adapters over the official SDKs (`anthropic`, `openai`, `google-genai`, gRPC `xai-sdk`; Meta ships no SDK, so its adapter drives `openai`), their imports guarded by `with require_sdk(extra):` so core stays dependency-free — users install per-provider extras. Don't hand-roll provider HTTP calls.
- Every adapter implements `BaseProvider` (`core/_providers/_base.py`), which owns `complete`/`stream`: `prepare()` is pure and returns the SDK's own request type (raising `RequestError` for what the model does not take), `send()`/`open_stream()` do the I/O, `assemble()` builds the one `Response`, and `map_error()` is the only place that knows SDK exceptions (architecture test). Per-model rules live in each adapter's profile table, resolved by `core/_model_id.py` (no `startswith` on model ids elsewhere; unlisted models get the current generation's rules). Cite the provider's documentation URL next to each rule.
- `create_provider()` routes by model prefix (`claude-`/`gpt-`/`o1-`/`o3-`/`o4-`/`grok-`/`gemini-`/`chat-`/`muse-spark-`). An unknown model with `base_url=` falls back to the OpenAI-compatible adapter (Ollama, LM Studio, vLLM); a loopback `base_url` needs no API key, and any other host than the provider's own API needs `api_key=` (environment keys are never sent there).
- Everything is async-first; every public coroutine gets a `_sync` wrapper (helpers in `core/_sync.py`).
- The recommended entry point is `Agent(ReasoningSpec(strategy=...), llm, tools)`, which compiles once to a `Flow`. The nine flow factories (`react`, `reflexion`, `rewoo`, `plan_execute`, `tot`, `lats`, `self_discovery`, `llm_compiler`, `generate_review`) are the level below; `completion` (single call, no tool loop) is the tenth strategy. `react`, `completion`, and `generate_review` support `output_schema` (for `generate_review`, it applies only to the generator; enforced in `agents/_compile.py`). Multi-phase strategies take per-phase overrides — LLM/tools as canonical deps (`planner_llm`, `executor_tools`, …), prompts as knobs (`planner_system`, …; a `{tools}` token is the only prompt substitution) — validated per strategy; agent manifests declare them under `strategy.phases` (see `docs/agents.md`).
- Metering vs budget: `core/_metering` is the neutral mechanism, `toolkit/budget` the opinion layer on top. Charges happen only at `LLM.complete/stream/stream_events` and the common tool executor. Three modes: no scope = unmetered; `MeterScope` alone = measure-only; scope + controller = enforce. Nested agent flows share the enclosing scope (one cumulative budget).
- Tool execution goes through the governed executor (`run_tools()`/`run_tools_sync()` or `ToolGroup` execute) — never the raw function; approval gates and metering live there.
- `toolkit/tools` is safe-by-default and stdlib-only. Side-effectful tools (shell, filesystem, Python eval, web fetch) live in `toolkit.tools.dangerous` and must be gated. Tools reach the network only through `toolkit/tools/_http.py` (an `Api` per HTTPS origin, each response read inside its `parse=`), and each declares its `capability`; architecture and invariant tests check both. Every tool call has an output cap and a deadline (`ToolRuntimePolicy`); a `ToolGroup` ceiling only tightens them.
- Flow factories take the four `Flow` options as `**options: Unpack[FlowOptions]` and leave their answer under the keys in `agents/flows/_keys.py` (`answer`, `response`), which is all the runner reads; an inner ReAct runs through `run_react`.
- Agent and prompt manifests have one declared shape each (`agents/_manifest_shape.py`, `prompts/_manifest_shape.py`, kinds in `toolkit/_shape.py`): the loaders check files against it, and the packaged JSON Schemas are generated from it (a test compares; the `schema()` docstring gives the command to rewrite them).
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
- The test tree mirrors the package (`tests/agents/`, `tests/flow/`, `tests/metering/`, `tests/budget/`, `tests/prompts/`, …). Cross-component system tests live in `tests/integration/` behind `integration`; tests that call real providers also carry `live_api`. `live_api` tests cost money and run only locally: CI deselects them (`-m "not live_api"`), so don't add workflows or secrets that call provider APIs.
- Provider tests: call an adapter through `tests/provider_calls.py` (`prepare`, `complete`, `stream`, `assembled`) and read the request from `prepare(...).params`. Build answers from the SDK's own types (`tests/sdk_streams.py` for streams), or serve them from a loopback server: `tests/integration/fakeserver.py` (HTTP/SSE) or `tests/integration/fakegrpc.py` (xAI). Every request a test prepares is checked against the SDK's types by the autouse `wire_log` fixture (`tests/wire_contract.py`); a test that sends a bad request on purpose declares it with `@pytest.mark.wire_contract(tolerate=[regex, ...])`.
- Agent tests: `make_response()`/`make_tool_call()` factories in `tests/agents/conftest.py`; mock the `LLM` with `AsyncMock` and feed `complete.side_effect` prebuilt `Response` objects.
- Metering/budget tests: use a real `LLM` over `tests/fake_provider.py` (`FakeProvider`, `fake_llm`), which honours the provider contract, so the charge site runs — mocking `llm.complete` bypasses metering entirely.
- Toolkit tool tests: simulate the network at the tools' one seam, `_http._open` (`HTTP_OPEN`, `respond`, `http_error` in `tests/toolkit/http_fakes.py`); every test under `tests/toolkit/` runs with sockets blocked and the throttle's waits recorded (`tests/conftest.py`). Use `tmp_path` for filesystem tools.

## Provider gotchas

- **Anthropic**: tools use `input_schema` (not `parameters`); `system` is a top-level param (not a message role); structured output uses native `output_config`; prompt-based output is available only with `structured_output_mode="prompt"` for schemas that exceed Anthropic's complexity limit. `max_tokens` is required. The Claude 5 family and Opus/Sonnet 4.6+ think adaptively (`thinking=True` → `{"type": "adaptive", "display": "summarized"}`, effort in `output_config.effort`); only the closed list of 4.5-and-older models takes `budget_tokens`, added to `max_tokens`. `anthropic` 1.x dropped `temperature`/`top_p`/`top_k` from its params: they travel in `extra_body`. A turn's tool results go in one `user` message.
- **OpenAI**: Chat Completions API only (no Responses API). The same adapter serves OpenAI-compatible servers via `base_url=`; vendor reasoning deltas (`reasoning_content`/`reasoning`) surface as thinking events. The official host gets `max_completion_tokens`, other servers `max_tokens` and no OpenAI model rule; `thinking_effort` is sent only with `thinking=True`.
- **Gemini**: `contents`/`parts` request shape (not `messages`/`content`); the SDK always sends through `httpx` (the adapter's transport; never `aiohttp`, which re-sends on its own). Known issue, fixed in code but not yet confirmed live: a turn's tool results now go in one `user` content with the call `id`s, and a streamed turn replays whole — keep avoiding Gemini for multi-tool agents until the owner's live run (`tests/integration/test_provider_hardening_live.py -k gemini`) passes (`docs/model-compatibility.md`).
- **xAI**: separate gRPC `xai-sdk` adapter (not OpenAI-compat); key from `XAI_API_KEY`. `prepare()` builds the request with the SDK's own `chat.create` (local, no RPC), so `Prepared.params` is the SDK's `Chat`. Grok models reason unasked: `thinking_effort` applies on its own where the model documents one; gRPC codes map by `google.rpc.Code`.
- **Meta**: Responses API through the `openai` SDK at `https://api.meta.ai/v1`; key from `MODEL_API_KEY`, never `OPENAI_API_KEY`. Requests are stateless (`store: false`): the encrypted reasoning comes back in the response and is replayed from `Response.to_message()["_raw"]`. Muse Spark always reasons — `thinking_effort` applies on its own, `thinking=True` only asks for summaries. Only `tool_choice="auto"` exists (`"none"` sends no tools; forced choices raise). `output_schema` is sent non-strict: Meta constrains decoding anyway, and strict would reject plain Pydantic schemas. The three places the request leaves the `openai` SDK's types are listed atop `_meta.py`, each with its live proof.
