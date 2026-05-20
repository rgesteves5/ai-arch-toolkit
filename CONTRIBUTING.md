# Contributing

Thanks for the interest — this guide covers how to get a working dev
environment, the conventions the codebase follows, and the patterns for
extending the framework.

## Setup

```bash
git clone https://github.com/rgesteves5/ai-arch-toolkit.git
cd ai-arch-toolkit
uv sync --extra dev          # all providers + dev tools (ruff, pyright, pytest, pre-commit)
uv run pre-commit install    # ruff + hygiene hooks on every commit
```

For running the examples or integration tests, copy `.env.example` to `.env`
and populate `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, and/or
`XAI_API_KEY`. Load them with `set -a && source .env && set +a` or via
`direnv`.

## Day-to-day commands

```bash
uv run pytest                                # full suite (1200+ tests, ~5s)
uv run pytest tests/test_llm.py              # one file
uv run pytest -k "stream"                    # by pattern
uv run pytest -m integration                 # live API tests (need keys)

uv run ruff check src tests examples         # lint
uv run ruff format src tests examples        # auto-format
uv run pyright src                           # type-check (standard mode)
uv run pre-commit run --all-files            # everything pre-commit will run
```

CI runs three jobs in parallel — `lint`, `typecheck` (non-blocking until the
error count is driven down), and `test` across Ubuntu and macOS on Python
3.13.

## Code conventions

- Python **3.13+** with `from __future__ import annotations` in every file.
- Ruff: line length 99; selected rules `E F W I UP B SIM RUF`. `ruff format`
  is the formatter — never hand-format.
- All dataclasses are `frozen=True, slots=True` (`kw_only=True` once they
  reach 3+ fields).
- PEP 695 `type` aliases for shared shapes:
  `type Content = str | list[ContentPart]`.
- Every package `__init__.py` declares `__all__`; internal modules are
  `_`-prefixed and reach the world only via re-export.
- Docstrings are Google-style. Don't restate types — the annotations are the
  truth.
- Toolkit tools return error strings (never raise) so agents can recover
  gracefully.

## Adding a provider

1. Add a class extending `BaseProvider` in `src/ai_arch_toolkit/core/_providers/_<name>.py`.
   Implement `complete()`, `stream()`, `stream_events()`, plus
   `to_provider_messages()`. Look at `_openai.py` as the most thorough
   reference.
2. Wire model-prefix routing in `core/_providers/__init__.py::create_provider()`.
3. If the provider has its own SDK, declare it in `[project.optional-dependencies]`
   in `pyproject.toml` (and the `dev` extra so CI has it) — never as a hard
   dependency.
4. Pricing: add the model prefixes to `core/_default_pricing.toml`.
5. Tests in `tests/test_<name>_provider.py`. Mock `urllib.request.urlopen` or
   the SDK's HTTP layer; mirror the response-shape fixtures the other
   provider tests use.
6. Update `docs/model-compatibility.md` and the README provider × feature
   matrix.

## Adding a toolkit tool

1. Drop it under `src/ai_arch_toolkit/toolkit/tools/_<file>.py`.
2. Decorate with `@tool` from `ai_arch_toolkit.core` — the schema is inferred
   from type hints + Google-style docstring.
3. Stdlib only. If an HTTP call is needed, use `urllib`; the existing
   weather/geo/wikipedia tools are the template.
4. **Return error strings, never raise.** Agents read the return value as the
   tool result.
5. Export from `toolkit/tools/__init__.py`.
6. Tests in `tests/toolkit/test_<file>.py`. Use the `mock_post` fixture or
   patch `urllib.request.urlopen`; use `tmp_path` for filesystem tools.

## Adding an agent flow

1. New module under `src/ai_arch_toolkit/toolkit/agents/flows/_<name>.py`.
2. Expose a `<name>_flow(...)` factory that builds and returns a `Flow`, plus
   a `<name>_initial_state(task)` helper that returns the operational state.
3. Build on the core primitives — `LLM`, `ToolGroup`, `State`, `Step`,
   `Result` — and on existing flow factories where possible. `react_flow` is
   the simplest reference; `lats_flow` shows search-based composition.
4. Add a numbered example under `examples/` that runs end-to-end with a real
   model.
5. Tests in `tests/agents/flows/test_<name>.py`. Mock `LLM.complete` with an
   `AsyncMock` and feed it a `side_effect` of pre-built `Response` objects
   from the `make_response` factory in `tests/agents/conftest.py`.
6. Mention it in the README matrix and `docs/agents-and-capabilities.md`.

## Commit + PR format

- Short imperative subject (under ~70 chars). Match the existing style:
  `Add X`, `Fix Y`, `Update Z`.
- Body explains the **why**, the surface area, and what tests/docs were
  touched.
- Add an `[Unreleased]` entry to `CHANGELOG.md` (Added/Changed/Fixed) when
  the change is user-visible.
- Open the PR against `main` with a summary, a test plan, and links to any
  related issues.

## Tests must pass before pushing

Run `uv run pytest`, `uv run ruff check src tests examples`, and
`uv run ruff format --check src tests examples` locally. `pre-commit` runs
the lint/format hooks on every commit, but the full test suite is still on
you.

Pyright is currently non-blocking in CI while the existing standard-mode
warnings (≈200) are driven down. If you add new code, please keep it
pyright-clean — and consider fixing a handful of pre-existing warnings while
you're in the area.
