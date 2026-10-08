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

CI checks that `uv.lock` matches `pyproject.toml` before installing
dependencies. Run `uv lock --check` before pushing when dependency metadata
changes, and update the lockfile with `uv lock` when needed.

For running the examples or the `live_api` tests, copy `.env.example` to `.env`
and populate `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY` (`GEMINI_API_KEY` is read
when it is unset), `XAI_API_KEY`, and/or `MODEL_API_KEY` (Meta). Load them with
`set -a && source .env && set +a` or via `direnv`.

## Day-to-day commands

```bash
uv run pytest                                # full suite (5700+ tests; live_api too if keys set)
uv run pytest tests/test_llm.py              # one file
uv run pytest -k "stream"                    # by pattern
uv run pytest -m "integration and not live_api"  # deterministic system tests
uv run pytest -m live_api                    # real APIs (needs keys; costs money; never in CI)

uv run ruff check src tests examples         # lint
uv run ruff format src tests examples        # auto-format
uv run pyright src                           # type-check (standard mode)
uv run pre-commit run --all-files            # everything pre-commit will run
uv lock --check                              # verify pyproject.toml and uv.lock agree
```

CI runs four jobs in parallel — blocking `lint` and `typecheck` jobs, `test`
across Ubuntu and macOS on Python 3.13 and 3.14, and `floors`, which installs
the lowest version of every direct dependency (`--resolution lowest-direct`)
and runs the suite, so the floors in `pyproject.toml` stay true. Hermetic
integration tests run in the normal matrix. Live-provider tests never run in
CI; run them locally.

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
- A toolkit tool that cannot answer raises `ToolFailure`; the executor turns it
  into a failed `ToolResult`, so agents recover gracefully.

For the practical writing style — how code should look, how to structure
docstrings/comments, and when to use classes vs functions — see
[`docs/code-style.md`](docs/code-style.md).

## Adding a provider

1. Add a class extending `BaseProvider` in `src/ai_arch_toolkit/core/_providers/_<name>.py`.
   The base owns `complete()` and `stream()`; the adapter implements `prepare()` (pure, returns
   the SDK's own request type in a `Prepared`, raises `RequestError` for what a model does not
   take), `send()`, `open_stream()`, `assemble()`, `usage()`, and `map_error()` (the only place
   that knows the SDK's exceptions, with each error's `delivery` from the provider's documented
   billing). Put per-model rules in a profile table resolved with `core/_model_id.py`.
   `_anthropic.py` is the reference; `_openai.py` and `_meta.py` show two profiles over a shared
   core (`_responses.py`, the Responses API).
2. Route the model family in `core/_providers/__init__.py`: its prefix in `_MODEL_PREFIXES`, and a
   branch in `create_provider()` that builds the adapter with its environment key.
3. If the provider has its own SDK, declare it as its own extra in
   `[project.optional-dependencies]` in `pyproject.toml` and add that extra to
   `all` (the `dev` extra installs `all`, so CI gets it) — never as a hard
   dependency. Cap the SDK at its next major and keep the floor at a version the
   `floors` CI job passes on.
4. Pricing: add each model id to `core/_default_pricing.toml` (ids billed at the same rates go in
   the entry's `aliases`; dated snapshots are found without an entry).
5. Tests in `tests/test_<name>_provider.py`, through `tests/provider_calls.py`, with the SDK's own
   response types; add the adapter to the parity, conversation, and transport tests
   (`tests/integration/fakeserver.py` serves HTTP and SSE on loopback) and to the wire net
   (`ADAPTERS` and `VALIDATORS` in `tests/wire_contract.py`, which a test checks).
6. Update `docs/model-compatibility.md` and the README provider × feature
   matrix.

## Adding a toolkit tool

1. Drop it under `src/ai_arch_toolkit/toolkit/tools/_<file>.py`.
2. Decorate with `@tool` from `ai_arch_toolkit.core` — the schema is inferred
   from type hints + Google-style docstring.
3. Stdlib only. Reach the network only through `toolkit/tools/_http.py`: declare an `Api` for
   the service's HTTPS origin and read each response inside `parse=`, so a malformed answer
   becomes the tool's failure. If the service explains errors in its answers (a JSON success or
   any error status: its status, headers or body), give its `Api` an `error_reader=`: it takes the
   `Reply` and returns `None`, the error in the service's words, or a typed `ToolFailure` when
   the service says what happened (a missing page is `not_found`), so no `parse` takes an error
   for an empty result; every `Api` on a MediaWiki `api.php` uses `mediawiki_error` (a test
   checks). A call that asks for one resource (a page, an entry, a DOI) passes `missing="…"`, the
   `not_found` message of its 404; any other 404 means the endpoint moved. A call to a service
   that answers "nothing found" with `204 No Content` or an empty body passes
   `allow_empty=True`, one that answers it with a 404 `empty_on_404=True`. The wiki family,
   `_wiki.py`, is the template: the first module migrated to the whole contract (T05). An
   architecture test refuses `urllib.request`, `urllib.error`, `http.client`, `socket` and `ssl`
   anywhere else in the package (`nanope/` aside).
4. Declare the tool's `capability` (`network`, `compute`, …); the invariants test compares it
   with what the tool reaches. Set `max_output_chars`/`timeout_s` on `@tool` when the defaults
   do not fit. Declare numeric limits in the signature, `Annotated[int, Range(1, 25)]`, instead of
   clamping inside the tool.
5. **Never cut without a way on.** Return part of something longer through
   `toolkit/tools/_window.py` — `text_window` (a document by characters, ending on a line),
   `find_window` (the passages around a term), `list_window` (a page the source cut) or
   `page_window` (a page of a list the tool holds) — and return `window.result()`, with a
   `heading=` that says in the tool's words what was read and where. Its footer
   tells the model what was shown, the total, and the exact call that reads on
   (`[chars 0-4000 of 34651 | next: offset=4000]`), and `metadata["window"]` tells the app. Where
   no call can read on (a command's output, results past how deep the source pages), give the
   window a `rest=` that says how to narrow instead. Every tool uses it, and the contract test
   holds each one to it.
6. **Fail with a type, never with a string.** When the tool cannot answer, raise
   `ToolFailure(type, message)` from `ai_arch_toolkit.core`: `not_found` (what was asked for
   does not exist), `validation_error` (an argument is wrong), `upstream` (the source failed or
   explained an error) or `rate_limited`. The message gives the source's reason and the next
   step. Let the door's `HttpError` through (it is a `ToolFailure`): what a status means is
   declared on the request (`missing=` for a 404) or read by the `Api`'s `error_reader`, never
   by catching it (architecture test). Zero results is a successful answer that says so, with
   the query.
7. Export from `toolkit/tools/__init__.py`, or from `toolkit/tools/dangerous.py` for a tool with
   side effects (files, a shell, an evaluator, any URL), which also requires approval.
8. Tests in `tests/toolkit/test_<file>.py`: patch `HTTP_OPEN` with `respond(...)` or
   `http_error(...)` from `tests/toolkit/http_fakes.py` (sockets are blocked there); use
   `tmp_path` for filesystem tools. `tests/toolkit/test_tool_invariants.py` also runs every tool
   against hostile arguments and response bodies, and `tests/toolkit/test_tool_contract.py` holds
   it to the contract (the window, the source's errors, `not_found`, zero results, limits): give
   the tool its kind and cases in `contract_cases.py` and its source's error answers in
   `error_bodies.py`. What a tool does not keep yet is listed in `contract_debt.py`, which only
   shrinks.

## Adding an agent flow

1. New module under `src/ai_arch_toolkit/toolkit/agents/flows/_<name>.py`.
2. Expose a `<name>_flow(...)` factory that builds and returns a `Flow`, plus
   a `<name>_initial_state(task)` helper that returns the operational state. The factory takes
   the `Flow` options as `**options: Unpack[FlowOptions]` and passes them on; the flow leaves its
   answer under `ANSWER` and the response under `RESPONSE` (`flows/_keys.py`); an inner ReAct
   loop runs through `run_react`.
3. Build on the core primitives — `LLM`, `ToolGroup`, `State`, `Step`,
   `Result` — and on existing flow factories where possible. `react_flow` is
   the simplest reference; `lats_flow` shows search-based composition.
4. Add a numbered example under `examples/` that runs end-to-end with a real
   model.
5. Tests in `tests/agents/flows/test_<name>_flow.py`. Mock `LLM.complete` with an
   `AsyncMock` and feed it a `side_effect` of pre-built `Response` objects
   from the `make_response` factory in `tests/agents/conftest.py`.
6. Export it from `flows/__init__.py` and `toolkit/agents/__init__.py`, register it as a strategy
   with `register_strategy(...)` in `toolkit/agents/_builders.py`, and add it to the README's
   Agent architectures table, the strategy table in `docs/agents.md` and
   `docs/agents-and-capabilities.md`.

## Extending resources and prompts

- Resource origin loaders, codecs, selectors, and serializers belong under
  `toolkit/resources/` and must respect `ResourcePolicy`.
- Prompt layouts must return a `LayoutResult` with one valid `SectionSpan` per section,
  subsections included, in preorder.
- Template engines are always explicit, strict on missing values, and must not
  expose arbitrary Python execution.
- Add focused tests under `tests/resources/` or `tests/prompts/`, including
  malformed input, escaping, deterministic ordering, and policy failures.
- Update `docs/prompt-extensibility.md` and add an offline runnable example for
  a new public extension point.

## Commit + PR format

- Short imperative subject (under ~70 chars). Match the existing style,
  Conventional Commits: `feat(tools): …`, `fix(agents): …`, `docs: …`.
- Body explains the **why**, the surface area, and what tests/docs were
  touched.
- Add an `[Unreleased]` entry to `CHANGELOG.md` (Added/Changed/Fixed) when
  the change is user-visible.
- Open the PR against `main` with a summary, a test plan, and links to any
  related issues.

## Tests must pass before pushing

Run `uv lock --check`, `uv run pytest`, `uv run ruff check src tests examples`,
`uv run ruff format --check src tests examples` and `uv run pyright src`
locally. `pre-commit` runs the lint/format hooks on every commit, but the full
test suite is still on you.

Pyright is blocking in CI. New code must remain clean in standard mode.
