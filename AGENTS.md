# Repository Guidelines

## Project Structure & Module Organization
- Core LLM client, providers, and tool execution live in `src/ai_arch_toolkit/core/` (`_providers/` for vendor adapters, `_tools/` for registry + schema).
- Agent helpers and reusable patterns sit in `src/ai_arch_toolkit/toolkit/` and the lightweight `nanope/` utilities; keep `_legacy/` untouched unless backporting fixes.
- Tests mirror the code under `tests/`, docs are in `docs/` (MkDocs), runnable samples in `examples/`, and helper scripts in `scripts/`.

## Build, Test, and Development Commands
- Install deps: `uv sync --dev` (creates `.venv/` automatically).
- Run samples: `uv run python examples/01_hello_world.py`.
- Lint/format: `uv run ruff check src/ tests/ examples/`; `uv run ruff format --check src/ tests/ examples/`.
- Tests: `uv run pytest` or `uv run pytest --cov=src --cov-report=term-missing`.
- Docs: `uv run mkdocs serve` for local docs; `uv run pdoc ai_arch_toolkit -o site/api` for API pages.

## Coding Style & Naming Conventions
- Python 3.13, Ruff line length 99; follow Ruff rules `E,F,W,I,UP,B,SIM,RUF`. Use 4-space indents and keep imports sorted.
- Use type hints for public functions; prefer `snake_case` for modules/vars/functions and `PascalCase` for classes.
- Keep new work in `core/` and `toolkit/`; avoid expanding `_legacy/` unless deleting or fixing defects.

## Testing Guidelines
- Place new tests in `tests/` mirroring module paths; name files and functions `test_*`.
- Leverage `pytest` features already enabled (`asyncio_mode=auto`) for async flows; prefer fixtures over ad-hoc sleeps.
- Add coverage-focused runs (`uv run pytest --cov=src --cov-report=term-missing`) for behavior changes and new providers.

## Commit & Pull Request Guidelines
- Write concise, imperative commits (`Add provider registry tests`); Conventional prefixes (`feat`, `fix`, etc.) are acceptable and appear in history.
- PRs should describe the change, link issues, list commands/tests executed, and mention doc updates or API surface changes. Include textual output over screenshots when possible.

## Security & Configuration Tips
- Keep API keys in `.env`; load with `set -a; source .env; set +a` before `uv run …`. Never commit secrets or `.env`.
- Default configs should be safe-by-default (no live-key usage); document any env vars needed for new integrations.
