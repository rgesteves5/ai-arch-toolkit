# F11 · Metadados de risco em `tools.dangerous`

- **Dono:** worker-C · **Estado:** done · **Depende de:** —
- **Plano:** C, F11 · **Decisões:** D4

## Problema

Das sete tools de `ai_arch_toolkit.toolkit.tools.dangerous`, cinco usam `@tool` nu (`capability=None`,
`risk_level="low"`, `requires_approval=False`): `read_file`, `list_directory`, `search_files`
(`src/ai_arch_toolkit/toolkit/tools/_filesystem.py:13,38,76`) e `http_get`, `scrape_text`
(`src/ai_arch_toolkit/toolkit/tools/_web.py:56,79`). `run_command` e `python_repl` já declaram risco.

## Decisão (D4)

- Filesystem: `@tool(capability="filesystem", risk_level="high", requires_approval=True, approval_reason=…)`.
- Web: `@tool(capability="network", risk_level="high", requires_approval=True, approval_reason=…)`.
- Quebra assumida: usá-las num `ToolGroup` sem `approval_handler` passa a dar `approval_denied`.

## Ficheiros

- `src/ai_arch_toolkit/toolkit/tools/_filesystem.py`, `src/ai_arch_toolkit/toolkit/tools/_web.py`
- Testes destas tools (procura em `tests/`); onde forem executadas por um grupo ou pelo executor sem
  handler, acrescentar um handler que aprova.
- `docs/safety.md` — **só** a secção "Dangerous tools" (o worker-B edita a secção "Gates").
- `docs/tools-catalog.md` se listar risco; `examples/24_toolkit_tools_showcase.py` se as executar por
  um grupo.
- Verificar se `src/ai_arch_toolkit/toolkit/tools/__init__.py` as reexporta fora de `dangerous`; se
  sim, registar em `FINDINGS.md` sem mudar exports.

## Prova

- As cinco políticas afirmadas campo a campo.
- `ToolGroup(read_file).execute(tc)` sem handler → `approval_denied`; com handler que aprova e um
  ficheiro em `tmp_path` → conteúdo lido.
- `http_get` idem, com `urllib.request.urlopen` mockado.

## Registo do dono (transcrito pelo coordenador; o worker não conseguiu escrever fora da worktree)

- Estado: done — feito pelo worker-C na worktree `.claude/worktrees/agent-abbcbffc136b5debb`.
- Ficheiros tocados: `toolkit/tools/_filesystem.py`, `toolkit/tools/_web.py`, `tests/toolkit/test_tools_exports.py`, `tests/toolkit/test_filesystem.py`, `tests/toolkit/test_web.py`, `docs/safety.md` (só "Dangerous tools", com tabela de capability/risco), `docs/tools-catalog.md` (uma frase).
- Testes novos: `test_dangerous_tools_declare_risk_and_require_approval[7 tools]` (cobre as sete, para uma tool nova sem aprovação falhar), `TestReadFileGovernance` (2), `TestHttpGetGovernance` (2).
- CHANGELOG (Changed, quebra): todas as tools de `toolkit.tools.dangerous` exigem aprovação; sem `approval_handler` dão `approval_denied`.
- Desvios e achados: `capability="network"` segundo D4 (o plano 3.2 dizia `web`). Nenhuma tool perigosa é reexportada fora de `dangerous`; nenhum teste ou exemplo precisou de handler. Achado C4 (nanope) em `FINDINGS.md`.
- Aplicado pelo coordenador ao checkout principal em 2026-09-13; suite integrada: 2812 passed, 7 skipped; ruff e pyright limpos.
