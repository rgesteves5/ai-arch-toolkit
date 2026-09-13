# F17 · Exportar a superfície de gates

- **Dono:** worker-B · **Estado:** done · **Depende de:** —
- **Plano:** F17

## Problema

`ToolGroup(gates=...)` recebe `ToolGate`; os gates recebem `ExecutionContext` e devolvem
`GateBlock`/`GateModify`/`GateDryRun`. Nenhum destes nomes é exportado por `ai_arch_toolkit.core._tools`,
`ai_arch_toolkit.core` ou `ai_arch_toolkit`: escrever um gate próprio obriga a importar módulos `_…`.

## Mudança

Exportar `ToolGate`, `ExecutionContext`, `GateBlock`, `GateModify`, `GateDryRun` e `GateResult` nos três
`__init__.py` (com `__all__`). Na secção "Gates" de `docs/safety.md`, um exemplo curto de gate próprio.

## Ficheiros

- `src/ai_arch_toolkit/core/_tools/__init__.py`, `src/ai_arch_toolkit/core/__init__.py`,
  `src/ai_arch_toolkit/__init__.py`
- `tests/test_core_exports.py` (actualizar se fixar `__all__`)
- `docs/safety.md` — **só** a secção "Gates" (o worker-C edita "Dangerous tools")

## Prova

- `from ai_arch_toolkit import ToolGate, ExecutionContext, GateBlock, GateModify, GateDryRun, GateResult`.
- Um gate próprio (classe com `check` e `check_sync`) em `ToolGroup(gates=[...])` bloqueia uma chamada, e
  `isinstance(gate, ToolGate)` é verdadeiro.

## Registo do dono

- Estado: review (feito na worktree `.claude/worktrees/agent-a120b7265ce857942`, por commitar).
- Ficheiros tocados:
  - `src/ai_arch_toolkit/core/_tools/__init__.py`, `src/ai_arch_toolkit/core/__init__.py`,
    `src/ai_arch_toolkit/__init__.py` — import e `__all__` de `ToolGate`, `ExecutionContext`,
    `GateBlock`, `GateModify`, `GateDryRun`, `GateResult`.
  - `tests/test_core_exports.py`.
  - `docs/safety.md` — nova subsecção "### Custom gates" dentro de "Gates", a seguir a "### Dry run"
    e antes de "### Executing a single tool call" (longe de "### Dangerous tools", para não colidir
    com o worker-C): o protocolo, o `ExecutionContext`, os três `GateResult`, um exemplo
    (`ReadOnlyGate`) e uma frase sobre gates sem estado e o encadeamento de `GateModify`.
- Testes novos (`tests/test_core_exports.py`):
  `test_gate_surface_is_exported[<nome>-<módulo>]` (6 nomes × 3 módulos: `hasattr` e presença em
  `__all__`), `test_custom_gate_built_from_public_imports_blocks_a_call` (`from ai_arch_toolkit
  import ...`, gate próprio com `check`/`check_sync`, `isinstance(gate, ToolGate)`, bloqueia
  `save_note` e deixa passar `read_notes`). Antes: `AssertionError: ... is not importable` e
  `ImportError`.
- Verificações: suite inteira `2658 passed, 7 skipped` (Python 3.14); `tests/test_core_exports.py`
  também em Python 3.13; `ruff check` limpo; `ruff format --check` limpo; `pyright src` 0 erros.
- Linhas propostas para o CHANGELOG:
  - *Added* — `ToolGate`, `ExecutionContext`, `GateBlock`, `GateModify`, `GateDryRun`, and
    `GateResult` are exported from `ai_arch_toolkit` and `ai_arch_toolkit.core`, so custom gates no
    longer import private modules; see
    [Tool Governance & Safety → Custom gates](docs/safety.md).
- Desvios, decisões propostas, achados:
  - O exemplo da doc foi executado tal como está escrito (script descartável): bloqueia `save_note`
    e deixa passar `read_notes`.
  - A doc descreve o encadeamento actual: cada gate vê os argumentos do modelo, o último
    `GateModify` ganha e, numa tool com aprovação, o gate de aprovação (sempre o último) descarta um
    `GateModify` anterior. Isto é um bug (registado em FINDINGS com reprodução); se for corrigido — o
    F13 mexe nesse ciclo — a última frase de "Custom gates" em `docs/safety.md` tem de mudar.
  - Nota: `GateBlock.error_type` é o `Literal` fechado `GovernanceOutcome`; um gate próprio só tipa
    com um desses valores (o exemplo usa `"dangerous_tool_blocked"`); em runtime aceita qualquer
    string. Não mexi (secção 4.9 do plano).
  - Proposta para F18: `docs/api.md` ("Core — Tools") não lista nenhum nome de governança
    (`DangerousToolGate`, `ApprovalGate`, ...), nem os agora exportados.
- Aplicado pelo coordenador ao checkout principal em 2026-09-13; suite integrada: 2812 passed, 7 skipped; ruff e pyright limpos.
