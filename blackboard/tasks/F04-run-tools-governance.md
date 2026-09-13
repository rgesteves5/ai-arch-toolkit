# F04 · `run_tools` usa a governança do `ToolGroup`

- **Dono:** worker-B · **Estado:** done · **Depende de:** —
- **Plano:** "run_tools perde gates", F4 · **Decisões:** D10

## Problema

`src/ai_arch_toolkit/toolkit/_runner.py::_normalize_tools` extrai `group.tools` e chama o executor
livre, que só instala `ApprovalGate`. Medido: grupo com `DangerousToolGate(blocked=["danger"])` e
`max_calls=0` → `group.async_execute` bloqueia (`dangerous_tool_blocked`); `run_tools(resp, group)`
devolve `'EXECUTED'`.

## Decisão (D10)

Com um `ToolGroup`, `run_tools`/`run_tools_sync` executam por `group.async_execute`/`group.execute`.
Tool desconhecida continua a levantar `KeyError` (contrato existente e documentado). Passar
`approval_handler=` junto com um `ToolGroup` levanta `ValueError` (o handler pertence ao grupo).
Listas de callables não mudam.

## Ficheiros

- `src/ai_arch_toolkit/toolkit/_runner.py`
- `tests/test_runner.py`
- `docs/tools.md` — secção "run_tools helper": uma frase sobre a governança do grupo

## Prova

- Grupo com `DangerousToolGate` → o conteúdo do `tool_result` contém `dangerous_tool_blocked`.
- `ToolGroup(..., max_calls=1)` com duas chamadas na resposta → a segunda dá `max_calls_exceeded`.
- É usado o `approval_handler` do grupo.
- `run_tools(resp, group, approval_handler=h)` → `ValueError`; idem em `run_tools_sync`.
- Os testes existentes com listas continuam verdes.

## Registo do dono

- Estado: review (feito na worktree `.claude/worktrees/agent-a120b7265ce857942`, por commitar).
- Ficheiros tocados:
  - `src/ai_arch_toolkit/toolkit/_runner.py` — removido `_normalize_tools`.
    `_reject_handler_with_group` levanta `ValueError` antes de tudo (também sem tool calls). Com um
    `ToolGroup`, cada chamada: `_require_in_group` (`KeyError` se desconhecida) e depois
    `tools.async_execute(tc)` / `tools.execute(tc)`. Listas inalteradas (`_resolve_fn` +
    `async_execute_tool` / `execute_tool`). Docstrings do módulo e das funções actualizadas, com
    `Raises`.
  - `tests/test_runner.py`.
  - `docs/tools.md` — secção "run_tools helper": uma frase sobre a governança do grupo e o
    `ValueError`.
- Testes novos (`tests/test_runner.py::TestRunToolsUsesGroupGovernance`): `test_group_gate_blocks`,
  `test_group_gate_blocks_sync`, `test_group_max_calls_blocks_the_second_call`,
  `test_group_max_calls_blocks_the_second_call_sync`, `test_group_approval_handler_is_used`,
  `test_group_approval_handler_is_used_sync`,
  `test_approval_handler_next_to_a_group_raises[with_tool_calls]`,
  `test_approval_handler_next_to_a_group_raises[without_tool_calls]`,
  `test_approval_handler_next_to_a_group_raises_sync`, `test_unknown_tool_still_raises_key_error`,
  `test_unknown_tool_still_raises_key_error_sync`, `test_group_calls_are_metered`. Antes da
  correcção: a tool bloqueada executava (`'Sunny in NYC'`), a 2.ª chamada corria (`'Sunny in LA'`),
  o handler do grupo perdia-se (`approval_denied`) e não havia `ValueError`. Os dois de `KeyError` e
  o de metering já passavam — fixam o contrato no caminho novo. Os testes existentes com listas
  continuam verdes sem alterações.
- Verificações: suite inteira `2658 passed, 7 skipped` (Python 3.14); `tests/test_runner.py` também
  em Python 3.13; `ruff check` limpo; `ruff format --check` limpo; `pyright src` 0 erros.
- Linhas propostas para o CHANGELOG:
  - *Fixed* — `run_tools()` / `run_tools_sync()` with a `ToolGroup` execute each call through the
    group (`async_execute()` / `execute()`), so its gates, `approval_handler`, and `max_calls` budget
    apply; they were silently dropped before.
  - *Changed* — Passing `approval_handler=` to `run_tools()` / `run_tools_sync()` together with a
    `ToolGroup` raises `ValueError`; configure it on the group (`ToolGroup(..., approval_handler=...)`).
    A plain list of callables still takes it.
- Desvios, decisões propostas, achados:
  - Decisão: o `ValueError` é verificado antes do atalho "sem tool calls", para o erro de uso
    aparecer sempre e não só quando o modelo chama tools (teste parametrizado com e sem tool calls).
  - Mantive a ordem existente — resolver e executar chamada a chamada. Consequência (já existia com
    listas e fica igual com grupos): numa resposta `[tool_conhecida, tool_desconhecida]` a primeira
    corre, com efeitos laterais, antes do `KeyError`, e o seu resultado perde-se. Registado em
    FINDINGS com proposta.
  - O `max_calls` de um grupo continua a ser por instância (não por chamada a `run_tools`), como
    `ToolGroup` documenta.
- Aplicado pelo coordenador ao checkout principal em 2026-09-13; suite integrada: 2812 passed, 7 skipped; ruff e pyright limpos.
