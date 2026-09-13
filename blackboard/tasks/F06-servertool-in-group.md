# F06 · `ToolGroup` rejeita `ServerTool` e não-callables

- **Dono:** worker-B · **Estado:** done · **Depende de:** —
- **Plano:** D, F6

## Problema

`ToolGroup(web_search())` → `AttributeError: 'ServerTool' object has no attribute '__name__'`.

## Mudança

`ToolGroup.add` levanta `TypeError` para não-callables. Para um `ServerTool`, a mensagem explica que é
executado pelo provider e que se passa ao lado do grupo: `llm.complete(..., tools=[group, web_search()])`.
Não criar um canal `server_tools=` nas estratégias (fora do âmbito).

## Ficheiros

- `src/ai_arch_toolkit/core/_tools/_group.py`
- `tests/test_tools_group.py`

## Prova

- `ToolGroup(web_search())` e `group.add(web_search())` → `TypeError` com `tools=[` na mensagem.
- `ToolGroup(42)` → `TypeError`.

## Registo do dono

- Estado: review (feito na worktree `.claude/worktrees/agent-a120b7265ce857942`, por commitar).
- Ficheiros tocados: `src/ai_arch_toolkit/core/_tools/_group.py` — `ToolGroup.add` verifica
  `isinstance(fn, ServerTool)` e depois `callable(fn)` antes de `_definition_for`; o `__init__` passa
  por `add`, logo os dois caminhos ficam cobertos. `tests/test_tools_group.py`.
- Testes novos (`tests/test_tools_group.py::TestRejectsNonLocalTools`):
  `test_server_tool_in_constructor_raises_type_error`,
  `test_server_tool_via_add_raises_and_leaves_group_unchanged`,
  `test_non_callable_raises_type_error`. Antes da correcção falhavam com
  `AttributeError: 'ServerTool' object has no attribute '__name__'` (e `'int' object ...`).
- Verificações: suite inteira `2658 passed, 7 skipped` (Python 3.14; base 2608); os ficheiros de
  tools também em Python 3.13 (125 passed); `ruff check src tests examples` limpo;
  `ruff format --check` nos ficheiros tocados limpo; `pyright src` 0 erros.
- Linhas propostas para o CHANGELOG:
  - *Fixed* — `ToolGroup(web_search())` / `group.add(code_execution())` raise a `TypeError`
    explaining that server tools run on the provider and are passed next to the group
    (`llm.complete(..., tools=[group, web_search()])`), instead of an `AttributeError`; any other
    non-callable also raises `TypeError`.
- Desvios, decisões propostas, achados:
  - A mensagem nomeia o tipo real (`'code_execution'`), mas o exemplo é sempre
    `tools=[group, web_search()]` (com "e.g."), para não inventar um helper para um
    `ServerTool(type=...)` personalizado.
  - Proposta para F18 (não editei, fora das minhas secções): em `docs/tools.md`, secção
    "Server tools", dizer que vão na lista `tools=` ao lado de um `ToolGroup`, nunca dentro dele.
- Aplicado pelo coordenador ao checkout principal em 2026-09-13; suite integrada: 2812 passed, 7 skipped; ruff e pyright limpos.
