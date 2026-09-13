# F20 · `run_tools` verifica todos os nomes antes de executar

- **Dono:** coordenador · **Estado:** done · **Depende de:** F04 aplicado
- **Plano:** achado do worker-B (`FINDINGS.md`, "`run_tools` levanta `KeyError` depois de...")

## Problema

A verificação de tool desconhecida era feita chamada a chamada. Numa resposta
`[send_email, does_not_exist]`, `send_email` corria e só depois saía o `KeyError`; o `tool_result` da
primeira perdia-se com a excepção. Igual com lista e com `ToolGroup`.

## Mudança

`_require_known_tools(response, tools)` corre antes do ciclo em `run_tools` e `run_tools_sync`: com
`ToolGroup`, `name not in tools`; com lista, `_resolve_fn`. O contrato do `KeyError` mantém-se; muda
só o momento.

## Prova

- Resposta `[efeito, desconhecida]` → `KeyError` e nenhum efeito, com lista e com `ToolGroup`, nos
  caminhos async e síncrono.

## Registo

- Estado: done (2026-09-13)
- Ficheiros tocados: `toolkit/_runner.py` (`_require_known_tools`; secção `Raises` das docstrings),
  `docs/tools.md` (frase na secção `run_tools`).
- Testes novos: `tests/test_runner_preflight.py` (4). Antes: falhavam com o efeito já feito.
- CHANGELOG (Fixed): `run_tools`/`run_tools_sync` verificam todos os nomes de tools antes de executar
  qualquer chamada; um nome desconhecido já não deixa efeitos de chamadas anteriores sem resultado.
