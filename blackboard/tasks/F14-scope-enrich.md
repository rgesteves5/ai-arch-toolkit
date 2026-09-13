# F14 · `Scope.enrich` sobre o snapshot filtrado

- **Dono:** coordenador · **Estado:** done · **Depende de:** —
- **Plano:** K, F14

## Problema

`apply_scope` filtra as camadas e depois passa ao `enrich` o snapshot **original**: com
`Scope(exclude={"api_key"}, enrich={"leak": lambda s: s["api_key"]})` o step vê `api_key=None` e
`leak='sk-SECRET'`.

## Mudança

O `enrich` recebe o snapshot já filtrado (e transformado), sem as saídas de outros `enrich`.
`transform` continua a correr por camada — documentar em `docs/flow-architecture.md`.

## Ficheiros

`src/ai_arch_toolkit/toolkit/flow/_scope.py`, `tests/flow/test_scope.py`, `docs/flow-architecture.md`.

## Prova

- `exclude={"api_key"}` + `enrich` que lê `api_key` → o step vê `leak is None`.
- `include={"x"}` + `enrich` que lê `y` → `None`.

## Registo

- Estado: done (2026-09-13)
- Ficheiros tocados: `toolkit/flow/_scope.py` (docstring e `apply_scope`), `docs/flow-architecture.md` (Scope).
- Testes novos: `tests/flow/test_scope.py::TestEnrichSeesWhatTheStepSees` (4). Antes: 3 falhavam; o
  quarto guarda que os enrichers não se vêem uns aos outros.
- Verificações: 2650 passed; ruff limpo; pyright 0 erros.
- CHANGELOG (Fixed): `Scope.enrich` lê o snapshot já filtrado e transformado, e deixa de ler chaves
  que o próprio scope exclui.
