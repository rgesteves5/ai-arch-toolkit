# F09 · `RunConfig` por execução no `Agent`

- **Dono:** coordenador · **Estado:** done · **Depende de:** —
- **Plano:** F9

## Mudança

`Agent.run`, `run_sync` e `iter` aceitam `config: RunConfig | None`, passado ao `Flow`; tal como em
`Flow.run`, `config` define o meter por inteiro e prevalece sobre `budget_policy`.

## Ficheiros

`src/ai_arch_toolkit/toolkit/agents/_agent.py`, `docs/agents.md`, `tests/agents/`.

## Prova

- `Agent.run(task, config=RunConfig(sinks=[sink]))` → o sink recebe os eventos da chamada LLM.
- `config` com controller que nega → `budget_exceeded`, mesmo com `budget_policy` permissiva.

## Registo

- Estado: done (2026-09-13)
- Ficheiros tocados: `toolkit/agents/_agent.py`, `docs/agents.md` (Budgets).
- Testes novos: `tests/agents/test_agent_runconfig.py` (4). Antes: 4 falhavam (`config` desconhecido).
- Verificações: 2650 passed; ruff limpo; pyright 0 erros.
- CHANGELOG (Added): `Agent.run`, `run_sync` e `iter` aceitam `config=RunConfig(...)` por execução.
