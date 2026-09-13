# F18 · Deriva de documentação e CHANGELOG

- **Dono:** coordenador · **Estado:** done · **Depende de:** todas as outras
- **Plano:** F18

## Mudanças

- `docs/flow-architecture.md:396` — gasto medido no scope partilhado; atribuição por nó via
  `StepTrace.children` (depois de 8e). `:219` passa a ser verdade.
- `docs/moderation.md:62`, `docs/middleware.md:115` — depois de F03.
- `docs/agents.md:53-54` — depois de F02 (e D1a).
- `docs/safety.md` — secção dangerous (F11), Gates (F17), pipeline (F13).
- `docs/tools.md:86` — depois de F07.
- `docs/flow-architecture.md` (Scope) — `transform` por camada, `enrich` sobre o snapshot filtrado.
- `CHANGELOG.md` `[Unreleased]` com as linhas propostas nas fichas.
- `AGENTS.md` aponta para o blackboard.
- Não tocar em `docs/concurrency.md` (já correcto).

## Registo

- Estado: done (2026-09-13)
- Já feitos com cada correcção, verificados agora: `docs/flow-architecture.md` (gasto de flows
  aninhados no meter partilhado, `children`, Scope), `docs/moderation.md`, `docs/middleware.md`,
  `docs/agents.md`, `docs/safety.md`, `AGENTS.md` (secção Coordination).
- Feitos no F18:
  - `docs/tools.md`: `execute()` corre tools `async def` e valida argumentos (proposta do F07);
    server tools na lista `tools=` ao lado do grupo, nunca dentro (proposta do F06); verificação
    prévia de nomes no `run_tools` (F20); nota do Gemini (F19).
  - `docs/api.md`: nomes de governança, resultados, executores e `run_tools` na tabela de tools
    (proposta do F17); `TraceCapture`; `FlowExecution`/`SyncFlowExecution`.
  - Deriva encontrada agora, causada pelo F03: o docstring de `RateLimitMiddleware`
    (`core/_rate_limit.py`) dizia que os streams contornavam o limitador; `docs/memory.md` sugeria
    que a injecção só corria no caminho async. Ambos corrigidos.
  - `CHANGELOG.md` `[Unreleased]`: 6 entradas em Added, 9 em Changed (5 quebras marcadas), 15 em
    Fixed, a partir das fichas F01–F20. Retirada a entrada antiga "`RateLimitMiddleware` docstring
    documents the streaming-bypass limitation", que o F03 tornou falsa.
- Não tocado: `docs/concurrency.md`.
- Verificações finais: `ruff check src tests examples` limpo; `ruff format --check` (410 ficheiros)
  limpo; `pyright src` 0 erros; `pytest` 2901 passed, 7 skipped (2 avisos de `TestAstra`, já no
  `HEAD`).
