# Quadro

## Frente activa: plano de correcção do toolkit

- **Estado:** concluída em 2026-09-13 (F01–F22 done), commitada e publicada em `main`.
- **Plano:** `docs/internal/toolkit-fix-plan.md` — a secção 0 tem os ajustes da revisão cruzada.
- **Origem:** `docs/internal/agentes-app-toolkit-review.md`.
- **Base:** `main` @ `48a43ac`. Baseline: 2608 passed, 7 skipped; pyright e ruff limpos.
- **Commits:** publicados em `main` a pedido do dono (`14e623d`..`cc83cb9` e o registo).
- **Coordenador:** sessão principal. Aplica os diffs das worktrees, corre a suite, escreve o `CHANGELOG`.
- **Não tocar:** `.claude/worktrees/exciting-germain-a41255` é de uma sessão anterior.

### Vaga 1 — independentes (workers em worktrees; coordenador no checkout principal)

| ID | Tarefa | Dono | Estado |
|---|---|---|---|
| F01 | Usage Anthropic: deltas cumulativos e campos `null` | worker-A | done |
| F02 | `Flow.policy` por step e `Flow(timeout=)` | coordenador | done |
| F04 | `run_tools` usa a governança do `ToolGroup` | worker-B | done |
| F05 | Fundir prompts de sistema nos adaptadores | worker-E | done |
| F06 | `ToolGroup` rejeita `ServerTool` e não-callables | worker-B | done |
| F07 | Tools async no caminho síncrono; positional-only | worker-B | done |
| F09 | `RunConfig` por execução no `Agent` | coordenador | done |
| F10 | `provider` nos pedidos de metering | worker-E | done |
| F11 | Metadados de risco em `tools.dangerous` | worker-C | done |
| F12 | Schema: uniões multi-tipo e varargs | worker-C | done |
| F14 | `Scope.enrich` sobre o snapshot filtrado | coordenador | done |
| F15 | Wrappers síncronos: cancelamento e backpressure | worker-D | done |
| F17 | Exportar a superfície de gates | worker-B | done |

### Vagas seguintes — coordenador, no checkout principal

| ID | Tarefa | Estado | Depende de |
|---|---|---|---|
| F03 | Middleware async em streaming | done | F05, F10 (aplicados) |
| F13 | Validar e coagir argumentos antes dos gates; encadear gates; C2/C3 | done | F07, F12 (aplicados) |
| F08 | Motor único de execução (8a–8g) | done | F02, F09 |
| F16 | Política de captura do trace | done | F08 |
| F19 | Gemini: schemas que `types.Schema` rejeita (C1) | done | F12 (aplicado) |
| F20 | `run_tools` verifica todos os nomes antes de executar | done | F04 (aplicado) |
| F18 | Deriva de documentação e `CHANGELOG` | done | todas |
| F21 | Achados restantes: nanope, conteúdo de sistema, `_stream_sync`, `Any`, middleware | done | F01–F20 |
| F22 | Revisão adversarial pós-implementação: 3 revisores, correcções confirmadas | done | F21 |

## Por fazer (dono do repositório)

- **xAI:** repor créditos na conta e depois correr `uv run pytest -m live_api -k xai` (custo por
  pedido) e um probe com uma tool cujo parâmetro seja `Any` (schema sem tipo) e com `system=` +
  `system()` ao mesmo tempo — únicas mudanças desta frente que o xAI ainda não confirmou.

