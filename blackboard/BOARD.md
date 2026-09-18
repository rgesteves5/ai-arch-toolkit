# Quadro

## Frente activa: robustez (três fases)

- **Estado:** aberta em 2026-09-17. **Base:** `main` a partir do commit que abre esta frente.
  Baseline: 3070 passed, 22 deselected; ruff, formatação e pyright limpos.
- **Plano:** `docs/internal/hardening-plan.md`. **Regras comuns:** `tasks/R00-rules.md`.
  **Decisões:** D15–D23.
- **R01:** commitada em `main` a pedido do dono (`a16e3f9` tools, `f95c81f` F25, `5a0ab3b` núcleo,
  e o registo); por publicar. Cada commit passa o gate sozinho (3097, 3107 e 3886 passed).
- **Como correr:** uma fase de cada vez, por ordem, cada uma num agente com contexto limpo. A fase
  seguinte só começa depois de o dono rever e commitar a anterior. Os agentes não fazem commits nem
  chamadas a fornecedores.

| ID | Fase | Dono | Estado | Depende de |
|---|---|---|---|---|
| F24 | Quatro contratos pequenos | coordenador | done | — |
| R01 | Núcleo de chamadas: F25 (nove correcções locais), erros tipados, meter com disposições e tecto incerto, pipeline de tentativa única | Codex; Claude (continuação) | done | nada |
| R02 | Fornecedores: ids e preços, contrato de três fases, um adaptador de cada vez (OpenAI, xAI, Gemini, Meta, Anthropic) | — | todo | R01 |
| R03 | Tools, motor de flows e dívida de manutenção | — | todo | R01, R02 |

## Frente em espera: capacidades em falta

- **Estado:** aberta em 2026-09-15. As nove fichas estão escritas; nenhuma tarefa começou.
- **Antes de codificar:** o dono fixa as "Decisões a fixar" de cada ficha. Cada decisão tomada entra
  em `DECISIONS.md` a partir de D15, com o número dado pelo coordenador.
- **Origem:** o que `docs/internal/agentes-app-toolkit-review.md` pediu ao toolkit (L1, L3–L7, L9, L13
  e o ponto D) e que `docs/internal/toolkit-fix-plan.md` §4 (itens 3 e 10) deixou de fora por ser
  âmbito, não contrato partido.
- **Base:** `main` @ `7ebf7ef`; baseline 3045 passed, 22 skipped. **Coordenador:** sessão principal.
- **Fora da frente:** `agent_as_tool` (a delegação é uma tool da app); scheduler, cofre, descoberta
  local, router `Auto` e escolha de arquitectura (app: L2, L8, L12); sandbox de código (L10).
- **Exemplos novos:** levam o próximo número livre (hoje 48), atribuído pelo coordenador ao aplicar.

| ID | Tarefa | Dono | Estado | Depende de |
|---|---|---|---|---|
| C01 | `Agent.stream()`: texto, thinking e tools através das estratégias | — | todo | nada |
| C02 | API pública para tools dinâmicas (`tool_from_schema`) | — | todo | nada |
| C03 | Cliente MCP (`toolkit.mcp`, extra `mcp`) | — | todo | C02 |
| C04 | Checkpoint e retoma de runs | — | todo | nada; aplicar depois do C01 |
| C05 | Server tools: config no fio e `server_tools=` nas estratégias | — | todo | nada; aplicar depois do C01 |
| C06 | Catálogo técnico de modelos no core | — | todo | nada; C06e depois do C01 e do C05 |
| C07 | Tools de escrita tipadas e `FilesystemPolicy` | — | todo | nada; C07c depois do C02 |
| C08 | Pesquisa web local (Brave e Tavily) | — | todo | nada |
| C09 | `FlowSpec` e máquinas de estados | — | blocked | C04a, C04b; formas validadas na app |

### Ordem de aplicação

Há poucas dependências lógicas; a ordem vem sobretudo dos ficheiros partilhados.

| Vaga | Tarefas | Porquê |
|---|---|---|
| 1 | C02, C06a–d, C07 sem C07c, C08 | Ficheiros quase disjuntos: `core/_tools`, módulos novos, `toolkit/tools`. |
| 2 | C01, C03, C07c | C01 mexe no motor, no `_llm.py`, nos adaptadores e nas estratégias (depois do C02a); C03 precisa do C02; C07c toca em `_definition.py` e `_decorator.py` depois do C02. |
| 3 | C04, C05 | Ambas depois do C01: C04 no motor; C05 nos adaptadores, no `_llm.py`, nas estratégias e no `budget/_estimator.py` (depois do C08c). Partilham `_lats.py` e `_manifest.py`: o coordenador aplica em série. |
| depois | C06e, C09 | C06e mexe nos adaptadores; C09 espera pela app. |

Quase todas tocam em `core/__init__.py`, `ai_arch_toolkit/__init__.py`, `docs/tools.md`,
`docs/safety.md`, `docs/agents.md` e `docs/api.md`; o coordenador junta-os ao aplicar.

### Decisões que mudam o contrato público

As primeiras a fixar. As alternativas e as razões estão nas fichas.

- **C01:** os deltas nascem em `LLM.complete(on_event=)`; `FlowEvent` ganha tipos e campos;
  `stream()` é novo e `iter()` não muda.
- **C02:** o handler recebe um `dict`; regra de nomes portátil em `ToolSchema`; nome repetido levanta
  `ValueError` (as duas últimas são quebras visíveis).
- **C03:** `mcp>=2.2,<3`; a ligação vive numa task própria; sem wrappers `_sync` (excepção à regra do
  `AGENTS.md`).
- **C04:** checkpoints só em fronteiras do motor; journal de tools no executor governado;
  `MeterScope(baseline=)` para o budget continuar.
- **C05:** config tipada que falha quando o fornecedor não a aplica; `ReasoningSpec.server_tools`;
  custo por uso na tabela de preços; o OpenAI passa a levantar (quebra visível).
- **C06:** um facto descreve o que funciona através do adaptador; ids exactos e aliases, nunca
  prefixo; os adaptadores não lêem o catálogo.
- **C07:** a policy é verificada no gate e outra vez na tool; hook `@tool(preview=)` e outcome
  `permission_denied` no core.
- **C08:** uma factory por fornecedor, com chave explícita; `capability="web_search"`, aprovação
  obrigatória, em `toolkit.tools`.

### Achados da abertura

Vinte entradas em `FINDINGS.md` (2026-09-15): catorze reproduzidas pelo coordenador, um risco medido e
cinco confirmadas no código e na documentação oficial. Onze não têm tarefa: são correcções, não
capacidades (ver "Por fazer").

## Frente anterior: achados em aberto

- **Estado:** concluída em 2026-09-15 (F23 done), commitada e publicada em `main` a pedido do dono
  (`b504998` tools, `dfaca5c` providers, `19a3905` pytest-timeout, `4509bc9` docs, e o registo).
- **Ficha:** `tasks/F23-open-findings.md`. **Base:** `main` @ `1334fa8`.

| ID | Tarefa | Dono | Estado |
|---|---|---|---|
| F23 | Corrigir os achados que ficaram em `FINDINGS.md` | coordenador | done |

## Frente anterior: fornecedor Meta (Muse Spark)

- **Estado:** concluída em 2026-09-13, commitada e publicada em `main` a pedido do dono (`9588f31` código e testes, `7967287` docs, e o registo).
- **Regra do dono:** nada de testes reais (`live_api`) no CI do GitHub; correm só localmente. O workflow de integração foi removido.
- **Ficha:** `tasks/M01-meta-provider.md`. **Decisões:** D11–D14.
- **Base:** `main` @ `a0807cc`. **Coordenador:** sessão principal (sem workers).

| ID | Tarefa | Dono | Estado |
|---|---|---|---|
| M01 | `MetaProvider` sobre a Responses API do SDK `openai` | coordenador | done |

## Frente anterior: plano de correcção do toolkit

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

- **Anthropic:** quando houver créditos, correr
  `uv run pytest tests/integration/test_provider_contracts_live.py -m live_api -k anthropic -q`
  (usa o `claude-haiku-4-5`, que leva `temperature` no corpo do pedido). Confirma a correcção do
  `temperature` com o `anthropic` 1.x, que só foi provada com o SDK real em loopback.
- **xAI:** repor créditos na conta e depois correr `uv run pytest -m live_api -k xai` (custo por
  pedido) e um probe com uma tool cujo parâmetro seja `Any` (schema sem tipo) e com `system=` +
  `system()` ao mesmo tempo — únicas mudanças desta frente que o xAI ainda não confirmou.
- **Frente C, decisões:** fixar as da vaga 1 (C02, C06, C07, C08) antes de atribuir donos.
- **Plano de robustez (2026-09-17):** `docs/internal/hardening-plan.md` agrupa os achados sem tarefa
  em seis causas e propõe corrigi-las antes da frente C. Um é impeditivo: qualquer chamada LLM falhada
  fica com custo desconhecido e, sob `max_cost`, nega retry, fallback e o resto do run. Falta fixar as
  decisões R1–R15 do plano; só depois se abre a frente com fichas. As dependências já estão actualizadas (2026-09-17; falta correr o CI e verificar os SDKs novos ao vivo). O plano absorve C01a, C02a, C02b,
  C08c e a validade do fio do C05a. Protótipos em `blackboard/prototypes/2026-09-hardening/`.
