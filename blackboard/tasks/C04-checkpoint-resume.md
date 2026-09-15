# C04 · Checkpoint e retoma de runs

- **Dono:** por atribuir · **Estado:** todo · **Depende de:** nada (em série com C01 no motor)
- **Origem:** `agentes-app-toolkit-review.md` L6 (`:309-330`), "O que escapou" 7 (`:486`), prioridade
  4 (`:514`); `toolkit-fix-plan.md` §4 item 10 (`:648`)
- **Decisões:** por fixar (ver abaixo)

## Problema

- O cursor vive em variáveis locais do motor (`toolkit/flow/_executor.py:329,333,370-377`); `_FlowRun`
  (`:206-247`) não aceita posição, resultados nem trace anteriores. `State.from_trace()` dá o estado
  inicial (`core/_state.py:230-237`).
- `State.to_dict()` não é JSON: o react guarda `Response`, `ToolResult` e o objecto do SDK em
  `messages[i]["_raw"]` (`agents/flows/_react.py:94-102,165-175`; `core/_response.py:186`), que o
  Gemini e a Meta relêem (`_providers/_gemini.py:190`, `_meta.py:165`); o LATS guarda `_MCTSNode`
  com `parent` (`_lats.py:26-35,269`), o ToT uma `deque` de tuplos (`_tot.py:216`).
- O `MeterStore` nasce a zero e conta o tempo desde que nasce (`core/_metering/_store.py:193,202,299`).
  Cancelar a meio de uma chamada LLM deixa custo desconhecido (`core/_llm.py:714-718`), que sob
  `max_cost` nega o resto (`toolkit/budget/_controller.py:102-108`).
- O executor governado não conhece o run nem regista chamadas (`core/_tools/_executor.py:311-368`);
  `ApprovalRequest` não leva `tool_call.id` (`core/_tools/_approval.py:21-34`).

Reprodução (provider falso, sem rede):

```text
json.dumps(react_state.to_dict())   -> TypeError: Object of type SdkLike is not JSON serializable
Flow(a, b, c) parado depois de a; run() de novo sobre o estado  -> a=2, b=1, c=1
react parado a meio de execute_tools; run() desde o fim de llm_call -> charge_card executa 2 vezes
BudgetPolicy(max_llm_calls=2): 1 chamada, parar, run() de novo  -> 1 + 2 = 3 chamadas
aclose() a meio de uma chamada LLM  -> llm_calls=1, unknown_cost_count=1
```

## Objectivo

Um run de `Flow` — e portanto de `Agent` — grava em fronteiras seguras um `FlowCheckpoint` só com
JSON, e `Flow.resume()` continua desse ponto com o mesmo tecto de budget, sem repetir steps concluídos
nem tools já executadas. O toolkit define formato, codecs, cursor, semântica e o protocolo
`CheckpointStore` com uma implementação em memória; a app fornece store durável, migrações, política
depois de crash e UI. O fingerprint e o cursor ficam genéricos para o C09 os reutilizar.

## API proposta

```python
cfg = CheckpointConfig(store=store, codecs=CodecRegistry.default(), tool_journal=store)
execution = flow.iter(state, checkpointing=cfg)     # idem run, iter_sync, Agent.run/iter
async for event in execution:
    if event.type == "checkpoint":                  # gravado antes de o evento sair
        ui.saved(event.checkpoint.run_id, event.checkpoint.sequence)
    if ui.pause_requested:
        execution.stop()                            # acaba na fronteira seguinte, sem cancelar
paused = execution.result.checkpoint                # FlowCheckpoint | None (só num run parado)

result = await flow.resume(paused, checkpointing=cfg, world={"db": db}, config=run_config)
# idem flow.resume_iter/resume_sync/resume_iter_sync e agent.resume/resume_iter/resume_sync
```

- `FlowCheckpoint` (frozen, só JSON, `to_dict`/`from_dict`): `version`, `run_id`, `sequence`,
  `flow_name`, `fingerprint`, `toolkit_version`, `cursor`, `state` (sem `world`), `results`, `steps`
  (`StepTrace.to_dict()` em modo `redacted`), `meter` (`MeterSnapshot.to_dict()`), `elapsed_s`.
- `CheckpointStore`: `async save`, `async load(run_id)`; `InMemoryCheckpointStore` também implementa o
  `ToolJournal` do core. Erros: `CodecError`, `CheckpointError`, `CheckpointMismatchError`.
- Cursor: `{mode: "sequential", iteration, next_index, any_executed}` ou `{mode: "dag", status,
  wave: {launched_at, finished}}`. `resume` valida versão e fingerprint, descodifica, junta `world`,
  repõe cursor, resultados, trace e tempo; serve qualquer checkpoint gravado, também o anterior a um
  `budget_exceeded` ou timeout, se a app der mais budget ou tempo.

## Decisões a fixar antes de codificar

1. **Fronteiras.** (a) a app guarda artefactos e reentra no step seguinte (L6: só serve sequências
   simples); (b) início, depois do merge de cada step executado e fim de cada vaga, mais cada irmão que
   termina nela; (c) dentro de steps. Recomendo (b): só o motor sabe quando o estado está fundido, e
   um step é opaco (D1a).
2. **Step interrompido.** (a) repeti-lo (o repro do `charge_card`); (b) repeti-lo com journal de
   tools. Recomendo (b): o motor liga o `ToolJournal` por `ContextVar` (como `bind_meter`) e
   `_arun_tool` consulta-o antes dos gates, chave `(run_id, fronteira de lançamento, step, tool_call.id,
   nome, digest dos args)`. Gravado → devolvido sem gates nem meter, salvo falha `retryable`. Iniciada
   sem resultado → repete se `risk_level="low"` sem `requires_approval`, senão
   `ToolResult.failure("interrupted")` e o modelo decide (D4). Só no caminho async.
3. **Loops internos.** O `react` é cíclico de dois steps (`_react.py:183-192`): retoma por chamada LLM
   e por lote de tools. `reflexion`, `generate_review`, `self_discovery`, `rewoo` e `tot` retomam nas
   fronteiras externas e repetem o react interno; `plan_execute` e `llm_compiler` (um só step,
   `_llm_compiler.py:202`) repetem o step que planeia e executa. Recomendo aceitar na v1, documentar
   por estratégia e dar ao LATS um codec de `_MCTSNode`.
4. **Codecs.** (a) pickle; (b) `to_dict` avulsos; (c) registo explícito, JSON etiquetado. Recomendo
   (c): `{"$codec": nome, "v": n, "data": …}`; embutidos `tuple`, `deque`, `set`, `bytes`, `Usage`,
   `ToolCall`, `Response`, `Result`, `ToolResult`; pydantic só por `register_model(cls, name=)`; nada
   se importa do payload. Sem codec, ciclo, chave não-string ou float não finito → `CodecError` com o
   caminho (`operational.mcts_root`) ao gravar. `world` fica de fora; o trace vai em modo `redacted`.
5. **`Response.raw`.** (a) largar; (b) codec por fornecedor. Recomendo (b) para os SDKs pydantic, com
   a classe numa tabela fixa: o Gemini reenvia thought signatures e a Meta o raciocínio (D12);
   `thought_signature` sobrevive a `model_dump(mode="json")` + `model_validate` (verificado). O xAI
   (protobuf) larga `raw`, que o adaptador não relê.
6. **Mesmo flow.** Callables não se identificam de forma fiável (todos os wrappers de `as_step` são
   `Flow.as_step.<locals>._run_flow`, `_flow.py:336`). Recomendo `sha256` do JSON canónico `{name,
   version, mode, steps: [{name, after, conditional}]}`, com `Flow(version=)` opcional; policy,
   timeout, budget, scope, `trace_capture`, `max_parallelism` e `max_iterations` ficam fora, para
   retomar com mais tempo ou budget. Divergência → `CheckpointMismatchError` com as diferenças, sem
   `force`. `json_fingerprint()` sai de `_manifest.py:968-976` sem mudar os fingerprints dos manifestos.
7. **Budget e tempo.** (a) controller que soma o anterior; (b) semear o store. Recomendo (b):
   `MeterScope(config, baseline=)` inicia os contadores da raiz e recua `started_at`, e a revalidação
   sob o lock (`_store.py:353-406`) vê o total (com (a) só veria o segmento). O checkpoint leva
   contagens, custo em pico e desconhecidos; operações vivas contam como no `close()`. Tempo activo:
   `max_wall_s` e `Flow(timeout=)` somam segmentos, pausas não contam. Gasto após a última fronteira:
   a app reconcilia pelo `UsageSink`. Sob scope herdado, `resume` com gasto levanta `CheckpointError`.
8. **Controlo.** Recomendo: `save` no motor, antes do evento (D9); `save` falhado → `checkpoint_error`
   no trace e run parado com `result.checkpoint` por gravar (não levanta, como no 8a); `stop()`
   gracioso; `aclose()` cancela e vale o último gravado; sem `inject()` (num evento fora de vaga nada
   corre e pode escrever-se em `execution.state`). Aprovação de horas: o handler guarda o pedido, a
   app faz `aclose()` e depois `resume`, e o handler responde pela decisão guardada com
   `ApprovalRequest.tool_call_id` (novo); não há gasto, porque a tool só abre operação depois dos
   gates (`core/_tools/_executor.py:325-350`).
9. **Store e síncrono.** Recomendo protocolo e `InMemoryCheckpointStore`, sem store em ficheiro
   (atomicidade, locks e retenção são da app); `resume_sync` e `SyncFlowExecution.stop()`.

## Sub-tarefas, por ordem

- **C04a** Codecs (`core/_codecs.py`), `json_fingerprint`, `MeterSnapshot.to_dict`/`from_dict`.
- **C04b** Motor: `run_id` (o de `FlowExecution` do C01c, se entrar antes), cursor explícito,
  fronteiras, `CheckpointConfig`, evento `checkpoint`, `FlowResult.checkpoint`, `stop()`,
  `Flow(version=)`, `flow_fingerprint()`. Ainda sem retoma.
- **C04c** Retoma sequencial e cíclica: `Flow.resume*`, validação, `world=`, trace, resultados, tempo.
- **C04d** Budget: `MeterScope(baseline=)`, `budget_scope(baseline=)`, meter no checkpoint.
- **C04e** Journal de tools no executor governado; `ApprovalRequest.tool_call_id`.
- **C04f** `Agent`: `checkpointing=`, `resume*`, `stop()`, codecs de `_MCTSNode` e de `output_schema`.
- **C04g** DAG (no fim: nenhuma estratégia o usa): estado por nó, vagas, irmãos, ordem de merge.

## Ficheiros

- Novos: `src/ai_arch_toolkit/core/_codecs.py`, `core/_fingerprint.py`, `core/_tools/_journal.py`,
  `toolkit/flow/_checkpoint.py`.
- `toolkit/flow/_executor.py`, `_flow.py` (partilhados com C01); `toolkit/agents/_agent.py`
  (partilhado com C01, C02); `toolkit/agents/flows/_lats.py` (partilhado com C01, C02, C05);
  `toolkit/agents/_manifest.py` (partilhado com C05; só importa o helper); `core/_tools/_approval.py`
  (partilhado com C07); `core/_tools/_executor.py`; `core/_metering/_admission.py`, `_store.py`,
  `_scope.py`; `toolkit/budget/_scope.py`.
- Exports: `core/__init__.py`, `core/_tools/__init__.py`, `toolkit/flow/__init__.py`,
  `toolkit/__init__.py`, `ai_arch_toolkit/__init__.py` (partilhados com C01, C02, C05, C06).
- Testes (em `tests/`): `test_codecs.py`, `test_fingerprint.py`, `test_tools_journal.py`,
  `test_core_exports.py` (partilhado com C02, C05, C06), `flow/test_checkpoint.py`,
  `flow/test_resume.py`, `flow/test_resume_dag.py`, `metering/test_baseline.py`,
  `agents/test_agent_resume.py`.
- Docs: `flow-architecture.md` (partilhado com C01), `agents.md` (com C01, C02, C05), `safety.md`
  (com C02, C05, C07, C08), `tools.md` (com C02, C03, C05, C07, C08), `api.md` (partilhado).

## Prova

- Codecs: estado de um react com `raw` pydantic volta igual e passa `json.dumps` (hoje `TypeError`);
  `_MCTSNode` sem codec → `CodecError` com `operational.mcts_root`; codec desconhecido → erro sem import.
- Checkpoint: fronteiras e cursor por modo; evento depois do `save`; `save` que levanta →
  `checkpoint_error`; `stop()` → `result.checkpoint`; dois `react_flow(...)` dão o mesmo fingerprint.
- Retoma: `Flow(a, b, c)` parado depois de `a` → corre só `b`, `c` (hoje `a=2`); cíclico cumpre
  `max_iterations` no total; react retomado após `llm_call` → mesmo texto, steps e chamadas LLM do run
  contínuo; `Flow(timeout=)` conta o segmento anterior; outra estrutura → `CheckpointMismatchError`.
- Budget: `max_llm_calls=2`, 1 chamada, parar, retomar → a 3.ª dá `budget_exceeded` (hoje passa);
  desconhecido com `max_cost` → negado; `max_wall_s` vê o tempo anterior; scope herdado → erro.
- Journal: repro do `charge_card` → uma execução (hoje duas); tool de risco alto iniciada sem
  resultado → `interrupted`; falha `retryable` repete; decisão guardada por `tool_call_id` responde.
- Agent: as dez estratégias com LLM falso de respostas fixas, `stop()` na primeira fronteira e retoma
  → mesmo `AgentResult.text` e mesmas chamadas LLM. DAG: três irmãos com conflito `last_wins`, crash
  depois do primeiro → corre só os outros; estado e trace iguais aos do run contínuo.

## Fora do âmbito

- Store durável, migrações, retenção, política depois de crash e UI: app. Agendador (L2), `inject()`.
- v2: reproduzir chamadas LLM dentro de um step (retoma exacta de `plan_execute`, `llm_compiler`,
  react internos); checkpoints de flows aninhados e incrementais.
- `ToolGroup.max_calls` entre segmentos: é por instância; o tecto do run é `BudgetPolicy.max_tool_calls`.

## Riscos

- O motor (F08) é o ficheiro mais sensível e o C01 mexe nele: aplicar em série. O estado inteiro por
  fronteira cresce com o quadrado dos turnos (como I/N7): medir 10 → 20 turnos no C04b.
- O checkpoint leva prompts e resultados inteiros (não passa pelo `Redactor`): a app cifra-o.
- Mudar chaves de estado de uma estratégia parte runs parados ("Changed" no `CHANGELOG`); `when`
  impuros decidem diferente na retoma; SDK novo pode recusar `raw` antigo (erro claro, nunca vazio).

## Registo do dono

- Estado: todo
- Ficheiros tocados:
- Testes novos:
- Verificações:
- CHANGELOG proposto:
- Desvios ao plano:
