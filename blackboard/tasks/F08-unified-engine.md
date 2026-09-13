# F08 · Motor único de execução

- **Dono:** coordenador · **Estado:** done · **Depende de:** F02, F09
- **Plano:** B, E, F, J, H.3, N3, N8, Q, children, F8 · **Decisões:** D5, D9

## Sub-tarefas, por ordem

- **8a** Erros de orquestração (`when`, `Scope.transform`, `Scope.enrich`) viram erro no trace e param o
  flow, em todos os modos.
- **8b** `MeterScope` fora do `State`; `FlowResult` guarda-o.
- **8c** Motor único segundo D9: gerador + task por step, eventos em tempo real, avanço a pedido;
  `iter()` devolve objecto de execução com `.result` (D5); `Agent.iter()` idem.
- **8d** Eventos `retry`/`fallback`/`timeout`/`policy_decision` emitidos pelo step engine.
- **8e** `StepTrace.children` com os traces dos flows aninhados (`as_step` e `inner.run()` dentro de steps).
- **8f** Numa negação de budget numa vaga paralela, processar todos os irmãos antes de levantar.
- **8g** Flow aninhado mantém o span corrente do step pai (N8).

## Prova

- 8a: `{when, transform, enrich}` × `{sequencial, cíclico, DAG single, DAG paralelo}` → `FlowResult`,
  erro no trace, flow parado.
- 8b: `"_meter_scope" not in state.world`; `trace.to_dict(trace_mode="full_debug")` serializável.
- 8c: repro de B igual em `run()` e `iter()`; paralelismo real em `iter()`; `execution.result`;
  `Agent.iter().result.text == Agent.run().text`; `break` sem `aclose()` não corre o step seguinte e
  fecha o scope; `retry` chega ao consumidor antes de o step acabar; exemplos e smoke tests verdes.
- 8d: `retry`, `timeout`, `fallback` observados no stream.
- 8e: `trace.flow("inner")` encontrado; reflexion expõe o react interno.
- 8f: repro de N3 com o step negado primeiro e no meio da vaga → trace e estado concordam.
- 8g: repro de N8 → `Cost exceeded`.

## Registo

- Estado: done (2026-09-13)
- Ficheiros tocados: `toolkit/flow/_executor.py` (reescrito: `_FlowRun`, `FlowExecution`,
  `SyncFlowExecution`), `toolkit/flow/_flow.py` (`FlowResult.meter_scope`; `iter`/`iter_sync`
  devolvem execuções), `core/_step_engine.py` (`on_decision`), `toolkit/agents/_agent.py`
  (`AgentExecution`, `_agent_result`), exports em `toolkit/flow`, `toolkit/agents`, `toolkit` e no
  pacote, `tests/flow/test_flow_metering.py` (lê `meter_scope` em vez de `state["_meter_scope"]`),
  `docs/flow-architecture.md` (Streaming, composição, `children`), `docs/agents.md` (Streaming).
- Desenho (D9): um gerador; cada step numa task; eventos das tasks num deque com um `asyncio.Event`,
  entregues em tempo real; `_wait` usa `asyncio.wait(timeout=…)` para o prazo; o `finally` fecha o
  sub-gerador, cancela e aguarda as tasks e fecha o scope. `step_end` sai do gerador depois do merge
  (step único) ou da task quando acaba (vaga paralela). `_child_traces` (ContextVar) recolhe os traces
  de flows corridos dentro de um step; o wrapper de `as_step()` liga directamente os steps do filho.
  Um run aninhado liga o meter com o span corrente (8g).
- Testes novos: `tests/flow/test_engine.py` (43 casos: 8a×22, 8b×2, 8c×7, 8d×3, 8e×3, 8f×3, 8g×1).
  Antes: falhavam 8a (excepções escapavam/skip silencioso), 8b, 8c (paralelismo, `.result`, retry em
  tempo real), 8d, 8e, 8f e 8g; o teste do `break` sem `aclose()` é guarda de regressão (passava com o
  gerador antigo e continua a passar).
- Verificações: 2691 passed antes da integração dos workers; 2812 depois; testes do motor estáveis em
  3 repetições; ruff limpo; pyright 0 erros.
- CHANGELOG (Fixed): `Flow.iter()` corre as vagas do DAG em paralelo e isola irmãos como `run()`;
  `when`/`Scope.transform`/`Scope.enrich` que levantam deixam de escapar de `Flow.run()` ou de virar
  skip silencioso — o step fica com o erro e o flow pára; negação de budget numa vaga paralela mantém
  os irmãos terminados no trace e no estado; `Policy(max_cost=)` de um step conta o gasto de flows que
  esse step corre.
- CHANGELOG (Added): eventos `retry`/`timeout`/`fallback`/`policy_decision` emitidos enquanto o step
  corre; `StepTrace.children` com os steps de flows aninhados; `FlowResult.meter_scope`;
  `FlowExecution`, `SyncFlowExecution`, `AgentExecution` com `.result`; `step_end.error`.
- CHANGELOG (Changed): `Flow.iter()`, `iter_flow()` e `Agent.iter()` devolvem objectos de execução
  (continuam a funcionar com `async for`); `iter_sync()` devolve `SyncFlowExecution`; o `MeterScope`
  deixa de ser guardado em `State.world["_meter_scope"]`.
