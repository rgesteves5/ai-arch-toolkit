# C01 · Agent.stream(): streaming de texto, thinking e tools

- **Dono:** por atribuir · **Estado:** todo · **Depende de:** nada (F03 e F08 feitos)
- **Origem:** revisão L5, E, F e "Nomes" em B
  (`docs/internal/agentes-app-toolkit-review.md:283-307`, `:733-737`, `:784-831`); plano §4 item 10
  (`docs/internal/toolkit-fix-plan.md:648-652`)
- **Decisões:** por fixar (ver abaixo)

## Problema

- `FlowEvent` só tem eventos de execução (`toolkit/flow/_flow.py:96-116`); `Agent.iter()` só
  embrulha `Flow.iter()` (`toolkit/agents/_agent.py:120-132`), sem tokens (`docs/agents.md:236`).
- Em `toolkit/agents/flows/`: 22 chamadas a `complete()` (`_react.py:83`, `_plan_execute.py:75,134`,
  `_rewoo.py:81,149`, `_reflexion.py:122`, `_generate_review.py:94,122`, `_self_discovery.py:94,
  109,124`, `_llm_compiler.py:111,181`, `_tot.py:93,112,139,159`, `_lats.py:157,196,213,228`,
  `agents/_builders.py:267`); 7 `inner.run()` (`_plan_execute.py:103`, `_reflexion.py:86`,
  `_generate_review.py:86,118`, `_self_discovery.py:156`, `_llm_compiler.py:164`, `_lats.py:148`);
  3 execuções de tools (`_react.py:116,148`, `_rewoo.py:127`). Paralelo só em `_llm_compiler.py:175`
  e `_react.py:132`; o ToT avalia candidatos em série (`_tot.py:138-153`).
- `stream_events()` não substitui `complete()`: retry e fallback só antes do 1.º item
  (`core/_llm.py:1069-1076`, `:1227-1239`), fora do `inference_limit` (`:1038-1040`), sem `parsed`,
  `citations` nem `response_id` (`:918-941`). O Gemini guarda como `raw` só o último chunk
  (`_gemini.py:634`), que o histórico reenvia (`:186-197`); o Gemini 3 dá 400 a uma
  `functionCall` sem assinatura ([docs](https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures)).

Repro (`uv run python`, providers falsos ou SDK simulado, sem rede):

```
Agent.iter() react, 2 voltas  flow_start, step_start/end llm_call, …execute_tools…, flow_end;
                              provider.complete 2×, stream_events 0×; texto só em result.text
sub-agente numa tool          nenhum evento do filho (tool async, ou síncrona com run_sync)
MeterScope, sucesso           complete = stream_events: calls=1 in=100 out=20 cost=0.0006
OpenAI, output_schema         complete parsed=Answer(answer=3) | stream parsed=None response_id=''
503 transitório, retry=2      complete attempts ['failed','ok'] | stream: excepção após 1.º evento
Gemini, 2 calls em 2 chunks   reenviado: complete [get_weather+sig, get_time] | stream [get_time]
stream sem usage (gpt-4o)     llm_calls=1 cost=0.0 unknown_cost_count=0
```

## Objectivo

`Agent.stream()` e `Flow.stream()` entregam, junto dos eventos de execução, deltas de texto e
thinking, tool calls e resultados de tools das dez estratégias e dos runs aninhados, com `run_id`,
`parent_run_id`, `iteration`, `phase`, `label` e `call_id`, e acabam com o resultado de `run()`
(texto, `parsed`, usage, custo, chamadas medidas). Retry, fallback ou erro depois de output visível
é anunciado para descartar. `run()` e `iter()` não mudam.

## API proposta

```python
response = await llm.complete(msgs, tools=group, on_event=cb)  # core; StreamEvent + "reset"
with event_labels(phase="planner"):                            # toolkit.flow, para steps
    response = await llm_complete(llm, msgs, system=planner)   # sem stream = llm.complete()
async with agent.stream(task, config=run_config) as stream:    # AgentExecution (D5)
    async for ev in stream:
        match ev.type:
            case "text_delta" | "thinking_delta": ui.append(ev.run_id, ev.call_id, ev.text)
            case "tool_call" | "tool_result": ui.tool(ev.tool_call, ev.tool_result)
            case "llm_end" if ev.error: ui.discard(ev.call_id)
            case "retry" | "fallback" | "timeout": ui.discard(ev.call_id or ev.step_name)
result = stream.result                                          # igual a agent.run(task)
```

- **`FlowEvent`:** tipos `llm_start`, `text_delta`, `thinking_delta`, `tool_call`, `llm_end`,
  `tool_result` (só em `stream()`); campos com default `run_id`, `parent_run_id`, `iteration`,
  `phase` (de `FlowStrategy.phases`), `label` (passo do plano, `$N`, candidato do ToT), `call_id`,
  `text`, `tool_call`, `tool_result`, `response`. O `flow_start` aninhado traz o `step_name` do pai
  e, numa tool, o `tool_call`.
- **Entradas:** `Flow.stream()`/`stream_sync()`, `Agent.stream()`/`stream_sync()` como `run()`;
  `SyncAgentExecution`, `Agent.iter_sync()` (falta), `FlowExecution.run_id`; nada no
  `ReasoningSpec`. Erros como `iter()`; provider sem `stream_events`, ou `logprobs=True`, sem deltas.
- **Metering e governança:** carga só em `LLM.complete` e no executor de tools, uma operação por
  tentativa física, preço igual (`mode="stream"` é etiqueta); `tool_result` sai depois de
  `ToolGroup.async_execute` (aprovação incluída) ou de `run_tools()`.

## Decisões a fixar antes de codificar

1. **Onde nasce o delta:** (a) helper sobre `stream_events()`; (b) `LLM.complete(on_event=)`; (c)
   observador ambiente no core. Recomendo (b): herda o metering por tentativa (`_llm.py:666-719`),
   `inference_slot` (`:682`), middleware (`:849-914`) e retry/fallback com output visível, que (a)
   não tem; (c) publicaria chamadas internas, como a do `LLMModerator`
   (`toolkit/moderation/_llm.py:67`).
2. **Como um step publica:** (a) flag por estratégia; (b) `ContextVar` de canal posto em `_execute`
   junto de `_child_traces` (`_executor.py:537-549`), lido por `llm_complete` e `run_tools`.
   Recomendo (b): o `Agent` compila uma vez (`_agent.py:56`) e, sem canal, o helper é
   `llm.complete()` tal e qual (os mocks dos testes actuais não mudam).
3. **Forma dos eventos:** (a) estender `FlowEvent`; (b) `AgentEvent` em união. Recomendo (a): um só
   `match ev.type`, as execuções de D5 intactas, e `iter()` nunca vê os tipos novos.
4. **Descartar o mostrado:** recomendo deltas provisórios; `retry` com `call_id` (o `reset`) ou
   `llm_end.error` descartam a chamada; `retry`/`fallback`/`timeout` com `step_name`, o step desde
   `step_start`; `timeout` do run, o que estava em curso. Cobre LLM, policy do step
   (`_step_engine.py:172-178`) e moderação de saída sobre texto já entregue (`docs/moderation.md:62`).
5. **Runs aninhados:** (a) isolados; (b) reencaminhados ao pai, com `run_id` e `parent_run_id`.
   Recomendo (b): é a árvore que a app mostra e o `ContextVar` já lá chega (repro: `_child_traces`
   numa tool async e numa síncrona com `run_sync`, noutro loop, daí `call_soon_threadsafe`).
6. **Nomes:** (a) deltas em `iter()`; (b) `stream()` novo; (c) `iter_events` e deprecar. Recomendo
   (b): D5 fixou `iter()` e quem o usa conta com um só `flow_end`; alias só com paridade provada.
7. **Paridade da `Response`:** (a) documentar; (b) corrigir nos adaptadores primeiro. Recomendo (b):
   senão perdem-se `parsed`, `citations`, `response_id` e as assinaturas do Gemini 3, e um stream
   sem usage fica a custo 0 conhecido, no `Agent` e no loop directo da app.
8. **Backpressure:** recomendo buffer sem limite dentro do step (a task corre até ao fim, D9), nada
   entre steps sem o consumidor, fila de 256 no `stream_sync` (`core/_sync.py:50`), sem coalescer:
   limitar dentro do step congelaria a chamada ao provider com a ligação aberta.

## Sub-tarefas, por ordem

- **C01a** (core) `stream()`/`stream_events()` finalizam como `complete()`: `parsed`, `citations`,
  `response_id` e texto pelos parsers de cada adaptador; `raw` do Gemini com todas as partes;
  `StreamState.usage` ausente → custo desconhecido.
- **C01b** (core) `complete(on_event=)`/`complete_sync(on_event=)`: `stream_events` do provider
  dentro de `_try_with_tracking`; `reset` antes de retry ou fallback que se segue a output entregue.
- **C01c** (precisa de C01b) canal no motor, `FlowEvent`, `run_id`, `Flow.stream()`,
  reencaminhamento (mesmo loop e entre threads), `event_labels`, `llm_complete`, `run_tools`.
- **C01d** (precisa de C01a e C01c) os 32 sítios das estratégias pelos helpers, com fases;
  `Agent.stream()`, `stream_sync()`, `SyncAgentExecution`, `Agent.iter_sync()`; exemplo novo.
- **C01e** ao vivo local (`live_api`, fora do CI), Anthropic, OpenAI, Gemini e Meta: react com tools
  e `output_schema` em `stream()` igual a `run()`.

## Ficheiros

- Em `src/ai_arch_toolkit/`: `core/_llm.py`, `core/_providers/_base.py`, `_anthropic.py`,
  `_openai.py`, `_gemini.py`, `_xai.py`, `_meta.py` (partilhados com C05, C06); `core/_response.py`.
- `toolkit/flow/_flow.py`, `_executor.py` (partilhados com C04), `_stream.py` (novo), `__init__.py`;
  `toolkit/_runner.py`; `toolkit/agents/flows/*.py` (os nove) e `toolkit/agents/_builders.py`
  (partilhados com C05; `_lats.py` também com C04); `toolkit/agents/_agent.py` (partilhado com
  C04), `toolkit/agents/__init__.py`; exports `toolkit/__init__.py`, `__init__.py` (partilhados).
- Testes novos: `tests/test_stream_parity.py`, `tests/test_llm_on_event.py`,
  `tests/flow/test_stream.py`, `tests/agents/test_agent_stream.py`,
  `tests/integration/test_stream_live.py`; alterado `tests/test_llm_metering.py`.
- `docs/llm.md` (partilhado com C06), `docs/agents.md` e `docs/flow-architecture.md` (partilhados
  com C04), `docs/moderation.md`, `docs/concurrency.md`, `docs/api.md`; `examples/<n>_agent_stream.py` (novo; o
  número é o próximo livre, atribuído pelo coordenador ao aplicar).

## Prova

- **C01a** SDK simulado por adaptador: `parsed`, `response_id` e `citations` de `stream_events()`
  iguais aos de `complete()` (antes `None`/`''`); Gemini com calls em 2 chunks → `_messages_to_sdk`
  reenvia as duas partes e a assinatura (antes só a última); sem usage → `unknown_cost_count == 1`.
- **C01b** eventos por ordem e `Response` igual a `complete()` (antes o callback nunca corre); 503
  após o 1.º evento, `max_retries=2` → `[text, reset, text, text]`, `attempts ['failed','ok']`,
  `llm_calls == 2`; fallback → `reset`; `inference_limit(1)` com 4 chamadas → pico 1; cancelar a
  meio fecha o iterador do provider e falha a operação; `abefore`/`aafter` uma vez.
- **C01c** step com `llm_complete` → `llm_start`, `text_delta`…, `llm_end` dentro do step e `result`
  igual a `run()`; `iter()` sem tipos novos nem `stream_events`; `inner.run()` e sub-agente numa
  tool síncrona → `parent_run_id`; steps paralelos intercalam; `break` + `aclose()` a meio fecha
  stream e scope; retry do step depois de deltas → `retry`; `stream_sync` igual.
- **C01d** dez estratégias, LLM real com provider falso: `stream().result` igual a `run()` (texto,
  `parsed`, usage, custo, `llm_calls`, trace); fases certas; `$1`/`$2` do LLMCompiler em `run_id`
  distintos; `tool_result` pelo `tool_call.id`; sub-agente com o `tool_call.id` da delegação.

## Fora do âmbito

`agent_as_tool` (app); retomar da resposta parcial ("Error recovery" em
https://platform.claude.com/docs/en/build-with-claude/streaming); thinking token a token no
Anthropic; eventos próprios, JSON parcial, serialização; filtros por fase; deltas no `Trace`;
`reset` e `inference_limit` em `stream_events()` directo; opt-out do reencaminhamento; CLI.

## Riscos

- `FlowEvent` com 16 campos sem estreitamento por `type`; um evento por token (medir 10 000 deltas).
- O Anthropic dá thinking em blocos (`_anthropic.py:772-781`); Gemini e xAI dão thinking e tool
  calls depois do texto (`_providers/_base.py:124-148`). O contrato é o mesmo, a cadência não.
- Assinaturas do Gemini e usage em stream só se confirmam ao vivo (C01e).
- Ordem com C04, C05 e C06 nos partilhados; um run filho que sobreviva ao pai perde eventos.

## Registo do dono

- Estado: todo
- Ficheiros tocados:
- Testes novos:
- Verificações:
- CHANGELOG proposto:
- Desvios ao plano:
