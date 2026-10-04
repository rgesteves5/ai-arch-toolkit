# A05 · O streaming dentro das estratégias: o `StepTrace` no fim de cada passo e os tokens (G-22)

- **Dono:** Claude (2026-10-04) · **Estado:** done · **Depende de:** nada
- **Origem:** o briefing do ai-network, grupo 5 (G-22) · **Decisões:** D54 · **Regras:** `R00-rules.md`

## Problema (verificado a 2026-10-04 no `ff6e8ba`)

- **`StepTrace`:** o `step_end` traz o `Result` do passo, mas não o `StepTrace`, que só chega no
  `flow_end`.
- **Passos que ficam abertos:** um passo com `step_start` fica sem `step_end` quando o flow acaba
  antes dele:
  - num timeout do flow, os passos em curso são cancelados, e só chega o evento `timeout`;
  - num budget negado a meio do passo, há o `policy_decision` e a entrada sintética
    `budget_exceeded`, mas nada para o passo;
  - numa excepção que sai do motor, não chega mais nada, nem o `flow_end`.
- **Tokens:** as estratégias chamam `llm.complete` (18 sítios), e o `iter()` só dá eventos de
  passo (`docs/agents.md`: "`iter()` is not token streaming").

## Nota de desenho (antes do código)

- **`StepTrace`:** o `FlowEvent` ganha `step_trace`, no `step_end` e no `step_skipped`.
- **Todo o passo que começa acaba:**
  - o `_FlowRun` guarda o início dos passos com `step_start` (`_started`), e o `_ended` tira-os
    de lá;
  - um passo cortado ganha o seu `step_end`, com um `StepTrace` que tem o erro, o tempo gasto e
    a razão em `policy_decisions`. As razões são `timeout` (o timeout do flow), `budget_exceeded`
    (o budget negado nesse passo) e `halt` (uma excepção que sai do motor, para os passos ainda
    em curso). O trace final também os regista;
  - as entradas sintéticas `flow_timeout` e `budget_exceeded` continuam, e em último lugar;
  - numa excepção, os `step_end` dos passos cortados saem antes de a excepção chegar a quem
    consome.
- **Tokens (D54):**
  - `core/_llm_events.py` (novo): `llm_events_to(sink)` e `llm_event_sink()`, exportado em
    `ai_arch_toolkit.core`;
  - o `LLM.complete`, com o canal ligado, corre pelo `stream_events` e entrega cada evento com
    um id por chamada;
  - o `_FlowRun` de um `iter()` (`tokens=True`) liga o canal em cada passo. O sink faz o
    `FlowEvent(type="llm_event", llm_event=..., llm_call=...)` e, chamado de outra thread (uma
    tool síncrona), entrega-o pelo `call_soon_threadsafe`;
  - o `run()` não liga o canal.
- **Provas:**
  - o `step_end` traz o `StepTrace`, o mesmo objecto que fica no trace final, e o
    `step_skipped` o seu;
  - num timeout, num budget negado e numa excepção, todo o `step_start` tem o seu `step_end`, com
    a razão, antes do `timeout`, do `policy_decision` ou da excepção;
  - num `iter()`, um passo que chama `llm.complete` dá `llm_event`s com os pedaços do texto, entre
    o `step_start` e o `step_end`. O `Result` tem a resposta inteira;
  - num `run()`, nada vai por stream;
  - um agente `react` iterado dá os tokens do `llm_call`, e um ReAct aninhado (o `plan_execute`)
    dá os seus marcados com o passo de fora;
  - uma tool síncrona que chama o LLM dentro de um flow iterado entrega os eventos.

## Ficheiros

- `src/ai_arch_toolkit/core/_llm_events.py` (novo), `core/_llm.py`, `core/_attempts.py`, `core/__init__.py`
- `src/ai_arch_toolkit/toolkit/flow/_flow.py`, `_executor.py`
- os testes de flow, de agentes e de LLM que mudarem; os testes novos
- `docs/flow-architecture.md`, `docs/agents.md`, `docs/llm.md`, `docs/api.md`,
  `docs/framework-overview.md`, `AGENTS.md`, `CHANGELOG.md`

## Registo do dono

- Estado: done (Claude, 2026-10-04 e 05), pela nota de desenho e pela revisão independente. O dono
  escolheu o canal no LLM e os tokens sempre no `iter()` (D54).
- **`StepTrace` e passos cortados:**
  - o `FlowEvent` ganhou `step_trace` (no `step_end` e no `step_skipped`), `llm_event` e
    `llm_call`, e o tipo `llm_event`;
  - o `_FlowRun` ganhou `_started`, `_cut`, `_cut_all` e `_queued`. O `_ended` recebe o nome do
    passo e tira-o de `_started`;
  - os casos que terminavam o run passaram do `events()` para um gerador próprio, o `_stopping`
    (o `events()` passava o limite de complexidade);
  - um passo cortado pelo timeout ganha a decisão `timeout`; um passo cujo budget foi negado,
    `budget_exceeded` (com o erro "stopped by the budget: ..."); e os passos em curso quando o
    motor levanta, `halt`.
- **Tokens:**
  - `core/_llm_events.py` (`llm_events_to`, `LLMEventSink`, exportados);
  - o `LLM.complete` com o canal ligado vai pelo `stream_events` (`_streamed`);
  - o `_FlowRun` de um `iter_flow` (`tokens=True`) liga o canal em cada passo.
- **Revisão independente.** Quatro defeitos, corrigidos com testes que falhavam antes:
  1. uma chamada deixada a correr pelo passo (a thread de uma tool que passou o prazo, uma task
     que ninguém espera) continuava a mandar eventos depois do `step_end`. Num `iter_sync`, a
     chamada falhava a meio, já cobrada, quando o loop do run fechava. Agora o sink deixa de
     ouvir quando o passo acaba e engole o `RuntimeError` do loop fechado;
  2. num corte, os eventos que estavam na fila (um `step_start`, os `llm_event`) perdiam-se antes
     dos `step_end` dos passos cortados. Agora saem primeiro (`_queued`);
  3. o `inference_limit` deixava de limitar num run iterado, porque o stream só segura a vaga
     até ao primeiro evento. Agora uma chamada que vai para o canal segura-a até ao fim
     (`Execution(whole_slot=True)`): quem a lê não cede o controlo a ninguém;
  4. o exemplo do `docs/flow-architecture.md` estava partido, e o `docs/api.md` e o
     `docs/framework-overview.md` não tinham o `llm_event`.

  Também corrigi um bug antigo que ela apontou: um `aclose()` logo a seguir ao `flow_start`
  saltava a limpeza do run. E o CHANGELOG diz agora que os passos cortados entram no
  `AgentResult.errors`, e que um servidor compatível sem usage no stream deixa o custo
  desconhecido.
- **Testes que afirmavam o contrato antigo, corrigidos:**
  - `tests/flow/event_grammar.py`: o evento de paragem fechava os passos em curso. Agora cada
    passo fecha com o seu `step_end`, que leva a sua entrada do trace;
  - `tests/flow/test_engine.py::test_a_timeout_in_a_parallel_wave_keeps_the_siblings_that_finished`:
    o passo cortado não estava no trace.
- **Testes novos (20):**
  - `tests/flow/test_flow_streaming.py` (14): o canal (3), os tokens num flow (4), todo o passo
    acaba (2), o agente `react` (1) e o que a revisão encontrou (4);
  - `tests/flow/test_event_grammar.py`: dois cenários com stream, um sequencial e outro com dois
    passos em paralelo, e o caso "still running at the stop" no verificador.
- **Riscos que ficam** (da revisão, sem verificação ao vivo):
  - num run iterado, toda a chamada vai por stream. Um servidor compatível que não mande usage
    no stream deixa o custo desconhecido;
  - o replay de um turno do Gemini feito em stream ainda espera a verificação ao vivo do dono;
  - as chamadas de dentro das tools e dos revisores também aparecem, e só o `llm_call` as
    distingue.
- Gate: 6333 passed, 42 skipped; ruff, formatação e pyright limpos.
- **Linha para o Registo do briefing do ai-network** (o dono leva-a): "o grupo 5 fechou:
  - o `step_end` traz o `StepTrace` (`event.step_trace`), e todo o `step_start` tem o seu
    `step_end`: também os passos que o timeout, um budget negado ou uma excepção cortam, com a
    razão no trace, antes do evento de paragem ou da excepção;
  - num `agent.iter()` ou `flow.iter()`, cada chamada ao LLM de um passo chega em stream como
    `FlowEvent(type='llm_event')`, com o `StreamEvent` e o id da chamada; um sub-agente chega
    marcado com o passo de fora;
  - o `run()` não muda.

  Na app, saem o ciclo copiado do `react_flow` no `chat/turns.py` (o `iter()` do agente `react`
  já dá os tokens e a mesma semântica) e o fecho à mão dos passos que ficavam abertos."
