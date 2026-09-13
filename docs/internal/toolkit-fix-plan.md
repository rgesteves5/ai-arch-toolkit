# Plano de correcção do toolkit

**Data:** 12 de setembro de 2026. **Código:** `main` em `48a43ac`.
**Ponto de partida:** `agentes-app-toolkit-review.md` (três rondas, várias conclusões revertidas).
**Método:** cada achado foi reconstruído e executado contra o código actual antes de entrar aqui;
os números abaixo saem dessas execuções. Nada foi inferido só da leitura.

O documento tem quatro partes: (1) o que se confirma e o que não se confirma na revisão;
(2) o que escapou à revisão; (3) o plano, com ordem, dependências, ficheiros e prova;
(4) o que recomendo não corrigir.

**Execução:** o trabalho é coordenado em `blackboard/` (protocolo, quadro, decisões, fichas por
tarefa). As fichas de lá são a versão operativa deste plano.

---

## 0. Adenda de 13 de setembro — revisão cruzada

Um segundo modelo reproduziu N1–N7 e a correcção ao ponto I. Propôs cinco ajustes; todos se
confirmam contra o código e estão integrados nas secções abaixo.

| # | Ajuste | Onde entra |
|---|---|---|
| 1 | `_extract_usage` da Anthropic deixa passar `None` nos campos de cache: `complete()` rebenta em `estimate_cost`. | F1 |
| 2 | D1 não dizia o que acontece nas seis estratégias que correm um flow interno dentro de um step. | D1a, F2 |
| 3 | Um motor numa `asyncio.Task` com fila sem limite continua a correr depois de um `break` sem `aclose()`. Hoje o flow só avança quando alguém pede o evento seguinte. | D9, F8c |
| 4 | Fundir o `system` na camada `LLM` tiraria as mensagens `system()` da sua posição no OpenAI (e em Ollama/vLLM). | D3 revisto, F5 |
| 5 | Validar argumentos depois dos gates faz um humano aprovar chamadas que depois falham, ou valores que depois mudam. | D7 revisto, F13 |

Achados novos confirmados por execução nesta ronda:

- **N3, detalhe:** se o step negado não for o primeiro da vaga, o trace fica com o irmão e o
  estado não — os dois discordam.
- **N8 · `Policy(max_cost=)` por step não vê o gasto de flows aninhados.** O flow interno volta a
  ligar o meter ao span da raiz. Step que chama o LLM directamente → `Cost exceeded`; step que
  corre um flow aninhado com o mesmo gasto (1.05 USD) → `error=None`. Entra em F8 (8g).
- **N9 · campos de uso `null` na Anthropic** (o ajuste 1): `_parse_sdk_response` levanta
  `TypeError` com `cache_creation_input_tokens=None`. Entra em F1.
- **N10 · parâmetro positional-only torna a tool impossível de chamar:** `def f(x: int, /)` →
  schema `required=['x']`, execução `validation_error`. Entra em F7.
- **N/O estão documentados** no diagrama de decisão de `docs/flow-architecture.md:147-159`; a
  secção 4 fica reduzida a uma frase de clarificação.

---

## 1. Veredicto sobre a revisão

### 1.1 Confirmado tal como está

| Ponto | Medido |
|---|---|
| **A** middleware async ignorado em streaming | `complete()` → `abefore/aafter`; `stream()`/`stream_events()` → `before/after` (sync). `ModerationMiddleware(input=…)` que sinaliza sempre: `complete()` bloqueia, `stream_events()` entrega `'hello'`. |
| **B** `Flow.iter()` viola o isolamento do DAG | `run()`: irmão `b` viu `'<NADA>'`; `iter()`: viu `'A wrote this'`. 4 steps de 50 ms com `max_parallelism=2`: `run()` 0.10 s, `iter()` 0.20 s. |
| **C** metadados de `tools.dangerous` | `read_file`, `list_directory`, `search_files`, `http_get`, `scrape_text` usam `@tool` nu (`_filesystem.py:13,38,76`; `_web.py:56,79`). |
| **D** `ToolGroup(web_search())` | `AttributeError: 'ServerTool' object has no attribute '__name__'`. |
| **E** eventos `retry`/`fallback`/`timeout` nunca emitidos | Zero ocorrências no pacote. Step que expira e step que retenta produzem ambos `[flow_start, step_start, step_end, flow_end]`. |
| **F** `Agent.iter()` sem resultado | Confirmado por leitura (`_agent.py:123-129`): o `State` é local, só saem `FlowEvent`s. |
| **G** `Flow.policy` inerte em execução directa | `Policy(timeout=0.001)` sobre step de 50 ms: directo → `error=None`; aninhado → `'Step timed out'`; no `Step` → `'Step timed out'`. `Agent(ReasoningSpec(strategy=…, timeout=0.001))` com provider de 50 ms devolve resposta normal e `errors=()` para `react`, `completion` e `rewoo`. |
| **G-bis** `policy` + `timeout` na factory perde o timeout | `react_flow(policy=Policy(retry=2), timeout=30)` → `flow.policy.timeout=None`. Mesmo padrão nas nove factories e em `_build_completion`. |
| **H.1/H.2** modo inferido | Só `when=` → `is_dag=False`, `n=3`; `when=` + um `after=` noutro step → `is_dag=True`, `n=1`, e `max_iterations` deixa de ser exigido. |
| **H.3/J** excepções que escapam de `Flow.run()` | Escapam: `when` em modo DAG, `Scope.transform`, `Scope.enrich` (sequencial e na vaga paralela do DAG). Em modo cíclico o `when` que levanta vira skip silencioso com o flow a "terminar bem". |
| **K** `Scope` incoerente | `exclude={"api_key"}` + `enrich` que a lê: o step vê `api_key=None` e `leak='sk-SECRET'`. `transform` numa chave presente em 3 camadas corre 3×. |
| **L** schema mentiroso | `procura(consulta: int \| str)` → `required=[]`; modelo omite → `validation_error`. `flexivel(a, *args, **kwargs)` → `required=['a','args','kwargs']`; a função recebe `args=() kwargs={'args': 2, 'kwargs': 'x'}`. |
| **M** argumentos não validados | `soma("1","2")` → `ok=True`, valor `"12"`. |
| **N/O** semântica de retry | Timeout + `max_retries=3` → 1 tentativa. `Result(error=)` + `max_retries=2` → 3 chamadas, `('retry','retry','halt')`, 0.17 s de backoff. |
| **P** (parte `_run_sync`) | Timeout do wrapper levanta `TimeoutError`; a coroutine termina na mesma 0.3 s depois. |
| **Q** `MeterScope` no trace | `input_state.world` contém `_meter_scope`; `to_dict(trace_mode="full_debug")` → `TypeError: Object of type MeterScope is not JSON serializable`. |
| **R** batch Anthropic perde `system()` | `batch_submit([{"messages": [system("A"), user("x")]}])` → `params.system=None`. |
| **S** `system=` apaga a mensagem `system()` | Anthropic e OpenAI: com os dois presentes só `system=` é enviado. |
| `run_tools(resp, ToolGroup)` perde gates/`max_calls` | Grupo com `DangerousToolGate` e `max_calls=0`: `group.async_execute` → `dangerous_tool_blocked`; `run_tools` → `'EXECUTED'`. |
| `UsageEvent.provider` vazio | Evento retido: `model='claude-sonnet-4-6' provider=None`. |
| `ToolGate`/`ExecutionContext`/`GateBlock`/`GateModify`/`GateDryRun` não exportados | Ausentes de `core/_tools/__init__.py`, `core/__init__.py` e `ai_arch_toolkit/__init__.py`. |
| `StepTrace.children` nunca preenchido | Nenhum `children=` fora de `_trace.py`. |
| `limits.timeout_seconds` do manifesto | Mapeia para `ReasoningSpec.timeout` (`_manifest.py:196,204`) — logo inerte por G. O exemplo de `docs/agents.md` anuncia exactamente este campo. |

### 1.2 Confirmado com correcção

**I — "a memória do trace cresce com o quadrado"**: o efeito quadrático existe, mas na
**serialização**, não na memória. Medido num `react_flow` de 12 turnos com observações de 2 KB:

| | bytes |
|---|---|
| histórico final (JSON) | 26 789 |
| soma dos `input_state` do trace (JSON) | 376 801 (×14.1) |
| crescimento real de memória durante o run (`tracemalloc`) | 120 009 |

`StepTrace.input_state` é uma cópia rasa (`dict(layer)`): os dicionários das mensagens são
**partilhados por referência** entre traces (verificado: `m1[0] is m0[0]`). O que é quadrático é
cada lista de mensagens ter os seus ponteiros, e sobretudo `Trace.to_dict()`, que serializa cada
`input_state` por inteiro: com 20 turnos e `Response.raw` de ~20 KB, `to_dict()` demora 1.39 s e
produz **16 MB** de JSON. A revisão acerta no impacto para quem persiste execuções; erra no
mecanismo ("tudo vivo em memória"). A correcção é a mesma (política de captura), ver F16.

**P (parte `_stream_sync`)**: "a thread nunca é aguardada" é literalmente verdade (não há `join`
quando `drained=False`), mas a thread **termina sozinha** ao chegar o item seguinte: o `stop` é
verificado a cada item. Medido: abandonar após 1 item, produtor a 20 ms → thread desaparece ao
2.º item. O problema real é só a fila ilimitada e o `aclose()` do provider adiado até ao chunk
seguinte. Severidade menor do que a revisão sugere.

### 1.3 O que não se confirma

- **"`docs/concurrency.md` devia ao menos mencionar a exclusão do streaming"** — já menciona
  (`docs/concurrency.md:50-51`: "Streaming calls are **not** throttled") e `docs/llm.md` repete-o.
  Nada a corrigir.
- **"Um agente de 50 turnos com 200 KB de contexto produz na ordem de 10 MB de trace, tudo vivo em
  memória"** — falso em memória (ver 1.2); verdadeiro à saída de `to_dict()`.

### 1.4 Resultados negativos meus (testado, está bem)

- `inference_limit(1)` com dois `run_sync()` consecutivos sob contenção (DAG de 2 chamadas
  paralelas): sem `RuntimeError` de loop; o semáforo não fica preso ao primeiro loop.
- `ToolGroup(max_calls=…)` com chamadas paralelas governadas em dois `asyncio.run()` sucessivos:
  o `asyncio.Lock` lazy não fica preso ao primeiro loop.
- O middleware do primário corre quando o fallback responde (já registado na revisão); ver N6
  para a parte que não bate certo.

---

## 2. O que escapou à revisão

Por ordem de gravidade. Todos executados.

### N1 · O adaptador Anthropic duplica os tokens de input em streaming

`_anthropic.py:600-615` (`stream`) e `:724-739` (`stream_events`) tratam `message_delta.usage`
como um **incremento** e somam-no ao `usage` do `message_start`. A API define-o como
**cumulativo**: "The token counts shown in the `usage` field of the `message_delta` event are
*cumulative*" (docs oficiais de streaming), e o SDK instalado (0.116.0) tipa
`MessageDeltaUsage.input_tokens` como "The cumulative number of input tokens which were used".
Desde 2025 o `message_delta` carrega `input_tokens`, `cache_*_input_tokens`, `output_tokens`.

Medido com os tipos reais do SDK (`RawMessageStartEvent` com `input_tokens=25, output_tokens=1`;
`RawMessageDeltaEvent` com `MessageDeltaUsage(input_tokens=25, output_tokens=15)`):

```
mensagem final:            input=25 output=15
state.usage acumulado:     input=50 output=16     <- input ×2, output +1
delta com input_tokens=None (forma antiga): TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'
```

Consequências: `Response.usage` errado em todos os streams Anthropic; o meter cobra 2× o input
(e 2× os tokens de cache); `max_input_tokens`/`max_total_tokens`/`max_cost` disparam cedo;
qualquer relatório de custo de um loop de streaming da app está inflacionado. As estratégias não
sofrem (usam `complete()`), mas o loop de streaming que a app planeia sofre directamente.

**Porque passou:** `tests/test_stream_tool_calls.py:51-54` falsifica `message_delta` só com
`output_tokens`, e `getattr(usage, "input_tokens", 0)` devolve 0 num `SimpleNamespace` sem o
campo. Os fakes não têm a forma da API.

### N2 · `ToolGroup.execute()` (síncrono) numa ferramenta `async def` devolve uma coroutine

`_run_tool_sync` (`core/_tools/_executor.py:224`) chama `definition.fn(**args)` sem verificar se
é coroutine function. Medido:

```
group.execute(tc)  -> ok=True, value=<coroutine>
result.to_model_text() -> TypeError: Object of type coroutine is not JSON serializable
RuntimeWarning: coroutine 'atool' was never awaited
```

`docs/tools.md:86` promete que `execute()` e `async_execute()` "never raise on tool failure".
Afecta também `execute_tool()` e `run_tools_sync()` (mesmo caminho). A via async faz o inverso
correctamente (`asyncio.to_thread` para funções sync).

### N3 · Uma negação de budget numa vaga paralela do DAG descarta o trabalho dos irmãos

`_execute_dag` (`_executor.py:414-428`) faz `gather(return_exceptions=True)` — os irmãos
terminam — mas ao percorrer os resultados levanta no primeiro `AdmissionDenied` e **não processa
os seguintes**. DAG com `llm_a`, `free_b`, `free_c` em paralelo sob `max_llm_calls=0`:

```
trace: [('budget_exceeded', 'admission denied: llm_calls')]
state.free_done = None      <- free_b e free_c correram, foram medidos, e desapareceram
```

O meter contou-os; o trace e o estado não. Para uma app com ambições de checkpoint é trabalho
pago e perdido.

### N4 · `StepTrace.input_state` é reescrito pelas mutações in-place dos steps

Como é cópia rasa, um step que mute um valor através do snapshot reescreve o `input_state` dos
traces anteriores. `tot_flow` e `lats_flow` fazem exactamente isso (`frontier.pop()`,
`_MCTSNode` mutado in-place). Medido com uma `deque` consumida em 3 iterações:

```
input_state.q registado em cada execução: [[], [], []]     <- devia ser [1,2,3], [2,3], [3]
```

O trace não é uma história fiel. Mesma raiz que I; mesma correcção (F16).

### N5 · O fallback recebe as mensagens **antes** do middleware mas o `system` **depois**

`_llm.py:739-753`: `_try_fallbacks(messages, …, system=system)` passa as `messages` originais
(pré-`abefore`) e o `system` já transformado. Middleware que injecte uma mensagem e altere o
system: o provider de fallback recebeu `messages=['hi']`, `system='INJECTED_SYSTEM'`. Incoerente;
menor.

### N6 · Compor `MemoryMiddleware` com uma mensagem `system()` apaga a mensagem

Variante de S com consequência prática: `MemoryMiddleware.abefore` escreve em `request.system`
(que é `None` quando o system vem nas mensagens); o adaptador aplica `system if system is not
None else msg_system` e a mensagem `system()` desaparece. A regra de precedência tem de viver
**antes** do middleware, no `LLM`, não nos adaptadores (F5).

### N7 · `Trace.to_dict()` é quadrático e deep-copia `Response.raw`

Complemento de I: `Redactor._redact_value` converte dataclasses com `asdict()`, que
**deep-copia** valores não-dataclass — incluindo o objecto SDK em `Response.raw`, presente em
cada `Response` do histórico, presente em cada `input_state`. 20 turnos com `raw` de 20 KB:
1.39 s, 16 MB. É o custo concreto que uma app que grava execuções vai pagar.

---

## 3. Plano de correcção

### 3.1 Regras para quem executar

1. **Uma correcção, um teste comportamental novo** que falhe antes e passe depois. O único
   teste que tocava em `Flow(policy=…)` verificava o getter (`tests/flow/test_flow.py:67-74`);
   é o padrão a evitar.
2. Os fakes de providers têm de ter a **forma da API real** (N1 nasceu de um fake sem
   `input_tokens`). Para Anthropic usar os tipos do SDK nos testes de usage.
3. Quebrar API pública é aceitável (`0.1.0.dev0`); cada quebra entra no `[Unreleased]` do
   `CHANGELOG.md` com a migração numa linha.
4. `uv run pytest -q`, `uv run ruff check --fix src tests examples`, `uv run ruff format`,
   `uv run pyright src` por PR. `tests/test_examples_smoke.py` cobre os exemplos que usam
   `Flow.iter`/`Agent.iter` (`examples/10`, `12`, `32`, `33`) — F8 muda o tipo devolvido e tem de
   os manter verdes.
5. Cada ficha abaixo lista os ficheiros; linhas referem-se a `48a43ac`.

### 3.2 Decisões a fixar antes de codificar

Estas são as escolhas que a revisão deixou em aberto. O plano assume as recomendadas; se se
escolher outra, as fichas mudam nos pontos indicados.

| Decisão | Recomendação | Alternativa e custo |
|---|---|---|
| **D1** Semântica de `Flow(policy=)` vs `timeout` (G) | Dois eixos: `Flow(policy=)` é a **policy por omissão de cada step** (`step.policy or flow.policy`, também em `_should_halt`); novo `Flow(timeout=)` é o **envelope wall-clock do run inteiro**. `ReasoningSpec.policy` → o primeiro; `ReasoningSpec.timeout` → o segundo. É o que `docs/agents.md:53-54` já diz. | Envelope para os dois: obriga a "repetir o flow inteiro" no retry, com efeitos externos duplicados. Não recomendo. |
| **D2** `as_step()` e a policy | O wrapper `Step` deixa de levar a policy do flow (`_flow.py:318`); o flow aninhado aplica a sua própria policy/timeout quando corre. Evita aplicar duas vezes. | Manter no wrapper: dupla aplicação depois de D1. |
| **D1a** Estratégias com flow interno num step (reflexion, plan_execute, llm_compiler, lats, self_discovery, generate_review com tools) | A policy **não** desce aos flows internos. Um timeout por step limita o loop interno inteiro desse step. Limites por chamada LLM pertencem a `LLM(timeout=, retry=)`; o limite do run a `ReasoningSpec.timeout`. Documentar em `docs/agents.md` e provar em reflexion. | Propagar a policy aos flows internos: aplica o mesmo timeout com dois significados e duplica o retry que o `LLM` já possui. |
| **D3** `system=` com mensagens `system()` (S, R, N6) — **revisto** | **Fundir nos adaptadores**, nunca escolher um: `system=` primeiro, depois as mensagens `system()` pela ordem. Anthropic, Gemini e xAI juntam tudo no parâmetro nativo; o OpenAI põe `system=` à cabeça e **mantém** as mensagens `system()` na sua posição. Inclui os caminhos de batch. | Fundir na camada `LLM`: move para o topo as mensagens `system()` a meio da conversa no OpenAI, Ollama e vLLM. |
| **D4** Metadados de `tools.dangerous` (C) | `capability` (`filesystem`/`web`), `risk_level="high"`, **`requires_approval=True`** nas cinco. Quem as usa sem handler passa a ver `approval_denied` — é o comportamento que `docs/safety.md` descreve para um módulo chamado `dangerous`. | Só `capability`+`risk_level`, sem aprovação: sem quebra, mas o hardening fica a meio. |
| **D5** Forma de `Flow.iter()`/`Agent.iter()` depois de F8 | Manter os nomes; devolver um objecto de execução que **é** `AsyncIterator[FlowEvent]` (o `async for` existente continua a funcionar) e expõe `.result` (`FlowResult`/`AgentResult`) no fim. | Renomear para `iter_events`/`start` como a revisão sugere: mais churn sem ganho de correcção. |
| **D6** Captura do trace (F16) | `Flow(trace_capture="keys" \| "full" \| "none")`, **default `"keys"`**: `input_state` vazio, nova `StepTrace.input_keys`; `"full"` faz deep copy (resolve N4 à custa de memória, explicitamente). | Default `"full"`: sem quebra, mas o custo quadrático continua por omissão. |
| **D7** Coerção de argumentos (F13) — **revisto** | Coagir strings numéricas/booleanas para `integer`/`number`/`boolean` quando o schema o diz; rejeitar com `validation_error` o que não coage; `enum` verificado; objectos/arrays intactos. **Antes dos gates**, para o `approval_handler` ver os valores que vão correr; validar outra vez depois de um `GateModify`. | Só validar (sem coagir): mais estrito, pior para modelos locais — que são o caso da app. |
| **D8** Operação de metering de um stream | Reservada na criação (a admissão continua eager); **iniciada** só quando começa a primeira tentativa no provider. Um stream rejeitado pelo middleware, ou nunca iterado, liberta a contagem em vez de ficar `incomplete` com custo desconhecido. | Manter o início na criação: uma rejeição da moderação passa a contar como chamada com custo desconhecido e bloqueia um run com `max_cost`. |
| **D9** Motor único de F8 | Um único gerador assíncrono. Cada step corre numa task; os eventos dessas tasks saem em tempo real; entre steps o flow só avança quando o consumidor pede. O `finally` cancela as tasks pendentes (inclui o `aclose()` que o loop agenda quando o gerador é abandonado). `run()` drena o gerador. O timeout do run usa esperas limitadas, nunca um cancel scope atravessado por `yield`. | Motor numa task com fila sem limite: depois de um `break` o flow continua a gastar budget em background. |

### 3.3 Ordem, dependências e paralelismo

```
Vaga 1 (independentes; um PR cada; podem correr em paralelo)
  F1  usage Anthropic em streaming          F6  ServerTool em ToolGroup → TypeError
  F2  Flow.policy + Flow(timeout)           F7  execute() sync em tool async
  F4  run_tools delega no ToolGroup         F10 provider no OperationRequest
  F5  fusão de system (LLM)                 F11 metadados de tools.dangerous
  F9  RunConfig no Agent                    F12 schema: uniões e varargs
  F14 Scope.enrich sobre o snapshot filtrado
  F17 exportar tipos de gates

Vaga 2
  F3  middleware async em streaming (+ N5)          — independente, mas maior; começar cedo

Vaga 3 (um PR, commits separados; depende de F2)
  F8  executor único: 8a erros de orquestração → 8b meter fora do State → 8c motor único
      (B, F) → 8d eventos de policy (E) → 8e children → 8f irmãos preservados (N3)

Vaga 4
  F16 política de captura do trace (depende de F8)
  F13 validação/coerção de argumentos (depende de F12)
  F15 wrappers sync: cancelamento e backpressure (independente; prioridade baixa)
  F18 deriva de docs (acompanha cada PR; fecha no fim)
```

Ponto de sequência: F3 e F8 antes de qualquer `Agent.stream()` — a revisão tem razão nisto.

### 3.4 Fichas

Formato: **Problema → Decisão → Ficheiros → Passos → Prova → Risco/quebra → Depende de**.

---

#### F1 · Usage cumulativo em streams Anthropic (N1) — pequeno, prioridade máxima

- **Problema:** `message_delta.usage` somado em vez de substituído; `None` rebenta.
- **Decisão:** `message_delta.usage` é cumulativo e **substitui**; campos `None` mantêm o valor
  anterior (do `message_start`). `_extract_usage` (`_anthropic.py:265`) converte `None` em 0 —
  é o único dos quatro adaptadores sem `or 0`, e sem isso até o `complete()` rebenta (N9).
- **Ficheiros:** `src/ai_arch_toolkit/core/_providers/_anthropic.py` (`stream` 600-615,
  `stream_events` 724-739 — extrair `_merge_delta_usage(prev, sdk_usage) -> Usage`);
  `tests/test_stream_tool_calls.py:51-54` (fake com `input_tokens`); `tests/test_anthropic_provider.py`.
- **Passos:** helper único; usar `getattr(u, campo, None)` e `prev.campo if valor is None else valor`.
- **Prova:** teste com tipos reais do SDK (`RawMessageStartEvent`, `RawMessageDeltaEvent`,
  `MessageDeltaUsage`) para `stream` e `stream_events`: `state.usage == Usage(input_tokens=25,
  output_tokens=15, cache_write_tokens=0, cache_read_tokens=0)`; delta só com `output_tokens`
  → sem `TypeError`, `input_tokens=25`; `LLM.stream_events` sob `MeterScope` →
  `snapshot.input_tokens == 25`.
- **Risco:** nenhum. `CHANGELOG` em *Fixed*.
- **Depende de:** nada.

#### F2 · `Flow.policy` por step e `Flow(timeout=)` envelope (G, G-bis, N7-manifesto) — médio

- **Problema:** policy do flow nunca chega ao step engine; `timeout` engolido quando há `policy`.
- **Decisão:** D1 + D1a + D2.
- **Prova adicional (D1a):** `Agent(ReasoningSpec(strategy="reflexion", policy=Policy(timeout=…)))`
  com LLM lento → o step `attempt` expira; `ReasoningSpec(strategy="plan_execute", timeout=…)` →
  o envelope corta o run inteiro, incluindo o flow interno em curso.
- **Ficheiros:** `core/_step_engine.py:21-27` (`execute_step(step, snapshot, *, policy=None)`);
  `toolkit/flow/_flow.py` (`__init__` 129-146: novo `timeout`; `as_step` 318: `policy=None`;
  propriedade `timeout`); `toolkit/flow/_executor.py` (`_run_step_with_scope` 552-561 recebe a
  policy efectiva; `_should_halt` 592-597 usa `fs.step.policy or flow.policy`; `execute_flow`
  40-51 e `iter_flow` 78-95 envolvem o corpo em `asyncio.timeout(flow.timeout)` e, ao expirar,
  acrescentam um pseudo-step `flow_timeout` com `Result(error="Flow timed out after Xs")`,
  `policy_decisions=("timeout",)` e emitem `FlowEvent(type="timeout")`); factories: `_react.py:180-191`,
  `_reflexion.py:140-152`, `_plan_execute.py:142-152`, `_llm_compiler.py:198-207`,
  `_generate_review.py:140-151`, `_rewoo.py:157-168`, `_tot.py:197-207`, `_lats.py:249-259`,
  `_self_discovery.py:162-174`, `_builders.py:277-280` — passar `policy=policy, timeout=timeout`
  sem a condição `if timeout is not None and flow_policy is None`; `docs/agents.md:53-54`;
  `docs/flow-architecture.md` (secção Policy: default por flow + envelope); `tests/flow/test_flow.py:67-74`.
- **Passos:** (1) kwarg em `execute_step`; (2) resolução no executor; (3) envelope; (4) factories;
  (5) `as_step`; (6) docs.
- **Prova:** `tests/flow/test_executor.py::TestFlowPolicy`: (a) `Flow(policy=Policy(timeout=0.001))`
  corrido directo → step com `'Step timed out'`; (b) `Step.policy` prevalece sobre `Flow.policy`;
  (c) `Flow(timeout=0.05)` com dois steps de 100 ms → `FlowResult` com erro de timeout, 2.º step
  nunca correu, `scope.has_live_ops(run)` falso, `iter()` emitiu `timeout`; (d) flow aninhado
  com `retry=1` num step contador → `attempts == 2`, não 4; (e) `on_exhausted="continue"` no
  flow deixa o flow prosseguir. `tests/agents/flows/test_react_flow.py`: `react_flow(policy=P,
  timeout=30)` → `flow.policy is P and flow.timeout == 30`. `tests/agents/`: `Agent(ReasoningSpec(
  timeout=0.001))` com provider de 50 ms → `result.errors` contém timeout e `result.text == ""`;
  manifesto com `limits.timeout_seconds: 0.001` → idem.
- **Risco/quebra:** quem tinha `Flow(policy=Policy(timeout=…))` passa a ter timeouts por step
  reais. `Policy(retry=…)` em `ReasoningSpec` passa a retentar `llm_call` — desejável, mas
  documentar que se soma ao retry do próprio `LLM`. O envelope cancela o step em curso: ops LLM
  são fechadas pelo `finally` de `_tracked` (metering correcto); ferramentas em thread continuam
  (ver F15).
- **Depende de:** nada. F8 assenta nisto.

#### F3 · Middleware async em streaming (A, + N5) — médio-grande

- **Problema:** `_prepare_stream_request` (`_llm.py:821`) corre `_run_before` e o finalizer
  (`:964-966`, `:1095-1096`) corre `_run_after`; moderação e memória não correm.
- **Decisão:** mover os hooks para **dentro do gerador**: `abefore` no arranque de `_items()`
  antes de abrir o stream do provider; `aafter` depois de o provider se esgotar, sobre a
  `Response` já construída e liquidada no meter; guardar a resposta final no `_StreamRun`; o
  finalizer devolve-a se existir e só constrói ele próprio no caminho de abandono (aí `aafter`
  não corre — documentar). Admissão continua eager (`_open_stream_op`, `:872`) com o pedido
  pré-middleware. Os wrappers sync drenam o gerador no loop da thread de trabalho, logo os hooks
  correm num loop. `_stream_with_fallbacks._items` faz o mesmo para o `aafter` do primário.
  N5: `_try_fallbacks` (`:739-753`) recebe `normalized` (pós-middleware) e não `messages`.
- **Ficheiros:** `core/_llm.py` (786-835, 837-977, 979-1115, 644-665); `docs/middleware.md:3,115`;
  `docs/moderation.md:62`; `tests/test_middleware.py`, `tests/test_stream_fallback.py`,
  `tests/test_rate_limit.py` (verificar se algum teste fixa o bypass em streaming; se sim, inverter).
- **Prova:** espião com os quatro hooks: `stream()`, `stream_events()`, `stream_sync()`,
  `stream_events_sync()` registam `['abefore', 'aafter']` exactamente uma vez, `aafter` recebe a
  `Response` com `usage` preenchido; `ModerationMiddleware(input=sinaliza-sempre)` →
  `ModerationError` no primeiro `__anext__` e o provider **nunca** foi chamado; `MemoryMiddleware`
  regista uma vez por stream; com fallback a responder, `aafter` do primário corre uma vez;
  retry antes do primeiro item não repete `abefore`; stream abandonado → `aafter` não corre.
- **Risco/quebra:** `RateLimitMiddleware` passa a aplicar-se a streams (correcto; actualizar a
  nota em `docs/middleware.md:115`). Código de ciclo de vida delicado — manter os testes de
  `test_stream_fallback.py` e `test_stream_response.py` verdes é a rede.
- **Depende de:** nada.

#### F4 · `run_tools`/`run_tools_sync` delegam no `ToolGroup` — trivial

- **Ficheiros:** `toolkit/_runner.py:25-29,55-60,85-90`; `tests/test_runner.py:137`.
- **Passos:** se `tools` é `ToolGroup`, chamar `group.async_execute(tc)`/`group.execute(tc)`;
  manter a pré-verificação `KeyError` para tool desconhecida (contrato existente).
- **Prova:** `DangerousToolGate` no grupo bloqueia via `run_tools`; `max_calls=1` bloqueia a 2.ª;
  lista de callables continua a funcionar.
- **Depende de:** nada.

#### F5 · Fundir os prompts de sistema nos adaptadores (S, R, N6) — pequeno

- **Decisão:** D3 revisto.
- **Ficheiros:** `core/_providers/_base.py` (helper `merge_system_prompts`);
  `_anthropic.py:497-498`, `:537-538`, `:648-649`, `:368-369`, `:775-782` (batch);
  `_openai.py:134-160` (com `system=`, manter as mensagens `system()` no lugar);
  `_gemini.py` (`complete`, `stream`, `count_tokens`); `_xai.py:446-447`, `:484-485`;
  testes que hoje fixam a precedência: `tests/test_openai_provider.py:102,415`,
  `tests/test_gemini_provider.py:415`, `tests/test_anthropic_provider.py:435`.
- **Prova:** por adaptador, `system="B"` com `system("A")` nas mensagens → o provider recebe os
  dois (`"B\n\nA"` nos que juntam; no OpenAI `"B"` à cabeça e `"A"` na posição original);
  `batch_submit` Anthropic só com `system()` → `params["system"] == "A"`; mensagem `system()` a
  meio da conversa no OpenAI mantém o índice com e sem `system=`; `LLM` com middleware que escreve
  `request.system` → o provider recebe memória e mensagem.
- **Risco/quebra:** quem usava `system=` para **substituir** a mensagem passa a ter os dois.
  `CHANGELOG` em *Changed*.
- **Depende de:** nada.

#### F6 · `ToolGroup.add(ServerTool)` → `TypeError` explícito (D) — trivial

- **Ficheiros:** `core/_tools/_group.py:63-72`; `tests/test_tools_group.py`.
- **Prova:** `ToolGroup(web_search())` → `TypeError` com a mensagem a apontar para
  `llm.complete(tools=[group, web_search()])`.
- **Depende de:** nada. (O canal `server_tools=` nas estratégias fica fora — ver 4.)

#### F7 · `execute()` síncrono em ferramenta async (N2) — pequeno

- **Ficheiros:** `core/_tools/_executor.py:193-237` (`_run_tool_sync`); `docs/tools.md:86`;
  `tests/test_tools_group.py`, `tests/test_runner.py`.
- **Passos:** se `inspect.iscoroutinefunction(definition.fn)`, executar com `_run_sync(...)`
  (fresh loop, ou thread se já houver loop — o contexto do meter é copiado). No caminho async,
  aguardar também um awaitable devolvido por uma função síncrona. Parâmetros positional-only
  (N10) passam por posição, pela ordem da assinatura.
- **Prova:** `execute()` numa tool `async def` → `ok=True`, valor string, sem `RuntimeWarning`;
  `run_tools_sync` idem; dentro de um loop a correr (caminho da thread) também;
  `def square(x: int, /)` → `ok=True`, `9`, nos dois caminhos.
- **Depende de:** nada.

#### F8 · Um executor, com eventos, e `run()` a drená-lo (B, E, F, J, H.3, N3, children, Q) — grande

Um PR, seis commits, nesta ordem. Cada commit é testável sozinho.

**8a · Erros de orquestração uniformes (J, H.3).** `when`, `Scope.transform` e `Scope.enrich`
que levantem, em qualquer modo, produzem `StepTrace(name, error="condition error: …" |
"scope error: …", policy_decisions=("halt",))`, `FlowEvent(type="step_end", error=…)` e param
o flow. O skip silencioso em modo cíclico (`_executor.py:143-155`) desaparece.
*Ficheiros:* `_executor.py` (143-165, 239-258, 339-343, 480-484, 543-549, 564-567).
*Prova:* parametrizado `{when, transform, enrich} × {sequencial, cíclico, DAG single, DAG
paralelo}` → `Flow.run()` devolve `FlowResult` (nunca levanta), `steps[-1].error` preenchido,
steps seguintes não correram.

**8b · `MeterScope` fora do `State` (Q).** O scope fica local em `execute_flow`/`iter_flow` e é
passado explicitamente (`_run_step_with_scope`, `_halt_if_over_budget`, `_trace_metadata`);
`FlowResult` ganha `meter_scope: MeterScope | None` (`field(repr=False)`) de que `meter`/`usage`
lêem; `State.set("_meter_scope", …, layer="world")` (`:618,626`) desaparece; o comentário em
`as_step` (`_flow.py:294-297`) fica obsoleto. Flows aninhados continuam a herdar por
`current_meter()`.
*Ficheiros:* `_executor.py:600-637`, `_flow.py:23-28,57-85,283-318`, `tests/flow/test_flow_metering.py`.
*Prova:* `"_meter_scope" not in state.world` após um run; `trace.to_dict(trace_mode="full_debug")`
serializa com `json.dumps` num flow sem `Response`; budget partilhado em flows aninhados
continua (testes existentes).

**8c · Motor único (B, F), segundo D9.** Um gerador assíncrono `_engine(...)` é o único motor.
Cada step corre numa task (um step sequencial é uma task; uma vaga do DAG são N tasks sob o
semáforo de `max_parallelism`). Os eventos dessas tasks vão para um deque com um sinal e o gerador
entrega-os em tempo real. Entre steps nada avança sem o consumidor pedir. O `finally` cancela e
aguarda as tasks pendentes e fecha o scope. `run()` = `async for _ in _engine(...)`. O timeout do
run é uma espera limitada (`asyncio.wait(timeout=…)`), nunca um cancel scope atravessado por
`yield`. `_iter_dag` e `_iter_sequential` são apagados. `Flow.iter()` devolve `FlowExecution` (D5: `__aiter__/__anext__`, `.result`,
`.trace`); `Agent.iter()` devolve `AgentExecution` com `.result: AgentResult | None`.
*Ficheiros:* `_executor.py` (todo), `_flow.py:260-281`, `_agent.py:123-129`, `toolkit/flow/__init__.py`,
`toolkit/agents/__init__.py`, `examples/10,12,32,33`, `docs/flow-architecture.md` (Streaming),
`docs/agents.md` (Streaming).
*Prova:* o repro de B como teste (`iter()` e `run()` dão o mesmo `b_saw`); 4 sleepers de 50 ms com
`max_parallelism=2` em `iter()` demoram < 3× 50 ms; `execution.result` é `FlowResult` após
drenar e `None` durante; `Agent.iter(...).result.text` igual a `Agent.run(...).text`; **`break`
sem `aclose()`** depois do primeiro `step_end` → passados 0.3 s o segundo step não correu e o
scope fechou (o teste existente usa `aclose()` explícito e não apanharia esta regressão); eventos
`retry` chegam ao consumidor antes de o step acabar; `TestFlowStreaming` e `test_examples_smoke`
verdes.

**8d · Eventos emitidos por quem decide (E).** `execute_step(..., on_decision=None)`; o motor
mapeia `retry`→`FlowEvent("retry")`, `fallback`→`"fallback"`, `timeout`→`"timeout"`, restantes
→ `"policy_decision"` com `policy_decision=<nome>`, todos com `step_name`, antes do `step_end`.
*Ficheiros:* `core/_step_engine.py:60-161`, `_executor.py`.
*Prova:* step com `retry=1` que falha uma vez → `[step_start, retry, step_end]`; timeout →
`timeout`; fallback → `[timeout, fallback]`; os três tipos do `Literal` passam a ter pelo menos
um teste que os observa.

**8e · `StepTrace.children` (traces de filhos).** `ContextVar[list[Trace] | None]` em
`_executor.py`: o motor abre uma lista à volta de cada step; `execute_flow` de um flow aninhado
(via `as_step()` **ou** `inner.run(state)` dentro de um step, como fazem seis estratégias) anexa
o seu `Trace` à lista corrente; o step envolvente recebe `children=(StepTrace(name=trace.flow_name,
duration=…, children=trace.steps), …)`. `Trace.flow(name)` passa a encontrar flows aninhados.
*Ficheiros:* `_executor.py`, `docs/flow-architecture.md:219,396`.
*Prova:* `Flow(inner, …)` → `trace.flow("inner")` não é `None` e tem os steps do filho;
`Agent(ReasoningSpec("reflexion"))` → `trace.flow("react")` encontrado;
`Trace.to_dict()`/`from_dict()` mantêm os filhos (existente).

**8f · Irmãos preservados numa negação (N3).** Processar todos os itens do `gather` (traces,
merge dos artefactos, contadores) e só depois levantar o `AdmissionDenied`.
*Ficheiros:* `_executor.py:414-446` (ou o equivalente no motor único).
*Prova:* o repro de N3 → `free_b`/`free_c` no trace, `state["free_done"] is True`, e o
`budget_exceeded` no fim; com o step negado **no meio** da vaga, trace e estado têm os mesmos
irmãos.

**8g · Atribuição de gasto a flows aninhados (N8).** Um flow que herda o scope liga-o mantendo o
span corrente do step pai, em vez de o repor na raiz.
*Prova:* o repro de N8 → o step que corre o flow aninhado falha com `Cost exceeded`, como o step
que chama o LLM directamente.

- **Depende de:** F2 (resolução de policy e envelope já no lugar).
- **Risco:** o maior do plano; a rede são os 260 testes de `tests/flow` + `tests/agents` mais os
  novos. Fazer 8a e 8b primeiro reduz o tamanho de 8c.

#### F9 · `RunConfig` por execução no `Agent` — trivial

- **Ficheiros:** `toolkit/agents/_agent.py:88-129` (`run`, `run_sync`, `iter` ganham `config=`,
  passado ao `Flow`); `docs/agents.md` (Budgets).
- **Prova:** `Agent.run(task, config=RunConfig(retain_meter_events=True))` →
  `result.report` e eventos retidos acessíveis; `config` prevalece sobre `budget_policy` como
  em `Flow.run` (docstring `_flow.py:238-245`).
- **Depende de:** nada (F8c actualiza `iter`).

#### F10 · `provider` no `OperationRequest` — trivial

- **Ficheiros:** `core/_llm.py:368-411` (resolver e guardar o nome: `provider` explícito, senão
  `_match_provider(model)`, senão `"openai"` com `base_url`), `:583-618` (`provider=self._provider_name`),
  `:620-642`.
- **Prova:** evento retido tem `provider="anthropic"`; num fallback cross-provider o evento do
  fallback tem o provider dele.
- **Depende de:** nada.

#### F11 · Metadados de `tools.dangerous` (C) — trivial, com quebra

- **Decisão:** D4.
- **Ficheiros:** `toolkit/tools/_filesystem.py:13,38,76`, `toolkit/tools/_web.py:56,79`,
  `docs/safety.md:86-101`, `docs/tools-catalog.md` (se listar risco), `CHANGELOG`.
- **Prova:** `read_file.__tool_definition__.policy == ToolRuntimePolicy(capability="filesystem",
  risk_level="high", requires_approval=True, …)` para as cinco; `ToolGroup(read_file).execute(call)`
  sem handler → `approval_denied`; com handler que aprova → executa.
- **Risco/quebra:** utilizadores de `dangerous` sem `approval_handler`.
- **Depende de:** nada.

#### F12 · Schema: uniões multi-tipo e varargs (L) — pequeno

- **Ficheiros:** `core/_tools/_schema.py:35-46` (união → `{"anyOf": [...]}`, `is_optional`
  só quando `None` está na união; `anyOf` e não `type: [...]` porque o Gemini não aceita listas
  de tipos), `:284-308` (ignorar `VAR_POSITIONAL`/`VAR_KEYWORD`: nunca em `properties` nem em
  `required`); `tests/test_tools_schema.py`.
- **Prova:** `procura(consulta: int | str)` → `required == ["consulta"]`, `properties.consulta ==
  {"anyOf": [{"type": "integer"}, {"type": "string"}]}`; `Optional[int | str]` → opcional com
  `anyOf`; `flexivel(a, *args, **kwargs)` → `required == ["a"]`, sem `args`/`kwargs` em
  `properties`; `prepare_tools` continua a produzir dicts aceites pelos quatro adaptadores
  (testes de adaptador com um schema `anyOf`).
- **Depende de:** nada.

#### F13 · Validação e coerção de argumentos (M) — médio; o item mais próximo de funcionalidade

- **Decisão:** D7 revisto.
- **Ficheiros:** `core/_tools/_executor.py` (novo `_bind_args(schema, args) -> dict | ToolResult`
  **antes** dos gates — o `ExecutionContext` recebe o `ToolCall` com os valores convertidos — e
  outra vez depois de um `GateModify`, em `_run_tool_sync` e `_arun_tool`); `docs/safety.md`
  (pipeline: "resolve → **validate/coerce** → gates → call-count → execute"); `tests/test_tools_executor.py`.
- **Prova adicional:** o `approval_handler` recebe `{"a": 1}` quando o modelo mandou `{"a": "1"}`;
  uma chamada inválida nunca chega ao handler; `modified_args` inválidos do handler →
  `validation_error`.
- **Prova:** a tabela de M: `{"a":"1","b":"2"}` → `3`; `{"a":"x"}` → `validation_error` com
  mensagem a nomear o parâmetro; chave desconhecida → `validation_error` listando as permitidas;
  `enum` fora do conjunto → `validation_error`; `GateModify` que reescreve args é validado depois
  da reescrita.
- **Depende de:** F12 (o schema tem de dizer a verdade primeiro).

#### F14 · `Scope.enrich` sobre o snapshot filtrado (K) — trivial

- **Ficheiros:** `toolkit/flow/_scope.py:29-59` (construir o `StateSnapshot` filtrado primeiro e
  passá-lo ao `enrich`); `docs/flow-architecture.md` (Scope: `transform` corre por camada — documentar,
  não mudar).
- **Prova:** `exclude={"api_key"}` + `enrich` que lê `api_key` → o step vê `leak is None`.
- **Depende de:** nada.

#### F15 · Wrappers síncronos: cancelamento e backpressure (P) — pequeno-médio, prioridade baixa

- **Ficheiros:** `core/_sync.py:55-88` (criar o loop e a task na thread, guardar referências;
  no timeout `loop.call_soon_threadsafe(task.cancel)` e `join` com tolerância), `:91-147`
  (`Queue(maxsize=…)`; ao abandonar, cancelar a task de drenagem via `call_soon_threadsafe` e
  `join`); `tests/test_sync.py`.
- **Prova:** timeout → a coroutine é cancelada (flag nunca fica `True` passado o dobro do tempo);
  abandonar `_stream_sync` → thread termina dentro de `_stream_join_timeout` sem esperar pelo
  item seguinte; consumidor lento → produtor bloqueia na fila cheia.
- **Depende de:** nada. Se se saltar, o impacto é o já descrito em 1.2.

#### F16 · Política de captura do trace (I, N4, N7) — pequeno-médio

- **Decisão:** D6.
- **Ficheiros:** `core/_trace.py:23-40` (`StepTrace.input_keys: dict[str, tuple[str, ...]]`),
  `core/_step_engine.py:43-55` (`capture` kwarg: `"keys"` → `input_state={}` e `input_keys`;
  `"full"` → `copy.deepcopy(snapshot.to_dict())`; `"none"` → nada), `toolkit/flow/_flow.py`
  (`trace_capture`), `_executor.py` (passa ao engine), `docs/flow-architecture.md` (Trace),
  `docs/safety.md` (modos de trace), `CHANGELOG` (*Changed*, quebra).
- **Prova:** default → `input_state == {}` e `input_keys["operational"]` lista as chaves;
  `"full"` com a `deque` de N4 → os três traces registam `[1,2,3]`, `[2,3]`, `[3]`;
  react de 20 turnos → `len(json.dumps(trace.to_dict()))` cresce linearmente com os turnos
  (assert contra um limite generoso); `from_dict` aceita traces antigos sem `input_keys`.
- **Depende de:** F8 (plumbing do motor); pode ser feito antes só pelo kwarg de `execute_step`.

#### F17 · Exportar a superfície de gates — trivial

- **Ficheiros:** `core/_tools/__init__.py:26-59`, `core/__init__.py`, `ai_arch_toolkit/__init__.py`,
  `tests/test_core_exports.py`, `docs/safety.md` (Gates: exemplo de gate próprio).
- **Prova:** `from ai_arch_toolkit import ToolGate, ExecutionContext, GateBlock, GateModify,
  GateDryRun` funciona; um gate próprio em `ToolGroup(gates=[…])` tipa sem `Any`.
- **Depende de:** nada.

#### F18 · Deriva de documentação — acompanha cada PR

- `docs/flow-architecture.md:396` "Cost, usage, confidence propagate up automatically" → o gasto
  é medido no scope partilhado (`result.meter`); a atribuição por nó vem de `StepTrace.children`
  (depois de 8e). `:219` (children) passa a ser verdade.
- `docs/moderation.md:62` e `docs/middleware.md:115` — depois de F3.
- `docs/agents.md:53-54` — depois de F2 (policy por step; timeout envelope).
- `docs/safety.md:88` — depois de F11; secção Gates depois de F17; pipeline depois de F13.
- `docs/tools.md:86` — depois de F7.
- `docs/flow-architecture.md` (Scope) — `transform` por camada; `enrich` vê o snapshot filtrado.
- Não tocar em `docs/concurrency.md` (já correcto).

### 3.5 Sequência de PRs sugerida

| PR | Conteúdo | Tamanho |
|---|---|---|
| 1 | F1 | S |
| 2 | F2 | M |
| 3 | F4 + F6 + F7 + F17 (tools, todos triviais) | S |
| 4 | F5 + F10 (LLM) | S |
| 5 | F11 + F12 | S |
| 6 | F9 + F14 | S |
| 7 | F3 (+ N5) | M-L |
| 8 | F8 (8a→8f, commits separados) | L |
| 9 | F16 | S-M |
| 10 | F13 | M |
| 11 | F15 | S-M |

PRs 1-6 são independentes entre si e do resto; podem ser feitos em paralelo por agentes diferentes
sem conflitos de ficheiro relevantes (só o `CHANGELOG.md` colide — resolver na junção).

---

## 4. O que recomendo não corrigir (e porquê)

1. **N/O — timeout nunca retentado; `Result(error=)` sempre retentado.** Comportamentos
   defensáveis e **já documentados** no diagrama de decisão (`docs/flow-architecture.md:147-159`)
   e no tipo `OnTimeout = Literal["halt", "fallback"]`. `Result.error` significa "este step
   falhou"; a convenção "erros como strings" do `AGENTS.md` aplica-se a *tools*. Acrescentar
   `Result.retryable` é API nova. **Fazer:** uma frase a dizer que um timeout nunca é retentado,
   mesmo com `retry` configurado.
2. **H.1/H.2 — modo inferido de `after=`.** É previsibilidade, não segurança (a própria revisão o
   corrigiu). Um `Flow(mode=…)` explícito é design, não correcção. **Fazer:** documentar a
   inferência e expor `Flow.mode` (propriedade read-only) se se quiser; nada mais agora.
3. **`ServerTool.config` descartado.** Encaminhar `max_uses`/`allowed_domains` exige as formas
   por provider (e o Anthropic mudou de `web_search_20250305` para `web_search_20260209` nos
   modelos actuais). É funcionalidade. **Fazer no máximo:** `warnings.warn` em `prepare_tools`
   quando `config` não está vazio.
4. **`Scope.transform` por camada.** É coerente com um contentor em camadas; a surpresa
   resolve-se com uma frase na doc (F14 inclui-a).
5. **`ToolGroup.max_calls` por instância.** Documentado e com alternativa correcta
   (`BudgetPolicy.max_tool_calls`). Repor por run dentro do `Agent` criaria corridas entre runs
   concorrentes — a revisão verificou que hoje não há contaminação e é melhor manter.
6. **`Response.to_message()` a embutir `_raw`.** O Gemini precisa dele (thought signatures);
   o custo de serialização é tratado por F16, não por o retirar.
7. **`inference_limit` não abranger streams.** Documentado em dois sítios e com razão (um stream
   abandonado seguraria o semáforo).
8. **`Agent.from_flow(init_state=…)` só semear `operational`.** É a escape hatch; quem precisa das
   quatro camadas desce a `flow.run(State(...))`.
9. **`ToolError.type` ser `str`.** `GovernanceOutcome` é fechado onde importa; fechar `ToolError.type`
   parte quem constrói `ToolResult.failure("network_error", …)` como a doc exemplifica.
10. **Renomear `iter`→`iter_events`, `Agent.stream()`, MCP, checkpoint/retoma, `FlowSpec`,
    máquinas de estados, catálogo de modelos, `server_tools=` nas estratégias, eventos de flows
    aninhados no stream do pai.** Tudo âmbito, não contrato partido. Concordo com a revisão em que
    F3 e F8 são pré-requisitos de `Agent.stream()`; não concordo em os fazer *como parte* de
    `Agent.stream()`.
11. **`StreamResponse.__anext__` a reparsear o JSON parcial a cada chunk** (`_response.py:240-243`,
    O(n²) em CPU em streams longos de texto). Micro-achado meu; só importa em streams de dezenas
    de KB e `partial_parsed` é uma conveniência. Deixar, ou tornar lazy quando alguém pedir.
12. **A parte `_stream_sync` de P como item autónomo.** Fica dentro de F15; se F15 se saltar, o
    impacto é limitado ao chunk seguinte (1.2).

---

## 5. Notas para quem pegar nisto

- Os repros que produziram os números desta secção foram scripts descartáveis; cada ficha indica
  o teste que os substitui. Não reconstruir os scripts — escrever os testes.
- A ordem das vagas é a ordem de risco: tudo o que está na vaga 1 é local e barato; F3 e F8 são
  onde o toolkit pode partir de formas novas, e é por isso que têm as provas mais longas.
- Três coisas mudam contrato público de forma visível para a app: F2 (timeouts passam a ser
  impostos), F5 (fusão de `system`), F11 (aprovação nas `dangerous`). Avisar antes de actualizar
  a versão instalada na app (`f210d6f`).
