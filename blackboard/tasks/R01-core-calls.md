# R01 · Fase 1 — Núcleo de chamadas: correcções locais, erros tipados, meter e pipeline única

- **Dono:** Codex (coordenador da fase) · **Estado:** done · **Depende de:** nada · **Regras:** `R00-rules.md`
- **Plano:** `docs/internal/hardening-plan.md` — causa 1, costuras A e C, secção 1
- **Achados (`FINDINGS.md`):** "Qualquer chamada LLM falhada fica com custo desconhecido…" (2026-09-17,
  impeditivo), "Cancelar à espera do `inference_limit`…", "Tentativas falhadas de fallbacks
  intermédios…", "`reserve="strict"` não reserva o custo de tools com preço", "Com `max_cost`, uma
  server tool é negada antes de correr", e os nove da `F25-local-fixes.md`
- **Decisões em vigor:** D7, D8, D9, D15–D23

## Objectivo

Com um tecto de custo activo, um 429, um 5xx, uma falha de rede, um timeout ou um cancelamento deixam
de matar o run: retry, fallback e os steps seguintes funcionam, e o tecto continua a ser respeitado.
O ciclo de vida de uma chamada ao LLM passa a existir uma só vez, para `complete`, `stream` e
`stream_events`. `core/_llm.py` sai mais pequeno e mais simples do que entrou.

## Passos, por ordem

### 0. `F25-local-fixes.md` (nove correcções locais)

Faz os nove itens como a ficha manda e regista-os lá. São independentes do resto.

### 1. Instrumentos (sem mudar comportamento)

- `tests/test_failure_matrix.py`, a partir de `prototypes/…/meter/`: matriz parametrizada de classe de
  falha (resposta 429, 5xx e 4xx; falha de ligação; timeout de leitura; cancelamento de quem chama;
  erro ao construir o pedido; resposta ilegível; erro a meio de um stream; stream abandonado) × modo
  do scope (sem scope, só medir, `max_cost` com `reserve="none"`, `max_cost` com `reserve="strict"`) ×
  caminho (`complete`, `stream`, `stream_events`) × recuperação (nenhuma, retry, fallback, chamada
  seguinte no mesmo scope, retry e fallback de step com `Policy`). Cada célula afirma chamadas ao
  provider, resultado, `llm_calls`, `cost`, tecto incerto, desconhecidos sem tecto e se a admissão
  seguinte passa. As células que hoje falham entram como `xfail(strict=True)`; no fim da fase não
  resta nenhuma.
- Teste de conservação do meter (`tests/metering/`): sequências aleatórias de operações com sementes
  fixas; nada pendente depois de fechar; contagens iguais às operações iniciadas; `replay(eventos)`
  igual à projecção.
- `tests/test_quality_budget.py`: corre `ruff check --select C901,PLR0912,PLR0915` (limiar 10) sobre
  `src/ai_arch_toolkit/core` e `toolkit` (sem `nanope`) e compara com uma linha de base em ficheiro;
  falha se aparecer uma função nova acima do limiar ou se uma da linha de base piorar, e obriga a
  encolher a linha de base quando uma melhora. Hoje: 35 funções em `core/`, 64 em `toolkit/`.
- `tests/test_architecture.py`: `core` nunca importa `toolkit`. As fases seguintes acrescentam regras.

### 2. Costura A — modelo de erros (`core/_exceptions.py`)

Base `ProviderError` com `delivery: Literal["not_sent", "unbilled", "indeterminate"]`, obrigatório na
construção. Por baixo: `RequestError` (também `ValueError`; erro ao preparar; `not_sent`), `APIError`
(mantém `status_code` e `body`) e `RateLimitError` (mantém `retry_after`), `TransportError` (também
`ConnectionError`) e `ProviderTimeout` (também `TimeoutError`), `ResponseError` (HTTP 200 inutilizável
ou falha dentro do stream). Quem já apanha `APIError`, `ConnectionError`, `TimeoutError` ou
`ValueError` continua a apanhar os mesmos casos. `PROVIDER_ERRORS` passa a ser a base. `network_error()`
devolve os tipos novos. O retry decide por tipo e estado, sem `getattr(exc, "status_code")`.
Classificação nesta fase: `RateLimitError` é `unbilled` em todos os fornecedores; os outros `APIError`
e as falhas de transporte são `indeterminate` (a política documentada por fornecedor e a detecção da
fase de ligação entram na R02). Exporta os nomes novos em `core/__init__.py` e no topo.

### 3. Meter: disposição da falha e desconhecido com tecto

- `MeterOperation.fail(disposition, …)`: `not_sent` e `unbilled` liquidam custo zero conhecido;
  `indeterminate` fica incerto. A contagem da chamada mantém-se nos três. `UsageEvent` leva a disposição.
- `Cost.unknown(reason, at_most=Money | None)`. O store soma os tectos num contador próprio
  (`MeterSnapshot`: tecto incerto e contagem de incertos com tecto); `unknown_cost_count` passa a
  contar só os desconhecidos **sem** tecto. O tecto entra nos checks de `max_cost` do store e do
  `BudgetController` e no tecto por step (`core/_step_engine.py::_span_spend`), nunca no `cost` reportado.
- De onde vem o tecto: com `reserve="strict"`, a reserva da própria operação fica retida em vez de
  libertada; com `reserve="none"`, o controller calcula-o na altura da falha com o mesmo estimador do
  modo estrito (o tamanho do pedido só se calcula nesse caminho); sem controller não há tecto. A
  capacidade opcional do controller exprime-se com um `Protocol` `runtime_checkable`, não com `getattr`
  (o `wants_request_size` actual passa pelo mesmo tratamento).
- `unpriced="fail_closed"` só dispara com desconhecidos sem tecto (modelo sem preço, server tool).
- `BudgetReport`: `cost` continua a ser o gasto conhecido; novo `cost_at_most`; `cost_uncertain`
  verdadeiro com qualquer incerto.
- C08c: sob `reserve="strict"`, o estimador reserva o preço de uma tool que o `Pricer` preça; preço
  desconhecido ou pricer que levanta nega.

### 4. Costura C — pipeline de tentativa única (`core/_attempts.py`, novo)

- Uma tentativa física é sempre: admitir → vaga de inferência (`inference_slot`) → `mark_started()` →
  despachar → liquidar ou falhar com a disposição do erro. Regista sempre um `Attempt`, também o de um
  fallback intermédio que falhou.
- Uma só regra para retry e fallback: permitidos enquanto nada foi entregue a quem chamou (`complete`
  nunca entrega antes do fim; um stream deixa de poder depois do primeiro item). `AdmissionDenied`
  continua terminal.
- `complete`, `stream`, `stream_events` e os wrappers síncronos usam esta pipeline. Mantêm-se: o
  middleware assíncrono à volta da cadeia inteira (F03), a reserva na criação do stream e o início na
  primeira tentativa, com libertação quando o stream nunca é iterado ou o middleware o rejeita (D8), a
  liquidação antes do `aafter`, e a finalização segura a partir de outra thread nos streams síncronos.
- O ciclo de vida do stream passa a ser um objecto explícito e tipado. Desaparecem os ganchos que
  viajam num `Callable` e se descobrem por `getattr` (`_stream_abandon`, `_stream_refresh`,
  `_stream_release`, `_stream_attempts`, `_meter_op`) em `_llm.py`, `_response.py` e `_sync.py`.
- A chamada ao provider fica isolada numa única função da pipeline: a R02 troca-a por
  `prepare()` + `send()` sem tocar no resto.
- `LLM` fica como fachada: assinaturas públicas iguais. Desaparecem `_try_with_tracking`, `_StreamRun`,
  `_FallbackStreamRun`, `_single_stream` e `_stream_with_fallbacks` como implementações paralelas.

### 5. Documentação e registo

`docs/safety.md` (tabela de custo por classe de falha, que é a especificação da matriz; desconhecido
com tecto; o que `unpriced` passa a querer dizer), `docs/llm.md` (erros, regra de retry e fallback),
`docs/budget` onde existir, `CHANGELOG.md`, blackboard.

## Aceitação

- A matriz de falhas passa inteira, sem `xfail`. Em particular, com `BudgetPolicy(max_cost=5.0)`: 503
  e 429 seguidos de sucesso com `retry=2` → duas chamadas ao provider e resposta `ok`; primário 503 com
  fallback → o fallback responde; timeout de quem chama seguido de nova chamada → a segunda corre;
  `Step` com `Policy(max_cost=1.0)` depois de um retry com sucesso → sem "could not be priced".
- Os testes que fixavam o comportamento antigo estão corrigidos e listados
  (`tests/test_llm_metering.py::test_stream_retry_meters_every_physical_attempt` e os que aparecerem).
- A suite antiga de streams, fallback, middleware, metering e sync passa; o que tiver de mudar fica
  justificado na ficha.
- `core/_llm.py` abaixo de 800 linhas (hoje 1527); nenhuma função acima de 60 linhas nem de
  complexidade 10 em `_llm.py` e `_attempts.py`; a linha de base de complexidade desce.
- Nenhum `getattr` sobre finalizers, providers ou controllers em `core/_llm.py`, `_attempts.py`,
  `_response.py` e `_sync.py` (regra nova em `tests/test_architecture.py`).

## Fora do âmbito

O interior dos adaptadores para lá dos tipos de erro que levantam (R02); regras de preço e D16 (R02);
tools e motor de flows para lá dos itens da F25 (R03); a frente C (`C01`–`C09`).

## Registo do dono

- **Estado:** done; passos 0–5 concluídos pela ordem. R02 e R03 não iniciadas.
- **Notas de desenho:** F25 conserva as notas anteriores às nove correcções; as notas dos passos 1–5 ficam abaixo. Não há caminho antigo paralelo, shim nem descoberta dinâmica de capacidades nos quatro módulos protegidos por AST.
- **Ficheiros tocados:** 32 módulos fonte enumerados no saldo abaixo; AGENTS, CHANGELOG, docs llm/safety/pricing/concurrency/tools-catalog/internal/metering-plan, fichas R01/F25 e BOARD/DECISIONS/FINDINGS/LOG. Testes novos de matriz, conservação, qualidade, arquitectura, erros, disposição e pipeline; ensaios locais dos SDKs em tests/integration/test_attempts_local_sdk.py; alterações aos testes antigos listadas no relatório.
- **Testes:** 3070 → 3886 passed (+816 casos, dos quais 2 da continuação), 22 live_api deselected; matriz de 720 células inteiramente verde, sem xfail. O ficheiro temporário failure_matrix_xfails.json e o código de tolerância foram apagados. quality_baseline.json conserva só a dívida legada medida e deve encolher quando ela melhora.
- **Verificação:** R00 completa no fim de cada passo. Gate final (continuação): 3886 passed, lint/formato limpos, pyright sem erros/avisos, lock actualizado. Há dois avisos antigos de top_logprobs nos testes Astra.
- **Saldo:** fonte tocada 10064 → 9801 (−263); fachada e novas casas da tentativa/ciclo de vida 1561 → 1094 (−467). Fachada isolada 532 linhas.
- **Complexidade:** dívida ruff 172 → 161 diagnósticos; C901 >10 core 35 → 29, toolkit 64 → 64. Fachada e pipeline têm C901 máximo 9 e funções máximas de 40 e 33 linhas.
- **Bloqueios:** nenhum na R01. Os bloqueios iniciais do import nanope e da consulta de preços foram resolvidos pela autorização explícita do dono ("sim"). Só se migrou o import CSV nessa árvore; só se consultaram gratuitamente as duas páginas oficiais de preços.
- **Desvios:** itens F25 independentes avançaram durante o bloqueio, como permite a R00; nenhum instrumento começou antes do gate verde. F25.8 é só prosa, com prova por leitura prevista na ficha. As correcções do instrumento e da revisão final estão explicadas abaixo; não se baixou o contrato nem se criou nova tolerância. O parâmetro temperature incompatível com anthropic 1.6.0 fica na R02, com reprodução em FINDINGS.
- **Restrições:** sem commits, pushes, branches ou worktrees; sem fornecedores pagos; sem leitura do .env; sem dependências novas. SDKs oficiais exercitados exclusivamente no servidor falso dos protótipos em 127.0.0.1.

### Histórico do gate F25 e desbloqueio

Cada linha incluiu os cinco comandos R00. Nos itens 1–8, as mesmas 17 falhas eram o import CSV nanope proibido; lint/formato/pyright/lock estavam limpos. O registo da F25 conserva os vermelhos e o pedido de autorização. Ambos os bloqueios estão resolvidos.

| F25 | Passed | Failed | Deselected | Ruff / formato / pyright / lock |
|---|---:|---:|---:|---|
| 1 | 3055 | 17 | 22 | limpos |
| 2 | 3058 | 17 | 22 | limpos |
| 3 | 3060 | 17 | 22 | limpos |
| 4 | 3061 | 17 | 22 | limpos |
| 5 | 3061 | 17 | 22 | limpos |
| 6 | 3062 | 17 | 22 | limpos |
| 7 | 3084 | 17 | 22 | limpos |
| 8 | 3084 | 17 | 22 | limpos |
| Migração autorizada do import | 3101 | 0 | 22 | limpos |
| 9 (preços verificados) | 3107 | 0 | 22 | limpos |

## Retoma · Passo 1: nota de desenho (antes de código)

A matriz cruza 10 falhas × 4 scopes × 3 caminhos × 6 recuperações. Um provider scripted que herda da interface actual suporta todos os caminhos sem rede. Uma tabela de resultados esperados determina entrega, recuperação, contagens e custo; as únicas tolerâncias são ids explícitos de células vermelhas em xfail estrito, apagados no fim. Conservação reconstrói a projecção a partir de eventos, com sementes fixas e operações deixadas pendentes para testar close. Qualidade é medida por ruff, com chaves por módulo/função/código e uma baseline JSON exacta: qualquer aumento, função nova ou tolerância obsoleta falha. AST fixa a independência core/toolkit com canário que injecta uma importação proibida. Não muda código fonte.

F25 concluída: 3107 passed, 22 deselected e R00 limpa. Autorização do dono resolveu import nanope e preços externos.

### Passo 1 · Resultado

Matriz: 176 passed e 544 xfails estritos, enumerados a partir do vermelho sem alteração de fonte. Mais 8 testes de conservação/canários/arquitectura/qualidade. Baseline: 172 diagnósticos ruff, dos quais C901 são 35 core + 64 toolkit. R00: 3291 passed, 544 xfailed, 22 deselected; lint/formato/pyright/lock limpos.

### Passo 2 · Nota de desenho

`ProviderError(message, *, delivery)` é a única base e casa da disposição, alias Literal `Delivery`. RequestError fixa not_sent e mantém ValueError; APIError conserva status/body e assume indeterminate nesta fase; RateLimitError fixa unbilled e mantém retry_after; TransportError/ProviderTimeout conservam os except builtins, ResponseError distingue resposta ilegível. As subclasses recebem a disposição por construção; a base exige-a. A compatibilidade dos construtores APIError(status, body) mantém-se como classificação por omissão especificada nesta fase, sem shim. `network_error` devolve os tipos novos. Retry lê retry_after por isinstance e status por APIError; desaparece a descoberta dinâmica nesses campos. PROVIDER_ERRORS contém só a base. Os duplos de testes de fallback que emitiam builtins crus passam a emitir os erros normalizados dos adaptadores, sem alterar os except públicos.

### Passo 2 · Resultado

11 novos testes de erros ficaram vermelhos antes dos nomes/disposições; depois passam. 87 testes focados verdes. R00: 3302 passed, 544 xfailed, 22 deselected; lint/formato/pyright/lock limpos. A linha de base de complexidade não piorou.

### Passo 3 · Nota de desenho

`Cost.unknown(reason, *, at_most=None)` distingue desconhecido limitado. `MeterSnapshot.uncertain_cost` (Money) e `uncertain_cost_count` conservam o tecto separado do gasto; unknown_cost_count só conta ilimitados. `MeterOperation.fail(disposition)` recebe a classificação. Store unifica falha/fecho numa transição terminal e usa a mesma projecção do settle, retirando `_fail_started` e a duplicação dos factos em `_LiveOp`. Dois Protocols runtime_checkable exprimem tamanho de pedido e `failure_bound(request, reservation)`. Uma callback tipada de factos (`failure_request`) só calcula tamanho na falha suave, fora do lock; em strict retém-se exactamente a reserva. O verificador neutro de limites fica no core e é partilhado pelo store e BudgetController, eliminando as duas tabelas de checks duplicadas. Report mantém cost conhecido e acrescenta cost_at_most. Estimador passa a preço de tools via Pricer, com erros a negar strict. Testes: custo zero com contagem, tectos exactos a bloquear admissão concorrente, callbacks fora do lock, report, conservation/replay e tools com preço.

### Passo 3 · Resultado

9 testes novos ficaram vermelhos antes da alteração. Conservação inclui agora desconhecidos limitados e ilimitados; corrigido o teste antigo que negava prematuramente uma server tool em reserve=none (a primeira corre, a liquidação desconhecida impede a próxima). 120 xfails estritos passaram e foram retirados, sem novas tolerâncias. R00: 3431 passed, 424 xfailed, 22 deselected; lint/formato/pyright/lock limpos.

### Passo 4 · Nota de desenho

`Execution(owner, request, arguments, path)` captura scope/span na criação; `items()` executa a cadeia e `finalize(text)`/`abandon()` são o ciclo de vida explícito dos wrappers. Uma tentativa física possui reserva, estado e registo; admitir, adquirir vaga, iniciar, despachar e terminar existem apenas nesse objecto. O despacho é uma função isolada sobre BaseProvider, Request e um Literal de caminho. Middleware envolve cada cadeia, com conjunto de candidatos visitados para não repetir fallbacks partilhados; os registos pertencem à execução, incluindo candidatos que levantam. Streams reservam na criação, refrescam os factos após abefore, libertam antes de começar quando recusados e preservam a liquidação antes de aafter. Sem middleware a confirmação de consumo nos wrappers liquida a operação, para abandono síncrono não ser confundido com o worker ter enchido a fila. O fecho usa Protocols tipados para iteradores e um ciclo de vida opcional explícito; finalizers públicos continuam callbacks simples, sem atributos mágicos. Desaparecem as cinco implementações paralelas citadas na ficha. Provas: matriz vermelha, cancelamento enquanto espera pela vaga, todos os fallbacks no histórico, canários AST de capacidade/dispatch e limites de linhas/complexidade.

Vermelho do passo 4 observado: cancelamento na fila já aparecia como chamada iniciada; complete omitia o fallback intermédio falhado; AST encontrou ganchos dinâmicos e fachada de 1564 linhas. Corrigido também um erro do instrumento: o tecto literal da matriz usava apenas 2 caracteres de conteúdo; o estimador vigente recebe 35 caracteres do pedido serializado, pelo que o tecto exacto é $0.000507 (9 input + 32 output). A estimativa/preço não mudou.

Correcções adicionais do instrumento, com os três exemplos vermelhos guardados em `/tmp/ai-arch-r01-instrument-corrections-red.log`: cancel/abandon só devem interromper a primeira tentativa scripted (antes interrompiam também o retry de step, ou abandonava-se a segunda resposta "ok"); num scope apenas de medição, sem controller, a falha continua ilimitada e o max_cost do step tem de recusar mesmo depois de sucesso. A matriz afirma agora explicitamente essa recusa fail-closed e também o custo conhecido da resposta bem sucedida; não atribui um tecto onde o contrato diz que não existe.

### Passo 4 · Resultado

Pipeline integrada em todos os caminhos e wrappers; fachada 1561 → 532 linhas. Máximos medidos por AST: funções de 40 linhas na fachada e 33 na pipeline; ruff C901 ≤10 em ambas. Dívida: 172 → 161 diagnósticos (core C901 35 → 29; toolkit 64 → 64). Os 720 casos da matriz passam; o ficheiro de xfails foi apagado. Corrigidos, com vermelho primeiro, consumo de uma Response vazia (conserva parsed/usage/metadata) e compatibilidade do construtor BudgetReport. Actualizado o duplo antigo de fallback que simulava `_complete` removido: agora injecta um provider num LLM real e mantém a prova de negação terminal. Os dois testes privados de sizing chamam a casa única request_facts; o adaptador privado `_meter_request` sem consumidores de produção foi apagado.

14 ensaios dos SDKs oficiais em 127.0.0.1 (2 vendors × 3 caminhos × 2 estados HTTP + 2 timeouts), com o servidor dos protótipos. Inspecção local: openai 3.14.1 e anthropic 1.6.0; AsyncOpenAI/AsyncAnthropic aceitam base_url, timeout e max_retries; retries SDK desligados na fixture para contar uma tentativa exacta. Referências de forma de API: https://github.com/openai/openai-python e https://github.com/anthropics/anthropic-sdk-python; confirmadas nas assinaturas instaladas, sem consultar hosts externos. Problema novo de temperature Anthropic registado em FINDINGS para R02, sem o ocultar na produção.

R00 completa: 3880 passed, 22 deselected, zero xfail; lint/formato/pyright/lock limpos. O sandbox bloqueia bind() local, por isso os ensaios e o gate com SDK correram com a permissão de socket local já autorizada pelo dono; nenhum fornecedor pago foi contactado.

### Passo 5 · Nota de desenho

A tabela de docs/safety especifica disposição, custo e recuperação por classe, distinguindo medir sem controller de soft/strict. docs/llm documenta erros normalizados, fronteira de entrega, histórico completo e negação terminal; docs/pricing é a documentação de orçamento disponível (não existe docs/budget). Actualizam-se os docstrings das projecções, estimador e relatório e a descrição da vaga inicial de streams em concorrência, para corresponderem à pipeline. Não muda comportamento. CHANGELOG identifica as duas quebras autorizadas (CSV e fail/disposição/contagem), registos conservam os vermelhos e gates históricos, removendo o relatório intermédio obsoleto que dizia fase incompleta. A prova documental é leitura cruzada com a matriz verde e gate R00 final.

### Revisão final do ciclo de vida

Alargada a prova de cancelamento na fila aos dois streams. Dois testes novos vermelhos descobriram que um worker síncrono rápido podia guardar "firstlast" antes de o consumidor fechar após "first"; o fecho já retinha a incerteza, mas o relatório parcial usava texto não consumido. finalize consulta agora o estado de consumo depois da liquidação e reconstrói a parcial quando abandonado. A matriz passou a afirmar também a classe exacta do resultado falhado, além de sucesso/contagem/custo. Não há nova tolerância. O gate R00 é repetido antes de aceitar a fase.


### Passo 5 · Resultado e aceitação

Docs e docstrings cruzados com a matriz; CHANGELOG e registos actualizados. Gate final repetido depois da revisão: 3884 passed, 22 deselected, zero xfail; ruff check e format --check limpos (433 ficheiros), pyright 0 errors/0 warnings, uv lock --check limpo. RequestError e AdmissionDenied são terminais na regra única can_recover, partilhada por retry e fallback.

### Continuação · verificação independente (Claude, 2026-09-18)

O Codex ficou sem créditos depois de escrever esta ficha, o BOARD e o LOG, e antes da mensagem
final. A verificação repetiu o gate R00 (3884 passed, 22 deselected; ruff, formato, pyright e lock
limpos), os critérios de aceitação medidos (fachada com 532 linhas; funções com máximo de 40 e 33
linhas; C901 ≤ 10 em `_llm.py` e `_attempts.py`; nenhum `getattr`/`hasattr` nos módulos
protegidos; nenhum `xfail`, `noqa` ou `type: ignore` novo) e os protótipos originais
`meter/retry_under_cap.py` e `meter/failure_matrix.py`, agora verdes sob `max_cost=5`. Encontrou
duas falhas de acabamento, ambas nesta fase:

- **Nota de desenho 1 (teste).** `tests/integration/test_attempts_local_sdk.py` importava o servidor
  falso de `blackboard/prototypes/`, que o README dos protótipos declara descartável e não
  importável: o CI ficava dependente dessa pasta. O servidor passa para
  `tests/integration/fakeserver.py`, só com os dois comportamentos usados (`status` e `hang`),
  tipados com `Literal`, e com a porta efémera atribuída pelo próprio `start_server`, em vez de
  `free_port()` (que fechava a porta antes de a reabrir). O protótipo fica como evidência para a
  R02, que parte deste módulo de testes. Sem mudança de comportamento: prova pelo gate.
- **Nota de desenho 2 (CHANGELOG).** O `fallback_on` por omissão passou de
  `(APIError, ConnectionError, TimeoutError, OSError)` para `(ProviderError,)` sem entrada
  própria. Os adaptadores só levantam os tipos normalizados (19 chamadas a `network_error`), por
  isso o efeito visível é só para `OSError` crus (um ficheiro local em falta deixa de accionar o
  fallback) e para providers próprios que levantem builtins. Acrescenta-se a entrada `Changed`.
- **Nota de desenho 3 (pipeline).** `SyncStreamResponse.close()` abandona pelo handle de ciclo de
  vida, a partir da thread do consumidor, antes de cancelar o worker. Se o worker ganha a vaga de
  inferência nesse intervalo, `attempt.start()` devolve `False`, `_physical_items` termina sem
  erro e `_Chain.items` rebenta no `assert self.response is not None` (`AssertionError()` nu; com
  `python -O`, `TypeError` em `dataclasses.replace(None)`). Uma tentativa que já não pode começar
  levanta `StreamAbandoned`, a mesma excepção do abandono depois do início: uma linha, nenhum caso
  novo. Como `closed` já está marcado, `can_recover` recusa retry e fallback.

### Continuação · resultado

- Vermelho da nota 3: `test_stream_abandoned_in_the_inference_queue_ends_typed_and_unstarted`
  (`stream` e `stream_events`) falhava com `AssertionError` em `_attempts.py:348`; verde com a
  correcção. O teste afirma ainda que o stream nunca chama o provider, que só o *holder* conta
  (`llm_calls == 1`) e que a reserva do stream termina `aborted`. Os 14 ensaios SDK passam com o
  servidor de `tests/integration/fakeserver.py`.
- Gate R00: 3886 passed (+2), 22 deselected; ruff e `format --check` limpos (434 ficheiros),
  pyright sem erros, `uv lock --check` limpo. `_attempts.py` mantém 526 linhas.
- Não verificado: o job `floors` do CI (`--resolution lowest-direct`) com os ensaios SDK novos; o
  cache do uv não tem as versões mínimas e a R00 proíbe rede. Fica para o primeiro CI depois do
  commit.
- Achado novo em `FINDINGS.md`: `meter/step_cap.py` (tecto por step sem `BudgetPolicy`) continua
  vermelho por contrato desta ficha; proposta de decisão para a R02.

## Relatório final · R01 concluída

### Feito por passo

- **0 / F25:** nove correcções concluídas e 37 casos novos. CSV perigoso/com aprovação e import nanope migrado; IP explícito válido antes de I/O; step_end antes da negação temporal; retry 529; documento Anthropic com title; timeout xAI encaminhado; restrição central MediaWiki; AGENTS corrigido; tarifas exactas Claude Fable/Mythos 5.1 e Gemini 3.8 Flash verificadas nas páginas oficiais autorizadas.
- **1 / Instrumentos:** matriz 10 × 4 × 3 × 6 (720 casos); conservação aleatória/replay com quatro sementes; linha de base exacta ruff e canários AST. Os xfails estritos enumerados só encolheram e foram apagados antes da aceitação.
- **2 / Erros:** base ProviderError com disposição obrigatória, subclasses normalizadas e excepts builtins preservados; exports core/topo; network_error e retry tipados. Onze casos novos antes da implementação.
- **3 / Meter:** disposição terminal, desconhecido com tecto separado do gasto, admissão/relatório/limite de step coerentes; uma projecção de falha; callbacks fora do lock; reserva estrita de tools com preço; soft server tools executam primeiro e só o desconhecido ilimitado bloqueia depois. Nove casos novos antes da alteração.
- **4 / Pipeline:** uma tentativa física para os três caminhos e wrappers síncronos; admissão antes da vaga, início depois da vaga; todos os fallbacks registados; middleware e liquidação preservados; ciclo de vida explícito e fecho de geradores delegados. Os novos canários provam casa única, ausência de ganchos dinâmicos e limites da fachada. Catorze ensaios de SDK real no servidor falso local. Revisão final corrigiu resposta parcial síncrona com texto ainda não consumido, com dois vermelhos anteriores à correcção.
- **5 / Documentação e registo:** tabela de falhas em safety, erros/recuperação em llm, orçamento em pricing, concorrência e docstrings actualizados; CHANGELOG e blackboard completos.

### O que não ficou e porquê

Nenhum passo da R01 pendente. Não se iniciou R02/R03. A preparação/envio específica dos adaptadores e a classificação documentada por fornecedor pertencem à R02; nesta fase só o 429 é unbilled por regra comum, os restantes erros enviados ficam indeterminate. O novo achado anthropic 1.6.0 rejeita a temperature que a fachada envia por omissão: a fixture local retira-a explicitamente só nesse SDK para testar pedidos válidos, e a produção não foi alterada fora do âmbito. D22 distingue a construção pura da fachada da preparação específica ainda por migrar. A escolha do transporte HTTP de ip_lookup continua pendente da R03 (D21), como previsto pela F25.

### Testes corrigidos, sem enfraquecimento

- F25: documento exige title e ausência de name; erro remoto de IP usa agora IP explícito válido; construtor xAI afirma timeout (incluindo None), preservando a prova de retries gRPC.
- tests/budget/test_budget.py: a antiga negação prévia de server tool em reserve=none foi substituída por execução inicial e recusa da chamada seguinte após liquidação desconhecida, o contrato pedido pela ficha.
- tests/metering/test_events.py e test_store.py: todos os fail recebem a disposição; os casos antigos conservam a incerteza/contagem que afirmavam.
- tests/test_llm_fallback.py, test_llm_metering.py e test_stream_middleware.py: os duplos lançam os novos TransportError/ProviderTimeout em vez de builtins crus; os excepts públicos continuam compatíveis. Os dois testes privados de sizing afirmam factos na função única request_facts.
- tests/test_admission_terminality.py: retirado o duplo de LLM._complete já apagado; LLM real com provider duplo prova a mesma negação terminal sem API interna paralela.
- test_stream_retry_meters_every_physical_attempt mantém a prova original: no scope apenas de medição, uma tentativa enviada indeterminate continua desconhecida ilimitada, e contam-se ambas as tentativas. Não se atribui ali o tecto de um controller inexistente.
- Instrumento: corrigidos o tamanho serializado/tecto exacto, a interrupção só da primeira tentativa, a recusa do step sem controller e a fronteira pura de RequestError. Os motivos e os vermelhos constam das notas acima; a matriz final também exige o tipo exacto de erro.

### Números e gates completos

Baseline 3070 passed → 3886 (+816). Distribuição: F25 +37, instrumentos +728, erros +11, meter +9, pipeline/ensaios/revisão +29 e continuação +2. Todos os gates excluíram os mesmos 22 live_api; a aceitação tem zero xfail. Não há nova dependência, Any público, type: ignore ou noqa.

| Gate R01 | Passed | Xfailed | Deselected | Ruff / formato / pyright / lock |
|---|---:|---:|---:|---|
| 0 / F25 concluída | 3107 | 0 | 22 | limpos |
| 1 / Instrumentos | 3291 | 544 | 22 | limpos |
| 2 / Erros | 3302 | 544 | 22 | limpos |
| 3 / Meter | 3431 | 424 | 22 | limpos |
| 4 / Pipeline | 3880 | 0 | 22 | limpos |
| 5 / Documentação e revisão final | 3884 | 0 | 22 | limpos |
| Continuação (Claude) | 3886 | 0 | 22 | limpos |

Comandos exactos, com UV_CACHE_DIR=/tmp/ai-arch-r01-uv e UV_OFFLINE=1 no ambiente:

```bash
uv run pytest -m "not live_api" -q
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv run pyright src
uv lock --check
```

A execução com sockets locais precisou da permissão para bind() em 127.0.0.1 bloqueado pelo sandbox; aprovação automática concedida para o servidor falso autorizado. Não se usou a permissão para fornecedores externos. Gate final guardado em /tmp/ai-arch-r01-accepted.log.

Complexidade: ruff C901/PLR0912/PLR0915 172 → 161 diagnósticos; funções C901 >10 core 35 → 29 e toolkit 64 → 64. Nenhuma dívida nova ou piorada. Em _llm.py/_attempts.py, C901 máximo 9 e funções máximas 40/33 linhas. A dívida legada remanescente é visível em quality_baseline.json para R03.

Saldo das 32 fontes tocadas, incluindo os dois módulos novos, contra o checkout inicial:

| Módulo fonte (src/ai_arch_toolkit/) | Antes | Depois | Saldo |
|---|---:|---:|---:|
| __init__.py | 419 | 429 | +10 |
| core/__init__.py | 258 | 271 | +13 |
| core/_concurrency.py | 80 | 80 | +0 |
| core/_default_pricing.toml | 965 | 999 | +34 |
| core/_exceptions.py | 27 | 73 | +46 |
| core/_llm.py | 1561 | 532 | -1029 |
| core/_metering/_admission.py | 175 | 267 | +92 |
| core/_metering/_cost.py | 83 | 93 | +10 |
| core/_metering/_events.py | 48 | 50 | +2 |
| core/_metering/_operation.py | 98 | 100 | +2 |
| core/_metering/_scope.py | 200 | 205 | +5 |
| core/_metering/_store.py | 564 | 533 | -31 |
| core/_providers/_anthropic.py | 896 | 896 | +0 |
| core/_providers/_base.py | 272 | 269 | -3 |
| core/_providers/_xai.py | 544 | 540 | -4 |
| core/_response.py | 484 | 491 | +7 |
| core/_retry.py | 97 | 97 | +0 |
| core/_step_engine.py | 246 | 246 | +0 |
| core/_sync.py | 261 | 260 | -1 |
| core/_tools/_executor.py | 418 | 418 | +0 |
| nanope/advanced_multi_purpose_configurable_agent/_tools.py | 264 | 264 | +0 |
| toolkit/budget/_controller.py | 119 | 81 | -38 |
| toolkit/budget/_estimator.py | 55 | 67 | +12 |
| toolkit/budget/_report.py | 81 | 97 | +16 |
| toolkit/flow/_executor.py | 831 | 831 | +0 |
| toolkit/tools/__init__.py | 289 | 288 | -1 |
| toolkit/tools/_geo.py | 301 | 305 | +4 |
| toolkit/tools/_json.py | 113 | 118 | +5 |
| toolkit/tools/_mediawiki.py | 292 | 314 | +22 |
| toolkit/tools/dangerous.py | 23 | 25 | +2 |
| core/_attempts.py | 0 | 526 | +526 |
| core/_stream_lifecycle.py | 0 | 36 | +36 |
| **Total** | **10064** | **9801** | **−263** |

Fachada + pipeline + ciclo de vida: 1561 → 532 + 526 + 36 = 1094 (−467). _store 564 → 533 (−31); BudgetController 119 → 81 (−38). O saldo negativo inclui as novas casas de responsabilidade, não apenas linhas retiradas da fachada. A contagem inicial real da fachada era 1561; os 1527 da ficha eram anteriores à F24.

### Mudanças visíveis para aplicações dependentes

Quebras explícitas no CHANGELOG: csv_read importa-se de toolkit.tools.dangerous e exige aprovação; MeterOperation.fail exige disposição; unknown_cost_count passa a contar só desconhecidos sem tecto. BudgetReport.cost conserva gasto conhecido e cost_at_most dá o tecto combinado (None se ilimitado); cost_uncertain inclui qualquer incerteza. Os excepts anteriores de APIError/ConnectionError/TimeoutError/ValueError e as assinaturas públicas LLM mantêm-se. Falhas com tecto deixam espaço para retry/fallback/steps dentro do orçamento; cancelamento na fila não conta; streams não repetem depois de entregar e fecham transportes imediatamente; histórico conserva fallbacks intermédios. As mudanças F25 listadas acima também estão documentadas no CHANGELOG.

### O que não foi verificado ao vivo

Sem fornecedores pagos nem leitura do .env. Os catorze ensaios locais validam SDK, estados, timeouts, contagem e meter, mas não provam facturação real nem disponibilidade dos modelos. Forma document/title e encaminhamento timeout xAI foram confirmados no SDK instalado e nos testes de construtor, sem sonda específica ao vivo. Páginas oficiais de preços foram verificadas na excepção gratuita autorizada, com URL/data no TOML e detalhes na F25.

O dono pode correr, com chaves já configuradas no ambiente, depois de resolver na R02 a incompatibilidade Anthropic registada:

```bash
uv run pytest -m live_api -q
```

Para só os contratos existentes: uv run pytest tests/integration/test_provider_contracts_live.py -m live_api -q. Estes comandos não substituem uma sonda específica de facturação/documento/timeout onde a suite existente não a tem.

### Divisão proposta em commits (a fazer pelo dono)

1. Tools locais: CSV/governança e import nanope, IP, MediaWiki, testes e catálogo/safety correspondentes.
2. F25 flow/providers/retry/preços: step_end, title, timeout, 529, tarifas exactas e testes.
3. Migração atómica do núcleo: erros, meter, admissão/budget, pipeline, ciclo de vida/sync, instrumentos, baseline e testes dos SDKs locais. Fazer juntas as mudanças de fail/disposição e todos os consumidores.
4. Documentação final e blackboard: contratos, CHANGELOG, decisões D21–D23, achados e passagem de fase.

Nenhum commit, push, branch ou worktree realizado. A revisão do diff e os commits pertencem ao dono; só depois começa a R02 com contexto limpo.

**Commits (2026-09-18, a pedido do dono, feitos pela continuação):** `a16e3f9` (1), `f95c81f` (2),
`5a0ab3b` (3) e o registo (4), em `main`, sem push. `docs/safety.md`, `docs/llm.md` e
`core/_retry.py` entraram por partes: a linha do `csv_read` no 1, o 529 no 2, o resto no 3 e no 4.
Cada árvore intermédia foi exportada do índice e passou o gate sozinha: 3097, 3107 e 3886 passed,
com ruff, formatação e pyright limpos. Antes do commit 1 corrigiu-se a formatação: a nota do
MediaWiki em `docs/tools-catalog.md` estava colada à lista (em Markdown virava parte do bullet
`wikidata_sparql`) e o TOML de preços tinha três linhas em branco seguidas.
