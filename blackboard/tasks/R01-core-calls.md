# R01 · Fase 1 — Núcleo de chamadas: correcções locais, erros tipados, meter e pipeline única

- **Dono:** por atribuir · **Estado:** todo · **Depende de:** nada · **Regras:** `R00-rules.md`
- **Plano:** `docs/internal/hardening-plan.md` — causa 1, costuras A e C, secção 1
- **Achados (`FINDINGS.md`):** "Qualquer chamada LLM falhada fica com custo desconhecido…" (2026-09-17,
  impeditivo), "Cancelar à espera do `inference_limit`…", "Tentativas falhadas de fallbacks
  intermédios…", "`reserve="strict"` não reserva o custo de tools com preço", "Com `max_cost`, uma
  server tool é negada antes de correr", e os nove da `F25-local-fixes.md`
- **Decisões em vigor:** D7, D8, D9, D15–D20

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

- Estado: todo
- Notas de desenho:
- Ficheiros tocados:
- Testes novos e corrigidos:
- Verificações:
- Saldo de linhas e complexidade:
- CHANGELOG:
- Bloqueios:
- Desvios ao plano:
- Commits propostos:
