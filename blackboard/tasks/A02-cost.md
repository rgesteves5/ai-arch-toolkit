# A02 · Custo: toda a falha tem tecto (G-29), preços com data e os que faltam (G-20)

- **Dono:** Claude (2026-10-04) · **Estado:** done · **Depende de:** nada
- **Origem:** o briefing do ai-network, grupo 2 (G-20, G-29) · **Decisões:** D49, D50 ·
  **Regras:** `R00-rules.md`

## Problema (verificado a 2026-10-04 no `6d1a989`)

- **G-29.**
  - Com um budget, uma falha já tem tecto, e uma nova tentativa num modelo com preço é admitida
    (D20).
  - Só a medir, a falha fica desconhecida, sem tecto. Um `Policy(max_cost=...)` por passo falha
    então depois de um retry com êxito (o achado de 2026-09-18; o protótipo `meter/step_cap.py`
    já não corre contra o contrato de hoje).
- **G-20.**
  - O `claude-fable-5-1` já está certo.
  - Faltam os modelos do catálogo do ai-network: `deepseek-flash`, `mistral-small-2603`,
    `codestral-2508` e `poolside/laguna-s-2.1` (OpenRouter).
  - As promoções (`gpt-5.6-sol` até 2026-11-21; Gemini 3.8, 3.7 e 3.6 Flash até 2026-12-31) só têm
    a data num comentário.

## Nota de desenho (antes do código)

- **G-29 (D49):**
  - `core/_metering/_worst_case.py` (novo): `worst_case(request, pricer) -> Reservation | None`,
    com a lógica e as constantes que estavam no `HeuristicEstimator` (os caracteres por token, a
    folga por parte de media e por imagem pedida);
  - o `HeuristicEstimator` do toolkit passa a delegar nele;
  - o `MeterStore` recebe do `MeterScope` uma função de tecto (o pior caso ao preço do pricer da
    execução, ou da tabela);
  - o `_fail_cost` usa a reserva estrita quando a há, e senão essa função com os factos medidos
    (`failure_request`), com ou sem controller;
  - saem o `FailureBoundController` e o `BudgetController.failure_bound`.
- **G-20 (D50):**
  - `ModelPricing` ganha `until: date | None` e `then: ModelPricing | None`;
  - `PricingRegistry.get(model, on=None)` devolve o preço do dia (UTC, por uma função `_today`
    que os testes fixam);
  - o `estimate_cost` e o `price` usam o `get`;
  - o TOML aceita `until` (uma data) e uma subtabela `then`;
  - os preços em falta e os de depois das promoções vêm das páginas oficiais, cada um com o URL.
- **Provas:**
  - um passo com `Policy(max_cost=1.0)` sem budget, com um 5xx e um retry com êxito, passa; antes
    falhava;
  - uma falha só a medir fica incerta com tecto; uma falha sem preço fica desconhecida;
  - o preço muda no dia a seguir ao `until`, no `estimate_cost` e no meter;
  - o TOML com `then` carrega;
  - os modelos novos têm preço.

## Ficheiros

- `src/ai_arch_toolkit/core/_metering/_worst_case.py` (novo), `_store.py`, `_scope.py`,
  `_admission.py`
- `src/ai_arch_toolkit/toolkit/budget/_estimator.py`, `_controller.py`
- `src/ai_arch_toolkit/core/_pricing.py`, `src/ai_arch_toolkit/core/_default_pricing.toml`
- os testes de metering, budget, pricing e passo que mudarem; os testes novos
- `docs/safety.md`, `docs/pricing.md`, `docs/internal/metering-plan.md` (se falar do tecto),
  `AGENTS.md`, `CHANGELOG.md`

## Registo do dono

- Estado: done (Claude, 2026-10-04), pela nota de desenho.
- **G-29 (D49):**
  - `core/_metering/_worst_case.py` (novo), com as constantes que eram do `HeuristicEstimator`;
  - o `MeterStore` recebe do `MeterScope` a função de tecto: o pior caso ao preço do pricer da
    execução, ou da tabela, com um import tardio porque o `_pricing` importa o pacote do meter;
  - o `_fail_cost` usa a reserva estrita quando há uma, e senão essa função;
  - saíram o `FailureBoundController` e o `BudgetController.failure_bound`;
  - o `HeuristicEstimator` delega no core.
- **Testes que afirmavam o contrato antigo, corrigidos:**
  - `tests/test_failure_matrix.py`: 150 casos do modo "measure" esperavam a falha desconhecida e
    o passo com `max_cost` recusado depois de um retry com êxito. A expectativa mudou num sítio só,
    e os 720 casos passam;
  - `tests/flow/test_flow_metering.py`: seis testes de abandono, erro e fim cedo de streams só a
    medir;
  - `tests/test_llm_metering.py`: seis testes (o primeiro mudou de nome, para
    `test_failed_attempt_keeps_the_count_and_a_cost_ceiling`);
  - `tests/metering/test_failure_disposition.py`: dois testes davam ao budget sem reserva estrita
    um estimador próprio para o tecto. Passam a dar o preço pelo pricer da execução
    (`FixedPrice`, `PeekingPricer`).
- **G-20 (D50):**
  - `ModelPricing.until`/`then` e `ModelPricing.on(day)`;
  - `PricingRegistry.get(model, on=)`, que lê o dia por `_pricing._today` (UTC);
  - o TOML aceita `until` e uma subtabela `then`, e um `until` que não seja data é recusado ao
    carregar;
  - todos os testes leem os preços num dia fixo (2026-10-04, fixture `price_day` no
    `conftest.py`), para não mudarem de veredicto quando uma promoção acaba.
- **Preços** (lidos a 2026-10-04 nas páginas oficiais, com os URLs na tabela):
  - `gpt-5.6-sol`: a OpenAI diz "at least through November 21, 2026" e não dá o preço seguinte.
    O `then` é o preço de antes da promoção (página arquivada de 2026-08-20), o que conta a mais
    se a promoção for prolongada. As escritas em cache de contexto longo e o fast do `then` seguem
    as proporções de hoje (2×);
  - Gemini 3.8, 3.7 e 3.6 Flash: até 2026-12-31, e depois o preço de 2027 que a Google publica;
  - DeepSeek: `deepseek-flash` (com os aliases v4) e `deepseek-v4-pro`, ao preço de hora de ponta
    (fora da ponta custa metade, o que a tabela não modela; conta a mais);
  - Mistral: `mistral-small-2603` e `codestral-2508`, com os `-latest`. O Magistral Small 1.2 e o
    Devstral Small 2 estão descontinuados e sem preço, e ficaram de fora;
  - Poolside: `poolside/laguna-s-2.1`, ao preço de lista. O OpenRouter aplica 10% de desconto, o
    que conta a mais, e anuncia o fim do endpoint para 2026-10-31.
- **Testes novos:** `tests/metering/test_worst_case.py` (5), `TestDatedPrices` (5) e
  `TestTablePromotionsAndVendors` (14) em `tests/test_pricing.py`.
- **Docs:** `docs/safety.md`, `docs/pricing.md` (o tecto, as promoções e os fornecedores),
  `AGENTS.md`, `CHANGELOG`.
- Gate: 6256 passed, 42 skipped; ruff, formatação e pyright limpos.
- **Linha para o Registo do briefing do ai-network** (o dono leva-a): "o grupo 2 fechou: toda a
  chamada que falha a meio tem tecto, com ou sem budget (só a medir também), e um passo com
  `max_cost` passa depois de um retry com êxito; os preços têm data de fim (`ModelPricing.until`
  e `then`, `pricing.get(model, on=)`), com o `gpt-5.6-sol` a mudar depois de 2026-11-21 e os
  Gemini Flash depois de 2026-12-31; o DeepSeek (`deepseek-flash`), o Mistral
  (`mistral-small-2603`, `codestral-2508`) e o Poolside (`poolside/laguna-s-2.1`) têm preço. Na
  app, o 'sem preço' das falhas medidas e o `unpriced` desses modelos no catálogo deixam de ser
  precisos."
