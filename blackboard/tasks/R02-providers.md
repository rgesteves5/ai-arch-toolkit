# R02 · Fase 2 — Fornecedores: ids e preços, contrato de três fases, um adaptador de cada vez

- **Dono:** Claude (coordenador da fase) · **Estado:** done · **Depende de:** R01 · **Regras:** `R00-rules.md`
- **Plano:** `docs/internal/hardening-plan.md` — causas 2, 3 e 4, costuras B e D, secção 4
- **Achados (`FINDINGS.md`, 2026-09-15 e 2026-09-17):** todos os que falam de adaptadores, streams,
  preços e regras por modelo; em particular "Erros de transporte a meio de um stream…", "Validação
  local do SDK…", "`stream_events()` perde `parsed`…", "Gemini em stream reenvia só o último chunk",
  "Stream sem usage…", "Gemini nunca junta os resultados de tools…", "Anthropic: um `user` por
  `tool_result`", "`thinking=True` no Anthropic…", "`tool_choice="required"` recusado no Fable 5.1",
  "xAI: `tool_choice` com nome…", "Gemini 3: `thinking_effort`…", "O Gemini muda de pilha HTTP…",
  "Server tools no fio…", "O preço por prefixo…", "Regras por modelo: o ramo por omissão é o antigo"
- **Decisões em vigor:** D3, D11–D14, D15, D16, D18, D20
- **Ordem dos adaptadores (D18):** OpenAI → xAI → Gemini → Meta → Anthropic. Acaba um antes de
  começar o seguinte.

## Objectivo

Cada adaptador fica reduzido a quatro peças pequenas — construir o pedido (tipado pelo SDK), mapear
erros (um só sítio), descodificar eventos, montar a resposta (uma só vez) — sobre uma base que é dona
do algoritmo. Streaming e `complete()` dão a mesma `Response` por construção. As regras por modelo e
os preços resolvem ids por uma gramática única, e um modelo novo recebe por omissão o comportamento
da geração mais recente.

## Passos, por ordem

### 1. Costura D — ids de modelo e preços (`core/_model_id.py`, novo)

- Uma função resolve um id em: exacto; id com sufixo de snapshot (gramática de datas e carimbos por
  fornecedor: `-20250929`, `-2025-04-14`, `@20251101`, `-latest`, `-001`, `-preview-05-20`, `-0709`);
  família; desconhecido. Casos que têm de dar "desconhecido": `o3-pro`, `o3-deep-research`,
  `gpt-4o-audio-preview`, `gpt-5-codex`, `gemini-2.5-flash-image`, `grok-4.6-mini`, `claude-fable-5-1`
  sem entrada própria.
- Usam-na o `PricingRegistry` (deixa de casar por prefixo; `register(..., match="prefix")` fica como
  opção explícita para famílias locais), o encaminhamento por fornecedor (`_providers/__init__.py`), a
  escolha do tokenizer (`core/_tokens.py`) e os perfis dos adaptadores. Regra de arquitectura:
  nenhuma verificação `startswith` sobre ids de modelo fora de `_model_id.py`.
- D16/D20: com um `MeterScope` ligado, um modelo sem preço falha antes de qualquer pedido, com um
  erro que diz como registar o preço; um modelo local regista zero de forma explícita; sem scope a
  chamada corre e `Response.cost` fica `None`.
- `scripts/audit_models.py`: compara as listas de modelos dos fornecedores (APIs de modelos, grátis)
  com a tabela de preços e os perfis. Só o dono o corre; não o corras.

### 2. Costura B — contrato do provider (`core/_providers/_base.py`)

- Três fases obrigatórias: `prepare()` síncrono e puro (levanta `RequestError`), `send()` e
  `open_stream()` com toda a E/S, e uma só montagem da `Response`. `complete`, `stream` e
  `stream_events` existem uma vez, na base. A pipeline da R01 passa a chamar `prepare()` antes de
  admitir a operação e `send()` depois: um erro de construção nunca abre operação nem conta como
  tentativa.
- Um duplo de teste partilhado (`tests/` — um `FakeProvider` que cumpre o contrato) substitui os
  providers duck-typed que só têm `complete`/`stream` (27 sítios em 21 ficheiros). Não fica nenhum
  caminho alternativo em `LLM` para providers sem contrato.
- Confirma SDK a SDK, nas versões instaladas, se o próprio SDK acumula o objecto final do stream; onde
  acumula, usa-o; só escreve junção à mão onde não há (Gemini).

### 3. Por adaptador, pela ordem de D18

Para cada um, tudo isto antes de passar ao seguinte:

- **Um mapeador de erros** à volta de cada `await` do SDK e de cada passo da iteração do stream.
  Nenhuma excepção de SDK, `httpx2`, `aiohttp` ou gRPC sai crua. Cada erro sai tipado (costura A) com
  `delivery` segundo a política de facturação documentada do fornecedor, citada em comentário com URL:
  Anthropic não cobra respostas de erro e cobra timeouts do cliente; Gemini não cobra 400 nem 500;
  429 é `unbilled` em todos; o que não está documentado fica `indeterminate`. Falhas da fase de ligação
  (`ConnectError`, `ConnectTimeout`, `PoolTimeout` na causa) são `not_sent`. Erros dentro do stream
  mapeiam para o estado documentado equivalente, nunca 200 nem 5xx inventado sem base.
- **Pedido tipado:** o construtor devolve o `TypedDict` do SDK (modelos pydantic no Gemini, protobuf
  no xAI), para o pyright verificar o fio. Regra de arquitectura: excepções de SDK só no mapeador.
- **Perfil por modelo:** uma tabela declarativa por adaptador (família → forma do pedido), resolvida
  pela costura D; o perfil por omissão é o da geração mais recente e as famílias antigas são lista
  fechada. O que o modelo não suporta levanta `RequestError` em `prepare()`, como a Meta (D13).
- **Paridade:** a mesma resposta lógica por `complete()` e por stream em chunks dá `Response` igual
  (texto, tool calls, thinking, `parsed`, `citations`, `response_id`, usage, e o que o histórico
  reenvia). Stream sem usage dá custo desconhecido.
- **Contrato de conversa:** históricos neutros canónicos (uma chamada; duas chamadas paralelas e dois
  resultados; texto com chamadas; thinking com chamadas; histórico depois de stream) convertidos
  segundo as regras documentadas do fornecedor: resultados de um turno juntos e pela ordem das
  chamadas, com `id` onde existe, assinaturas preservadas.
- **Testes de transporte** com `prototypes/…/adapter-phases/fakeserver.py` como ponto de partida:
  ligação recusada, servidor mudo, fecho sem resposta, RST, 429, 500, 529, erro dentro do stream,
  corte a meio, e o caminho de sucesso. Cada caso dá o tipo de erro e o `delivery` certos e bate com a
  matriz da R01.

Específico de cada um (além do que os achados listam):

- **OpenAI:** o erro dentro do stream deixa de sair como `openai.APIError` cru; o limite de tokens
  vai em `max_completion_tokens` no host oficial e em `max_tokens` só com `base_url` próprio; server
  tools no Chat Completions e `ServerTool.config` não vazio levantam `RequestError` (a funcionalidade
  é do `C05`); o corpo JSONL do batch passa pelo mesmo construtor tipado.
- **xAI:** `tool_choice` com nome de tool usa a forma do SDK; os códigos gRPC deixam de virar 5xx
  inventados (`UNAVAILABLE` e `DEADLINE_EXCEEDED` são falhas de transporte); o comentário sobre server
  tools fica certo.
- **Gemini:** resultados de tools juntos num só `Content`, com o `id` da chamada; o `raw` de um stream
  guarda todas as partes e assinaturas; pilha fixa em `httpx` por `HttpOptions.httpx_async_client`
  (acaba o reenvio escondido); `thinking_effort` inválido levanta `RequestError`. A nota "Known issue"
  em `docs/model-compatibility.md` e no `AGENTS.md` só sai depois de o dono confirmar ao vivo.
- **Meta:** os desvios aos tipos do SDK `openai` que o servidor aceita ficam numa lista explícita,
  cada um com a nota de verificação ao vivo do M01; o usage de um `response.failed` deixa de se perder.
- **Anthropic:** thinking segundo a documentação actual — 4.6 e seguintes com `{"type": "adaptive"}`
  e `output_config.effort` (fundido com `output_config.format`), `display: "summarized"` quando se pede
  thinking, `thinking=False` omite o parâmetro onde o thinking vem ligado, só as famílias antigas levam
  `budget_tokens` com `max_tokens = budget + max_tokens` (D20); `tool_choice` forçado em Fable 5.1 e
  Mythos 5.1 levanta `RequestError`; todos os `tool_result` de um turno num só `user`; server tool com
  `name`; ramo 429 em `count_tokens` e batch. Confirma cada regra na documentação oficial antes de a
  codificar. Já feito antes da fase (2026-09-18): `temperature`, `top_p` e `top_k` vão em
  `extra_body`, porque o `anthropic` 1.x os tirou das assinaturas; `_TEMPERATURE_DEPRECATED_PREFIXES`
  passa para a tabela de perfis.

### 4. Rede de testes do fio

A fixture que valida cada pedido construído contra os tipos do SDK (protótipos em
`prototypes/…/wire-contract/`) fica confinada a `tests/`, com os canários: os pedidos maus conhecidos
têm de falhar e os bons de passar. Serve para o que a tipagem estática não vê.

### 5. Verificação ao vivo (só preparar)

Não há créditos Anthropic nem xAI e o agente nunca chama fornecedores. Deixa testes `live_api`
pequenos e baratos por adaptador (modelo mais barato, `max_tokens` baixo): chamadas paralelas com
reenvio em `complete` e em stream, thinking ligado e desligado, um erro 4xx sob `max_cost` seguido de
sucesso. Escreve na ficha o comando exacto por fornecedor para o dono correr.

### 6. Documentação e registo

`docs/llm.md`, `docs/pricing.md`, `docs/model-compatibility.md`, `docs/safety.md`, `AGENTS.md`
(gotchas e a deriva sobre o structured output, se a F25 não a tiver fechado), `CHANGELOG.md`, blackboard.

## Aceitação

- Testes de paridade, de contrato de conversa e de transporte verdes nos cinco adaptadores.
- Regras de arquitectura verdes: excepções de SDK só nos mapeadores; ids de modelo só em
  `_model_id.py`; nenhum provider sem contrato.
- Saldo de linhas negativo em `core/_providers/` (hoje 4294 linhas nos cinco adaptadores; 44 blocos
  `except` de SDK); nenhuma função nova acima de complexidade 10; pyright limpo com os pedidos tipados.
- A matriz da R01 continua verde, agora com erros de construção a não abrirem operação.

## Fora do âmbito

Funcionalidades novas de server tools, `server_tools=` nas estratégias e custo por uso (`C05`);
`Agent.stream()` (`C01`); catálogo de modelos (`C06`); porta para a Responses API do OpenAI; tools e
motor (R03).

## Registo do dono

- Estado: done (passos 1 a 6), commitada a pedido do dono (`cf2aa43`, `a8c3eea` e o registo). Falta
  o dono correr as verificações ao vivo (relatório final, no fim desta ficha).
- Notas de desenho: abaixo, uma por passo, escritas antes do código (salvo a do Meta: desvio
  assumido, ver "Passo 3 · Meta · resultado").
- Por adaptador (feito, testes, linhas antes e depois): nas secções "Passo 3 · <adaptador> ·
  resultado". Linhas: OpenAI 890 → 741, xAI 540 → 492, Gemini 655 → 597, Meta 757 → 673,
  Anthropic 910 → 777.
- Ficheiros tocados: `src/` 19 ficheiros (tabela no relatório final); `scripts/audit_models.py`
  (novo); `pyproject.toml` e `uv.lock` (mínimo do `xai-sdk`); 70 ficheiros de `tests/`, 14 deles
  novos; `docs/llm.md`, `docs/pricing.md`, `docs/model-compatibility.md`, `docs/safety.md`,
  `docs/tools.md`, `docs/getting-started.md`, `docs/framework-overview.md`, `README.md`,
  `AGENTS.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `examples/25_server_tools.py`; blackboard
  (esta ficha, `BOARD.md`, `DECISIONS.md` D24–D27, `FINDINGS.md`, `LOG.md`).
- Testes novos e corrigidos: por passo, nas secções de resultado.
- Verificações: o gate da R00 verde no fim de cada passo; tabela no relatório final.
- CHANGELOG: entradas em `[Unreleased]` para os preços por id exacto e o `UnpricedModelError`
  (dois **Breaking**), o despacho e a entrega, a paridade dos streams, o custo sem usage, os cinco
  adaptadores, o `MeterOperation.fail(..., usage=, cost=)` e o mínimo do `xai-sdk` (na entrada
  **Breaking** dos SDKs).
- Comandos de verificação ao vivo para o dono: no relatório final.
- Bloqueios: nenhum.
- Desvios ao plano: no relatório final.
- Commits: feitos a pedido do dono, como propostos no relatório final.

### Início (2026-09-18)

Base: `main` @ `b3dae3f`, publicada. Leitura obrigatória feita pela ordem da R00. SDKs instalados:
`anthropic` 1.6.0, `openai` 3.14.1, `google-genai` 2.24.0, `xai-sdk` 1.19.0 (`grpcio` 1.84.0),
`httpx` 0.28.1 e `httpx2` 2.13.0 (o transporte dos dois primeiros), `aiohttp` 3.14.3. A documentação
oficial é lida só como página pública (sem chaves, sem APIs de fornecedores); cada regra cita o URL.

### Passo 1 · Nota de desenho (costura D)

- **Interface.** `core/_model_id.py`, novo. `snapshot_base(model) -> str | None` devolve o id sem o
  sufixo de snapshot, por uma só expressão: `-AAAAMMDD` (Anthropic), `-AAAA-MM-DD` (OpenAI),
  `@AAAAMMDD` (Vertex), `-NNN` e `-preview-MM-DD`/`-preview-MM-AAAA` (Gemini), `-MMDD` (xAI e OpenAI
  antigos), `-latest`. `lookup(model, exact, families=None) -> ModelMatch[T] | None` resolve por
  ordem fixa: entrada própria; entrada do id sem snapshot; família (prefixo mais longo, só quando o
  chamador passa famílias); senão `None`, que é o desconhecido. `ModelMatch` é um dataclass congelado
  com `kind` (`Literal["exact", "snapshot", "family"]`), `key` e `value`.
- **Quem a usa:** `PricingRegistry.get`, o encaminhamento (`_match_provider`), o tokenizer
  (`_get_correction`, `_get_encoding`) e, no passo 3, as tabelas de perfil de cada adaptador.
- **Preços.** A tabela passa a ids exactos. Um id real que partilha a tarifa de outro entra como
  `aliases = [...]` na própria entrada do TOML: são os ids que o TOML dizia cobrir por prefixo
  (`grok-4-1-fast-reasoning`, `grok-3-fast`, `gemini-3.1-pro-preview`…) e os que a página de modelos
  lista. `register(model, pricing, *, match="exact" | "prefix")`: o prefixo passa a opção explícita
  para famílias locais; o mesmo `match` vale numa entrada de TOML carregada com `load()`. Um snapshot
  com tarifa própria ganha pela entrada exacta (`gpt-4o-2024-05-13`, `gpt-3.5-turbo-1106`, verificados
  na página oficial). `_load_toml` deixa de listar os 26 campos à mão (`fields(ModelPricing)`).
- **D16/D20.** Com um `MeterScope`, a pipeline verifica antes de abrir a operação que o pricer do
  scope dá um custo conhecido ao modelo (`price(factos sem server tools, Usage())`; o custo das server
  tools é do C05). Se não dá, levanta `UnpricedModelError`, subclasse de `RequestError` (logo
  terminal: nem retry nem fallback), com a instrução para registar o preço e a de registar zero para
  um modelo local. Sem scope nada muda: a chamada corre e `Response.cost` fica `None`.
- **Desaparece:** os quatro casadores de prefixo do `core` fora dos adaptadores (`PricingRegistry.get`,
  `_get_correction`, `_get_encoding`, `_match_provider`).
- **Regra de arquitectura.** Em `core/`, nenhum `startswith`/`endswith` sobre uma expressão cujo nome
  fala de modelo, fora de `_model_id.py`. Com canário. Os quatro adaptadores que ainda os têm
  (`_openai`, `_xai`, `_gemini`, `_anthropic`) ficam numa lista de tolerância que encolhe no passo 3 e
  é apagada no fim da fase.
- **`scripts/audit_models.py`.** Lê as listas de modelos pelos SDKs (grátis) e compara-as com os
  preços e os perfis; a comparação é uma função pura testada com ids falsos. Só o dono o corre.
- **Testes.** Gramática com os casos da ficha, incluindo os sete que têm de dar desconhecido; preços
  (snapshot com tarifa própria, alias, prefixo explícito, `unregister`); encaminhamento e tokenizer
  com os mesmos resultados de hoje; admissão sem preço (nenhum pedido, nenhuma operação, mensagem
  accionável; sem scope, `cost=None`); arquitectura.

### Passo 1 · Resultado

- **Vermelho observado:** 16 falhas antes do código em `tests/test_pricing.py` e
  `tests/test_architecture.py` (variantes com o preço da base, `register` sem `match`, entradas em
  falta, `startswith` sobre ids em `_pricing`, `_tokens` e no encaminhamento); `test_model_id.py` e
  `metering/test_unpriced_model.py` nem importavam.
- **Feito:** `core/_model_id.py`; preços por id exacto ou snapshot, com `aliases` e `match="prefix"`
  (TOML e `register`); tokenizer e encaminhamento pela gramática; `UnpricedModelError` e a
  verificação D16 antes de abrir a operação (`_require_price` em `_attempts.py`); a regra de
  arquitectura com a lista de tolerância dos quatro adaptadores; `scripts/audit_models.py`.
- **Tabela de preços:** os blocos que o TOML dizia serem aliases passaram a `aliases` (os ids
  antigos do Grok, que o xAI factura como `grok-4.3`; Daybreak Blue/Red; `grok-code-fast*`;
  `grok-build-latest`). Ids reais acrescentados como alias, cada um confirmado numa página
  oficial a 2026-09-18: `https://ai.google.dev/gemini-api/docs/models` (os `-preview` do Gemini
  3.x), `https://docs.x.ai/docs/models` (`grok-4.20-0309-*`, `grok-4.20-multi-agent-0309`),
  `https://developers.openai.com/api/docs/pricing` (`gpt-5.1` à tarifa do `gpt-5`; entradas novas
  `gpt-4o-2024-05-13` a 5/15, `gpt-3.5-turbo-1106` a 1/2, `gpt-5.5-cyber` a 12,50/1,25/75). O
  `o3-pro` aparece na página de preços, mas o TOML regista-o como retirado da `/v1/models` e a
  ficha exige-o desconhecido: fica sem entrada. Os ids que a gramática não cobre e que eu não pude
  confirmar ficam para o `scripts/audit_models.py` do dono.
- **Testes corrigidos (afirmavam a herança por prefixo, que a D16/R6 retira):** em
  `tests/test_pricing.py`, `test_register_custom_model` (agora
  `test_register_is_exact_and_covers_dated_snapshots`), `test_unregister`,
  `test_reset_clears_custom` e os cinco `TestLoad` que liam `my-model-v1`; `tests/test_core_exports.py::test_core_exports_pricing`.
  **Afirmavam o contrato antigo da D16** (um modelo sem preço corria sob um scope):
  `tests/flow/test_flow_metering.py` — os dois testes de `unpriced` passam a usar server tools como
  fonte de custo desconhecido (a intenção, fail-closed e `allow`, fica) e há um teste novo que prova
  que o modelo sem preço não corre, com `fail_closed` e com `allow`;
  `test_policy_max_cost_fails_closed_on_unknown_cost` idem; `tests/test_foreign_code_safety.py` —
  os pricers que rebentam e que estimam respondem à sonda de preço e só falham ao liquidar (a
  intenção N6 fica), mais um teste novo para um pricer que rebenta já na sonda;
  `tests/test_llm_metering.py::test_openai_compatible_server_is_attributed_to_openai` regista o
  preço zero do modelo local, que é a via D16.
- **Testes novos:** `tests/test_model_id.py` (26), `TestIdGrammar` e os de `register`/TOML em
  `tests/test_pricing.py`, `tests/metering/test_unpriced_model.py` (7), dois de arquitectura, dois
  do `audit_models`, um de pricer.
- **Gate R00:** 3954 passed, 22 deselected; ruff, formato, pyright (0/0/0) e lock limpos.
- **Saldo:** Python +94 (`_model_id.py` +69 novo; `_attempts.py` +28 com a verificação D16;
  `_pricing.py` −5, `_tokens.py` −4, encaminhamento −2, `_exceptions.py` +4, exports +4); TOML
  −126. Total −32.

### Passo 2 · Nota de desenho (costura B)

- **Confirmado nos SDKs instalados.** Acumulam o objecto final do stream: `anthropic`
  (`AsyncMessageStream.get_final_message()`), `openai` (`ChatCompletionStreamState`; uso o
  `current_completion_snapshot`, porque `get_final_completion()` levanta `LengthFinishReasonError`
  com `finish_reason="length"`), `xai-sdk` (`chat.stream()` dá pares `(resposta acumulada, chunk)`)
  e a Responses API da Meta (o evento `response.completed` traz a resposta inteira). O
  `google-genai` não acumula: o Gemini leva uma junção dos chunks. Os clientes `httpx2` do
  `anthropic` e do `openai` aceitam `http_client=` com `event_hooks`.
- **Interface.** `BaseProvider[P, F]` (P: o pedido preparado; F: o objecto final do SDK), com seis
  peças por adaptador: `prepare(request: Request) -> P`, síncrono e puro (levanta `RequestError`);
  `async send(p) -> F`; `open_stream(p) -> AsyncIterator[StreamEvent | Done[F]]`, que dá texto e
  thinking à medida e termina com `Done(final)`; `assemble(final, p) -> Response` (a única
  montagem); `usage(final) -> Usage | None`; `map_error(exc) -> ProviderError | None` (o único
  sítio que conhece excepções do SDK). A base é dona do algoritmo: `complete(p) -> Answer` e
  `stream(p) -> AsyncIterator[StreamEvent | Answer]` correm as fases dentro do mapeador (o `send`
  e cada passo do stream); a montagem é a mesma nos dois caminhos; os eventos `tool_call` saem da
  resposta montada, por isso são sempre os de `Response.tool_calls`; o custo calcula-se num só
  sítio (custo do fornecedor, senão tabela, senão `None` quando não houve usage). `Answer` junta a
  `Response` e `usage_reported`, que a liquidação usa: sem usage, custo desconhecido (C01a).
- **Pipeline.** `dispatch(provider, prepared, path)` passa a chamar só `complete`/`stream`. O
  `prepare` corre em `Execution` antes de qualquer admissão (um `RequestError` nunca abre operação)
  e outra vez depois do `abefore` quando há middleware; se falha aí, a reserva do stream é libertada
  sem contar. A tentativa guarda os eventos que viu, para a resposta parcial de um stream
  abandonado; `StreamState` e a montagem de `_attempts._response` desaparecem. `stream()` e
  `stream_events()` do `LLM` são vistas da mesma sequência (só texto, ou os eventos).
- **Comum a todos:** `parse_options` lê as opções do toolkit uma vez (os cinco `kwargs.pop` e os
  cinco avisos de parâmetro desconhecido passam a um) e `parse_structured` faz o `parsed` (quatro
  cópias passam a uma).
- **Adaptadores, neste passo:** só a reorganização nas seis peças, com o comportamento de hoje (os
  mesmos tipos de erro, o mesmo fio). O que a ficha pede por adaptador (entrega documentada,
  `not_sent` na ligação, pedido tipado, perfis, testes de paridade, conversa e transporte) é o
  passo 3, pela ordem da D18. O Gemini ganha já a junção dos chunks, porque a montagem única
  precisa de um objecto final.
- **Duplo de teste.** `tests/fake_provider.py`: `FakeProvider(*replies)` cumpre o contrato;
  `Reply` diz resposta, texto em chunks, erro (antes do primeiro chunk ou depois de N), espera
  num `asyncio.Event`, usage reportado ou não; o duplo regista os pedidos, conta chamadas e
  concorrência, e aceita recusas no `prepare`. Substitui os 29 duplos duck-typed em 23 ficheiros
  e o `ScriptedProvider` da matriz. Não há caminho para um provider sem contrato: a pipeline chama
  `prepare`, que um duplo duck-typed não tem.
- **Desaparece:** `StreamState`, o `stream_events` por omissão da base, `_attempts._response`, os
  ciclos de stream duplicados da Anthropic (um só `open_stream`), a acumulação manual de tool calls
  em quatro adaptadores, e os `# type: ignore[assignment]` dos duplos.
- **Testes (vermelho primeiro):** contrato da base (paridade por construção num provider mínimo;
  `tool_call` a partir da resposta; stream sem usage → custo desconhecido; erro do SDK a meio do
  stream passa pelo mapeador; `prepare` falhado não abre operação nem conta, também num stream com
  middleware); a matriz da R01 com a célula `request` a falhar no `prepare` do provider.

### Passo 2 · Resultado

- **Vermelho observado:** `tests/test_provider_contract.py` nem importava (`Answer`, `Done` não
  existiam); os comportamentos que prova (paridade por construção, `tool_call` a partir da
  resposta, stream sem usage com custo desconhecido, recusa do adaptador sem operação) eram os
  achados C01a e "Erro do adaptador a montar o pedido envenena o budget".
- **Feito:** `BaseProvider[P, F]` com as seis peças e o algoritmo na base (`complete`, `stream`,
  mapeador por passo, montagem única, custo num só sítio, `Answer`); `parse_options` e
  `parse_structured` comuns (esta usada já pelos cinco; `parse_options` entra adaptador a
  adaptador no passo 3); `StreamState` e o `stream_events` por omissão apagados; pipeline com
  `prepare` antes de qualquer admissão e outra vez depois do `abefore` com middleware;
  `dispatch` só com `complete`/`stream`; liquidação pelo `Answer` (sem usage → custo
  desconhecido); resposta parcial a partir dos eventos vistos. Os cinco adaptadores reorganizados
  nas seis peças com o comportamento de hoje; os dois ciclos de stream da Anthropic passam a um e o
  usage cumulativo vem do acumulador do SDK (sai o `_merge_delta_usage`); o OpenAI acumula com o
  `ChatCompletionStreamState` do SDK; o Gemini ganha a junção dos chunks; a Meta lê a resposta
  final do evento terminal. Os eventos `tool_call` saem da resposta montada (sai a acumulação de
  tool calls em quatro adaptadores).
- **Duplo de teste:** `tests/fake_provider.py` (`FakeProvider`, `Reply`, `fake_llm`) substitui os
  29 duplos duck-typed e os ~180 providers `AsyncMock`/`MagicMock` em 33 ficheiros; a matriz da R01
  usa-o (a célula `request` falha agora no `prepare` do provider e o `ConstructionFailureLLM`
  sai — a D22 fica cumprida). `tests/provider_calls.py` chama um adaptador pelo contrato;
  `tests/sdk_streams.py` dá os streams dos SDKs com os seus tipos e acumuladores reais (o
  `accumulate_event`/`build_events` do `anthropic`, `ChatCompletionChunk` do `openai`), em vez de
  namespaces feitos à mão. A migração dos 33 ficheiros foi feita por três agentes em paralelo,
  com regras escritas (sem git, sem `src/`, sem enfraquecer testes, cada adaptação justificada) e
  revista aqui; um deles apontou um teste fraco
  (`test_complete_primary_denial_does_not_enter_fallbacks`: o fallback também era negado pelo
  mesmo tecto, por isso o teste passava com a negação mascarada), reforçado com um controller que
  só nega o modelo primário e provado com um canário que torna tudo recuperável.
- **Testes adaptados ao contrato (nenhum apagado para passar):** a lista completa está nos
  relatórios dos agentes, resumida aqui. Mudança de semântica, não bug: os eventos `tool_call`
  vêm depois do texto e na ordem da resposta (`test_stream_events::test_mixed_event_types`, Meta
  `test_stream_events`); a resposta final é a montada, não a concatenação dos chunks
  (`test_text_chunk_concatenation`, `test_llm::test_stream_sync`); um `cost=` escrito no duplo é
  recalculado pela base; `request.kwargs` inclui os valores por omissão do `LLM`
  (`test_phase_overrides::test_llm_kwargs_reach_reviewer_with_precedence` passa a usar 0,7 para
  distinguir os casos); um erro que o duplo antigo levantava ao abrir o stream sai agora no
  primeiro passo (seis testes de `test_stream_fallback`, dois de `test_admission_terminality`);
  o `raw` de um stream xAI e as tool calls vêm da resposta acumulada pelo SDK (os duplos passam a
  acumular como o SDK); com o acumulador do SDK da Anthropic, JSON de tool inválido rebenta o
  stream (`test_malformed_tool_args_fail_the_stream`; o mapeamento tipado é do passo 3). Testes
  retirados com o que testavam: os três `TestStreamState` (o tipo deixou de existir) e as
  parametrizações `stream`/`stream_events` dos testes de usage e de sistema da Anthropic (o
  provider tem um só stream); por ficheiro: `test_anthropic_provider.py` 83 → 78,
  `test_stream_tool_calls.py` 29 → 28, todos os outros iguais ou maiores.
- **Testes novos:** `test_provider_contract.py` (14), `test_provider_base.py` reescrito (1),
  Gemini `test_stream_raw_keeps_every_chunk_part`, OpenAI `TestUsageReport`, o teste de terminalidade
  reforçado.
- **Gate R00:** 3963 passed, 22 deselected; ruff, formato, pyright (0/0/0) e lock limpos. Os 14
  ensaios dos SDKs reais em loopback passam com os adaptadores novos.
- **Saldo:** os cinco adaptadores 3752 → 3022 (−730); `_base.py` 269 → 393 (+124); `_attempts.py`
  554 → 568 (+14). Blocos `except` de excepções de SDK nos adaptadores: 44 → 0 (um mapeador por
  adaptador). Dívida de complexidade 161 → 137 (−24: os ciclos de stream da Anthropic, OpenAI, Meta
  e Gemini e o `_parse_sdk_response` do Gemini).
- **Para o passo 3 (achado aqui):** um 200 com corpo ilegível (cobrado) e uma recusa local do SDK
  (nada enviado) saem ambos como `ValueError`/`TypeError` — `json.JSONDecodeError` no primeiro;
  `TypeError` (um `set` no corpo) e `ValueError` ("Streaming is required…", Anthropic) no segundo;
  provado em loopback. O tipo não chega para a entrega: o mapeador precisa de saber se o pedido foi
  despachado (hook `request` do cliente HTTP, como previa o protótipo `exp_hook.py`).

### Passo 3 · Nota de desenho (comum aos cinco)

- **Despachado ou não.** O passo 2 provou em loopback que o tipo da excepção não chega (um 200
  ilegível e uma recusa local do SDK saem ambos como `ValueError`). A base cria um marcador por
  chamada (um por stream inteiro) num `ContextVar`; o hook `request` do cliente HTTP do SDK
  (`event_hooks`, aceite pelos clientes `httpx2` do `anthropic`/`openai` e pelo `httpx` que o
  Gemini vai usar) marca-o quando o SDK entrega o pedido ao transporte; o xAI marca-o antes da RPC,
  porque todo o trabalho local já está no `prepare`. `map_error(exc, *, sent)` recebe o marcador.
  Uma excepção que não é do SDK sai por um helper da base: antes do despacho é `RequestError` (o SDK
  recusou o pedido; nada enviado), depois é `ResponseError` (a resposta não se leu; incerto). O
  duplo de teste não o usa, por isso as excepções arbitrárias dos testes da pipeline continuam como
  estão.
- **Transporte:** um helper da base classifica a causa `httpx`/`httpx2`: `ConnectError`,
  `ConnectTimeout` e `PoolTimeout` são `not_sent`; os outros timeouts são `ProviderTimeout` e o
  resto `TransportError`, `indeterminate`. Sai o `network_error`.
- **Entrega documentada, com URL no comentário do mapeador.** 429 `unbilled` em todos (D20); o
  resto segundo a página de cada fornecedor; sem documento, `indeterminate`.
- **Pedido tipado:** cada `prepare` devolve o tipo do SDK (`TypedDict` no `openai`/`anthropic`,
  modelos pydantic no Gemini, protobuf no xAI) e o pyright verifica o fio.
- **Perfil por modelo:** uma tabela por adaptador resolvida com `_model_id.lookup`; o perfil por
  omissão é o da geração actual e as famílias antigas são lista fechada; o que o modelo não aceita
  levanta `RequestError` no `prepare`.
- **Opções:** cada adaptador passa a ler as opções com `parse_options` (sai o bloco de `pop`).
- **Testes por adaptador:** paridade (`tests/test_provider_parity.py`: a mesma resposta pelo
  `send` e pelo stream em chunks dá campos iguais), contrato de conversa
  (`tests/test_provider_conversations.py`: históricos neutros canónicos → fio), transporte
  (`tests/integration/test_provider_transport.py`, com o servidor falso em loopback alargado a
  ligação recusada, servidor mudo, fecho sem resposta, RST, 429, 500, 529, erro no stream e corte a
  meio). A lista de tolerância da regra de ids encolhe adaptador a adaptador.

### Passo 3 · OpenAI · nota de desenho

- **Documentação (2026-09-18):** `max_tokens` está obsoleto em favor de `max_completion_tokens` e
  não serve os modelos o-series (https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create);
  valores de `reasoning_effort` por modelo (https://developers.openai.com/api/docs/models/gpt-5:
  minimal–high; https://developers.openai.com/api/docs/models/gpt-5.2: none–xhigh); Astra sem
  `temperature`/`top_p`/`logprobs`/`top_logprobs`, sem `none`, e tools só pela Responses API
  (https://developers.openai.com/api/docs/guides/latest-model); o gpt-4o não tem raciocínio
  (https://developers.openai.com/api/docs/models/gpt-4o); a cobrança de respostas de erro não está
  documentada (https://developers.openai.com/api/docs/guides/error-codes), só o 429 do Flex
  (https://platform.openai.com/docs/guides/flex-processing); no SDK 3.14.1 o `tools` do Chat
  Completions é `ChatCompletionFunctionToolParam | ChatCompletionCustomToolParam` (sem
  `web_search`). A regra "sem `temperature` diferente de 1 quando raciocina" vem da prova ao vivo
  de 2026-04-28 (`scripts/model_probe_notes.md`) e fica como está.
- **Perfis:** `_CURRENT` (raciocina: com thinking tira a `temperature` que não seja 1, salvo
  `none`), `_LEGACY` (lista fechada: gpt-4o, gpt-4o-mini, gpt-4.1*, gpt-4-turbo, gpt-4,
  gpt-3.5-turbo — thinking levanta `RequestError`), `_ASTRA`, e `_COMPATIBLE` para um `base_url`
  próprio (sem regras da OpenAI; passa `max_tokens`). O limite de tokens vai por host:
  `max_completion_tokens` no host oficial, `max_tokens` só com `base_url` próprio.
- **Server tools** e `ServerTool.config` não vazio levantam `RequestError` (a funcionalidade é do
  C05). O corpo JSONL do batch sai do mesmo `prepare`.
- **Erros:** `RateLimitError` (unbilled); `APIStatusError` → `APIError` (indeterminate);
  `APIConnectionError` pela causa `httpx2`; `httpx2.TransportError` a meio do stream; o
  `openai.APIError` de um evento de erro dentro do stream → `ResponseError`; o resto pelo helper do
  despacho.

### Passo 3 · OpenAI · resultado

- **Vermelho observado:** transporte em loopback com o SDK real — o 200 ilegível e o erro dentro
  do stream saíam crus (`JSONDecodeError`, `openai.APIError`), a ligação recusada saía
  `indeterminate` e a recusa local do SDK (`TypeError` de um `set`) saía crua; unitários — `gpt-4o`
  recebia `max_tokens` no host oficial, `thinking=True` num modelo sem raciocínio e server tools
  iam para o fio, o Astra levantava `ValueError` sem tipo, e o corpo do batch levava `max_tokens`
  (11 falhas).
- **Feito:** a infra-estrutura comum na base (marcador de despacho por `ContextVar`,
  `on_request` para os `event_hooks`, `map_error(exc, *, sent)`, `transport_error`,
  `refused_or_unread`); o adaptador OpenAI com `prepare` tipado pelo SDK
  (`CompletionCreateParamsNonStreaming`, mensagens, partes, tools e `tool_choice` tipados; o
  pyright apanhou dois erros reais no fio ao tipar: o `response_format` sem tipo e
  `stream`/`stream_options` passados duas vezes), `parse_options`, tabela de perfis resolvida pela
  gramática, limite de tokens por host, server tools recusadas, batch pelo `prepare`, e o mapeador
  com entrega documentada e despacho.
- **Testes novos:** `tests/integration/test_provider_transport.py` (13 casos com o SDK real em
  loopback: 429, 500, 529, fecho sem resposta, RST, servidor mudo, 200 ilegível, erro no stream,
  corte a meio, sucesso, ligação recusada, recusa local do SDK), `TestProfiles` (13),
  `tests/test_provider_parity.py` e `tests/test_provider_conversations.py` para o OpenAI (8; a
  paridade já vinha do passo 2 e a conversa já estava certa: ficam como guarda), a regra de
  arquitectura "excepções de SDK só no mapeador" (com canário) e a lista `_PENDING_ADAPTERS`, que
  junta as duas regras por adaptador e só encolhe.
- **Testes corrigidos:** três testes usavam `gpt-4o` para provar o envio de `reasoning_effort`
  (o gpt-4o não raciocina, segundo a página do modelo): passam ao `gpt-5.4-mini`; o do corpo do
  batch afirmava `max_tokens` (obsoleto e recusado pelos o-series): passa a
  `max_completion_tokens`; os dois do Astra chamavam o `_build_sdk_kwargs` apagado: passam pelo
  `prepare` e esperam `RequestError`; o do construtor do cliente verifica também o `http_client`
  com o hook.
- **Gate R00:** 3998 passed, 26 deselected; ruff, formato, pyright (0/0/0) e lock limpos.
- **Saldo:** `_openai.py` 890 (início) → 690 (passo 2) → 741 (+51: construtores tipados, perfis e
  URLs); `_base.py` 393 → 457 (+64: o despacho e o transporte, comuns aos cinco). Dívida de
  complexidade 137 → 134.
- **Ao vivo (só preparado):** `tests/integration/test_provider_hardening_live.py` —
  `uv run pytest tests/integration/test_provider_hardening_live.py -m live_api -k openai -q`
  (gpt-5-nano, 1024 tokens; quatro testes, uns cêntimos).

### Passo 3 · xAI · nota de desenho

- **Documentação (2026-09-18).** Esforços por modelo, nas páginas de cada um
  (https://docs.x.ai/developers/models/grok-4.6, `…/grok-4.5`, `…/grok-4.3`,
  `…/grok-4.20-0309-reasoning`, `…/grok-4.20-0309-non-reasoning`, `…/grok-build-0.1`): o `grok-4.6`
  e o `grok-4.5` aceitam low, medium, high (omissão) e xhigh, que o `grok-4.5` serve como high
  (https://docs.x.ai/developers/model-capabilities/text/reasoning), e o raciocínio não se desliga;
  o `grok-4.3` aceita none, low (omissão), medium, high e xhigh; o `grok-4.20-reasoning` e o
  `grok-build-0.1` raciocinam sem esforço documentado; o `grok-4.20-non-reasoning` não raciocina.
  Os modelos de raciocínio recusam `presence_penalty`, `frequency_penalty` e `stop` com erro (mesma
  página). O multi-agente escolhe 4 agentes com low/medium e 16 com high/xhigh (`agent_count` no
  SDK) e não aceita `max_tokens` nem tools do cliente
  (https://docs.x.ai/developers/model-capabilities/text/multi-agent). Os ids retirados a
  2026-05-15 passam para o `grok-4.3`: os de raciocínio com esforço low, os outros com none, e o
  `grok-code-fast-1` para o `grok-build-0.1` (https://docs.x.ai/developers/migration/may-15-retirement).
  Os resultados de tools voltam um por chamada, pela ordem de `response.tool_calls`
  (https://docs.x.ai/docs/guides/function-calling). As violações das regras de uso são cobradas
  (https://docs.x.ai/developers/pricing): só o `RESOURCE_EXHAUSTED` é `unbilled` (D20). Estados HTTP
  pelo `google.rpc.Code` (https://github.com/googleapis/googleapis/blob/master/google/rpc/code.proto):
  `FAILED_PRECONDITION` é 400 (o adaptador dizia 412) e `UNIMPLEMENTED` 501 (dizia 500).
- **SDK.** No `xai-sdk` 1.19.0, `chat.create(...)` não faz RPC: constrói o `GetCompletionsRequest`
  (valida tipos com `TypeError` e valores com `ValueError`); forçar uma tool é
  `chat.required_tool(name)`; o SDK tem server tools em `xai_sdk.tools`, cujo suporte é do C05;
  `AsyncClient(api_host=, use_insecure_channel=True)` permite um servidor gRPC em loopback. O
  acumulador do stream (`Response.process_chunk`) copia o usage de cada chunk, por isso um stream
  sem usage fica com um usage vazio mas "presente": o usage só conta se tiver algum campo. O
  `xai-sdk` 1.7, o mínimo declarado, não tem `medium` nem `xhigh`, nem `agent_count`, nem
  `cost_usd` (o adaptador já usava `agent_count` e lia o custo por `getattr`); o `xhigh` chegou na
  1.18.0 (https://github.com/xai-org/xai-sdk-python/blob/main/CHANGELOG.md).
- **Correcções à primeira versão desta nota:** o `grok-4.5` não precisa de perfil próprio (aceita
  `xhigh`); o `grok-4.3` precisa (aceita `none`); o `chat.create` passa para o `prepare`.
- **Perfis:** `_CURRENT` (grok-4.6, grok-4.5 e qualquer modelo novo: esforço low–xhigh, raciocina),
  `_GROK_4_3` (none–xhigh; também os ids de raciocínio retirados), `_AUTO` (raciocina sem esforço:
  `grok-4.20-reasoning`, `grok-build-0.1` e os seus ids), `_PLAIN` (`grok-4.20-non-reasoning` e os
  ids sem raciocínio retirados), `_MULTI_AGENT`. Opções: `thinking_effort` aplica-se sozinho (os
  modelos raciocinam sem pedir, como a Muse Spark, D13) e é validado contra o perfil;
  `thinking=True` não pede nada a mais, salvo num modelo que não raciocina, onde levanta
  `RequestError`; `thinking_effort` num modelo sem esforço levanta `RequestError`;
  `thinking_budget` só avisa. Nos que raciocinam, `stop` e as penalidades levantam `RequestError`.
  No multi-agente, tools levantam `RequestError`, `max_tokens` fica de fora (o `LLM` põe sempre um)
  e o esforço escolhe o `agent_count`.
- **Pedido:** o `prepare` constrói os argumentos num `TypedDict` e chama o `chat.create` do cliente
  (local, sem RPC); `Prepared.params` é o `Chat` do SDK, cujo `proto` é o pedido do fio. Um erro de
  construção do SDK sai do `prepare` como `RequestError`, antes de qualquer operação. O pyright
  verifica o `TypedDict` contra a assinatura do `create` (um acesso tipado ao cliente). O
  `tool_choice` com nome passa a `required_tool`; server tools levantam `RequestError`.
- **Erros:** `RESOURCE_EXHAUSTED` → `RateLimitError` (429); `UNAVAILABLE` → `TransportError` e
  `DEADLINE_EXCEEDED` → `ProviderTimeout`, sem estado HTTP inventado; os outros códigos → `APIError`
  com o estado do `google.rpc.Code`; tudo `indeterminate` salvo o 429. Uma ligação recusada também
  chega como `UNAVAILABLE`: o gRPC não diz se o pedido saiu, por isso fica `indeterminate`. O
  despacho marca-se ao entrar no `send`/`open_stream` (todo o trabalho local ficou no `prepare`).
- **Mínimo do SDK:** o extra `xai` passa a `xai-sdk>=1.18,<2` (D15: o mínimo é o que o adaptador
  usa; sem ramos por versão). Não posso instalar a 1.18.0 aqui: fica para o job `floors` do CI.
- **Transporte:** servidor gRPC falso em 127.0.0.1 (`tests/integration/fakegrpc.py`) com estados,
  erro a meio do stream, servidor mudo e sucesso; mais ligação recusada e o servidor TCP do
  `fakeserver` (fecho sem resposta, RST). A paridade e o contrato de conversa usam o SDK real: o
  mesmo servidor para `sample`/`stream`, e o `proto` do pedido para o fio.

### Passo 3 · xAI · resultado

- **Vermelho observado:** 96 dos 136 testes novos de `tests/test_xai_provider.py` falhavam antes do
  código. O `prepare` antigo devolvia um dict, e uma sonda que o passava ao `chat.create` do SDK real
  mostrou oito defeitos. O `tool_choice` com nome rebentava no SDK dentro do `send` (`ValueError:
  Protocol message ToolChoice has no "type" field`, já com a operação aberta). As server tools
  levantavam `NotImplementedError` sem tipo. O `xhigh` do `grok-4.6` (como qualquer esforço) caía
  com um aviso. O `stop` seguia para um modelo de raciocínio, que o recusa. `thinking=True` num
  modelo sem raciocínio caía com aviso. As tools seguiam para o multi-agente, que não as aceita.
  `temperature="hot"` dava `TypeError` cru no `send`. E um stream sem usage dava
  `usage_reported=True` com custo 0,0, um custo conhecido e falso. No mapeador, `UNAVAILABLE` e
  `DEADLINE_EXCEEDED` saíam como `APIError` 503 e 504, `FAILED_PRECONDITION` como 412 e
  `UNIMPLEMENTED` como 500, e uma excepção que não fosse gRPC saía crua.
- **Feito:** o `prepare` constrói os argumentos num `TypedDict` (`_Create`, verificado pelo pyright
  contra a assinatura de `chat.create`) e chama o `chat.create` do cliente, que é local. O
  `Prepared.params` é o `Chat` do SDK, e um erro de construção do SDK sai do `prepare` como
  `RequestError`. Tabela de perfis pela gramática (`_CURRENT`, `_GROK_4_3`, `_AUTO`, `_PLAIN`,
  `_MULTI_AGENT`, com os ids retirados). `parse_options` substitui o bloco de `pop`. O `tool_choice`
  com nome usa `required_tool`, as server tools levantam `RequestError`, e o multi-agente recusa
  tools. O custo reportado vem do `Response.cost_usd` do SDK (sai o `getattr`). O usage só conta se
  tiver campos. O mapeador segue o `google.rpc.Code`, com despacho marcado ao entrar no
  `send`/`open_stream`. O `import grpc` passa para dentro da guarda do SDK.
- **Comum, feito aqui:** a R00 não admite `# noqa` novos, e o passo OpenAI e o passo 2 tinham
  deixado sete (`# noqa: E402` nos imports tipados do OpenAI, Anthropic e Meta). O `require_sdk`
  passa a gestor de contexto (`with require_sdk("openai"): import openai …`), e os imports dos
  SDKs ficam dentro do bloco, sem E402. Saem os 18 `# noqa: E402` dos cinco adaptadores e da
  moderação: `src/` tem 4 `# noqa` (14 no início da fase). O erro de import passa a levar a causa
  (`from missing`).
- **Mínimo do SDK:** `xai-sdk>=1.18,<2` (era 1.7, sem `medium`, `xhigh`, `agent_count` nem
  `cost_usd`). `uv lock` feito sem rede. O README está actualizado.
- **Testes novos:** `tests/test_xai_provider.py` reescrito com o SDK real (53 → 136). O pedido lê-se
  do `proto` do SDK, as respostas vêm do tipo do SDK ou do servidor gRPC em loopback, e há os
  perfis, os 16 códigos gRPC e as chamadas. A paridade xAI tem 2 testes, a conversa xAI 6 e o
  transporte xAI 11 (429, 500, `UNAVAILABLE`, servidor mudo, erro no stream, corte a meio, sucesso,
  fecho sem resposta, RST, ligação recusada, recusa do SDK sem pedido). `tests/integration/fakegrpc.py`
  é o servidor gRPC falso, com estados, espera, corte e registo dos pedidos. Os testes ao vivo do
  xAI estão em `test_provider_hardening_live.py`.
- **Testes corrigidos ou retirados:**
  - `TestGrpcCodeToHttp` (7) sai com o `_grpc_code_to_http`, substituído pela tabela de
    `TestErrors`. O `test_failed_precondition` afirmava 412, e o `google.rpc.Code` diz 400.
  - `test_is_multi_agent_model` sai com o helper; a resolução do multi-agente, snapshot incluído,
    fica em `TestProfiles`.
  - `test_input_output_tokens_fields` sai com o `getattr` que testava: o `SamplingUsage` do SDK
    não tem esses campos.
  - `test_reasoning_effort_ignored` e `test_thinking_true_is_ignored` afirmavam que o
    `grok-4.20-reasoning` ignora o esforço com aviso. Agora o esforço levanta `RequestError` e
    `thinking=True` passa sem nada no fio. É uma mudança de desenho, pelos esforços documentados.
  - Os testes de stream e de erro com `MagicMock` (que escondia o `chat.create` real, como dizia
    o achado) passam ao servidor gRPC em loopback.
  - `test_finish_reason` afirmava o valor falso `FINISH_REASON_STOP`; o SDK dá `REASON_STOP`.
  - `TestRequireSdk` segue a guarda nova.
- **Gate R00:** 4100 passed, 30 deselected; ruff, formato, pyright (0/0/0) e lock limpos. O caso
  "corte a meio" falhou uma vez na suite inteira, porque o servidor falso parava-se de dentro da
  chamada. Passou a parar-se numa tarefa própria, e cinco execuções seguidas deram verde.
- **Saldo:** `_xai.py` 540 (início) → 460 (passo 2) → 492 (+32: pedido tipado, perfis e URLs,
  tabela de estados). `_imports.py` 12 → 20. Blocos `except` de SDK no xAI: 0. Dívida de
  complexidade 134 → 129 (sai o `_build_create_kwargs`, 17/17/52, e o `_messages_to_sdk`, 14/15).
- **Ao vivo (só preparado):**
  `uv run pytest tests/integration/test_provider_hardening_live.py -m live_api -k xai -q` (grok-4.3,
  1024 tokens, quatro testes, uns cêntimos). O `temperature=5.0` é o pedido que a API deve recusar
  com um 4xx; se a API o aceitar, o teste falha nesse ponto e o `oversized` precisa de outro valor.
  O custo exacto continua em `tests/integration/test_xai_cost.py`.
- **Não verificável aqui:** o `xai-sdk` 1.18.0 (o novo mínimo) não está instalado nem em cache,
  por isso fica para o job `floors` do CI. O `grok-4.5` com `xhigh` servido como `high` e a recusa de
  `stop` por cada modelo são documentação, não prova ao vivo.

### Passo 3 · Gemini · nota de desenho

- **Documentação (2026-09-18).** Thinking no `generateContent`
  (https://ai.google.dev/gemini-api/docs/generate-content/thinking). O 3.8 e o 3.7 Flash aceitam
  `thinkingLevel` low, medium (omissão) e high; o 3.6 e o 3.5 Flash, o 3.5 e o 3.1 Flash-Lite e o 3
  Flash aceitam também minimal; o 3.1 Pro aceita low, medium e high (omissão) e não desliga o
  thinking; o 3 Pro aceita low e high (tabela de https://ai.google.dev/gemini-api/docs/thinking).
  No 2.5 usa-se `thinkingBudget`: Pro de 128 a 32768 sem se desligar, Flash de 0 a 24576, Flash-Lite
  0 ou de 512 a 24576, e -1 é dinâmico. O `thinkingBudget` é aceite no Gemini 3 só por
  compatibilidade. `includeThoughts` pede resumos. As assinaturas das function calls são
  obrigatórias: devolve-se o turno inteiro, com todas as partes. Function calling
  (https://ai.google.dev/gemini-api/docs/generate-content/function-calling): os resultados de
  chamadas paralelas vão num só `Content` `user`; o Gemini 3 dá sempre um `id` a cada
  `functionCall` e pede o mesmo `id` na `functionResponse`. Facturação
  (https://ai.google.dev/gemini-api/docs/billing): "If your request fails with a 400 or 500 error,
  you won't be charged for the tokens used."
- **SDK `google-genai` 2.24.0.** `HttpOptions.httpx_async_client` desliga o `aiohttp`
  (`_use_aiohttp()`). O caminho `aiohttp` reenvia o pedido depois de um erro de ligação
  (`_api_client.py`: `except (aiohttp.ClientConnectorError, …)`, espera 1 a 10 s e repete), fora
  do `retry_options`. O SDK não fecha um cliente que recebeu, e o timeout de cada pedido continua
  a vir do `HttpOptions.timeout`. Um erro HTTP sai como `ClientError` (4xx) ou `ServerError` (5xx),
  com `.code` e com o `httpx.Response` em `.response`. Um objecto de erro dentro de um stream 200
  levanta as mesmas classes, mas com o invólucro interno do SDK em `.response`. Um corpo que não é
  JSON dá `UnknownApiResponseError(ValueError)`. `FunctionCall.id` e `FunctionResponse.id` existem;
  `ThinkingLevel` tem MINIMAL, LOW, MEDIUM e HIGH.
- **Perfis:** `_CURRENT` (por omissão: 3.8/3.7 Flash, 3.1 Pro e modelos novos, com níveis low,
  medium e high), `_MINIMAL` (acrescenta minimal), `_GEMINI_3_PRO` (low e high), e os orçamentos
  do 2.5 (`_PRO_2_5`, `_FLASH_2_5`, `_FLASH_LITE_2_5`) com os intervalos documentados.
  `thinking_effort` aplica-se sozinho, porque estes modelos pensam sem pedir (D13/D25), e é
  validado contra o perfil (sai o aviso do SDK sobre `xhigh` e o pedido que seguia); no 2.5 vira
  orçamento (low 2048, medium 5000, high 10000, como hoje). `thinking=True` pede resumos
  (`include_thoughts`) e, sem esforço, fica como hoje: nível high no 3.x, orçamento 10000 no 2.5.
  `thinking_budget` fora do intervalo do modelo levanta `RequestError`; no Gemini 3 só avisa e
  fica de fora.
- **Pedido tipado:** `Prepared` com um `TypedDict` (`model`, `contents: list[types.Content]`,
  `config: types.GenerateContentConfig`), e a config construída com argumentos nomeados. Um
  `ValidationError` do pydantic (um `ValueError`) nos parâmetros do chamador sai do `prepare` como
  `RequestError`. O `generate_content(**params)` corre por um acesso tipado ao `AsyncModels`, para
  o pyright verificar o fio.
- **Conversa:** os resultados de um turno vão juntos num só `Content` `user`, pela ordem das
  chamadas, cada um com o `id` da chamada quando o turno do modelo o trouxe (Gemini 3). O turno do
  modelo é reenviado do `_raw` (todas as partes e assinaturas) quando é uma resposta Gemini. Um
  papel que não seja `user`, `assistant` ou `system` levanta `RequestError`.
- **Pilha HTTP:** o adaptador passa o seu `httpx.AsyncClient` (`follow_redirects=True`, como o do
  SDK, com o hook `request` do despacho) e fecha-o no `close()`. Acaba o reenvio escondido.
- **Server tools:** `web_search` → `google_search`, `code_execution` → `code_execution`. Uma
  config (chaves além de `type`) ou outro tipo levanta `RequestError`: a config é do C05, e hoje
  ambas se perdiam em silêncio.
- **Erros:** 429 → `RateLimitError` (`unbilled`, D20); um HTTP 400 ou 500 → `APIError`
  `unbilled` (página de facturação); os outros estados → `APIError` `indeterminate`. Um erro
  dentro de um stream 200 → `APIError` com o código do próprio objecto de erro, `indeterminate`,
  porque pode já ter havido tokens. Transporte `httpx` pela causa (a fase de ligação é `not_sent`).
  O resto pelo helper do despacho: um corpo ilegível é `ResponseError`, e uma recusa local do SDK
  antes do envio (`contents are required.`) é `RequestError`.
- **Testes:** perfis e fio no `prepare`; conversa (resultados juntos, `id`, assinaturas, turno
  depois de stream); paridade com os tipos do SDK; transporte em loopback com o SDK real (429, 400,
  500, 503, fecho sem resposta, RST, servidor mudo, 200 ilegível, erro dentro do stream, corte a
  meio, sucesso, ligação recusada, recusa local); e um só pedido no servidor quando a ligação
  cai, a prova de que o reenvio escondido acabou.

### Passo 3 · Gemini · resultado

- **Vermelho observado:** 59 dos 110 testes novos de `tests/test_gemini_provider.py` falhavam
  antes do código: os resultados de um turno saíam em `Content`s separados e sem `id`; um papel
  desconhecido seguia; uma server tool com config, ou de tipo desconhecido, perdia-se em silêncio;
  `thinking_effort` sem `thinking=True` era ignorado e um esforço inválido seguia; um
  `ValidationError` do pydantic saía cru; 400 e 500 ficavam `indeterminate`; uma falha de ligação
  não dizia `not_sent`; um corpo ilegível (`UnknownApiResponseError`) saía cru. Na sonda com o SDK
  real em loopback, uma ligação que caía sem resposta dava **dois** pedidos ao servidor e 9 s de
  espera (o reenvio do caminho `aiohttp`), contra um pedido com o `httpx`. A conversa com duas
  chamadas dava `[user, model, user, user]`.
- **Feito:** o `prepare` constrói a `GenerateContentConfig` com argumentos nomeados, num
  `TypedDict` (`_Generate`) que o pyright verifica contra o `generate_content` por um acesso tipado
  ao `AsyncModels`. Um `ValidationError` do SDK sai como `RequestError`. Tabela de perfis pela
  gramática: níveis por modelo no Gemini 3 e orçamentos com intervalo no 2.5. `thinking_effort`
  aplica-se sozinho e é validado. Os resultados de um turno vão num só `Content`, com o `id` da
  chamada quando o Gemini o deu. Um papel desconhecido, uma config de server tool e um tipo de
  server tool desconhecido levantam `RequestError`. A pilha fica fixa em `httpx` por um transporte
  próprio em `async_client_args`: o SDK constrói o cliente com o hook do despacho e fecha-o, e o
  `close()` fecha também o cliente assíncrono. O mapeador tem entrega documentada (400 e 500
  `unbilled`, 429 `unbilled`, erro dentro do stream `indeterminate`). Sai o `aiohttp` do adaptador.
- **Testes novos:** `tests/test_gemini_provider.py` reescrito com os tipos do SDK (52 → 110:
  mensagens, pedido, perfis, uso, montagem, chamadas, erros). A paridade Gemini tem 2 testes e a
  conversa Gemini 6 (os quatro históricos canónicos, o turno Gemini com `id` e assinaturas, e o
  texto antes das chamadas). O transporte Gemini tem 15, com o SDK real e HTTP em loopback: 429,
  400, 500, 503, fecho sem resposta, RST, servidor mudo, 200 ilegível, erro dentro do stream, corte
  a meio e sucesso, mais um pedido só quando a ligação cai, ligação recusada `not_sent`, recusa
  local do SDK sem pedido e o fio do sucesso. Os testes ao vivo do Gemini estão em
  `test_provider_hardening_live.py`.
- **Testes corrigidos ou retirados:**
  - `TestBuildThinkingConfig` (5) sai com o `_build_thinking_config` e passa a `TestThinking`,
    pelo `prepare`. Os casos antigos continuam lá; o esforço vale agora também sem
    `thinking=True`.
  - `test_provider_timeout_is_converted_to_sdk_milliseconds` segue as `HttpOptions` tipadas, com
    o transporte e o hook.
  - Os testes de `complete` com `MagicMock`/`SimpleNamespace` passam a `TestRequest` (o fio no
    `prepare`) e a `TestCalls` (tipos do SDK). O `test_thinking_forwarded` usava o
    `gemini-2.0-flash`, que o Gemini não documenta com thinking; passa ao `gemini-2.5-pro`.
  - `TestGeminiProviderNetworkErrors` perde o caso `aiohttp` com a pilha; os casos `httpx`
    passam ao mapeador, com `not_sent` na ligação.
- **Gate R00:** 4178 passed, 34 deselected; ruff, formato, pyright (0/0/0) e lock limpos. O
  transporte correu três vezes seguidas verde.
- **Saldo:** `_gemini.py` 655 (início) → 600 (passo 2) → 597. Blocos `except` de SDK fora do
  mapeador: 0. Dívida de complexidade 129 → 124 (saem o `_build_config`, 15/15, e o
  `_messages_to_sdk`, 15/13/51).
- **Ao vivo (só preparado):**
  `uv run pytest tests/integration/test_provider_hardening_live.py -m live_api -k gemini -q`
  (gemini-3.1-flash-lite, 1024 tokens, quatro testes, cêntimos). As chamadas paralelas com
  reenvio em `complete` e em stream são a verificação que a nota "Known issue" de
  `docs/model-compatibility.md` e do `AGENTS.md` espera: só sai depois de o dono as ver verdes.
- **Não verificável aqui:** que a API aceita as `functionResponse` com `id` no Gemini 3 e sem `id`
  no 2.5, e que um `thinking_budget` fora do intervalo seria mesmo recusado (a regra vem da
  documentação).

### Passo 3 · Meta · nota de desenho

- **Documentação (2026-09-18).** Erros da Meta Model API (https://dev.meta.ai/docs/error-handling):
  a tabela dá, a cada código, o estado HTTP (`invalid_api_key` 401, `billing_not_configured` 402,
  `model_not_found`/`file_not_found` 404, `payload_too_large` 413, `rate_limit_exceeded` 429,
  `server_shutting_down`/`service_overloaded`/`backend_unavailable` 503, `gateway_timeout` 504; o
  500 e o 400 vêm com código nulo). As falhas a meio do stream chegam como `response.failed` ou
  `error`, com `code` e `message`. "Failed requests may still incur charges depending on where
  processing occurred": salvo o 429 (D20), tudo `indeterminate`. No `openai` 3.14.1 a Responses API
  tem o esforço `max`, o `phase` e o `file_url`; o SDK levanta `APIError` para um payload
  `{"error": …}` dentro do stream e entrega o evento `error` como evento tipado.
- **Pedido tipado:** o `prepare` devolve um `ResponseCreateParamsNonStreaming` com os `TypedDict`
  do SDK (itens de entrada, partes, tools, `reasoning`, `text`), e o pyright verifica o fio. Os
  sítios onde o fio sai dos tipos do SDK ficam numa lista no topo do adaptador, cada um com a
  prova ao vivo do M01 (`blackboard/tasks/M01-meta-provider.md`); o que o pyright apontar e o M01
  não tiver provado fica nos tipos do SDK.
- **Erros:** um só mapeador (saem o `_api_error` e o `_sdk_error`, e com eles a Meta sai da lista
  de tolerância). HTTP 429 → `RateLimitError`; outro estado → `APIError`; `APIConnectionError` pela
  causa `httpx2` (a fase de ligação é `not_sent`), como no OpenAI, com o hook do despacho no
  cliente. Uma falha reportada dentro de uma resposta (`response.failed`, evento `error`, payload
  de erro) leva o estado que a tabela da Meta dá ao seu código; um código fora da tabela, ou nulo
  (o 400 e o 500 partilham o nulo), sai como `ResponseError`, sem estado inventado. Sai o "código
  desconhecido → 400", que era inventado. O `server_error` (código de erro da Responses API) fica
  500, o estado que a Meta dá ao tipo `server_error`.
- **Usage de um `response.failed`:** o erro passa a levar o usage que a resposta reportou
  (`ProviderError.usage`). A pipeline regista-o na tentativa (`Response.attempts`) e o meter
  liquida a falha com esse usage e o custo do pricer (`MeterOperation.fail(..., usage=, cost=)`),
  em vez de um custo desconhecido. É a única mudança no núcleo de metering: um argumento opcional.
- **Perfil:** a Muse Spark é uma só família (D13); não há tabela de perfis.
- **Server tools:** só `web_search` chega à Meta (confirmado no M01). Um `code_execution`, que hoje
  cai com aviso, e uma config levantam `RequestError`.
- **Testes:** fio tipado e desvios; mapeador (estados, códigos dentro da resposta, transporte);
  usage de uma falha até ao meter; paridade e conversa; transporte em loopback com o SDK real
  (429, 500, 503, fecho sem resposta, RST, servidor mudo, 200 ilegível, `response.failed`, evento
  `error`, corte a meio, sucesso, ligação recusada, recusa local).

### Passo 3 · Meta · resultado

- **Vermelho observado:** os três testes do usage de uma falha
  (`tests/metering/test_failure_disposition.py`) falharam antes do código: nem o erro nem o
  `fail` do meter aceitavam usage. **Desvio à R00, assumido:** escrevi o adaptador Meta antes dos
  seus testes. Para ver o vermelho, reconstruí a versão do passo 2 num ficheiro do scratchpad (sem
  git), troquei-a de lugar, corri os testes e repus o ficheiro novo, confirmado com `diff`. Os
  testes novos falharam pelas razões esperadas:
  - `logprobs=True` largado em silêncio; `max` aceite no 1.2 e num contributor, e um esforço
    inventado seguia;
  - o papel `tool` sem `tool_use_id` seguia;
  - `code_execution` só avisava, e a config do `web_search` perdia-se;
  - um `response.failed` saía sem usage;
  - o código nulo virava `APIError` 500, e o `invalid_prompt` um 400 inventado;
  - uma ligação recusada ficava `indeterminate`;
  - o `timeout` fazia `import httpx`, que o extra `meta` não instala (o `openai` 3.x só depende do
    `httpx2`), e o cliente não tinha o hook do despacho.

  A reconstrução usa os helpers do HEAD, e por isso partiu também testes sem relação com isto.
- **Feito:** pedido tipado pelo SDK (`ResponseCreateParamsNonStreaming` e os `TypedDict` dos
  itens, partes, tools, `reasoning` e `text`); o pyright verifica o fio. Os três desvios aos tipos
  do SDK ficam listados no topo do adaptador, cada um com a prova ao vivo do M01: a mensagem de
  assistente reconstruída sem `id`, `status` nem `annotations`; a function tool sem `strict`; a
  imagem sem `detail`. Cada um é construído como dict e convertido uma só vez. Outros pontos:
  - um só mapeador (saem o `_api_error`, o `_sdk_error` e o `_STREAM_ERROR_STATUS`);
  - códigos de falha pela tabela documentada da Meta; um código nulo ou fora da tabela sai como
    `ResponseError`;
  - `ProviderError.usage` e `MeterOperation.fail(..., usage=, cost=)`: uma falha com usage fica
    liquidada com esse usage e o custo do pricer, e a tentativa guarda-o em `Response.attempts`;
  - esforços validados por modelo (`max` só no 1.3 standard);
  - `logprobs=True`, papéis desconhecidos, `code_execution` e configs de server tools levantam
    `RequestError`;
  - o cliente leva o hook do despacho e um `timeout` simples, e o `openai.APIConnectionError` é
    classificado pela causa `httpx2`.
- **Testes novos:** 3 de metering; `tests/test_meta_provider.py` (56 → 67: esforços por modelo,
  papel desconhecido, server tools, usage de uma falha em `complete` e em stream, ligação `not_sent`,
  cliente sem `httpx`); paridade Meta (2); conversa Meta (5); transporte Meta (16 com o SDK real em
  loopback: 429, 500, 503, fecho sem resposta, RST, servidor mudo, 200 ilegível, `response.failed`,
  evento `error` sem código, payload de erro, corte a meio, sucesso, usage da falha, ligação
  recusada, recusa local do SDK, fio do sucesso). Ao vivo: `test_provider_hardening_live.py`.
- **Testes corrigidos:** `test_logprobs_are_not_sent` → `test_logprobs_are_refused` (a Meta
  responde 400: https://dev.meta.ai/docs/reasoning). `test_unsupported_server_tool_warns` →
  `test_a_server_tool_meta_does_not_run_is_refused`. Em `test_stream_errors`, o código nulo
  afirmava um 500 e o `invalid_prompt` um 400, ambos inventados: passam a `ResponseError`, e os
  `response.failed` passam a respostas reais do SDK. `test_a_deterministic_stream_failure_is_not_retried`
  espera `ResponseError` (continua sem retry). `test_a_failed_response_raises` verifica também o usage.
- **Gate R00:** 4215 passed, 38 deselected; ruff, formato, pyright (0/0/0) e lock limpos.
- **Saldo:** `_meta.py` 757 (início) → 650 (passo 2) → 673 (+23: tipos e a lista de desvios);
  metering +32 (`_store.py` +24, `_operation.py` +8); `_exceptions.py` +14; `_attempts.py` +13.
  Dívida de complexidade 124 → 123 (sai o `_build_request`, 13).
- **Ao vivo (só preparado):**
  `uv run pytest tests/integration/test_provider_hardening_live.py -m live_api -k meta -q`
  (muse-spark-1.3, 1024 tokens, quatro testes, cêntimos). A suite do M01 continua em
  `uv run pytest tests/integration/test_meta_live.py -m live_api -q`.
- **Não verificável aqui:** que a Meta devolve usage num `response.failed` real; que o
  `max_tokens=10_000_000` dá um 4xx (se a Meta o aceitar, o teste ao vivo diz-o e o `oversized`
  muda).

### Passo 3 · Anthropic · nota de desenho e resultado

- **Documentação (2026-09-18).** Thinking por modelo
  (https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting):
  - Fable 5.1, Mythos 5.1, Fable 5 e Mythos 5: só adaptativo, sempre ligado.
  - Opus 5 e Sonnet 5: adaptativo e ligado por omissão (o `disabled` do Opus 5 só até `high`).
  - Opus 4.8 e 4.7: só adaptativo, desligado por omissão.
  - Opus 4.6 e Sonnet 4.6: adaptativo, e ainda o `enabled` obsoleto.
  - Opus 4.5, Haiku 4.5, Sonnet 4.5 e os Claude 4 anteriores: só `budget_tokens`, e recusam o
    `adaptive`.

  O `display` é `"omitted"` por omissão nos modelos novos
  (https://platform.claude.com/docs/en/build-with-claude/thinking). Esforços
  (https://platform.claude.com/docs/en/build-with-claude/effort): `xhigh` e `max` na família 5,
  no Opus 4.8 e no 4.7; os 4.6 têm `max` sem `xhigh`; o Opus 4.5 só com orçamento. O `tool_choice`
  `any`/`tool` dá 400 no Fable 5.1 e no Mythos 5.1 (https://platform.claude.com/docs/en/api/errors),
  página que dá também a tabela tipo → estado; um erro a meio do stream chega depois do 200.
  Resultados paralelos numa só mensagem `user`, antes de qualquer texto
  (https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use). Server tools
  com `name`: `web_search_20250305` e `code_execution_20250825`, versões que todos os modelos
  actuais aceitam sem header beta (o `code_execution_20250522` do adaptador é legado). Facturação:
  "Failed requests aren't charged", mas um corte ou timeout do cliente a meio é cobrado
  (https://support.claude.com/en/articles/8977456-how-do-i-pay-for-my-claude-api-usage).
- **SDK `anthropic` 1.6.0:** o `MessageParam` aceita os blocos da resposta (`ContentBlock`) tal
  como vieram, e é assim que o turno do assistente é reenviado a partir do `_raw` enquanto
  coincidir com a mensagem (assinaturas de thinking e resultados de server tools incluídos).
  `messages.stream` não tem `stream` (o `Params` assenta no `MessageCreateParamsBase`) e aceita
  `extra_body`, por onde vão `temperature`, `top_p` e `top_k`. Um erro dentro do stream chega como
  `APIStatusError` com o 200 do stream.
- **Perfis:**
  - `_CURRENT` (família 5, Opus 4.8 e 4.7, modelos novos): adaptativo, esforço low–max, sem
    amostragem.
  - Fable 5.1 e Mythos 5.1 recusam `tool_choice` forçado; o Mythos Preview não tem `xhigh`.
  - Os 4.6 não têm `xhigh` e aceitam amostragem.
  - `_EXTENDED` (lista fechada das famílias antigas): orçamento. Um esforço ou um orçamento liga o
    thinking sozinho, com mínimo de 1024, e `max_tokens` passa a orçamento + `max_tokens` (D20).

  Regras comuns:
  - `thinking=True` nos adaptativos → `{"type": "adaptive", "display": "summarized"}`;
    `thinking=False` não envia nada (a ficha).
  - `thinking_effort` → `output_config.effort`, sozinho e fundido com o `format`.
  - `thinking_budget` num adaptativo só avisa.
  - Sem amostragem, a `temperature` sai em silêncio (o `LLM` envia-a sempre) e `top_p`/`top_k`
    levantam `RequestError`.
  - `max_tokens` é obrigatório (`RequestError`; o `LLM` envia-o sempre).
- **Vermelho observado (antes do código):** 58 testes novos em `tests/test_anthropic_provider.py`,
  4 da conversa (resultados em mensagens separadas; thinking sem reenvio) e 9 de transporte com o
  SDK real em loopback. Nestes, 400, 500 e 529 saíam `indeterminate`; o 200 ilegível e o JSON de
  tool inválido saíam como `ValueError` cru; o corte a meio saía como erro `httpx2` cru; a ligação
  recusada não dizia `not_sent`; e a recusa local do SDK saía crua. O evento de erro dentro do
  stream virava `APIError` com estado 200.
- **Feito:**
  - pedido tipado (`Params` sobre o `MessageCreateParamsBase`, com `extra_body`), com os perfis
    acima;
  - todos os `tool_result` de um turno numa só mensagem, e o turno reenviado do `_raw`;
  - papéis desconhecidos recusados; server tools com `name` (config e tipo desconhecido recusados);
  - um mapeador: 429 `unbilled`, outros estados `unbilled` (facturação documentada), erro dentro do
    stream pelo tipo documentado (`indeterminate`, ou `ResponseError` se o tipo não estiver na
    tabela), transporte pela causa `httpx2`, e o resto pelo helper do despacho;
  - hook do despacho no cliente, e despacho marcado ao entrar no stream (chegou uma resposta);
  - o batch pelo `prepare`, com a amostragem no corpo, e os resultados do batch tipados;
  - thinking com texto vazio (`display: "omitted"`) já não entra em `Response.thinking`;
  - sai o `_TEMPERATURE_DEPRECATED_PREFIXES` (a tabela de perfis cobre-o) e, na base, sai o
    `network_error`; a lista `_PENDING_ADAPTERS` ficou vazia e foi apagada (as duas regras de
    arquitectura exigem agora zero infractores).
- **Testes novos:** `TestThinkingByModel`, `TestSamplingByModel`, `TestToolChoiceAndServerTools`,
  `TestErrorsR02` (78 → 146 no ficheiro); conversa Anthropic (6); paridade Anthropic (1, guarda);
  transporte Anthropic (15). Ao vivo: `test_provider_hardening_live.py`.
- **Testes corrigidos ou retirados:**
  - `TestBuildThinkingParam` (6) sai com o helper, e os casos passam a `TestThinkingByModel`, num
    modelo de orçamento.
  - `test_thinking_forwarded` esperava `enabled` no `claude-sonnet-4-6`; a documentação dá
    adaptativo.
  - `test_does_not_inject_max_tokens` afirmava que o pedido seguia sem `max_tokens`, que a API
    exige: passa a `test_max_tokens_is_required`.
  - `test_missing_cache_fields` testava uma forma que o `Usage` do SDK nunca tem (sai com o
    `getattr`).
  - `test_malformed_tool_args_fail_the_stream` espera `ResponseError` `indeterminate`.
  - Os duplos `SimpleNamespace` ganham o `id` que a `Message` do SDK tem sempre, e as chamadas
    directas ao adaptador passam `max_tokens`.
  - Da R01, `test_local_sdk_error_delivery_and_bounded_meter` esperava `indeterminate` num 503 da
    Anthropic; a política documentada dá `unbilled` (o do OpenAI continua `indeterminate`).
  - `test_network_mapper_raises_normalized_error_without_losing_builtin_handler` passa ao
    `transport_error`, e há um caso novo para `not_sent`.
- **Gate R00:** 4305 passed, 42 deselected; ruff, formato, pyright (0/0/0) e lock limpos.
- **Saldo:** `_anthropic.py` 910 (início) → 622 (passo 2) → 777 (+155: pedido tipado, tabela de
  perfis com URLs, reenvio do turno, tabela de erros, batch tipado). `_base.py` 457 → 452 (sai o
  `network_error`). Os cinco adaptadores: 3752 no início da fase → 3280 (−472); o pacote
  `core/_providers/` inteiro: 4301 → 4018 (−283; os "4294" da ficha eram o pacote, não os cinco
  adaptadores). Dívida de complexidade 123 → 121 (sai o `_build_sdk_kwargs`, 17/18).
- **Ao vivo (só preparado; sem créditos Anthropic):**
  `uv run pytest tests/integration/test_provider_hardening_live.py -m live_api -k anthropic -q`
  (claude-haiku-4-5, 1024 tokens mais um orçamento de 2048 no teste de thinking, quatro testes).
- **Não verificável aqui:** as regras de thinking e esforço por modelo, a recusa do `tool_choice`
  forçado no Fable 5.1, e que a API aceita o turno reenviado com as assinaturas, tudo contra a API
  real.

### Passo 4 · Nota de desenho (rede do fio)

- **Onde.** Só em `tests/`: o validador em `tests/wire_contract.py` e uma fixture `autouse` em
  `tests/conftest.py` que envolve o `prepare` dos cinco adaptadores. Cada pedido que um teste
  constrói é validado no momento em que sai do `prepare`, e uma violação faz falhar o teste. O
  `prepare` é o único construtor: o `send`, o stream e os corpos de batch saem dele, e o
  `count_tokens` usa os mesmos construtores de mensagens e tools.
- **Por SDK** (o protótipo `wire-contract/`, com o que mudou desde então):
  - `openai` e `anthropic` (Stainless): o `TypedDict` do pedido compilado num validador estrito.
    Chaves a mais são proibidas, os iteráveis validam-se já, um modelo do SDK só passa como
    instância, e uma união com discriminador literal (`type`, `role`) valida-se pelo membro certo,
    com um só erro. OpenAI: `CompletionCreateParamsNonStreaming`. Anthropic:
    `MessageCreateParamsNonStreaming` sem o `extra_body`, que é uma opção de pedido; o `extra_body`
    só pode levar `temperature`, `top_p` e `top_k`, com os tipos da documentação (o `anthropic`
    1.6.0 tirou-os do `TypedDict`). Meta: `ResponseCreateParamsNonStreaming`, com os três desvios
    que o adaptador lista preenchidos antes da validação (`id` e `status` no turno reconstruído e
    `annotations` no seu texto, `strict` na function tool, `detail` na imagem), para o resto ser
    validado sem ruído. Um teste prova que o adaptador ainda produz cada desvio; quando deixar de
    o produzir, o preenchimento sai.
  - `google-genai`: o pedido volta a passar pela validação do pydantic (`model_validate` do
    `model_dump`, que apanha o que foi atribuído sem validação) e pelo conversor offline do SDK
    para a Gemini Developer API (`_GenerateContentParameters_to_mldev`), que recusa os campos que
    só o Vertex aceita. O aviso "is not a valid" dos enums tolerantes do SDK conta como erro.
  - `xai-sdk`: o pedido tem de ser um `Chat` do SDK. O protobuf aceita e serializa um valor de
    enum que não existe (provado no `xai-sdk` 1.19.0: `reasoning_effort = 99` serializa), por isso
    a rede percorre o `proto` e recusa enums fora da definição.
- **Canários** (`tests/test_wire_contract.py`): por SDK, pedidos maus conhecidos dão a violação
  certa (chave a mais, tipo errado, membro de união errado, papel inválido, campo só do Vertex,
  enum inválido no Gemini e no xAI, amostragem desconhecida no `extra_body`), e pedidos bons
  construídos pelo `prepare` de cada adaptador passam. Outro canário prova que a fixture está
  ligada (o pedido de um `prepare` fica registado e validado) e que uma violação faz falhar.
- **Excepção declarada:** um teste que envia de propósito um pedido fora do contrato (a recusa
  local do SDK nos testes de transporte) declara-o com
  `@pytest.mark.wire_contract(tolerate=[regex, ...])`, linha a linha. Não há outra forma de
  desligar a rede.
- **Vermelho esperado:** os canários não importam sem o módulo; com a fixture ligada, a suite
  mostra onde os pedidos de hoje saem do contrato, e cada caso é corrigido no adaptador (com o
  teste que o prova) ou é um teste que o faz de propósito e o declara.

### Passo 4 · Resultado

- **Vermelho observado:** os canários (`tests/test_wire_contract.py`) não importavam sem o
  módulo. Com a fixture ligada, a suite inteira validou 490 pedidos preparados em 402 testes
  (Anthropic 147, OpenAI 98, xAI 87, Gemini 81, Meta 77) e falhou em três, e só em três: os
  testes de transporte que enviam de propósito um pedido que o SDK recusa (um `set` no `stop` do
  OpenAI e no `prompt_cache_key` da Meta, `contents` vazio no Gemini). Os pedidos que os cinco
  adaptadores constroem já cumpriam o contrato dos SDKs depois do passo 3; a rede fica como
  guarda.
- **Feito:** `tests/wire_contract.py` (o validador, a partir do protótipo) e a fixture `autouse`
  `wire_log` em `tests/conftest.py`, que envolve o `prepare` dos cinco adaptadores e faz falhar o
  teste com as violações. Os três testes propositados declaram-nas com
  `@pytest.mark.wire_contract(tolerate=[...])`. Nada em `src/`.
- **Diferenças para o protótipo:** o protótipo validava as chamadas aos clientes simulados e os
  construtores privados; com a costura B, o `prepare` é o único construtor e é ele que se valida.
  Os escalares passam a estritos (o SDK envia-os tal como estão: `"yes"` não é booleano no fio, e
  o modo tolerante do pydantic aceitava-o). Os desvios da Meta deixam de ser tolerados por
  expressões regulares sobre as mensagens de erro (que exigiam limpar o ruído das uniões) e passam
  a ser preenchidos antes da validação, só onde faltam. O `extra_body` da Anthropic é validado
  contra os campos de amostragem documentados. O Gemini volta a passar pela validação do pydantic
  antes do conversor. O xAI, que agora constrói o pedido no próprio SDK, é verificado nos enums do
  `proto`.
- **Canários (36):** OpenAI 9 (pedido completo, servidor compatível, chave a mais, tipo errado,
  esforço inválido, papel inválido, `tool` sem `tool_call_id`, function tool plana, server tool
  `web_search`); Anthropic 9 (três pedidos completos por família; `temperature` no topo, que é o
  erro do b3dae3f; `extra_body` desconhecido ou mal tipado; `adaptive` com `budget_tokens`; server
  tool sem `name`; `tool_result` sem `tool_use_id`); Meta 7 (pedido completo; o adaptador ainda
  faz cada desvio; chave a mais, esforço inválido, item sem `output`, `strict` e `detail` com
  valores errados, que o preenchimento não esconde); Gemini 5 (dois pedidos completos, campo só do
  Vertex, valor atribuído depois da construção, enum inválido); xAI 3 (pedido completo, enum fora
  da definição no pedido e numa mensagem, pedido que não é um `Chat`); fixture 3 (cada `prepare`
  fica verificado, violação declarada ou não, e todos os adaptadores de `core/_providers` estão
  debaixo da rede).
- **Um canário corrigido antes do código:** escrevi um `function_call_output` sem `call_id` como
  pedido mau, mas o `openai` 3.14.1 declara o `call_id` opcional; o validador estava certo e o
  canário passou a um item sem `output`, que é obrigatório.
- **Gate R00:** 4341 passed, 42 deselected; ruff, formato, pyright (0/0/0) e lock limpos.
- **Saldo:** `tests/` +706 (`wire_contract.py` 338, `test_wire_contract.py` 327, `conftest.py`
  +38, três marcadores no transporte +3). `src/` 0. Sem entrada no CHANGELOG: não é visível para
  quem usa a biblioteca.

### Passo 5 · Resultado (só preparado)

`tests/integration/test_provider_hardening_live.py`, 20 testes `live_api` (quatro por fornecedor,
escritos com cada adaptador no passo 3): chamadas paralelas com reenvio em `complete` e em stream,
thinking ligado e desligado, e um 4xx sob `max_cost` seguido de uma chamada admitida. Modelo mais
barato de cada um e `max_tokens` 1024. Recolhem sem erros (`--collect-only`); não correram, porque
o agente não chama fornecedores. Os comandos estão no relatório final.

### Passo 6 · Resultado

- **Docs:** `docs/llm.md` (erros com a entrega de cada fornecedor e a recusa antes da admissão;
  thinking por fornecedor numa tabela; `grok-4.3` no encaminhamento), `docs/pricing.md` (preço
  por id exacto, snapshots, `aliases`, `match="prefix"`; o modelo sem preço debaixo de um meter;
  custo `None` sem usage; a falha com usage), `docs/model-compatibility.md` (as regras actuais de
  cada fornecedor por cima de cada tabela; o xAI reescrito; o thinking do Gemini e da Anthropic;
  os limites da Meta), `docs/safety.md` (a tabela de falhas com a entrega por fornecedor e o
  `not_sent` da ligação; o `unpriced` já não fala de modelos sem preço). Fora da lista da ficha,
  porque afirmavam o que a R02 mudou: `docs/tools.md` (quem corre server tools, e a config
  recusada), `docs/getting-started.md` e `docs/framework-overview.md` (`grok-2`, retirado),
  `README.md` (o quadro de funcionalidades dizia server tools no OpenAI e imagens no xAI),
  `CONTRIBUTING.md` ("Adding a provider" descrevia o contrato antigo) e
  `examples/25_server_tools.py` (usava `web_search()` com o `gpt-4.1-nano`, que agora levanta
  `RequestError`; passa ao `claude-haiku-4-5`).
- **`AGENTS.md`:** o contrato do provider e as tabelas de perfil na arquitectura; os testes de
  provider pelo `tests/provider_calls.py`, os tipos do SDK, os servidores em loopback e a rede do
  fio; o `FakeProvider` nos testes de metering; os gotchas da Anthropic (thinking adaptativo,
  `max_tokens` obrigatório, amostragem no `extra_body`), do OpenAI (limite por host, esforço só
  com `thinking`), do Gemini, do xAI e da Meta. A deriva do structured output já estava fechada
  pela F25 (item 8).
- **Nota "Known issue" do Gemini:** fica, como a ficha manda, até o dono ver verdes as chamadas
  paralelas ao vivo; reescrita para dizer que está corrigida no código e à espera dessa prova
  (dizia que os resultados iam separados e sem `id`, o que deixou de ser verdade).
- **CHANGELOG:** acertadas duas entradas que a R02 contradizia dentro do próprio `[Unreleased]`:
  a do passo 2 sobre o JSON de tool inválido na Anthropic (`ValueError`; agora `ResponseError`,
  junta à entrada da Anthropic) e o mínimo `xai-sdk>=1.7` da entrada dos SDKs (uma só entrada com
  `>=1.18`). A entrada da Anthropic diz agora que a config de uma server tool, antes largada, levanta
  `RequestError`; a das recusas do adaptador deixa de falar de `ValueError` por tipar.
- **Achado novo (`FINDINGS.md`):** no OpenAI, `thinking_effort` sem `thinking=True` perde-se em
  silêncio, ao contrário dos outros quatro adaptadores; reproduzido, fica para decisão do dono.
- **Gate R00:** 4341 passed, 42 deselected; ruff, formato, pyright (0/0/0) e lock limpos.

## Relatório final · R02 concluída

### Feito por passo

- **1 · Costura D (ids e preços):** `core/_model_id.py` com a gramática de snapshots e o
  `lookup`; preços por id exacto ou snapshot, com `aliases` e `match="prefix"` explícito; o
  encaminhamento e o tokenizer pela gramática; `UnpricedModelError` antes de abrir a operação
  (D16/D20); regra de arquitectura sem `startswith` sobre ids fora de `_model_id.py`;
  `scripts/audit_models.py` para o dono.
- **2 · Costura B (contrato):** `BaseProvider[P, F]` com `prepare`/`send`/`open_stream`/
  `assemble`/`usage`/`map_error` e o algoritmo na base; `prepare` antes de qualquer admissão;
  paridade `complete`/stream por construção; custo desconhecido sem usage; `FakeProvider` em vez
  dos duplos duck-typed.
- **3 · Os cinco adaptadores (OpenAI, xAI, Gemini, Meta, Anthropic, por esta ordem):** pedido
  tipado pelo SDK, tabela de perfis por modelo com URLs, um mapeador com a entrega documentada e o
  despacho (`not_sent` na ligação), paridade, contrato de conversa e transporte com o SDK real em
  loopback. Comum: marcador de despacho (D24), `require_sdk` como gestor de contexto (saem 18
  `# noqa`), usage das falhas até ao meter (D26).
- **4 · Rede do fio:** `tests/wire_contract.py` e a fixture `wire_log`; 490 pedidos da suite
  validados, 36 canários; só os três testes que enviam de propósito um pedido mau o declaram.
- **5 · Verificação ao vivo:** 20 testes preparados; comandos abaixo.
- **6 · Documentação e registo:** docs, `AGENTS.md`, `CHANGELOG.md` e blackboard.

### O que não ficou e porquê

- **Nada ao vivo.** O agente não chama fornecedores (R00); não há créditos Anthropic nem xAI.
- **A nota "Known issue" do Gemini** fica até o dono confirmar ao vivo (ficha, passo 3).
- **O `xai-sdk` 1.18.0**, o novo mínimo, não está instalado nem em cache: prova-o o job `floors`
  do CI.
- **O tecto por step sem `BudgetPolicy`** (`FINDINGS.md`, 2026-09-18, "decisão do dono (R02)")
  não estava na ficha e espera a decisão do dono: não foi tocado.
- **Encontrado e deixado de fora (em `FINDINGS.md`, com reprodução):** os resultados de batch à
  tarifa normal; o `LLM` do xAI que não se constrói depois de um `asyncio.run`; o xAI que larga
  imagens que os Grok aceitam; o `thinking_effort` do OpenAI sem `thinking`. As server tools no
  OpenAI e no xAI e as configs de todas continuam para o C05.

### Números

| Gate R00 | Passed | Deselected | Ruff / formato / pyright / lock |
|---|---:|---:|---|
| Início (`b3dae3f`) | 3886 | 22 | limpos |
| 1 · Costura D | 3954 | 22 | limpos |
| 2 · Costura B | 3963 | 22 | limpos |
| 3 · OpenAI | 3998 | 26 | limpos |
| 3 · xAI | 4100 | 30 | limpos |
| 3 · Gemini | 4178 | 34 | limpos |
| 3 · Meta | 4215 | 38 | limpos |
| 3 · Anthropic | 4305 | 42 | limpos |
| 4 · Rede do fio | 4341 | 42 | limpos |
| 6 · Documentação | 4341 | 42 | limpos |

Os 20 deselected a mais são os testes ao vivo novos. Zero `xfail`. Comandos, sem rede:

```bash
uv run pytest -m "not live_api" -q --timeout=120
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv run pyright src
uv lock --check --offline
```

- **Complexidade** (`tests/quality_baseline.json`, a dívida que só desce): 161 → 121 entradas
  (−40: passo 2 −24, OpenAI −3, xAI −5, Gemini −5, Meta −1, Anthropic −2). Nenhuma função nova
  acima de complexidade 10 nem de 60 linhas.
- **Supressões:** `# noqa` em `src/` 14 → 4; `# type: ignore` em `src/` 17 → 15 e em `tests/`
  155 → 122. Nenhuma nova. Nenhum `Any` novo em assinaturas públicas.
- **Blocos `except` de excepções de SDK fora dos mapeadores:** 44 → 0 (regra de arquitectura).
- **Saldo de linhas em `src/`** (antes → depois, contra `b3dae3f`):

| Módulo (`src/ai_arch_toolkit/`) | Antes | Depois | Saldo |
|---|---:|---:|---:|
| `__init__.py` | 429 | 431 | +2 |
| `core/__init__.py` | 271 | 273 | +2 |
| `core/_attempts.py` | 526 | 586 | +60 |
| `core/_default_pricing.toml` | 997 | 871 | −126 |
| `core/_exceptions.py` | 73 | 85 | +12 |
| `core/_metering/_operation.py` | 100 | 106 | +6 |
| `core/_metering/_store.py` | 533 | 551 | +18 |
| `core/_model_id.py` (novo) | 0 | 69 | +69 |
| `core/_pricing.py` | 339 | 334 | −5 |
| `core/_providers/__init__.py` | 268 | 266 | −2 |
| `core/_providers/_anthropic.py` | 910 | 777 | −133 |
| `core/_providers/_base.py` | 269 | 452 | +183 |
| `core/_providers/_gemini.py` | 655 | 597 | −58 |
| `core/_providers/_imports.py` | 12 | 20 | +8 |
| `core/_providers/_meta.py` | 757 | 673 | −84 |
| `core/_providers/_openai.py` | 890 | 741 | −149 |
| `core/_providers/_xai.py` | 540 | 492 | −48 |
| `core/_tokens.py` | 104 | 100 | −4 |
| `toolkit/moderation/_openai.py` | 77 | 77 | 0 |

  Python −123, TOML −126, total −249. O pacote `core/_providers/`: 4301 → 4018 (−283); os cinco
  adaptadores 3752 → 3280 (−472). Fora de `src/`: `scripts/audit_models.py` +155; `tests/`
  +2651 (14 ficheiros novos com 3389 linhas, os outros −738).

### Mudanças visíveis para quem depende do toolkit

Todas no `[Unreleased]` do `CHANGELOG.md`. As que partem código:

- **Breaking:** o preço casa por id exacto ou snapshot; uma variante já não herda o preço do
  prefixo (`o3-pro`, `gpt-4o-audio-preview`…), e `register()` é exacto (o primeiro parâmetro
  passa a `model`).
- **Breaking:** um modelo sem preço não corre debaixo de um meter (todo o `Flow` e `Agent` abre
  um): `UnpricedModelError`; um modelo local regista zero.
- **Breaking:** o extra `xai` pede `xai-sdk>=1.18`.
- Pedidos que o modelo não aceita levantam `RequestError` antes de enviar, em vez de seguirem
  (ou caírem com aviso): thinking num modelo sem raciocínio, esforços fora do perfil, server
  tools no OpenAI e no xAI, configs de server tools, `tool_choice` forçado no Fable 5.1 e no
  Mythos 5.1, `top_p`/`top_k` onde não há amostragem, `logprobs` na Meta, papéis desconhecidos.
- `delivery` segue a facturação documentada (Anthropic e os 400/500 do Gemini `unbilled`) e a
  ligação falhada é `not_sent`; nenhuma excepção de SDK sai crua.
- O thinking segue cada modelo: adaptativo na Anthropic 4.6+, esforço aplicado sozinho no xAI,
  Gemini, Meta e Anthropic, orçamento somado ao `max_tokens` nos Claude antigos.
- Os resultados de um turno de tools vão juntos (Gemini com `id`, Anthropic numa só mensagem), e
  os turnos são reenviados com as assinaturas.
- Um stream acaba com a mesma `Response` que o `complete()`; uma resposta sem usage tem custo
  `None`; o usage de uma falha reportada chega ao meter.

### O que não pôde ser verificado sem chamadas ao vivo

Comandos para o dono, com as chaves no ambiente (`set -a && source .env && set +a`); cada linha
custa cêntimos (modelo mais barato, 1024 tokens):

```bash
uv run pytest tests/integration/test_provider_hardening_live.py -m live_api -k openai -q
uv run pytest tests/integration/test_provider_hardening_live.py -m live_api -k xai -q
uv run pytest tests/integration/test_provider_hardening_live.py -m live_api -k gemini -q
uv run pytest tests/integration/test_provider_hardening_live.py -m live_api -k meta -q
uv run pytest tests/integration/test_provider_hardening_live.py -m live_api -k anthropic -q
uv run pytest tests/integration/test_meta_live.py -m live_api -q
uv run pytest tests/integration/test_xai_cost.py -m live_api -q
```

O que cada um prova e o que fica por provar está nas secções "Ao vivo" e "Não verificável aqui"
de cada adaptador. Em resumo: as chamadas paralelas com reenvio (a do Gemini decide a nota "Known
issue"); o thinking e os esforços por modelo; que o `temperature=5.0` no xAI e o
`max_tokens=10_000_000` na Meta dão mesmo um 4xx (se não derem, o teste falha nesse ponto e o
valor muda); que a Meta devolve usage num `response.failed`; que a API aceita os turnos
reenviados com assinaturas. O `xai-sdk` 1.18.0 prova-se no job `floors` do CI.

### Desvios ao plano

- **Meta:** o adaptador foi escrito antes dos testes; o vermelho foi visto contra uma
  reconstrução do passo 2 (secção do Meta).
- **xAI:** o `prepare` chama o `chat.create` do SDK (local, sem RPC) e o `Prepared.params` é o
  `Chat`, em vez de um dict; é o pedido do fio, e um erro de construção sai no `prepare`.
- **Import dos SDKs:** o `require_sdk` passou a gestor de contexto para cumprir a R00 (sem
  `# noqa` novos), o que tocou os cinco adaptadores e a moderação.
- **Docs fora da lista da ficha:** `tools.md`, `getting-started.md`, `framework-overview.md`,
  `README.md`, `CONTRIBUTING.md` e o exemplo 25, porque afirmavam o que a R02 mudou.
- **Contagem da ficha:** os "4294 linhas nos cinco adaptadores" eram o pacote
  `core/_providers/` inteiro (4301 no início); os cinco adaptadores tinham 3752.

### Divisão proposta em commits

1. `refactor(providers): three-phase contract, exact-id pricing, per-model rules` — todo o
   `src/`, `scripts/audit_models.py`, `pyproject.toml`, `uv.lock` e `tests/`, salvo a rede do
   fio. Os passos 1 a 3 partilham `_attempts.py`, `_exceptions.py` e uma dúzia de ficheiros de
   teste: separá-los pedia `git add -p` e árvores intermédias que ninguém correu, por isso vão
   juntos. Os três marcadores `wire_contract` de `tests/integration/test_provider_transport.py`
   (ficheiro novo) vão aqui; sem a fixture, são só marcas desconhecidas (aviso, não erro).
2. `test: check every prepared request against its SDK's contract` — `tests/wire_contract.py`,
   `tests/test_wire_contract.py`, `tests/conftest.py`.
3. `docs: provider rules, pricing, failed-call accounting and the R02 record` — `docs/`,
   `README.md`, `AGENTS.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `examples/25_server_tools.py` e o
   blackboard.

**Feitos a pedido do dono (2026-09-18), sem push:** `cf2aa43` (1), `a8c3eea` (2) e o commit do
registo (3). Os três marcadores `wire_contract` foram com o commit 2, junto da fixture que os lê: o
commit 1 leva o ficheiro de transporte sem eles. Cada árvore foi verificada sozinha, exportando o
índice com `git checkout-index` e correndo a suite com o `PYTHONPATH` na exportação: commit 1, 4305
passed; commit 2, 4341 passed; ruff e formato limpos nas duas. O `src/`, o `pyproject.toml` e o
`uv.lock` das duas são iguais aos da árvore final, que passou no pyright e no `uv lock --check`. O
commit 3 não muda código nem testes.
