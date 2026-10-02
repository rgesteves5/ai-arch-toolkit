# O02 · Núcleo Responses partilhado, extraído do `_meta.py`

- **Dono:** Claude (agente, 2026-10-02) · **Estado:** done · **Depende de:** nada (pode correr
  em paralelo com a O01)
- **Origem:** D43 · **Decisões:** D11 a D14 (Meta), D43 · **Regras:** `R00-rules.md`

## Problema

- Tudo o que é da Responses API vive no `_meta.py` (674 linhas): itens de entrada, function tools,
  `text.format`, montagem da `Response`, eventos de stream, reenvio do `_raw`, usage, falhas
  dentro da resposta e do stream, `input_tokens.count`. A O03 precisa do mesmo para o OpenAI;
  copiá-lo daria duas casas ao mesmo assunto.
- O reenvio só verifica o tipo do `_raw` (`_replayable_output`: `isinstance(raw, SDKResponse)`).
  Hoje só a Meta devolve esse tipo. Com a O03 o OpenAI também o devolve, e um `fallback=` entre os
  dois mandaria a um fornecedor o raciocínio cifrado do outro. A OpenAI também não reaproveita
  raciocínio de outra família de modelos ("Persisted reasoning can be reused only within the same
  model family", https://developers.openai.com/api/docs/guides/reasoning).

## Objectivo

- `core/_providers/_responses.py` com o que é da Responses API. O `_meta.py` fica só com o que é
  da Meta: host, chave, códigos de erro, esforços, `tool_choice` só `auto`, `strict: false`, a
  `web_search` alojada.
- O comportamento da Meta não muda: os testes dela passam sem alteração, salvo os novos do
  reenvio.

## Desenho (confirma-o na nota de desenho, antes do código)

- O núcleo recebe um perfil, uma dataclass congelada com o que varia entre fornecedores: as regras
  de esforço, o `tool_choice`, o `strict`, as tools alojadas aceites, o mapa de códigos de erro e
  a família do modelo.
- **Reenvio:** um `_raw` só se reenvia quando o fornecedor e a família do modelo coincidem com os
  do pedido. A família é uma regra do perfil, resolvida por `core/_model_id.py` (nada de
  `startswith` sobre ids). Fora disso a mensagem reconstrói-se dos campos, sem raciocínio, como
  hoje quando o texto não coincide.
- Sem caminhos duplos nem shims: o `_meta.py` usa o núcleo e perde o código que se mudou.
- Confirma as formas no SDK instalado (`openai` 3.19.2) e regista o que confirmaste.

## Ficheiros

- `src/ai_arch_toolkit/core/_providers/_responses.py` (novo), `src/ai_arch_toolkit/core/_providers/_meta.py`
- `src/ai_arch_toolkit/core/_model_id.py` (só se a família precisar de uma regra nova)
- `tests/test_meta_provider.py` (só testes novos), `tests/test_responses_core.py` (novo),
  `tests/test_architecture.py` (se a casa única precisar de teste)

## Provas

- Os testes da Meta passam sem alteração.
- Teste novo, que falha antes: um `_raw` de outro fornecedor, ou de outra família de modelos, não
  é reenviado; a mensagem reconstrói-se sem raciocínio.
- Saldo de linhas registado: o par `_meta.py` + `_responses.py` contra o `_meta.py` de hoje. A
  descida a sério vem na O03, quando o OpenAI usar o núcleo.

## Registo do dono

- Estado: review (Claude, 2026-10-02). Nada commitado, por ordem do dono.

### Nota de desenho (escrita antes do código)

- **Casa.** `core/_providers/_responses.py` fica com tudo o que é da Responses API: itens de
  entrada e reenvio, function tools, tools alojadas, `tool_choice`, `text.format`, montagem da
  `Response`, usage, eventos de stream, falhas dentro da resposta e do stream,
  `input_tokens.count` e o mapa dos erros do SDK.
- **Perfil.** `ResponsesProfile`, dataclass `frozen=True, slots=True, kw_only=True`, só com dados:
  - `provider: str`: o nome nas mensagens de erro ("Meta").
  - `families: Mapping[str, str]`: prefixo de id → família, resolvido por `_model_id.family`.
    O método `family(model)` dá a família do perfil, senão o próprio id sem sufixo de snapshot
    (`snapshot_base`).
  - `include: tuple[ResponseIncludable, ...]`: a Meta pede `reasoning.encrypted_content`; o
    OpenAI fica com `()`, porque com `store: false` o conteúdo cifrado vem por omissão (O01, 2).
  - `takes_tool_choice: bool`: com `False` (Meta) só há `auto`, `none` vai sem tools e forçar
    levanta; com `True` (OpenAI) `auto`, `none`, `required` e um nome vão em `tool_choice`.
  - `function_strict: bool | None`: `None` omite a chave (Meta, desvio 2); `False` manda
    `strict: false` (OpenAI, O01, 5).
  - `hosted_tools: Mapping[str, ToolParam]`: o tipo da server tool do toolkit → a tool da
    Responses que se envia. A Meta só tem `web_search`. Uma config continua recusada (C05).
  - `codes: Mapping[str, int]`: o estado HTTP de cada código de erro de uma falha dentro da
    resposta ou do stream.
- **Esforços: ficam fora do perfil (desvio ao desenho da ficha).** São regras do modelo, e
  vêm agarradas a outras que não são dados. A Meta raciocina sempre e recusa logprobs. O OpenAI
  só manda o esforço com `thinking`, tem modelos que não raciocinam, retira o sampling a
  raciocinar e precisa de `include` para os logprobs (O01, 7). Cada adaptador resolve-as pela
  sua tabela, com `lookup`, e põe o `reasoning` no seu `prepare`.
- **Divisão.**
  - `ResponsesProvider(LoopAwareClientCache, BaseProvider[Prepared[Params], SDKResponse])` é
    abstracta e recebe `(model, profile, client)`. Faz o I/O e o que não depende do pedido:
    `send`, `open_stream`, `usage`, `map_error`, `close` e `_count_input_tokens`.
  - O adaptador compõe `prepare`, `assemble` e `count_tokens` com as funções do núcleo
    (`input_items`, `request_params`, `_parse_sdk_response`). A conversão das mensagens leva a
    família do pedido, que é do adaptador, e o OpenAI ainda vai juntar os logprobs à montagem
    (O01, 7).
  - O `_input_items` e o `_parse_sdk_response` continuam a importar-se do `_meta.py`, sem shim:
    o primeiro é a ligação da Meta (perfil e família), usada pelo `prepare` e pelo
    `count_tokens`; o segundo é o parser do núcleo, com o nome de sempre, usado pelo `assemble`.
- **Reenvio (mudança 1).**
  - `input_items(messages, profile, family)`. Um `_raw` reenvia-se só quando é uma resposta da
    Responses API, `profile.family(raw.model) == family` e o texto e as calls ainda coincidem.
    Fora disso, a mensagem reconstrói-se dos campos, sem raciocínio, como hoje.
  - O fornecedor reconhece-se pelo `model` do `_raw`, lido na tabela de famílias do perfil do
    pedido. Os prefixos de um fornecedor são só dele (`muse-spark-`, `gpt-…`), e um id que
    nenhuma família do perfil reclama é a sua própria família. Por isso a resposta de outro
    fornecedor nunca cai na família do pedido.
  - A Meta serve uma família, a Muse Spark (D14: só `muse-spark-` vai para a Meta). O
    `_input_items` da Meta liga a família `muse-spark` sem olhar para o modelo do pedido. Um id
    forçado com `provider="meta"` fica sem reenvio, e a Meta só serve Muse Spark na mesma.
  - O OpenAI (O03) vai passar `profile.family(self._model)`.
- **Itens com o nome do fio (mudança 2).** O reenvio faz `model_dump(..., by_alias=True)`: o
  `async_` do SDK sai como `async`, a chave do `ResponseFunctionToolCallParam`.
- **O que sai do `_meta.py`:** a conversão de conteúdos e mensagens, o reenvio, as function e as
  hosted tools, o `text.format`, usage, `stop_reason`, thinking, citações, o parser, as falhas, o
  `_TextJoiner`, `send`, `open_stream`, `usage`, `map_error`, `close`, `_NOT_SENT`, `_TIMEOUTS`
  e a lista dos três desvios. A lista passa para o topo do `_responses.py`, onde estão os casts.
- **O que fica no `_meta.py`:** host, cliente sem os cabeçalhos da OpenAI, `_FORWARDED`,
  esforços, `_CODE_STATUS`, `_PROFILE`, `_input_items`, `prepare` (com `_reasoning`),
  `assemble`, `count_tokens`.
- **Provas.**
  - Os testes de `tests/test_meta_provider.py` correm sem alteração.
  - Novos testes na Meta: uma volta do OpenAI (`gpt-6-luna`) não é reenviada à Meta; uma volta
    da `muse-spark-1.2` é reenviada à 1.3 (mesma família); uma function call com `async`
    reenvia-se com `async`. O primeiro e o terceiro falham antes.
  - `tests/test_responses_core.py`: famílias, guarda entre famílias e ids sem família, e o
    lado OpenAI do perfil (`include` vazio, `strict: false`, `tool_choice`, tools alojadas,
    códigos).
- **Ficheiros fora da lista:**
  - `tests/test_wire_contract.py`: `_adapters()` passa a ignorar classes abstractas, porque o
    `ResponsesProvider` não é um adaptador.
  - `tests/wire_contract.py` e `AGENTS.md`: o ponteiro para a lista de desvios, que passa para
    o `_responses.py`.

### Formas confirmadas no SDK instalado (`openai` 3.19.2, com `uv run python`)

- `ResponseFunctionToolCall.async_` tem o alias `async`, e o mesmo acontece em `FunctionTool`,
  `CustomTool`, `ToolFunction` e `ResponseCustomToolCall`. Sem `by_alias`, o `model_dump` dá
  `async_`; a chave do `ResponseFunctionToolCallParam` é `async`. O construtor do SDK
  (`model_construct`) aceita `async` nos itens aninhados.
- `Response.model` é `str` (uma união com os literais dos modelos da OpenAI).
- `ResponseIncludable` tem `reasoning.encrypted_content` e `message.output_text.logprobs`.
- O `ToolChoice` da `responses.create` aceita `"none" | "auto" | "required"` e
  `ToolChoiceFunctionParam` (`{"type": "function", "name": ...}`).
- Os códigos de `ResponseError.code` são `server_error`, `rate_limit_exceeded`,
  `invalid_prompt`, `vector_store_timeout`, os `invalid_image*`/`image_*`, entre outros. Ficam
  para o perfil do OpenAI na O03.
- `InputTokenCountParams` tem `model`, `input`, `instructions`, `tools` e `tool_choice`, entre
  outros.

### Feito

- **Núcleo.** `core/_providers/_responses.py` (novo) tem `ResponsesProfile`,
  `ResponsesProvider`, `input_items`, `request_params` e `_parse_sdk_response`, com a lista dos
  três desvios no topo.
- **Meta.** O `_meta.py` ficou com o host e o cliente, `_FORWARDED`, os esforços (`_EFFORTS`, e
  `_MODEL_EFFORTS`, que era `_PROFILES` e mudou de nome para não se confundir com `_PROFILE`),
  `_CODE_STATUS`, `_PROFILE`, `_input_items`, `prepare` com `_reasoning`, `assemble` e
  `count_tokens`.
- **Mudança 1, a guarda do reenvio.** Um `_raw` de outro fornecedor, ou de outra família, já não
  se reenvia: a mensagem reconstrói-se sem raciocínio.
- **Mudança 2.** Os itens reenviados saem com os nomes do fio (`by_alias=True`).
- **Casa única.** Teste novo de arquitectura: só o `_responses.py` chega a
  `client.responses…`. Falha sobre o `_meta.py` antigo, na linha 522, e guarda a O03 contra uma
  segunda cópia no `_openai.py`.

### Ficheiros tocados

- `src/ai_arch_toolkit/core/_providers/_responses.py` (novo)
- `src/ai_arch_toolkit/core/_providers/_meta.py`
- `tests/test_meta_provider.py`: 48 linhas acrescentadas e nenhuma retirada; os testes que já
  existiam não mudaram.
- `tests/test_responses_core.py` (novo)
- `tests/test_architecture.py`
- Fora da lista da ficha, os três pela razão dada na nota de desenho:
  - `tests/test_wire_contract.py`: `_adapters()` ignora as classes abstractas. Sem isto, o
    censo dos adaptadores falha, porque conta o `ResponsesProvider`.
  - `tests/wire_contract.py`: o ponteiro da docstring de `_meta`.
  - `AGENTS.md`: a frase dos três desvios aponta agora para o `_responses.py`.

### Testes novos (24)

- `tests/test_meta_provider.py::TestReplayGuard`, 3 testes:
  - `test_a_turn_from_another_provider_is_rebuilt_without_its_reasoning`: uma volta da
    `gpt-6-luna` não é reenviada à Meta. Falhava antes: reenviava `reasoning`.
  - `test_a_turn_from_another_muse_spark_model_is_replayed`: a família Muse Spark mantém o
    reenvio entre modelos (fixa o comportamento de hoje).
  - `test_replayed_items_carry_the_wire_names`: a call reenviada leva `async` e não `async_`.
    Falhava antes, com `KeyError`, e a rede do contrato do fio denunciava
    `function_call.async_ | Extra inputs are not permitted`.
- `tests/test_responses_core.py`, 19 testes, sobre um perfil "Acme" com ids inventados, à
  maneira do OpenAI:
  - famílias e ids sem família, que contam sem o sufixo de snapshot;
  - a guarda em cinco casos: a mesma família, outra família, um id sem família datado, outro
    id, outro fornecedor;
  - uma volta reconstruída passa no `prepare`;
  - sem `include`, `strict: false` nas function tools;
  - `tool_choice` `auto`, `none`, `required` e um nome, sem `tool_choice` quando não há tools;
  - tools alojadas copiadas em profundidade, e uma server tool que o fornecedor não corre é
    recusada;
  - falhas tipadas pelos códigos do perfil.

  Os pedidos validam-se contra o `ResponseCreateParamsNonStreaming`, com a rede do contrato.
- `tests/test_architecture.py`, 2 testes: `test_the_responses_api_is_reached_only_by_its_core`
  e o canário do detector.

### Números

- **Linhas:**
  - `_meta.py`: 674 antes, 207 depois.
  - `_responses.py`: 662, novo.
  - O par soma 869 contra 674, mais 195. Vem dos dois blocos de imports, do perfil documentado,
    da composição do pedido (`request_params`, `_tools`, `_sends_tools`, `_tool_choice`) e do
    construtor da base. A descida chega na O03, quando o OpenAI não precisar de outra cópia de
    cerca de 450 linhas.
- **Complexidade:**
  - O máximo é 9 antes e depois (`open_stream`, igual).
  - O `prepare` da Meta desceu de 6 para 2; o `request_params` tem 5.
  - A função mais longa tem 43 linhas (`_replayable_output`).
  - A linha de base da qualidade não mudou.
- **Verificações:**
  - `uv run ruff format src tests examples`: nada a mudar (474 ficheiros).
  - `uv run ruff check src tests examples`: limpo.
  - `uv run pyright src`: 0 erros.
  - `uv run pytest -q`: 5795 passed, 42 skipped. A linha de base era 5771 passed e 42 skipped;
    são mais 24, os testes novos.
  - `uv lock --check`: limpo.
  - Os testes de integração herméticos da Meta (SDK real contra o servidor falso em
    `127.0.0.1`) passam: 16.

### Linhas propostas para o `CHANGELOG` (`[Unreleased]`)

- **Fixed:** "Meta: a replayed turn keeps the API's field names, so a function call that carries
  `async` is no longer sent with the SDK's `async_`."
- **Changed** (pode esperar pela O03, que é quando se torna alcançável: hoje só a Meta devolve
  uma resposta da Responses API): "A turn's encrypted reasoning is replayed only to the provider
  and the model family that produced it; any other assistant turn is rebuilt without it."
- A extracção do núcleo não é visível para quem usa o toolkit.

### Desvios ao plano

1. **Os esforços ficam fora do perfil**, no `_reasoning` de cada adaptador. A razão está na nota
   de desenho: são regras do modelo, agarradas a código.
2. **`prepare`, `assemble` e `count_tokens` ficam no adaptador, não na base.**
   - A conversão leva a família do pedido.
   - Os testes da Meta importam `_input_items` e `_parse_sdk_response` do `_meta.py` e tinham de
     passar sem alteração.
   - O OpenAI vai juntar logprobs à montagem.

   Para isso, `_parse_sdk_response` manteve o nome privado no núcleo e o `_meta.py` importa-o
   para o seu `assemble`. As funções novas do núcleo têm nomes públicos (`input_items`,
   `request_params`), como as do `_base.py`.
3. **A família de um pedido à Meta é a constante `muse-spark`**, não a do modelo do pedido. Um
   id que não é Muse Spark, forçado com `provider="meta"`, nunca reenvia. A Meta só serve Muse
   Spark, por isso esse pedido falharia na mesma.
4. **A mensagem de `tool_choice` forçado** passou de "Meta Model API only supports…" a "Meta
   only supports…", com o nome do perfil. O teste procura "only supports tool_choice='auto'".
   A ordem dos erros no `prepare` manteve-se: conversão, raciocínio, tools.
5. **O `text.format` vai com `strict: false` em todos os perfis**, como hoje na Meta. Se o OpenAI
   tiver de honrar o `OutputSchema.strict` (a Chat Completions honra-o, com `_strict_schema`), a
   O03 junta um campo ao perfil.

### Para a O03

- Definir a tabela de famílias do OpenAI. A O01 mostrou que a OpenAI aceita raciocínio de outra
  família sem erro; a guarda evita carga inútil.
- Definir o mapa de códigos (os do `ResponseError.code`, com os estados a decidir), o
  `takes_tool_choice=True`, o `function_strict=False` e o `include=()`.
- O adaptador novo precisa de entrar no `ADAPTERS` e nos `VALIDATORS` da rede do fio.
- As quatro kwargs que levantam `RequestError` não são do perfil: são do `prepare` do OpenAI.
- O `_http()` continua repetido no `_openai.py` e no `_meta.py`.

### Divisão proposta em commits

- `refactor(providers): extract the Responses API core from the Meta adapter (O02)`. Leva o
  `src`, os testes e o ponteiro do `AGENTS.md`. As duas mudanças de comportamento nasceram com
  os seus testes nos mesmos ficheiros, por isso separá-las não compensa.
- `docs(blackboard): record O02`, com esta ficha.
