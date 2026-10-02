# O03 · OpenAI pela Responses no host oficial; Chat Completions só para servidores compatíveis

- **Dono:** Claude (agente, 2026-10-02) · **Estado:** done · **Depende de:** O02 (a O01 está feita: os factos
  estão no "O que isto fixa" da ficha dela)
- **Origem:** D43 · **Decisões:** D43 · **Regras:** `R00-rules.md`

## Problema

- O `_openai.py` só usa `chat.completions.create`. Do GPT-5.4 em diante as tool calls só vão com
  effort `none` ("Starting with GPT-5.4, Chat Completions does not support tool calling with
  `reasoning_effort` values other than `none`",
  https://developers.openai.com/api/docs/guides/migrate-to-responses).
- O GPT-6 Astra e o GPT-6.1 Sol não têm `none` e não chamam tools por esta API.
- A OpenAI não dá aqui resumos de raciocínio nem raciocínio reenviável, e as server tools
  levantam `RequestError`.

## Objectivo

- No host oficial (sem `base_url`, ou com `api.openai.com`) o OpenAI vai pela Responses, sobre o
  núcleo da O02.
- Noutro host a Chat Completions fica como está (perfil `_COMPATIBLE`), num adaptador só dela.
- Nenhuma escolha de endpoint na API pública: o `create_provider` escolhe pelo host. Os nomes das
  classes são internos.

## Desenho (confirma-o na nota de desenho; os factos vêm da O01, medidos a 2026-10-02)

- **Pedido:**
  - `store: false`, sem `include`: o raciocínio vem cifrado por omissão (O01);
  - `reasoning.effort` pelas regras de cada modelo, e `reasoning.summary` com `thinking=True`;
  - `max_output_tokens`, e `text.format` para `output_schema` e `json_mode`;
  - logprobs com `top_logprobs` e o `include` deles;
  - `strict: false` explícito nas function tools: sem ele, a OpenAI reescreve o schema em modo
    strict e torna obrigatórios os parâmetros opcionais (O01).
- **Regras por modelo:** a tabela de perfis passa às regras da Responses. Deixa de haver "tools só
  a `none`". O sampling dos GPT-6 vai só a `none`, como na Chat Completions, e o Astra e o 6.1 Sol
  continuam sem `none`. O `max` volta para os GPT-6: a Responses aceita-o e a Chat Completions não
  (O01). Cada regra leva o URL da documentação, ou a prova ao vivo.
- **Parâmetros sem lugar na Responses:** `stop`, `seed`, `frequency_penalty` e `presence_penalty`
  levantam `RequestError` no host oficial (D43). Ao vivo, os dois primeiros dão 400 e os outros
  dois 500 ao fim de ~90 s (O01). O `response_format` cru fica por fixar (abaixo).
- **Batch:** endpoint `/v1/responses` (o SDK aceita-o). Os resultados lêem-se pelo endpoint de
  cada batch, para os batches submetidos antes continuarem a ler-se.
- **Contagem de tokens:** `responses.input_tokens.count`, como a Meta. O OpenAI hoje não conta
  tokens no fornecedor.
- **Erros:** as falhas dentro da resposta e do stream vêm do núcleo; os códigos da OpenAI ficam no
  perfil.

## Decisões a fixar (dono, antes do código) — fixadas na D44

1. `response_format` cru no host oficial: mapear para `text.format`, ou recusar a favor de
   `output_schema`/`json_mode`. Proposta: recusar, para ficar uma só maneira.
2. Tools alojadas do OpenAI (`web_search` e as outras): nesta ficha só as que o mecanismo actual já
   leva sem config, ou tudo na C05 (decisão 6 da ficha dela, opção c). Proposta: a C05.
3. Servidores compatíveis que já implementam a Responses (vLLM, Ollama, LM Studio): ficam na Chat
   Completions até alguém pedir. Proposta: sim.

## Ficheiros

- `src/ai_arch_toolkit/core/_providers/_openai.py` (passa à Responses)
- `src/ai_arch_toolkit/core/_providers/_openai_compatible.py` (novo: a Chat Completions de hoje,
  sem as regras dos modelos OpenAI)
- `src/ai_arch_toolkit/core/_providers/__init__.py`, `src/ai_arch_toolkit/core/_providers/_responses.py`
  (só o que o perfil OpenAI precisar), `src/ai_arch_toolkit/core/_batch.py` (se for preciso)
- `tests/test_openai_provider.py`, `tests/test_provider_registry.py`,
  `tests/integration/test_provider_transport.py` (Responses no `fakeserver.py`),
  `tests/test_architecture.py`

## Provas

- Os testes que hoje afirmam "requires the Responses API" passam a afirmar que a tool call sai com
  o effort pedido. Ficam listados na ficha, não se apagam em silêncio.
- Um loop de tools com raciocínio reenvia os itens cifrados (`fakeserver.py`, SDK real).
- Um `base_url` de outro host continua na Chat Completions, com o mesmo pedido de hoje (comparação
  do `prepare`).
- O `wire_contract` passa contra os tipos do SDK.

## Registo do dono

- Estado: done (Claude, 2026-10-02). Revista e commitada pelo coordenador; as linhas do `CHANGELOG` entraram consolidadas.

### Nota de desenho (escrita antes do código)

- **Casas.**
  - `_openai.py`: o OpenAI no host oficial, pela Responses, sobre o núcleo da O02. Fica com o
    perfil (famílias, códigos), a tabela de regras por modelo, as kwargs que a Responses não tem,
    `count_tokens` e o batch em `/v1/responses`.
  - `_openai_compatible.py` (novo): a Chat Completions de hoje, só para os outros hosts e sem
    regras de modelo (o `_COMPATIBLE` de hoje deixa de ser um perfil: é o adaptador). Fica também
    com o Batch API da OpenAI (o ficheiro JSONL e os `batches`), que os dois adaptadores
    partilham, e com o parser do corpo da Chat Completions, que o adaptador oficial usa para ler
    os batches submetidos antes da O03. A dependência vai num só sentido: oficial → compatível.
  - `_responses.py`: só o que o perfil OpenAI pede (abaixo), e o cliente HTTP com o gancho do
    envio, que sai do `_meta.py` para os dois adaptadores da Responses.
  - `_base.py`: `strict_schema`, o subconjunto strict da OpenAI, sai do `_openai.py` porque os
    dois adaptadores o usam (o compatível no `response_format`, o oficial no `text.format`).
    Mudar de casa é tocar-lhe: parte-se em funções de complexidade ≤ 10, e a entrada
    `_openai.py:_ensure_strict` (17) sai da linha de base.
  - `__init__.py`: o `create_provider` escolhe pelo host (sem `base_url`, ou com o host
    `api.openai.com` de `_OWN_HOSTS`, vai a Responses; outro host, Chat Completions). O nome do
    fornecedor continua `openai`; a API pública não muda (D43).
- **Interface.**
  - `OpenAIProvider(model, api_key, *, base_url=None, timeout=None)` fala sempre Responses.
    Construído à mão com outro `base_url`, fala Responses com ele: é assim que os testes o levam
    ao `127.0.0.1`.
  - `OpenAICompatibleProvider(model, api_key, *, base_url, timeout=None)`.
  - `ResponsesProfile.output_strict: bool`: o fornecedor honra `OutputSchema.strict` (o OpenAI
    sim, como hoje na Chat Completions; a Meta não, D13). Fecha o desvio 5 da O02.
  - Batch, no `_openai_compatible.py`: `batch_request(req, model) -> Request`,
    `submit_batch(client, endpoint, bodies) -> str`, `batch_status(client, batch_id) -> str`,
    `batch_output(client, batch_id) -> tuple[str, list[dict]]` (o endpoint do batch e as linhas
    do ficheiro de saída), `batch_results(entries, parse) -> list[BatchResult]` e
    `chat_batch_response(body, model) -> Response`.
- **O pedido no host oficial** (`prepare`):
  - antes de tudo, `stop`, `seed`, `frequency_penalty`, `presence_penalty` e `response_format`
    levantam `RequestError` (D43, D44);
  - `store: false`, sem `include` (O01, 2); `instructions`; `max_output_tokens` (de
    `max_tokens`, ou de `max_completion_tokens`, que ganha, como hoje);
  - `reasoning.effort` só com `thinking=True` (o `thinking_effort`, senão `high`, como hoje),
    verificado contra os esforços do modelo; `reasoning.summary: "auto"` quando o esforço não é
    `none`;
  - function tools com `strict: false` (O01, 5), `tool_choice` como vem, `web_search()` sem
    config como tool alojada (D44); a qualquer esforço;
  - `text.format` para `output_schema` (strict honrado, com `strict_schema`) e `json_mode`;
  - logprobs: `include: ["message.output_text.logprobs"]` e o `top_logprobs` que vier (O01, 7).
- **Regras por modelo (Responses).** Dataclass `_Model` (`efforts`, `default_effort`,
  `sampling_while_reasoning`), resolvida por `lookup`; um modelo fora da tabela tem as regras da
  geração actual.
  - Modelos que não raciocinam (`gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini`): sem
    esforços, `thinking=True` levanta.
  - GPT-5 a GPT-5.6, `o3` e um modelo novo: todos os esforços do SDK; com `thinking` a um esforço
    que não é `none`, uma `temperature` que não é 1 cai (sonda de 2026-04-28, como hoje).
  - Astra: `low` a `max`; 6.1 Sol: `low` a `max`, `medium` por omissão; Sol e Luna: `none` a
    `max`, `medium` por omissão. A raciocinar (pelo esforço enviado ou pelo de omissão; sem `none`,
    sempre), caem `temperature`, `top_p`, `top_logprobs` e o `include` dos logprobs.
  - Desaparece o "tools só a `none`", e com ele o envio forçado de `none` às tools do Sol e da
    Luna sem `thinking`: um pedido com tools corre como um sem tools.
- **Famílias (reenvio).** Uma família por geração, com o prefixo `<geração>-` e o nome da
  geração: `gpt-6.1`, `gpt-6` (Astra, Sol, Luna), `gpt-5.6` (Sol, Terra, Luna, o exemplo do guia),
  `gpt-5.5` a `gpt-5.1` e `gpt-5`. O id da geração sozinho é a própria família; um id fora da
  tabela é a sua própria família, sem o sufixo de snapshot (`o3`, `gpt-4o`).
- **Códigos.** `rate_limit_exceeded` → 429 e `server_error` → 500 (os dois que o guia de erros
  nomeia); os outros códigos do `ResponseError` ficam sem estado (`ResponseError`, sem retry).
- **Batch.** O oficial submete a `/v1/responses`; o compatível a `/v1/chat/completions`. Os
  resultados lêem-se pelo endpoint de cada batch: no oficial, um batch de `/v1/chat/completions`
  lê-se com o parser da Chat Completions, qualquer outro com o da Responses. O corpo constrói-se
  com o construtor do SDK e passa pela montagem de sempre: o parser de dicts de hoje desaparece.
- **O que desaparece.** `_Profile.thinking`, `tools_while_reasoning`, `_EARLIER`, `_COMPATIBLE`,
  `_tool_effort`, `self._official`/`_OFFICIAL_HOST`, `max_completion_tokens` no fio oficial, a
  Chat Completions no host oficial, `_parse_batch_response`, o `_http()` do `_meta.py`.
- **Provas.**
  - Os testes do OpenAI que afirmam a Chat Completions no host oficial são reescritos para a
    Responses; os que afirmam o fio da Chat Completions passam ao `OpenAICompatibleProvider`.
  - O pedido de um servidor compatível compara-se, byte a byte (`json.dumps` do `prepare`), com o
    de hoje, capturado antes da mudança.
  - Testes novos para cada regra; um loop de tools com raciocínio que reenvia os itens cifrados
    pelo SDK real contra o `fakeserver.py` (que ganha respostas em sequência).
  - Arquitectura: só o `_openai_compatible.py` chega a `chat.completions`.

### Formas e factos confirmados (SDK `openai` 3.19.2 e documentação, 2026-10-02)

- SDK, com `uv run python`:
  - `batches.create(endpoint=...)` aceita `/v1/responses`; `Batch.endpoint` é `str`.
  - `ReasoningEffort` tem `none`, `minimal`, `low`, `medium`, `high`, `xhigh` e `max`;
    `Reasoning` tem `effort`, `summary`, `generate_summary`, `context` e `mode`.
  - `ResponseIncludable` tem `message.output_text.logprobs`; `ResponseOutputText.logprobs` é
    `list[Logprob]` (`token`, `bytes`, `logprob`, `top_logprobs`).
  - `ResponseError.code` é um `Literal` de 21 códigos (`server_error`, `rate_limit_exceeded`,
    `invalid_prompt`, os de imagem, os de política…); `ResponseErrorEvent.code` é `str | None`.
  - `ResponseCreateParamsNonStreaming` não tem chaves obrigatórias, nem `stop`, `seed` ou as
    penalidades.
  - `EasyInputMessageParam` aceita `role: "assistant"` com `phase`; o núcleo continua a
    reconstruir com a forma de saída (desvio 1), como na Meta.
  - O `model_construct` do SDK constrói os modelos aninhados e guarda os campos extra: é o que lê
    os corpos dos batches.
- Documentação:
  - https://developers.openai.com/api/docs/guides/reasoning: famílias ("gpt-5.6-sol,
    gpt-5.6-terra, and gpt-5.6-luna can reuse each other's reasoning, but reasoning does not
    carry between the GPT-5.6 and GPT-5.5 families"); `summary: "auto"`; o
    `encrypted_content` por omissão com `store: false`. Avisa que os resumos podem exigir a
    verificação da organização (ponto aberto 1).
  - https://developers.openai.com/api/docs/guides/latest-model: "When reasoning effort is not
    `none`, remove `temperature`, `top_p`, and `top_logprobs`".
  - Páginas dos modelos: Astra `low` a `max`; 6.1 Sol `low` a `max`, `medium` por omissão, sem
    `none` nem `minimal`; Sol e Luna `none` a `max`, `medium` por omissão.
  - https://developers.openai.com/api/docs/guides/migrate-to-responses: "omitting `strict`
    attempts strict mode"; `text.format` em vez de `response_format`.
  - https://developers.openai.com/api/docs/guides/batch: `/v1/responses` está entre os
    endpoints; a linha de saída é `{custom_id, response: {status_code, body}, error}`.
  - https://developers.openai.com/api/docs/guides/images-vision: sem `detail`, "it defaults to
    `auto` in both the Responses API and the Chat Completions API" (desvio 3 provado para o
    OpenAI).
  - https://developers.openai.com/api/docs/guides/error-codes: só nomeia estados HTTP; dos
    códigos do `ResponseError` só `rate_limit_exceeded` (429) e `server_error` (500) têm par.

### Feito

- **`_openai.py` (Responses).** `OpenAIProvider(ResponsesProvider)` com o perfil `OpenAI`:
  - famílias por geração (`gpt-6.1`, `gpt-6`, `gpt-5.6` … `gpt-5`, com o prefixo `<geração>-`);
  - `include=()`, `takes_tool_choice=True`, `function_strict=False`, `output_strict=True`;
  - `hosted_tools={"web_search": {"type": "web_search"}}` e os códigos
    `{"rate_limit_exceeded": 429, "server_error": 500}`.
  - `prepare`: recusa as kwargs da Chat Completions, aplica a tabela `_Model` (por `lookup`),
    compõe o pedido do núcleo, junta `reasoning` e passa por `_sampling` (que também junta o
    `include` dos logprobs).
  - `count_tokens` por `responses.input_tokens.count`; batch em `/v1/responses`, lido pelo
    endpoint de cada batch.
- **`_openai_compatible.py` (novo).** A Chat Completions de hoje, sem perfis: o `prepare` é o do
  `_COMPATIBLE` (effort só com `thinking`, verificado contra os esforços do SDK; `max_tokens`;
  sem regras de sampling). E o Batch API partilhado: `batch_request`, `submit_batch`,
  `batch_status`, `batch_output`, `batch_results`, `chat_batch_response`.
- **Núcleo.**
  - `ResponsesProfile.output_strict` (a Meta fica `False`: os testes dela não mudaram).
  - `request_params` aceita `max_completion_tokens`, que ganha, como nome do limite.
  - `_parse_sdk_response` monta `Response.logprobs` (os `Logprob` do texto, por ordem, ou
    `None`).
  - `http_client()` sai do `_meta.py`.
- **`_base.py`.** `strict_schema`, partido em cinco funções de complexidade ≤ 10; a saída é a
  mesma de antes (comparada em esquemas Pydantic com `$ref`, listas, uniões e auto-referência).
- **Encaminhamento.** `_openai_by_host` no `__init__.py`: sem `base_url`, ou com o host
  `api.openai.com` (`_OWN_HOSTS`), `OpenAIProvider`; outro host, `OpenAICompatibleProvider`.
  `resolve_provider_name` continua a dar `openai`.
- **Rede do fio.** O `OpenAIProvider` valida-se contra `ResponseCreateParamsNonStreaming`, com
  os desvios 1 e 3 e sem o 2 (um tool sem `strict` é apanhado); o `OpenAICompatibleProvider`
  contra `CompletionCreateParamsNonStreaming`.

### Tabela de regras por modelo (Responses)

| Modelos | Esforços | Sem esforço enviado | Sampling e logprobs a raciocinar | `thinking=True` |
|---|---|---|---|---|
| `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini` | nenhum | não raciocina | ficam | `RequestError` |
| GPT-5 a GPT-5.6, `o3`, modelo novo | os 7 do SDK | o padrão do modelo (não conta como raciocinar) | ficam; uma `temperature` ≠ 1 cai | effort (senão `high`) + `summary: auto` |
| `gpt-6-astra` | `low` a `max` | raciocina sempre | caem | idem |
| `gpt-6.1-sol` | `low` a `max` | `medium` | caem | idem |
| `gpt-6-sol`, `gpt-6-luna` | `none` a `max` | `medium` | caem, salvo a `none` | idem; a `none` sem `summary` |

Tools a qualquer esforço em todos. `stop`, `seed`, `frequency_penalty`, `presence_penalty` e
`response_format` levantam `RequestError` antes de enviar.

### Ficheiros tocados

- `src/ai_arch_toolkit/core/_providers/_openai.py` (reescrito)
- `src/ai_arch_toolkit/core/_providers/_openai_compatible.py` (novo)
- `src/ai_arch_toolkit/core/_providers/_responses.py`, `.../__init__.py`
- Fora da lista da ficha:
  - `src/ai_arch_toolkit/core/_providers/_base.py` (`strict_schema`) e `.../_meta.py`
    (`output_strict=False`, `http_client`), pela razão da nota de desenho.
  - `tests/test_openai_compatible_provider.py` (novo).
  - `tests/test_batch.py`, `tests/test_provider_conversations.py`,
    `tests/test_provider_parity.py`, `tests/test_provider_system_content.py`,
    `tests/test_runner.py`, `tests/test_stream_tool_calls.py`, `tests/test_tools_schema.py`,
    `tests/test_wire_contract.py`, `tests/wire_contract.py`,
    `tests/test_provider_loop_safety.py` e `tests/test_responses_core.py`: fixavam o módulo antigo
    ou a Chat Completions no host oficial.
  - `tests/integration/fakeserver.py` (respostas em sequência) e `tests/quality_baseline.json`
    (sai a dívida do `_ensure_strict`).
- Nos da lista: `tests/test_openai_provider.py`, `tests/test_provider_registry.py`,
  `tests/integration/test_provider_transport.py`, `tests/test_architecture.py`.

### Testes mudados (nenhum apagado sem equivalente)

- `tests/test_openai_provider.py` (136 testes antes, 132 agora) reescrito para a Responses.
  Passaram ao `tests/test_openai_compatible_provider.py`, por fixarem o fio da Chat
  Completions: `TestMessagesToSdk`, `TestToolToSdk`, `TestParseToolArgs`,
  `TestBuildOutputSchemaFormat`, `TestExtractUsage`, `TestParseSdkResponse`, `TestUsageReport`,
  `test_complete_surfaces_reasoning`, `test_system_passed_as_message`,
  `TestProfiles::test_a_compatible_server_gets_max_tokens` e
  `test_a_compatible_server_calls_tools_while_reasoning`. Reescritos para a Responses:
  - os que afirmavam "requires the Responses API":
    `TestAlwaysReasoning::test_tools_require_responses` → `test_they_call_tools`;
    `TestSolAndLuna::test_tool_calls_while_reasoning_need_the_responses_api` →
    `test_tool_calls_go_at_any_effort`;
    `TestProfiles::test_from_gpt_5_4_tool_calls_while_reasoning_are_refused` e
    `test_earlier_reasoning_models_call_tools_while_reasoning` →
    `TestReasoning::test_tool_calls_go_at_the_effort_asked`;
    `TestProfiles::test_astra_refusals_are_request_errors` → `test_they_call_tools` e
    `test_invalid_reasoning_effort`;
  - o `max`: `TestAlwaysReasoning::test_invalid_reasoning_effort` deixa de o ter;
    `TestSolAndLuna::test_minimal_and_max_are_not_among_their_efforts` →
    `test_minimal_is_not_among_their_efforts`; os dois `test_every_documented_effort_is_sent`
    ganham `max`;
  - `TestSolAndLuna::test_tool_calls_without_thinking_are_sent_at_none` →
    `test_tool_calls_without_thinking_run_at_the_models_default` (já não se força `none`);
  - `max_completion_tokens` → `max_output_tokens`: `TestProfiles::test_the_official_host_sends_max_completion_tokens_for_every_model`,
    `test_max_completion_tokens_forwarded`, `test_gpt5_translates_max_tokens_to_max_completion_tokens`,
    `test_exact_o3_translates_max_tokens_to_max_completion_tokens`,
    `test_gpt5_prefers_explicit_max_completion_tokens`, `TestAlwaysReasoning::test_request_parameters`,
    `TestSolAndLuna::test_by_default_they_reason_so_sampling_is_dropped`;
  - `reasoning_effort` → `reasoning`: os de `thinking`/`temperature` do
    `TestOpenAIProviderComplete`, `test_at_none_they_sample`, `test_a_new_model_gets_the_current_generation_rules`;
  - `response_format` → `text.format`: `test_output_schema_forwarded`,
    `test_output_schema_with_tools_coexist`;
  - `TestProfiles::test_server_tools_are_refused_before_sending` →
    `test_web_search_without_config_is_a_hosted_tool` e
    `test_a_configured_or_another_server_tool_is_refused` (no compatível os três continuam
    recusados);
  - os de erros, rede, retry do `LLM` e ciclo de vida passam a simular `responses.create`.
- `tests/test_batch.py`: os três de `batch_submit` afirmam `/v1/responses`,
  `max_output_tokens` e `instructions`; os de `batch_results` dão ao batch o endpoint
  `/v1/chat/completions` (um batch de antes da O03), em vez de um `MagicMock`.
- `tests/test_provider_conversations.py`, `tests/test_provider_parity.py`,
  `tests/test_stream_tool_calls.py`, `tests/test_runner.py`,
  `tests/test_provider_system_content.py`: o fio da Chat Completions passa ao adaptador
  compatível.
- `tests/test_provider_registry.py`: as seis rotas de outro host fazem patch ao adaptador
  compatível; `test_openai_route` fica sem `base_url`.
- `tests/test_wire_contract.py`: `TestOpenAI` passa à Responses; as canárias da Chat Completions
  passam a `TestOpenAICompatible`.
- `tests/integration/test_provider_transport.py`: os casos OpenAI de hoje correm no adaptador
  compatível (o servidor é `127.0.0.1`).
- `tests/test_tools_schema.py`: três testes verificam a tool da Chat Completions e a da
  Responses.

### Testes novos

- `tests/test_openai_compatible_provider.py`, 65 testes, entre eles
  `test_a_compatible_server_gets_the_request_it_got_before` (9 casos): o `json.dumps` do
  `prepare` é igual, byte a byte, ao do `_openai.py` antigo, capturado antes de qualquer
  mudança (histórico com imagem, documento, calls e resultados; tools e `tool_choice`;
  `thinking`; sampling e logprobs; `output_schema` strict e não strict; `json_mode` com `stop`,
  `seed` e penalidades; `response_format` cru; os dois limites).
- `tests/test_openai_provider.py`: os testes de cada regra acima, mais as famílias (8 casos), um
  turno da Meta reconstruído, o loop de tools pelo `LLM` com o raciocínio cifrado reenviado, os
  códigos, o stream de resumos, `count_tokens` e os batches (submissão, leitura da Responses,
  leitura de um batch da Chat Completions).
- `tests/integration/test_provider_transport.py`: os casos de transporte da Responses do OpenAI
  (8), a resposta falhada com a usage, a ligação recusada, o pedido que o SDK não serializa, e
  `test_openai_a_tool_loop_replays_its_encrypted_reasoning` (completo e em stream). O SDK real
  fala com o `fakeserver.py`: a primeira volta traz raciocínio cifrado, comentário e call; a
  segunda leva `reasoning` (com o `encrypted_content`), `message` (com o `phase`),
  `function_call` e `function_call_output`.
- `tests/test_provider_registry.py`: `test_openai_compatible_route` e
  `test_the_host_chooses_the_openai_api` (5 casos, um deles um host que só começa por
  `api.openai.com`).
- `tests/test_architecture.py`: `test_chat_completions_is_reached_only_by_the_compatible_adapter`
  e a canária do detector.
- `tests/test_responses_core.py`: o limite (`max_completion_tokens` ganha), o `text.format`
  strict e não strict, os logprobs na montagem.
- `tests/test_provider_conversations.py` (5), `tests/test_provider_parity.py` (1),
  `tests/test_provider_system_content.py` (2), `tests/test_wire_contract.py` (os pedidos e as
  canárias da Responses) e o censo do `tests/test_provider_loop_safety.py`, que ganha a Meta e o
  compatível.
- **Falham antes.** Corridos sobre uma exportação de `HEAD` no scratchpad (sem tocar no git), o
  novo `tests/test_openai_provider.py` e o `tests/test_architecture.py` dão 108 falhas e 50
  passagens. As que passam são as invariantes que se mantêm (recusas dos modelos sem raciocínio,
  avisos, a regra da `temperature` dos GPT-5, o cliente, a leitura de um batch da Chat
  Completions). O novo `tests/test_responses_core.py` nem carrega (o perfil não tinha
  `output_strict`).

### Números

- **Linhas:**

  | Módulo | Antes | Depois |
  |---|---|---|
  | `_openai.py` | 810 | 332 |
  | `_openai_compatible.py` | — | 589 |
  | `_responses.py` | 662 | 702 |
  | `_base.py` | 452 | 525 |
  | `_meta.py` | 207 | 204 |
  | `__init__.py` | 266 | 273 |

  O par OpenAI soma 921 contra 810 (mais 111), e o `src` tocado mais 228. Vem do adaptador da
  Responses (perfil, famílias, códigos, tabela citada, batch, `count_tokens`), dos ajudantes de
  batch partilhados e do `strict_schema` partido em funções. Sem o núcleo da O02 seriam mais
  cerca de 450 linhas copiadas. As regras do OpenAI para a Chat Completions desapareceram em vez
  de ficarem duas vezes (o `_Profile`, `_EARLIER`, `_COMPATIBLE`, `_tool_effort`,
  `self._official`), e o parser de dicts dos batches foi substituído pela montagem de sempre.
- **Complexidade:** a dívida `_openai.py:_ensure_strict` (C901 17, PLR0912 16) saiu da linha de
  base. Nos módulos tocados, `ruff --select C901` com o máximo a 10 passa. A função mais longa
  dos módulos novos tem 34 linhas (`_parse_sdk_response`, movida). O `create_provider` continua
  com mais de 60 linhas (94, eram 104): é de antes.
- **Verificações:**
  - `uv run ruff format src tests examples`: 476 ficheiros sem mudança.
  - `uv run ruff check src tests examples`: limpo.
  - `uv run pyright src`: 0 erros.
  - `uv run pytest -q`: 5897 passed, 42 skipped (linha de base 5795 e 42; mais 102).
  - `uv run pytest -m "integration and not live_api" -q`: 100 passed.
  - `uv lock --check`: limpo.

### Linhas propostas para o `CHANGELOG` (`[Unreleased]`)

- **Changed:** "**Breaking:** OpenAI's own host (no `base_url`, or `api.openai.com`) is driven
  through the Responses API instead of Chat Completions; any other `base_url` host keeps Chat
  Completions, with no OpenAI model rule. `stop`, `seed`, `frequency_penalty`,
  `presence_penalty` and a raw `response_format` raise `RequestError` on OpenAI's host before
  sending (structured output is `output_schema` or `json_mode`); OpenAI-compatible servers still
  take them."
- **Changed:** "**Breaking:** OpenAI: `Response.logprobs` holds the Responses API's token
  logprobs of the output text (a tuple of the SDK's `Logprob`), no longer Chat Completions'
  `ChoiceLogprobs`."
- **Changed:** "**Breaking:** OpenAI: GPT-6 Sol and Luna tool calls without `thinking=True` are
  no longer forced to the `none` effort; they run at the model's default (`medium`), like a
  request without tools, so their sampling parameters are dropped."
- **Changed:** "OpenAI: tool calls work at any reasoning effort (`thinking=True` with tools no
  longer raises from GPT-5.4 on), and GPT-6 Astra and GPT-6.1 Sol call tools; the GPT-6 models
  take the `max` effort."
- **Added:** "OpenAI: `thinking=True` returns reasoning summaries as thinking blocks, and a tool
  loop replays each turn's encrypted reasoning from `Response.to_message()` without
  server-side state (`store: false`)."
- **Changed:** "A turn's encrypted reasoning is replayed only to the provider and the model family
  that produced it; any other assistant turn is rebuilt without it." (da O02, alcançável agora)
- **Added:** "OpenAI: `web_search()` without config runs as a hosted tool, and
  `LLM.count_tokens()` counts with `POST /v1/responses/input_tokens`."
- **Changed:** "OpenAI: function tools are sent with `strict: false`, so the Responses API keeps
  their optional parameters optional."
- **Changed:** "OpenAI: batches are submitted to `/v1/responses`; results are read by each
  batch's endpoint, so batches submitted to Chat Completions before still read. A batch result's
  `raw` is the SDK's response object, no longer the parsed JSON dict (OpenAI-compatible servers
  too)."

### Desvios ao plano

1. **`strict_schema` mudou para o `_base.py`**, fora da lista: os dois adaptadores da OpenAI
   usam-no e nenhum deve importar do outro para isso. Mudar de casa obrigou a parti-lo (≤ 10).
2. **`ResponsesProfile.output_strict`**: o OpenAI honra o `OutputSchema.strict` no `text.format`,
   como honrava na Chat Completions (o desvio 5 da O02 previa-o).
3. **`max_completion_tokens` continua aceite no host oficial**, como nome do limite: a D43 e a
   D44 não o recusam, e tem par na Responses (`max_output_tokens`).
4. **Os ajudantes do Batch API ficam no `_openai_compatible.py`**, sem módulo novo: o oficial já
   depende dele para ler os batches de antes, por isso a dependência fica num só sentido.
5. **O `raw` de um resultado de batch passa a ser o objecto do SDK** (era o dict), nos dois
   adaptadores: o corpo passa pela montagem de sempre e o parser de dicts desapareceu. Na
   Responses, o turno de um batch reenvia o raciocínio como o de uma chamada.
6. **O Sol e a Luna deixam de mandar `none` às tools sem `thinking`**: era a volta da Chat
   Completions, e com tools a qualquer esforço um pedido com tools corre como um sem tools.
7. **`reasoning.summary` só quando o esforço não é `none`**: a `none` não há raciocínio a resumir.
8. **O `_http()` da Meta passou ao núcleo (`http_client`)**; o compatível guarda o seu, como
   antes guardava o `_openai.py`: ficam duas cópias de duas linhas, uma por API.
9. **A mensagem do compatível para uma server tool** passou a "only function tools reach an
   OpenAI-compatible server" (o teste procura "server tool").

### Pontos abertos (para a O04, ao vivo)

1. **Resumos e verificação da organização.** O guia de raciocínio diz que os resumos podem exigir
   a verificação da organização e não diz o que acontece sem ela. Se for um 400, `thinking=True`
   passa a falhar numa organização por verificar. A O01 viu os resumos chegar na do dono.
2. **Desvio 1 no OpenAI.** A volta reconstruída (mensagem sem `id`, `status` nem `annotations`)
   não foi provada ao vivo no OpenAI: a O01 reenviou as voltas inteiras. A documentação mostra
   `{"role": "assistant", "content": "..."}`.
3. **Batch com `store: false`** em `/v1/responses`: não provado ao vivo.
4. **Famílias.** O GPT-6 (Astra, Sol, Luna) como uma família, e o 6.1 Sol à parte, são por
   analogia com o exemplo do GPT-5.6. A OpenAI aceita raciocínio de outra família sem erro
   (O01): um engano aqui só custa carga, nunca um erro.
5. **Códigos.** Só `rate_limit_exceeded` e `server_error` têm estado. Os outros (prompt ou imagem
   inválidos, políticas) dão `ResponseError`, sem retry.
6. **Para a O04, além da documentação:**
   - `AGENTS.md` (a linha do OpenAI diz "Chat Completions API only");
   - `docs/model-compatibility.md`, `docs/llm.md`, `docs/framework-overview.md` e `docs/tools.md`;
   - `scripts/model_probe_models.toml`, cujos comentários dizem que o Astra e o 6.1 Sol não
     chamam tools: falta-lhes o cenário `tools_loop`.
7. **Comandos ao vivo para o dono** (pagos, com `set -a && source .env && set +a`):
   - `uv run pytest -m live_api tests/integration/test_provider_hardening_live.py -k openai`;
   - `uv run pytest -m live_api tests/integration/test_provider_contracts_live.py -k openai`;
   - `uv run python scripts/probe_models.py --suite full --model gpt-6-luna --model gpt-6.1-sol --model gpt-6-astra`,
     depois de juntar `tools_loop` ao Astra e ao 6.1 Sol no inventário.

### Divisão proposta em commits

- `feat(openai)!: drive OpenAI's own host through the Responses API (O03)`: o `src` e os
  testes. A separação da Chat Completions e a passagem à Responses nasceram juntas e os testes
  de cada lado dependem do encaminhamento novo; separá-las pediria um estado intermédio que não
  existiu.
- `docs(blackboard): record O03`, com esta ficha.
