# O03 · OpenAI pela Responses no host oficial; Chat Completions só para servidores compatíveis

- **Dono:** por atribuir · **Estado:** todo · **Depende de:** O01 (resultados das verificações 2
  a 7), O02
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

## Desenho (confirma-o na nota de desenho; os factos vêm da O01)

- **Pedido:**
  - `store: false`, e o raciocínio cifrado da maneira que a O01 mostrar (por omissão ou com
    `include`);
  - `reasoning.effort` pelas regras de cada modelo, e `reasoning.summary` com `thinking=True`;
  - `max_output_tokens`, e `text.format` para `output_schema` e `json_mode`;
  - logprobs com `top_logprobs` e o `include` deles;
  - `strict: false` explícito nas function tools, para manter o comportamento de hoje.
- **Regras por modelo:** a tabela de perfis passa às regras da Responses. Deixa de haver "tools só
  a `none`". O sampling dos GPT-6 vai só a `none`, e o Astra e o 6.1 Sol continuam sem `none`.
  Cada regra leva o URL da documentação.
- **Parâmetros sem lugar na Responses:** `stop`, `seed`, `frequency_penalty` e `presence_penalty`
  levantam `RequestError` no host oficial (D43). O `response_format` cru fica por fixar (abaixo).
- **Batch:** endpoint `/v1/responses` (o SDK aceita-o). Os resultados lêem-se pelo endpoint de
  cada batch, para os batches submetidos antes continuarem a ler-se.
- **Contagem de tokens:** `responses.input_tokens.count`, como a Meta. O OpenAI hoje não conta
  tokens no fornecedor.
- **Erros:** as falhas dentro da resposta e do stream vêm do núcleo; os códigos da OpenAI ficam no
  perfil.

## Decisões a fixar (dono, antes do código)

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

- Estado: todo.
