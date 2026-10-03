# I03 · Os adaptadores: gerar e editar imagens no OpenAI, no Gemini, no xAI e na Meta

- **Dono:** Claude (2026-10-03) · **Estado:** done · **Depende de:** I01 (verificações 1, 4 e 5;
  a 6 e a 7 para o xAI e a Meta), I02
- **Origem:** D46 · **Decisões:** D46, D47 · **Regras:** `R00-rules.md`

## Problema

Depois da I02, o `generate_image` existe, mas nenhum adaptador o sabe fazer. O Gemini deita fora
as partes `inline_data` (`_gemini.py`, `_parse_sdk_response`), e o núcleo da Responses ignora os
itens `image_generation_call`.

## Objectivo

- **OpenAI** (`_openai.py`):
  - `images.generate`, ou `images.edit` com `images=` (até 16;
    https://developers.openai.com/api/reference/resources/images);
  - `size` em `WxH` calculado a partir de `aspect_ratio` + `resolution` (D47);
  - `quality`, `n` e `output_format` pelas regras de cada modelo;
  - o custo pelo `usage` de tokens de imagem, ou pela tabela se a I01 mostrar que não vem.
- **Gemini** (`_gemini.py`):
  - `generate_content` com `response_modalities` e `image_config`, e `images=` como partes do
    pedido;
  - as partes `inline_data` passam a `Response.images`, e as "thought images" ficam de fora;
  - o custo pela modalidade `IMAGE` do `usage`. Isto corrige também o `complete` num modelo de
    imagem.
- **xAI** (`_xai.py`):
  - `image.sample` (ou `sample_batch` para `n > 1`), com `image_format="base64"`, e `image_urls`
    para editar (até 5);
  - o custo é o `provider_cost` (`cost_usd`).
- **Meta** (`_meta.py` e `_responses.py`):
  - `muse-image-1.0` pela Responses;
  - um parser de `image_generation_call` no núcleo partilhado, que a I04 reutiliza para o
    OpenAI;
  - preço por imagem ($0.01; https://dev.meta.ai/docs/pricing-rate-limits).
- **Encaminhamento:** o prefixo `muse-image-` no `create_provider`. Os outros já entram:
  `gpt-image-`, `gemini-…-image` e `grok-imagine-`.
- **Regras por modelo:** proporções, resoluções, qualidades, `n` e o limite de imagens de
  entrada, na tabela de cada adaptador, com o URL da documentação ou a prova da I01.
- **Preços** (`_default_pricing.toml`, cada entrada com a fonte):
  - `gpt-image-2.5-sunburst`, `gpt-image-2.5-flare`, `gpt-image-2`, `gpt-image-1.5`,
    `chatgpt-image-latest`, `gpt-image-1`, `gpt-image-1-mini`;
  - `gemini-3.1-flash-image`, `gemini-3.1-flash-lite-image`, `gemini-3-pro-image`;
  - `grok-imagine-image-2.0`, `grok-imagine-image`, `grok-imagine-image-quality` (este último
    retirado a 2026-11-02);
  - `muse-image-1.0`.

## Ficheiros

- `src/ai_arch_toolkit/core/_providers/_openai.py`, `_gemini.py`, `_xai.py`, `_meta.py`,
  `_responses.py`, `__init__.py`
- `src/ai_arch_toolkit/core/_default_pricing.toml`
- `tests/test_openai_provider.py`, `tests/test_gemini_provider.py`, `tests/test_xai_provider.py`,
  `tests/test_meta_provider.py`, `tests/test_responses_core.py`,
  `tests/test_provider_registry.py`, `tests/test_pricing.py`
- `tests/integration/fakeserver.py` (a Images API), `tests/integration/fakegrpc.py` (o serviço de
  imagem do xAI), `tests/integration/test_provider_transport.py`

## Provas

- Cada adaptador, com o SDK real sobre o loopback, devolve os bytes, o mime e o custo.
- Uma edição leva as imagens de entrada no fio certo de cada fornecedor.
- Um parâmetro que o modelo não aceita levanta `RequestError` antes de enviar.
- O `wire_contract` passa contra os tipos do SDK.
- Um `complete` num modelo de imagem do Gemini devolve as imagens. Antes perdiam-se: o teste
  falha antes.

## Registo do dono

- Estado: done (Claude, 2026-10-03).
- **Desenho:**
  - **OpenAI e Meta** partilham a Images API num módulo novo, `_openai_images.py`:
    - `ImageModel` (as regras), `ImagesCall`, o tamanho portável, o pedido, a leitura da resposta
      e o `usage`;
    - o núcleo `ResponsesProvider` envia um `ImagesCall` para `images.generate`/`edit` e lê o
      `ImagesResponse`, e o `assemble` passou para o núcleo (era igual nos dois adaptadores);
    - o OpenAI tem a tabela `_IMAGE_MODELS` (2.5 e 2: tamanho livre; 1.5, 1, 1-mini e
      chatgpt-image-latest: três tamanhos fixos; um `gpt-image-` não listado leva as regras do 2.5);
    - a Meta tem o `muse-image-` e o prefixo de encaminhamento novo.
  - **Gemini:** `prepare_image` com `response_modalities=["IMAGE"]` e `image_config`, por uma tabela
    fechada de três modelos.
  - **xAI:** `prepare_image` com os argumentos do `image.sample`, ou do `sample_batch` para
    `n > 1`; o custo é o do proto.
  - **Partilhado:** `image_prompt`, `image_bytes` e `image_media_type` no `_base.py`.
  - O `complete` num modelo de imagem do Gemini ou da Meta devolve as imagens: o parser do Gemini e
    o do núcleo da Responses leram-nas.
- **Tamanho portável** (D47):
  - a resolução é a área de um quadrado com esse lado (como o "1K" do Gemini);
  - no OpenAI dá `WxH` em múltiplos de 16, dentro de 655 360 a 8 294 400 píxeis e 3840 de lado;
  - nos modelos fixos, só 1:1, 2:3 e 3:2;
  - na Meta, só a proporção.
- **Bugs encontrados e corrigidos:**
  - Gemini: um `ImagePart` com `data:` URL ia como `file_uri` em vez de bytes (teste próprio);
  - Meta: editar com a lista de ficheiros dava 400 (o SDK nomeia-os `image[]`, a Meta quer
    `image[N]`). Agora uma imagem vai como um ficheiro só, e a Meta edita uma imagem de cada vez;
  - Meta: o `complete` no `muse-image` saía a custo zero, porque o `usage` da Responses não conta
    imagens. O núcleo conta agora os `image_generation_call` em `image_count`.
- **Verificação ao vivo (2026-10-03, cerca de $0.06):**
  - OpenAI: `generate_image` no `gpt-image-2.5-flare` (16:9 deu 1360x768, 110 tokens de imagem,
    $0.003385, e o meter igual), uma edição (1032 tokens de imagem à entrada) e o
    `gpt-image-1-mini` a 3:2;
  - Meta: geração ($0.01, e o meter igual), edição (céu roxo, visto) e `complete` (uma imagem,
    $0.01);
  - o Gemini e o xAI ficam por verificar (sem faturação, sem créditos).
- **Testes:**
  - `tests/test_image_generation_providers.py` (novo, 66 testes);
  - `tests/integration/test_image_transport.py` (novo: os quatro SDKs reais em loopback);
  - `fakegrpc.py` com o serviço Image;
  - a rede do fio (`conftest.py`, `wire_contract.py`) verifica também o `prepare_image`, com
    canários em `tests/test_wire_contract.py`;
  - `tests/provider_calls.py` ganhou `image_request`, `prepare_image` e `generate_image`.
- Gate: 6196 passed, 42 skipped; ruff, formatação e pyright limpos.
