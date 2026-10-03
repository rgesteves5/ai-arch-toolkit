# I04 · Imagens dentro de um turno: a server tool do OpenAI, o stream e o reenvio

- **Dono:** Claude (2026-10-03) · **Estado:** done · **Depende de:** I01 (verificações 2, 3 e 4),
  I03
- **Origem:** D46 · **Decisões:** D46, D47 · **Regras:** `R00-rules.md`

## Problema

- Um modelo de texto do OpenAI pode desenhar a meio de um turno, com a ferramenta
  `image_generation` da Responses: gpt-5 em diante e os GPT-6
  (https://developers.openai.com/api/docs/guides/tools-image-generation). O toolkit não lhe dá
  essa ferramenta.
- O stream não tem evento para imagens. A OpenAI manda até 3 parciais
  (`response.image_generation_call.partial_image`;
  https://developers.openai.com/api/reference/resources/responses/streaming-events).
- Uma imagem gerada tem de voltar no histórico para ser editada no turno seguinte:
  - no OpenAI, o item `image_generation_call` (sem estado, se a I01 o provar);
  - no Gemini, as partes de imagem com a `thought_signature`.

## Objectivo

- **Server tool `image_generation()`:**
  - Config tipada: `model`, `quality`, `aspect_ratio`, `resolution`, `output_format` e
    `partial_images`, com o vocabulário da D47.
  - Levanta onde o fornecedor não a corre.
  - A tool tem de levar sempre o `model`: por omissão a OpenAI usa o `gpt-image-1`.
  - É a primeira server tool com config tipada; a C05 generaliza depois a partir dela (a D44
    deixou a config das server tools para a C05).
- **`Response.images` no `complete` e no `stream`:**
  - no OpenAI, pelo parser de `image_generation_call` que a I03 pôs no núcleo;
  - no Gemini, já vem da I03.
- **Stream:**
  - `StreamEvent(kind="image", image=GeneratedImage, partial=...)`: as parciais com
    `partial=True`, a final com `False`.
  - O `RichStreamResponse` não muda: a `Response` final traz as imagens.
- **Reenvio:**
  - o `_raw` já leva os itens; confirmar com testes no OpenAI (sem estado) e no Gemini (as
    assinaturas);
  - sem `_raw`, decidir na nota de desenho o que uma mensagem reconstruída faz com uma imagem
    gerada.
- **Custo da imagem no turno:** conforme a verificação 2 da I01.
  - Se o `usage` já a incluir, não há nada a fazer.
  - Se não, o adaptador estima-a pela tabela do modelo de imagem e o custo fica `estimated`, sem
    se fingir exacto.
- **Fora desta ficha:** a server tool do xAI. O adaptador dele ainda não leva server tools; fica
  para a C05.

## Ficheiros

- `src/ai_arch_toolkit/core/_server_tools.py`, `src/ai_arch_toolkit/core/_response.py`
- `src/ai_arch_toolkit/core/_providers/_responses.py`, `_openai.py`, `_gemini.py`
- `src/ai_arch_toolkit/core/__init__.py`, `src/ai_arch_toolkit/__init__.py`
- `tests/test_openai_provider.py`, `tests/test_gemini_provider.py`,
  `tests/test_responses_core.py`, `tests/sdk_streams.py`, `tests/integration/fakeserver.py`

## Provas

- Um turno com a tool devolve texto e imagem. Em stream, as parciais e a final chegam como
  eventos `image`.
- O segundo turno reenvia a imagem e o pedido sai no formato que a I01 provou.
- Uma config que o fornecedor não aplica levanta `RequestError`.
- O custo do turno inclui a imagem, ou fica marcado como estimado.

## Registo do dono

- Estado: done (Claude, 2026-10-03).
- **A tool.** `image_generation(model=..., quality, aspect_ratio, resolution, output_format,
  partial_images)` no `_server_tools.py`, exportada.
  - O `model` é obrigatório: a OpenAI usaria o `gpt-image-1` por omissão.
  - O `ResponsesProfile` ganhou `image_tool`, o construtor da tool com as regras do modelo de
    imagem (no OpenAI, o tamanho portável da I03). A Meta não o tem e recusa a tool.
- **O custo.** A OpenAI manda os tokens da imagem em `tool_usage.image_gen` (I01).
  - O `usage` do turno soma-os, porque contam nos tectos de tokens.
  - O `provider_cost` é o custo do turno mais o da imagem às tarifas do modelo de imagem,
    calculado pela tabela por omissão. Assim o meter fica com o custo conhecido, apesar de uma
    server tool deixar o preço por tokens desconhecido (C05).
- **O stream.**
  - `StreamEvent` ganhou `kind="image"` e `image`.
  - OpenAI: as parciais com `partial=True`, e a imagem do `output_item.done` com `False`.
  - Gemini: as "thought images" com `partial=True`, a final com `False`.
  - O `_partial` de um stream abandonado guarda as imagens acabadas.
- **O reenvio** (`ResponsesProfile.image_replay`):
  - na OpenAI, o `image_generation_call` sai e a imagem volta como `input_image` numa mensagem do
    utilizador depois do turno, ou antes, se o turno chamou uma função;
  - na Meta, volta por referência (`result: null`), e o `muse-image` é uma família própria.
  - Sem `_raw` (histórico de uma base de dados), a imagem não volta: a app junta-a à mensagem
    seguinte, se quiser editá-la.
- **Bug encontrado ao vivo e corrigido:** sem stream, a OpenAI recusa `partial_images` com 400
  ("Partial images are only supported with streaming"). O envio sem stream tira-o.
- **Complexidade:** o `open_stream` do núcleo passou de 12 para baixo do limite, com o mapeamento
  dos eventos numa função `_streamed`.
- **Verificação ao vivo (2026-10-03, cerca de $0.05):**
  - `gpt-5-nano` com a tool (`gpt-image-2.5-flare`, `low`): uma imagem, 229 tokens de imagem,
    $0.00723 com o meter igual e 0 custos desconhecidos;
  - segundo turno pelo `to_message()`: `action: "edit"`, céu roxo (visto), $0.0198;
  - stream com `partial_images=2`: a imagem final como evento (nesta corrida a OpenAI não mandou
    parcial; na I01 mandou uma);
  - `muse-image-1.0`: segundo turno por referência, uma imagem, $0.01.
- **Testes:** `tests/test_images_in_the_turn.py` (novo, 16 testes).
- Gate: 6215 passed, 42 skipped; ruff, formatação e pyright limpos.
