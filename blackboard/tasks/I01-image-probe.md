# I01 · Sonda ao vivo: as APIs de imagem do OpenAI, do Gemini, do xAI e da Meta

- **Dono:** Claude (o script e a execução, autorizados pelo dono a 2026-10-03) · **Estado:** done ·
  **Depende de:** nada
- **Origem:** D46 · **Decisões:** D46, D47 · **Regras:** `R00-rules.md`
- **Excepção às regras, como na O01:** o Claude corre o script, que faz chamadas pagas com um
  tecto de custo. Corre só depois de o dono autorizar as chamadas.

## Problema

A documentação oficial (lida a 2026-10-02, fontes na D46) deixa em aberto factos de que a I03 e a
I04 dependem:

1. **`usage` da Images API nos modelos novos.** A referência diz que o `usage` é "for
   gpt-image-1 only". O guia diz para o ler para calcular o custo dos 2.5
   (https://developers.openai.com/api/reference/resources/images,
   https://developers.openai.com/api/docs/guides/image-generation). Se não vier, o custo tem de
   ser estimado pela tabela de imagens.
2. **Onde a ferramenta `image_generation` da Responses cobra.** "Responses API requests include
   the mainline model's token usage in addition to image generation costs" (guia), mas o
   `ResponseUsage` não separa os tokens de imagem. Sem isto, o meter conta mal uma imagem gerada
   no turno.
3. **Editar sem estado.** O guia mostra a edição com `previous_response_id`, ou com o id da
   imagem, que precisa de estado guardado. O schema dos itens de entrada aceita o `result`, mas
   nenhuma página diz que `store: false` com o item reenviado funciona. A Meta documenta o
   reenvio com `result: null`.
4. **Gemini, modelos de imagem:**
   - se o `usage` separa a modalidade `IMAGE` em `candidatesTokensDetails`;
   - se as "thought images" (`thought: true`, não cobradas) chegam no `generate_content`;
   - se reenviar uma parte de imagem sem a `thought_signature` dá 400 (a validação estrita só está
     documentada para as chamadas a funções:
     https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures);
   - se o stream entrega a imagem inteira num chunk;
   - se `candidate_count > 1` é aceite;
   - se o `image_config` do google-genai 2.25.0 chega (o guia já mostra `response_format`).
5. **Tamanhos do OpenAI:**
   - se o `gpt-image-1.5` recusa um tamanho fora dos três fixos (`1536x864`);
   - `n=2` no 2.5;
   - `output_format="webp"`.
6. **xAI**, só se a conta tiver créditos (ver "Por fazer" no `BOARD.md`):
   - `grok-imagine-image-2.0` com `image_format="base64"`;
   - o `cost_usd`;
   - uma edição com um data URL em `image_url`.
7. **Meta, `muse-image-1.0`:**
   - a saída e o `usage` na Responses;
   - o `size` como proporção;
   - um segundo turno sem estado, com `result: null`.

## Desenho

- `scripts/probe_images.py`, com os SDKs crus (os adaptadores ainda não geram imagens), no estilo
  de `scripts/probe_openai_responses.py`:
  - chaves do ambiente;
  - fornecedores por argumento;
  - tecto de custo por argumento;
  - sempre a qualidade e o tamanho mais baratos.
- Estimativa: menos de $1. Um `low` do OpenAI custa cerca de $0.006, um 1K do Flash Lite $0.034,
  uma imagem do xAI $0.04 e uma da Meta $0.01.
- Cada verificação é um pedido, e o resultado fica registado tal como vem: aceite, 400 com a
  mensagem, ou ignorado.
- As imagens ficam em `scripts/output/` (ignorado pelo git), para se verem; a tabela, em
  Markdown.

## Ficheiros

- `scripts/probe_images.py` (novo)
- `tests/test_probe_images.py` (novo; só a lógica sem rede)
- `scripts/model_probe_notes.md` (os resultados)

## Critério

- O script corre com a autorização do dono. Os números ficam aqui e em
  `scripts/model_probe_notes.md`.
- A I03 só começa com as verificações 1, 4 e 5 respondidas, mais a 6 e a 7 para o xAI e a Meta. A
  I04 só começa com a 2, a 3 e a 4.

## Registo do dono

- Estado: done (2026-10-03), com duas perguntas por responder por razões da conta: a 4 (Gemini) e
  a 6 (xAI).
- Corridas: `images-20261003T031041Z` e `images-20261003T031426Z`, em `scripts/output/model-probes/`
  (ignorado pelo git), com as imagens ao lado. Custo pelo `usage`: cerca de $0.25. Os piores casos
  cobrados pelo tecto foram $1.47 e $0.38.
- Ficheiros:
  - `scripts/probe_images.py` (novo);
  - `tests/test_probe_images.py` (novo, 8 testes sem rede);
  - `scripts/model_probe_notes.md` (as notas em inglês, entrada de 2026-10-03).

### O que isto fixa

1. **O `usage` da Images API vem em todos os modelos medidos** (2.5-flare, 2.5-sunburst,
   gpt-image-2, 1-mini), com tokens de texto e de imagem à entrada e à saída. O custo lê-se do
   `usage` com as tarifas de imagem.
   - Uma 1024x1024 em `low`: 196 tokens de saída no 2.5 e no 2; 272 no 1-mini.
   - Uma edição conta a imagem de entrada como 1024 tokens de imagem.
2. **A ferramenta da Responses cobra a imagem à parte.** O `Response.usage` só traz o modelo
   principal. A imagem vem num campo de topo `tool_usage.image_gen`, que o SDK 3.19.2 não tipa
   (está no `model_extra`), com a forma do `usage` da Images API. O meter pode cobrá-la às tarifas
   do modelo de imagem, sem estimar.
3. **Editar sem estado não pode reenviar o `image_generation_call`:** com o `result`, ou com
   `result: null`, dá 404 ("Items are not persisted when `store` is set to false … remove this
   item from your input").
   - Funciona tirar o item e mandar a imagem como `input_image` numa mensagem do utilizador; o
     modelo edita (`action: "edit"`).
   - Isto parte o reenvio de hoje: o `_replayable_output` reenviaria o item e levaria 404. Fica
     para a I04.
4. **Gemini: por responder.** A chave está no free tier, onde os modelos de imagem têm quota 0
   (429, `limit: 0`). O dono tem de activar a faturação no projecto.
   - Visto sem pedido: o google-genai 2.25.0 recusa `image_config.output_mime_type` no modo
     Developer API.
   - A I03 faz o Gemini pela documentação e pelos tipos do SDK; a verificação ao vivo fica para
     quando houver faturação.
5. **Tamanhos do OpenAI:**
   - o `gpt-image-1.5` só aceita 1024x1024, 1024x1536, 1536x1024 e `auto` (400 com a lista);
   - o 2.5 aceita `WxH` arbitrário;
   - `n=2` e `webp` funcionam.
6. **xAI: por responder.** A conta não tem créditos (`PERMISSION_DENIED`). A I03 faz o xAI pela
   documentação e pelos tipos do `xai-sdk`.
7. **Meta (`muse-image-1.0`):**
   - a saída é `reasoning`, `message` e `image_generation_call`, em WebP por omissão;
   - o `usage` vem em tokens (cerca de 10 000 à entrada, 8 000 em cache), e o preço é $0.01 por
     imagem;
   - o `size` só fixa a proporção: 1024x1024 dá 1600x1600, e 1024x1792 dá 1152x2016;
   - o segundo turno sem estado, com `result: null`, funciona;
   - `/v1/images/generations` também funciona.
- Streaming com parciais: há uma parcial com `partial_images=2`, na Images API e na ferramenta da
  Responses.
