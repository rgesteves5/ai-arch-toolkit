# I02 · Tipos, preços e o charge site do `generate_image`

- **Dono:** Claude (2026-10-03) · **Estado:** done · **Depende de:** nada (corre em paralelo com a
  I01)
- **Origem:** D46 · **Decisões:** D46, D47 · **Regras:** `R00-rules.md`

## Problema

- O `LLM` não tem chamada para modelos de imagem: só `complete`, `stream`, `stream_events`, o
  batch e o `count_tokens`.
- A `Response` não tem onde guardar uma imagem.
- O `Usage` e o `ModelPricing` só conhecem tokens de texto. Uma imagem do OpenAI paga tokens de
  imagem a outra tarifa: no `gpt-image-2.5`, $8/M à entrada e $30/M à saída, contra $5/M de texto
  (https://developers.openai.com/api/docs/pricing). A Meta cobra por imagem.

## Objectivo

- **Tipos:**
  - `GeneratedImage` (`frozen`, `slots`, `kw_only`): `data: bytes`, `media_type: str`,
    `revised_prompt: str = ""`. Exportado em `ai_arch_toolkit.core` e no topo.
  - `Response.images: tuple[GeneratedImage, ...] = ()`. O `Response.__bool__` passa a ser
    verdadeiro também com imagens.
- **Chamada:**
  - `LLM.generate_image()` e `generate_image_sync()`, com a assinatura da D47.
  - Corre pelo mesmo `Execution` do `complete`, com as opções de imagem no `Request`. Não há
    segundo pipeline: retries, fallbacks (para outro modelo de imagem), middleware, attempts e o
    meter valem tal como estão.
- **Contrato do adaptador** (`BaseProvider`):
  - Um pedido de imagem prepara-se no `prepare` (puro, devolve o tipo do SDK) e envia-se no
    `send`. O `assemble` devolve a `Response` com as `images`.
  - Um adaptador que não gera imagens recusa antes de qualquer reserva no meter.
  - O stream de imagens fica para a I04: aqui, um `generate_image` é sempre `complete`.
- **Custo:**
  - O `Usage` ganha `image_input_tokens` e `image_output_tokens`, disjuntos dos outros contadores
    como diz a docstring dele. `tokens` passa a somá-los.
  - O `ModelPricing` ganha `image_input`, `image_output`, `image_cache_read` (só na Responses) e
    as variantes de batch que existirem, mais um `per_image` para os preços por imagem.
  - O `_estimate_response_cost` usa-os todos.
  - O `provider_cost` continua a ganhar quando vem (xAI).
- **Meter:**
  - Uma operação de imagem é `kind="llm"`, `mode="complete"`.
  - Sob enforcement, a reserva é o pior caso do pedido: `n` imagens ao preço mais alto que o
    pedido permite. Quem a calcula é o estimador (`toolkit/budget/_estimator.py`), a partir de um
    facto neutro do `OperationRequest`, sem heurísticas no core.
- **Testes:**
  - O `FakeProvider` (`tests/fake_provider.py`) passa a dar respostas com imagens.
  - Um `LLM` real sobre ele prova o charge com um `MeterScope`. É o "feito quando" do
    ai-network: a imagem em bytes, com o mime, e o custo no meter, provado com um fornecedor
    falso.

## Nota de desenho (escrita antes do código, 2026-10-03)

- **O pedido.**
  - Um módulo novo, `core/_images.py`, define o `ImageRequest` (`frozen`, `slots`, `kw_only`):
    `n`, `aspect_ratio`, `resolution`, `quality` e `output_format`, com `Literal`s para
    `resolution` (`"512"`, `"1K"`, `"2K"`, `"4K"`) e `output_format` (`"png"`, `"jpeg"`,
    `"webp"`). A validação comum vem no `__post_init__` (`n >= 1`, a forma `"W:H"` da
    proporção, a `quality` não vazia) e levanta `RequestError`.
  - O `Request` do middleware ganha `image: ImageRequest | None = None`.
  - O prompt e as imagens de entrada vão nas `messages`, como um `user([prompt, *images])`. Assim
    o middleware que lê o texto do pedido (moderação, redacção) continua a vê-lo.
  - Um pedido de imagem não leva kwargs: nem os defaults do `LLM` (`temperature`, `max_tokens`),
    nem opções de texto. O adaptador recusa com `RequestError` as que um middleware lhe juntar,
    em vez de as ignorar em silêncio (como na D45).
- **O caminho.**
  - O `Arguments` ganha `image`. O `_prepare_call` constrói o `Request` com ele; um fallback
    reconstrói o pedido de imagem para o seu modelo pelo mesmo `_prepare_call`, e o `rewritten`
    passa o `image` de um middleware.
  - O `LLM.generate_image()` monta o `Arguments` e corre `Execution(..., path="complete")`.
  - Uma única função, `prepare(provider, request)` no `_attempts.py`, escolhe entre
    `provider.prepare` e `provider.prepare_image`. É usada nos dois sítios que hoje chamam o
    `prepare`.
- **O contrato do adaptador.**
  - O `BaseProvider` ganha `prepare_image(request) -> P`, que por omissão levanta `RequestError`
    ("… does not generate images"). Como o `Execution` prepara antes de abrir a operação, um
    adaptador sem imagens nunca chega ao meter.
  - O `P` e o `F` de cada adaptador cobrem os dois pedidos: o `send` escolhe o endpoint e o
    `assemble` lê as imagens. Não há segundo `send` nem segundo `assemble`: o Gemini e a Meta
    usam o mesmo endpoint, e o OpenAI e o xAI escolhem pelo tipo do `P`.
- **A resposta.**
  - `GeneratedImage` (`data`, `media_type`, `revised_prompt`) e `Response.images`, no
    `_response.py`.
  - O `__bool__` passa a contar as imagens. O `to_message()` não muda: a imagem viaja no `_raw`,
    e a I04 trata do reenvio.
- **O custo.**
  - O `Usage` ganha `image_input_tokens`, `image_output_tokens` e `image_count`, disjuntos dos
    outros contadores; `tokens` soma os de imagem.
  - O `image_count` existe para os preços por imagem: a Meta cobra $0.01 por imagem, e o `usage`
    dela vem em tokens que não são a fatura (I01).
  - O `ModelPricing` ganha `image_input`, `image_output`, `batch_image_input`,
    `batch_image_output` e `per_image`.
  - O `estimate_cost` e o `price` cobram cada contador à sua tarifa. Uma tarifa de imagem que
    falte cai na de texto (`_first_rate`), e o `per_image` que falte vale zero.
  - Os agregadores (meter, trace, flow, relatório do budget) somam os campos novos.
- **O pior caso.**
  - O `OperationRequest` ganha `declared_images` (o `n` pedido; 0 fora das imagens).
  - O `HeuristicEstimator` reserva, por imagem declarada, uma folga de tokens de saída de imagem
    e o `per_image` do modelo, ambos pelo pricer. A folga é uma opinião, por isso fica no
    toolkit, como a `_NON_TEXT_TOKEN_ALLOWANCE`.
- **Bug encontrado ao desenhar (corrigido aqui):** o `_request_size` conta uma imagem como texto e
  não como parte não textual. O `str()` de um `ImagePart` com bytes entra no tamanho: uma imagem
  de 1 MB dá 2,9 milhões de caracteres, cerca de 735 mil tokens reservados. E o `ImagePart` não é
  `dict`, por isso `non_text_parts` fica a 0.
  - Correcção: as `ImagePart` e `DocumentPart` contam como partes não textuais, sem a fonte no
    tamanho.
  - Afecta qualquer pedido com imagens sob enforcement, não só as edições.
- **Testes que provam que não nasce um segundo caminho:**
  - o `generate_image` passa pelo middleware, pelo retry e pelo fallback do `complete`;
  - o charge é um só, no `_PhysicalAttempt`;
  - o teste de arquitectura continua a ver um só sítio que chama o `prepare` dos adaptadores.

## Ficheiros

- `src/ai_arch_toolkit/core/_response.py`, `src/ai_arch_toolkit/core/_llm.py`,
  `src/ai_arch_toolkit/core/_attempts.py`, `src/ai_arch_toolkit/core/_providers/_base.py`
- `src/ai_arch_toolkit/core/_pricing.py`, `src/ai_arch_toolkit/core/_metering/_operation.py` (se
  o facto for preciso), `src/ai_arch_toolkit/toolkit/budget/_estimator.py`
- `src/ai_arch_toolkit/core/__init__.py`, `src/ai_arch_toolkit/__init__.py`
- `tests/fake_provider.py`, `tests/test_generate_image.py` (novo), os testes de metering e
  budget que mudarem, `tests/test_pricing.py`, `tests/test_architecture.py` (os charge sites)

## Provas

- `generate_image` num adaptador sem suporte levanta antes de reservar (o meter fica a zeros).
- Com um `MeterScope` e uma `BudgetPolicy`: a imagem é cobrada uma vez, a reserva cobre `n`
  imagens, e um retry não cobra a dobrar.
- O custo por tokens de imagem e o custo por imagem batem com as tarifas registadas.
- Um fallback de um modelo de imagem para outro funciona como no `complete`.

## Registo do dono

- Estado: done (Claude, 2026-10-03), pela nota de desenho e sem desvios.
- Ficheiros:
  - `core/_images.py` (novo);
  - `core/_middleware.py` (`Request.image`);
  - `core/_response.py` (`GeneratedImage`, `Response.images`, contadores de imagem no `Usage`,
    `Usage.__add__`);
  - `core/_pricing.py` (tarifas de imagem, `_image_cost`);
  - `core/_metering/_operation.py` (`declared_images`);
  - `core/_metering/_store.py` (os tokens de imagem contam nos tectos de tokens);
  - `core/_attempts.py` (`Arguments.image`, `prepare()`, `_request_size`);
  - `core/_llm.py` (`generate_image`, `generate_image_sync`);
  - `core/_providers/_base.py` (`prepare_image`);
  - `core/_trace.py` e `core/_step.py` (o `usage` serializado com todos os campos, e a soma pelo
    `Usage.__add__`);
  - `toolkit/budget/_estimator.py` (a folga por imagem);
  - as duas `__init__.py`.
- Testes:
  - `tests/test_generate_image.py` (novo, 26 testes);
  - `tests/test_pricing.py` (`TestImagePricing`, 6);
  - `tests/test_architecture.py` (a preparação tem uma só casa, a função `prepare`);
  - `tests/fake_provider.py` (`images=True`).
- O bug do `_request_size` tem teste próprio, que falhava antes: 2 940 090 caracteres e 0 partes
  para uma imagem de 1 MB.
- Gate: 6120 passed, 42 skipped; ruff, formatação e pyright limpos.
- Linhas para o `CHANGELOG` (o coordenador junta-as na I05):
  - Added: `LLM.generate_image()`/`generate_image_sync()`, `Response.images` e `GeneratedImage`,
    `ImageRequest` e `Request.image`, os contadores de imagem do `Usage` e o `Usage.__add__`, as
    tarifas de imagem do `ModelPricing`;
  - Fixed: um pedido com imagens reservava, sob enforcement, o tamanho dos bytes como texto e não
    contava a imagem como parte.
