# A13 · Que modelos vêem imagens (G-39)

- **Dono:** Claude (2026-10-05) · **Estado:** done · **Depende de:** nada
- **Origem:** o briefing do ai-network, grupo 14 (G-39) · **Decisões:** nenhuma ·
  **Regras:** `R00-rules.md`

## Problema (verificado a 2026-10-05 no `3273a9a`)

- A matriz (`docs/model-compatibility.md`) não dizia, de nenhum modelo, se ele lê uma imagem num
  pedido. A app não tinha como saber se uma imagem anexada ia ser vista.
- O adaptador xAI deitava fora as imagens de um turno do utilizador, com o aviso "xAI does not
  support image input". As páginas dos modelos Grok dizem "text, image → text" (o achado de
  2026-09-18 em `FINDINGS.md`, que tinha ficado sem tarefa).

## Nota de desenho

- O dono não quer sondas ao vivo nem chamadas pagas: a resposta vem das páginas de cada modelo
  nos sites dos fornecedores, lidas a 2026-10-05.
- A sonda ganha o cenário `vision` (um quadrado vermelho de 64 píxeis, construído com a
  biblioteca padrão), listado no inventário para os modelos documentados. Fica para o dono
  correr.
- O adaptador xAI passa a mandar as imagens, JPEG ou PNG (os tipos que a página do xAI dá), e
  recusa outro tipo com `RequestError` antes de enviar. Os documentos continuam de fora, com um
  aviso que já não diz que o xAI não os aceita.

## Registo do dono

- Estado: done (Claude, 2026-10-05).
- **A matriz:** a secção "Image Input" de `docs/model-compatibility.md` dá, para os 35 modelos do
  inventário, se lêem imagens, com a página de onde vem. Um agente leu as páginas por resumo,
  e conferi três (o `grok-4.7`, o `muse-spark-1.3` e o `gemini-3-flash-preview`). Lêem imagens:
  - os 13 modelos OpenAI (e os GPT-5.6);
  - seis Grok;
  - cinco Gemini;
  - sete Claude do inventário (o oitavo, o Sonnet 4, foi retirado), mais o Fable 5.1 e o Sonnet 5.5;
  - o Muse Spark 1.3.
- **Os retirados:** o inventário guarda cinco modelos que já não servem ou não se chamam por
  aqui:
  - `grok-4-1-fast-*` (retirados a 2026-05-15);
  - `gemini-3.1-flash-lite-preview` (desligado a 2026-05-25);
  - `gemini-3.1-flash-live-preview` (só pela Live API);
  - `claude-sonnet-4-0` (retirado a 2026-06-15).
  Ficam sem `vision`, e a tabela di-lo.
- **A sonda:** `scripts/probe_models.py` tem o `vision`, e 30 modelos do inventário listam-no.
  Os testes (`tests/test_probe_models.py`, três novos) usam o fornecedor falso e não chamam
  nenhum fornecedor.
- **O xAI:** o `_user_content` de `_xai.py` manda o texto e as imagens pela ordem
  (`xai_chat.image`). Três testes ao fio em `tests/test_xai_provider.py`: imagens em bytes e por
  URL, um WebP recusado e um documento largado com aviso.
- **Revisão independente:** o fio confere com o SDK, e nenhum outro adaptador larga imagens.
  Corrigido: o tipo de uma imagem dada em linha vê-se pelos bytes, não pelo `media_type`
  declarado (um GIF declarado PNG passava), e um URL fica com o xAI. Na tabela, os modelos fora
  do inventário passaram a "also".
- **Achado resolvido de passagem:** o `LLM("grok-…")` criado depois de um `asyncio.run` já não
  levanta (resolvido na A03, G-17; verificado agora).
- **Docs:** `docs/model-compatibility.md`, `CHANGELOG.md` (Added, Fixed), `FINDINGS.md`.
- **Linha para o Registo do briefing do ai-network** (o dono leva-a): "o grupo 14 fechou. A
  secção 'Image Input' do `docs/model-compatibility.md` diz, modelo a modelo, quem lê imagens,
  pelas páginas dos fornecedores. Lêem-nas os 13 OpenAI, os seis Grok, os cinco Gemini, os Claude
  e o Muse Spark do inventário. O cenário `vision` da sonda espera por uma corrida do dono.

  O adaptador xAI passou a mandar as imagens (só JPEG e PNG), que antes largava. Na app, o
  `vision` do catálogo pode vir desta tabela, e a imagem pode ir aos Grok."
