# A12 · A memória num pedido com imagens (G-38)

- **Dono:** Claude (2026-10-05) · **Estado:** done · **Depende de:** nada
- **Origem:** o briefing do ai-network, grupo 13 (G-38) · **Decisões:** nenhuma ·
  **Regras:** `R00-rules.md`

## Problema (verificado a 2026-10-05 no `3273a9a`)

- O `_extract_query` do `MemoryMiddleware` (`toolkit/memory/_middleware.py`) lia de um conteúdo
  em lista só as partes `dict` (`p.get("text")`).
- O texto de um `user([texto, image(...)])` vem como `str`, por isso a procura ficava vazia e o
  `abefore` devolvia o pedido sem memórias. A app manda sem memórias todo o pedido com uma imagem
  anexada (E06-08).

## Registo do dono

- Estado: done (Claude, 2026-10-05).
- O `_extract_query` junta, por um espaço, o texto das partes de texto: as `str`, as `cache()` e
  os `dict` com `text`. As imagens e os documentos ficam de fora. Uma mensagem sem texto não dá
  procura, como antes.
- O `aafter` regista a interacção pelo mesmo texto.
- **Testes:** `tests/memory/test_middleware.py` ganhou quatro. Um pedido com uma imagem procura
  pela pergunta e leva as memórias; todos os tipos de parte; uma mensagem só com uma imagem não
  procura; o registo. Três falham no código antigo.
- **Docs:** `docs/memory.md`, `CHANGELOG.md` (Fixed).
- **Linha para o Registo do briefing do ai-network** (o dono leva-a): "o grupo 13 fechou. O
  `MemoryMiddleware` procura as memórias pelo texto das partes de um pedido (as `str`, as
  `cache()` e as `{"text": …}`), e um pedido com uma imagem anexada leva-as. Na app, a E06-08
  pode tirar o contorno."
