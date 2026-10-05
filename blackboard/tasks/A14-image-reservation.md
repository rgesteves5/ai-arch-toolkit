# A14 · A reserva de uma imagem pela qualidade e pelo tamanho (G-40)

- **Dono:** Claude (2026-10-05) · **Estado:** done · **Depende de:** nada
- **Origem:** o briefing do ai-network, grupo 15 (G-40) · **Decisões:** D61 ·
  **Regras:** `R00-rules.md`

## Problema (verificado a 2026-10-05 no `3273a9a`)

- O pior caso (`core/_metering/_worst_case.py`) reservava 16 000 tokens de imagem por imagem,
  olhando só para o número de imagens (`OperationRequest.declared_images`).
- O briefing pedia o `HeuristicEstimator`, mas ele já só delega no pior caso do core (D49).
- Uma imagem em baixa no Flare custa 196 tokens, e a app contorna com a sua estimativa (D-81).
- Os 16 000 nem eram um tecto: o gpt-image-2 em `high` chega a 23 719 a 2880x2880.

## Registo do dono

- Estado: done (Claude, 2026-10-05), pela D61. Sem chamadas pagas: as contagens vêm das páginas
  dos fornecedores.
- **As contagens publicadas**, lidas por um agente nas páginas oficiais, com a calculadora da
  OpenAI descarregada:
  - **GPT Image 1, 1.5 e `chatgpt-image-latest`:** a tabela do guia (272 a 6 240);
  - **gpt-image-1-mini:** os seus preços por imagem a $8 por milhão;
  - **gpt-image-2 e os 2.5:** a fórmula da calculadora (uma grelha de mosaicos por qualidade).
    A implementação confere com o código da calculadora em todos os 145 730 tamanhos válidos;
  - **Gemini:** 747, 1 120, 1 680 e 2 520 por tamanho;
  - **xAI e Meta:** cobram por imagem.
- **Confere com o que a app mediu:** 1 167 (gpt-image-2, média, 2:3 de 1K), 292 (Flare, a
  mesma), 5 488 (gpt-image-2, alta, 1024x1536) e 196 (Flare, baixa, 1024x1024; corrida da
  frente I).
- **O código:**
  - `SizeTokens`, `TileTokens` e `image_token_bound` em `_openai_images.py`, e as tabelas no
    `_IMAGE_MODELS` do `_openai.py`;
  - os tamanhos do Gemini com os seus tokens (`_ImageProfile.sizes`);
  - `BaseProvider.image_token_bound`, com o valor por omissão `None`;
  - o `request_facts` leva o tecto ao meter (`OperationRequest.declared_image_tokens`), e o
    `worst_case` reserva-o por imagem;
  - sem contagens, a reserva sobe de 16 000 para 24 000 tokens.
- **Revisão independente:** nenhuma falha grave. A fórmula confere com a calculadora (233 168
  tamanhos, por node). Corrigido o que apontou:
  - **média:** o Gemini pensa sempre, ao preço do texto, e a reserva exacta da imagem deixava de
    o cobrir (1 500 tokens de pensamento passavam-na). Agora o `image_text_token_bound` dá o
    limite de saída do modelo, que se reserva quando a geração não declara `max_tokens`;
  - **média:** os modelos cobrados por imagem reservavam 24 000 tokens que nunca pagam, e um tecto
    de tokens recusava-os. O xAI e a Meta dão agora 0;
  - **baixas:**
    - a tabela de imagens misturava modelos fora do inventário;
    - o tipo das imagens do xAI passou a ver-se pelos bytes (`image_media_type`);
    - a redacção do CHANGELOG e da D61;
    - uma entrada Changed para a chamada sem qualidade nem tamanho;
    - um teste do arredondamento a meio;
    - os factos com os adaptadores reais.
- **Testes:** `tests/test_image_token_bounds.py` (50) e quatro em `tests/test_generate_image.py`.
  Um orçamento de $0,02 admite duas imagens em baixa e recusa a qualidade sem contagem. Os
  factos levam as contagens dos adaptadores reais.
- **Docs:** `docs/images.md`, `AGENTS.md`, `CHANGELOG.md` (Added, Changed), D61.
- **Linha para o Registo do briefing do ai-network** (o dono leva-a): "o grupo 15 fechou. A
  reserva estrita de uma imagem sai do modelo, da qualidade e do tamanho pedidos, pelas contagens
  que cada fornecedor publica. Para a OpenAI, a tabela até ao gpt-image-1.5 e a calculadora do
  gpt-image-2 e dos 2.5 (196 tokens em baixa a 1024x1024 no Flare). Para o Gemini, os tokens por
  tamanho.

  Uma qualidade ou um tamanho deixados ao modelo reservam o mais caro que ele pode escolher. Um
  modelo sem contagens reserva 24 000. Na app, a estimativa da D-81 pode sair."
