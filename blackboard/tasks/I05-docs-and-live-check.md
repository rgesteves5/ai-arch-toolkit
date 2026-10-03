# I05 · Documentação, exemplo e verificação ao vivo final

- **Dono:** Claude (os docs e a verificação ao vivo, autorizada pelo dono a 2026-10-03) ·
  **Estado:** done · **Depende de:** I03, I04
- **Origem:** D46 · **Decisões:** D46, D47 · **Regras:** `R00-rules.md`

## Problema

A I02 a I04 acrescentam uma chamada, um campo na `Response`, um evento de stream, uma server tool
e tarifas novas. Os docs, o `AGENTS.md` e a matriz do `README` ainda não falam de imagens geradas.

## Objectivo

- **Docs:**
  - `docs/images.md` (novo, no `mkdocs.yml`): o `generate_image`, a edição, os parâmetros da D47
    por fornecedor, as imagens no turno, o custo;
  - `docs/llm.md` (o campo `images` e o evento `image`), `docs/tools.md` (a `image_generation()`),
    `docs/pricing.md` (as tarifas de imagem), `docs/model-compatibility.md` (os modelos de imagem
    e o que deu ao vivo), `docs/content.md` (se falar de imagens só à entrada);
  - `README.md` (uma coluna ou linha de imagens na matriz) e `docs/framework-overview.md`.
- **`AGENTS.md`:** há charges também no `generate_image`; uma linha por fornecedor nos
  "Provider gotchas".
- **Exemplo:** `examples/48_generate_image.py` (o número livre de hoje; o coordenador confirma ao
  aplicar), a gravar a imagem em disco.
- **`CHANGELOG`:** as linhas propostas (Added; o evento `image` como mudança visível).
- **Verificação ao vivo** (o Claude, com autorização do dono para as chamadas pagas):
  - o `generate_image` no modelo mais barato de cada fornecedor;
  - uma edição;
  - um turno do OpenAI com a tool, em stream, com parciais;
  - o custo no meter comparado com o da tabela.
- **Nota para o ai-network:** o texto para o dono lhe mandar, com a API que ficou (a E06-09 lê-a
  no registo desta frente).

## Ficheiros

- `docs/images.md` (novo), `mkdocs.yml`, `docs/llm.md`, `docs/tools.md`, `docs/pricing.md`,
  `docs/model-compatibility.md`, `docs/content.md`, `docs/framework-overview.md`, `README.md`,
  `AGENTS.md`
- `examples/48_generate_image.py` (novo), `examples/README.md`
- `scripts/model_probe_models.toml` e `scripts/model_probe_notes.md` (se a matriz ganhar um
  cenário de imagem)

## Critério

- Os docs dizem o que o código faz. O `uv run mkdocs build --strict` passa, se o extra `docs`
  estiver instalado.
- A verificação ao vivo fica registada aqui, com o custo.

## Registo do dono

- Estado: done (Claude, 2026-10-03).
- **Docs:**
  - `docs/images.md` (novo, no `mkdocs.yml`): a chamada, os modelos, as opções por fornecedor e
    o tamanho portável, o custo, as imagens no turno, o stream e a edição no turno seguinte;
  - `docs/llm.md`: `Response.images`, o evento `image`, os contadores de imagem e `Usage` com `+`;
  - `docs/tools.md`: a `image_generation()`;
  - `docs/pricing.md`: as tarifas de imagem;
  - `docs/content.md`: os `data:` URLs e como devolver uma imagem gerada;
  - `docs/model-compatibility.md`: uma secção "Image Models" com o que correu ao vivo, e o Gemini
    e o xAI nas lacunas;
  - `docs/framework-overview.md`, `README.md` (uma linha na matriz), `docs/examples.md`.
- **`AGENTS.md`:**
  - o charge site do `generate_image`;
  - o `prepare_image` no contrato;
  - os prefixos novos;
  - uma frase por fornecedor nos "Provider gotchas".
- **Exemplo:** `examples/48_generate_image.py` (gerar, editar, desenhar no turno), corrido ao vivo
  a 2026-10-03 ($0.0034 + $0.0122 + $0.0077).
- **`CHANGELOG`:** Added (o `generate_image`, a tool, as imagens no `complete` do Gemini e da
  Meta), Changed (o evento `image`, as recusas entre modelos de chat e de imagem), Fixed (o tamanho
  dos pedidos com imagens, o `data:` URL no Gemini).
- **Verificação ao vivo da frente** (I01, I03, I04 e o exemplo): cerca de $0.45 no total. O Gemini
  e o xAI ficam por verificar: o projecto Gemini não tem faturação, e a conta xAI não tem créditos.
- **Não corrido:** `uv run mkdocs build --strict` (o extra `docs` não está instalado).
- Gate: 6217 passed, 42 skipped; ruff, formatação e pyright limpos.
- **Nota para o ai-network:** no relatório final ao dono, para ele a mandar.
