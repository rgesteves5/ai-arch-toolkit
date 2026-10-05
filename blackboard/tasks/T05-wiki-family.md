# T05 · Família wiki: MediaWiki, Wikipedia e definições

- **Dono:** Claude (2026-10-05) · **Estado:** done · **Depende de:** T04b
- **Módulos:** `_mediawiki` (4 tools), `_wikipedia` (3), `_dictionary` (1)
- **Origem:** plano, secções 1, 3.5 e 3.6, e os anexos A a E destes módulos · **Decisões:** D39,
  D40, D41 · **Regras:** `T00-rules.md`
- **Já feito:** `6a34668`: o `wikipedia_related` diz porque pesquisa (não há página, ou a página não
  tem ligações), em vez de trocar em silêncio.

## Problema

É o caso que abriu a frente.

- O `mediawiki_page` corta aos 4000 caracteres, sem secção nem posição.
- O `mediawiki_sections` devolve índices que nenhuma tool aceita.
- O `wikipedia_article` aceita `max_chars` até 100 000, mas pede só a introdução (`exintro`).
- O wikitexto é limpo por uma regex de uma só passagem: ficam predefinições, `|-`, `rowspan` e
  `80px`.
- O `_TEXT_RE` recusa `&`, `?`, `!` e `"`, por isso "AT&T" não passa.
- As tools MediaWiki usam o Wiktionary por omissão.
- Há três tools de definições e duas famílias para a Wikipedia.

## Objectivo

Uma família wiki que responde às duas perguntas das conversas em uma ou duas chamadas:

- **"Qual foi o motivo do Nobel da Física de 1960?"** Um `find="1960"` na página da lista devolve a
  linha da tabela com o motivo.
- **"O que diz o Wikibooks sobre personagens, estilo, edição e publicação?"** O índice, com o tamanho
  de cada secção, e a leitura de cada uma.

## Desenho (decide-o na nota de desenho, dentro de D39 a D41)

- **Tools propostas:**
  - `wiki_search(query, wiki, offset)`;
  - `wiki_outline(title, wiki)`: as secções, com índice e tamanho;
  - `wiki_read(title, wiki, section, offset, find, max_chars)`;
  - `wiki_summary(title, wiki)`: a introdução;
  - `wiktionary_entry(term, language)`.

  O `wiki` é a `en.wikipedia` por omissão, ou outro host Wikimedia permitido (o `_DOMAINS` actual).
  As tools antigas (`mediawiki_*`, `wikipedia_*`, `define_word`) saem, com a tabela de migração.
- **Leitura:** `action=parse&prop=text|sections`, com `section=N`, `disableeditsection=1`,
  `disabletoc=1` e `disablelimitreport=1`, convertida pelo `html.parser` da stdlib.
  - Ficam parágrafos, listas, títulos e tabelas em linhas, com `rowspan` e `colspan` resolvidos.
  - Saem `<style>`, `<script>`, as referências, as ligações de edição, as caixas de navegação e o
    que está escondido. Confirma as classes na documentação da MediaWiki e escreve os URLs.
- **`define_word`:** sai. A fonte (dictionaryapi.dev) responde 522, e o Wiktionary cobre as
  definições.
- **Títulos:** aceita o que a API aceita. Um título inválido dá `validation_error`, com o
  `invalidreason` da API (`c259e0b` já o lê); uma página que não existe dá `not_found`, com "search
  with `wiki_search`".
- **Quem usa os nomes antigos:** os exemplos 10, 12, 15, 19 e 24 (actualizam-se nesta ficha) e o
  `nanope` (ver T00: pergunta ao dono).

## Nota de desenho (2026-10-05)

- **Tools finais:**
  - `wiki_search(query, wiki, max_results, offset)`;
  - `wiki_outline(title, wiki, offset)`: as secções, com o número que o `wiki_read` aceita e o
    tamanho (com as subsecções: o que o `wiki_read` devolve, ao carácter);
  - `wiki_read(title, wiki, section, find, offset, max_chars)`;
  - `wiktionary_entry(term, language, offset, max_chars)`.

  Sai o `wiki_summary` da proposta: `wiki_read(section=0)` é a introdução, e uma tool a mais
  para o mesmo trabalho contraria a D41. O `wikipedia_related` sai sem tool própria: a pesquisa
  da wiki tem `morelike:Título`, que o `wiki_search` documenta.
- **Módulos:**
  - `_wiki.py` (as quatro tools);
  - `_wiki_html.py` (o conversor);
  - `_mediawiki.py` fica só com o que as wikis partilham: o `mediawiki_error` (agora também
    `nosuchsection` como `validation_error`), os domínios Wikimedia e o `wiki_api(host)`;
  - saem `_wikipedia.py` e `_dictionary.py`.
- **`wiki`:** o host (`"en.wikibooks.org"`), ou o seu URL; a `en.wikipedia.org` por omissão. Fora
  dos domínios Wikimedia, `validation_error` antes do pedido. Nas duas wikis das tools (Wikipedia
  e Wiktionary em inglês) um 404 é um endpoint que mudou; noutro host, o URL de quem chama
  (`Api.within`, T02).
- **Leitura:** `action=parse` com `prop=text`, `redirects=1`, `disableeditsection=1`,
  `disabletoc=1`, `disablelimitreport=1` e `formatversion=2` (o `text` vem em string). Tudo se lê
  dentro do `parse=` da porta.
- **Secções pela posição, não pedidas à wiki** (pela investigação de 2026-10-05, com as fontes no
  rascunho da sessão): o `prop=sections` está obsoleto desde a 1.46; um `section=N` é analisado de
  novo pela wiki, enquanto a página inteira vem da cache dela; e as secções de predefinições
  (`T-N`) não se pedem. Por isso o `wiki_read(section=N)` lê a página inteira e corta a secção do
  texto convertido, pela N-ésima heading; o tamanho no `wiki_outline` é o mesmo corte. O
  `wiktionary_entry` corta a secção da língua (o `h2` com o nome dela) numa só chamada.
- **Cortesia:** no máximo 4 pedidos por segundo por wiki (`min_interval_s`, com o `Api.within` a
  aceitá-lo), a `User-Agent` da porta, o 429 com `Retry-After` da T02. A política de robôs da
  Wikimedia desaconselha o Action API para o HTML de páginas em escala; para um agente que lê
  algumas páginas, fica o `action=parse`, que é o que dá os redirects e o texto de uma vez. Se a
  Wikimedia apertar, a troca é o `rest.php/v1/page/{title}/html` (Parsoid), que o conversor já
  lê.
- **Conversor (`html.parser` da stdlib):**
  - títulos `## Título` com o nível, a âncora e a posição no texto (a marcação nova, `div.mw-heading`
    com o `id` no `hN`, e a antiga, `span.mw-headline`);
  - parágrafos, listas (`- `, `1. `, com o `start` e a profundidade) e listas de definição numa
    linha cada;
  - tabelas linha a linha, células com ` | `: a que ocupa várias linhas repete-se em cada uma, a
    que ocupa várias colunas deixa as outras vazias; legenda como `Table: …`; uma tabela dentro de
    uma célula fica nessa célula; spans absurdos limitados;
  - fórmulas pelo TeX (`alttext`, ou o `alt` da imagem de recurso);
  - saem os marcadores e as listas de referências, as ligações de edição, o índice (`#toc`, com o
    seu `h2`), as caixas de navegação (também pelo `role="navigation"`), os avisos de manutenção,
    o texto de erro, o escondido e o que a wiki deixa de fora das vistas móveis e dos excertos
    (`display:none`, `noprint`, `nomobile`, `noexcerpt`, `sortkey`), `style`/`script`/`link` e
    `audio`/`video`. As hatnotes ficam: dizem que página ler a seguir;
  - lê o HTML do analisador antigo, que o `action=parse` devolve, e o do Parsoid (`<section>`,
    `sup.mw-ref`), para quando a Wikimedia mudar.
- **Janela:** o `Window` ganha `heading=` em `text()` e `result()`, a linha da tool por cima do
  corpo. As tools devolvem `window.result()`; a invariante e o contrato lêem o texto de um
  `ToolResult` de sucesso.
- **Títulos:** o que a API aceita; antes do pedido, só o vazio e o que passa os 255 bytes. Um
  título que a API recusa é `validation_error` com o motivo dela; uma página que não existe é
  `not_found`, com "find the exact title with `wiki_search`" (com o `wiki=` quando não é a
  Wikipedia).
- **Contrato:** as quatro tools cumprem os cinco pontos e saem da dívida, com as oito antigas. O
  ponto dos erros da fonte passa a valer para as tools de rede salvo os módulos `SOURCELESS`
  (`_web`, `_youtube`), porque o `Api` das wikis vive no `_mediawiki`. A invariante da capacidade
  segue as funções importadas de outro módulo de tools.
- **nanope (dono, 2026-10-05):** a T05 tira os nomes; o nanope fica por actualizar pelo dono. Os
  testes de `tests/nanope` que constroem as suas tools saltam até lá
  (`tests/nanope/pending.py`) e voltam a correr sozinhos quando o nanope as construir.

## Passos

1. Nota de desenho, com a lista final de tools e a tabela de migração.
2. O conversor de HTML, com testes sobre fixtures construídas a partir de HTML real da MediaWiki.
3. As tools, a janela, os limites e as falhas tipadas.
4. Exemplos, `docs/tools-catalog.md`, `docs/tools.md`, `CONTRIBUTING.md` (a família wiki passa a ser
   o modelo) e `CHANGELOG` (**Breaking**, com a tabela de migração).
5. As 8 linhas da lista de dívida saem.

## Ficheiros

`toolkit/tools/_mediawiki.py`, `_wikipedia.py` e `_dictionary.py` (ou um `_wiki.py` que os
substitua), `toolkit/tools/__init__.py`, os seus testes em `tests/toolkit/`,
`tests/toolkit/contract_debt.py`, os exemplos 10, 12, 15, 19 e 24, `docs/tools-catalog.md`,
`docs/tools.md`, `CONTRIBUTING.md`.

## Prova

- As duas perguntas das conversas como testes:
  - a linha de 1960 aparece com o motivo, numa chamada `find`;
  - o Wikibooks lê-se secção a secção.
- A tabela com `rowspan` sai em linhas completas.
- Página inexistente, título inválido e zero resultados dão o que o contrato diz.
- Nenhuma tool da família fica na lista de dívida.
- **Ao vivo, pelo dono:** as duas conversas repetidas num agente, e os comandos que escreveres no fim
  da ficha.

## Riscos

- O HTML é maior do que o wikitexto, e cada leitura pede a página inteira (as secções cortam-se
  do texto): a porta aceita até ao limite da própria wiki (12 MiB, `_MAX_BYTES`), e uma página
  maior chega como aviso da wiki, que a falha cita.
- As classes CSS da MediaWiki mudam com o tempo: testa sobre fixtures e deixa ao dono a verificação
  ao vivo.

## Registo do dono

- **Estado:** done (Claude, 2026-10-05), pela nota de desenho.
- **Feito:**
  - `_wiki.py` (`wiki_search`, `wiki_outline`, `wiki_read`, `wiktionary_entry`) e
    `_wiki_html.py` (o conversor); o `_mediawiki.py` só com o partilhado; saem `_wikipedia.py`,
    `_dictionary.py` e as oito tools antigas, com a tabela de migração no `CHANGELOG`;
  - o `Window` com `heading=`; o `Api.within` com `min_interval_s`;
  - exemplos 10, 12, 15, 19 e 24; `docs/tools.md`, `docs/tools-catalog.md`,
    `docs/framework-overview.md`, `docs/agents.md`, `docs/agents-and-capabilities.md`,
    `CONTRIBUTING.md` (a família wiki passa a ser o modelo).
- **Prova:** as duas conversas são testes (a linha de 1960 numa chamada `find`; o Wikibooks secção
  a secção); a tabela com `rowspan` sai em linhas completas; página inexistente, título inválido e
  zero resultados dão o que o contrato diz; as quatro tools saem da dívida (com as oito antigas), e
  `CUT_HELPERS` desce para 14.
- **Investigação (documentação, sem pedidos às APIs):** a marcação de headings de hoje e a antiga,
  o que a MediaWiki deixa de fora dos excertos, o `formatversion=2`, o `prop=sections` obsoleto, o
  `section=N` analisado de novo pela wiki. Daí o desenho por posição. Um pedido dessa investigação
  levou o email do dono no User-Agent (a mediawiki.org); o dono foi avisado.
- **Revisão independente:** uma falha grave, no core: um `Range` atrás de um alias PEP 695
  (`type Chars = Annotated[...]`) fazia do parâmetro uma `string` sem limites. Corrigi o `_schema.py`
  (o alias e o `Annotated` lidos por um helper, que baixa a dívida de qualidade), escrevi o
  `max_chars` sem alias, e o contrato passa a falhar um `int` cujo schema não é inteiro. Corrigi as
  médias:
  - spans de tabela sem limite total (orçamento de células por tabela);
  - indentação de listas que crescia com o quadrado da profundidade (até 10 níveis).

  Corrigi também as baixas:
  - o `next:` de uma secção sem a `section`;
  - o `wiki=` na pista do esboço;
  - devolvem sempre `ToolResult`;
  - o `CHANGELOG` com as mudanças de comportamento;
  - um heading ou um `pre` numa célula, `start`/`value` nas listas, `dl` aninhados;
  - fins de tag sem par e fins de secção em tempo linear;
  - `kw_only`;
  - o ramo `nosuchsection` sem uso;
  - o teste do `Api.within(min_interval_s=)`.
- **nanope:** como o dono decidiu, ficou por actualizar; os testes que constroem as suas tools
  saltam (`tests/nanope/pending.py`), e o BOARD diz o que trocar.
- Gate: 6981 passed, 101 skipped (42 ao vivo, 59 do nanope à espera); ruff, formatação e pyright
  limpos.

### Verificação ao vivo (dono)

APIs gratuitas e sem chave; cada comando faz um ou dois pedidos à Wikimedia.

```bash
uv run python -c "from ai_arch_toolkit.toolkit.tools import wiki_read; print(wiki_read('List of Nobel laureates in Physics', find='1960').value)"
```

```bash
uv run python -c "from ai_arch_toolkit.toolkit.tools import wiki_search; print(wiki_search('Creative Writing novels', wiki='en.wikibooks.org').value)"
```

```bash
uv run python -c "from ai_arch_toolkit.toolkit.tools import wiki_outline; print(wiki_outline('Python (programming language)').value)"
```

```bash
uv run python -c "from ai_arch_toolkit.toolkit.tools import wiktionary_entry; print(wiktionary_entry('serendipity').value)"
```

O que confirmar: a linha de 1960 com o motivo; um livro do Wikibooks que se lê secção a secção
(pelo título que a pesquisa der); os tamanhos do esboço iguais ao total do `wiki_read(section=N)`;
o HTML de hoje sem lixo (navegação, referências, avisos) no texto.
