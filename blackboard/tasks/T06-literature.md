# T06 · Literatura e identificadores

- **Dono:** Claude (um agente, 2026-10-08) · **Estado:** done · **Depende de:** T05 (o modelo a seguir)
- **Módulos:** `_arxiv` (2 tools), `_crossref` (2), `_datacite` (2), `_europe_pmc` (3), `_pubmed` (2),
  `_semantic_scholar` (3), `_ror` (2), `_nvd` (2): 18 tools
- **Origem:** os anexos A a E do plano para estes módulos · **Decisões:** D37 a D42 · **Regras:**
  `T00-rules.md`
- **Já feito:** `c259e0b` (o `ERROR` do ESearch do PubMed) e `6a34668` (a entrada de erro do arXiv,
  num 400 ou num 200, é a falha e não um artigo).

## O que está partido

- **Pesquisas às cegas:** `arxiv_search`, `crossref_search`, `datacite_search`, `pubmed_search`,
  `semantic_scholar_search`, `semantic_scholar_citations` e `nvd_cve_search` paginam, mas não dizem
  o total nem se há mais. O `europe_pmc_citations` diz o total, mas não tem parâmetro de página,
  embora a API o tenha.
- **Registos cortados, sem caminho para o resto:**
  - resumos a 700, 900 ou 1200 caracteres, conforme a tool (`arxiv_paper`, `crossref_work`,
    `datacite_doi`, `europe_pmc_article`, `pubmed_article`, `semantic_scholar_paper`, `nvd_cve`);
  - listas cortadas em silêncio: autores, referências, ligações, MeSH, CPEs, nomes alternativos;
  - o `ror_organization` fica só com a primeira localização;
  - o `semantic_scholar_citations` fica com o primeiro contexto de cada citação, a 260 caracteres.
- **Promessas:**
  - a paginação do `ror_search` está partida: o `max_results` corta do nosso lado a página fixa de 20
    da ROR, e as linhas 6 a 20 de cada página nunca se alcançam;
  - o `semantic_scholar_search` diz aceitar "DOI", mas é uma pesquisa por palavras;
  - o `datacite_search` põe o `resource_type` em minúsculas no `resource-type-id`, o que parte os
    tipos de várias palavras (suspeita);
  - o `nvd_cve_search` não diz o limite de 120 dias da NVD para intervalos de datas, e o motivo do
    404 que daí vem perde-se.
- **Saída:** a pontuação CVSS da NVD aparece sem a versão.

## Objectivo

As 18 tools cumprem o contrato de `T00-rules.md` e saem da lista de dívida.

- Um registo dá os seus campos inteiros, pela janela.
- Uma lista dentro de um registo diz quantos itens faltam e como os ler.
- Todas as pesquisas dizem o total ou "more available".

## Passos

1. Nota de desenho por módulo: parâmetros de continuação, limites (`Range`), falhas tipadas e
   leitores de erro da porta. Confirma na documentação de cada API como pagina e como sinaliza
   erros, e escreve os URLs.
2. Testes primeiro, por módulo, a partir de respostas construídas com a documentação.
3. A migração: janela, limites, tipos e saída; os `_truncate`/`_trim` destes módulos desaparecem.
4. `docs/tools-catalog.md` e `CHANGELOG`, com as mudanças visíveis e as quebras.

## Ficheiros

Os 8 módulos, os seus testes em `tests/toolkit/`, `tests/toolkit/contract_debt.py` (só as linhas
destas tools), `tests/toolkit/error_bodies.py` (as entradas destas fontes), `docs/tools-catalog.md`.

## Prova

- A invariante de contrato passa nas 18 tools, que saem da lista de dívida.
- Há testes de comportamento para cada item acima, incluindo a paginação do `ror_search` e o erro do
  arXiv.
- **Ao vivo, pelo dono:** os comandos que escreveres no fim da ficha.

## Registo do dono

- **Estado:** done. Feito por agentes em worktrees, juntado e revisto pelo coordenador; uma revisão independente por parte, e as correcções dela (secção "Correcções da revisão").
- **Gate final, no checkout principal com todas as fichas da vaga:** 8195 passed, 118 skipped (os ao vivo e os 59 do nanope à espera); ruff, formatação, pyright e `uv lock --check` limpos (2026-10-08).
- **CHANGELOG:** as linhas entraram em `[Unreleased]` (Upgrade notes, Added, Changed, Fixed).

- **Worktree:** `/Users/rge/Documents/dev/pessoal/ai-arch-toolkit/ai-arch-toolkit/.claude/worktrees/agent-a1e5c9071ef1d95eb`
- **Estado:** review (feito na worktree, por commitar). As 18 tools cumprem o contrato e saem da
  lista de dívida.

### Nota de desenho

Regras comuns aos oito módulos:

- **Devolvem sempre `ToolResult`**, como a família wiki: um zero resultados é
  `ToolResult.success`, uma página ou um registo é `window.result(heading=...)`.
- **Pesquisas:** `list_window` com o total da fonte e a chamada seguinte no parâmetro de
  continuação que a fonte usa (`start`, `page` ou `cursor_mark` + `offset`). Cada fonte tem um
  fundo; passado ele, o rodapé diz "the rest cannot be read here", e o parâmetro de
  continuação tem esse fundo no `Range`. A última página dentro do fundo pede só o que resta
  (PubMed, Semantic Scholar), sem `min`/`max`.
- **Registos:** o texto inteiro (todos os autores, referências, ligações, MeSH, CPEs, locais) pela
  janela, com `offset` (`Range(0)`) e `max_chars` (`Range(500, 20000)`, 6000 por omissão, como o
  `wiki_read`). O resumo vem antes das listas longas, para que a primeira janela o mostre mesmo
  num artigo com milhares de autores.
- **Listas dentro de um resultado de pesquisa:** os primeiros 8 nomes e "(+N more;
  arxiv_paper('…') lists all)": diz quantos faltam e a chamada exacta que os dá todos. As listas
  que a pesquisa cortava em silêncio (licenças, ligações, assuntos, relacionados) saem da
  pesquisa; o registo dá-as inteiras.
- **Paginação por número de página** (DataCite, citações do Europe PMC): o `next_call` leva
  `page` e `max_results`, porque a página se conta em páginas desse tamanho.
- **Casa comum:** `toolkit/tools/_records.py` (novo, 70 linhas): `record()` (texto → janela),
  `names()` (os primeiros nomes e quantos faltam), `call()` (a chamada do registo, vazia sem
  ID), `doi_of()` (o DOI tal como as fontes o
  aceitam; substitui as cópias do `_normalize_doi` no Crossref e no DataCite e o ramo DOI do
  Semantic Scholar) e as constantes da janela.
- **Ponto 5:** datas ISO 8601 (Crossref `date-parts`, PubMed com o mês por extenso, NVD ao
  segundo), sinalizadores Y/N do Europe PMC como yes/no, coordenadas do ROR em graus decimais
  sem notação científica, CVSS com a versão.

Por módulo:

- **`_arxiv`** (https://info.arxiv.org/help/api/user-manual.html, secções "Paging", "Errors" e
  "Atom"): o feed traz `opensearch:totalResults`; pesquisa até ao resultado 30 000
  (`start: Range(0, 29999)`). O leitor de erros da D38 fica (entrada de erro num 400, ou num
  200 dentro do `parse`). Uma entrada sem título nem resumo não é um artigo (`not_found`). O
  manual pede 3 s entre pedidos: `min_interval_s=3.0`. A pesquisa mostra o resumo inteiro de cada
  artigo (os resumos do arXiv têm no máximo 1920 caracteres; uma página de 20 cabe em ~45 000).
  O `arxiv_paper` dá os autores com as afiliações (`arxiv:affiliation`) e as categorias.
- **`_crossref`** (https://github.com/CrossRef/rest-api-doc: "Offsets for /works are limited to
  10K", `rows` até 1000; a fonte do serviço, https://github.com/CrossRef/cayenne,
  `api/v1/query.clj` `max-offset 10000` e `api/v1/validate.clj`): `total-results`;
  `start: Range(0, 10000)`. Leitor novo: um 400 `{"status": "failed", "message-type":
  "validation-failure", "message": [{type, value, message}]}` é `validation_error` com as
  palavras do validador. O registo dá autores com afiliações e ORCID, licenças (versão, início),
  ligações (tipo de conteúdo, aplicação) e todas as referências; quando o Crossref tem
  `reference-count` mas não a lista, di-lo.
- **`_datacite`** (https://support.datacite.org/docs/pagination: `page[number]` até aos 10 000
  registos, `page[size]` até 1000, `meta.total`/`totalPages`;
  https://support.datacite.org/docs/api-queries: `resource-type-id` em kebab case,
  "output-management-plan"; https://support.datacite.org/docs/api-error-codes e
  https://jsonapi.org/format/#error-objects): `page: Range(1, 10000)`, e uma página para lá dos
  10 000 registos é `validation_error` antes do pedido. **Suspeita confirmada:** o
  `.lower()` dava `journalarticle`; agora `JournalArticle`, `Journal Article` ou
  `journal-article` vão como `journal-article`. Leitor novo para o `errors` do JSON:API (400 é
  `validation_error`). O registo dá todos os títulos (com o tipo), criadores com afiliações,
  descrições pelo tipo (Abstract, Methods…), assuntos, direitos (com o URI) e identificadores
  relacionados (relação, identificador, tipo).
- **`_europe_pmc`** (https://europepmc.org/RestfulWebService; ver "Bloqueios"): pesquisa por
  cursor com `hitCount`; o `europe_pmc_search` ganha `offset` (só para numerar a página do
  cursor, como pede o `contract_cases.py`) e o rodapé dá `cursor_mark` e `offset`. O
  `europe_pmc_citations` ganha `page` (`/{source}/{id}/citations?page=&pageSize=`), com o
  `hitCount` como total; `max_results: Range(1, 25)` (25 é a página por omissão da fonte). A
  lista de citações responde a um registo que não existe como a um que ninguém cita, por isso,
  com zero citações na página 1, um pedido `SRC:… AND EXT_ID:…` (`resultType=idlist`) decide
  entre `not_found` e "No articles in Europe PMC cite …". Leitor novo: `errCode`/`errMsg` num
  200 é o erro (antes lia-se como sem resultados). O `europe_pmc_article` aceita `SOURCE/ID`
  (o id que a pesquisa mostra) e diz quando o mesmo ID existe em várias fontes.
- **`_pubmed`** (https://www.nlm.nih.gov/pubs/techbull/so22/so22_updated_pubmed_e_utilities.html:
  `retstart + retmax <= 10,000`; https://www.ncbi.nlm.nih.gov/books/NBK25499/;
  https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/):
  `count` como total, `start: Range(0, 9999)`. Zero resultados dizem as frases que o PubMed não
  encontrou (`quotedphrasesnotfound`, `phrasesnotfound`). Um PMID que o EFetch não devolve
  fica no seu lugar, com a chamada para o tentar. O EFetch que responde `eFetchResult/ERROR` é
  `upstream` com as palavras dele (antes, "no PubMed article"). O registo dá o PMCID (que o
  `europe_pmc_article` aceita), o resumo por secção (BACKGROUND, METHODS…), autores com
  afiliações, MeSH com tópico principal e qualificadores.
- **`_semantic_scholar`** (https://api.semanticscholar.org/api-docs/graph; o limite de 1000
  desde 2023-10-31 em https://github.com/allenai/s2-folks/blob/main/API_RELEASE_NOTES.md; os
  esquemas `PaperSearchBatch`, `CitationBatch`, `Error400`, `Error404` como os gera
  https://github.com/novafacing/semanticscholar-rs/tree/main/docs): pesquisa com `total` e
  `next`, `offset + limit` abaixo de 1000 (`start: Range(0, 998)`); citações sem total, com
  `next`, abaixo de 10 000 (`start: Range(0, 9998)`). **Promessa:** a pesquisa é "plain-text,
  no special query syntax": um DOI ou um ID do arXiv dá `validation_error` com a chamada
  `semantic_scholar_paper('…')`; o `year` valida-se antes. Leitor novo: um 400 "Unacceptable
  query params" é `validation_error`; "Unrecognized or unsupported fields" (os campos são da
  tool) fica `upstream`. As citações mostram todos os contextos, inteiros. Os IDs externos
  saem na forma que o `semantic_scholar_paper` aceita (`DOI:…`, `ARXIV:…`, `PMID:…`,
  `CorpusId:…`). O `_normalize_paper_id` passou a uma tabela de prefixos (complexidade 16 → 9).
- **`_ror`** (https://ror.readme.io/v2/docs/rest-api: "The maximum number of results that can be
  retrieved via the API is 10,000"; https://github.com/ror-community/ror-api,
  `rorapi/settings.py` `PAGE_SIZE 20`, `MAX_PAGE 500`, e `rorapi/common/queries.py`, as palavras
  dos erros): **paginação corrigida**: sai o `max_results`, a página de 20 da ROR mostra-se
  inteira e numerada (21–40 na página 2), `page: Range(1, 500)`. A consulta aceita qualquer
  texto imprimível até 200 caracteres (o `_TEXT_RE` recusava `&`, e a própria documentação da
  ROR pesquisa "Franklin & Marshall College"). O registo dá todos os locais (com subdivisão,
  coordenadas e ID GeoNames), nomes (tipos e língua), ligações, IDs externos (o preferido
  marcado), relações e datas do registo.
- **`_nvd`** (https://nvd.nist.gov/developers/vulnerabilities: `totalResults`, `startIndex`, "The
  maximum allowable range when using any date range parameters is 120 consecutive days";
  https://nvd.nist.gov/developers/start-here e
  https://nvd.nist.gov/general/news/api-20-announcements: "examine the response header for a
  field named message"): o leitor do cabeçalho `message` já existia e está certo; o limite de
  120 dias verifica-se antes do pedido (`validation_error` que diz como partir o intervalo), e
  o motivo do 404 continua a chegar pelo leitor. CVSS com a versão: uma pontuação por versão
  na pesquisa (`CVSS 3.1: 10.0 CRITICAL | CVSS 2.0: 9.3 HIGH`, a do avaliador principal), e no
  registo todas, com avaliador e vector. CPEs com "vulnerable" e o intervalo de versões;
  referências com as etiquetas; fraquezas com quem as nomeou.

### Fusões e nomes tirados

Nenhuma fusão nem nome tirado: as 18 tools mantêm o nome. Uma quebra de assinatura:

- `ror_search` perde `max_results` (a ROR não tem tamanho de página; o parâmetro só cortava do
  nosso lado).

Parâmetros novos (compatíveis): `offset` e `max_chars` nos oito registos; `page` no
`europe_pmc_citations`; `offset` no `europe_pmc_search`. Limites que passam a ser recusados em vez
de ajustados: `max_results` fora de 1–20 (1–25 nas citações do Europe PMC) e os fundos de cada
fonte acima. O `nanope` não importa nenhuma destas tools (procurei em
`src/ai_arch_toolkit/nanope`, só leitura).

### Mudanças em ficheiros partilhados

- `core/`, `_http.py`, `_window.py`: nenhuma.
- Novo `src/ai_arch_toolkit/toolkit/tools/_records.py` (só usado pelos oito módulos).
- Novo `tests/toolkit/literature_answers.py`: as respostas das oito fontes, para os testes e
  para os casos de contrato.
- `tests/toolkit/contract_cases.py`: `import literature_answers as lit` e as minhas entradas
  (janela 18, `not_found` 5 novas, zero 8), no fim de cada dicionário.
- `tests/toolkit/error_bodies.py`: `import literature_answers as lit` e as entradas `_arxiv`,
  `_crossref`, `_datacite`, `_europe_pmc`, `_pubmed`, `_semantic_scholar`, `_ror`, `_nvd`, no
  início do dicionário.
- `tests/toolkit/contract_debt.py`: 18 linhas apagadas; `CUT_HELPERS` 14 → 7.
- `tests/quality_baseline.json`: 11 entradas apagadas (os `_format_*` dos seis módulos e o
  `_normalize_paper_id`); 39 → 28.
- `docs/tools-catalog.md`: as secções "Papers", "Academic graph & metadata" e "Security". **Para o
  coordenador:** a frase do topo do catálogo ("The wiki family declares its limits as Range
  bounds … the other tools still move a numeric argument …") e o passo 4 do
  `CONTRIBUTING.md` ("the existing tools, the templates included, clamp") ficam por actualizar
  quando as T06 a T09 estiverem juntas.

### Prova

Os testes novos, pelos itens da ficha:

| Item da ficha | Testes |
|---|---|
| Pesquisas às cegas (total ou "more") | `test_a_page_says_the_total_and_the_next_start` (arXiv, Crossref, PubMed, S2, NVD), `test_a_page_says_the_total_and_the_next_page` (DataCite), `..._next_cursor_with_the_position` (Europe PMC), `test_a_page_says_there_is_more_and_the_next_start` (citações S2), `test_a_page_of_citations_says_the_total_and_the_next_page` (Europe PMC); os fundos: `test_past_crossrefs_offset_limit…`, `test_past_the_10000_records…` (DataCite), `test_esearch_reaches_the_first_10000_records`, `test_relevance_search_reaches_the_first_1000_results`, `test_past_rors_10000_results…` |
| `europe_pmc_citations` sem página | `test_a_page_of_citations_says_the_total_and_the_next_page`, `test_a_record_nobody_cites_says_so`, `test_a_record_europe_pmc_does_not_have_is_not_found` |
| Resumos cortados a 700/900/1200 | `test_a_result_shows_the_whole_summary…` (arXiv), `test_the_record_is_whole*` e `test_a_long_*_reads_on_through_the_window` nos oito registos |
| Listas cortadas em silêncio | `test_a_long_author_list_says_how_many_more_and_where` (5 módulos), `test_long_creator_lists…` (DataCite), `test_the_record_is_whole*` (todos os autores, referências, ligações, MeSH, CPEs, nomes) |
| ROR só com a primeira localização | `test_a_row_shows_every_location…`, `TestRorOrganization::test_the_record_is_whole` |
| Paginação do `ror_search` | `test_every_row_of_rors_page_shows_and_the_next_page_follows`, `test_the_page_limits_are_the_schemas` |
| Primeiro contexto a 260 | `test_a_page_says_there_is_more_and_the_next_start` (S2: os três contextos, inteiros) |
| `semantic_scholar_search` promete DOI | `test_an_identifier_is_looked_up_with_the_paper_tool_not_searched` |
| `resource_type` em minúsculas | `test_a_resource_type_goes_as_the_filter_takes_it` (5 formas) |
| 120 dias da NVD e o motivo do 404 | `test_a_range_over_nvds_120_days_is_refused_before_the_request`, `test_a_refused_request_says_why_from_the_message_header` |
| CVSS sem versão | `test_a_page_says_the_total…` (NVD), `test_the_record_is_whole_every_cvss_with_its_version` |
| Erro do arXiv | `test_a_query_arxiv_cannot_read_is_explained`, `test_an_error_entry_is_the_error_not_a_paper` |
| Limites na assinatura | `test_the_limits_are_the_schemas*` (pelo executor) em todas as pesquisas |
| Erros novos das fontes | `test_a_refused_filter_says_crossrefs_reason`, `test_a_refused_request_says_datacites_reason`, `test_an_error_in_place_of_the_result_is_the_error` (Europe PMC), `test_an_error_efetch_reports_is_the_error…`, `test_an_unacceptable_parameter_is_the_callers_to_fix` (S2), `test_a_refused_filter_says_rors_reason` |
| Contrato | as 18 tools passam `test_each_tool_keeps_the_contract…` sem linha na dívida |
| Revisão | `test_citations_past_the_apis_reach…`, `test_a_paper_without_an_id…` (S2), `test_an_empty_page_inside_the_total…`, `test_the_last_page_within_30000…` (arXiv), `test_without_totalpages…` (DataCite), `test_a_null_list_in_one_work…`, `test_a_work_without_a_doi…` (Crossref), `test_a_ror_id_is_taken_with_any_url_form`, `test_a_publication_date_range_alone_is_a_search` (NVD) |

Cada ficheiro de teste novo correu primeiro contra o módulo antigo e falhou pela razão certa
(o texto, o tipo de retorno, os parâmetros novos), excepto o do DataCite, que escrevi antes do
módulo mas só corri depois.

### Ao vivo, pelo dono

APIs gratuitas e sem chave (o Semantic Scholar sem chave partilha um limite que muitas vezes
está gasto: com `SEMANTIC_SCHOLAR_API_KEY` vai melhor). Cada comando faz um ou dois pedidos.

```bash
uv run python -c "from ai_arch_toolkit.toolkit.tools import arxiv_search; print(arxiv_search('LLM agents', max_results=3).value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import arxiv_paper; print(arxiv_paper('1706.03762').value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import arxiv_paper; print(arxiv_paper('2501.99999'))"
uv run python -c "from ai_arch_toolkit.toolkit.tools import crossref_search; print(crossref_search('deep learning', max_results=3).value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import crossref_work; print(crossref_work('10.1038/nature14539', max_chars=3000).value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import crossref_search; crossref_search('x', type_filter='journal')"
uv run python -c "from ai_arch_toolkit.toolkit.tools import datacite_search; print(datacite_search('climate', resource_type='JournalArticle', max_results=3).value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import datacite_doi; print(datacite_doi('10.5061/dryad.8515').value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import europe_pmc_search; print(europe_pmc_search('deep learning', max_results=2).value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import europe_pmc_citations; print(europe_pmc_citations('MED', '26017442', max_results=5, page=2).value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import europe_pmc_citations; europe_pmc_citations('MED', '99999999999')"
uv run python -c "from ai_arch_toolkit.toolkit.tools import pubmed_search; print(pubmed_search('deep learning', max_results=3).value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import pubmed_search; print(pubmed_search('cancer', max_results=5, start=9998).value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import pubmed_article; print(pubmed_article('26017442').value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import semantic_scholar_search; print(semantic_scholar_search('attention is all you need', max_results=3).value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import semantic_scholar_citations; print(semantic_scholar_citations('ARXIV:1706.03762', max_results=2).value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import ror_search; print(ror_search('university', page=2).value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import ror_search; print(ror_search('Franklin & Marshall College').value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import ror_organization; print(ror_organization('01c27hj86').value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import nvd_cve; print(nvd_cve('CVE-2021-44228', max_chars=4000).value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import nvd_cve_search; print(nvd_cve_search(query='log4j', pub_start_date='2021-12-01', pub_end_date='2022-03-30', max_results=3).value)"
```

E três respostas cruas, para confirmar o que a documentação não deixou ler daqui:

```bash
## Europe PMC: o erro num 200 ({"errCode": …, "errMsg": …}).
curl -s 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=cancer&format=json&pageSize=1001'
## NVD: o cabeçalho message de um intervalo de 121 dias (um 404).
curl -s -D - -o /dev/null 'https://services.nvd.nist.gov/rest/json/cves/2.0?pubStartDate=2021-01-01T00:00:00.000&pubEndDate=2021-05-01T23:59:59.999'
## Europe PMC: as citações de um registo que não existe (espera-se hitCount 0, não um erro).
curl -s 'https://www.ebi.ac.uk/europepmc/webservices/rest/MED/99999999999/citations?format=json&pageSize=1'
```

O que confirmar:

- os totais e os rodapés (`[results 1-3 of … | next: start=3]`); a página 2 do ROR numerada 21–40;
- o `datacite_search(resource_type='JournalArticle')` com resultados (a suspeita do `.lower()`);
- o `europe_pmc_citations(page=2)` numerado a partir de 6, e o registo inexistente como
  `not_found`;
- o `pubmed_search(start=9998)` com dois resultados e "the rest cannot be read here";
- a forma do `errMsg` do Europe PMC e o texto do cabeçalho `message` da NVD;
- se um intervalo de exactamente 120 dias (2021-12-01 a 2022-03-30) passa na NVD; se não passar,
  a contagem é de diferença e não de dias incluídos, e o limite desce um dia;
- se os carimbos da NVD são UTC (a resposta não traz fuso; mostram-se como vêm, ao segundo).

### Bloqueios e achados

- **Documentação que não se deixou ler daqui:** `europepmc.org/RestfulWebService` e o PDF de
  referência respondem 403; os livros da NCBI pedem um captcha; as páginas de
  `nvd.nist.gov/developers` chegam vazias; `api.semanticscholar.org/api-docs` é uma aplicação
  JavaScript. Usei fontes oficiais equivalentes quando as havia (o código da ROR e do
  Crossref, as notas de versão do Semantic Scholar e o seu esquema gerado, o boletim da NLM,
  o NCBI Insights) e cópias da documentação da NVD (o texto dos 120 dias, igual no nvdlib e no
  pypi `nvd-api`). Para o Europe PMC, a forma `errCode`/`errMsg` vem de uma gravação de
  terceiros (K-Dense-AI/scientific-agent-skills) e o `page` das citações do cliente R da
  rOpenSci (`epmc_citations`, `utils.r`): a confirmar ao vivo, acima.
- **Palavras de erro que a documentação não dá:** o cabeçalho `message` da NVD e o `title` 400
  do DataCite ("Bad Request", da página de códigos) são palavras que os testes põem; o contrato
  prova que chegam ao modelo, não o texto exacto da fonte.
- **Saldo de linhas:** os oito módulos tinham 2954 linhas e têm 3541, mais 70 do `_records.py`
  (+657). A R00 pede saldo negativo nos módulos redesenhados; aqui não é: entram os totais, a
  continuação, os registos inteiros (que antes não se liam), cinco leitores de erros novos e a
  verificação do Europe PMC. Saem as sete cópias do `_truncate`, as duas do `_normalize_doi` e
  11 entradas da dívida de complexidade.
- **Revisão adversarial (um agente, só leitura):** 2 médias e 10 baixas, todas corrigidas, com
  testes escritos depois da correcção (não os vi falhar antes):
  - média: nas citações do Semantic Scholar, passado o fundo de 10 000, o rodapé dizia "end"
    com mais citações por ler; agora pede o `citationCount` do artigo e diz "of N | the rest
    cannot be read here";
  - média: a docstring do `europe_pmc_search` prometia resumos com `result_type="core"`, que a
    lista não mostra; a docstring diz agora o que acontece;
  - baixas: uma página vazia do arXiv dentro do total era um falso fim (agora `upstream`
    com repetição, e o `start` seguinte conta as entradas, não os artigos lidos); o arXiv pede
    só o que resta antes dos 30 000; o DataCite sem `totalPages` decide pelo total; o
    `next_call` paginado por `page` leva o `max_results` (DataCite, citações do Europe PMC);
    as mensagens de erro do EFetch e do `errMsg` do Europe PMC ganham o passo seguinte; o
    teste do EFetch usava um erro que a tool nunca provoca ("Empty id list"); "lists all"
    sem ID nomeava uma chamada que falhava; os IDs externos do Semantic Scholar que nenhuma
    tool aceita (DBLP) saem (ponto 6); um `null` numa lista de um trabalho do Crossref deitava
    abaixo a pesquisa inteira; o `ror_organization` aceita `http://ror.org/…` e `ror.org/…`; o
    `nvd_cve_search` aceita só um intervalo de datas como filtro (a NVD aceita).
  - **Fica por fazer (achado da revisão, fora do âmbito):** o `window_kept` do
    `test_tool_contract.py` serve as respostas pela ordem, seja qual for o pedido: uma próxima
    chamada que relesse a mesma página passava a verificação. Só os testes de cada módulo
    (os parâmetros enviados) o apanham. Proposta para `FINDINGS.md`.
  - **Ponto 5, parcial:** as categorias do arXiv (`cs.CL`) e as fontes do Europe PMC (`MED`)
    saem sem rótulo, porque a resposta não o traz (a T00 pede o rótulo "que a resposta já
    traz").
- **Fora do âmbito (para `FINDINGS.md`, pelo coordenador):** a frase do topo de
  `docs/tools-catalog.md` e o passo 4 do `CONTRIBUTING.md` dizem que as tools fora da família
  wiki ajustam os limites em silêncio; deixam de ser verdade à medida que as T06 a T09 entram.
- **Decisão minha:** o `arxiv_search` mostra os resumos inteiros (era o que a pesquisa do arXiv
  tinha de útil, cortado a 700); as outras pesquisas continuam sem resumo. Uma página de 20 do
  arXiv pode chegar aos ~45 000 caracteres; a omissão é 5.
- **Decisão minha:** no Semantic Scholar, só o "Unacceptable query params" é `validation_error`;
  "Unrecognized or unsupported fields" diz respeito aos campos que a tool pede, que o modelo não
  pode corrigir, e fica `upstream`.
