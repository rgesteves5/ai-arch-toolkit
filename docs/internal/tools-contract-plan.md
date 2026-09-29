# Contrato das tools: levantamento e plano

**Data:** 29 de setembro de 2026. **Código:** `main` em `1433016`; as linhas citadas são desse commit.
**Origem:** duas conversas do ai-network (28/09) em que um agente react com a GPT-6 Luna não chegou
à informação que as páginas tinham.
**Método:** leitura dos 44 módulos de `toolkit/tools/` (132 tools: 113 de rede, 13 de cálculo, 4 de
ficheiros, 1 de shell, 1 de Python), com as respostas das fontes simuladas na costura
`_http._open`, sem rede. Ao vivo, só as APIs gratuitas da MediaWiki e a base de dados local do
ai-network. *Confirmado*: o caminho foi reproduzido sem rede e o comportamento da fonte está
documentado. *Suspeito*: o caminho existe, mas o comportamento da fonte não foi visto ao vivo.
**Estado:** decisões D37–D42 tomadas (`blackboard/DECISIONS.md`); frente T aberta a 30/09 no
`BOARD.md`, com as fichas T00–T09. O `c259e0b` (29/09) já resolveu parte do anexo C.

## 0. Resumo

As surpresas das conversas não são da MediaWiki: nenhuma tool tem um contrato com o agente. Cada uma
decide sozinha como corta, como falha e como formata.

| Classe | Tools | Exemplo |
|---|---|---|
| A · Cortam sem saída | 94 cortam; 60 são becos sem saída | `mediawiki_page` corta a 4000 caracteres, sem secção nem posição |
| B · Prometem o que não fazem | 12 confirmadas, 7 suspeitas | `wikidata_sparql` devolve 20 linhas a uma query com `LIMIT 100` |
| C · Erro mostrado como sucesso | 19 confirmadas, 17 suspeitas | uma página que não existe volta como cabeçalho vazio, com `ok: true` |
| D · Saída por limpar | 16 | marcação de tabelas do wikitexto, datas em milissegundos epoch |
| E · Repetidas sobre a mesma fonte | 10 grupos | `wikipedia_article` e `mediawiki_page` na mesma Wikipedia |

Por cima disto:

- qualquer string de erro que uma tool devolve conta como sucesso no executor, no meter e na app
  (`core/_tools/_executor.py:240`);
- o teste de invariantes só garante que as tools não rebentam e devolvem `str`.

## 1. O caso que o revelou

| Página | Wikitexto | Texto limpo | Onde está o que interessa | A saída a 4000 acaba em |
|---|---|---|---|---|
| *List of Nobel laureates in Physics* | 102 078 | 34 651 | "1960" no carácter 13 888 | 1908 (Lippmann) |
| *Creative Writing/Novels* (Wikibooks) | 14 146 | 13 988 | "Creating an Outline" no 3 984 | "Creating an O..." |

- O agente pediu as duas páginas com `max_chars: 4000`, o tecto; a do Nobel três vezes, com os
  mesmos argumentos.
- No Wikibooks tentou contornar o corte pedindo as secções como subpáginas
  (`Creative Writing/Novels/Editing`, …). Não existem, e cada uma voltou `ok: true` só com o
  cabeçalho.
- Nas 72 chamadas a `mediawiki_page` registadas no ai-network, o agente passou sempre `max_chars`:
  39 vezes 4000 e 29 vezes 3000, nunca o valor por omissão (1200). Nas 37 a `wikipedia_article`
  pediu até 10 000, e a tool devolve só a introdução.
- A outra limitação dessas conversas (a Luna não raciocina com ferramentas porque o adaptador OpenAI
  só usa a Chat Completions) é do fornecedor e fica fora deste plano; ver o `LOG.md` de 29/09.

## 2. O contrato

O que um agente pode sempre assumir de uma tool:

1. **Sabe o que aconteceu:** resultado, zero resultados, não existe, argumento inválido ou falha da
   fonte, com o motivo da fonte e o passo seguinte.
2. **Sabe o que ficou de fora e como lá chegar:** o rodapé diz o que foi mostrado, o total e a
   chamada exacta para o resto, por exemplo `[chars 0–4000 of 34,651 · next: offset=4000]`.
3. **O schema diz a verdade:** os limites estão declarados e são validados à entrada; nada muda em
   silêncio; cada identificador que uma tool devolve é aceite por outra.
4. **Lê sem descodificar:** texto limpo, tabelas em linhas, datas ISO 8601, números com unidade,
   códigos com rótulo.
5. **Há uma tool por trabalho e por fonte.**

## 3. Causas e correcções estruturais

### 3.1 As falhas não têm tipo (D37)

- **Hoje:** as tools devolvem strings de erro, com seis ou mais formulações. O executor embrulha
  qualquer valor em `ToolResult.success` (`_executor.py:240-244`) e o meter liquida a chamada como
  sucesso. O `tool_result()` (`core/_content.py:110-126`) não tem campo de erro, e nenhum adaptador
  manda o `is_error` do Anthropic.
- **Correcção:** as tools lançam `ToolError` com um tipo de um conjunto fechado (`not_found`,
  `invalid_argument`, `upstream`, `rate_limited`) e o `retryable` certo. O executor, que já converte
  excepções em falhas sem parar o agente (`_executor.py:247`), usa esse tipo, e o erro chega a cada
  fornecedor como ele o aceita. Zero resultados é um sucesso que o diz.
- **Desaparece:** a convenção de devolver strings de erro e o `try/except HttpError` de cada tool
  (o `HttpError` passa a ser um `ToolError`).

### 3.2 A porta HTTP aceita qualquer 2xx (D38)

- **Hoje:** o `_fetch` não entrega estado nem cabeçalhos às tools. O corpo de um erro só chega à
  mensagem em 2 dos 44 módulos. Doze módulos traduzem qualquer 404 por "no matching records found.",
  e os erros que vêm dentro de um 200 viram "sem resultados".
- **Correcção:** cada `Api` declara uma vez como a sua fonte sinaliza erro (uma função sobre estado,
  cabeçalhos e corpo, corrida antes do `parse=`) e o que um 404 quer dizer em cada endpoint.
- **Desaparece:** os `status_messages={404: …}` e as guardas `isinstance` que nunca disparam
  (`_mediawiki.py:183,200,217`, `_eurostat.py:226`).

### 3.3 Cada tool corta à sua maneira (D39)

- **Hoje:** cerca de 72 cortes `[:N]` silenciosos em 22 módulos; `_truncate` copiado em 12 módulos e
  `_trim` em 3, nenhum com o tamanho original. O executor corta com "kept X of Y", mas sem
  continuação.
- **Correcção:** uma primitiva de janela para texto (caracteres) e para listas (itens), com rodapé e
  `metadata` (`truncated`, `total`, `next`). Documentos navegam-se por `section`, `offset` e `find`;
  listas por `offset` ou `cursor`, como a fonte pagina.
- **Desaparece:** os 15 helpers copiados e os cortes soltos.

### 3.4 A documentação não é contrato

- **Hoje:** limites que o código aperta sem dizer (`_eonet.py:89`, `_eurostat.py:167`), parâmetros
  que não fazem o que prometem (`wikipedia_article`, `wikidata_sparql`, `ror_search`), índices que
  nenhuma tool aceita (`mediawiki_sections`), validação mais estreita do que a da fonte (o
  `_TEXT_RE` da MediaWiki recusa `&`, `?`, `!` e `"`, logo "AT&T").
- **Correcção:** os limites vivem na assinatura e o schema enviado ao modelo mostra-os; fora deles,
  `invalid_argument` com o intervalo aceite. A validação aceita o que a fonte aceita, e cada
  identificador devolvido tem uma tool que o recebe. Antes de escolher a forma de declarar os
  limites, ver como o `@tool` deriva o schema.

### 3.5 A saída é a da fonte, não a de um leitor (D40)

- **Hoje:** wikitexto limpo por uma regex de uma só passagem (ficam as predefinições exteriores, `|-`,
  `rowspan`, `80px`), datas em milissegundos epoch, códigos sem rótulo, notação científica, autores
  como chaves `/authors/OL…A`.
- **Correcção:** a MediaWiki lê-se pelo HTML que o servidor renderiza (`action=parse&prop=text`),
  convertido pelo `html.parser` da stdlib. Datas, números e tabelas ganham uma formatação comum, e os
  rótulos que já vêm na resposta são usados.
- **Desaparece:** o `_clean_wikitext`.
- Tirar o `exintro` ao `wikipedia_article` não chegava: o TextExtracts deita fora as tabelas, e o
  texto completo da página do Nobel (3 286 caracteres) não tem nenhum laureado.

### 3.6 Várias tools para o mesmo trabalho (D41)

Dez grupos (anexo E). Fica uma família por fonte, com um trabalho por tool e sem aliases; a família
wiki usa a Wikipedia por omissão, não o Wiktionary.

## 4. Garantias

- **Invariante de contrato no CI**, em `tests/toolkit/test_tool_invariants.py`, que já descobre todas
  as tools e simula a rede:
  1. um corpo maior do que a janela dá um rodapé, e a continuação do rodapé, chamada, devolve a
     janela seguinte;
  2. cada corpo de erro real da fonte (uma tabela por API, tirada da documentação ou gravada ao vivo)
     dá uma falha com o código da fonte;
  3. um recurso que não existe dá `not_found`; zero resultados dá um sucesso que o diz;
  4. os limites do schema são os que o código aplica.
- **Lista de dívida que só encolhe:** as tools ainda fora do contrato. O teste falha se a lista
  crescer ou se uma tool da lista já cumprir.
- **Avaliação por tarefas, local e barata, nunca no CI:** perguntas reais por fonte, medidas em
  sucesso, chamadas e tokens. As duas conversas do ai-network são as primeiras. Referência: Anthropic,
  *Writing effective tools for agents* (paginação e truncagem com instruções, erros accionáveis,
  avaliação por tarefas).

## 5. Ordem

1. As costuras: `ToolError` e o executor (D37), a porta HTTP (D38), a primitiva de janela (D39).
2. A invariante de contrato, com todas as tools na lista de dívida.
3. MediaWiki e Wikipedia: HTML (D40), navegação e fusão (D41). A correcção do `missingtitle` feita em
   paralelo a 29/09 (ainda com strings de erro) converte-se aqui em `not_found`.
4. As 19 tools que mostram erros como sucesso.
5. Os 60 becos sem saída.
6. O resto: a saída por limpar e os grupos repetidos que sobrarem.

Cada passo muda comportamento visível (`CHANGELOG`) e é avisado ao ai-network.

---

## Anexo A · Cortes

### Contagens

- **94 das 132 tools cortam alguma coisa.** As outras 38 nunca cortam: datas, cálculo,
  `python_repl`, tempo e consultas de um só registo como `pdb_entry` e `chembl_molecule`.
- **33 têm parâmetro de continuação.**
  - 21 também dizem o total ou o token seguinte.
  - 12 paginam às cegas, sem total nem sinal de "há mais": `arxiv_search`, `crossref_search`,
    `datacite_search`, `earthquake_search`, `internet_archive_search`, `nvd_cve_search`,
    `open_library_search`, `pubmed_search`, `semantic_scholar_search`,
    `semantic_scholar_citations`, `who_indicators`, `who_series`.
  - A paginação do `ror_search` está partida (anexo B).
- **60 são becos sem saída** (cortam, e nada deixa ler o resto):
  - **25 cortam o resultado principal sem total:** `air_quality_forecast`, `define_word`,
    `eonet_events`, `eurostat_dimensions`, `eurostat_compare`, `gdelt_news_search`,
    `gdelt_timeline`, `geocode`, `mediawiki_page`, `mediawiki_sections`, `wiktionary_entry`,
    `hacker_news`, `osm_search_place`, `rxnorm_drug_search`, `rxnorm_related`, `rxnorm_ndcs`,
    `dailymed_label`, `uniprot_features`, `uniprot_crossrefs`, `wikidata_search`,
    `wikidata_entity`, `wikidata_sparql`, `wikipedia_search`, `wikipedia_article`,
    `wikipedia_related`.
  - **16 dizem o corte ou o total, mas não oferecem continuação:** `europe_pmc_citations`,
    `eurostat_series`, `overpass_query`, `overpass_pois`, `world_bank_compare`,
    `youtube_transcript`, `youtube_transcript_search`, `youtube_transcript_languages`, `http_get`,
    `scrape_text`, `read_file`, `list_directory`, `search_files`, `csv_read`, `run_command`,
    `regex_search`.
  - **19 tools de um registo cortam campos** (resumos, descrições, listas), e cada uma é o único
    caminho para esse registo: `arxiv_paper`, `clinical_trial_study`, `crossref_work`,
    `datacite_doi`, `eonet_event`, `europe_pmc_article`, `eurostat_dataset`,
    `internet_archive_item`, `nvd_cve`, `open_food_facts_product`/`_nutrition`/`_compare`,
    `open_library_work`/`_isbn`, `pubmed_article`, `ror_organization`, `semantic_scholar_paper`,
    `uniprot_entry`, `world_bank_indicator`.
  - Cerca de 50 dos 54 becos de rede deitam fora dados que já vinham na resposta, ou que a API
    paginaria. As excepções são `eonet_events`, `gdelt_news_search`, `http_get` e `scrape_text`.
- **Como se corta:** cerca de 72 `[:N]` silenciosos em 22 módulos, e 25 cortes de texto por helpers
  copiados: `_truncate` (" ... [truncated]", em 12 módulos) e `_trim` ("...", em 3). Nenhum diz o
  tamanho original. Só dizem tamanhos: web, ficheiros, shell, regex, overpass, `eurostat_series`, os
  cabeçalhos do World Bank e o youtube.

### Piores casos

| Tool | Onde | Efeito |
|---|---|---|
| `wikidata_sparql` | `_wikidata.py:176` | Uma query com `LIMIT 100` recebe 20 linhas, sem nota (sondado) |
| `rxnorm_ndcs`, `rxnorm_drug_search` | `_rxnorm_dailymed.py:236`, `:191` | Só as primeiras 25; sem total nem posição |
| `uniprot_crossrefs`, `uniprot_features` | `_uniprot.py:227`, `:201` | Só as primeiras 25; sem contagem |
| `gdelt_timeline` | `_gdelt.py:190` | Fica com os 20 pontos mais antigos: os dados recentes perdem-se |
| `wikidata_entity` | `_wikidata.py:289,293,232` | 15 afirmações tiradas das primeiras 20 propriedades, 3 valores cada |
| `nvd_cve` | `_nvd.py:279-281` | 5 CPEs e 5 referências, em silêncio |
| `internet_archive_item` | `_internet_archive.py:213` | 8 ficheiros, sem contagem |
| `dailymed_label` | `_rxnorm_dailymed.py:268` | Só os títulos das secções; nenhuma tool devolve o texto |
| `mediawiki_page` | `_mediawiki.py:190,195` | Corte a 4000 caracteres com "..."; nenhuma tool aceita `section`, embora `mediawiki_sections` devolva índices |
| `world_bank_compare` | `_world_bank.py:319-329` | Só a página 1; se as linhas vierem agrupadas por país (não visto ao vivo), caem países inteiros |
| `clinical_trial_study` | `_clinical_trials.py:241,264` | 5 locais e 5 referências, em silêncio |

### Por módulo

Legenda: P = parâmetro de continuação; T = diz o total ou o tamanho; M = marca de corte sem tamanho;
S = silencioso; BS = beco sem saída.

- **`_air_quality`:** `forecast` mostra 1–72 linhas horárias de até 336 (7 dias de previsão e 7
  passados); o cabeçalho só dá as linhas mostradas. S, BS.
- **`_arxiv`:** `search`: 1–20, P (`start`), sem total; resumo a 700 M; autores 8, depois
  "(+N more)". `paper`: os mesmos cortes. BS.
- **`_chembl`:** as três pesquisas: 1–25, P, T.
- **`_clinical_trials`:** `search`: 1–20, P (`page_token`), mostra o token seguinte, sem total;
  resumo 900 M; condições e intervenções 8 S; locais 5 S. `study`: resumo 900 M; elegibilidade 1200 M;
  braços e resultados 6 cada (220–240 caracteres M, a contagem S); locais 5 S; referências 5 S. BS.
- **`_crossref`:** `search`: 1–20, P, sem total; licença e ligações 3 S. `work`: resumo 900 M;
  referências 5 "(+N)" a 220 caracteres M; autores 8 "(+N)"; ligações 3 S. BS.
- **`_datacite`:** `search`: 1–20, P, sem total; criadores 8, assuntos 10, direitos e relacionados
  5, tudo S. `doi`: só a primeira descrição, 900 M; as mesmas listas S. BS.
- **`_dictionary`:** `define_word` fica com a primeira entrada e 3 definições por classe de palavra.
  S, BS.
- **`_earthquake`:** `search`: 1–50, P. O "count" do cabeçalho é `metadata.count` (suspeito: as
  linhas devolvidas, não o número de resultados).
- **`_eonet`:** `events`: 1–50, sem posição (a API também não tem). BS. `event`: fontes 5 S; o
  percurso fica reduzido ao último ponto. BS.
- **`_europe_pmc`:** `search`: 1–20, P (cursor), T — a melhor do repositório. `article`: resumo
  1200 M; URLs de texto completo 5 S. BS. `citations`: 1–20, T, mas sem parâmetro de página,
  embora a API o tenha. BS.
- **`_eurostat`:** `dataset_search`: 1–50, P, T. `dataset`: descrição 500 M. BS. `dimensions`:
  1–50 códigos por dimensão, sem contagem. BS. `series`: 1–50 pontos, "X of Y". BS. `compare`:
  os primeiros `last_time_periods` pontos por geografia, S. BS.
- **`_filesystem`:** `read_file`: 1–10 000 linhas / 100 000 caracteres, T. BS. `list_directory`:
  1000 entradas, T. BS. `search_files`: 1–1000 resultados, T; linhas cortadas a 300 caracteres S; só
  o primeiro milhão de caracteres de cada ficheiro é lido, S. BS.
- **`_json`:** `csv_read`: 1–10 000 linhas, "N of M", mas o M só conta o primeiro milhão de
  caracteres. BS.
- **`_foodon`:** `search`: 1–20, P, T; definição 700 M (o `foodon_term` dá o texto completo).
- **`_gbif`:** as duas pesquisas: 1–50, P, T.
- **`_gdelt`:** `news_search`: 1–20 (a API permite 250), sem total. BS. `timeline`: os primeiros 20
  pontos, S. BS.
- **`_geo`:** `geocode`: 3 resultados, fixo, S. BS.
- **`_internet_archive`:** `search`: 1–20, P, sem total; listas S. `item`: descrição 1000 M;
  ficheiros 8 S; criadores e colecções 8 S; assuntos 12 S. BS.
- **`_mediawiki`:** `search`: 1–25, P, T. `page`: 200–4000 caracteres, "..."; lista de secções 15 S.
  BS. `sections`: 25, S. BS. `wiktionary_entry`: 200–4000, "..."; lista de secções 20 S.
- **`_news`:** `hacker_news`: 1–30 de cerca de 500 histórias. BS.
- **`_nvd`:** `search`: 1–20, P, sem total; descrição 900 M; fraquezas 8, CPEs 5, referências 5,
  tudo S. `cve`: os mesmos cortes. BS.
- **`_open_food_facts`:** `search`: 1–20, P, T. `product`: categorias, rótulos e países 8 S;
  alergénios, vestígios e aditivos 10 S. BS. `nutrition`: alergénios e vestígios 10 S. BS.
  `compare`: alergénios 5 S. BS.
- **`_open_library`:** `search`: 1–20, P, sem total; listas S. `work` e `isbn`: descrição 1000 M;
  listas S. BS.
- **`_openfda_food`:** `recall_search`: 1–20, P, T.
- **`_osm`:** `search_place`: 1–10 (a API permite 40), sem paginação. BS. `reverse_geocode`: as
  etiquetas extra são as primeiras 8 de 12 por ordem alfabética, S (menor).
- **`_overpass`:** as duas tools: 1–50, "N of M". BS.
- **`_pdb`:** `search`: 1–25, P, T.
- **`_pubmed`:** `search`: 1–20, P, sem contagem. `article`: resumo 900 M; MeSH e palavras-chave
  12 S; tipos de publicação 8 S. BS.
- **`_ror`:** `search`: P e T, mas a paginação está partida (anexo B). `organization`: nomes
  alternativos, siglas e relações 10 S; só a primeira localização. BS.
- **`_rxnorm_dailymed`:** `drug_search`: 25 S. BS. `related`: 1–25 S. BS. `ndcs`: 25 S. BS.
  `label_search`: 1–25, P, T. `label`: só títulos. BS.
- **`_semantic_scholar`:** `search`: 1–20, P, sem total. `paper`: resumo 900 M; listas 8 S. BS.
  `citations`: 1–20, P, sem total; só o primeiro contexto, 260 M.
- **`_shell`:** `run_command`: T. BS.
- **`_text`:** `regex_search`: 1000 ocorrências, "more not shown". BS.
- **`_uniprot`:** `search`: 1–25, P (o `offset` é suspeito de não fazer nada na fonte), T. `entry`:
  texto FUNCTION a 500, "...". BS. `features` e `crossrefs`: 1–25 S. BS.
- **`_web`:** `http_get` e `scrape_text`: T. BS.
- **`_who_gho`:** `indicators` e `series`: 1–100, P, sem total.
- **`_wikidata`:** `search`, `entity` e `sparql`: todas BS, como acima.
- **`_wikipedia`:** `search`: 1–10, sem posição. `article`: só a introdução, depois "[Truncated]"
  sem tamanho. `related`: 1–20, sem continuação. As três BS.
- **`_world_bank`:** `topics`, `sources`, `countries`, `indicators`, `series`: 1–100, P, T; notas
  500 M. `indicator`: definição 500 M. BS. `compare`: só a página 1. BS.
- **`_youtube`:** `transcript`: 50 000 caracteres, T. BS. `transcript_search`: 1–20, "N more". BS.

## Anexo B · Promessas que o código não cumpre

Cerca de 12 confirmadas e 7 suspeitas.

| Tool | Onde | Diferença |
|---|---|---|
| `wikipedia_article` | `_wikipedia.py:43-51,117-118` | `max_chars` aceita até 100 000, mas o `exintro` devolve só a introdução. O ramo "[Truncated]" perde o título. Não manda `redirects=1`, que o `wikipedia_related` (`:80`) manda; suspeito que um título de redirecionamento volte vazio |
| `wikidata_sparql` | `_wikidata.py:125,176` | A docstring diz que `max_results` só põe um `LIMIT` quando a query não tem; o código corta qualquer resultado a 20 linhas |
| `ror_search` | `_ror.py:47,88` | `max_results` corta do nosso lado a página fixa de 20 da ROR: com `page=2` mostra as linhas 21–25, e as linhas 6–20 de cada página nunca se alcançam |
| `wiktionary_entry` | `_mediawiki.py:221,225` | O cabeçalho diz "(Portuguese)", mas quando essa secção falta devolve a página inteira |
| `youtube_transcript` | `_youtube.py:417` | Diz "Increase max_chars" mesmo quando já está no tecto de 50 000 |
| `csv_read` | `_json.py:88,117-120` | O total de "N of M rows" só cobre o primeiro milhão de caracteres; o resto cai em silêncio |
| `wikipedia_related` | `_wikipedia.py:63-80,90-91` | "Related" são as primeiras N ligações pela ordem da API; uma página que não existe passa, calada, a pesquisa |
| `eurostat_compare` | `_eurostat.py:182` | Fica com os primeiros N pontos pela ordem do índice, por geografia: uma série arbitrária quando as outras dimensões ficam em aberto |
| `mediawiki_sections` | `_mediawiki.py:207-211` | Devolve índices de secção que nenhuma tool aceita |
| Limites não documentados | `_eonet.py:89` (dias até 365); `_eurostat.py:167` (`last_time_periods` até 20) | Nenhum está na docstring |
| Arredondamento calado | `unit_convert` `_math.py:351,357` (4 ou 6 algarismos significativos); World Bank `_world_bank.py:899-902` | Arredonda sem o dizer |
| `wikipedia_search` | `_wikipedia.py:16` | Promete "summaries"; devolve os excertos da pesquisa |

**Suspeitas:**

- o `offset` do `uniprot_search` (`_uniprot.py:52`): a UniProt pagina por cursor, por isso o
  `offset` é provavelmente ignorado;
- o "count" do `earthquake_search` (`_earthquake.py:131`) é provavelmente o da página, não o número
  de resultados;
- o `gbif_species_match` promete "common names" (`_gbif.py:29`); a API de correspondência resolve
  nomes científicos;
- o `semantic_scholar_search` diz aceitar "DOI" (`:107`); é uma pesquisa por palavras;
- o `datacite_search` põe o `resource_type` em minúsculas em `resource-type-id` (`:60`), o que parte
  os tipos de várias palavras;
- o `nvd_cve_search` não diz o limite de 120 dias da NVD para intervalos de datas, e o motivo do 404
  que daí vem perde-se.

## Anexo C · Erros mostrados como sucesso

19 tools confirmadas e 17 suspeitas. O padrão de raiz é `.get("x", {})` a alimentar uma guarda
`isinstance` que nunca falha; a mesma guarda morta está em `_mediawiki.py:183,200,217` e
`_eurostat.py:226`.

**Confirmadas (sondadas):**

| Tools | Onde | O que o modelo vê |
|---|---|---|
| `mediawiki_page`, `wiktionary_entry` | `_mediawiki.py:182-196`, `:216-230` | Só um cabeçalho, como se a página estivesse vazia |
| `mediawiki_sections`, `mediawiki_search` | `:199-205`, `:162-165` | "No sections found" / "No MediaWiki pages found" |
| `wikipedia_search`, `wikipedia_article`, `wikipedia_related` | `_wikipedia.py:95-98`, `:109-121`, `:124-144` | "No results", "Article not found", ou uma troca calada para a pesquisa |
| `wikidata_search`, `country_info` | `_wikidata.py:151-154`, `_geo.py:291-293` | "No results" / "Country not found"; por exemplo, `language="english"` passa a regex de `_wikidata.py:22`, e o erro `badvalue` da API vira "No Wikidata results" |
| As 7 `world_bank_*` | `_world_bank.py:334-343` | O `_page` transforma de propósito o erro `[{"message": …}]` da API numa página vazia: um país inválido dá "No World Bank series observations found." |
| `overpass_query`, `overpass_pois` | `_overpass.py:105-110` | O campo `remark` é ignorado: um timeout vira "No Overpass POIs found."; um resultado parcial por falta de memória aparece como "returned 1 of 1" |
| `hacker_news` | `_news.py:31-36` | As histórias que falham ao carregar caem sem nota |

**Suspeitas:**

| Tools | Onde | Porquê |
|---|---|---|
| `gdelt_timeline` | `_gdelt.py:128-160` | Suspeita alta: o parser e o fixture do teste (`tests/toolkit/test_gdelt.py:80`) esperam uma lista plana de pontos, mas o GDELT aninha-os em `timeline[].data`. Se assim for, qualquer chamada ao vivo dá "No GDELT timeline points found" |
| As duas tools do GDELT | `_gdelt.py:84-87,106-109` | O GDELT devolve erros em texto com HTTP 200, que viram "could not parse API response" |
| `pdb_search` | `_pdb.py:53-58` | A RCSB responde "sem resultados" com HTTP 204 e corpo vazio, reportado como falha de leitura |
| `eurostat_dataset`, `eurostat_dimensions`, `eurostat_series` | `_eurostat.py:206-252` | Nenhuma verificação de erro |
| `eonet_event` | `_eonet.py:119-122` | Mostra um evento em branco |
| `arxiv_paper`, `arxiv_search` | `_arxiv.py:214-250` | A entrada de erro do arXiv apareceria como um artigo com o título "Error" |
| `pubmed_search` | `_pubmed.py:169-171` | A chave ERROR lê-se como "No results" |
| `internet_archive_search` | `_internet_archive.py:120-125` | Um erro lê-se como "No items found" |
| `open_library_work`, `open_library_isbn` | `_open_library.py:176-255` | Registos redireccionados ou apagados aparecem como "(untitled)" |
| `wikidata_entity` | `_wikidata.py:157-161` | Um QID fundido lê-se como "not found" |
| `uniprot_entry`, `uniprot_features`, `uniprot_crossrefs` | `_uniprot.py:179-185` | Entradas inactivas mostram "(no protein name) … entry: Inactive" e perdem o destino da fusão |

**Relacionado, confirmado:** 12 módulos declaram `404: "no matching records found."`. Qualquer 404
— incluindo um endpoint errado, que era exactamente o defeito do `uniprot_search` antes de 28/09 —
lê-se como "sem resultados".

## Anexo D · Saída por limpar

16 tools.

| Tools | Onde | Problema |
|---|---|---|
| `mediawiki_page`, `wiktionary_entry` | `_mediawiki.py:258-268` | Uma só passagem de limpeza deixa cascas de predefinições aninhadas, marcação de tabelas e parâmetros de imagens |
| `earthquake_search`, `earthquake_event` | `_earthquake.py:275` | Tempos em milissegundos epoch |
| `wikidata_entity` | `_wikidata.py:285-322` | Códigos de propriedade e de item sem rótulo ("P31: Q5"), unidades perdidas, coordenadas escritas como um dict de Python |
| `world_bank_series`, `world_bank_compare` | `_world_bank.py:899-902` | Valores grandes em notação científica, por exemplo "2.91849e+13" |
| `open_library_work`, `open_library_isbn` | `_open_library.py:182-190` | Autores como chaves `/authors/OL…A`, não nomes |
| `eurostat_series`, `eurostat_compare` | `_eurostat.py:255,348` | Códigos de dimensão; os rótulos da mesma resposta não são usados |
| `clinical_trial_study` | `_clinical_trials.py:202` | A lista de critérios de elegibilidade fica numa só linha |
| `who_series` | — | Códigos crus |
| `nvd_cve`, `nvd_cve_search` | `_nvd.py:198-215` | Pontuação CVSS sem a versão |
| `pdb_search` | — | Só identificadores: cada título custa outra chamada |

## Anexo E · Tools repetidas sobre a mesma fonte

1. **A Wikipedia por duas famílias.** `wikipedia_search` dá no máximo 10 resultados, sem posição nem
   total, e passa a pesquisa tal como vem; `mediawiki_search` apontada à Wikipedia dá até 25, com
   posição e total, mas o `_TEXT_RE` recusa `&`, `?`, `!` e `"`. `wikipedia_article` dá a
   introdução (até 100 000 caracteres); `mediawiki_page` dá wikitexto (até 4000) e recusa "AT&T".
2. **As tools MediaWiki usam o Wiktionary inglês por omissão** (`_mediawiki.py:12`), não a
   Wikipedia. Com `wiktionary_entry` e `define_word`, são três tools de definições com saídas
   diferentes.
3. **Geocodificação inversa do Nominatim:** `reverse_geocode` (zoom 10, `_geo.py:85`) e
   `osm_reverse_geocode` (zoom 18, `_osm.py:98`) dão respostas diferentes para o mesmo ponto.
4. **Geocodificação do Open-Meteo:** `geocode` mostra 3 resultados; as tools de tempo ficam, caladas,
   com o primeiro (`_weather.py:53`).
5. **Três tools para o tempo actual:** `get_weather`, `weather_units` e `get_weather_by_coords`. A de
   coordenadas não valida, e quando o Open-Meteo recusa coordenadas más o motivo perde-se.
6. **Open Food Facts:** `product`, `nutrition` e `compare` partilham um pedido mas cortam os
   alergénios a 10, 10 e 5.
7. **UniProt:** `entry`, `features` e `crossrefs` descarregam cada uma a mesma entrada inteira e
   cortam-na de maneira diferente.
8. **Eurostat:** `eurostat_dataset` e `eurostat_dimensions` fazem o mesmo pedido com cortes
   diferentes.
9. **World Bank:** `series` pagina; `compare` só lê a página 1.
10. **O mesmo resumo corta-se em comprimentos diferentes** conforme a tool: 700, 900, 1000 ou 1200
    caracteres (por exemplo, `pubmed_article` a 900 e `europe_pmc_article` a 1200).

## Anexo F · A porta HTTP, o executor e os testes

**`_http.py`:**

- Uma resposta que não é 2xx lança `HttpError(message, status=…, body=primeiros 2000 caracteres)`
  (`_http.py:35-46`, `:168-175`, `:213-215`). A mensagem é o `status_messages[status]` do módulo,
  se o declarou; senão, um 429 dá "rate limited by {name}…" e o resto "HTTP error {status}:
  {reason}" (`:407-412`). Falhas de rede, timeouts, redirecionamentos recusados e corpos grandes
  demais lançam `HttpError` com `status=None`.
- O corpo nunca chega à mensagem: só 2 dos 44 módulos lêem o `e.body` — `_air_quality.py:146` (o
  `reason` do Open-Meteo) e `_gdelt.py:114` (só o detalhe do 429).
- Qualquer 2xx é sucesso: o `_fetch` devolve só o corpo, o charset e se veio completo
  (`:197-225`), sem estado nem cabeçalhos. Isso impede verificações como o cabeçalho
  `MediaWiki-API-Error` ou o cabeçalho `message` da NVD. Um 204 ou um corpo vazio dá "could not parse
  API response".
- Não há gancho para erros dentro de um 200. O único ponto de extensão é o `parse=` de cada chamada:
  excepções de forma dão "could not parse API response" (`:254-271`), e um `parse` pode lançar
  `HttpError` — só `_pubmed.py:183`, `_rxnorm_dailymed.py:261` e `_world_bank.py:343` o fazem, e
  nenhum por um erro da fonte. Só três tools detectam erros no corpo, cada uma à sua maneira:
  `_geo.py:276` (`success`), `_open_food_facts.py:229` (`status==0`) e `_world_bank.py:341`, que
  depois transforma o erro numa página vazia.

**Executor e `ToolResult`:**

- O `_bounded` (`core/_tools/_executor.py:177-194`) corta o texto do modelo em `max_output_chars`
  (200 000 por omissão, `_definition.py:22`; um `ToolGroup` só o pode baixar), acrescenta
  "[Output truncated: kept X of Y characters.]" e marca `metadata["truncated"]`. O modelo fica a
  saber o total, mas não tem continuação, e o corte pode cair a meio de uma linha; aplica-se também
  às mensagens de erro. Na prática raramente dispara, porque os cortes das tools vêm primeiro.
- O `ToolResult` (`_result.py:31-38`) tem `ok`, `value`, um `error: ToolError` (tipo, mensagem,
  `retryable`, `safe_to_show`, detalhes) e `metadata`. `ok=False` só acontece para tool
  desconhecida, falha de validação, bloqueio de um gate ou de `max_calls`, timeout, ou uma excepção
  lançada pela tool (`runtime_error`, `:247-260`).
- As strings de erro devolvidas pelas tools contam como sucesso: o `_coerce_result` embrulha qualquer
  valor em `ToolResult.success` (`:240-244`), com `ok=True` e sem erro, e o meter liquida a chamada
  como sucesso (`:438-444`). Nenhuma tool de `toolkit/tools` devolve um `ToolResult`.

**Testes que já correm sobre todas as tools:**

- `tests/toolkit/test_tool_invariants.py` descobre todas as `@tool` sozinho (`:51-67`) e verifica que
  cada uma: é exportada por exactamente um namespace (`:81`); declara a capacidade que o código usa,
  por AST (`:159`); não sai dos hosts do módulo com argumentos hostis (`:339`); nunca rebenta — com
  argumentos hostis (`:359`), corpos hostis incluindo `{}`, 500, 404 e 429 (`:404`), e corpos
  construídos com todas as chaves que o módulo lê (`:446`); e fica dentro de `max_output_chars` + 200
  através do executor (`:495`). Só verifica que o resultado é um `str`: nunca o que o texto diz.
- Outros: `tests/test_architecture.py:260` (só o `_http.py` chega à rede), `tests/toolkit/test_http.py`
  (a própria porta), `tests/test_tool_limits.py` (limites de saída e de tempo do executor),
  `tests/toolkit/test_tools_exports.py` (seguras e perigosas), `tests/conftest.py` (bloqueia sockets
  nos testes das tools).
- Erros no corpo só têm teste no `ip_lookup` (`test_geo.py:107,116`) e no Open Food Facts
  (`test_open_food_facts.py:76`).
- A invariante de contrato (secção 4) entra como invariante nova neste ficheiro, reutilizando
  `TOOLS`, `NETWORK`, `_benign` e `_answering`, ou num `tests/toolkit/test_tool_contract.py` ao lado
  que os importa, com uma tabela de corpos de erro reais por API.
