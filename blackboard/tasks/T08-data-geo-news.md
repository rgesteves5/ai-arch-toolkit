# T08 · Dados, geo e notícias

- **Dono:** Claude (dois agentes, T08a e T08b, 2026-10-08) · **Estado:** done · **Depende de:** T05 (o modelo a seguir)
- **Módulos:** `_eurostat` (5 tools), `_world_bank` (7), `_who_gho` (3), `_gdelt` (2), `_overpass` (2),
  `_osm` (2), `_geo` (6), `_weather` (5), `_air_quality` (2), `_eonet` (3), `_earthquake` (3),
  `_wikidata` (3), `_news` (1): 44 tools
- **Origem:** os anexos A a E do plano para estes módulos · **Decisões:** D37 a D42 · **Regras:**
  `T00-rules.md`
- **Já feito:** `c259e0b` (erros num 200 do World Bank, do Overpass, do GDELT e da Wikidata; o
  `gdelt_timeline` lê as séries) e `6a34668` (o Eurostat explica os seus 404 e 413; o `eonet_event`
  explica o 500 e recusa uma resposta sem evento; o `wikidata_entity` segue um QID fundido; o
  `hacker_news` diz que histórias não carregou).

## O que está partido

- **Becos sem saída:**
  - o `wikidata_sparql` devolve 20 linhas a uma query com `LIMIT 100`, sem nota, embora a docstring
    diga que só põe `LIMIT` quando a query não o tem;
  - o `wikidata_entity` mostra 15 afirmações, tiradas das primeiras 20 propriedades, com 3 valores
    cada;
  - o `world_bank_compare` só lê a página 1;
  - o `gdelt_news_search` fica em 20 resultados (a API permite 250), sem total;
  - o `gdelt_timeline` fica com os primeiros 20 pontos (desde `c259e0b` diz que corta, mas não como
    continuar);
  - o `osm_search_place` fica em 10 resultados (a API permite 40);
  - o `geocode` mostra sempre 3;
  - o `hacker_news` mostra 30 de cerca de 500;
  - o `air_quality_forecast` mostra até 72 de 336 linhas horárias;
  - `overpass_*`, `eurostat_series`, `eurostat_dimensions`, `eurostat_compare` e o `eonet_*`
    (percurso reduzido ao último ponto) cortam sem continuação;
  - os `who_*` paginam sem total.
- **Promessas:**
  - o `eurostat_compare` fica com os primeiros N pontos pela ordem do índice, uma série arbitrária
    quando as outras dimensões ficam em aberto;
  - os limites de dias do EONET (365) e de `last_time_periods` do Eurostat (20) não estão na
    docstring;
  - o "count" do `earthquake_search` é provavelmente o da página (suspeita);
  - o World Bank arredonda sem o dizer.
- **Saída:**
  - os tempos do `earthquake_*` vêm em milissegundos epoch;
  - o `wikidata_entity` mostra códigos sem rótulo ("P31: Q5"), perde as unidades e escreve as
    coordenadas como um dict de Python;
  - o World Bank usa notação científica;
  - o Eurostat mostra códigos de dimensão, e os rótulos que vêm na resposta não são usados;
  - o `who_series` mostra códigos crus.
- **Repetidas (decide as fusões na nota de desenho, D41):**
  - `reverse_geocode` (zoom 10) e `osm_reverse_geocode` (zoom 18) dão respostas diferentes para o
    mesmo ponto;
  - o `geocode` mostra 3 resultados, e as tools de tempo ficam, sem dizer, com o primeiro;
  - há três tools de tempo actual (`get_weather`, `weather_units`, `get_weather_by_coords`), e a de
    coordenadas não valida e perde o motivo da recusa;
  - `eurostat_dataset` e `eurostat_dimensions` fazem o mesmo pedido;
  - o `world_bank_series` pagina, e o `world_bank_compare` não.

## Objectivo

As 44 tools cumprem o contrato de `T00-rules.md` e saem da lista de dívida.

## Passos

1. Nota de desenho por módulo: continuação, limites, falhas, leitores de erro, formatação (datas ISO,
   números, rótulos) e fusões. Confirma na documentação de cada API e escreve os URLs.
2. Testes primeiro, por módulo.
3. A migração; os `_truncate`/`_trim` destes módulos desaparecem.
4. `docs/tools-catalog.md` e `CHANGELOG`, com as quebras e a tabela de migração se houver fusões.

## Ficheiros

Os 13 módulos, os seus testes em `tests/toolkit/`, `tests/toolkit/contract_debt.py` (só as linhas
destas tools), `tests/toolkit/error_bodies.py` (as entradas destas fontes), `docs/tools-catalog.md`.

## Prova

- A invariante de contrato passa nas 44 tools, que saem da lista de dívida.
- Há testes de comportamento para cada item acima, incluindo o `LIMIT` do SPARQL, a página 2 do
  `world_bank_compare` e os rótulos do Wikidata e do Eurostat.
- **Ao vivo, pelo dono:** os comandos no fim da ficha.

## Registo do dono

- **Estado:** done. Feito por agentes em worktrees, juntado e revisto pelo coordenador; uma revisão independente por parte, e as correcções dela (secção "Correcções da revisão").
- **Gate final, no checkout principal com todas as fichas da vaga:** 8195 passed, 118 skipped (os ao vivo e os 59 do nanope à espera); ruff, formatação, pyright e `uv lock --check` limpos (2026-10-08).
- **CHANGELOG:** as linhas entraram em `[Unreleased]` (Upgrade notes, Added, Changed, Fixed).

### T08a

- **Dono:** Claude (agente T08a), 2026-10-08 · **Estado:** review (diff por commitar na worktree)
- **Worktree:** `/Users/rge/Documents/dev/pessoal/ai-arch-toolkit/ai-arch-toolkit/.claude/worktrees/agent-a8201c43080719d4f`
- **Módulos:** `_eurostat` (5 → 3 tools), `_world_bank` (7 → 6), `_who_gho` (3), `_gdelt` (2),
  `_wikidata` (3), `_news` (1). Eram 21 tools; ficam 18, todas fora da dívida, com as três tiradas.
- **Resumo:** as 18 tools cumprem os cinco pontos do contrato. Toda a lista lê-se pela janela
  (`offset`, `page` ou `skip`, como a fonte pagina), os limites são `Range`, as falhas têm tipo e
  levam as palavras da fonte, zero resultados é sucesso com a consulta, e a saída lê-se sem
  descodificar: números com todos os dígitos e sem notação científica, datas ISO 8601 UTC, códigos
  com o rótulo. Saem os dois helpers de corte (`_trim` do Eurostat, `_truncate` do World Bank):
  `CUT_HELPERS` desce de 14 para 12.

#### Nota de desenho

##### `_eurostat`

Documentação: o guia da API de estatísticas
(https://ec.europa.eu/eurostat/web/user-guides/data-browser/api-data-access/api-detailed-guidelines/api-statistics),
as FAQ (https://ec.europa.eu/eurostat/web/user-guides/data-browser/api-data-access/api-faq) e o
formato JSON-stat 2.0 (https://json-stat.org/format/). O wiki antigo do Eurostat
(`wikis.ec.europa.eu`) pede login: não se lê.

- **O que o Eurostat responde quando um filtro não encontra nada** (o pedido da ficha): o guia tem
  uma tabela de erros. Um 400 com o erro 100 ("No results found") é uma consulta cujo resultado é
  vazio; um 404 com o mesmo 100 é "the requested resource is not available" (o dataset); 140 e 150
  com 400 são consultas recusadas (sintaxe, ou um código fora da restrição do dataset: as FAQ dão
  `INVALID_QUERY_DIMENSION_VALUE … GEO=EU27`); 500 é o resto. Por isso:
  - 404 → `not_found`, "no dataset X" (deixa de dizer "ou não há dados para os filtros": o guia diz
    que é o recurso);
  - 400 com erro 100 → `not_found`, "Eurostat has no data for this query (No results found);
    other codes or periods may have some: eurostat_dataset lists the codes". O `eurostat_series` é
    uma consulta de um recurso (o pedaço do dataset), por isso `not_found` e não zero resultados;
  - outro 400 → `validation_error`, com a etiqueta e "list the codes with
    eurostat_dataset(dataset_id, dimension=...)";
  - 413, ou o `{"warning": {"status": 413}}` que o guia mostra → `upstream` com nova tentativa;
    outro `warning` não é erro.
- **Vários códigos de uma dimensão:** o guia manda repetir o parâmetro (`geo=FR&geo=DE`). O código
  mandava `geo=PT+ES` como um valor só, que o guia não define. O filtro continua a escrever-se
  `geo=PT+ES`, mas sai como parâmetros repetidos.
- **Tempo:** só um parâmetro de tempo por consulta (o guia). `last_time_periods` só vai quando os
  filtros não trazem `time`, `time_period`, `sinceTimePeriod`, `untilTimePeriod` ou
  `lastTimePeriod`. O tecto calado de 20 sai: `Range(1, 100)` na assinatura.
- **Rótulos:** a resposta JSON-stat traz o rótulo de cada código (`dimension.*.category.label`) e
  de cada dimensão. As linhas mostram `Portugal (PT)`; as dimensões com um só código vão para o
  cabeçalho ("fixed: Time frequency (freq) = Annual (A)"), e as linhas só nomeiam as que variam.
- **A série arbitrária do `eurostat_compare`:** cada observação é uma linha, pela ordem do índice
  (série a série, tempo mais rápido), e o cabeçalho diz que outras dimensões além de `geo` variam
  ("unit varies too: each row names its series; filter it (e.g. unit=NR) to compare one series").
- **Janela:** `eurostat_series` por observações (`offset`, `max_points` 1-100);
  `eurostat_dataset` por linhas de códigos (60 por janela, `offset`, e `dimension=` para uma só,
  que a chamada seguinte repete); `eurostat_dataset_search` pelos datasets que casam (o catálogo
  vem inteiro em stubs), com total.
- **Saída:** `updated` em UTC (`2026-04-30T23:00:00+0200` → `2026-04-30T21:00:00Z`), a descrição
  inteira (não corta aos 500), o período das anotações (`OBS_PERIOD_OVERALL_*`), as flags
  (`(flag p)`), números sem notação científica.
- **Fusões (D41):** ver "Fusões e nomes tirados".

##### `_world_bank`

Documentação: estrutura das chamadas
(https://datahelpdesk.worldbank.org/knowledgebase/articles/898581-api-basic-call-structures) e
códigos de erro (https://datahelpdesk.worldbank.org/knowledgebase/articles/898620-api-error-codes).

- **Paginação:** `page` e `per_page`; a resposta diz `page`, `pages`, `per_page` e `total`. Cada
  lista é uma janela numerada a partir do início da página, com o total e `next: page=N+1`. O guia
  diz que a ordem não se controla: a docstring do `world_bank_series` diz "in the API's order".
- **Vários países:** um só pedido, os códigos juntos por `;` (o guia). Daí a fusão do
  `world_bank_compare` no `world_bank_series`, que agora pagina (o compare só lia a página 1).
- **Erros:** a API manda `[{"message": [{"id", "key", "value"}]}]` com HTTP 200 (visto a
  2026-09-29). O leitor tipa pelos ids da tabela: 115, 120, 150 e 160 são do pedido
  (`validation_error`, com "check the codes and IDs given (world_bank_countries, …)"); 105 é o
  serviço em baixo (`upstream`, nova tentativa); o resto são as palavras da fonte. Nos pedidos de
  códigos com nome (`world_bank_indicator`, `world_bank_series`) o 120 é `not_found`, com as
  palavras da fonte e o passo seguinte; na série, o 120 não diz se falta o indicador ou o país, e a
  mensagem di-lo.
- **Números:** o `.6g` arredondava a seis algarismos e escrevia `2.91849e+13`. Agora vão todos os
  dígitos que a API mandou (`29184912345600`); o valor nulo é "no value"; períodos `2021M07` →
  `2021-07`, `2021Q1` → `2021-Q1`.
- **Cortes de campo:** a nota de cada tópico, a definição e a organização do indicador iam a 500
  caracteres; os tópicos de um indicador, a 6. Agora vão inteiros (D39: um campo não se corta
  abaixo da janela). As listas de indicadores deixam a definição para o `world_bank_indicator`, que
  o cabeçalho nomeia. O `world_bank_indicator` passa a `WHOLE`.
- **Pesquisa de indicadores:** a API não pesquisa texto; a tool lê `scan_pages` páginas de 1000 do
  catálogo (`Range(1, 30)`), desde a primeira, e ordena os resultados. `page` passa a ser a página
  dos resultados (antes era a primeira página lida). O cabeçalho diz quantos indicadores leu e de
  quantos, e como ler o resto (`scan_pages=30`, ou "narrow it with topic or source" se o catálogo
  passar das 30 páginas). Com `topic` e `source` ao mesmo tempo (não há endpoint que junte os
  dois), segue o mesmo caminho, sem consulta.
- **Países com filtros:** um pedido de 1000 (a API lista 296 em 2026) e a página dos que casam, com
  o total; se a API disser que tem mais do que leu, o cabeçalho di-lo.

##### `_who_gho`

Documentação: https://www.who.int/data/gho/info/gho-odata-api (não fala de `$top`, `$skip` nem
`$count`), e o OData v4.01 para o objecto de erro e o `@odata.nextLink`
(https://docs.oasis-open.org/odata/odata-json-format/v4.01/odata-json-format-v4.01.html#sec_ErrorResponse,
`#sec_ControlInformationnextLinkodatanextLink`).

- **Total:** nenhuma documentação garante `$count`; mandá-lo a um servidor que não o aceita dá 400.
  As listas pedem uma linha a mais do que mostram (`$top = max_results + 1`): essa linha, ou um
  `@odata.nextLink`, diz que há mais, e o rodapé dá o `skip` seguinte. Sem total (D39: "há mais"
  quando a fonte não dá total).
- **Erros:** leitor do objecto OData `{"error": {"code", "message"}}`; um 400 é
  `validation_error` ("check the code and the filters (who_indicators lists the codes)").
- **Códigos crus do `who_series`:** cada linha diz o tipo de cada código e os rótulos que a linha
  traz: `PRT (country, region Europe EUR), 2019: 81.31234 (80.9 to 81.7) | sex: SEX_BTSX`. O
  valor é o `NumericValue` com todos os dígitos e o intervalo `Low`/`High`; sem número, o `Value`
  que a OMS escreve. Não pedi os rótulos dos códigos ao `/api/DIMENSION/…/DimensionValues`: seriam
  mais pedidos por chamada, e o ponto 5 pede os rótulos que a resposta já traz.
- **Validação:** `skip` e `max_results` passam a `Range`; o `_check_skip` sai.

##### `_gdelt`

Documentação: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/.

- **Pesquisa:** `maxrecords` até 250, sem offset e sem total. A tool pede os artigos até ao fim da
  página e mais um (`maxrecords = min(offset + max_results + 1, 250)`), e mostra
  `[offset, offset + max_results)`. O a mais diz que há página seguinte; uma lista mais curta do
  que o pedido é tudo, com total; no tecto dos 250 o cabeçalho diz como ver outros ("narrow the
  timespan or the query, or change the sort"). `max_results` `Range(1, 50)`, `offset`
  `Range(0, 249)`. É o mesmo desenho do `_first_results.py` da T08b (ver achados).
- **Timeline:** o passo é de 15 minutos abaixo de 72 h, de uma hora até uma semana, de um dia além
  disso (até 288 pontos). Todos os pontos lêem-se pela janela (100 por janela, `offset`), dos mais
  antigos para os mais recentes. O valor é a percentagem de toda a cobertura monitorizada (o
  `timelinevol`, segundo a documentação): sai com `%`, e o cabeçalho di-lo.
- **Datas:** `20260611T120000Z` → `2026-06-11T12:00:00Z`.
- **`timespan`:** a documentação dá `min` para minutos e `m` para meses; o regex antigo lia `m` sem
  dizer o quê. Agora aceita `min`, `h`, `d`, `w` e `m`, e a mensagem diz "(months)".

##### `_wikidata`

Documentação: `wbsearchentities` e `wbgetentities`
(https://www.wikidata.org/w/api.php?action=help&modules=wbsearchentities,
`…&modules=wbgetentities`), o modelo JSON
(https://doc.wikimedia.org/Wikibase/master/php/docs_topics_json.html), o manual do WDQS
(https://www.mediawiki.org/wiki/Wikidata_Query_Service/User_Manual) e
https://www.wikidata.org/wiki/Wikidata:Data_access.

- **`wikidata_search`:** `limit` 1-50 e `continue` 0-10 000 (a ajuda). O rodapé dá o
  `search-continue` que a API devolve (que a ajuda não documenta; sem ele, não há página
  seguinte). O `_API` continua com `error_reader=mediawiki_error` (`test_mediawiki` verifica).
- **`wikidata_entity`:** as afirmações todas, 40 por janela (`offset`), uma por linha, pela ordem
  das propriedades. Os códigos da página (propriedades, valores, unidades e as propriedades dos
  qualificadores de tempo) ganham rótulo num pedido `wbgetentities` por cada 50 códigos (o limite
  da API), na língua pedida e em inglês quando falta (`languagefallback`). No máximo quatro pedidos
  por página: um código para além disso fica sem rótulo, mas é um código que a tool lê. Valores
  pelo modelo: datas à sua precisão (`1952-03-11`, `1952-03`, `1952`, `1001 (to the century,
  Julian calendar)`), quantidades com a unidade (`1.96 metre (Q11573)`), coordenadas como
  `latitude 52.5, longitude 13.4`, texto monolingue com a língua, `unknown value`/`no value`,
  rank `[preferred]`/`[deprecated]`, qualificadores de tempo (`(point in time: 2021)`). Aceita
  também `P…`: cada código mostrado é lido pela própria tool. O QID fundido continua a funcionar.
- **`wikidata_sparql`:** guarda todas as linhas que a consulta pediu e lê-as pela janela
  (`max_results` 1-100 por janela, `offset`). Um SELECT sem `LIMIT` nos modificadores finais
  (depois do último `}`; um `LIMIT` de uma subconsulta não conta) leva `LIMIT 1000`, e o cabeçalho
  diz quando chegou a ele e como passar (ORDER BY, LIMIT e OFFSET). O paginar é local: a mesma
  consulta repete-se tal e qual, e o WDQS guarda em cache os GET (o manual: "POST queries are not
  cached"). URIs de entidades saem como código (`Q42`), literais numéricos sem notação científica.
- **Erros do SPARQL:** leitor do texto de erro: `MalformedQueryException` → `validation_error`
  com o texto do analisador; `TimeoutException` → `upstream` sem nova tentativa (a mesma consulta
  volta a esgotar os 60 s), com como estreitar. O `timeout_s` do cliente passa de 15 para 65 s,
  porque o serviço deixa uma consulta correr 60 s.
- **Erros do Special:EntityData:** a página não documenta erros além do 429 (Data_access); o 404 é
  o `missing=`.

##### `_news` (`hacker_news`)

Documentação: https://github.com/HackerNews/API e, para o erro, o Firebase
(https://firebase.google.com/docs/reference/rest/database#section-error-conditions).

- O README diz "up to 500" histórias. A tool lê a lista inteira (um pedido) e mostra `count`
  histórias desde `offset` (`Range(1, 30)` e `Range(0, 499)`), com o total. Uma história que
  falha aparece no seu lugar, com o motivo, e a numeração não salta. O nome fica (o nanope usa-o).
- O título é HTML (o README): sai com `html.unescape`. O `time` (Unix) sai em ISO 8601 UTC.
- Erros: o Firebase manda `{"error": "…"}` com o estado; o leitor comum da porta já o lê.

##### Comum

- `_numbers.plain_number` (módulo novo): números com todos os dígitos, sem notação científica,
  para floats (`repr`) e para texto decimal (`"+1.750"`, `"1.0E7"`, da Wikidata e do SPARQL).
- Todas as tools devolvem `ToolResult` (como a T05), com `metadata["window"]` quando cortam.

#### Fusões e nomes tirados

Sem aliases (D41). O nanope não usa nenhum dos nomes tirados (só o `hacker_news`, que fica).

| Tirado | Agora | Porquê |
|---|---|---|
| `eurostat_dimensions(dataset_id, max_values)` | `eurostat_dataset(dataset_id, dimension="geo")` | Faziam o mesmo pedido (`lastTimePeriod=1`) com cortes diferentes; o `eurostat_dataset` mostra todos os códigos com rótulos, página a página |
| `eurostat_compare(dataset_id, geo_codes, filters, last_time_periods)` | `eurostat_series(dataset_id, filters="geo=PT+ES+FR,…", last_time_periods=…)` | É a série com vários códigos de `geo`: o mesmo pedido (agora um só, em vez de um por país) e a mesma saída |
| `world_bank_compare(indicator, countries, year, start_year, end_year, max_points)` | `world_bank_series("PRT,ESP,DEU", indicator, start_year=…, end_year=…, max_results=…, page=…)` | A API leva vários países num pedido; a série pagina, o compare só lia a página 1. Um ano é o `start_year` sozinho |

Parâmetros que mudam de sentido ou de limites:

- `world_bank_indicators(query=…, page=…)`: `page` é a página dos resultados; a pesquisa lê sempre
  desde a primeira página do catálogo.
- `wikidata_sparql(max_results)`: linhas por janela (1-100), já não o `LIMIT` acrescentado (era
  1-20 e cortava qualquer resposta a 20 linhas).
- `eurostat_series`: `last_time_periods` 1-100 (era cortado a 20 sem aviso), `max_points` 1-100
  (era 1-50), `offset` novo.
- `gdelt_news_search(max_results)`: 1-50 por janela, até 250 por `offset` (era 1-20).
- `wikidata_search(max_results)`: 1-50 (era 1-20), `offset` novo.
- `hacker_news(count, offset)`: `offset` novo.
- `wikidata_entity(qid)` aceita também IDs `P…`, e ganha `offset`.
- `world_bank_series(country)`: um ou vários códigos; "all" sozinho.
- Os `validation_error` de `page < 1`, `skip < 0`, `scan_pages < 1` e `offset < 0` saem das tools:
  o executor recusa-os pelo `Range`.

#### Mudanças em ficheiros partilhados

- **Novo `src/ai_arch_toolkit/toolkit/tools/_numbers.py`** (`plain_number`). A T08b criou
  `_values.py` com `plain` para o mesmo assunto. São parecidos mas não iguais: o `plain` dela
  deixa o `.0` dos floats inteiros (`10578174.0`) e não converte texto decimal (`"1.0E7"`,
  `"+1.750"` ficam como vêm), que a Wikidata e o SPARQL mandam. Proposta ao coordenador: um só
  módulo, com o tratamento de texto decimal deste; as minhas tools importam-no em cinco linhas
  (`_eurostat`, `_world_bank`, `_who_gho`, `_gdelt`, `_wikidata`).
- **Novo `tests/toolkit/data_bodies.py`**: as respostas das minhas fontes para os casos do
  contrato (como o `wiki_pages.py` da T05).
- `tests/toolkit/contract_cases.py`: importa `data_bodies`; `_LOOKUPS` perde os três nomes tirados;
  `world_bank_indicator` entra em `_WHOLE` (não mexi no "38" do comentário: a T08b também mexe);
  blocos `_DATA_NEWS_WINDOWS` e `_DATA_NEWS_ZEROS` (entram com uma linha `**…` em `WINDOW_CASES` e
  `ZERO_CASES`); em `NOT_FOUND_CASES`, saem `eurostat_dimensions` e `eurostat_compare` do
  `_missing_by_404` e entram `who_indicator` e `world_bank_series`; um helper `_answers`.
- `tests/toolkit/error_bodies.py`: `_eurostat` (o 413 passa a valer para as três tools, mais o
  `warning` 413, o erro 100 e o erro 150), `_world_bank`, `_who_gho`, `_gdelt`, `_news`, e o
  `_wikidata` ganha o 429 do entity e do SPARQL e os dois erros do SPARQL; constantes
  `_EUROSTAT_GUIDE`, `_EUROSTAT_DATA`, `_WORLD_BANK_ERRORS`.
- `tests/toolkit/contract_debt.py`: saem 21 linhas (as 21 tools, as três tiradas incluídas);
  `CUT_HELPERS` 14 → 12.
- `tests/toolkit/tool_catalog.py`: saem os valores benignos `countries` e `geo_codes` (parâmetros
  das tools tiradas).
- `src/ai_arch_toolkit/toolkit/tools/__init__.py`: saem os três nomes.
- `docs/tools-catalog.md`: as secções Wikidata, News & events e Official statistics. A frase de
  abertura ("the other tools still move a numeric argument … to the nearest limit") fica para o
  coordenador.
- Não toquei em `core/`, `_http.py` nem `_window.py`.

#### Prova

Os itens da ficha e os testes (todos em `tests/toolkit/`):

- `wikidata_sparql` e o `LIMIT 100`: `test_wikidata.py::TestWikidataSparql::test_a_query_with_its_own_limit_reads_every_row`;
  o `LIMIT` acrescentado e dito: `test_a_query_without_a_limit_gets_one_and_says_when_it_is_reached`,
  `test_a_limit_inside_a_subquery_is_not_the_querys`.
- `wikidata_entity`, 15 afirmações de 20 propriedades com 3 valores: `TestWikidataEntity::test_every_statement_reads_on`;
  códigos sem rótulo, unidades, coordenadas: `test_codes_come_with_their_labels_units_and_readable_values`;
  os rótulos em lotes de 50 e no máximo quatro pedidos: `test_labels_are_asked_fifty_codes_at_a_time`,
  `test_labels_take_four_requests_at_most`; língua com recurso ao inglês:
  `test_labels_in_another_language_fall_back_to_english`; `P31`: `test_a_property_is_read_like_an_item`.
- `world_bank_compare` só na página 1: `test_world_bank.py::TestWorldBankSeries::test_several_countries_compare_and_read_on_to_page_two`.
- World Bank arredonda e usa notação científica: `test_series_values_are_whole_numbers_not_scientific_notation`
  e `test_numbers.py`.
- `gdelt_news_search` fica nos 20, sem total: `test_gdelt.py::TestGdeltNewsSearch::test_the_results_read_on_past_the_first_page`,
  `test_a_page_that_ends_the_list_has_no_next_call`, `test_fewer_articles_than_asked_are_all_there_is`,
  `test_at_gdelts_cap_the_answer_says_how_to_see_others`.
- `gdelt_timeline`, os primeiros 20 pontos: `TestGdeltTimeline::test_a_long_timeline_reads_on_to_its_latest_points`.
- `hacker_news`, 30 de cerca de 500: `test_news.py::TestHackerNews::test_the_list_reads_on_past_the_first_stories`,
  `test_the_last_stories_say_end`, `test_an_offset_past_the_list_says_so`.
- `eurostat_series`, `eurostat_dimensions` e `eurostat_compare` cortam sem continuação:
  `test_eurostat.py::TestSeries::test_observations_read_on`, `TestDataset::test_one_dimensions_codes_read_on`,
  `TestDatasetSearch::test_matches_read_on_with_the_total`.
- `eurostat_compare` e a série arbitrária: `TestSeries::test_codes_come_with_labels_and_open_dimensions_are_named`.
- Eurostat mostra códigos sem os rótulos: o mesmo, e `TestDataset::test_reads_the_details_and_every_dimensions_codes_with_labels`,
  `TestSeries::test_one_series_has_only_time_in_its_rows`.
- O filtro que não encontra nada: `test_a_query_that_matches_no_data_is_not_found`; o código fora
  do dataset: `test_a_code_the_dataset_does_not_have_is_a_validation_error`,
  `test_a_400_without_an_id_is_a_validation_error`; o `warning`:
  `test_an_asynchronous_warning_in_a_success_is_worth_a_retry`; os parâmetros repetidos:
  `test_several_codes_go_as_repeated_parameters`, `test_a_time_filter_replaces_the_last_periods`.
- Limites na assinatura: um `test_the_limits_are_in_the_schema` por módulo (pelo executor), e o
  ponto `limits` do contrato.
- `who_*` paginam sem total: `test_who_gho.py::TestWhoIndicators::test_one_more_than_shown_says_there_is_more`,
  `test_the_last_page_has_no_next_call`, `test_an_odata_next_link_also_says_there_is_more`,
  `TestWhoSeries::test_series_read_on_with_skip`.
- `who_series` mostra códigos crus: `test_rows_name_each_codes_dimension_and_the_labels_the_answer_brings`,
  `test_a_row_without_a_number_keeps_the_displayed_value`; o erro OData:
  `test_a_query_the_service_refuses_is_a_validation_error_in_its_words`.
- Erros do World Bank pela tabela: `test_an_invalid_value_on_a_list_is_a_validation_error_in_its_words`,
  `test_a_series_of_an_unknown_indicator_or_country_is_not_found`,
  `test_a_service_the_api_says_is_unavailable_is_worth_a_retry`.
- Erros do SPARQL: `test_a_query_the_service_cannot_parse_is_a_validation_error`,
  `test_a_query_past_the_services_deadline_says_how_to_narrow_it`.
- A invariante de contrato passa nas 18 tools, sem nenhuma linha de dívida; a invariante das tools
  (`test_tool_invariants.py`) passa.

Testes que afirmavam o comportamento antigo e mudaram (nenhum enfraquecido: dizem o novo
contrato):

- `test_world_bank.py`: o erro 120 numa lista era `upstream` e passa a `validation_error`; numa
  série, `not_found`; as mensagens de `not_found` levam as palavras da fonte; os testes do
  `world_bank_compare` passam para o `world_bank_series`.
- `test_wikidata.py`: a consulta mal formada era `upstream` e passa a `validation_error`; o texto do
  `wikidata_entity` e o do QID fundido mudam de forma.
- `test_eurostat.py`: o 404 com filtros dizia "dados ou dataset"; passa a dataset, pelo guia; os
  testes do `eurostat_dimensions` e do `eurostat_compare` passam para as tools que os absorvem.
- `test_gdelt.py`: "first 20 of 31 points" passa à janela; as mensagens de zero resultados mudam.
- `test_news.py`: o teste do corte calado (`count=99`) passa ao `Range`; a história que falha
  aparece no seu lugar.
- `test_who_gho.py`: as recusas de `skip` negativo passam ao `Range`.

Saldo de linhas dos seis módulos: 2369 → 2654 (+285), mais 26 do `_numbers.py`. Sobem com os
rótulos, as janelas e os leitores de erro; o World Bank desce (980 → 813).

#### Ao vivo, pelo dono

APIs gratuitas e sem chave. Cada bloco faz poucos pedidos.

```bash
uv run python - <<'EOF'
from ai_arch_toolkit.core import ToolFailure
from ai_arch_toolkit.toolkit.tools import eurostat_dataset, eurostat_series
print(eurostat_dataset("tps00001", dimension="geo").value)
for filters in ("geo=PT+ES", "geo=PT,time=1900", "geo=EU27"):
    try:
        print(eurostat_series("tps00001", filters=filters).value)
    except ToolFailure as failure:
        print(failure.error)
EOF
```

Confirmar: os rótulos dos códigos; duas linhas de `geo` (os parâmetros repetidos aceites); que
`time=1900` dá o 400 com o erro 100 (`not_found`) e `geo=EU27` o 400 com o 150
(`validation_error`), e com que corpo: é a verificação que a ficha pede. Se o Eurostat responder
outra coisa, o leitor `_eurostat_error` é o único sítio a mudar.

```bash
uv run python - <<'EOF'
from ai_arch_toolkit.core import ToolFailure
from ai_arch_toolkit.toolkit.tools import world_bank_indicators, world_bank_series
print(world_bank_series("PRT,ESP,DEU", "NY.GDP.MKTP.CD", "2015", "2023", max_results=10, page=2).value)
print(world_bank_indicators("inflation consumer prices", scan_pages=3).value)
try:
    world_bank_series("PRT", "NOT.AN.INDICATOR")
except ToolFailure as failure:
    print(failure.error)
EOF
```

Confirmar: a página 2 com os três países; valores sem notação científica; o 120 como `not_found`.

```bash
uv run python - <<'EOF'
from ai_arch_toolkit.toolkit.tools import who_indicators, who_series
print(who_indicators("life expectancy", max_results=3).value)
print(who_series("WHOSIS_000001", country="PRT", max_results=4).value)
EOF
```

Confirmar: que `ghoapi.azureedge.net` ainda responde (ver achados), o `next: skip=…` e os campos
`Low`/`High`.

```bash
uv run python - <<'EOF'
from ai_arch_toolkit.toolkit.tools import gdelt_news_search, gdelt_timeline
print(gdelt_news_search("climate", max_results=5, offset=5).value)
import time; time.sleep(6)
print(gdelt_timeline("climate", timespan="7d").value[:2000])
EOF
```

Confirmar: o `next: offset=10`; o `date_resolution` no cabeçalho do timeline; as datas em ISO. O
GDELT anónimo aceita poucos pedidos por minuto: um 429 é o esperado de vez em quando (D53).

```bash
uv run python - <<'EOF'
from ai_arch_toolkit.core import ToolFailure
from ai_arch_toolkit.toolkit.tools import wikidata_entity, wikidata_sparql
print(wikidata_entity("Q42").value[:3000])
print(wikidata_entity("Q65439041").value[:500])
print(wikidata_entity("P31").value[:800])
query = "SELECT ?item ?itemLabel WHERE { ?item wdt:P31 wd:Q146 . SERVICE wikibase:label { bd:serviceParam wikibase:language 'en'. } } LIMIT 100"
print(wikidata_sparql(query, offset=80).value)
try:
    wikidata_sparql("SELECT ?x WHERE {")
except ToolFailure as failure:
    print(failure.error)
EOF
```

Confirmar: os rótulos (o `languagefallback` do `wbgetentities`), o QID fundido, as linhas 81-100,
e o texto do erro do analisador (se o WDQS já não for Blazegraph, o `_sparql_error` lê outra
coisa: o tipo cai para o estado, 400 → `upstream`).

```bash
uv run python -c "from ai_arch_toolkit.toolkit.tools import hacker_news; print(hacker_news(count=3, offset=30).value)"
```

Confirmar: as histórias 31-33 e o `next: offset=33`.

#### Bloqueios e achados

- **A API do GHO pode estar desactivada.** A página https://www.who.int/data/gho/legacy diz que "the
  current GHO OData API will be deprecated near the end of 2025", e que a sucessora (World Health
  Data Hub OData API) tem link vazio. Não mudei o endereço (sem prova ao vivo). O dono confirma no
  comando acima; se falhar, é uma ficha nova.
- **`$count` no GHO:** nenhuma documentação o garante, por isso a tool pede uma linha a mais. Se o
  servidor aceitar `$count=true`, as listas podem passar a dizer o total.
- **A página de erros do Firebase** (o `#section-error-conditions`) não se lê pelo WebFetch: é quase
  toda navegação. Usei a forma `{"error": "…"}`, que a porta já lê.
- **Ponto 6, falta pouco:** o `who_series` mostra códigos de `Dim2` e `Dim3` que nenhuma tool
  aceita como filtro; o `wikidata_sparql` devolve códigos de lexemas (`L…`) que o
  `wikidata_entity` não aceita. Não acrescentei parâmetros: fica para decidir.
- **`_geo.py`** (T08b) tem o seu próprio `Api` do WDQS com `timeout_s=15`, abaixo dos 60 s do
  serviço.
- **Sobreposição com a T08b:** o `_values.py` e o `_first_results.py` dela resolvem casos meus
  (números; uma página de uma fonte sem offset, com um a mais). O GDELT usa aqui o mesmo desenho do
  `_first_results`, escrito no módulo; o coordenador pode juntá-los ao aplicar.
- **Interpretação:** o 400 com o erro 100 do Eurostat é `not_found` e não zero resultados, porque
  o `eurostat_series` é uma consulta de um recurso (`KINDS`: lookup) e a porta não deixa um 400
  ser sucesso sem mexer no `_http.py`.
- Sem bloqueios.

#### Correcções da revisão

2026-10-08, na checkout principal, sem commits. Cada teste foi escrito antes e visto a falhar.

##### Médios

1. **World Bank, erro 175.** `_lookup_error` passa a ler o 120 e o 175 (`_NOT_HELD`) como um
   código que o World Bank não tem: `not_found`, com o número do erro, as palavras da fonte e o
   passo seguinte. O 175 não está na tabela de erros (vem do teste do próprio repositório e da
   prova da revisão). No `_api_message`, o caso genérico (outros ids) ganha o passo seguinte:
   "check the codes and IDs given (…), or try again later". Testes:
   - `test_a_series_of_an_indicator_the_api_does_not_hold_is_not_found`;
   - `test_an_indicator_lookup_of_one_the_api_does_not_hold_is_not_found`;
   - `test_every_message_the_api_reports_is_kept_with_a_next_step` (era `…_is_kept`);
   - `test_another_error_on_an_indicator_lookup_stays_the_sources_words`, agora com o passo;
   - em `error_bodies.py`, uma entrada nova: o 175 com 200, `not_found`, para
     `world_bank_indicator` e `world_bank_series`.
2. **Wikidata, rótulos.** O `_labels` só manda ao `wbgetentities` os códigos `Q…` e `P…`. Um
   `E10` fazia falhar o lote inteiro (`no-such-entity`). Também deixei de fora os `L…`, embora a
   revisão os pusesse: um lexema tem lemas, não rótulos. Os rótulos passam a ser best effort. Um
   lote que falha (429, `maxlag`, `readonly`) deixa os seus códigos crus, e o cabeçalho di-lo:
   "(14 codes without their label; the label request failed: readonly: …)". Testes:
   - `test_only_items_and_properties_are_asked_for_labels`;
   - `test_a_later_label_request_that_fails_keeps_the_labels_already_read`;
   - `test_a_label_request_the_wiki_refuses_leaves_the_codes_bare_and_says_why`. Substitui o
     `…_says_why`, que afirmava a falha da chamada inteira.
   O `except ToolFailure` não tem `return`: o teste de arquitectura passa.
3. **SPARQL, `_with_limit`.** Novo `_shape(query)`, uma cópia da consulta com o mesmo
   comprimento, onde (pela gramática do SPARQL 1.1):
   - os comentários (de `#` ao fim da linha, fora de strings e IRIs) passam a espaços;
   - o texto das strings e dos IRIs passa a `_`.
   O `LIMIT` procura-se no shape, entre a última `}` do WHERE e uma cláusula `VALUES` final; se
   faltar, entra antes dessa cláusula. A mesma causa (uma regex sobre o texto cru) recusava
   consultas válidas em mais dois sítios, que agora também lêem o shape:
   - uma consulta que começa por um comentário ("must start with SELECT or ASK");
   - "copy" numa string, ou uma variável `?add`, no `_UNSAFE_SPARQL`. Este também deixa de ler
     `?x`, `$x` e nomes prefixados como palavras-chave.
   Testes:
   - `test_the_limit_goes_before_a_trailing_values_clause`;
   - `test_comments_strings_and_iris_are_not_the_querys_words`, com cinco casos: `LIMIT 5 # see
     {docs}`, um comentário inicial, uma string com "Copy, then DELETE", um IRI com `#`, e
     `?add ?copy`;
   - `test_a_limit_in_a_comment_is_not_the_querys`.
4. **Wikidata, qualificadores.** Cada afirmação mostra todos os qualificadores, pela ordem da
   resposta, com propriedade e valor rotulados. `_Claim.when` passa a `qualifiers`, e os códigos
   dos qualificadores entram no pedido de rótulos. Não há tecto por afirmação: os quatro pedidos
   rotulam 200 códigos, e os que passam disso ficam crus. Teste:
   `test_every_qualifier_is_shown_with_its_label` ("educated at" com início, grau e duas
   especialidades).
5. **WHO, códigos de lugar.** O `country` do `who_series` aceita `^[A-Za-z0-9_]{2,40}$`: o
   `SpatialDim` que as linhas mostram (`SEAR`, `GLOBAL`, `WB_LMI`). A docstring diz "place code".
   Pus o máximo em 40, e não nos 12 da revisão, porque o mais longo não está documentado. Só
   passam letras, algarismos e `_`, que não saem da string OData. Testes:
   - `test_every_place_code_a_row_shows_is_a_filter`, com quatro casos;
   - em `test_invalid_arguments_do_not_call_api`, `P'T` e `X` substituem `PT`, que é agora um
     código bem formado (dá zero resultados, com a consulta).

##### Baixos

6. **Eurostat, flags rotuladas.** As flags levam o rótulo de `extension.status.label`:
   `(flag p: provisional)`. Com várias flags juntas (`ep`), cada uma leva o seu; uma flag sem
   rótulo fica em código. Testes: `test_flags_come_with_the_labels_the_answer_brings` e
   `test_combined_and_unknown_flags`. **Ao vivo:** confirmar esta forma no primeiro bloco da
   ficha (`eurostat_series("tps00001", …)`).
7. **Eurostat, mensagens de erro.**
   - Num `eurostat_series` com filtros, o 404 acrescenta "(a 404 does not say whether the filters
     geo=XX were at fault: eurostat_dataset lists the codes)".
   - O erro 100 nomeia a consulta, com um leitor por chamada (`replace(_DATA,
     error_reader=…)`): "Eurostat has no data of TPS00001 for geo=PT, time=1960 (No results
     found); …".
   Testes: `test_a_404_with_filters_names_the_dataset_and_the_filters`; e, actualizados,
   `test_a_query_that_matches_no_data_is_not_found` e `test_a_dataset_eurostat_does_not_have_is_not_found`
   (este passa a `startswith`).
8. **Eurostat, filtros.** `format` e `lang`, em qualquer caixa, são recusados como filtros
   (`_FIXED_KEYS`); os códigos de `geo` passam a maiúsculas. Testes: dois casos em
   `test_invalid_arguments_do_not_call_api`, e `test_geo_codes_are_upper_case`.
9. **Não feito.** A cache por processo precisava de duas coisas:
   - uma validade, porque o catálogo muda;
   - um reset entre testes, num `conftest` fora do meu âmbito.
   E o risco do prazo de 120 s já existe na primeira chamada, com o `Range(1, 30)` do
   `scan_pages`. Proposta ao dono: guardar a última pesquisa alguns minutos, com a chave
   `(query, topic, source, scan_pages)`, e fazer o reset em `tests/conftest.py`.
10. **World Bank, rodapés.** O rodapé diz `next: page=N+1, max_results=K`. Teste:
    `test_the_next_page_keeps_the_pages_size`, e cinco rodapés actualizados. `_open_food_facts`,
    `_rxnorm_dailymed` e `_internet_archive` seguem a convenção antiga; não são meus.
11. **`_values`.** O `decimal_text` tem um limite (`_MAX_EXPONENT = 100`): além dele, fica o
    texto da fonte. O `plain(True)` escreve `true`.
12. **Eurostat, janela a meio de uma dimensão.** Quando a janela começa entre os códigos de uma
    dimensão, o cabeçalho nomeia-a: "codes (geo: Geopolitical entity, continued):". Testes:
    `test_one_dimensions_codes_read_on` (actualizado) e
    `test_a_window_that_starts_on_a_dimensions_line_needs_no_name`.
13. **SPARQL, mensagem do analisador.** A mensagem fica inteira: é uma linha, e a porta já limita
    o corpo de erro. Teste: `test_the_parsers_message_is_kept_whole`.

##### Formatadores (`_values.py`)

- **Conflito: o `.0` fica.** Tirar o `.0` aos floats inteiros parte 7 testes da T08b, que não
  posso editar:
  - `test_geo.py::TestGeocode::test_lists_the_places_labelled_with_signed_coordinates`
    (`182.0 m`);
  - dois em `test_weather.py`: `TestThePoint::test_coordinates_need_no_geocoding_and_name_the_point`
    (`21.0 °C`) e `TestUnitsAndForecast::test_imperial_units_are_asked_of_open_meteo_and_shown_as_it_names_them`
    (`0.0 mm`);
  - três em `test_eonet.py` (`latitude 16.0`, `magnitude 36.0 kts`, `38.0 to 38.4`);
  - `test_earthquake.py::TestEarthquakeEvent::test_an_event_with_its_details_and_times_in_utc`.
  Por isso a regra única é a contrária: um float mantém o ponto que o seu número JSON tinha
  (`21.0`), e um inteiro fica inteiro. Agora também sem excepções: o `plain` antigo escrevia
  `1e16` como `10000000000000000`, e agora escreve `10000000000000000.0`. O `-0.0` sai `0.0`.
- **Para tirar o `.0` mais tarde,** quando a T08b acabar: uma linha no `plain`, os 7 testes
  acima e três casos de `test_values.py`.
- **`decimal_text(text)`** é novo: serve os números que chegam como texto (`"+1.750"` → `1.750`,
  `"1.0E7"` → `10000000`). Só aceita algarismos ASCII, tem o limite de expoente, e não entra no
  `plain` (`"007"` fica `007` no `plain`).
- **As T08a no `_values`:**
  - `_eurostat` e `_world_bank`: número → `plain`, texto → `decimal_text`;
  - `_who_gho`: `plain`;
  - `_gdelt`: `plain`; o `_float_or_none` passa a `_number`, que mantém inteiro um inteiro;
  - `_wikidata`: quantidades e literais do SPARQL → `decimal_text`, coordenadas → `plain`;
  - `_news`: o `_iso` usa o `utc`.
  Saídas que mudam: um float inteiro do World Bank ou do Eurostat lê-se agora
  `29184912345600.0` ou `15000000.0` (dois testes actualizados).
- **`_numbers.py` e `test_numbers.py` saem da árvore de trabalho.** No índice continuam "A": o
  `git rm` fica para o coordenador. Os testes juntam-se em `tests/toolkit/test_values.py` (novo,
  por adicionar), com a `TestValues` que estava em `test_first_results.py`. O agente da T08b já a
  tinha tirado de lá, com uma nota a apontar para `test_values.py`.

##### GDELT no `_first_results`

Cabe. O `maxrecords` é `asked(offset, max_results, 250)`, e a página sai de
`first_results_window(…, requested=asked(…), depth=250, narrow=…)`. O `_first_results.py` mudou
enquanto eu trabalhava: o agente da T08b acrescentou o `requested=` e o modo "a profundidade toda"
para o Nominatim e o Open-Meteo. Não lhe toquei. O GDELT continua a pedir "a página e mais um",
como antes.

A única mudança visível: no tecto dos 250, a nota passa do cabeçalho para o fim da página, "(the
source returns no more than 250 results; narrow the timespan or the query, or change the sort, to
see others)". Teste: `test_at_gdelts_cap_the_answer_says_how_to_see_others` (actualizado); os
outros quatro testes de paginação passam como estavam.

Opção para o dono: pedir sempre os 250. Todas as páginas sairiam da mesma resposta, com total
abaixo dos 250, mas cada resposta seria maior.

##### Verificação

- Os testes dos meus módulos, dos valores, da T08b (com o overpass), do contrato e das
  invariantes: 1255 passam.
- `tests/toolkit` inteiro, com `test_architecture` e `test_quality_budget`: 2651 passam, 2 xfail.
- `uv run pyright src`: 0 erros. Ruff (`check` e `format --check`) limpo nos meus ficheiros.
  Não corri `ruff format` em `contract_cases.py` nem em `error_bodies.py`.

##### Mudanças propostas ao CHANGELOG (bloco "Data and news (T08a)")

Substituir a linha do `wikidata_entity` por:

```markdown
  - `wikidata_entity` reads every statement, 40 at a time, each code with its label
    (`instance of (P31): human (Q5)`), quantities with their unit, dates to their precision,
    coordinates as latitude and longitude, ranks and every qualifier; it takes property IDs too.
    Labels are best effort: a label request that fails leaves its codes bare, and says why.
```

No fim da linha do `wikidata_sparql`, acrescentar:

```markdown
    Comments, strings and IRIs are not read as the query's words: a trailing `VALUES` clause
    gets the `LIMIT` before it, and a query that starts with a comment or has a variable such as
    `?add` is accepted.
```

Em "`who_series` names each code's dimension", acrescentar ", and takes any place code its rows
show (`SEAR`, `GLOBAL`, `WB_LMI`) as `country`".

Substituir a linha dos erros por:

```markdown
  - Eurostat errors follow its guide: no data (error 100) is `not_found`, naming the query; a
    code the dataset does not have is a `validation_error`. World Bank errors follow its table:
    an unknown indicator or country is `not_found`, and so is error 175 (an indicator deleted
    or archived); error 105 is worth a retry; any other error says what to check.
  - `eurostat_series` gives each flag its label (`p: provisional`), upper-cases `geo` codes and
    refuses `format` and `lang` as filters; `eurostat_dataset` names the dimension a window of
    codes continues. A World Bank list's next call keeps its `max_results`.

### T08b

- **Dono:** Claude (agente T08b), 2026-10-08 · **Estado:** review (feito na worktree, por commitar)
- **Worktree:** `/Users/rge/Documents/dev/pessoal/ai-arch-toolkit/ai-arch-toolkit/.claude/worktrees/agent-ae53281a80e2c1a45`
- **Módulos:** `_overpass` (2), `_osm` (2), `_geo` (6), `_weather` (5), `_air_quality` (2), `_eonet` (3),
  `_earthquake` (3): 23 tools à entrada, 19 à saída (quatro fundidas, D41).
- **Resultado:** as 21 linhas destas tools saem de `contract_debt.py`; as 19 tools que ficam cumprem
  os cinco pontos do contrato. `CUT_HELPERS` não muda (14): nenhum destes módulos tinha `_truncate`
  ou `_trim`.

#### Nota de desenho

##### Peças comuns (módulos novos, sem tools)

- **`_open_meteo.py`:** os três `Api` do Open-Meteo (`GEOCODING`, `FORECAST`, `AIR_QUALITY`), um só
  leitor de erros (`open_meteo_error`), a pesquisa de lugares (`places`, `Place`) e `measured` (um
  valor com a unidade que a resposta lhe dá). Antes, o `_geo` e o `_weather` declaravam cada um o
  seu `_GEOCODING` e o seu `_FORECAST`, e só o `_air_quality` lia os erros. O Open-Meteo recusa um
  pedido com HTTP 400 e `{"error": true, "reason": …}` nas três APIs
  (https://open-meteo.com/en/docs, https://open-meteo.com/en/docs/geocoding-api,
  https://open-meteo.com/en/docs/air-quality-api, secção "Errors"): é `validation_error`, com o
  motivo e "correct the argument it names".
- **`_first_results.py`:** a página de uma fonte que não tem offset (o Open-Meteo dá até 100 lugares,
  o Nominatim até 40, o EONET tantos eventos quantos o `limit` pedir). A tool pede, desde o início,
  um resultado a mais do que a página acaba (`asked`), e mostra `limit` a partir de `offset` pela
  janela (`first_results_window`). O resultado a mais diz se há mais; o total sabe-se quando a fonte
  devolve menos do que foi pedido. No fundo da fonte, a página diz como estreitar a consulta.
- **`_values.py`:** `plain` (números sem notação científica, tal como a fonte os mandou) e `utc`
  (um instante em ISO 8601 UTC).

##### `_open_meteo` + `_weather` (fusão: ver abaixo)

- **`get_weather(city="", latitude=None, longitude=None, units="metric")`** e
  **`get_forecast(city="", latitude=None, longitude=None, days=3, units="metric")`**.
- **Lugar:** com coordenadas, o pedido vai direito à previsão e a `city` só dá nome ao ponto; sem
  elas, a geocodificação pede `count=2`. Fica com o primeiro, como antes, mas agora di-lo quando há
  outros com o mesmo nome e diz como escolher outro (`geocode(...)` e as coordenadas). Coordenadas
  fora de alcance, ou só uma das duas, dão `validation_error` antes de qualquer pedido. Um nome sem
  lugar dá `not_found`, com o passo seguinte.
- **Unidades:** o Open-Meteo converte (`temperature_unit`, `wind_speed_unit`,
  `precipitation_unit`, https://open-meteo.com/en/docs) e nomeia cada unidade
  (`current_units`, `daily_units`); a tool mostra cada valor com a unidade que veio. Desaparecem
  as conversões feitas à mão (`_c_to_f`, `_kmh_to_mph`).
- **Limites:** `days` é `Range(1, 16)`, como a documentação ("Up to 16 days"); antes cortava-se a 7
  sem dizer. `units` é `Literal["metric", "imperial"]` (enum no schema) e é validado no código.
- **Saída:** o código WMO com o rótulo ("Partly cloudy (WMO code 2)"), a tabela completa da
  documentação (faltavam 56 e 57); o tempo actual em ISO 8601 UTC (`timeformat=unixtime`), com o
  fuso e o seu desvio no cabeçalho; as datas da previsão são dias locais (`YYYY-MM-DD`).

##### `_geo`

- **`geocode(city, max_results=10, offset=0)`:** `Range(1, 100)` e `Range(0, 99)`; mostrava sempre
  3. Pede `count` = página + 1 (até 100, o máximo da API, que não tem offset). Cada lugar diz região,
  país com código, `latitude`/`longitude` com sinal (antes "-9.1°E"), fuso, população e altitude.
  Zero resultados é sucesso, com o nome e o passo seguinte (`osm_search_place`).
- **`timezone_lookup`:** o leitor do Open-Meteo; acrescenta a abreviatura do fuso; uma resposta sem
  fuso passa a `upstream` (era "No timezone found", como se fosse sucesso).
- **`ip_lookup`:** um leitor próprio (`_ipwhois_error`). O ipwho.is responde `success: false` com
  HTTP 200 para um endereço que não localiza ("Reserved range", "Invalid IP address") e com 4xx no
  resto ("Rate limit exceeded" com 429) (https://ipwhois.io/documentation, "Errors"). Um endereço
  privado ou reservado passa a `validation_error` com o passo seguinte (era `upstream`, que se lê
  "tenta mais tarde"). Saída com o código do país e o desvio UTC.
- **`country_info`:** sem mudança de código; o `_WIKIDATA` mantém `error_reader=mediawiki_error`
  (`test_mediawiki.py`). Ganha os casos de contrato (erros MediaWiki e `not_found`).
- **`distance_between`:** sem mudança.
- **`reverse_geocode`:** sai, fundida em `osm_reverse_geocode`.

##### `_air_quality`

- `air_quality_forecast` ganha `offset` e passa pela janela (`page_window`): "[results 1-72 of 336
  | next: offset=72]". Antes mostrava até 72 de 336 linhas e o resto perdia-se.
- Limites na assinatura: `forecast_days` `Range(1, 7)`, `past_days` `Range(0, 92)` (a documentação
  dá 0-92; a tool parava em 7), `max_hours` `Range(1, 72)`, `offset` `Range(0)`.
- Horas em ISO 8601 UTC (`timeformat=unixtime`): uma hora local escondia a mudança de hora a meio
  da previsão; o fuso do ponto, que marca onde começa cada dia, vai no cabeçalho.
- O leitor passa a ser o comum (`open_meteo_error`).

##### `_osm` (Nominatim)

- **`osm_search_place(query, max_results=5, offset=0, …)`:** `Range(1, 40)` e `Range(0, 39)`; a API
  permite 40 (https://nominatim.org/release-docs/latest/api/Search/, `limit`: "Cannot be more than
  40"); a tool parava em 10. Pede `limit` = página + 1. Considerei o `exclude_place_ids`, que a
  documentação dá como caminho para mais resultados, mas o cursor cresce a cada página; fico pelos
  40 e, no fundo, a página diz como estreitar (país, camada, consulta mais precisa).
- **`osm_reverse_geocode(latitude, longitude, zoom=18, …)`:** `zoom` `Range(0, 18)`, como a API
  (https://nominatim.org/release-docs/latest/api/Reverse/); a docstring dá a tabela (18 edifício,
  10 cidade, 3 país). Morada completa, na ordem do Nominatim, e todas as etiquetas extra (eram as
  primeiras 8 de 12): a tool deixa de cortar e passa para `WHOLE`.
- **Leitor (`_nominatim_error`):** o Nominatim recusa com `{"error": {"code": 400, "message": …}}`
  (`_format_error` em `src/nominatim_api/v1/format.py` e `raise_error` em
  `src/nominatim_api/server/asgi_adaptor.py`, https://github.com/osm-search/Nominatim): 400 é
  `validation_error` com a mensagem. O "nada aqui" de um reverse é HTTP 200 com
  `{"error": "Unable to geocode"}` (a documentação diz "exactly one result or an error"; o texto
  está em `test/bdd/features/api/reverse/v1_json.feature`): fica sucesso, "No OpenStreetMap place
  at …". Outro texto de erro num 200 é `upstream` com as palavras da fonte.
- **Saída:** o objecto OSM `relation/540` (o `overpass_query` lê-o: `relation(540);out;`), a caixa
  rotulada (south, north, west, east, pela ordem da documentação
  https://nominatim.org/release-docs/latest/api/Output/), a atribuição ODbL uma vez no cabeçalho.
  Sai a `importance` (formatada com `.4g`, notação científica possível).

##### `_overpass`

- As duas tools ganham `offset` e passam pela janela (`page_window`): o Overpass responde inteiro,
  sem paginação, e a continuação corre a consulta de novo. `max_results` `Range(1, 50)`, `radius_m`
  `Range(1, 50000)` (estes limites eram validados ou cortados à mão).
- O leitor (`remark` "runtime error" num 200, páginas HTML de erro) fica como estava
  (https://dev.overpass-api.de/overpass-doc/en/preface/commons.html,
  https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL). Considerei pedir o total com
  `out count;` na mesma consulta das `overpass_pois`, mas não o posso provar sem uma chamada ao
  vivo; fica para o dono, se quiser.
- Zero resultados diz a consulta (ou a etiqueta e a área). Coordenadas rotuladas; uma `way` ou
  `relation` usa o `center`.

##### `_eonet`

- **`eonet_events`:** `days` `Range(1, 365)` (o 365 era um corte nosso, calado; a documentação não dá
  máximo, https://eonet.gsfc.nasa.gov/docs/v3) e a docstring manda usar datas para mais; `max_results`
  `Range(1, 50)`; `offset` `Range(0, 999)`. O EONET não tem offset: pede `limit` = página + 1, até
  1000, e no fundo diz para estreitar o período, a categoria ou a caixa.
- **Erro encontrado (corrigido):** o `bbox` ia tal como a tool o recebe (west,south,east,north), mas
  o EONET lê "min lon, max lat, max lon, min lat" (o canto superior esquerdo e o inferior direito):
  a latitude ia trocada. A tool converte para a ordem do EONET.
- As datas vão como `YYYY-MM-DD` (o `date.fromisoformat` aceita `20260901` e a tool mandava-o assim).
- **`eonet_event(event_id, offset=0, max_points=50)`:** o percurso inteiro, uma posição datada por
  linha, pela janela (antes, só o último ponto); fontes todas (eram 5), descrição inteira, estado
  ("closed …"). Polígonos mostram a extensão.
- **Leitor:** o 500 que o EONET dá a um ID desconhecido (visto a 2026-09-30) é lido por um `Api` só
  do evento (`_EVENT_API`, `error_reader=_unknown_event`); continua `upstream` e retryable, com o que
  pode querer dizer. Assim nenhuma tool lê o estado de uma falha apanhada: o `_STATUS_READERS` de
  `tests/test_architecture.py` fica vazio.
- Coordenadas GeoJSON (longitude primeiro) passam a rotuladas; datas já vinham em ISO 8601 UTC.

##### `_earthquake` (USGS)

- **204 confirmado na documentação:** `nodata` vale 204 por omissão
  (https://earthquake.usgs.gov/fdsnws/event/1/; e a especificação FDSN, "No data selected",
  https://www.fdsn.org/webservices/FDSN-WS-Specification-Commonalities-1.2.pdf). A porta tratava o
  204 como "could not parse". Agora `allow_empty=True` na página e no evento: uma página vazia é
  "no results from N of T", um evento vazio é `not_found`.
- **O "count" era o da página (suspeita confirmada pelo desenho):** a documentação não define o
  `metadata.count` como total. A pesquisa pede primeiro o método `count` com os mesmos filtros
  (documentado: "uses the same parameters as the query method") e depois a página; o rodapé dá o
  total ("[results 1-10 of 1234 | next: offset=11]"). Com total 0 não pede página.
- **Limites:** `max_results` `Range(1, 50)`, `offset` `Range(1)` (1-based, como a API: mantive para
  não mudar em silêncio o sentido do argumento).
- **Leitor (`_fdsn_error`):** o texto FDSN ("Error 400: …", o detalhe, "Usage details are available
  from …"). 400 e 413 (pedido grande demais) são `validation_error` com o detalhe; 409 (evento
  apagado, ver `includedeleted` na documentação) é `not_found` e diz para procurar o que o
  substituiu.
- **Saída:** tempos em ISO 8601 UTC (eram milissegundos epoch; a página do GeoJSON só diz "Long
  Integer", https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php, e o valor do fixture,
  1710000000000, é de 2024-03-09); magnitude com o tipo (`mww`); latitude, longitude e profundidade
  em km rotuladas; no evento, estado de revisão, relatos sentidos, alerta PAGER, bandeira de
  tsunami, significância, actualização e página. As datas vão como `YYYY-MM-DD`.
- `earthquake_count` diz o que contou.

#### Fusões e nomes tirados

Decididas pela D41 (uma tool por trabalho e por fonte), sem aliases:

| Antes | Agora |
|---|---|
| `reverse_geocode(lat, lon)` | `osm_reverse_geocode(latitude, longitude, zoom=10)` |
| `get_weather_by_coords(lat, lon)` | `get_weather(latitude=lat, longitude=lon)` |
| `weather_units(city, unit="f")` | `get_weather(city, units="imperial")` |
| `get_forecast_by_coords(lat, lon, days)` | `get_forecast(latitude=lat, longitude=lon, days=days)` |

- **Porquê:** o `reverse_geocode` (zoom 10) e o `osm_reverse_geocode` (zoom 18) eram o mesmo pedido
  ao Nominatim com respostas diferentes; o nível passa a argumento. As três tools de tempo actual
  faziam o mesmo trabalho com argumentos diferentes, e a de coordenadas não validava nem guardava o
  motivo da recusa. Fica o prefixo `osm_` para a família Nominatim.
- **Não decidi:** renomear a família Open-Meteo para um prefixo comum (`geocode`, `get_weather`,
  `get_forecast`, `timezone_lookup`, `air_quality_*`). O `get_weather` e o `geocode` estão no
  README, em cinco exemplos e em três docs; a quebra seria grande para pouco ganho. Fica para o dono.
- **nanope:** importa pelo nome `reverse_geocode`, `get_weather_by_coords`, `get_forecast_by_coords`
  e `weather_units`, que saem. O nanope já não importa desde a T05 (os testes de `tests/nanope`
  saltam por `pending.py`); estes quatro nomes juntam-se à lista que o dono troca. Não toquei em
  `src/ai_arch_toolkit/nanope/` nem em `tests/nanope/`. O texto de `tests/nanope/pending.py` só
  fala da família wiki.

#### Mudanças em ficheiros partilhados

- `tests/test_architecture.py`: só o `_STATUS_READERS`, que fica `set()` (o `_eonet` deixou de ler
  o estado de uma falha apanhada; o comentário diz porquê).
- `tests/toolkit/contract_cases.py`: as minhas entradas em `WINDOW_CASES`, `NOT_FOUND_CASES` e
  `ZERO_CASES`; as quatro tools tiradas saem de `_OTHERS` e `_WHOLE`; `osm_reverse_geocode` entra
  em `_WHOLE`; o comentário do `_WHOLE` perde o "38". Importa `geo_answers` (novo).
- `tests/toolkit/error_bodies.py`: entradas `_geo`, `_weather`, `_osm`, `_overpass`, `_eonet`,
  `_earthquake`, e um helper `_open_meteo`. O `_air_quality` fica como estava (passa com o leitor
  comum).
- `tests/toolkit/contract_debt.py`: saem 21 linhas (as minhas, as quatro tiradas incluídas).
- `tests/toolkit/tool_catalog.py`: sai `("weather_units", "unit")`.
- `src/ai_arch_toolkit/toolkit/tools/__init__.py`: saem os quatro nomes.
- **Módulos novos que os irmãos podem querer:** `_first_results.py` (a página de uma fonte sem
  offset: o GDELT, com 250, e o Hacker News, com ~500, da T08a, têm o mesmo caso) e `_values.py`
  (números sem notação científica, como o World Bank precisa). Não toquei em `_http.py`, `_window.py`
  nem em `core/`.
- `docs/tools-catalog.md`: as secções "Weather, geo & places" e "Natural events". A frase de
  abertura ("the other tools still move a numeric argument … to the nearest limit") fica para o
  coordenador, que junta as frentes.

#### Prova

Testes novos ou reescritos (148 nos ficheiros dos módulos, mais os casos de contrato), por item
da ficha:

- **`osm_search_place` parava em 10 (a API dá 40):**
  `test_osm.py::test_more_than_ten_places_page_on_up_to_the_forty_nominatim_gives`,
  `::test_the_forty_is_in_the_schema`; caso de janela e de zero no contrato.
- **`geocode` mostrava sempre 3:** `test_geo.py::TestGeocode::test_shows_more_than_three_and_pages_on`,
  `::test_the_next_page_says_the_total_when_the_source_has_no_more`,
  `::test_past_the_hundred_places_it_says_how_to_narrow`; janela e zero no contrato.
- **As tools de tempo ficavam caladas com o primeiro lugar:**
  `test_weather.py::TestTheCity::test_a_city_shared_by_several_places_says_which_one_and_how_to_pick_another`.
- **`air_quality_forecast` mostrava até 72 de 336:**
  `test_air_quality.py::test_every_hour_is_reachable_through_the_window`; janela no contrato.
- **`overpass_*` e `eonet_*` cortavam sem continuação; percurso reduzido ao último ponto:**
  `test_overpass.py::test_every_element_is_reachable_through_the_window`,
  `test_eonet.py::test_events_past_the_page_are_reachable`,
  `::TestEonetEvent::test_the_whole_track_is_read_not_only_its_last_point`; três casos de janela.
- **O limite de 365 dias do EONET fora da assinatura:** `test_eonet.py::test_the_365_days_are_in_the_schema`,
  `::test_more_days_are_refused_not_cut`.
- **O "count" do `earthquake_search`:** `test_earthquake.py::test_the_total_is_the_count_methods_not_the_pages`,
  `::test_the_next_page_numbers_on_from_the_offset`; janela no contrato (count e página por chamada).
- **Tempos `earthquake_*` em milissegundos:** `::test_times_are_iso_8601_utc_and_positions_are_labelled`,
  `::TestEarthquakeEvent::test_an_event_with_its_details_and_times_in_utc`.
- **USGS 204:** `::test_a_page_with_no_content_is_an_empty_page_not_a_parse_error`,
  `::test_no_content_for_an_event_is_not_found`, `::test_no_events_is_a_success_that_names_the_query_without_asking_for_a_page`.
- **Fusão `reverse_geocode`/`osm_reverse_geocode`:** `test_osm.py::test_zoom_ten_answers_what_reverse_geocode_did_the_city`,
  `::test_the_zoom_nominatim_takes_is_in_the_schema`, `::test_nothing_at_the_point_is_a_success_that_says_so`.
- **Fusão das três tools de tempo actual; a de coordenadas não validava e perdia o motivo:**
  `test_weather.py::test_the_three_current_weather_tools_are_one`,
  `::TestThePoint::test_bad_coordinates_are_refused_before_any_request`,
  `::test_open_meteos_reason_for_a_refusal_reaches_the_agent`, `::test_with_coordinates_a_city_only_names_the_place`.
- **Achados de caminho:** `test_eonet.py::test_the_box_goes_in_eonets_corner_order`, `::test_dates_go_as_yyyy_mm_dd`,
  `test_earthquake.py::test_a_deleted_event_is_not_found_and_says_so`, `::test_too_much_data_says_to_narrow`,
  `test_geo.py::TestIpLookup::test_a_reserved_address_is_the_callers_to_change`,
  `test_osm.py::test_a_parameter_nominatim_refuses_is_a_validation_error_in_its_words`.
- **Peças comuns:** `test_first_results.py` (a página com um a mais, o total, o fundo; números e
  instantes).
- **Contrato:** `test_tool_contract.py` passa (139 testes, sobre as 126 tools que ficam); as 19
  desta ficha cumprem os cinco pontos e a dívida desce de 113 para 92 entradas, com corpos de erro
  da documentação de cada fonte (`error_bodies.py`).
- **Vermelho antes:** corri os testes novos sobre os módulos de `HEAD` (repostos e devolvidos
  depois): 91 falhas, todas pela razão certa (saídas antigas, janela em falta, limites cortados,
  `bbox` trocado, 204 como erro de leitura, nomes que ainda existiam).

**Saldo de linhas (src):** os sete módulos passam de 2011 para 2301 linhas, mais 216 nos três
módulos novos (+506). Não é negativo: entram as janelas, o percurso do EONET, o total da USGS,
cinco leitores de erros e as docstrings com as fontes; saem os `_GEOCODING`/`_FORECAST` duplicados,
as conversões de unidades à mão, os cortes e os clamps.

#### Ao vivo, pelo dono

APIs gratuitas e sem chave (Open-Meteo, Nominatim, Overpass, EONET, USGS, ipwho.is). Cada comando
faz um ou dois pedidos. Para ver uma falha tipada, o executor imprime-a:

```bash
uv run python -c "from ai_arch_toolkit.toolkit.tools import geocode; print(geocode('Springfield', max_results=5).value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import get_weather; print(get_weather('Springfield'))"
uv run python -c "from ai_arch_toolkit.toolkit.tools import get_forecast; print(get_forecast(latitude=38.72, longitude=-9.14, days=3, units='imperial'))"
uv run python -c "from ai_arch_toolkit.core import ToolCall, ToolGroup; from ai_arch_toolkit.toolkit.tools import air_quality_forecast as t; print(ToolGroup(t).execute(ToolCall(id='c', name=t.__name__, input={'latitude': 38.72, 'longitude': -9.14, 'timezone': 'Mars/Olympus'})).to_model_text())"
uv run python -c "from ai_arch_toolkit.toolkit.tools import air_quality_forecast; print(air_quality_forecast(38.72, -9.14, forecast_days=7, past_days=7, max_hours=72, offset=288).value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import osm_search_place; print(osm_search_place('Springfield', max_results=10, offset=30).value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import osm_reverse_geocode; print(osm_reverse_geocode(38.6916, -9.2160, zoom=10)); print(osm_reverse_geocode(0, -30))"
uv run python -c "from ai_arch_toolkit.toolkit.tools import overpass_pois; print(overpass_pois('amenity', 'cafe', latitude=38.7139, longitude=-9.1394, radius_m=500, max_results=5).value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import eonet_events; print(eonet_events(bbox='-125,32,-114,42', status='all', days=365, max_results=3).value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import eonet_events; print(eonet_events(category='severeStorms', status='all', days=120, max_results=3).value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import earthquake_search; print(earthquake_search(start_time='2024-01-01', end_time='2024-01-31', min_magnitude=6, max_results=3).value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import earthquake_search; print(earthquake_search(start_time='2024-01-01', end_time='2024-01-31', min_magnitude=6, offset=500).value)"
uv run python -c "from ai_arch_toolkit.core import ToolCall, ToolGroup; from ai_arch_toolkit.toolkit.tools import earthquake_search as t; print(ToolGroup(t).execute(ToolCall(id='c', name=t.__name__, input={'max_depth_km': 5000.0})).to_model_text())"
uv run python -c "from ai_arch_toolkit.core import ToolCall, ToolGroup; from ai_arch_toolkit.toolkit.tools import ip_lookup as t; print(ToolGroup(t).execute(ToolCall(id='c', name=t.__name__, input={'ip': '10.0.0.1'})).to_model_text())"
uv run python -c "from ai_arch_toolkit.toolkit.tools import timezone_lookup; print(timezone_lookup(38.72, -9.14))"
```

Depois do segundo `eonet_events`, ler um evento com
`uv run python -c "from ai_arch_toolkit.toolkit.tools import eonet_event; print(eonet_event('<ID>', max_points=5).value)"`.

O que confirmar:

- o Open-Meteo sem resultados deixa `results` de fora (a tool aceita as duas formas), e o `current.time`
  vem em segundos com `timeformat=unixtime`;
- o motivo do 400 do Open-Meteo para um fuso inválido chega ao texto;
- o Nominatim aceita `limit=40`; o reverse no mar dá "No OpenStreetMap place";
- o `bbox` do EONET devolve eventos da Califórnia (a ordem corrigida); se vier vazio, a ordem da
  documentação está errada;
- o `count` da USGS em texto simples; a página depois do total dá 204 e lê-se "no results from 500
  of N"; o 400 de `maxdepth=5000` traz o detalhe FDSN; o `metadata.count` de uma página com
  `limit=3` é 3 (a suspeita);
- o `center` das `way` no Overpass;
- o ipwho.is responde "Reserved range" com HTTP 200 para 10.0.0.1.

#### Bloqueios e achados

- **Sem bloqueios.**
- **Achado corrigido (fora da ficha, mas no módulo):** o `bbox` do EONET ia com as latitudes
  trocadas (ver `_eonet`). Proponho que conste em `Fixed`.
- **Achado, não corrigido:** o `earthquake_search` e o `earthquake_count` mandam `minmagnitude=0.0`
  por omissão, o que deixa de fora os sismos de magnitude negativa (a USGS não filtra sem o
  parâmetro). É uma promessa da docstring ("Defaults to 0"), não um corte calado; fica para o dono
  decidir se o valor por omissão deve ser "nenhum".
- **Achado, não corrigido:** o `osm_search_place` e o `geocode` não limitam o comprimento da
  consulta do lado do Nominatim (o `geocode` limita a 200 caracteres); uma consulta enorme dá o erro
  do servidor (414 ou rede), sem passo seguinte. Menor.
- **Achado para a T08a/coordenador:** o `_first_results.py` e o `_values.py` resolvem casos que a
  T08a também tem (GDELT 250, Hacker News, notação científica do World Bank). Se a T08a fizer os
  seus, o coordenador pode juntá-los.
- **Fontes que não consegui ler:** o glossário de termos da USGS
  (https://earthquake.usgs.gov/data/comcat/data-eventterms.php) não devolveu as definições ao
  WebFetch; a unidade do `time` (milissegundos) vem do fixture existente e da ficha (anexo D), e
  fica nos comandos ao vivo.
- **Nenhum pedido às APIs:** só páginas de documentação e o código-fonte do Nominatim no GitHub;
  nenhum dado pessoal em pedidos.

#### Correcções da revisão

Feitas na checkout principal (2026-10-08), sobre `notes/T08b-review.md`. Para cada achado, o teste
de comportamento foi escrito primeiro e visto a falhar pela razão certa. Só mexi nos nove módulos,
nos seus testes, em `geo_answers.py` e num comentário das minhas entradas de `contract_cases.py`.
O `_values.py` é da T08a e só o importo. A T08a mudou duas vezes a regra de `plain` para os floats
inteiros enquanto eu trabalhava (sem `.0`, depois de novo com `.0`), por isso os testes que mostram
um float inteiro esperam `plain(10.0)` e não um texto fixo. O `TestValues` saiu de
`test_first_results.py`: o `_values` tem agora os seus testes em `test_values.py`, que são da T08a.

##### 1. O último dia do `earthquake_*` ficava de fora (médio-alto)

- **Mudança:** o `_period` manda `endtime` como o último microssegundo do dia
  (`2024-01-31T23:59:59.999999`), tanto no `count` como na página. A USGS lê uma data sem hora como
  00:00:00 (https://earthquake.usgs.gov/fdsnws/event/1/) e o `endtime` selecciona "on or before".
  Os subsegundos, até ao microssegundo, vêm da especificação FDSN (Commonalities 1.2, "Time
  parameter values"). A página da USGS não os menciona, por isso fica nos comandos ao vivo.
  `start_time == end_time` é agora um dia inteiro. As docstrings dizem "included".
- **Testes:** `test_earthquake.py::TestEarthquakeSearch::test_the_end_day_is_included` (o `count`
  e a página levam o mesmo `endtime`, e o cabeçalho também o mostra) e
  `::TestEarthquakeCount::test_the_end_day_is_counted`.

##### 2. O `overpass_query` aceitava consultas que não podia ler (médio)

- **Mudança:** um `_check_query` recusa com `validation_error`, antes de qualquer pedido, nestes
  casos:
  - uma consulta sem `[out:json]` (regex com espaços opcionais), porque XML ou CSV davam um
    `upstream` "could not parse";
  - uma consulta sem instrução `out`, porque o Overpass não devolvia elementos e a tool dizia "No
    OpenStreetMap elements match".

  A instrução `out` é reconhecida a seguir a `;` ou `{` e aceita o conjunto de entrada (`.a out`).
  Cada mensagem traz o exemplo a seguir.
- **Testes:**
  - `test_overpass.py::test_a_query_the_tool_cannot_read_is_refused_before_any_request`: sem `out`,
    "Shout" num valor, `[out:xml]`, `[out:csv(...)]`, sem definições;
  - `::test_every_form_of_the_out_statement_is_taken`: `out;`, `out center tags;` depois de uma
    união, `[ out : json ]` com `.a out body;`, `out` com mudança de linha, `foreach{out;}`.

##### 3. Nominatim e Open-Meteo pedem sempre o fundo (médio)

- **Mudança:** o `first_results_window` ganha `requested=`, o número pedido à fonte, de que sai o
  total. Antes calculava-o com `asked()`.
  - O `osm_search_place` pede sempre `limit=40` e o `geocode` pede sempre `count=100`: cada página
    sai da mesma resposta, e abaixo do fundo o total é exacto.
  - O EONET continua com `asked()`, uma página e mais um, porque é estável e o fundo (1000) pesa.
  - A docstring do módulo explica os dois modos e cita o código do Nominatim (`v1/server_glue.py`,
    `geocoder.py`).
- **Testes:**
  - `test_osm.py::test_pages_are_cut_from_one_answer_whatever_order_the_limit_gives`: uma fonte
    simulada cuja ordem muda com o `limit`; antes, a página 2 repetia lugares;
  - `::test_the_first_page_knows_the_total_below_the_forty`;
  - `test_geo.py::TestGeocode::test_shows_more_than_three_and_pages_on`, que agora pede 100 e mostra
    "of 6";
  - `test_first_results.py::test_asked_for_the_whole_depth_the_first_page_has_the_total`.

##### 4. Limites e enums na assinatura (baixo-médio)

- **Mudança:**
  - **Coordenadas:** `Annotated[float, Range(-90, 90)]` e `Range(-180, 180)` em `get_weather` e
    `get_forecast`, `osm_reverse_geocode`, `air_quality_current` e `air_quality_forecast`,
    `timezone_lookup`, `distance_between` (as quatro), `overpass_pois` e `earthquake_search`.
  - **Intervalos da USGS** no `earthquake_search`: `max_radius_km` `Range(0, 20001.6)`,
    `min_depth_km` e `max_depth_km` `Range(-100, 1000)` (https://earthquake.usgs.gov/fdsnws/event/1/).
  - **Enums:** `order_by` passa a `Literal["time", "time-asc", "magnitude", "magnitude-asc"]`, o
    `status` do EONET a `Literal["open", "closed", "all"]` e o `unit` do `distance_between` a
    `Literal["km", "mi"]`.
  - **Código:** os `_validate_coords` e `_validate_location` copiados saem. Fica o código que cruza
    argumentos: as duas coordenadas juntas, o círculo da USGS, a bbox. Para quem chama a função
    crua, o `Literal` continua a ser verificado no código, como o `units`.
- **Testes:**
  - por módulo, `test_the_coordinates_are_bounded_in_the_schema` (ou `test_the_limits_are_in_the_schema`);
  - recusa pelo executor antes de qualquer pedido: `test_coordinates_out_of_range_are_refused_*`,
    `test_values_usgs_does_not_take_are_refused_before_any_request` (depth 5000, raio -1, `order_by`);
  - os enums: `test_eonet.py::test_the_statuses_are_in_the_schema` e
    `test_geo.py::test_the_coordinates_and_units_are_in_the_schema`.

##### 6. Floats escritos com `plain()` (baixo)

- **Mudança:** `plain()` nos pedidos e nos cabeçalhos:
  - a bbox e o `around` da QL do Overpass;
  - os parâmetros da USGS (coordenadas, raio, profundidades, magnitudes, e o cabeçalho que se faz a
    partir deles); a especificação FDSN recusa notação científica ("Float type parameters");
  - `lat` e `lon` do Nominatim;
  - as coordenadas do Open-Meteo no tempo, na qualidade do ar e no fuso;
  - a saída do `distance_between`.
- **Testes:** `test_coordinates_go_in_decimal_notation` em cada módulo (0.00001 vai como `0.00001`,
  não `1e-05`), `test_earthquake.py::test_numbers_go_in_decimal_notation` e
  `test_geo.py::test_the_points_are_written_in_decimal_notation`.

##### 7. Diversos (baixo)

- **Feito:**
  - `ip_lookup(ip: str)` passa a obrigatório (`test_the_address_is_required`).
  - Sai o "Defaults to kilometers" do `distance_between`.
  - O EONET aceita listas de categorias e de fontes separadas por vírgulas, que a documentação lê
    como OR (https://eonet.gsfc.nasa.gov/docs/v3): `_IDS_RE`. Um item vazio é recusado. Testes:
    `test_several_categories_and_sources_go_as_eonet_takes_them` e os casos novos de
    `test_invalid_options_do_not_call_api`.
- **Não feito, para o dono:**
  - O `min_magnitude=0.0` por omissão deixa de fora as magnitudes negativas. Mudar o valor por
    omissão para "nenhum", como a USGS, muda uma promessa da assinatura, e o autor já o tinha
    deixado ao dono.
  - O `earthquake_search` volta a pedir o `count` a cada continuação. Evitá-lo pedia um argumento
    escondido ou uma cache, o que não é barato nem claramente certo.

##### Mudanças nas linhas propostas para o CHANGELOG

- **`### Changed`, na linha Breaking da T08b:**
  - Depois de "`osm_search_place` up to 40 (it stopped at 10)", acrescentar: "each page cut from
    one answer, with the total below that depth".
  - Depois de "`eonet_events` up to 365 days (cut without a word)", acrescentar: "Coordinates are
    `Range` bounds (latitude -90 to 90, longitude -180 to 180), as are `earthquake_search`'s
    radius and depths (USGS's ranges). `order_by`, EONET's `status` and `distance_between`'s
    `unit` are enums, so `"KM"` or `"Open"` are refused. `ip_lookup` requires its `ip`."
- **`### Changed`, nova linha:** "`eonet_events` takes several categories or sources,
  comma-separated (EONET reads them as any of them)."
- **`### Fixed`:**
  - "`earthquake_search` and `earthquake_count` include the last day of the period: USGS read the
    bare date as its first instant, so a one-day search found nothing (T08b)."
  - "`overpass_query` refuses a query without `[out:json]` or without an out statement: the first
    failed as an unreadable answer to retry, the second answered that nothing matched (T08b)."
  - "The geo, weather and earthquake tools send coordinates and other floats in decimal notation:
    a coordinate of 0.00001 went as `1e-05`, which Overpass QL and USGS refuse (T08b)."

##### Comandos ao vivo, actualizados

- O comando com `max_depth_km: 5000.0` pelo executor é agora recusado antes do pedido, pelo
  `Range`. Para ver o detalhe FDSN do 400, chama-se a função crua:
  `uv run python -c "from ai_arch_toolkit.toolkit.tools import earthquake_search as t; t(max_depth_km=5000.0)"`.
- **Novo:** `uv run python -c "from ai_arch_toolkit.toolkit.tools import earthquake_count; print(earthquake_count(start_time='2024-01-01', end_time='2024-01-01'))"`.
  - Confirmar que a USGS aceita `endtime=2024-01-01T23:59:59.999999` (os microssegundos são da
    especificação FDSN, não da página da USGS).
  - Confirmar que a contagem é maior do que 0.
  - Se a USGS recusar os microssegundos, trocar `_END_OF_DAY` por `T23:59:59.999` ou mandar o dia
    seguinte.
- **Novo:** `uv run python -c "from ai_arch_toolkit.toolkit.tools import osm_search_place; print(osm_search_place('Springfield, Illinois', max_results=3).value)"`.
  O rodapé deve ter o total ("of N") quando o Nominatim devolve menos de 40.

##### Prova

- **Testes:** os oito ficheiros dos módulos, mais `test_tool_contract.py`, `test_tool_invariants.py`
  e `tests/test_architecture.py`, dão **1030 passed**. `tests/test_quality_budget.py` passa.
- **Verificações:** `uv run pyright src` dá 0 erros. `ruff check` e `ruff format --check` estão
  limpos nos meus ficheiros. Não corri `ruff format` em `contract_cases.py` nem em
  `error_bodies.py`.
- **`tests/toolkit` inteiro:** 9 falhas, todas em `test_eurostat.py`, que é trabalho da T08a em
  curso e não toca nos meus módulos.
- **Rede:** nenhum pedido às APIs. Li só documentação: a página FDSN da USGS, a especificação FDSN
  Commonalities 1.2 e a documentação v3 do EONET.
