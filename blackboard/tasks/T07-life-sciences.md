# T07 · Vida e saúde

- **Dono:** Claude (dois agentes, T07a e T07b, 2026-10-08) · **Estado:** done · **Depende de:** T05 (o modelo a seguir)
- **Módulos:** `_clinical_trials` (2 tools), `_uniprot` (5), `_pdb` (4), `_chembl` (5), `_foodon` (2),
  `_gbif` (4), `_rxnorm_dailymed` (6), `_openfda_food` (2), `_open_food_facts` (4): 34 tools
- **Origem:** os anexos A a E do plano para estes módulos · **Decisões:** D37 a D42 · **Regras:**
  `T00-rules.md`
- **Já feito:** `6a34668`: o 204 da RCSB lê-se como sem resultados (`allow_empty`); a UniProt diz o
  destino de uma entrada inactiva e o motivo de um pedido recusado (`messages`).

## O que está partido

- **Becos sem saída:**
  - `rxnorm_drug_search`, `rxnorm_related` e `rxnorm_ndcs` ficam com os primeiros 25, sem total;
  - `uniprot_features` e `uniprot_crossrefs` ficam com os primeiros 25, sem contagem;
  - o `dailymed_label` só dá os títulos das secções, e nenhuma tool devolve o texto de uma secção.
- **Registos cortados:**
  - o `clinical_trial_study` corta o resumo a 900 caracteres, a elegibilidade a 1200, 6 braços e 6
    resultados, e os locais e as referências a 5, em silêncio;
  - o `clinical_trial_search` não diz o total;
  - o texto FUNCTION do `uniprot_entry` fica nos 500 caracteres;
  - as listas do `open_food_facts_*` são cortadas em silêncio: os alergénios ficam em 10, 10 e 5,
    conforme a tool, a partir do mesmo pedido.
- **Promessas:**
  - o `offset` do `uniprot_search` é provavelmente ignorado, porque a UniProt pagina por cursor
    (suspeita);
  - o `gbif_species_match` promete "common names", mas a API resolve nomes científicos (suspeita).
- **Saída:**
  - a lista de elegibilidade do `clinical_trial_study` fica numa só linha;
  - o `pdb_search` só dá identificadores, e cada título custa outra chamada.
- **Repetidas:**
  - `uniprot_entry`, `uniprot_features` e `uniprot_crossrefs` descarregam a mesma entrada inteira;
  - `open_food_facts_product`, `_nutrition` e `_compare` partilham um pedido, mas cortam de maneira
    diferente.

  Decide na nota de desenho se se fundem (D41).

## Objectivo

As 34 tools cumprem o contrato de `T00-rules.md` e saem da lista de dívida.

## Passos

1. Nota de desenho por módulo: continuação, limites, falhas, leitores de erro e fusões. Confirma na
   documentação de cada API como pagina e como sinaliza erros, e escreve os URLs.
2. Testes primeiro, por módulo.
3. A migração; os `_truncate`/`_trim` destes módulos desaparecem.
4. `docs/tools-catalog.md` e `CHANGELOG`, com as quebras e a tabela de migração se houver fusões.

## Ficheiros

Os 9 módulos, os seus testes em `tests/toolkit/`, `tests/toolkit/contract_debt.py` (só as linhas
destas tools), `tests/toolkit/error_bodies.py` (as entradas destas fontes), `docs/tools-catalog.md`.

## Prova

- A invariante de contrato passa nas 34 tools, que saem da lista de dívida.
- Há testes de comportamento para cada item acima, incluindo o 204 da RCSB, o cursor da UniProt e o
  texto das secções do DailyMed.
- **Ao vivo, pelo dono:** os comandos no fim da ficha.

## Registo do dono

- **Estado:** done. Feito por agentes em worktrees, juntado e revisto pelo coordenador; uma revisão independente por parte, e as correcções dela (secção "Correcções da revisão").
- **Gate final, no checkout principal com todas as fichas da vaga:** 8195 passed, 118 skipped (os ao vivo e os 59 do nanope à espera); ruff, formatação, pyright e `uv lock --check` limpos (2026-10-08).
- **CHANGELOG:** as linhas entraram em `[Unreleased]` (Upgrade notes, Added, Changed, Fixed).

### T07a

- **Dono:** Claude (agente T07a), 2026-10-08 · **Estado:** review (feito na worktree, por aplicar)
- **Worktree:** `/Users/rge/Documents/dev/pessoal/ai-arch-toolkit/ai-arch-toolkit/.claude/worktrees/agent-aa9b6584a5202b379`
- **Módulos:** `_clinical_trials` (2), `_rxnorm_dailymed` (6, agora 7), `_openfda_food` (2),
  `_open_food_facts` (4, agora 3), `_foodon` (2). Novo: `_spl.py` (leitor do SPL, sem tools).
- **Resultado:** as 16 tools saem da lista de dívida (15 continuam, uma fundiu-se) e a tool nova
  `dailymed_label_text` entra já a cumprir o contrato. `CUT_HELPERS` desce de 14 para 12.

#### Nota de desenho

Regra comum a todos os módulos: os limites estão na assinatura (`Annotated[int, Range(...)]`), os
tectos de resultados ficam onde estavam (D39), todo o corte passa pela janela, zero resultados é
um sucesso que diz a pesquisa, e cada falha leva o motivo da fonte e o passo seguinte. Nenhum
módulo tem `_truncate` nem `_trim`.

##### `_clinical_trials`

- **Documentação:** o OpenAPI oficial, https://clinicaltrials.gov/api/oas/v2 (as páginas
  `data-api/api` e `data-api/about-api` são JavaScript e não se lêem sem browser).
- **Paginação:** por cursor. O `countTotal=true` dá o `totalCount` com a primeira página (é
  ignorado nas seguintes); o `nextPageToken` dá a página seguinte; a última não o traz; uma
  página pode vir com `studies` vazio. As páginas seguintes têm de levar os mesmos parâmetros.
- **`clinical_trials_search`:** ganha `offset` (só numera a página que o `page_token` lê). O
  rodapé diz `[results 1-5 of 57 | next: page_token="…", offset=5]`. Sem termos de pesquisa é
  `validation_error` (antes um `page_token` sozinho passava, contra a regra da API); um `offset`
  sem `page_token` também. A lista deixa de trazer resumo e locais (cortados a 900 e 5): cada
  estudo numa entrada com estado, tipo, fase, condições, intervenções, patrocinador, datas e
  inscrição ("1062 participants (actual)"); o cabeçalho manda ler o registo com
  `clinical_trial_study`. Uma página vazia com token diz como continuar.
- **`clinical_trial_study`:** o registo inteiro em secções (`overview`, `summary`,
  `description`, `eligibility`, `arms`, `outcomes`, `locations`, `references`), servido pela
  janela: `section=` (um `Literal`), `find=`, `offset=`, `max_chars` (500–20 000). O cabeçalho
  diz as secções que o registo tem, com o tamanho e o número de itens. Saem os cortes a 900, 1200,
  6 e 5. Os campos `markup` vêm em markdown, uma linha por parágrafo ou item: a elegibilidade
  mantém as linhas (era o item "numa só linha").
- **Erros:** um 400 é `text/plain` (`errorMessage`). O `error_reader` faz dele
  `validation_error` com as palavras da fonte e o passo seguinte (rever a sintaxe e os filtros).
  Um 404 num estudo é `not_found` (`missing=`); um 301 de um alias segue no mesmo host.
- **PMIDs** das referências são aceites pelo `pubmed_article`.

##### `_rxnorm_dailymed` (+ `_spl.py`)

- **Documentação RxNav:** https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html,
  `api-RxNorm.getDrugs.html`, `api-RxNorm.getRelatedByType.html`,
  `api-RxNorm.getAllRelatedInfo.html`, `api-RxNorm.getNDCs.html`; limite de 20 pedidos/s por IP
  em https://lhncbc.nlm.nih.gov/RxNav/TermsofService.html; um 400 para parâmetros inválidos em
  https://lhncbc.nlm.nih.gov/RxNav/news/API-Changes-202107.html (a página dá 403 ao WebFetch;
  citado pelo resultado de pesquisa); os nomes dos TTY em
  https://www.nlm.nih.gov/research/umls/rxnorm/docs/appendix5.html.
- **Paginação RxNav:** nenhuma; as listas vêm inteiras. `rxnorm_drug_search`, `rxnorm_related` e
  `rxnorm_ndcs` ganham `max_results` (1–25) e `offset`, e paginam com `page_window`: o total e a
  chamada seguinte (era "os primeiros 25, sem total").
- **`rxnorm_related`:** o `tty` é obrigatório no `related.json` e separa-se por espaços (o código
  antigo mandava `IN+BN` codificado como `IN%2BBN`). Sem `tty`, passa a usar o `allrelated.json`.
  Aceita `+`, vírgulas ou espaços.
- **Rótulos:** cada TTY com o nome (`SCD (Semantic Clinical Drug)`), ponto 5.
- **Erros RxNav:** `error_reader` que faz do 400 um `validation_error` com o passo seguinte. Um
  RxCUI desconhecido no `properties.json` vem vazio (já era `not_found`); no `related`/`ndcs` a
  documentação não diz, e a lista vazia diz "RxNorm lists no …" (zero resultados).
- **Documentação DailyMed:** https://dailymed.nlm.nih.gov/dailymed/webservices-help/v2/spls_api.cfm
  (`page`, `pagesize` até 100, `metadata.total_elements`/`total_pages`, filtro `rxcui`),
  `spls_setid_api.cfm` (o SPL em XML), `ndcs_api.cfm` (NDC com hífenes) e
  https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm (erros só por estado: 404,
  415, 5xx; "informação adicional no cabeçalho", sem dizer qual). Sem leitor de erros.
- **`dailymed_label_search`:** ganha `rxcui` (os RxCUI das tools RxNorm passam a ter onde entrar,
  ponto 6); `page` com `Range(1)`; o rodapé dá o total e `next: page=N`; datas ISO
  (`Jun 10, 2026` → `2026-06-10`).
- **`dailymed_label`:** perde `max_sections` (o corte); dá o título, a versão, a data efectiva
  (ISO), o titular e as secções numeradas, com o código LOINC e o nome que a resposta traz, e o
  tamanho; 50 por página (`offset`), como o `wiki_outline`.
- **`dailymed_label_text` (nova):** lê o texto do rótulo inteiro, uma secção (`section=N`, com as
  subsecções) ou as passagens de um termo (`find=`), pela janela; o mesmo desenho do `wiki_read`.
  É a forma de ler o texto de uma secção que a ficha pedia.
- **`_spl.py`:** lê o SPL com `xml.etree` (já usado). Secções pela ordem do documento; narrativa
  CDA (https://hl7.org/cda/stds/core/narrative.html): parágrafos, listas (item por linha,
  aninhadas com indentação limitada), tabelas (linha por linha, ` | `, legenda como `Table:`),
  `br` como quebra, imagens sem texto. As quebras de linha do XML são espaço. Cada elemento lê-se
  uma vez (uma tabela dentro de uma célula fica na célula). Secções LOINC:
  https://www.fda.gov/industry/structured-product-labeling/section-headings-loinc.
- **NDCs:** vêm na forma CMS de 11 dígitos; o cabeçalho manda para
  `dailymed_label_search(rxcui=…)`, que a documentação garante.

##### `_openfda_food`

- **Documentação:** https://open.fda.gov/apis/query-parameters/ (`limit` até 1000, `skip` até
  25 000), https://open.fda.gov/apis/paging/ (`search_after` para além disso),
  https://open.fda.gov/apis/authentication/ (240 pedidos/min, 1000/dia por IP sem chave). Os erros
  vêm do código do servidor: https://github.com/FDA/openfda/blob/master/api/faers/api.js
  (`NOT_FOUND` 404 "No matches found!", `SERVER_ERROR` 500 "Check your request and try again") e
  `api_request.js` (os `BAD_REQUEST` 400, por exemplo "Skip value must 25000 or less.").
- **`openfda_food_recall_search`:** `max_results` 1–20 e `skip` 0–25 000 na assinatura. O rodapé
  dá o total e `next: skip=N`; quando a página seguinte passaria os 25 000, o rodapé diz que o
  resto não se lê aqui e o cabeçalho manda estreitar por datas. Zero resultados (o 404 da fonte)
  diz a pesquisa.
- **Erros:** `error_reader`: `BAD_REQUEST` é `validation_error` com as palavras da fonte e "check
  the filters and the dates"; os outros códigos vão como texto, tipados pelo estado (500
  `upstream`, 429 `rate_limited`).
- Sem `search_after`: o tecto de 25 000 fica dito, não escondido.

##### `_open_food_facts`

- **Documentação:** o OpenAPI do servidor,
  https://github.com/openfoodfacts/openfoodfacts-server/blob/main/docs/api/ref/api.yaml
  (`/api/v2/product/{code}` responde 404 a um produto desconhecido; a pesquisa dá `count`,
  `page`, `page_size`, `page_count` = produtos nesta página, e `skip`), e os limites em
  https://openfoodfacts.github.io/openfoodfacts-server/api/#rate-limits (15 leituras e 10
  pesquisas por minuto; 503 no limite global).
- **Fusão (D41):** `open_food_facts_nutrition` funde-se no `open_food_facts_product`. As duas liam
  o mesmo produto e cortavam-no de maneira diferente; o produto passa a ter tudo, com o
  Nutri-Score e o grupo NOVA rotulados. O `open_food_facts_compare` fica: é outro trabalho
  (vários produtos lado a lado, uma linha cada).
- **Listas inteiras:** alergénios, vestígios, aditivos, categorias, rótulos e países sem corte (o
  10/10/5 e o 8 saem). Nada corta: `product` e `compare` passam a `WHOLE`.
- **Nutrientes:** por 100 g, com a unidade (`<nutriente>_unit` da resposta, ou kcal/g), sem
  notação científica (`0.00001 g`, era `1e-05`).
- **`open_food_facts_search`:** `page` com `Range(1)`; o rodapé dá `count` e `next: page=N`; a
  numeração segue a página. Pede só os campos que lista. O texto antigo chamava "page_count" ao
  número de páginas; a fonte diz que são os produtos da página.

##### `_foodon`

- **Documentação:** o código do OLS4,
  https://github.com/EBISPOT/ols4/blob/dev/backend/src/main/java/uk/ac/ebi/spot/ols/controller/api/v1/V1SearchController.java
  (`rows`, `start`, `queryFields`, `exact`, `response.numFound`, os sinónimos por omissão) e
  `controller/api/exception/GlobalExceptionHandler.java` (erros como `{"status", "message"}`). A
  página de ajuda do OLS é JavaScript.
- **`foodon_search`:** `max_results` 1–20, `start` com `Range(0)`; rodapé com o total e
  `next: start=N`. A definição já não se corta aos 700.
- **`foodon_term`:** pede com `queryFields=obo_id`; devolve a definição inteira e os sinónimos
  (exactos, relacionados, …). Um termo importado mantém o prefixo da sua ontologia
  (`NCBITaxon:3750`). Sem termo: `not_found` com "search with foodon_search".
- **Erros:** sem leitor; a porta já cita a `message` do JSON.

#### Fusões e nomes tirados

| Antes | Agora |
|---|---|
| `open_food_facts_nutrition(barcode)` | `open_food_facts_product(barcode)`: o mesmo produto, com tudo, e o Nutri-Score e o NOVA rotulados |
| `dailymed_label(setid, max_sections)` | `dailymed_label(setid, offset=…)`: todas as secções, numeradas; o texto de uma secção em `dailymed_label_text(setid, section=N)` (nova) |

O `nanope` não usa nenhuma destas tools (confirmado por grep). Mudanças de parâmetros sem nome
tirado: `rxnorm_drug_search` e `rxnorm_ndcs` ganham `max_results`/`offset`; `rxnorm_related`
ganha `offset` e o `tty` separa-se por espaços; `dailymed_label_search` ganha `rxcui`;
`clinical_trials_search` ganha `offset`; `clinical_trial_study` ganha `section`, `find`,
`offset`, `max_chars`.

#### Mudanças em ficheiros partilhados

- Nenhuma em `core/`, `_http.py` ou `_window.py`.
- `toolkit/tools/__init__.py`: sai `open_food_facts_nutrition`, entra `dailymed_label_text`.
- `tests/toolkit/contract_cases.py`: só as entradas destas fontes (um bloco "T07a" em cada
  dicionário), mais `dailymed_label_text` em `_LOOKUPS` e `open_food_facts_compare`/`_product` em
  `_WHOLE` (numa linha própria no fim de cada bloco, para o merge).
- `tests/toolkit/error_bodies.py`: as entradas `_clinical_trials`, `_rxnorm_dailymed`,
  `_openfda_food`, `_open_food_facts`, `_foodon`.
- `tests/toolkit/contract_debt.py`: 16 linhas apagadas; `CUT_HELPERS` 14 → 12.
- `tests/quality_baseline.json`: 6 entradas apagadas (`_clinical_trials._format_trials` C901 e
  PLR0912, `_open_food_facts._format_products` C901 e PLR0912, `_format_nutrition` C901,
  `_openfda_food._format_recalls` C901). Só desceu.
- Ficheiro novo de testes: `tests/toolkit/health_pages.py` (respostas para os casos do contrato).

#### Prova

Testes novos ou reescritos, por item da ficha:

| Item da ficha | Testes |
|---|---|
| `rxnorm_drug_search`, `rxnorm_related`, `rxnorm_ndcs`: 25 sem total | `test_rxnorm_dailymed.py::TestRxNormLists::test_a_search_lists_25_then_reads_on_with_the_total`, `test_related_reads_on_past_its_page`, `test_ndcs_read_on_and_point_to_the_labels`, `test_max_results_is_refused_outside_its_limits` |
| `dailymed_label` só títulos; texto de uma secção | `TestDailyMedLabel::test_the_label_gives_its_facts_and_numbered_sections_with_sizes`, `test_a_section_size_is_what_dailymed_label_text_returns`, `test_a_long_outline_reads_on`; `TestDailyMedLabelText::test_a_section_reads_with_its_subsections_lists_and_tables`, `test_a_long_section_reads_on_and_names_the_section`, `test_find_searches_the_whole_label`, `test_a_section_the_label_does_not_have_is_a_validation_error`; `TestSpl` (4 testes) |
| `clinical_trial_study` corta resumo, elegibilidade, braços, resultados, locais, referências | `test_clinical_trials.py::TestStudy::test_the_record_comes_whole_with_its_sections_and_sizes`, `test_every_arm_outcome_site_and_reference_is_there` (4 casos), `test_a_long_section_reads_on_window_by_window_and_names_the_section`, `test_find_gives_the_passages_that_mention_a_term` |
| elegibilidade numa só linha | `TestStudy::test_the_eligibility_criteria_keep_their_lines` |
| `clinical_trials_search` sem total | `TestSearch::test_a_page_says_the_total_and_the_call_for_the_next`, `test_the_next_page_is_numbered_on_from_the_offset` |
| `open_food_facts_*` cortam alergénios a 10, 10, 5 | `test_open_food_facts.py::TestProduct::test_every_list_comes_whole` (5 casos), `TestCompare::test_products_compare_on_one_line_each_with_every_allergen`, `TestProduct::test_the_nutrition_tool_is_gone_into_the_product` |
| D41 (fusão) | o mesmo `test_the_nutrition_tool_is_gone_into_the_product` e `test_the_record_comes_whole_with_labels_next_to_the_scores` |
| Fora da ficha, encontrados | `test_term_types_go_space_separated` (3 casos), `test_related_without_a_term_type_asks_for_all_related`, `test_a_request_rxnav_cannot_process_is_a_validation_error`, `test_a_refused_search_is_a_validation_error_with_the_sources_words` (CT.gov e openFDA), `test_past_the_deepest_skip_the_heading_says_how_to_narrow`, `test_a_table_in_a_cell_is_read_once`, `test_the_sources_newlines_are_whitespace_…` |

Contrato: `test_tool_contract.py` passa nas 16 (15 + a nova) sem nenhuma linha de dívida;
`test_the_copied_cut_helpers_only_go` com 12.

Falhar antes: no ClinicalTrials.gov (19 falhas) e no FoodOn (5) corri os testes antes do código.
Nos outros três escrevi os testes primeiro mas só os corri depois; para o provar, corri-os contra
o módulo original (reposto de `HEAD` por momentos, sem stash, e depois devolvido): openFDA 5
falhas, Open Food Facts 12, RxNorm/DailyMed 30 (com um `dailymed_label_text = None` a fingir a tool
nova). Cada falha é a do item que o teste prova. Testes antigos que mudaram de expectativa, por
desenho: o 400 do openFDA passa de `upstream` a `validation_error`; os textos dos cabeçalhos e
das listas; o `page_token` sozinho passa a ser recusado.

#### Ao vivo, pelo dono

APIs gratuitas e sem chave. Cada comando faz um ou dois pedidos.

```bash
uv run python -c "from ai_arch_toolkit.toolkit.tools import clinical_trials_search as s; print(s(condition='asthma', status='recruiting', max_results=3).value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import clinical_trial_study as t; print(t('NCT04280705', section='eligibility').value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import clinical_trials_search as s; print(s('covid', status='recruitng'))"
uv run python -c "from ai_arch_toolkit.toolkit.tools import rxnorm_drug_search as s; print(s('ibuprofen').value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import rxnorm_related as r; print(r('5640', tty='BN SBD').value); print(r('5640', max_results=5).value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import rxnorm_ndcs as n; print(n('213269').value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import rxnorm_related as r; print(r('999999999').value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import dailymed_label_search as s; print(s(rxcui='213269').value); print(s(ndc='00069420030').value)"
uv run python -c "import re; from ai_arch_toolkit.toolkit.tools import dailymed_label_search as s, dailymed_label as l, dailymed_label_text as t; r = s(drug_name='ibuprofen', max_results=1).value; sid = re.search(r'setid: (\S+)', r)[1]; print(l(sid).value); print(t(sid, section=1).value); print(t(sid, find='overdose').value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import openfda_food_recall_search as s; print(s(query='listeria', max_results=3).value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import open_food_facts_product as p; print(p('3017620422003'))"
uv run python -c "from ai_arch_toolkit.toolkit.tools import open_food_facts_search as s; print(s(brand='ferrero', max_results=3).value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import foodon_search as s, foodon_term as t; print(s('apple', max_results=3).value); print(t('FOODON:00002473'))"
```

O que confirmar:

- o `totalCount` e o `nextPageToken` do ClinicalTrials.gov, e a elegibilidade em linhas;
- as palavras reais do 400 do ClinicalTrials.gov (o `status='recruitng'`): a entrada em
  `error_bodies.py` usa palavras ilustrativas e deve passar a ser a resposta gravada;
- o `related.json?tty=BN+SBD` (espaço) e o `allrelated.json` do RxNav; o que o RxNav responde a um
  RxCUI que não existe no `related`/`allrelated` (vazio, 404 ou 400?);
- se o filtro `ndc` do DailyMed aceita a forma de 11 dígitos sem hífenes que o RxNav devolve
  (se não, o rodapé do `rxnorm_ndcs` já manda pelo `rxcui`);
- o formato do `published_date` do DailyMed (`Jun 10, 2026`?) e o `total_pages` em texto;
- um SPL real: secções com subsecções, tabelas e listas legíveis, tamanhos do `dailymed_label`
  iguais ao total do `dailymed_label_text(section=N)`;
- o 404 do openFDA para zero resultados; o `count` do Open Food Facts; os sinónimos do OLS com
  `queryFields=obo_id`.

#### Bloqueios e achados

- **Sem bloqueios.**
- **Investigação:** só documentação, nenhum pedido às APIs. Os OpenAPI do Open Food Facts e o
  código do OLS4 e do openFDA, públicos no GitHub, li-os com `gh api` (que leva a autenticação do
  `gh` do dono, sem email nem User-Agent próprio); o OpenAPI do ClinicalTrials.gov com WebFetch
  em `clinicaltrials.gov/api/oas/v2`, que é a especificação, não dados.
- **Achado (partilhado, `_window.py`):** uma janela de lista com o corpo vazio e uma chamada
  seguinte escreve `[no results from N | end]`: o `footer()` diz "end" quando `last < first`, sem
  olhar para o `next_call`. O ClinicalTrials.gov documenta que uma página pode vir vazia com
  `nextPageToken`; contornei no módulo (uma frase com a chamada seguinte). Outras fontes por
  cursor podem precisar do mesmo; vale corrigir no `_window.py` (dizer `next:` também aí).
- **Achado (ClinicalTrials.gov):** uma página de 20 estudos com resultados pode passar os 10 MB do
  `max_bytes` da porta (o `resultsSection` é grande). O parâmetro `fields=` cortaria o pedido, mas
  os nomes das peças pedem verificação ao vivo antes de se usarem.
- **Achado (DailyMed):** a documentação diz que os erros trazem "informação adicional no
  cabeçalho" sem dizer qual; com uma resposta gravada, o `error_reader` pode passar a lê-lo.
- **Achado (RxNav):** a página de 2021 que documenta o 400 dá 403 a pedidos automáticos; está
  citada pelo resultado da pesquisa, não lida.
- **Achado (openFDA):** sem `search_after`, os resultados para além de `skip=25000` não se lêem;
  a tool di-lo e manda estreitar por datas. Se for preciso, o `search_after` vem no cabeçalho
  `Link` (https://open.fda.gov/apis/paging/), que a porta hoje não entrega às tools.

#### Correcções da revisão

Feitas a 2026-10-08 sobre a árvore de trabalho principal, a partir de `notes/T07a-review.md`. Em
cada achado, o teste foi escrito primeiro e visto a falhar (27 falhas antes do código, todas pelo
motivo do achado); depois, o código. Verificação final: os testes dos seis módulos, `test_wiki_html`,
`test_wiki`, `test_http`, `test_tool_contract`, `test_tool_invariants`, `test_architecture` e
`test_quality_budget` passam (1242); `uv run pyright src`: 0 erros; `ruff check` e `ruff format`
limpos nos ficheiros tocados (`contract_cases.py` não foi formatado, como pedido).

##### 1. Alta: unidades dos nutrientes do Open Food Facts
- **Mudança:** `_nutrients` deixa de ler `<nutriente>_unit` (a unidade em que o contribuidor
  escreveu `_value`). O `_100g` vem normalizado: kcal na energia, g no resto (schema
  `docs/api/ref/schemas/product_nutrition.yaml` do openfoodfacts-server, citado na docstring).
- **Teste:** `test_open_food_facts.py::TestProduct::test_nutrients_per_100_g_are_in_standard_units_whatever_unit_was_entered`
  (sódio em mg de um rótulo dos EUA: `sodium 0.4 g`, `salt 1 g`, no produto e no `compare`).

##### 2. Média-alta: tabelas do SPL com `rowspan`/`colspan`
- **Mudança:** módulo novo `toolkit/tools/_tables.py` com `Cell`, `cell()` (spans lidos dos
  atributos, com os tectos 500/50), `grid()` (o antigo `_wiki_html._grid`, igual, com o orçamento
  de 100 000 células) e `table_lines()` (legenda `Table:`, uma linha por linha, células vazias do
  fim e linhas vazias fora). O `_wiki_html` passa a usá-lo (sai o `_Cell`, o `_grid`, o `_span` e
  as três constantes; fica o `_MAX_INDENT`); o `_spl._table` também. Os testes do `_wiki_html` não
  mudaram e passam iguais.
- **Testes:** `test_rxnorm_dailymed.py::TestSpl::test_a_cell_that_spans_rows_or_columns_is_laid_out`
  (a tabela de reacções adversas da revisão) e `test_spans_are_capped_as_the_wikis_are`.

##### 3. Média: só o `NOT_FOUND` do openFDA é zero resultados
- **Mudança na porta (`_http.py`, aditiva e compatível):** `empty_on_404` aceita `True` (como
  antes) ou um teste sobre o `Reply` do 404 (corpo descodificado se for JSON), o tipo
  `EmptyOn404`. Se o teste recusar, ou tropeçar no corpo, o 404 segue o caminho normal: leitor de
  erros e "endpoint not found" (`upstream`). Docstrings do `Api` e de `_status_failure` dizem-no.
- **openFDA:** `_no_matches` aceita só `{"error": {"code": "NOT_FOUND"}}`. Usado na pesquisa e
  também no `openfda_food_recall`, que trocou `missing=` por `empty_on_404=_no_matches` (o pedido
  é a mesma pesquisa; um 404 de CDN já não diz "openFDA has no food recall …"). O caso de
  `not_found` do contrato para `openfda_food_recall` passa a usar o corpo `NOT_FOUND` do openFDA
  (saiu de `_missing_by_404`; entrada própria no bloco T07a de `contract_cases.py`).
- **Testes:** `test_http.py::TestDeclaredAnswers::test_a_404_is_nothing_only_where_the_sources_answer_says_so`
  (`get_json` e `get_json_list`; outro código, outra mensagem e HTML são `upstream`);
  `test_openfda_food.py::TestRecallSearch::test_a_404_that_is_not_openfdas_no_match_is_a_moved_endpoint`
  (3 corpos) e `TestRecall::test_a_404_that_is_not_openfdas_no_match_is_not_a_missing_recall`.

##### 4. Média-baixa: texto que o `_spl` deixava cair
- **Mudança:** cada `excerpt/highlight/text` de uma secção escreve-se depois do texto dela, sob
  uma linha `Highlights:` (SPL Implementation Guide v1, 2.2.4 Highlights,
  https://www.fda.gov/media/84201/download, lido). As Recent Major Changes (43683-2) passam a ter
  texto, e o tamanho do `dailymed_label` conta-o. Legendas: `List: …` antes dos itens (indentada
  com a lista) e `Figure: …` numa linha própria para um `renderMultiMedia` com `caption` (numa
  célula, junta-se com ` / `).
- **Testes:** `TestSpl::test_the_highlights_of_a_section_are_read_under_their_own_line`,
  `TestSpl::test_list_and_figure_captions_are_kept`,
  `TestDailyMedLabel::test_a_section_with_only_highlights_lists_their_size`.

##### 5. Baixa-média: o ClinicalTrials.gov pede só o que lê
- **Mudança:** a pesquisa manda `fields=IdentificationModule,StatusModule,SponsorCollaboratorsModule,ConditionsModule,DesignModule,ArmsInterventionsModule,HasResults`;
  o registo, `fields=ProtocolSection,HasResults`. Nunca vem o `resultsSection`. O OpenAPI
  (https://clinicaltrials.gov/api/oas/v2, lido com WebFetch) diz que cada item é "area name, piece
  name, field name" e dá como exemplos `ProtocolSection`, `HasResults` e `ConditionsModule`; os
  outros nomes de módulo seguem o mesmo padrão dos esquemas do OAS, mas **pedem confirmação ao
  vivo** (comando abaixo). Se algum falhar, um 400 do CT.gov diz qual, como `validation_error`.
- **Testes:** `TestSearch::test_the_search_asks_only_for_the_modules_it_lists`,
  `TestStudy::test_the_record_asks_for_the_protocol_and_not_the_results`.

##### 6. Baixa: elegibilidade com aninhamento e sem escapes
- **Mudança:** `_markup` guarda a indentação por níveis (uma pilha de indentações; dois espaços
  por nível, no máximo 10) e tira os escapes de barra do CommonMark
  (https://spec.commonmark.org/0.31.2/#backslash-escapes): `* aged \>= 18` lê-se `* aged >= 18`.
  Vale para o resumo e a descrição também.
- **Teste:** `TestStudy::test_the_criteria_keep_their_nesting_and_read_without_markdown_escapes`.
  Falta a verificação num registo real (comando abaixo).

##### 7. Baixa: datas do DailyMed sem depender do locale
- **Mudança:** `_published` lê `Mon DD, YYYY` com uma tabela fixa de meses em inglês (sai o
  `strptime("%b …")`).
- **Teste:** `TestDailyMedSearch::test_published_dates_read_whatever_the_locale` (fr_FR, de_DE,
  pt_PT, com `Oct 10, 2026`; salta se o locale não existir).

##### 8. Baixa: páginas para lá do fim e tectos
- **DailyMed e Open Food Facts:** uma página vazia com total diz "Page 9 is past the end: 57
  DailyMed labels match drug name 'ibuprofen', on 6 pages of 10; the last is page=6." (o número
  de páginas da resposta, ou calculado). **FoodOn**, mesma causa, também: "start=100 is past the
  end: 57 FoodOn terms match 'apple'; the last page is start=50."
- **ClinicalTrials.gov:** a página vazia com token mantém o texto e ganha o `metadata["window"]`
  (first, last, total, `next_call` com o token). O rodapé do `_window.py` diria "end" num corpo
  vazio; por isso o texto continua a ser a frase do módulo (o achado do `_window.py` mantém-se).
- **openFDA:** o conselho de estreitar por datas sai do cabeçalho e vai para `Window.rest`: o
  rodapé diz `[results 24991-25010 of 40000 | openFDA reads no further than skip=25000: narrow the
  search with from_date and to_date for the rest]`.
- **Testes:** `TestDailyMedSearch::test_a_page_past_the_last_gives_the_total_and_the_last_page`,
  `test_open_food_facts.py::TestSearch::test_a_page_past_the_last_gives_the_count_and_the_last_page`,
  `test_foodon.py::TestSearch::test_a_start_past_the_last_term_gives_the_total`,
  `TestSearch::test_an_empty_page_with_a_token_keeps_the_window` (CT.gov),
  `TestRecallSearch::test_past_the_deepest_skip_the_footer_says_how_to_narrow` (reescrito).

##### 9. Baixa: palavras da fonte e notas sem rótulo
- **RxNav 400:** o leitor junta as palavras da fonte (`message`/`error` de um JSON, ou o texto
  que não seja uma página): "RxNorm could not process the request (HTTP 400, <palavras>); check
  the arguments: …". Sem palavras, fica "invalid parameters". A entrada de `error_bodies.py` não
  mudou (o corpo real do 400 continua por gravar).
- **Nutri-Score:** `not-applicable` e `unknown` lêem-se "Nutri-Score: not applicable (it does not
  apply to this kind of product)" e "Nutri-Score: unknown (not computed: data is missing)"; nas
  listas, só "not applicable"/"unknown". O Eco-Score, com os mesmos valores, também.
- **Testes:** `TestRxNormConcept::test_rxnavs_400_keeps_the_sources_words` (texto e JSON),
  `test_rxnavs_400_page_is_not_quoted`; `TestProduct::test_a_nutri_score_that_is_no_grade_says_what_it_is`
  (2 casos).

##### Ficheiros tocados nesta ronda
- Código: os seis módulos, `_wiki_html.py`, `_tables.py` (novo), `_http.py`.
- Testes: os cinco ficheiros de teste dos módulos, `test_http.py`; em `contract_cases.py`, só a
  entrada `openfda_food_recall` (tirada de `_missing_by_404`, entrada própria no bloco T07a).
  `error_bodies.py`, `health_pages.py` e `test_wiki_html.py` sem mudanças.
- A secção "Mudanças em ficheiros partilhados" acima deixa de valer para o `_http.py`: muda o
  `empty_on_404` (aditivo).

##### CHANGELOG (linhas a juntar às propostas acima)
Em **Changed**, na entrada "Breaking: the health tools…", trocar "nutrients per 100 g with units"
por "nutrients per 100 g in grams (energy in kcal)". Em **Fixed**:

```
- Open Food Facts nutrients per 100 g are in grams (energy in kcal): the unit a contributor
  entered (`mg` of sodium on a US label) was printed beside the gram value, 1000 times too low.
- An openFDA 404 reads as no recalls only when openFDA says `NOT_FOUND`; any other 404 is an
  `upstream` "endpoint not found", where it read as no recalls (or no such recall).
- `dailymed_label_text` lays out table cells that span rows or columns, and reads the label's
  Highlights (Recent Major Changes included) and the captions of lists and figures.
- `clinical_trial_study` keeps the nesting of the eligibility criteria and drops markdown's
  backslash escapes (`\>=` reads `>=`); both ClinicalTrials.gov tools ask only for the parts
  they read, so a page of studies with posted results no longer passes the 10 MB limit.
- DailyMed publication dates read as ISO 8601 under any locale.
- A page past the last one of `dailymed_label_search`, `open_food_facts_search` and
  `foodon_search` gives the total and the last page; an RxNav 400 carries RxNav's reason.
```

##### Ao vivo, pelo dono (a juntar aos comandos acima)

```bash
uv run python -c "from ai_arch_toolkit.toolkit.tools import clinical_trials_search as s; print(s(condition='asthma', status='completed', max_results=3).value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import clinical_trial_study as t; print(t('NCT04280705', section='eligibility').value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import openfda_food_recall_search as s; print(s(query='zzqqxx'))"
```

Confirmar: que o CT.gov aceita os nomes de módulo no `fields` (um 400 diria qual não aceita) e que
a lista traz o mesmo que antes; que a elegibilidade de um registo real mantém os subníveis e não
mostra `\`; que o 404 do openFDA para zero resultados traz mesmo `"code": "NOT_FOUND"`.

### T07b

- **Worktree:** `/Users/rge/Documents/dev/pessoal/ai-arch-toolkit/ai-arch-toolkit/.claude/worktrees/agent-a82b78b3111b85b8f`
  (sem commits; base `601e38c`).
- **Módulos:** `_uniprot` (5 tools), `_pdb` (4), `_chembl` (5), `_gbif` (4): 18 tools.
- **Dívida:** saem as 17 linhas destas tools (o `uniprot_sequence` já não estava na lista).
  `CUT_HELPERS` desce de 14 para 13 (sai o `_trim` do `_uniprot`; os outros três módulos não
  tinham helper de corte). Nenhum `_truncate`/`_trim` fica nos quatro módulos.

#### Nota de desenho

##### Comum aos quatro

- Os limites passam para a assinatura (`Annotated[int, Range(...)]`), com os tectos de hoje
  (D39); sai cada `_bounded` (o `min`/`max` que ajustava em silêncio).
- Cada `Api` ganha um `error_reader`. Um 400 é um pedido que a fonte recusa, e quem o corrige é
  quem chama: `validation_error`, com as palavras da fonte e o passo seguinte. Antes era
  `upstream` (UniProt) ou "HTTP error 400" (os outros).
- Zero resultados: sucesso com a consulta ("No UniProtKB proteins match 'x'.").
- Listas: `list_window` (a fonte pagina) ou `page_window` (a tool tem a lista inteira), com o
  total e a chamada seguinte. Registos de uma só coisa (`*_entry`, `*_molecule`, `gbif_species`,
  `pdb_ligands`) ficam inteiros (`WHOLE`), como no anexo A.
- A validação aceita o que a fonte aceita: texto livre (sem caracteres de controlo, com tecto de
  comprimento) em vez das regex que recusavam aspas, `[`, `]` e `*` da sintaxe das fontes.

##### `_uniprot`

Documentação: o manual oficial da UniProt está no GitHub (`ebi-uniprot/uniprot-manual`), porque
as páginas `www.uniprot.org/help/*` são uma aplicação JavaScript:
https://www.uniprot.org/help/pagination, https://www.uniprot.org/help/api_queries,
https://www.uniprot.org/help/rest-api-headers, https://www.uniprot.org/help/accession_numbers,
https://www.uniprot.org/help/deleted_accessions, https://www.uniprot.org/help/return_fields;
as formas dos comentários e das features, pelo site da própria UniProt
(`ebi-uniprot/uniprot-website`, `src/uniprotkb/types/commentTypes.ts`, `featureType.ts`).

- **O `offset` era ignorado (suspeita confirmada).** A pesquisa só aceita `size` (até 500) e
  `cursor`; a página seguinte vem no cabeçalho `Link: <…&cursor=…>; rel="next"`, que falta na
  última; o total vem em `x-total-results`, não no corpo (o `totalResults` que o código lia nunca
  existiu: o total saía sempre "?"). Cada "página" era a primeira.
  - `uniprot_search(query, organism, reviewed, max_results, cursor, offset)`: o rodapé dá
    `next: cursor="…", offset=N`; o `offset` só numera a página que o cursor alcança. Um `offset`
    sem `cursor` é `validation_error` antes do pedido (andar cursores até lá seria trabalho sem
    tecto).
  - A porta não dava cabeçalhos ao `parse`: `Api.get_json_reply` (ver "Mudanças em ficheiros
    partilhados").
- **404 ou 400 para uma acessão desconhecida:** a documentação dá 404
  (`{"messages": ["Resource not found"]}`) para uma acessão bem formada que não existe, e 400 para
  um pedido mal formado. A acessão valida-se antes pela regex oficial (com o `-N` das isoformas),
  por isso uma mal formada nunca sai; a desconhecida é `not_found` (o `missing=` de antes). Um
  400 é `validation_error` com as `messages`.
- **Acessão fundida:** a documentação dá `303 See Other` para a entrada nova
  (`/uniprotkb/P23141?from=Q00015`); a porta segue o redirect no mesmo host, e a resposta é a da
  outra acessão. O cabeçalho diz "UniProtKB entry P23141 (EST1_HUMAN), which UniProt gives for
  Q00015". As inactivas que respondem 200 (`entryType: Inactive`, vistas ao vivo) continuam
  `not_found` com o destino.
- **Fusão (D41): não se fundem.** `uniprot_entry`, `uniprot_features` e `uniprot_crossrefs` lêem
  a mesma entrada, mas fazem três trabalhos, cada um com o seu filtro e a sua unidade de janela:
  ler a anotação (texto, por caracteres), listar as features por tipo e listar as referências
  por base de dados (itens). Fundir não poupava pedidos (a fusão continuava a pedir a entrada
  inteira a cada chamada) e misturava duas unidades de `offset` numa tool. O defeito do anexo E
  era cortarem cada uma à sua maneira; agora nenhuma corta sem janela. O `fields=` não reduz o
  pedido de forma documentada no endpoint de uma entrada (e as referências não têm um campo
  "todas"), por isso fica a entrada inteira.
- **`uniprot_entry(accession, offset, max_chars)`:** a anotação inteira por `text_window`: resumo
  (estado, existência, nomes alternativos, genes, organismo com táxon, comprimento em aa e massa
  em Da, datas ISO, contagem de features e referências com a tool que as lê, palavras-chave) e
  cada comentário sob o seu título (função, actividade catalítica com EC, localização, doença,
  cofactores, isoformas, interacções, cinética com unidades sem notação científica, …). O texto
  FUNCTION já não pára nos 500.
- **`uniprot_features` e `uniprot_crossrefs`:** `page_window` sobre todos os itens
  (`max_results` 1–25, `offset`); a primeira página conta os itens por tipo / por base; um filtro
  sem resultado diz os que a entrada tem. As referências mostram todas as propriedades (eram 3);
  as features mostram a mudança de uma variante (`H -> D`), o ligando e o `featureId`.
- **Corrigido de passagem:** um organismo com espaço ia sem aspas (`organism_name:Homo sapiens`);
  um `OR` na consulta levava os filtros consigo (agora `(query) AND …`); uma entrada TrEMBL saía
  "(no protein name)" (lê o `submissionNames`).

##### `_pdb`

Documentação: https://search.rcsb.org/ (paginação, "Empty results" = 204),
https://data.rcsb.org/index.html (REST e GraphQL: o GraphQL responde sempre 200 e põe os erros no
corpo; até 1000 IDs por pedido). Os campos das consultas GraphQL são os que o cliente da própria
RCSB usa (`rcsb/rcsb-mcp`, `src/rcsb_mcp/report/tables.py` e `queries.py`; `rcsb/py-rcsb-api`).

- **`pdb_search` só dava identificadores:** cada página faz agora dois pedidos, a pesquisa e um
  GraphQL `entries(entry_ids: $ids)` para todos os IDs dessa página: título, método, resolução em
  Å e data de lançamento ISO. Um ID sem registo diz-se na linha. Paginação pelo `start`, com o
  total (`total_count`).
- **204:** continua sem resultados (`allow_empty`, desde `6a34668`), agora com a consulta.
- **`pdb_ligands`:** fazia um pedido por ligando (sem tecto: uma entrada grande tem dezenas).
  Agora: a entrada por REST (o 404 é `not_found`, documentado) e um GraphQL
  `nonpolymer_entities(entity_ids: ["4HHB_3", …])`. Um ligando listado sem registo é uma linha,
  não uma falha da tool inteira.
- **`pdb_entry`:** datas ISO, resolução em Å, contagem de entidades (os IDs de entidade e de
  assembly saem: nenhuma tool os aceita), a citação com DOI e PubMed ID (aceites pelas tools de
  literatura). **`pdb_chemical_component`:** peso em Da, InChIKey.
- **Erros (`_rcsb_error`):** os `errors[].message` do GraphQL (num 200) são `upstream` com as
  palavras da RCSB; um 400 com `message` é `validation_error`.

##### `_chembl`

Documentação: https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services
(`limit`, `offset`, `page_meta.total_count`/`next`) e
https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/drug-and-compound-questions
(o que cada `max_phase` quer dizer). Os erros, pelo código dos web services
(`chembl/chembl_webservices_py3`, `src/chembl_webservices/core/resource.py`, arquivado): um 400 com
`{"error_message": …}`, um ID desconhecido 404 sem corpo, uma pesquisa com menos de 3 caracteres
recusada ("Search query too short").

- As três pesquisas: `list_window` com o total; `next` só quando há mais.
- A pesquisa com menos de 3 caracteres recusa-se antes do pedido.
- Saída: a fase com o rótulo ("4 (approved)", "0.5 (early phase 1)", "-1 (unknown)", "none
  (preclinical)"); propriedades com unidade (MW em Da, PSA em Å²); InChIKey; actividades com o nome
  da molécula e do alvo ao lado dos IDs, pChEMBL, a descrição do ensaio, a revista e o ano, e o
  aviso de validade quando o há. Os componentes de um alvo são acessões UniProt, lidas pelo
  `uniprot_entry`.

##### `_gbif`

Documentação: https://techdocs.gbif.org/en/openapi/ (paginação e 429); o código dos serviços, onde
a documentação OpenAPI é gerada: `gbif/matching-ws` (`MatchV1Controller`: `name` é "the
scientific name to fuzzy match against"; o v1 está marcado obsoleto),
`gbif/checklistbank` (`SpeciesResource`: a pesquisa cobre "the scientific and vernacular names";
`offset` até 100 000), `gbif/occurrence` (`OccurrenceSearchResource`: 300 por página,
`offset + limit` até 100 000; `hasCoordinate=false` dá só registos **sem** coordenadas),
`gbif/gbif-common-ws` (`IllegalArgumentExceptionMapper`: um 400 em texto simples).

- **"Common names" (suspeita confirmada):** o serviço de correspondência só resolve nomes
  científicos. A docstring di-lo e manda os nomes comuns para o `gbif_species_search`, que os
  cobre. Um nome sem correspondência (`matchType: NONE`) passa a `not_found` com o passo seguinte
  (antes, sucesso "No GBIF species match found."); "Multiple equal matches" diz para dar o reino
  ou a autoria. `FUZZY` e `HIGHERRANK` dizem o que querem dizer; um sinónimo dá a chave aceite.
- **`gbif_species_search`:** `list_window` com `count`; mostra os nomes comuns que contêm a
  consulta (porque é que bateu) e, num táxon de outra checklist, a chave do backbone (a que a
  pesquisa de ocorrências aceita). `offset` `Range(0, 100_000)`.
- **`gbif_occurrence_search`:** `offset` `Range(0, 99_999)`; o `limit` pedido é
  `min(max_results, 100 000 − offset)` (o tecto da fonte, como o `wiki_search`); no fim do
  alcance o rodapé diz "the rest cannot be read here" e o cabeçalho diz para estreitar os filtros
  ou usar o serviço de download. Cada registo leva a ligação `https://www.gbif.org/occurrence/N`
  (a chave sozinha não é aceite por nenhuma tool), as coordenadas sem notação científica e a base
  do registo em palavras.
- **Corrigido de passagem:** `has_coordinate=False` mandava `hasCoordinate=false`, que pede só
  registos sem coordenadas; agora não filtra, como a docstring sempre disse.
- `gbif_species` recusa uma chave maior do que um inteiro de 32 bits antes do pedido.

#### Fusões e nomes tirados

Nenhuma fusão e nenhum nome tirado ou renomeado (justificação em `_uniprot`, acima). Parâmetros
novos: `uniprot_search(cursor=)`; `uniprot_entry(offset=, max_chars=)`;
`uniprot_features(offset=)`; `uniprot_crossrefs(offset=)`. O `nanope` não usa nenhuma destas
tools (grep).

#### Mudanças em ficheiros partilhados

- `src/ai_arch_toolkit/toolkit/tools/_http.py` (aditiva): `Api.get_json_reply(*segments, parse,
  params)`, em que o `parse` recebe o `Reply` inteiro (estado, cabeçalhos, o objecto JSON). O
  `_json_answer` passa a ser `_json_reply(...).body` (o mesmo caminho: leitor de erros, vazio
  permitido). Testes: `tests/toolkit/test_http.py::TestWholeReply` (4). Serve a qualquer fonte
  que pagine por `Link` ou conte num cabeçalho; o `CONTRIBUTING.md` (fora do meu âmbito) pode
  ganhar uma linha.
- `tests/toolkit/biodata_answers.py` (novo): as respostas das quatro fontes para os casos de
  contrato, como o `wiki_pages.py`.
- `tests/toolkit/contract_cases.py`: só entradas destas fontes (um bloco "Life sciences (T07b)" em
  `WINDOW_CASES` e em `ZERO_CASES`, e `gbif_species_match` em `NOT_FOUND_CASES`).
- `tests/toolkit/error_bodies.py`: `_uniprot` (o 400 passa a `validation_error`, mais o exemplo da
  documentação), `_pdb`, `_chembl`, `_gbif` (novas).
- Para o coordenador: a T09a acrescenta `Window.rest`; o teste do fim de alcance do GBIF afirma o
  rodapé de hoje ("the rest cannot be read here"). Se a T09a mudar o texto por omissão, esse teste
  acompanha; o GBIF pode usar o `rest` para "use GBIF's download service".

#### Prova

Itens da ficha → testes:

- **`uniprot_features`/`uniprot_crossrefs` ficavam com 25 sem contagem:**
  `test_uniprot.py::TestFeatures::test_every_feature_can_be_read_page_by_page`,
  `TestCrossrefs::test_every_cross_reference_can_be_read_page_by_page`,
  `test_a_reference_keeps_all_its_properties`; filtros: `test_a_type_with_no_features_names_the_types_there_are`,
  `test_the_type_filter_ignores_case`, `test_a_database_with_none_names_the_databases_there_are`.
- **FUNCTION a 500:** `TestEntry::test_the_entry_reads_every_annotation_whole`,
  `test_every_kind_of_annotation_reads_as_text`, `test_a_long_entry_reads_window_by_window`.
- **`offset` ignorado / cursor:** `TestSearch::test_a_search_reads_on_by_the_cursor_the_link_header_gives`,
  `test_the_last_page_has_no_link_and_says_the_end`,
  `test_an_offset_without_its_cursor_is_refused_before_any_request`.
- **"Common names" do `gbif_species_match`:**
  `test_gbif.py::TestMatch::test_no_match_is_not_found_and_points_common_names_to_the_search`,
  `test_several_equal_matches_say_why`, `TestSpeciesSearch::test_a_search_pages_and_shows_the_common_names_that_matched`.
- **`pdb_search` só com IDs:** `test_pdb.py::TestSearch::test_each_hit_comes_with_its_title_from_one_more_request`,
  `test_the_next_page_is_numbered_on`, `test_an_nmr_entry_has_no_resolution_and_a_missing_record_says_so`.
- **Fusão (D41):** decidida contra, na nota; `pdb_ligands` com dois pedidos:
  `TestLigands::test_every_ligand_comes_from_one_more_request`,
  `test_an_entry_without_ligands_says_so_in_one_request`.
- **204 da RCSB:** `TestSearch::test_a_search_without_hits_is_a_success_that_names_the_query`, e o
  `ZERO_CASES["pdb_search"]` do contrato.
- **404 vs 400 da UniProt:** `test_a_404_on_an_accession_is_not_found` (corpo da documentação),
  `TestSearch::test_a_search_uniprot_refuses_is_the_callers_to_fix_in_uniprots_words` (400 →
  `validation_error`; o teste antigo afirmava `upstream`, mudou com o comportamento),
  `test_invalid_arguments_do_not_call_the_api[ABC123]` (formato oficial).
- **Acessão fundida (303):** `TestEntry::test_an_accession_merged_into_another_says_which_entry_answered`.
- **Erros tipados das quatro fontes:** `test_a_search_rcsb_refuses_is_the_callers_to_fix_in_its_words`,
  `test_an_error_the_graphql_endpoint_reports_in_a_200_is_a_failure`,
  `test_chembl.py::TestSearches::test_a_request_chembl_refuses_is_the_callers_to_fix_in_its_words`,
  `test_gbif.py::test_a_request_gbif_refuses_is_the_callers_to_fix_in_its_words`.
- **Saída (ponto 5):** fases com rótulo (`test_every_phase_has_its_label`,
  `test_a_molecule_without_a_phase_is_preclinical`), unidades (`test_a_molecule_reads_with_phase_labels_and_units`,
  `test_a_component_reads_with_its_weight_in_daltons`), datas e Å
  (`test_an_entry_reads_with_units_dates_and_ids_other_tools_take`), coordenadas sem notação
  científica (`test_occurrences_read_with_labels_dates_and_a_link`).
- **Corrigidos de passagem:** `test_filters_go_in_the_query_and_a_name_with_spaces_stays_one_value`,
  `test_an_unreviewed_entry_is_named_by_its_submission_name`, `test_the_query_syntax_uniprot_takes_passes`;
  o `has_coordinate` no `test_occurrences_read_with_labels_dates_and_a_link` (só `true` vai).
- **Profundidade do GBIF:** `TestOccurrences::test_past_gbifs_depth_the_footer_says_the_rest_is_not_here`;
  chave de 32 bits: `test_a_key_past_gbifs_integers_is_refused_before_any_request`.
- **Porta:** `test_http.py::TestWholeReply` (4).
- **Contrato:** `test_tool_contract.py` passa nas 18 tools, sem dívida; `test_tool_invariants.py`
  passa nelas.

#### Ao vivo, pelo dono

APIs gratuitas e sem chave; um a três pedidos por comando. Chamadas cruas: uma falha aparece como
`ToolFailure`.

```bash
uv run python -c "
from ai_arch_toolkit.toolkit.tools import uniprot_search
first = uniprot_search('insulin', organism='Homo sapiens', max_results=3)
print(first.value)
print(uniprot_search('insulin', organism='Homo sapiens', max_results=3, **first.metadata['window']['next_call']).value)"
```

```bash
uv run python -c "from ai_arch_toolkit.toolkit.tools import uniprot_entry; print(uniprot_entry('P04637', max_chars=4000).value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import uniprot_entry; print(uniprot_entry('Q00015').value[:400])"
uv run python -c "from ai_arch_toolkit.toolkit.tools import uniprot_features; print(uniprot_features('P04637', feature_type='Natural variant', offset=25).value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import uniprot_crossrefs; print(uniprot_crossrefs('P04637', database='PDB').value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import uniprot_search; uniprot_search('nosuchfield:abc')"
```

```bash
uv run python -c "from ai_arch_toolkit.toolkit.tools import pdb_search; print(pdb_search('hemoglobin', max_results=3).value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import pdb_search; print(pdb_search('zzqqxxyyvvww').value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import pdb_ligands, pdb_entry; print(pdb_ligands('4HHB')); print(pdb_entry('4HHB'))"
```

```bash
uv run python -c "from ai_arch_toolkit.toolkit.tools import chembl_molecule, chembl_target; print(chembl_molecule('CHEMBL25')); print(chembl_target('CHEMBL203'))"
uv run python -c "from ai_arch_toolkit.toolkit.tools import chembl_activity_search; print(chembl_activity_search(molecule_chembl_id='CHEMBL25', standard_type='IC50', max_results=3).value)"
```

```bash
uv run python -c "from ai_arch_toolkit.toolkit.tools import gbif_species_match; print(gbif_species_match('Puma concolor'))"
uv run python -c "from ai_arch_toolkit.toolkit.tools import gbif_species_match; gbif_species_match('mountain lion')"
uv run python -c "from ai_arch_toolkit.toolkit.tools import gbif_species_search; print(gbif_species_search('cougar', max_results=3).value)"
uv run python -c "from ai_arch_toolkit.toolkit.tools import gbif_occurrence_search; print(gbif_occurrence_search(country='US', max_results=3, offset=99998).value)"
```

O que confirmar:

- UniProt: o segundo `uniprot_search` começa no 4 e são outras proteínas (o cursor anda); o total
  bate com o site; o `uniprot_entry('Q00015')` chega em JSON depois do 303 (o redirect larga o
  `format=json`; a porta não manda `Accept`) e o cabeçalho diz "which UniProt gives for Q00015";
  o `nosuchfield:abc` é `validation_error` com a mensagem da UniProt.
- RCSB: títulos, métodos, resoluções e datas reais no `pdb_search`; o 204 dá "No RCSB PDB entries
  match"; os ligandos do 4HHB com os componentes (HEM, PO4); o GraphQL aceita as duas consultas
  tal como estão (um campo errado volta como `errors`, e a tool falha com a mensagem da RCSB).
- ChEMBL: a fase vem como texto `"4.0"` e sai "4 (approved)"; as actividades têm nomes e pChEMBL.
- GBIF: "mountain lion" é `not_found` a apontar para a pesquisa; "cougar" acha *Puma concolor*
  com o nome comum; a ocorrência no fim do alcance diz "the rest cannot be read here".

#### Bloqueios e achados

- **Pedidos à documentação no host da API:** duas tentativas de ler documentação por WebFetch no
  host da UniProt (`rest.uniprot.org/help/pagination` e `rest.uniprot.org/uniprotkb/api/docs`,
  as páginas que o site renderiza) responderam 500, sem conteúdo. Não houve outros pedidos a
  APIs. Li o manual pelo repositório oficial no GitHub.
- **O `/v1/species/match` do GBIF está obsoleto:** o `MatchV1Controller` diz "this method will be
  removed and users are advised to migrate to the version 2 API v2/species/match". Proposta para
  uma ficha nova: passar o `gbif_species_match` para o v2 (resposta com outra forma), antes que o
  v1 dê 404 ("endpoint not found").
- **Corpo do 400 da RCSB:** nenhuma página da RCSB documenta os corpos de erro; o
  `{"status": 400, "message": …}` vem do teste da T02 e o cliente da própria RCSB lê o corpo de um
  400 como motivo. Confirmar ao vivo com uma pesquisa mal formada, se o dono quiser.
- **Erros do ChEMBL:** lidos do código dos web services (repositório arquivado em 2026-06); o
  serviço vivo pode ter mudado de implementação.
- **Saldo de linhas:** os quatro módulos passam de 1314 para cerca de 2010 linhas: a anotação
  completa da UniProt, o GraphQL da RCSB e os rótulos não existiam. Sai o `_trim`, saem os quatro
  `_bounded` e as guardas `_check_offset`.
- **Revisão:** fiz a revisão adversarial do diff eu próprio; dela saíram as correcções do `OR`
  na consulta, das aspas no organismo, da confiança 0 do GBIF, da cinética sem texto (que caía
  calada), da chave do backbone e da guarda do FASTA (`uniprot_sequence` recusa uma resposta que
  não é FASTA, que um redirect de acessão fundida pode trazer). Lancei também um revisor
  independente, só de leitura, mas o relatório dele não chegou antes da entrega: fica por
  incorporar, e convém repetir essa revisão antes do commit.
- **Ao vivo, também:** `uniprot_sequence('Q00015')` (o 303 de uma acessão fundida com o sufixo
  `.fasta`).

#### Correcções da revisão

Na checkout principal (`main`, sem commits), só nos quatro módulos, nos seus testes e nas entradas
destas fontes em `error_bodies.py` (`contract_cases.py` e `biodata_answers.py` ficaram como
estavam). Cada teste foi escrito antes da correcção e visto a falhar.

1. **`pdb_search`: as linhas fora do `parse=` (Média).** As linhas passam a ser feitas dentro do
   `parse` do GraphQL (`_described`, que substitui o `_records` e serve também o `pdb_ligands`), e
   cada leitura de um objecto passa por `_object` (o `rcsb_entry_info` em texto ou em lista dá
   `{}`; o mesmo no `pdb_entry` e no `pdb_chemical_component`). Um registo assim lê-se sem a
   resolução; uma forma que nem isso deixa ler é a falha de parse da porta, tipada. Testes:
   `test_pdb.py::TestSearch::test_a_record_of_another_shape_reads_without_what_it_lacks` (texto e
   lista; antes, `AttributeError` cru) e `test_records_the_parse_cannot_read_are_a_typed_failure`.
2. **`uniprot_search`: cursor sem offset (Média).** Recusa-se antes do pedido, como o offset sem
   cursor (uma só condição: os dois vêm juntos do rodapé), com "pass both from the previous page's
   footer, or neither for the first page". Teste:
   `TestSearch::test_a_cursor_without_its_offset_is_refused_before_any_request`.
3. **O 303 da acessão fundida, pelo opener (Média).** Os testes servem agora o 303 pelo opener
   real da porta (`_SameHostRedirect` incluído), com um transporte enlatado (`_Transport` +
   fixture `web`, como em `test_http.py`): `TestRedirect::test_the_documented_redirect_is_followed_and_the_heading_names_the_entry`
   (o `Location: /uniprotkb/P23141?from=Q00015` da documentação, relativo: segue-se, e o
   cabeçalho nomeia a P23141). Um `Location` em `http://` é recusado pela porta como downgrade;
   o `_uniprot` lê então o URL recusado e responde `not_found` com a acessão de destino e a mesma
   tool ("Q00015 is inactive: UniProt redirects it to P23141; look up P23141 with
   uniprot_entry"), nas quatro tools de acessão (`_entry` e o `uniprot_sequence`):
   `test_a_redirect_down_to_http_names_the_accession_to_look_up` (×4; antes, `upstream` sem
   passo). Um redirect recusado para a mesma acessão (o `http://…:8080` da colecção) continua a
   falha da porta: `test_a_refused_redirect_to_the_same_accession_stays_the_doors_failure`.
   **Mudança proposta à porta (não feita: o `_http.py` é de outro agente nesta ronda):** o
   `_uniprot` lê o URL recusado no `__cause__` do `HttpError` (o `target` do `_Redirected`), por
   `getattr`, sem importar o nome privado; se a porta mudar esse nome, o teste acima falha. A
   correcção de raiz seria a porta dar o destino recusado num atributo público
   (`HttpError.redirect_to`) ou, melhor, seguir um `http://` do mesmo host como `https://` (o
   balanceador da UniProt escreve `http` nos URLs que devolve: o `url` dos corpos de erro e o
   redirect da colecção); com isso, o `_redirected` deixa de ser preciso.
4. **GBIF: o `min(max_results, depth - offset)` invisível ao harness (Baixa).** Fica o tecto,
   explícito: `_occurrence_page(offset, max_results)` devolve o `limit` e o alcance, com a razão
   na docstring, e o rodapé da página que o tecto encurta nomeia o `max_results`: "[results
   99996-100000 of 400000 | GBIF's search reaches the first 100000 (max_results=10 stops there:
   5 on this page); narrow the filters, or use GBIF's download service, for the rest]" (o
   `Window.rest` da T09a, por `dataclasses.replace`, porque o `list_window` não o recebe). Testes:
   `TestOccurrences::test_past_gbifs_depth_the_footer_says_the_rest_is_not_here` (actualizado:
   afirmava "the rest cannot be read here") e
   `test_a_page_that_ends_at_gbifs_depth_names_no_shortfall`.
5. **GBIF: profundidade das espécies com um a menos (Baixa).** Cada lista declara o maior offset
   que a fonte aceita (`_Reach.last_offset`: 100 000 nas espécies, `SpeciesResource.checkDeepPaging`
   só recusa acima; 99 999 nas ocorrências, `offset + limit ≤ 100 000`), e o rodapé dá a chamada
   seguinte enquanto o offset seguinte cabe (`shown <= last_offset`). Cabeçalho e rodapé dizem o
   alcance de cada pesquisa (as espécies não têm serviço de download: "narrow the query, the rank
   or the higher taxon"). Teste:
   `TestSpeciesSearch::test_the_search_pages_to_the_last_offset_gbif_takes` (99 990 →
   `next: offset=100000`; em 100 000, o fim com o motivo).
6. **ChEMBL: notação científica (Baixa).** `_number`: `decimal_text` de `_values.py` para o que a
   ChEMBL manda em texto ("1E-7" → 0.0000001; "10.0" fica "10.0") e `plain` para um número JSON;
   no valor medido e no pChEMBL. O `plain` sozinho não chegava: deixa o texto como veio. Teste:
   `TestSearches::test_a_value_reads_without_scientific_notation` ("1E-7", `1e-07`, "2.50E+3").
7. **PDB: um 204 com `start>0` dizia "of 0" (Baixa).** `_Hits.total` é `None` quando a resposta
   não traz `total_count` (o 204); o rodapé fica "[no results from 11 | end]". Teste:
   `TestSearch::test_a_page_past_the_hits_does_not_claim_a_total_of_zero`.
8. **PDB: mensagem do GraphQL (Baixa).** O leitor único `_rcsb_error` parte-se em três, um por
   API: `_search_error` (400 → "rephrase the query in plain words"), `_data_error` (400 → "check
   the ID, or find one with pdb_search"; antes dizia "rephrase the query" a um ID) e
   `_graphql_error` ("RCSB PDB's GraphQL endpoint said: …; pdb_entry reads an entry without it,
   or try again later"). A pesquisa deixa de ler `errors`, que só o GraphQL manda. Testes:
   `test_an_error_the_graphql_endpoint_reports_in_a_200_is_a_failure` (agora no `pdb_search` e no
   `pdb_ligands`), `test_the_search_api_does_not_read_graphqls_errors`,
   `TestEntry::test_a_request_the_data_api_refuses_names_a_step_that_fits`.
9. **`error_bodies.py`, o caso GraphQL do `_pdb` (Baixa).** O harness serve uma só resposta a
   todos os pedidos, e o erro do GraphQL só chega ao segundo pedido do `pdb_search` e do
   `pdb_ligands`; o caso passava porque o leitor da pesquisa lia `errors`. Com a correcção 8, o
   caso deixava a pesquisa sem resultados (o teste de contrato falhou, como devia). Saiu; fica um
   comentário a apontar para o teste do `test_pdb.py` que serve as duas respostas em sequência.
   O ponto 2 do contrato continua coberto, nas quatro tools, pelo 400 da pesquisa.
10. **IDs devolvidos que nenhuma tool aceita (menor).**
    - **Ensaios ChEMBL: aceitam-se.** `chembl_activity_search(assay_chembl_id=…)`, último
      parâmetro (não muda posições), filtro que a documentação mostra
      (`activity?molecule_chembl_id=CHEMBL998&assay_chembl_id=CHEMBL1909156`,
      https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services).
      Listar as outras medições do mesmo ensaio é um passo real de quem compara compostos.
      Testes: `TestSearches::test_the_assay_an_activity_names_lists_its_activities` e o caso
      `invalid assay_chembl_id` em `test_invalid_options_do_not_call_api`.
    - **Features UniProt: rotulam-se.** O `featureId` (VAR_…, PRO_…) não é aceite pela pesquisa
      da UniProt nem cabe noutra tool; serve para citar (literatura, ClinVar). Fica na linha, e a
      página que mostra algum diz uma vez, no cabeçalho, "In brackets: UniProt's feature ID, for
      citing (no tool takes it)." (`_Part.own_id`). Testes:
      `TestFeatures::test_a_feature_shows_its_range_its_change_and_its_id` e
      `test_a_page_without_feature_ids_says_nothing_of_them`.

##### Para o coordenador

- `CHANGELOG` (Added): "`chembl_activity_search` takes `assay_chembl_id`, the assay each
  measurement names." (Fixed): "`uniprot_search` refuses a `cursor` without its `offset` (the
  page was numbered from 1); `pdb_search` no longer fails on a record of an unexpected shape, nor
  says 'of 0' past the hits; `gbif_species_search` reads on to offset 100,000, which GBIF serves;
  ChEMBL values show no scientific notation; an inactive UniProt accession redirected to plain
  HTTP is `not_found` naming its entry."
- `docs/tools-catalog.md`, linha do `chembl_activity_search`: "of a molecule, on a target, in an
  assay, or any of them together".
- A mudança proposta à porta (ponto 3).

##### Verificação

`uv run pytest tests/toolkit/test_{uniprot,pdb,chembl,gbif}.py tests/toolkit/test_tool_contract.py
tests/toolkit/test_tool_invariants.py tests/test_architecture.py tests/test_quality_budget.py -q`:
999 passed. `uv run pytest tests/toolkit -q`: 2590 passed, 2 xfailed. `uv run pyright src`: 0
erros. `ruff check` limpo nos ficheiros tocados; `ruff format` só nos quatro módulos e nos quatro
ficheiros de teste (não no `error_bodies.py`, já formatado).
