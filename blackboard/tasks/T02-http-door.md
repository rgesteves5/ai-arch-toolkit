# T02 · A porta HTTP valida cada fonte

- **Dono:** Claude (2026-10-05) · **Estado:** done · **Depende de:** T01
- **Origem:** plano, secção 3.2 e anexos C e F · **Decisões:** D38, D42 · **Regras:** `T00-rules.md`
- **Já feito:** `c259e0b` deu ao `Api` o `body_error`, que lê os erros que uma fonte manda num 2xx,
  e declarou leitores para a MediaWiki (`mediawiki_error`), o World Bank, o ESearch do PubMed, o
  Internet Archive, o Overpass e o GDELT. Ainda devolvem strings. `6a34668` pôs o `body_error` a
  ler também o corpo de um 4xx ou 5xx (o texto toma o lugar da razão do estado; leitores novos no
  Eurostat, no arXiv e na UniProt) e criou a declaração de vazio por chamada (`allow_empty=True`,
  usada pelo `pdb_search`). O estado já chega ao `_json_answer`; os cabeçalhos ainda não.

## Problema

- **Pela T01 (2026-10-05):** o `HttpError` já é um `ToolFailure`. A revisão da T01 deixou aqui:
  - os onze `status_messages={404: "no matching records found."}`, que hoje saem `upstream` com
    uma mensagem que parece "sem resultados". As docstrings do `earthquake_event` e do
    `rxnorm_concept` prometem `not_found`;
  - o `body_error` só sabe dar `upstream`, quando deve poder dar um tipo:
    - as entradas inactivas da UniProt são `not_found`;
    - o erro 120 do World Bank (um indicador desconhecido, num 200) é `not_found`;
    - o `missingtitle` e o `invalidtitle` da MediaWiki, hoje lidos pela mensagem;
    - o 413 `ASYNCHRONOUS_RESPONSE` do Eurostat;
    - o `ratelimited` da MediaWiki e o 503 do Open Food Facts são `rate_limited`;
  - uma chave em falta ("no key: set X") é hoje `upstream`;
  - os 400 do Overpass e do SPARQL perdem a razão da fonte.

- O `body_error` só vê o corpo de um 2xx. O estado e os cabeçalhos não chegam aos módulos, o que
  impede ler o `MediaWiki-API-Error` ou o cabeçalho `message` da NVD.
- Doze módulos declaram `status_messages={404: "no matching records found."}`: `_chembl`,
  `_earthquake`, `_eonet`, `_eurostat`, `_gbif`, `_mediawiki`, `_openfda_food`, `_pdb`, `_ror`,
  `_rxnorm_dailymed`, `_uniprot` e `_who_gho`. Qualquer 404, incluindo o de um endpoint errado (o
  defeito do `uniprot_search` antes de 28/09), lê-se como "sem resultados".
- O corpo de um erro não chega à mensagem: só o `_air_quality` e o `_gdelt` lêem o `e.body`.
- Um 204 ou um corpo vazio dá "could not parse API response", e a RCSB responde "sem resultados" com
  204.

## Objectivo

Cada `Api` declara uma vez como a sua fonte sinaliza erro e o que cada estado quer dizer. A falha sai
como `ToolFailure` (T01), com o código e o texto da fonte.

## Desenho (confirma-o na nota de desenho)

- **Um leitor por `Api`** recebe o estado, os cabeçalhos e o corpo (JSON ou texto) e devolve
  `ToolFailure | None`. Corre para todas as respostas, antes do `parse=`. O `body_error` e o
  `status_messages` fundem-se nele: sem caminhos duplos.
- **O 404 declara-se por chamada.** Num recurso (uma página, uma entrada, um DOI), `not_found`; por
  omissão, `upstream` com "endpoint not found; the API may have changed".
- **Sem leitor próprio**, a mensagem de um 4xx ou 5xx leva o texto de erro da fonte (os campos
  `message`, `error` ou `detail`, ou os primeiros 300 caracteres).
- **O corpo vazio e o 204** declaram-se por chamada, para as fontes que os usam para "sem
  resultados".
- **Os leitores de `c259e0b`** (`_gdelt`, `_geo`, `_internet_archive`, `_mediawiki`, `_overpass`,
  `_pubmed`, `_wikidata`, `_wikipedia`, `_world_bank`) passam para o leitor novo e ficam tipados.

## Nota de desenho (2026-10-05)

- **Um leitor por `Api`, `error_reader`**, no lugar do `body_error` e do `status_messages`. Recebe
  uma `Reply` (o estado, os cabeçalhos e o corpo: o JSON lido, ou o início do texto) e devolve:
  - `None` quando a resposta não reporta erro;
  - um texto, que é a falha nas palavras da fonte, com o tipo do estado (429 `rate_limited`, 5xx
    `upstream` retryable, o resto `upstream`);
  - um `ToolFailure`, quando a fonte diz o que aconteceu: uma entrada inactiva da UniProt é
    `not_found`, o `ratelimited` da MediaWiki é `rate_limited`.
  Corre em todas as respostas JSON, antes do `parse=`, e nos estados de erro de qualquer pedido.
  Um leitor que tropeça no JSON de um sucesso é um erro de parse (uma forma que ninguém esperava);
  noutro corpo, ou se devolve texto vazio, não explica nada. O texto dele entra na frase do
  estado ("endpoint not found …: texto", "rate limited … X said: texto"); um `ToolFailure`
  substitui-a. Um `rate_limited` tipado descansa o host como um 429.
- **O 404 declara-se por chamada:** `missing="…"` nos `get_*`/`post_*` diz que a chamada pede um
  recurso, e um 404 é então `not_found` com essa mensagem (o passo seguinte incluído). Sem a
  declaração, um 404 é `upstream`: "{API}: endpoint not found (HTTP 404); the API may have
  changed", com o texto da fonte se o leitor o der. A declaração ganha ao leitor. Num URL que veio
  de quem chama (`Api.within`), um 404 sem declaração é `validation_error`: "nothing answers at
  …; check the URL".
- **Sem leitor**, a mensagem de um 4xx/5xx leva o texto de erro da fonte: o `message`, o `error`
  ou o `detail` de um corpo JSON (texto, ou o `message` de um objecto), ou os primeiros 300
  caracteres de um corpo de texto que não é HTML.
- **O vazio:** o `allow_empty` fica como está (204 ou corpo vazio declarados por chamada); o
  `empty_on_404` declara só o 404, como o do openFDA.
- **Desaparece:** `status_messages`, `body_error`, os `except HttpError` que traduzem um 404 em
  `not_found` (passam a `missing=`), e o `startswith("missingtitle")` da MediaWiki (o leitor
  devolve o tipo).
- **Uma chave em falta** ("no key: set X") é `validation_error`: o agente não a pode dar, mas a
  chamada não deve repetir-se, e a mensagem diz como a pôr. Nenhum dos quatro tipos é melhor.
- **Arquitectura:** nenhum módulo declara `status_messages` ou `body_error` (desaparecem do
  `Api`), e os `except HttpError` que traduziam um 404 dão lugar ao `missing=`. Fica o do EONET,
  que acrescenta uma pista ao 500 com que a fonte responde a um ID desconhecido.

## Passos

1. Nota de desenho.
2. `_http.py`, com testes da porta em `tests/toolkit/test_http.py`, escritos primeiro.
3. Migrar os leitores de `c259e0b` e os `status_messages` dos catorze módulos que os declaram (os doze
   acima, mais `_open_food_facts` e `_overpass`), escolhendo em cada chamada o que é um recurso.
4. As guardas `isinstance` que nunca disparam e que restem (o anexo C cita `_eurostat.py:226`).
5. Docs: a secção das tools de rede no `CONTRIBUTING.md` e no `AGENTS.md`, e `docs/tools.md`.
   `CHANGELOG` onde o comportamento visível mudar (um 404 de endpoint deixa de ser "sem resultados").

## Ficheiros

`toolkit/tools/_http.py`, os módulos com `status_messages` ou `body_error` listados acima e os seus
testes, `tests/toolkit/test_http.py`, `CONTRIBUTING.md`, `AGENTS.md`, `docs/tools.md`.

## Prova

- **Porta:**
  - um 404 sem declaração dá `upstream` com a mensagem de endpoint;
  - com a declaração de recurso, dá `not_found`;
  - um 200 com cabeçalho de erro dá falha;
  - o corpo JSON de um 400 aparece na mensagem;
  - um 204 declarado chega ao `parse` como vazio.
- Os testes de `c259e0b` continuam verdes, agora com o tipo.
- **Arquitectura:** nenhum módulo fora da porta monta mensagens a partir de estados HTTP.

## Fora do âmbito

As suspeitas por módulo do anexo C que não têm leitor (`hacker_news`, o 204 da RCSB no `_pdb`, os
`eurostat_*`, o `eonet_event`, o arXiv, a Open Library, o `wikidata_entity` e as entradas inactivas da
UniProt). Cada uma fica na ficha do seu grupo (T06 a T09), já com a porta nova.

## Registo do dono

- **Estado:** done (Claude, 2026-10-05), pela nota de desenho. A porta, o teste de arquitectura,
  as docs e o CHANGELOG fê-los o coordenador; os 31 módulos, quatro agentes em paralelo, com
  módulos e testes disjuntos e o mesmo guia.
- **Porta (`_http.py`):**
  - `Reply` e `ErrorReader`; o `Api(error_reader=)` substitui o `body_error` e o
    `status_messages`;
  - `missing=` e `empty_on_404=` por chamada;
  - o 404 sem declaração é "endpoint not found"; num URL de quem chama (`Api.within`), é
    `validation_error`;
  - sem leitor, o texto de erro da fonte;
  - uma chave em falta é `validation_error`;
  - um `rate_limited` tipado descansa o host, e o `retry_after_s` chega aos `details`.
- **Módulos:**
  - `missing=` em cada chamada que pede um recurso (ChEMBL, GBIF, PDB, ROR, UniProt, Crossref,
    DataCite, ClinicalTrials, Semantic Scholar, Open Library, Wikidata, WHO, World Bank, RxNorm,
    DailyMed, openFDA, USGS, EONET, dicionário, Internet Archive, Eurostat);
  - leitores tipados: MediaWiki (`missingtitle`, `invalidtitle`, `ratelimited`/`maxlag`, também
    pelo cabeçalho), UniProt (entradas inactivas), World Bank (o 120 no `world_bank_indicator`),
    Eurostat (413), OFF (503), Overpass (o 400), ROR, NVD (o cabeçalho `message`), Brave e Tavily
    (chave recusada, 432/433), Open-Meteo e arXiv (400), GDELT (texto num 200).
- **Arquitectura:** nenhum `except` de uma tool compara o `.status` de uma falha (só o do EONET,
  que acrescenta a pista ao 500).
- **Revisão independente:** sem falhas graves. Corrigi as médias (o `retry_after_s` perdido num
  leitor tipado, o 4xx da NVD todo `validation_error`, o `missing=` do Eurostat com filtros, a prova
  de arquitectura em falta) e as baixas (`empty_on_404` a ligar o `allow_empty`, o corpo JSON
  cortado a 2000 caracteres antes de ser lido, o texto vazio de um leitor, o 404 num URL de quem
  chama, as docstrings).
- Gate: 6898 passed, 42 skipped; ruff, formatação e pyright limpos.
- **Para as T06 a T09** (por confirmar ao vivo): o 204 da USGS (`nodata`), o 404 do Eurostat
  para filtros sem dados, o 404 da UniProt para um acesso desconhecido, e o cabeçalho `message`
  da NVD só nos erros.
