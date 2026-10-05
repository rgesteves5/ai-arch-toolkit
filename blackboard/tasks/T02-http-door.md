# T02 · A porta HTTP valida cada fonte

- **Dono:** por atribuir · **Estado:** todo · **Depende de:** T01
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

- **Estado:** todo.
