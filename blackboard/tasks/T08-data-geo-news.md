# T08 · Dados, geo e notícias

- **Dono:** por atribuir · **Estado:** todo · **Depende de:** T05 (o modelo a seguir)
- **Módulos:** `_eurostat` (5 tools), `_world_bank` (7), `_who_gho` (3), `_gdelt` (2), `_overpass` (2),
  `_osm` (2), `_geo` (6), `_weather` (5), `_air_quality` (2), `_eonet` (3), `_earthquake` (3),
  `_wikidata` (3), `_news` (1): 44 tools
- **Origem:** os anexos A a E do plano para estes módulos · **Decisões:** D37 a D42 · **Regras:**
  `T00-rules.md`

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
- **Erro como sucesso (suspeitas):**
  - os `eurostat_dataset`, `_dimensions` e `_series` não verificam erros;
  - o `eonet_event` mostra um evento em branco;
  - um QID fundido lê-se como "not found" no `wikidata_entity`;
  - o `hacker_news` deixa cair, sem nota, as histórias que falham.
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

- **Estado:** todo.
