# T07 · Vida e saúde

- **Dono:** por atribuir · **Estado:** todo · **Depende de:** T05 (o modelo a seguir)
- **Módulos:** `_clinical_trials` (2 tools), `_uniprot` (5), `_pdb` (4), `_chembl` (5), `_foodon` (2),
  `_gbif` (4), `_rxnorm_dailymed` (6), `_openfda_food` (2), `_open_food_facts` (4): 34 tools
- **Origem:** os anexos A a E do plano para estes módulos · **Decisões:** D37 a D42 · **Regras:**
  `T00-rules.md`

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
- **Erro como sucesso (suspeitas):**
  - a RCSB responde "sem resultados" com HTTP 204 e corpo vazio (usa a declaração de vazio da T02);
  - as entradas inactivas da UniProt perdem o destino da fusão.
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

- **Estado:** todo.
