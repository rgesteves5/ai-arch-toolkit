# T06 · Literatura e identificadores

- **Dono:** por atribuir · **Estado:** todo · **Depende de:** T05 (o modelo a seguir)
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
- **Erro como sucesso (suspeita):** a entrada de erro do arXiv apareceria como um artigo com o título
  "Error".
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

- **Estado:** todo.
