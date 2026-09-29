# T05 · Família wiki: MediaWiki, Wikipedia e definições

- **Dono:** por atribuir · **Estado:** todo · **Depende de:** T04b
- **Módulos:** `_mediawiki` (4 tools), `_wikipedia` (3), `_dictionary` (1)
- **Origem:** plano, secções 1, 3.5 e 3.6, e os anexos A a E destes módulos · **Decisões:** D39,
  D40, D41 · **Regras:** `T00-rules.md`

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

- O HTML é maior do que o wikitexto: pede por secção e respeita o tecto de 10 MB da porta.
- As classes CSS da MediaWiki mudam com o tempo: testa sobre fixtures e deixa ao dono a verificação
  ao vivo.

## Registo do dono

- **Estado:** todo.
