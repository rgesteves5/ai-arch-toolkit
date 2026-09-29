# T09 · Ficheiros, web e o resto

- **Dono:** por atribuir · **Estado:** todo · **Depende de:** T05 (o modelo a seguir)
- **Módulos:** `_filesystem` (3 tools), `_json` (2), `_text` (4), `_web` (2), `_shell` (1),
  `_python` (1), `_youtube` (3), `_internet_archive` (2), `_open_library` (3), `_math` (2),
  `_datetime` (5): 28 tools
- **Origem:** os anexos A a E do plano para estes módulos · **Decisões:** D37 a D42 · **Regras:**
  `T00-rules.md`

## O que está partido

- **Dizem o tamanho, mas não deixam continuar:**
  - `read_file`, `list_directory` e `search_files`;
  - `http_get` e `scrape_text`;
  - `youtube_transcript`, `youtube_transcript_search` e `youtube_transcript_languages`;
  - `csv_read` e `regex_search`;
  - `run_command`.
- **Cortes silenciosos:**
  - o `search_files` corta as linhas aos 300 caracteres e só lê o primeiro milhão de caracteres de
    cada ficheiro;
  - o total do `csv_read` ("N of M rows") só conta o primeiro milhão de caracteres;
  - as listas dos `internet_archive_*` e `open_library_*` são cortadas em silêncio, e as descrições
    ficam nos 1000 caracteres;
  - as pesquisas do Internet Archive e da Open Library paginam sem total.
- **Promessas:**
  - o `youtube_transcript` diz "Increase max_chars" mesmo quando já está no tecto de 50 000;
  - o `unit_convert` arredonda para 4 ou 6 algarismos significativos sem o dizer.
- **Erro como sucesso (suspeita):** registos da Open Library redireccionados ou apagados aparecem
  como "(untitled)".
- **Saída:** os autores da Open Library aparecem como chaves `/authors/OL…A`, não como nomes.
- **O que não se repete:** a saída de um comando (`run_command`) ou de código (`python_repl`) não se
  relê com `offset` sem o correr outra vez. Aí o rodapé diz o tamanho e como estreitar a saída, por
  exemplo com um filtro ou um intervalo de linhas, em vez de uma continuação. Decide-o na nota de
  desenho.

## Objectivo

As 28 tools cumprem o contrato de `T00-rules.md` e saem da lista de dívida.

As tools de ficheiros, shell, Python e web continuam em `toolkit.tools.dangerous`, com os mesmos
gates e capacidades: a janela não muda a governança.

## Passos

1. Nota de desenho por módulo: continuação (linhas e caracteres nos ficheiros, `offset` no texto
   web e nas transcrições, `find` onde servir), limites, falhas tipadas e o caso da shell e do
   Python.
2. Testes primeiro, por módulo (ficheiros com `tmp_path`, rede pela costura `_http._open`).
3. A migração; os `_truncate`/`_trim` destes módulos desaparecem.
4. `docs/tools-catalog.md`, `docs/safety.md` se a governança for tocada (não devia ser), e
   `CHANGELOG`.

## Ficheiros

Os 11 módulos, os seus testes em `tests/toolkit/`, `tests/toolkit/contract_debt.py` (só as linhas
destas tools), `tests/toolkit/error_bodies.py` (as entradas destas fontes), `docs/tools-catalog.md`.

## Prova

- A invariante de contrato passa nas 28 tools, que saem da lista de dívida.
- Seguir os rodapés do `read_file` e do `http_get` reconstrói o conteúdo inteiro.
- O total do `csv_read` é o do ficheiro inteiro.
- O `unit_convert` diz a precisão que usa.
- Os testes de governança (`test_tools_exports.py`, gates) continuam verdes sem mudanças.

## Registo do dono

- **Estado:** todo.
