# T09 · Ficheiros, web e o resto

- **Dono:** Claude (dois agentes, T09a e T09b, 2026-10-08) · **Estado:** done · **Depende de:** T05 (o modelo a seguir)
- **Módulos:** `_filesystem` (3 tools), `_json` (2), `_text` (4), `_web` (2), `_shell` (1),
  `_python` (1), `_youtube` (3), `_internet_archive` (2), `_open_library` (3), `_math` (2),
  `_datetime` (5): 28 tools
- **Origem:** os anexos A a E do plano para estes módulos · **Decisões:** D37 a D42 · **Regras:**
  `T00-rules.md`
- **Já feito:** `c259e0b` (o `error` do Internet Archive) e `6a34668` (a Open Library segue um
  registo fundido e diz que um apagado foi apagado).

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

- **Estado:** done. Feito por agentes em worktrees, juntado e revisto pelo coordenador; uma revisão independente por parte, e as correcções dela (secção "Correcções da revisão").
- **Gate final, no checkout principal com todas as fichas da vaga:** 8195 passed, 118 skipped (os ao vivo e os 59 do nanope à espera); ruff, formatação, pyright e `uv lock --check` limpos (2026-10-08).
- **CHANGELOG:** as linhas entraram em `[Unreleased]` (Upgrade notes, Added, Changed, Fixed).

### T09a

- **Dono:** Claude (agente T09a), worktree
  `.claude/worktrees/agent-a7176ffcd25ed17e5`, 2026-10-08. **Estado:** review (por commitar na
  worktree).
- **Módulos:** `_filesystem` (3), `_json` (2), `_text` (4), `_shell` (1), `_python` (1),
  `_math` (2), `_datetime` (5): 18 tools.
- **Base:** `main` @ `601e38c`; gate antes: 6981 passed, 101 skipped.

#### Nota de desenho

Escrita antes do código. O modelo é a T05: limites `Range` na assinatura, `ToolFailure` com o
passo seguinte, zero resultados como sucesso que diz a consulta, e todo o corte pela janela.

##### Comum: o que corre não se relê (`run_command`, `python_repl`)

- A saída de um comando ou de um programa só existe para a chamada que a fez. Ler o resto com
  `offset` obrigava a correr outra vez, com os efeitos (um comando) ou o custo (um programa). Por
  isso estas duas tools não têm continuação.
- Quando a saída não cabe, mostra-se o início, a acabar numa linha, e o rodapé da janela diz o que
  foi mostrado, o tamanho e como estreitar a saída:
  `[chars 0-7998 of 48893 | the rest is not kept: run the command again narrowed, e.g. | grep PATTERN, …]`.
- A janela ganha um campo `Window.rest` (aditivo): o que o rodapé diz em vez de "the rest cannot
  be read here". Sem ele, o rodapé destas tools seria um beco sem saída.
- Um módulo novo, `_output.py`, guarda o que as duas partilham. O `Head` acumula o início de um
  fluxo e conta o total sem o guardar: a memória fica limitada, por muito que o comando escreva.
  O `fit` divide o limite pelas partes (stdout e stderr; o que o programa escreveu e o seu valor
  ou erro). Uma parte mais curta do que a sua parte igual mostra-se inteira e deixa o resto às
  outras. O código de saída e o erro nunca se perdem por baixo de um stdout longo.
- **Contrato:** o ponto da janela, para estas duas, é esse rodapé. `contract_cases.ONCE`
  declara-as, com o porquê. `test_tool_contract.once_kept` verifica o rodapé: unidade `chars`,
  total, nenhuma chamada seguinte, e nem "end" nem o beco sem saída.
- **`NOT_CALLED`:** o caso do `run_command` é um comando fixo e inofensivo, `seq 1 3000`, como os
  do `test_shell.py`. O `NOT_CALLED` do catálogo continua a afastar dele os argumentos hostis que a
  invariante gera. O contrato não gera argumentos para ele: corre só esse caso, uma vez, sem
  continuação. Se o coordenador preferir que o contrato nunca o corra, a alternativa é declarar o
  ponto em `ONCE` sem caso, com a prova no `test_shell.py`; perde-se a prova pelo contrato.

##### `_filesystem`

- **`read_file(path, offset=0, max_lines=200)`:** janela por caracteres (a unidade da janela).
  - Uma janela tem até `max_lines` linhas e no máximo 100 000 caracteres (o tecto actual), e acaba
    numa linha.
  - O ficheiro lê-se em pedaços de 1 Mi caracteres: salta `offset`, lê a janela e conta o resto
    sem o guardar. A memória fica limitada.
  - O total conta-se quando o resto cabe em 100 milhões de caracteres; para lá disso, o rodapé dá
    só a chamada seguinte, e a janela perto do fim já dá o total. Medido: texto descodifica a cerca
    de 3 GB/s, bytes que não são UTF-8 a 0,12 GB/s. Sem o limite, um ficheiro binário de 10 GB
    passava o prazo do executor só para dizer o total.
  - `newline=""`: os offsets são os dos caracteres do próprio ficheiro, e seguir os rodapés
    reconstrói-o exactamente.
  - Sem cabeçalho: um ficheiro que cabe devolve o texto tal qual (o teste do nanope lê "hello").
- **`list_directory(path, pattern, offset=0)`:**
  - página de 1000 entradas (o tecto actual), por ordem de nome, com o total;
  - o glob pára aos 100 000 resultados, dentro ou fora da pasta, como na G-27. Um padrão que passa
    disso é `validation_error` com "narrow it". Antes, a listagem parava aos 1000 e calava o resto;
  - só as entradas da página fazem `stat`, e cada pasta-mãe resolve-se uma vez.
- **`search_files(directory, pattern, max_results=50, offset=0)`:**
  - cada linha vem como `path:line:offset: text`. O offset é onde começa o texto mostrado, e
    `read_file(path, offset=…)` lê dali (ponto 6 do contrato: um identificador que outra tool
    aceita);
  - os ficheiros lêem-se inteiros, linha a linha, em pedaços de até 64 Ki caracteres: acaba o
    corte calado ao primeiro milhão de caracteres. Uma ocorrência partida entre dois pedaços
    encontra-se, porque os últimos `len(pattern) - 1` caracteres passam ao pedaço seguinte;
  - uma linha com mais de 300 caracteres mostra o troço à volta da primeira ocorrência e di-lo:
    `[part of a 2000008-char line; read_file(offset=…) reads on]`. Acaba o corte calado aos 300;
  - a pesquisa pára quando tem `offset + max_results + 1` linhas. Sem total quando há mais (o
    rodapé dá a chamada seguinte), com total quando acabou;
  - a ordem é a do `os.walk` com nomes ordenados, para que as páginas sejam estáveis. O
    `os.walk` não segue ligações para pastas, como o `rglob` de antes;
  - um padrão vazio é `validation_error`. Binários (o sufixo, ou o ponto onde o UTF-8 falha) e
    ligações para fora da pasta continuam a saltar-se. Só as ligações se resolvem (antes,
    resolvia-se cada ficheiro).
- **Governança:** nada muda. As tools ficam em `dangerous`, com a mesma capacidade, risco,
  aprovação e nomes de argumentos (`path`, `directory`, `pattern`: a D64 vai mapeá-los). As
  leituras continuam sem raízes; isso é da C07.
- Saem `read_prefix` e `clamp`.

##### `_json`

- **`csv_read(path, offset=0, max_rows=100)`:** o ficheiro lê-se linha a linha pelo
  `csv.reader`. Guarda-se só a página e contam-se todas as linhas de dados: o total é o do
  ficheiro inteiro. Antes contava `\n` no primeiro milhão de caracteres, e errava também com
  campos de várias linhas. Cada página repete o cabeçalho, e o rodapé dá o `offset` seguinte. Uma
  linha que não é CSV diz o número da linha e "read it with read_file".
- **`json_extract`:** já dava `not_found`; faltava-lhe o caso do contrato. A mensagem passa a dizer
  as chaves do objecto (até 20, e quantas mais), o tamanho da lista, ou que está vazio.

##### `_text`

- **`regex_search(text, pattern, offset=0)`:** todas as ocorrências contam-se (o texto tem no
  máximo 20 000 caracteres, logo no máximo 20 001 ocorrências). Medi o pior caso que achei,
  `(?=(a*))a` sobre 20 000 `a`: 0,07 s, abaixo do varrimento sem ocorrências que já existia
  (0,9 s). Uma página tem até 1000 ocorrências (o tecto actual) e até 20 000 caracteres, com o
  total e o `offset` seguinte. Zero ocorrências é "No matches for 'p'.", com o padrão.
- `text_stats`, `base64_encode`, `base64_decode`: não cortam e já cumpriam; não mudam.

##### `_shell`

- **`run_command`:** `timeout` e `max_output` passam a `Range` (1–600, 1–100 000). A saída passa
  pelo `fit` (ver "Comum").
- O processo corre por `Popen`, com uma thread por fluxo que o esvazia para um `Head`: a memória
  fica em `max_output` caracteres por fluxo. Antes, o `capture_output` guardava tudo (um `yes`
  de 30 s eram gigabytes), e o corte `[:max_output]` deitava fora o stderr e o código de saída.
- A saída lê-se em UTF-8, com `errors="replace"`. Antes, uma saída binária dava
  `UnicodeDecodeError`, apanhado como "remove the null byte".
- O tempo limite conta também o fecho dos fluxos, como no `subprocess.run`: um processo deixado a
  segurar a saída acaba em "timed out". Os leitores têm meio segundo para esvaziar o que ficou no
  fim.

##### `_python`

- **`python_repl`:** a saída fica nos 20 000 caracteres, pelo `fit`. O que o programa escreve vai
  para um `Head` (antes acumulava tudo), e o valor da última expressão ou o erro vêm a seguir, com
  a sua parte. Sem parâmetro novo: estreitar é mudar o código (imprimir uma fatia ou um filtro).
- Uma saída que cabe não muda; os testes de `tests/test_python_eval.py` e do nanope passam sem
  mudanças.

##### `_math`

- **`unit_convert`:**
  - diz a precisão, "(rounded to 6 significant digits)", e escreve os números sem notação
    científica;
  - a temperatura passava a 4 algarismos e o resto a 6; agora tudo a 6;
  - os factores passam aos valores exactos das definições (libra 453,59237 g, galão americano
    3,785411784 L e as suas partes, acre 4046,8564224 m², km/h = 1000/3600, nó = 1852/3600), para
    que o sexto algarismo seja verdadeiro. Antes, uma colher de sopa eram 3,00001 colheres de chá;
  - um valor ou um resultado que não é finito é `validation_error`.
- `math_eval` não muda.

##### `_datetime`

- **`date_add`:** `days`, `hours` e `minutes` com `Range` no alcance do calendário (anos 1 a 9999),
  derivado de `datetime.max - datetime.min` por divisão inteira de `timedelta` (o
  `total_seconds()` arredondava uma hora a mais). Uma soma dentro dos limites que sai do
  calendário continua a ser `validation_error`.
- As outras quatro não cortam nem têm inteiros; não mudam.

##### Saldo de linhas

Os sete módulos e o `_window.py`: 2221 antes, 2702 depois (com o `_output.py` novo, 76): +481. O
saldo positivo vem da navegação (offsets, páginas, contagem em pedaços), do leitor de linhas do
`search_files` e da captura limitada do `run_command`. Nenhuma função nova passa a complexidade 10
nem as 60 linhas; a linha de base de qualidade não mudou.

#### Fusões e nomes tirados

Nenhuma. Nenhum nome de tool muda, nenhum argumento muda de nome. O nanope continua a importar os
mesmos nomes.

Argumentos novos: `offset` em `read_file`, `list_directory`, `search_files`, `csv_read` e
`regex_search`.

Quebras sem mudança de nome:

- chamadas directas a `read_file`, `list_directory`, `search_files`, `csv_read` e
  `regex_search` devolvem um `ToolResult` (o `value` é o texto), como as da família wiki;
- os limites destas tools e do `date_add` recusam-se pelo executor em vez de se ajustarem;
- formatos de saída: o rodapé da janela; `search_files` com `path:line:offset: text`;
  `unit_convert` com a precisão; `regex_search` com o padrão no cabeçalho;
- `list_directory` recusa um padrão com mais de 100 000 resultados;
- `python_repl` corta aos 20 000 caracteres (antes, só o executor cortava, aos 200 000).

#### Mudanças em ficheiros partilhados

- `src/ai_arch_toolkit/toolkit/tools/_window.py`: `Window.rest` (campo novo, opcional, com
  docstring), dito no rodapé em vez de "the rest cannot be read here". Teste em
  `tests/toolkit/test_window.py` (`test_a_rest_no_call_reaches_says_how_to_narrow_instead`).
- `tests/toolkit/test_tool_contract.py`:
  - `once_kept` e o ramo das tools `ONCE` no `unmet`;
  - o grupo `total` no `_FOOTER`;
  - a verificação de que `ONCE` só nomeia tools que cortam;
  - os testes do próprio `once_kept` em `TestTheChecks` (`narrowed`, `dead_end`).
- `tests/toolkit/contract_cases.py`: além das entradas das minhas tools, a declaração `ONCE` e a
  sua menção na docstring. O `python_repl` sai de `WHOLE` (passou a cortar).
- `tests/toolkit/contract_debt.py`: saem as linhas das minhas oito tools que lá estavam.
  `CUT_HELPERS` fica em 14: os meus módulos não tinham `_truncate` nem `_trim`, e continuam sem.
- Ficheiros novos: `src/ai_arch_toolkit/toolkit/tools/_output.py`, `tests/toolkit/test_output.py`,
  `tests/toolkit/test_python.py`.
- Não toquei: `core/`, `_http.py`, `__init__.py`, `dangerous.py`, `test_tools_exports.py`,
  `tool_catalog.py`, `tests/quality_baseline.json`, `docs/safety.md`, o nanope, o blackboard,
  o `CHANGELOG.md`.

#### Prova

Os testes novos, por item da ficha:

- **Dizem o tamanho mas não deixam continuar** (`read_file`, `list_directory`, `search_files`,
  `csv_read`, `regex_search`, `run_command`):
  - `test_filesystem.py`: `test_a_window_holds_max_lines_and_names_the_next_offset`,
    `test_a_listing_longer_than_a_page_is_paged_with_its_total`,
    `test_results_are_paged_with_the_next_offset`;
  - `test_json.py`: `test_a_page_has_the_header_and_the_next_offset`,
    `test_following_the_footers_reads_every_row_once`;
  - `test_text.py`: `test_a_page_holds_a_thousand_matches_with_the_total_and_the_next_offset`,
    `test_following_the_footers_reaches_every_match_once`,
    `test_a_page_stays_within_its_characters_however_long_the_matches`;
  - `test_shell.py`: `TestLongOutput` (o início, o tamanho e como estreitar; o stderr e o código
    de saída sobrevivem; o tamanho conta tudo, 30 MB);
  - contrato: os casos de janela das cinco, e o `once_kept` do `run_command`.
- **Seguir os rodapés do `read_file` reconstrói o conteúdo inteiro:**
  `test_following_the_footers_rebuilds_the_whole_file` (acentos, `\r\n`, linhas vazias, uma linha
  de 250 000 caracteres, sem quebra final; os pedaços juntos são o ficheiro, e cada janela dá o
  total).
- **`read_file` não guarda o ficheiro inteiro:** `test_the_file_is_read_a_chunk_at_a_time`,
  `test_the_total_counts_the_whole_file_past_the_old_read_limit`,
  `test_the_total_waits_until_the_rest_is_within_the_count`.
- **`search_files` corta aos 300 e lê só o primeiro milhão:** `test_the_whole_file_is_searched`
  (uma ocorrência depois de 2,5 milhões de caracteres),
  `test_a_long_line_shows_the_part_around_the_match_and_says_so` (e `read_file` no offset dado
  continua o troço), `test_a_match_split_between_two_pieces_of_a_line_is_found`,
  `test_each_hit_gives_the_offset_read_file_reads_from`,
  `test_files_come_in_name_order_folder_by_folder`.
- **O total do `csv_read` é o do ficheiro inteiro:**
  `test_the_total_is_the_whole_files_past_the_old_read_limit` (60 000 linhas, 2 MB),
  `test_a_quoted_field_over_several_lines_is_one_row`.
- **O `unit_convert` diz a precisão:** `test_the_answer_states_the_precision_it_rounds_to`,
  `test_temperatures_round_to_the_same_precision`, `test_numbers_are_never_in_scientific_notation`,
  `test_the_factors_are_the_exact_definitions`, `test_a_result_beyond_a_float_is_a_validation_error`.
- **Limites do `date_add`:** `TestDateAddLimits` (o schema, o alcance inteiro, a recusa do
  executor) e o ponto `limits` do contrato.
- **`json_extract` `not_found`:** o caso do contrato;
  `test_a_missing_key_is_not_found_and_names_the_keys_there`,
  `test_many_keys_are_named_with_how_many_more`,
  `test_an_index_out_of_range_is_not_found_and_gives_the_length`,
  `test_an_empty_list_or_object_says_it_is_empty`.
- **O caso do comando e do código:** `test_python.py` (cinco testes: o rodapé, o valor e o erro
  depois de uma saída longa, o `Head` a guardar só o limite, uma saída curta igual, a recusa com a
  saída cortada), `test_output.py` (seis), e
  `TestTheChecks::test_an_output_that_says_its_size_and_how_to_narrow_it_is_kept_once`.
- **Zero resultados:** os casos do contrato de `search_files` e `regex_search`;
  `test_no_matches_say_the_pattern` nos dois.
- **Governança:** `test_tools_exports.py`, `TestReadFileGovernance` e os testes de gates passam sem
  mudanças.
- **Contrato:** as 18 tools passam com a dívida vazia; as oito que lá estavam saíram.

Testes antigos que afirmavam o comportamento que mudou (substituídos, nenhum enfraquecido):

- os de "Truncated", "Showing N of M", "Stopped at", "more not shown": passam a afirmar o rodapé;
- os de "clamped": passam a afirmar a recusa do executor (`validation_error`, T04a);
- `test_list_stops_its_walk_at_the_cap_when_every_match_is_outside`: o passeio continua a parar
  no tecto (1001 com o tecto a 1000), mas a resposta diz que o padrão passou dele, em vez de "no
  entries";
- os da G-27 sobre o `search_files`: as mesmas linhas, no formato novo (`notes.txt:1:0: …`);
- `test_a_shell_that_cannot_start_is_an_upstream_failure` simula o `Popen` em vez do `run`.

#### Bloqueios e achados

Nenhum bloqueio.

Achados fora do âmbito, com reprodução, para o `FINDINGS.md` (o coordenador decide):

1. **`python_repl`: repetir uma sequência não tem guarda.** O `**` tem `_MAX_POWER`, mas
   `"x" * 10**9` reserva 1 GB, e `"x" * 10**10` acaba em `MemoryError`, que não está na lista de
   erros do programa. Reprodução: `python_repl('len("x" * 10**9)')`.
2. **`run_command`: o tempo limite mata só a shell.** Um processo que ela lançou (`sleep 100; …`
   numa sequência) sobrevive, como antes com o `subprocess.run`. Se segurar a saída, os dois
   leitores (threads daemon) ficam à espera dele. Um grupo de processos resolvia, mas muda o
   tratamento de sinais: é uma decisão do dono. Reprodução:
   `run_command("sleep 100; echo x", timeout=1)`, depois `pgrep -f "sleep 100"`.
3. **`run_command` herda o stdin do processo.** Um comando que lê o stdin (`cat`) espera pelo
   terminal do anfitrião até ao tempo limite. `stdin=DEVNULL` resolvia; já era assim.
4. **`regex_search`: uma ocorrência com muitos grupos longos** faz uma linha de milhões de
   caracteres. O limite da página aplica-se entre linhas, e a linha passa ao corte do executor
   ("kept X of Y"). Reprodução: um padrão com 100 grupos aninhados à volta de `a*`, sobre 20 000
   `a`.
5. **`read_file` e `search_files` verificam e depois abrem:** um ficheiro trocado por um FIFO entre
   as duas operações bloqueia a leitura. Já era assim; o lugar é a caminhada com
   `O_NOFOLLOW`/`dir_fd` da C07.
6. **`math_eval` devolve floats em notação científica** (`str(float)`), contra o ponto 5 do
   contrato. Não está na ficha.
7. **A frase de abertura do `docs/tools-catalog.md`** ("the other tools still move a numeric
   argument … to the nearest limit") deixou de valer para as minhas tools. É partilhada: o
   coordenador actualiza-a ao juntar as T06 a T09.

Decisão a registar pelo coordenador (proposta): as tools cuja saída pertence à execução
(`ONCE`) não têm continuação; o rodapé diz o tamanho e como estreitar (`Window.rest`).

Verificação ao vivo: nenhuma; são tools locais.

Divisão proposta em commits:

1. `feat(tools): Window.rest and the output of a run` (`_window.py`, `_output.py`,
   `test_window.py`, `test_output.py`, o `ONCE` e o `once_kept` no contrato);
2. `feat(tools): files read window by window (T09)` (`_filesystem.py`, `_json.py`, os seus testes,
   os casos e a dívida);
3. `feat(tools): regex pages, unit precision, date limits (T09)` (`_text.py`, `_math.py`,
   `_datetime.py`, os seus testes);
4. `feat(tools): command and program outputs within a limit (T09)` (`_shell.py`, `_python.py`,
   os seus testes);
5. `docs(tools): the local tools in the catalog (T09)`.

#### Correcções da revisão

Agente de correcção da T09a, na checkout principal, 2026-10-08, sem commits. Corrigidas as três
Altas, as três Médias e as Baixas 7 a 11 e 13 (o estilo); a 12 fica documentada. Para cada uma,
o teste veio primeiro e vi-o falhar. Ficheiros tocados: os oito módulos, os seus testes em
`tests/toolkit/`, o `once_kept` (e os seus autotestes) em `test_tool_contract.py` e, no
`contract_cases.py`, só a declaração `ONCE`.

##### 1. Alta: `run_command` não deixa processos, fds nem leitores

- **Mudança (`_shell.py`):**
  - o comando corre num grupo de processos próprio (`start_new_session=True`), com
    `stdin=DEVNULL`: um comando que lê a entrada (`cat`) acaba logo, em vez de esperar pelo
    terminal do anfitrião (era o achado 3 do autor);
  - a thread que chama lê os dois pipes sozinha, com um `selectors.DefaultSelector`. Cada pipe
    passa por um descodificador incremental UTF-8 (`replace`), com newlines universais como o
    `subprocess` em modo texto (`io.IncrementalNewlineDecoder`), para o seu `Head`. Já não há
    threads;
  - o fim da shell vê-se com `os.waitid(P_PID, pid, WEXITED | WNOHANG | WNOWAIT)`. A shell fica
    por colher até ao `killpg`, por isso o id do grupo não pode ser reutilizado entre a
    verificação e a morte;
  - o grupo é morto no prazo, ou 0,5 s depois de a shell acabar com os pipes ainda abertos. Segue-se
    o `os.killpg(pid, SIGKILL)`, o `wait()` e o fecho dos fds. O `killpg` corre sempre, também no
    fim normal: um `sleep 30 >/dev/null 2>&1 &` também morre ("nunca deixar um processo"). Um
    `PermissionError` do macOS, num grupo onde só resta a shell zombie, ignora-se (medido);
  - um registo `_RUNNING` e um `atexit` matam os grupos ainda vivos quando o programa sai. O
    executor corre a tool numa thread daemon, que morre a meio sem chegar ao `finally`;
  - só fica fora de alcance um processo que sai do grupo por si (`setsid`, `set -m`). Está dito na
    docstring do módulo.
- **Windows:** fica explicitamente sem suporte. A chamada dá
  `ToolFailure("upstream", "run_command needs a POSIX system (Linux, macOS): …")`, porque os grupos
  de processos são POSIX e o `select` do Windows não aceita pipes. O CI não tem Windows; é o
  critério da D64 (C07.8).
- **Testes (`test_shell.py`, `TestNothingIsLeftBehind`):**
  - `test_no_process_fd_or_thread_outlives_the_call`, para `yes W &`, `sleep N &`, `tail -f F &`,
    `tail -f F` (no prazo) e `sleep N >/dev/null 2>&1 &`. Mede os fds (`/dev/fd`) e as threads
    (`threading.active_count()`) antes e depois. Mede os processos com `pgrep -f` e um marcador
    único (o fixture faz `pkill -9` no fim), e o tempo (menos de 3 s com `timeout=1`). Com o código
    anterior falharam quatro dos cinco casos (`(14, 3) != (12, 1)` no `yes`);
  - `test_a_command_still_running_when_the_program_exits_is_stopped`: um Python filho corre o
    comando numa thread daemon e sai; o `sleep` tem de desaparecer. Falhava antes do `atexit`;
  - a resposta no prazo, o código de saída com um processo em fundo, o `wait`, o `cat` sem entrada e
    o Windows.
- **Medido com os scripts da revisão:**
  - `bg_yes.py`: 0,00 s de CPU do anfitrião nos 3 s seguintes (eram 2,7 s);
  - cinco `sleep 6 &`: 0 fds e 0 threads deixados (eram 6 e 6);
  - `echo hi` leva 3,6 ms; `yes | head -c 30000000` leva 13 ms.

##### 10. Baixa: as respostas do `run_command` (vai com a 1)

- **A shell acabou, mas um filho segura os pipes:** a resposta dá o que o comando escreveu, o
  `[exit code: N]` e
  `[the command ended and left processes in the background holding its output: they were stopped; end the command with wait to let them finish]`.
  Chega 0,5 s depois, não no prazo. Antes era "timed out", sem saída e sem código.
- **No prazo:** fica o que o comando escreveu (stdout e `[stderr]`, pelo `fit`), e depois
  `[timed out after Ns: the command and every process it started were stopped; give it a longer timeout (at most 600) or make it do less]`.
- **`max_output`:** a docstring diz agora que conta os caracteres do comando e que as notas do
  corte vêm por cima. `max_output=200` dá 342 caracteres com a nota.
- **Testes:** `test_the_timeout_keeps_what_was_printed_and_says_what_to_do`,
  `test_a_command_that_leaves_a_process_in_the_background_ends_with_its_exit_code` e
  `test_wait_lets_the_background_finish`. O antigo
  `test_a_process_left_holding_the_output_ends_at_the_timeout` afirmava a resposta errada e saiu.

##### 2. Alta: as páginas do `csv_read` têm um limite de caracteres

- **Mudança (`_json.py`):**
  - uma página tem até `max_rows` linhas e no máximo `_PAGE_CHARS` = 100 000 caracteres, como a
    janela do `read_file`, com pelo menos uma linha, seja qual for o seu tamanho. As linhas
    juntam-se primeiro pelo tamanho sem preenchimento, que é um limite inferior (`_rows`);
    depois, já com as larguras, corta-se no limite (`_fitted`);
  - o preenchimento de cada coluna pára nos `_PAD_CHARS` = 40 caracteres. Uma célula mais longa
    mostra-se inteira, desalinhada; nenhum campo é cortado (D39);
  - o rodapé dá o `offset` seguinte, como antes.
- **Testes (`test_json.py`, `TestCsvPageBounds`):** o CSV da revisão tem 9999 linhas e uma linha
  com um campo de 130 000 caracteres.
  - a primeira página pára antes dessa linha;
  - essa linha vem sozinha;
  - as linhas curtas a seguir enchem cerca de 100 000 caracteres, nenhuma com mais de 60;
  - seguir os rodapés lê cada linha uma vez;
  - o preenchimento pára nos 40 caracteres.
- **Medido:** o `wide.csv` da revisão com `max_rows=10000` dá 130 309 caracteres e 55 MB de RSS
  (eram 1,3 GB de texto e 4 GB de RSS).

##### 3. Alta: o `regex_search` não congela o processo

- **Escolha:** o match corre num processo Python filho (`sys.executable -I -S -c _WORKER`), morto
  ao fim de `_MATCH_S` = 5 s por `subprocess.run(timeout=…)`. O padrão e o texto vão em JSON pelo
  stdin, como dados; o filho devolve o total e os spans da página. O processo pai espera sem o
  GIL. A guarda estática que já existia (exponencial, retro-referências, tamanhos) fica e recusa
  logo. O comentário no topo do `_text.py` justifica a escolha: uma verificação estática que
  recuse `a*a*b` também teria de recusar `\d+-\d+` e `\w+@\w+\.\w+`. A alternativa seria uma
  análise de sobreposição de conjuntos de caracteres (classes Unicode, flags, lookarounds) que não
  consigo provar correcta.
- **Pior caso permitido:** 5 s de CPU no filho, com o anfitrião sempre a correr. Medido com o
  `regex_driver.py` da revisão:
  - `a*a*b` com n=20 000: recusado aos 5,01 s (eram cerca de 18 minutos com o GIL preso);
  - `\w*\w*\w*x` com n=600: recusado aos 5,00 s;
  - `a*a*a*b` com n=600: 4,77 s, dentro do limite;
  - o pior padrão de um só quantificador que medi, `[a-z]+\d` sobre `"a" * 20000`: 1,72 s.
  - custo de uma chamada: 18 a 25 ms, o arranque do filho.
- **Programa congelado ou sem `sys.executable`:** `upstream`, sem tentar arrancar o próprio
  binário da aplicação.
- **Testes (`test_text.py`, `TestPolynomialBacktracking`):**
  - `a*a*b` sobre 20 000 caracteres, num filho com prazo de 30 s, é recusado com
    `validation_error` antes de `_MATCH_S` + 3 s. Antes, o pytest-timeout teve de o matar;
  - com `_MATCH_S=0.5`, uma thread do anfitrião continua a contar durante o match. Antes falhava;
  - o filho encontra o mesmo que o `re` local: grupos `None`, Unicode, padrão vazio e lookahead;
  - o caso do programa congelado.
- **Conflito para o coordenador:**
  `tests/toolkit/test_tool_invariants.py::test_the_declared_capability_is_what_the_code_reaches[regex_search]`
  falha. O detector AST lê o `import subprocess` como capacidade "shell". O ficheiro está fora do
  meu âmbito e não lhe toquei.
  - A tool continua `compute`: não corre comandos nem código do utilizador, só o programa fixo
    `_WORKER`, com o padrão e o texto como dados;
  - a governança declarada não muda;
  - proposta: uma excepção nomeada no invariante, com o porquê:

    ```python
    # Tools that run a fixed program of their own in a child process, and why: the arguments
    # are data, so the child reaches nothing the declared capability does not.
    _OWN_CHILD = {"regex_search": "matches in a child Python (_text._WORKER), killable at 5 s"}
    ...
    assert declared == reached or (declared, reached) == ("python", "compute") or (
        name in _OWN_CHILD and (declared, reached) == ("compute", "shell")
    )
    ```

  - Se o dono preferir que nenhuma tool `compute` arranque processos, a alternativa é a análise
    estática, mais cara e com o risco que digo acima.

##### 4. Média: o `search_files` não guarda as ocorrências antes do `offset`

- **Mudança:** `_hits` dá pares `(caminho, _Line)` sem formatar. As primeiras `offset` contam-se
  com `itertools.islice` e não se guardam; só a página (`max_results + 1`) se formata.
- **Teste:** `test_the_hits_before_the_offset_are_not_kept`, com `tracemalloc`, offset 49 990 em
  50 000 ocorrências: pico abaixo de 1 MB. Antes era 4,2 MB.
- **Medido como na revisão:** offset 1 999 990 num ficheiro de 151 MB dá 2,7 s e 55 MB de RSS (eram
  395 MB).

##### 5. Média: cada caminho do `search_files` é aceite pelo `read_file`

- **Mudança:** o caminho mostrado é `Path(directory) / relativo`, com o `directory` tal como foi
  dado. `"."` dá `notes.txt`, `"sub"` dá `sub/notes.txt`, e um caminho absoluto dá um caminho
  absoluto. O `read_file` resolve-o como o `search_files` resolveu a pasta: em relação ao cwd, com
  `~` expandido. O mesmo vale para o ficheiro onde a pesquisa parou e para os que não se leram.
- **Teste:** `TestSearchPaths`. Com o cwd em `tmp_path`, `search_files("sub", …)` dá
  `sub/notes.txt:3:10: …`, e `read_file("sub/notes.txt", offset=10)` lê dali. Antes dava
  `notes.txt`, que é outro ficheiro.
- Os testes antigos passam a tirar o prefixo da pasta com `_hits(result, under)`, que verifica que
  ele lá está. O caso do contrato usa `"."` e não muda.

##### 6. Média: limites de varrimento

- **Binários:** um ficheiro com um NUL nos primeiros 8192 bytes (`raw.peek`) salta-se, como no
  grep e no git. Teste: `test_a_file_with_a_nul_in_its_first_chunk_is_skipped_as_binary`. Medido:
  um ficheiro esparso de 2 GB com o nome `zeros.log` leva 0,000 s (eram 10 s).
- **Orçamento da pesquisa:** `_SCAN_CHARS` = 500 milhões de caracteres em todos os ficheiros, cerca
  de 4,2 s medidos (118 milhões de caracteres por segundo).
  - Se o orçamento acaba antes de a página encher, a resposta mostra o que se encontrou e acaba em
    `[results 1-2 | stopped at the scan limit (500000000 characters in 2 files) in <caminho>; search a narrower directory to see the rest]`.
    Sem `next`, porque um `offset` repetiria o mesmo varrimento, e sem total;
  - com zero ocorrências: `No matches for 'p' in d up to the scan limit (…), which stopped in <caminho>; search a narrower directory to see the rest`;
  - o rodapé faz-se à mão, porque a `Window` diria `end` sem total nem `next`, e o `_window.py`
    não é meu;
  - testes: `test_the_search_stops_at_its_budget_and_says_where_and_how_to_narrow`,
    `test_no_match_within_the_budget_says_the_search_stopped` e
    `test_a_full_page_within_the_budget_reads_on_as_usual`.
- **Contagem do `csv_read`:** depois da página, as linhas contam-se até `_COUNT_CHARS` = 50
  milhões de caracteres. O parser lê cerca de 80 MB/s com o contador, logo são no máximo cerca de
  0,6 s.
  - Para lá disso, o cabeçalho diz
    `(at least N rows, counted up to 50000000 characters past this page)`, e o rodapé dá o `next`
    sem total;
  - testes: `test_the_count_stops_past_its_cap_and_says_so`, com o limite a 1000, e
    `test_the_count_still_reaches_the_end_of_a_file_within_the_cap`;
  - medido: no `big.csv` de 83 MB, uma página leva 0,67 s, que não cresce com o ficheiro (eram
    0,93 s e cresciam);
  - uma página funda continua a custar O(offset).

##### 7. Baixa: `python_repl` com uma string vazia

- **Mudança:** decide-se pela existência do valor (`is not None`), não pelo `Head.total`.
  `'"a".strip("a")'` dá `""` outra vez; `print("x")` seguido de `""` dá `"x\n\n"`, como o código
  antigo.
- **Teste:** `test_a_value_shows_when_there_is_one_even_empty`, com cinco casos.

##### 8. Baixa: `unit_convert(10**400, …)`

- **Mudança:** o `math.isfinite` corre dentro de um `try`, e o `OverflowError` passa a
  `validation_error` ("the value is beyond a float's range; convert a smaller value").
- **Teste:** `test_an_integer_beyond_a_float_is_a_validation_error`, também `10**20`, que converte.

##### 9. Baixa: `once_kept` mais estrito

- **Mudança:** o `ONCE` passa a `dict[str, Once]`, com `Once(why, narrow)`; `narrow` é o que o
  rodapé da tool tem de dizer. O `once_kept` rejeita um `onward` que comece por `next`, mesmo mal
  formado, e exige essas palavras. Exige também `last < total`. O `_DEAD_ENDS` saiu, porque a
  frase exigida o cobre.
- **Testes (`TestTheChecks`):** as tools de teste `misnamed` (`next: run it again printing less`)
  e `reworded` (`the rest cannot be read`), e uma frase errada (`| grep`). Verifiquei que o
  `once_kept` antigo aceitava as duas (`True | True`).

##### 11. Baixa: pastas que não se lêem

- **`search_files`:**
  - uma raiz que não se lê dá `upstream`, pelo `path_failure` ("permission denied to search …");
  - uma subpasta ou um ficheiro que não se lê nomeia-se: até três, e "and N more", no cabeçalho ou
    na resposta de zero ("; this process cannot read …"). O `onerror` do `os.walk` e o `OSError`
    de cada ficheiro alimentam a lista.
- **`list_directory`:** um `os.scandir` à pasta antes do glob, para que uma pasta que não se lê
  dê `upstream` em vez de "No entries".
- **Testes:** `TestUnreadableFolders`, com `chmod 0`, saltado como root e reposto no fim.

##### 12. Baixa: o custo O(N²) do `read_file`

- Não há correcção barata. Um offset em caracteres não tem posição em bytes no UTF-8 sem
  descodificar o que vem antes, e o caso caro, bytes que não são UTF-8, é precisamente o que não
  se pode saltar.
- Ficou documentado na docstring do `_skip` (3 GB/s para texto, 0,12 GB/s para o resto, cerca de
  80 s a 10 GB de profundidade, O(N²) para percorrer tudo). A docstring da tool diz agora: "Reaching
  an offset reads the file up to it, so a deep one in a huge file takes longer."

##### 13. Estilo

- **Docstrings:** saíram os "Defaults to …" que repetiam a assinatura no `date_add` e no
  `list_directory`.
- **Passos seguintes nas mensagens:**
  - `_out_of_range` acaba em "; use a smaller shift or a date further from the calendar's ends.";
  - o `upstream` genérico do `path_failure` acaba em "; try again, or pick another path.";
  - o prazo do `run_command` está no ponto 10.
- **Testes:** os de `TestRange` em `test_datetime.py` e `test_another_os_error_is_upstream`.

##### Verificação

- `uv run pytest -q` nos testes dos módulos (`test_shell`, `test_filesystem`, `test_json`,
  `test_text`, `test_python`, `test_python_nodes`, `test_math`, `test_datetime`, `test_output`,
  `test_window`, `tests/test_python_eval.py`), mais `test_tool_contract.py`,
  `test_tool_invariants.py`, `test_tools_exports.py`, `tests/test_tools_*`, `test_tool_failures`,
  `test_tool_limits`, `test_quality_budget`, `test_architecture` e `tests/nanope`: **1725 passed,
  59 skipped, 1 failed**. A falha é o detector de capacidade do `regex_search` (ponto 3, para o
  coordenador).
- `uv run pyright src`: 0 erros. `ruff check` e `ruff format` estão limpos nos meus ficheiros; no
  `contract_cases.py` só acertei as linhas em branco do meu bloco, sem formatar o ficheiro.
- **Processos:** nenhum ficou. Verifiquei com `pgrep` depois de cada corrida; o filho órfão da
  primeira corrida falhada do teste de 20 000 caracteres foi morto à mão.

##### Mudanças às linhas propostas para o CHANGELOG

Nas Upgrade notes (Tools), acrescentar:

```markdown
  - `search_files` gives each path under `directory` as you passed it (`sub/notes.txt`, or an
    absolute path), which `read_file` reads as it is; it was relative to `directory`.
  - `run_command` stops every process the command starts when the call returns (one left in the
    background half a second after the command ends: end the command with `wait`), gives the
    command no input, and refuses to run on Windows (a typed `upstream` failure).
```

No `### Changed`, substituir o item do `run_command` por:

```markdown
- **Breaking:** `run_command` and `python_repl` show the start of an output that does not fit,
  with its size and how to narrow the command or print less: the output is not kept, so no call
  reads on. `run_command` keeps stderr and the exit code under a long stdout (a cut at
  `max_output` dropped them), runs the command in a process group of its own, which it kills when
  the call returns, so no process or reader is left behind (a process left in the background is
  stopped half a second after the command ends, with a note that says to end the command with
  `wait`), keeps what was printed when the command times out and says what to do, gives the
  command no input (`/dev/null`), and refuses to run on Windows. `python_repl` shows at most
  20,000 characters, and a long print never pushes out the last value or the error.
```

No item "local tools keep the tools contract", acrescentar:

```markdown
  A `csv_read` page holds at most 100,000 characters (a longer row comes alone) and pads a column
  to 40 at most; its rows are counted up to 50 million characters past the page, then the
  heading says "at least". `search_files` skips a file with a NUL in its first 8 KiB, reads at
  most 500 million characters a call (then it says where it stopped and asks for a narrower
  `directory`), and names the folders and files it cannot read. A folder that `search_files` or
  `list_directory` cannot read is an `upstream` failure, not an empty answer.
  `regex_search` matches in a child Python process given 5 seconds: a pattern that backtracks
  polynomially (`a*a*b`) is refused when they run out, and the program never freezes.
```

No `### Fixed`, acrescentar:

```markdown
- `run_command` left a process the command started in the background running, with two threads
  reading its output, after the call returned.
- `regex_search` froze the whole process (it held the GIL) on a pattern that backtracks
  polynomially, such as `a*a*b` on 20,000 characters (about 18 minutes).
- `csv_read` built a page as wide as its widest cell on every row (1.3 GB for a 200 KB file).
- `python_repl` showed `None` for an expression whose value is an empty string.
- `unit_convert` failed untyped on an integer beyond a float's range.
```

##### Para o coordenador: documentação partilhada (não lhe toquei)

- **`docs/tools-catalog.md`:**
  - linha 35: o `regex_search` corre num processo filho com 5 s;
  - linha 345: os caminhos do `search_files`, o binário pelo NUL, o orçamento de 500 milhões e as
    pastas que não se lêem;
  - linha 346: os limites do `csv_read`;
  - linha 350: o `run_command`, que mata o grupo, não dá entrada e não corre em Windows. A frase
    "the command runs on until it ends or its own timeout stops it" continua certa.
- **`docs/safety.md`, linha 54:** continua verdadeira (as recusas mantêm-se) mas está incompleta.
  Pode ganhar "and runs each match in a child process it kills after 5 seconds". A governança não
  mudou, por isso não lhe toquei.

##### Achado fora do âmbito

- O `python_repl` expõe `re.findall`, `re.search`, `re.match`, `re.sub` e `re.split`, que correm
  no processo, sem a guarda nem o filho do `regex_search`. Por exemplo,
  `python_repl('len(re.findall("a*a*b", "a" * 20000))')` congela o processo como o achado 3. É a
  mesma causa, e corrige-se passando essas funções pelo mesmo filho, ou tirando-as.

### T09b

- **Dono:** Claude (agente T09b), 2026-10-08 · **Estado:** review (na worktree
  `.claude/worktrees/agent-aa9a3ba0434bb07d0`, por commitar)
- **Módulos:** `_web` (`http_get`, `scrape_text`), `_youtube` (3), `_internet_archive` (2),
  `_open_library` (3): 10 tools. As 10 saem da lista de dívida.

#### Nota de desenho

##### `_web` (`http_get`, `scrape_text`)

- **Continuação:** `offset` e `find` nas duas, pela janela (`text_window`, `find_window`). Cada
  chamada busca a página outra vez; o rodapé dá a chamada seguinte, e segui-lo reconstrói o texto
  inteiro (teste).
- **`http_get` lê só o que a janela precisa:** `(offset + max_chars) × 4` bytes (UTF-8 tem no
  máximo 4 bytes por carácter), até 10 MB (o limite por omissão da porta). Se a leitura não
  chegou ao fim da página, o total não se sabe: o rodapé sai sem "of N" e com o `offset`
  seguinte. Uma leitura cortada pode acabar a meio de um carácter: esse carácter de substituição
  final sai, e a janela seguinte lê-o inteiro. Com `find`, lê até 10 MB. Uma página maior diz-o
  numa linha por cima: "(URL goes on past its first 10000000 bytes, all http_get reads)".
- **`scrape_text`:** continua a ler 2 MB de HTML (o tecto actual, D39); o texto visível vai à
  janela. Um HTML maior diz-o por cima.
- **Limites:** `max_chars` `Range(1, 100_000)` (os mesmos de hoje), `offset` `Range(0)`. Sem
  `_clamp`.
- **Falhas:** as da porta (`fetch_page`); continuam `SOURCELESS` (qualquer URL).
- **Saída:** sem cabeçalho quando a página cabe (o texto é a página): o teste de governança que
  compara `result.value == "Hello World"` fica igual. Uma resposta vazia diz-o.
- **Governança:** a mesma (`network`, `high`, `requires_approval`); cada continuação volta a
  pedir aprovação (teste). `docs/safety.md` não muda.
- **Parâmetros novos no fim** (`url, max_chars, offset, find`): quem chama por posição não parte.

##### `_youtube` (3 tools)

- **Fonte:** `youtube-transcript-api`, não a porta. Os erros vêm das classes de excepção da
  biblioteca (`_errors.py`, lido na versão instalada 1.2.4 e em
  https://github.com/jdepoix/youtube-transcript-api/blob/master/youtube_transcript_api/_errors.py).
  O mapeamento tipado (T01) fica; o módulo continua `SOURCELESS`, com o motivo actualizado.
- **Costura:** `_load_youtube_transcript_api()`, que os testes já usavam. Passa a ser a costura
  do contrato: o `Case` ganha `patches` (ver "Mudanças em ficheiros partilhados"), e
  `tests/toolkit/youtube_fakes.py` tem a biblioteca falsa (`library()`, `raising(nome)`),
  partilhada por `test_youtube.py` e `contract_cases.py`.
- **`youtube_transcript`:** `offset` (no fim da assinatura) e a janela sobre o texto formatado
  (qualquer `output_format`). O "Increase max_chars", que aparecia mesmo no tecto de 50 000, sai:
  o rodapé dá `next: offset=N`. Cabeçalho: vídeo, código e nome da língua, `manual`/`generated`,
  e "translated from xx" quando traduzido. `max_chars` `Range(1, 50_000)`.
- **`youtube_transcript_search`:** é o `find` das transcrições (com tempos), por isso não há
  `find` no `youtube_transcript` (D41). Pagina as ocorrências com `offset`, com total:
  `[matches 21-40 of 50 for "chorus" | next: offset=40]`. `max_results` `Range(1, 20)`,
  `context_segments` `Range(0, 3)`.
- **`youtube_transcript_languages`:** todas as línguas de tradução (cortava em 8 com "+N more"),
  uma por linha, uma vez por vídeo (o YouTube dá a mesma lista a cada transcrição que traduz);
  janela de 4000 caracteres com `offset`.
- **`not_found`:** `VideoUnavailable`/`InvalidVideoId`/`TranscriptsDisabled`/`NoTranscriptFound`
  (já da T01), agora provado pelo contrato nas três.

##### `_internet_archive` (2 tools)

- **Documentação:** pesquisa avançada https://archive.org/advancedsearch.php (formulário; `rows`,
  `page`, `output=json`) e https://archive.org/help/aboutsearch.htm ("We limit the number of sorted
  paged results returnable to 10,000"); metadados https://archive.org/developers/md-read.html
  (`error` legível; `extended_err=1` dá `errcode`: 101 pendente, 102 indisponível, 104 apagado,
  105 nó primário não encontrado; item desconhecido: "returns an empty array").
- **Duas `Api`, cada uma com o seu leitor:** `_SEARCH` (`/advancedsearch.php`) e `_METADATA`
  (`/metadata`).
  - Pesquisa: o `error` num 200 (visto a 2026-09-29) vira "the Internet Archive's search could not
    run the query: …; check its syntax (…)"; num 5xx, só as palavras da fonte (o estado dá o
    `retryable`).
  - Metadados: pede `extended_err=1`; 104 é `not_found` ("deleted"), 101/102/105 `upstream`
    retryable ("try again later"), o resto as palavras da fonte.
- **`internet_archive_search`:** mantém `page` (a fonte pagina por página). Numerada a partir do
  `response.start`, com o total `response.numFound` e `next: page=N+1`. Recusa antes de pedir uma
  página que começa depois do resultado 10 000 (`validation_error`, com "narrow the query"); na
  última página alcançável o rodapé diz "the rest cannot be read here" e o cabeçalho explica. Zero
  resultados: "No Internet Archive items match 'q' (mediatype: …)". Listas de cada resultado:
  as primeiras (3 criadores, 3 colecções, 5 assuntos) com "(+N more)"; o item mostra-as todas.
  `max_results` `Range(1, 20)`, `page` `Range(1, 10_000)`.
- **`internet_archive_item`:** o registo inteiro (todos os ficheiros, descrição sem corte), com
  `find`, `offset` e `max_chars` `Range(500, 20_000)` (o modelo da T05). Os ficheiros dizem onde se
  descarregam (`https://archive.org/download/{id}/<name>`). Usa `item_size` e `files_count` do
  topo da resposta.
- **Datas:** a pesquisa dá `1888-01-01T00:00:00Z`; sai `1888-01-01`.
- Sai o `_truncate` e o `_format_items` (C901 12).

##### `_open_library` (3 tools)

- **Documentação:** https://openlibrary.org/dev/docs/api/search (`q` é "the solr query";
  `limit`/`offset`; `fields`; resposta com `numFound`/`num_found` e `start`),
  https://openlibrary.org/dev/docs/api/books (works, editions, `/isbn/{isbn}.json` redirecciona
  para a edição), https://openlibrary.org/dev/docs/api/authors ("we do not have a way to get the
  full data for multiple authors in one request"; a pesquisa de autores com
  `key:(/authors/OL…A OR /authors/OL…A)` dá o nome; o `key` vem sem prefixo) e
  https://openlibrary.org/developers/api (1 pedido por segundo sem email; sem documentação de
  erros).
- **Ritmo:** `min_interval_s=1.0` na `Api` (não mandamos email nem outro User-Agent).
- **`open_library_search`:** mantém `start`. Pede `fields` explícitos. Numerada pelo `start` da
  fonte, com total e `next: start=N`. Autores pelo nome com a chave ao lado
  (`J.R.R. Tolkien (OL26320A)`), que o `query="author_key:OL26320A"` aceita (ponto 6; a docstring
  di-lo). ISBN e editoras: os 3 primeiros com "(+N more)". Zero resultados com os filtros.
  `max_results` `Range(1, 20)`, `start` `Range(0)`.
- **Nomes dos autores nos registos (ponto 5):** um pedido a `/search/authors.json` com
  `key:(… OR …)`, `fields=key,name`, `limit`, para os primeiros 50 autores (limite do pedido; os
  outros ficam pela chave). Um autor que a pesquisa não conhece: `OL2A (no name on Open Library)`.
  Os nomes não vêm na resposta da obra nem da edição; a alternativa (um pedido por autor) seria
  ilimitada. Uma falha desse pedido é a falha da tool (tipada, retryable), não texto.
- **`open_library_work` / `open_library_isbn`:** o registo inteiro (descrição, assuntos, ligações,
  ISBN-13/10, editoras), janela com `offset` e `max_chars` `Range(500, 20_000)`. A edição diz a
  obra (`Work: OL45804W (open_library_work reads it)`). Datas em ISO 8601 (`October 1, 1988` →
  `1988-10-01`, `September 1970` → `1970-09`; o que não é data fica como está).
- **Fusões e apagados** (`6a34668`): ficam.
- Sai o `_truncate` e o `_format_books` (C901 20, PLR0912 19).

#### Fusões e nomes tirados

- **Nenhuma fusão, nenhum nome de tool tirado ou mudado.** O `nanope` importa `http_get` e
  `scrape_text` pelo nome: continuam.
- **Parâmetros novos** (nenhum tirado nem renomeado): `offset` e `find` no `http_get` e no
  `scrape_text` (no fim); `offset` nas três `youtube_*` (no fim); `find`, `offset`, `max_chars`
  no `internet_archive_item`; `offset`, `max_chars` no `open_library_work` e no
  `open_library_isbn`.
- **Quebras visíveis:** as 10 devolvem `ToolResult` (como a família wiki); chamadas cruas lêem
  `.value`. Valores fora dos limites são `validation_error` em vez de ajustados. Textos de saída
  novos (cabeçalhos, rodapés, nomes dos autores).

#### Mudanças em ficheiros partilhados

- `tests/toolkit/test_tool_contract.py`: o `_serve` põe no lugar os `case.patches` (duas linhas),
  a docstring do módulo di-lo, e um teste novo em `TestTheChecks`
  (`test_a_cases_patches_stand_in_for_a_source_reached_without_the_door`, com a tool de teste
  `through_a_library`). É a costura própria que a `contract_cases.py` prometia à T09 para as
  `youtube_*`.
- `tests/toolkit/contract_cases.py`: campo novo `Case.patches` (`Mapping[str, object]`, vazio por
  omissão), a linha do YouTube na docstring e o motivo do `SOURCELESS["_youtube"]`; o resto são as
  minhas entradas (janela 10, `not_found` 3, zero 2).
- `tests/toolkit/youtube_fakes.py`: novo.
- `tests/quality_baseline.json`: só desce (3 entradas: `_internet_archive.py:_format_items:C901`,
  `_open_library.py:_format_books:C901` e `:PLR0912`).
- `core/`, `_http.py`, `_window.py`: sem mudanças.
- `docs/tools-catalog.md`: só as minhas secções. A frase de abertura do catálogo ("the other tools
  still move a numeric argument outside a limit … without a word") é partilhada e deixei-a; o
  coordenador deve actualizá-la ao juntar as fichas.

#### Prova

Testes novos, por item da ficha (testes antes: 76 nos quatro ficheiros e 143 no contrato; depois
143 e 144: +68). Cada um falhou antes pela razão certa (os de IA e Open Library corridos contra o
módulo antigo: 17 e 24 falhas).

- **Dizem o tamanho mas não deixam continuar (web e YouTube):**
  - `test_web.py::TestHttpGet::test_following_the_footers_rebuilds_the_whole_text` (o item
    "seguir os rodapés do `http_get` reconstrói o conteúdo inteiro", com caracteres de 1 a 4
    bytes), `test_the_smallest_window_still_reads_on_to_the_end` (`max_chars=1`, emoji; e uma
    página que começa por espaços), `test_a_cut_page_names_the_next_offset_and_reads_no_more_than_it_needs`,
    `test_a_page_read_whole_gives_its_length`, `test_find_returns_the_passages_around_a_term`,
    `test_a_page_longer_than_what_http_get_reads_says_so`;
  - `TestScrapeText::test_following_the_footers_rebuilds_the_visible_text`,
    `test_find_returns_the_passages_around_a_term`,
    `test_html_longer_than_what_scrape_text_reads_says_so`;
  - `test_youtube.py::test_following_the_footers_rebuilds_the_whole_transcript`,
    `test_every_format_reads_window_by_window` (json, srt, vtt),
    `test_matches_page_by_offset_with_their_total`, `test_an_offset_past_the_last_match_says_so`,
    `test_every_translation_is_listed_window_by_window`.
- **"Increase max_chars" no tecto:**
  `test_a_cut_at_the_ceiling_names_the_next_offset_not_a_larger_max_chars`.
- **Listas cortadas em silêncio e descrições a 1000:**
  `test_internet_archive.py::test_no_list_or_description_is_cut_short`,
  `test_long_lists_in_a_result_say_how_many_more_the_item_has`,
  `test_a_long_item_reads_window_by_window`; `test_open_library.py::
  test_no_description_or_list_is_cut_short_and_a_long_work_reads_on`,
  `test_a_long_work_names_the_next_offset`, `test_long_lists_in_a_result_say_how_many_more`.
- **Paginação sem total:** `test_returns_items_numbered_with_the_total_and_the_next_page`,
  `test_the_search_stops_at_the_depth_the_archive_pages_and_says_so`,
  `test_a_page_past_that_depth_is_refused_before_asking`;
  `test_returns_results_numbered_with_the_total_and_the_next_start`,
  `test_the_next_page_is_numbered_on`.
- **Autores como chaves:** `test_returns_the_work_with_its_authors_names`,
  `test_every_author_is_named_in_one_request`,
  `test_names_are_asked_for_the_first_fifty_authors_at_most`,
  `test_a_work_without_authors_asks_for_no_names`, `test_returns_the_edition`,
  `test_a_failure_of_the_names_request_is_the_tools_failure`.
- **Saída (ponto 5):** `test_dates_come_in_iso_8601` (7 casos); a data do IA no teste da pesquisa.
- **Falhas tipadas:** `test_an_extended_error_code_says_what_happened` (104, 102, 101, 400),
  `test_a_query_the_search_cannot_run_says_why_and_what_to_check`,
  `test_a_server_error_is_retryable_and_says_only_what_the_search_said`,
  `test_a_429_is_rate_limited`, `test_a_video_that_is_not_there_is_not_found_for_every_tool`.
- **Limites pelo executor:** `test_max_chars_is_refused_outside_its_limits_through_the_executor`
  (web, YouTube, item), `test_limits_are_refused_through_the_executor` (as duas pesquisas). Sem o
  `_clamp`, uma chamada crua com `context_segments` negativo partia o `_passage` na minha primeira
  versão: `test_a_raw_call_with_a_negative_context_shows_the_match_alone` (achado da revisão).
- **Governança igual:** `test_tools_exports.py` intocado e verde; os dois testes originais de
  `TestHttpGetGovernance` intocados; novos `test_the_window_does_not_change_the_web_tools_governance`
  e `test_reading_on_is_approved_call_by_call`.
- **Contrato:** as 10 tools cumprem os pontos que deviam e saem da dívida;
  `test_a_cases_patches_stand_in_for_a_source_reached_without_the_door`.
- **Testes substituídos** (afirmavam o comportamento antigo): no `test_web.py`, os dois
  `test_truncation`, `test_a_negative_or_huge_max_chars_is_clamped` e
  `test_reads_no_more_of_the_body_than_it_can_return` (passou a cobrir o `offset`); no
  `test_open_library.py`, o `max_results=99` ajustado a 20 e o `start=-1` cru (agora recusados
  pelo executor) e as asserções das chaves `/authors/…`; no `test_internet_archive.py`, o
  `page=0` cru (agora pelo executor). Os do YouTube passaram para as falsas partilhadas, com as
  mesmas asserções de erro.

#### Ao vivo, pelo dono

APIs gratuitas e sem chave. A Open Library pede 1 pedido por segundo: a porta espaça-os.

```bash
uv run python -c "from ai_arch_toolkit.toolkit.tools import internet_archive_search as s; print(s('alice in wonderland', mediatype='texts', max_results=3).value)"
```

```bash
uv run python -c "from ai_arch_toolkit.toolkit.tools import internet_archive_search as s; print(s('alice in wonderland', mediatype='texts', max_results=3, page=2).value)"
```

```bash
uv run python -c "from ai_arch_toolkit.toolkit.tools import internet_archive_item as i; print(i('IDENTIFIER_DO_PRIMEIRO_COMANDO', max_chars=3000).value)"
```

```bash
uv run python -c "from ai_arch_toolkit.toolkit.tools import internet_archive_item as i; i('zzqq-no-such-item-zzqq')"
```

```bash
uv run python -c "from ai_arch_toolkit.toolkit.tools import internet_archive_search as s; s('title:(')"
```

```bash
uv run python -c "from ai_arch_toolkit.toolkit.tools import open_library_search as s; print(s('the lord of the rings', max_results=3).value)"
```

```bash
uv run python -c "from ai_arch_toolkit.toolkit.tools import open_library_search as s; print(s('author_key:OL26320A', max_results=3).value)"
```

```bash
uv run python -c "from ai_arch_toolkit.toolkit.tools import open_library_work as w; print(w('OL27448W').value)"
```

```bash
uv run python -c "from ai_arch_toolkit.toolkit.tools import open_library_isbn as i; print(i('9780140328721').value)"
```

```bash
uv run python -c "from ai_arch_toolkit.toolkit.tools import youtube_transcript as t; print(t('dQw4w9WgXcQ', max_chars=500).value)"
```

```bash
uv run python -c "from ai_arch_toolkit.toolkit.tools import youtube_transcript_languages as l; print(l('dQw4w9WgXcQ').value)"
```

```bash
uv run python -c "from ai_arch_toolkit.toolkit.tools.dangerous import http_get; r = http_get('https://www.rfc-editor.org/rfc/rfc9110.txt', max_chars=2000); print(r.value[-300:]); print(r.metadata['window'])"
```

```bash
uv run python -c "from ai_arch_toolkit.toolkit.tools.dangerous import scrape_text; print(scrape_text('https://www.python.org/', find='download', max_chars=1500).value)"
```

O que confirmar:

- IA: o total e o `next: page=2`; a segunda página numerada a partir de 4; o item inteiro, com
  todos os ficheiros; o item desconhecido dá `not_found` (a documentação diz "empty array"; o
  código espera `{}`: ver achados); o `title:(` dá a falha com "check its syntax"; o
  `extended_err=1` não muda a resposta de um item que existe.
- Open Library: o total e o `next: start=3`; o `author_key:OL26320A` devolve livros do Tolkien
  (prova o ponto 6); a obra e a edição com o nome dos autores (prova o `key:(…)` da pesquisa de
  autores, e que `fields`/`limit` não a partem); a data `1988-10-01`.
- YouTube: o cabeçalho com língua e tipo, e o rodapé com `next: offset`; as línguas de tradução
  todas.
- Web: `total` `None` enquanto a página não foi lida até ao fim, e o `offset` seguinte.

#### Bloqueios e achados

- **Sem bloqueios.**
- **IA, item desconhecido:** https://archive.org/developers/md-read.html diz "returns an empty
  array"; o código (e o teste, de antes) tratam `{}`. Se ao vivo vier `[]`, o `get_json` falha
  com "expected a JSON object, got an array" (`upstream`) em vez de `not_found`. A correcção
  precisaria de um pedido na porta que aceite objecto ou lista; não a fiz sem a prova ao vivo.
- **IA, `extended_err=1`:** documentado, não verificado ao vivo.
- **IA, descrições com HTML:** muitas descrições trazem `<p>`, `<br>`, ligações; saem tal como
  vêm. Não está na ficha; candidato a usar um conversor de HTML partilhado.
- **Open Library, pesquisa de autores:** `fields` e `limit` estão documentados para a pesquisa de
  obras, não para a de autores; se forem ignorados, a resposta só fica maior. O limite por omissão
  dessa pesquisa não está documentado; peço `limit` igual ao número de chaves (até 50).
- **Open Library, `author_key:` no `q`:** a documentação diz só que o `q` é "the solr query" e
  mostra o campo `author_key`; confirmar ao vivo.
- **Open Library, ritmo:** com 1 pedido por segundo, uma obra custa pelo menos 1 s (dois
  pedidos), mais 1 s por fusão seguida.
- **`_record` da Open Library:** "more than 3 redirects between Open Library records" não diz o
  passo seguinte (antigo, fora do âmbito desta ficha).
- **`docs/tools-catalog.md`:** a frase de abertura, partilhada, ainda diz que as outras tools
  ajustam os limites em silêncio; actualiza-se ao juntar as fichas.
- **Divisão proposta em commits:** (1) `test(tools): a library stand-in for the contract
  (Case.patches)` com `youtube_fakes.py`, `contract_cases.py` (campo) e `test_tool_contract.py`;
  (2) `feat(tools): the web tools read window by window (T09)`; (3) `feat(tools): YouTube
  transcripts window by window (T09)`; (4) `feat(tools): Internet Archive and Open Library
  records whole, searches with totals, authors by name (T09)`; cada um com os seus testes, casos,
  linhas da dívida e secção do catálogo.

#### Correcções da revisão

Revisão: `notes/T09b-C08-review.md` (2026-10-08). Na checkout principal, por commitar. Cada teste
foi escrito antes da correcção e falhou pela razão certa; os que já passavam são guardas, e
digo-o. Nenhuma mudança em `_http.py`, `_window.py`, `_values.py` nem `core/`.

##### M2 · `http_get`: o rodapé avança sempre

- **Mudança (`_web.py`):** depois da primeira leitura, `(offset + max_chars) × 4` bytes, enquanto
  a página não veio inteira, o texto lido é mais curto do que `offset + max_chars` e o orçamento
  ainda não chegou aos 10 MB, o `http_get` volta a pedir a página com o dobro dos bytes (no
  mínimo 64 KiB, uma leitura do socket da porta). Cobre o BOM do UTF-16/32, os escapes do
  ISO-2022-JP e qualquer carácter de mais de 4 bytes.
- **A invariante:** `_window` afirma (`assert`, como o `core/` já faz para as suas invariantes)
  que um rodapé com chamada seguinte tem `offset` maior do que o actual.
- **No tecto:** um `offset` para lá de tudo o que o `http_get` lê já não recua. Mostra
  `[chars 300-300 | end]` nesse `offset`, com a linha "goes on past its first N bytes" por cima.
- **Porquê na tool:** o `fetch_page` só devolve o texto já descodificado, sem bytes nem charset, e
  a porta não é minha. Verifiquei que o prefixo descodificado de um corte em qualquer byte (sem o
  U+FFFD final) é sempre um prefixo do texto inteiro, em 15 codificações: UTF-8/16/32, UTF-7,
  ISO-2022-JP/-JP-2/-KR, HZ, GB18030, Shift_JIS, CP932, EUC-JP e Big5. Só faltava comprimento.
  Cada nova leitura é outro GET ao mesmo URL, sob a mesma aprovação. A docstring di-lo.
- **Testes:**
  - `test_every_footer_reads_on_past_its_offset_whatever_the_encoding` (7 casos). Cobre UTF-16 e
    UTF-32 com BOM e carácter astral a `max_chars=1`, UTF-32 a 3, ISO-2022-JP `"a日"*50` a 1, 2
    e 3, e kanji com kana a 7. Afirma `next > offset` a cada passo e reconstrói o texto. Os casos
    `utf16-bom`, `utf32-bom` e `jis-1/2/3` falharam; `utf32-bom-3` e `jis-kana-7` já avançavam e
    ficam como guarda.
  - `test_a_short_read_is_read_again_further_only_as_far_as_it_needs`: 2 pedidos.
  - `test_an_offset_past_what_http_get_reads_is_not_moved_back`: falhou com
    `[chars 200-200 | end]`.
  - Os probes do revisor dão agora "ok" em todos os casos: `web_stall.py`, e `web_follow.py`
    (`bad 0` em 9 codificações).

##### M4 · Open Library: a pesquisa de autores falha, o registo fica

- **Mudança (`_open_library.py`):** o `_named` apanha o `ToolFailure` do pedido a
  `/search/authors.json`. Faz como o `_news.py` faz por história: sem `return` dentro da excepção
  e sem ler `.status`, por isso os testes de arquitectura passam. Deixa as chaves e diz porquê:
  `Authors: OL26320A (names not read: HTTP error 503: Unavailable; call again to read them)`.
  O "call again" só aparece quando a falha é retryable. Campo novo: `_Book.names_unread`.
- **Testes:**
  - `test_a_failure_of_the_names_request_keeps_the_keys_and_says_so`: substitui
    `test_a_failure_of_the_names_request_is_the_tools_failure`, que afirmava o contrário.
  - `test_a_names_failure_that_will_not_pass_does_not_ask_to_call_again`: um ISBN cuja pesquisa
    de autores responde HTML.
- **A nota de desenho mudou:** dizia "Uma falha desse pedido é a falha da tool". Deixa de ser.

##### M5 · Internet Archive: o item desconhecido `[]` é `not_found`

- **Sem mudar a porta.** O `internet_archive_item` lê pelo `get_text` e descodifica o JSON no
  `_item_answer`, dentro do `parse=`. Por isso um corpo que não é JSON, ou JSON de outra forma,
  continua a ser "could not parse API response" (`upstream`).
  - `[]` (documentado em https://archive.org/developers/md-read.html) e `{}` dão `None`, que é
    `not_found` com o passo seguinte.
  - Uma lista com conteúdo é um erro de parse.
- **O leitor de erros continua a funcionar.** A `Api` mantém `error_reader=_metadata_error`, que
  a porta corre em todos os estados de erro (o `get_text` passa pelo `_status_failure`). O
  `_item_answer` corre-o sobre o corpo de um 200, como o `get_json` fazia.
- **Testes:**
  - `test_an_unknown_identifier_answered_with_an_empty_array_is_not_found`: falhou com
    `upstream`.
  - Guardas, verdes antes e depois: `test_an_array_with_something_in_it_is_a_parse_error` e
    `test_an_error_status_is_still_read_by_the_metadata_reader` (um 500 com `errcode` 102 dá
    `upstream` retryable, com as palavras da fonte). Os testes de `errcode` 104/102/101/400 e o
    do `{}` continuam verdes.
- **Contrato (`contract_cases.py`, só a minha entrada):** o caso `not_found` do
  `internet_archive_item` passou do 404 para a resposta que a documentação dá, `[]`, com o URL
  no comentário. O nome saiu da lista do `_missing_by_404`.
- **Sem `xfail`:** não é precisa nenhuma mudança na porta.

##### L6 · IA: um `error` ao lado dos dados

- **Mudança:** o `_metadata_error` devolve `None` quando o corpo traz um `metadata`. O código 106
  diz que se usou uma cópia secundária, por isso os dados vêm com ele. O aviso sai no cabeçalho:
  `Internet Archive item X (the Internet Archive says: Unbalanced locations):`.
- **Teste:** `test_a_warning_beside_the_item_does_not_fail_the_read`. Falhou com
  `upstream: Unbalanced locations`.

##### L7 · YouTube: os erros de forma da biblioteca

- **Mudança:** as três tools apanham as excepções de forma (`_SHAPE_ERRORS`) como `upstream` não
  retryable: "could not parse the transcript response: KeyError('runs')". A tupla é a da porta,
  copiada no módulo, porque o YouTube não passa pela porta e o nome é privado do `_http`.
- **Teste:** `test_a_page_the_library_cannot_read_is_an_upstream_parse_failure`, com 3 casos:
  - `KeyError` na lista, no `youtube_transcript`;
  - `KeyError` na lista, no `youtube_transcript_languages`;
  - `ParseError` no `fetch`, no `youtube_transcript_search`.
  Os três falharam como `runtime_error`.

##### L9 · `scrape_text`: a nota dos 2 MB dá a chamada

- **Mudança:** a nota diz "...its text stops there; http_get with offset=N reads the HTML on, or
  find= searches it)". `N` é o número de caracteres de HTML lidos, não de bytes: é o `offset` do
  `http_get`.
- **Testes:**
  - o teste da nota, actualizado (falhou);
  - `test_the_html_scrape_text_did_not_read_is_where_http_get_reads_on`: segue esse `offset`
    pelo `http_get` (com aprovação) e cai no HTML seguinte, com um "á" de 2 bytes.

##### L10 · Gramática do YouTube

- **Mudança:** "The YouTube transcript of X, en (English), manual, has no passage that mentions
  'q'."
- **Teste:** `test_search_handles_no_matches`, actualizado (falhou).
- As duas frases partilhadas que o revisor assinala (`docs/tools.md`, `docs/tools-catalog.md`)
  são do coordenador. A de `docs/tools.md` já está actualizada na checkout.

##### Linhas do CHANGELOG (o que muda nas propostas)

- **Changed (`http_get`, acrescentar):** "A page whose characters take more than four bytes (a
  byte-order mark, ISO-2022-JP) is read again, further, so every footer reads on."
- **Changed (Open Library, acrescentar à linha dos autores):** "when the author search fails, the
  record still reads, with its authors by key and why their names are missing."
- **Fixed (nova):** "`internet_archive_item` reads the empty array the metadata API answers for an
  unknown identifier as `not_found`, and an item that comes with a warning (code 106) as the
  item, with the warning in its heading."
- **Fixed (nova):** "The `youtube_*` tools report a page youtube-transcript-api cannot read (a
  missing key, malformed XML) as an `upstream` parse failure, not a retryable `runtime_error`."
- **Changed (`scrape_text`, acrescentar):** "When it stops at its 2 MB of HTML, it names the
  `http_get` offset that reads the rest."

##### Verificação

- `uv run pytest -q` dos quatro ficheiros de teste, mais `test_web_search.py`,
  `test_tool_contract.py`, `test_tool_invariants.py` e `test_tools_exports.py`: **1054 passed, 2
  xfailed**. Os dois `xfail` são da C08.
- `tests/test_architecture.py` e `tests/test_quality_budget.py`: 34 passed.
- `uv run pyright src`: 0 erros.
- `ruff check` e `ruff format --check` nos meus 10 ficheiros: limpos. Nos dois ficheiros
  partilhados corri só o `ruff check`, que passou.
- Em `tests/toolkit` inteiro, as únicas falhas são de `test_eurostat.py`: o módulo e os testes de
  outro agente estão a ser mudados agora.
