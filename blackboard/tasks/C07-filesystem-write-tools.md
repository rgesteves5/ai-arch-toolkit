# C07 · Tools de escrita tipadas e `FilesystemPolicy`

- **Dono:** Claude (agentes: C07a, b, d, e, f numa worktree; C07c e as correcções no checkout; 2026-10-08 e 09) · **Estado:** done · **Depende de:** nada (coordenar `core/_tools` com C02)
- **Origem:** `docs/internal/agentes-app-toolkit-review.md` — L9 (`:360-379`), §10 (`:184-190`), §4
  `ResourcePolicy.check_path` para âmbitos (`:460-462`), ponto C (`:739-757`)
- **Decisões:** fixadas pelo dono a 2026-10-08, D64; respeita D4 (aprovação) e D7 (validar antes dos gates)
- **Actualização (2026-09-30):** espera pela T01 e pela T03 e nasce com o contrato das tools (D37 a
  D42). As referências a ficheiros e linhas são de 2026-09-15, anteriores às frentes R e T: relê-as
  contra o `main` antes de começar.

## Problema

- **Sem escrita tipada:** as tools de ficheiros só lêem (`toolkit/tools/_filesystem.py:13-129`);
  escrever só por `run_command` com `shell=True` (`_shell.py:32-38`), sem schema nem âmbito.
- **Leituras sem raízes** (`Path(path).expanduser()`, `_filesystem.py:26,56,100`): com
  `root/leaf_link.txt -> outside/secret.txt`, `search_files(root, "secret")` →
  `leaf_link.txt:1: OUTSIDE-SECRET`; `list_directory(root, pattern="../outside/*")` → `secret.txt`.
- **`ResourcePolicy.check_path`** (`toolkit/resources/_policy.py:45-60`) só serve os resources: sem
  acções, relativos contra o cwd do processo (`:52`), `allow_symlinks` só vê a folha (`:50`).
- **Gate sozinho não garante âmbito:** corre antes do `ApprovalGate` (`core/_tools/_group.py:57-59`)
  e não revê os `modified_args` (gate "só dentro" + aprovador que troca `path` → `'OUTSIDE'`); com
  lista, `execute_tool`/`run_tools` só põem o `ApprovalGate` (`_executor.py:389,414`); `path` omitido
  chega ao gate como `{}` (`_validation.py:79-82`).
- **Preview = JSON dos argumentos** (`_approval.py:127,160-165`): 200 KB de conteúdo → `preview` de
  200 063 caracteres. `GovernanceOutcome` (`_governance.py:26-33`) não tem recusa por permissão.

## Objectivo

Escrita tipada (`write_file`, `append_file`, `make_directory`, `move_path`) e uma `FilesystemPolicy`
com raízes por acção, aplicada com o mesmo `check` no `PathScopeGate` (recusa antes do humano) e nas
tools de `filesystem_tools(policy)` imediatamente antes do syscall (a garantia). Atómica, sem seguir
symlinks, com preview legível e dry-run; stdlib, falhas tipadas (D37: o `ToolFailure` da T01),
leituras actuais intactas.

## API proposta

```python
type FilesystemAction = Literal["read", "write", "delete"]
@dataclass(frozen=True, slots=True, kw_only=True)
class FilesystemPolicy:
    read_roots: tuple[Path, ...] = ()
    write_roots: tuple[Path, ...] = ()
    delete_roots: tuple[Path, ...] = ()  # nenhuma tool de v1 remove; para tools da app/MCP
    cwd: Path | None = None              # base dos relativos; None → cwd na construção
    max_write_bytes: int = 1_048_576
    def check(self, path: str | os.PathLike[str], action: FilesystemAction) -> Path: ...
class PathScopeGate:  # ToolGate
    def __init__(self, policy: FilesystemPolicy, *,
                 paths: Mapping[str, Mapping[str, FilesystemAction]] | None = None) -> None: ...
def filesystem_tools(policy: FilesystemPolicy) -> tuple[Callable[..., str], ...]: ...
# leituras (se read_roots) com nome e schema actuais; escritas (se write_roots):
# write_file(path, content, overwrite=False, create_parents=False) · append_file(path, content)
# make_directory(path, parents=False) · move_path(source, destination, overwrite=False)
@tool(..., preview=fn)  # core: ToolDefinition.preview: Callable[[dict[str, Any]], str] | None
```

- **`check`** devolve o caminho canónico ou levanta `FilesystemPolicyError(PermissionError)`: `read` →
  alvo em `read_roots`; `write`/`delete` → pai nas raízes da acção e folha não resolvida (nunca
  symlink, `..` ou nome reservado; a própria raiz não se cria, move nem substitui).
- **Gate:** `check` por argumento mapeado (omitido → `default` do schema; sem default → recusa);
  recusa → `GateBlock(error_type="permission_denied")` com `audit["filesystem"]`; aceite →
  `GateModify` com os caminhos canónicos, que o aprovador vê. Mapa por omissão: as sete tools;
  `capability="filesystem"` sem mapa → bloqueada. No `check` async o I/O vai para `to_thread`.
- **Tools:** metadados D4; `check` de novo antes do syscall; recusa → string, como `"File not found"`;
  sucesso → caminho canónico e bytes. Preview: `replace /…/a.md (1204 → 1311 bytes)` + diff truncado
  (80 linhas / 8 KB; nenhum se binário ou > 256 KB). `DryRunGate` depois do gate: nada escrito nem
  medido, preview no audit. Só o executor mede (`_executor.py:223-252`): recusa do gate não conta, a
  da tool conta como chamada executada.

## Decisões a fixar antes de codificar

**Fixadas a 2026-10-08 (D64), todas na opção recomendada; a lista abaixo fica como estava.**

1. **Onde se aplica.** (a) só gate: não vê `modified_args`, falta com lista, janela aberta durante a
   aprovação; (b) só tool: pede aprovação ao que depois recusa; (c) ambos. Recomendo (c): o gate
   poupa o humano e o budget, a tool fecha os três buracos.
2. **Como o gate sabe caminhos e acção.** (a) `path_args` em `ToolRuntimePolicy` (filesystem no core,
   contrato de C02); (b) mapa no construtor. Recomendo (b), porque não toca no core e falha fechado;
   tools MCP (C03) mapeiam-se por nome.
3. **Leituras existentes.** (a) parâmetro `policy` (entra no schema); (b) contextvar (implícito); (c)
   factory. Recomendo (c): `read_file` & co. não mudam; as ligadas recusam `..` no `pattern` e saltam
   entradas cujo alvo sai das raízes. Não há escrita sem policy.
4. **Canonicalização e TOCTOU.** Recomendo `os.path.realpath(strict=os.path.ALLOW_MISSING)` (3.13.4+:
   ENOTDIR e laços levantam; antes, `strict=False` e recusa de `..` residual), `is_relative_to` só
   depois (é lexical), `os.path.isreserved` em Windows e nenhum `casefold` (em macOS `realpath` mantém
   a caixa dada e o APFS pode ser sensível: recusa falsa, nunca aceitação falsa). Antes do syscall, em
   POSIX, abrir da raiz componente a componente com `O_DIRECTORY|O_NOFOLLOW` e operar relativo ao
   `dir_fd` do pai (symlink trocado → `ENOTDIR`; testar `os.rename in os.supports_dir_fd`, porque em
   macOS `os.replace` aceita `dir_fd` sem constar lá); em Windows, sem os dois, repetir `check` e
   documentar a janela. Comportamento POSIX verificado em macOS/3.13.7.
5. **Atomicidade e conteúdo.** Recomendo temporário no mesmo directório (`O_CREAT|O_EXCL|O_NOFOLLOW`,
   `O_BINARY` em Windows, `0o666` com umask; `mkstemp` dá `0o600`) → `fsync` → sem `overwrite`,
   `os.link` + `unlink` (POSIX) ou `os.rename` (Windows), que recusam destino existente (o `os.rename`
   POSIX substitui; sem hard links, `lstat` + `os.replace` com janela documentada); com `overwrite`,
   `os.replace` com o modo antigo (atómico como requisito POSIX; substitui um symlink em vez de o
   seguir, daí a recusa) e `fsync` do directório. `append_file` (regular, `st_nlink == 1`) não é
   atómico; `move_path` não copia entre volumes. UTF-8 estrito, fins de linha intactos, `content`
   não-`str` → erro (`_validation.py:13-15` não impõe `string`). Porquê: `react_flow` corre tool
   calls em paralelo por omissão (`_react.py:27,112`).
6. **Preview e recusa.** (a) hook `@tool(preview=)` lido por `approval_request_for` e `DryRunGate`;
   (b) helper para o handler da app. Recomendo (a): serve app, CLI e MCP e chega ao dry-run. Recusa:
   `"permission_denied"` novo em `GovernanceOutcome` (`dangerous_tool_blocked` fala de flags de CLI).
7. **Remoção:** fora de v1. O lixo do SO não tem API stdlib e é da app (L9); a quarentena (`rename`
   para `trash_dir` + manifesto) é viável, mas retenção, restauro e UI são da app e `move_path` já dá
   o mecanismo. `delete_roots` serve tools da app/MCP.

Docs: https://docs.python.org/3.13/library/os.html (`#os.replace`, `#os.rename`, `#os.open`) e
https://docs.python.org/3.13/library/os.path.html (`#os.path.realpath`, `#os.path.isreserved`).

## Sub-tarefas, por ordem

- **C07a** `FilesystemPolicy`, `FilesystemPolicyError`, `check`, canonicalização.
- **C07b** `"permission_denied"`; `PathScopeGate` (mapa, defaults, bloqueio, `GateModify` canónico).
- **C07c** Hook de preview no core (fallback, `to_thread`, `DryRunGate`; `tool_from_schema` de C02
  aceita-o). Independente de a/b.
- **C07d** Operações privadas: caminhada `dir_fd`, escrita atómica, append, mkdir, rename, fallback.
- **C07e** Tools e previews de escrita, leituras ligadas, `filesystem_tools`. **C07f** Exports e docs.

## Ficheiros

- `src/ai_arch_toolkit/toolkit/tools/`: `_filesystem_policy.py` e `_filesystem_write.py` (novos),
  `_filesystem.py` (helpers, factory), `dangerous.py` (exporta `FilesystemAction`, `FilesystemPolicy`,
  `FilesystemPolicyError`, `PathScopeGate`, `filesystem_tools`).
- `src/ai_arch_toolkit/core/_tools/`: `_definition.py`, `_decorator.py` (partilhados com C02),
  `_approval.py` (partilhado com C04), `_governance.py`.
- `tests/toolkit/test_filesystem_policy.py`, `tests/toolkit/test_filesystem_write.py` (novos),
  `tests/toolkit/test_filesystem.py`, `tests/toolkit/test_tools_exports.py` (partilhado com C08),
  `tests/test_tools_decorator.py`, `tests/test_tools_group.py` (partilhados com C02).
- `docs/safety.md` (`:57-60`, `:104-113`, `:158-167`; partilhado com C02, C04, C08),
  `docs/tools-catalog.md` (`:248-256`; partilhado com C08), `docs/tools.md` (`:145`; partilhado com
  C02–C05, C08), `docs/framework-overview.md` (`:200`; partilhado com C03, C06).

## Prova

- `test_filesystem_policy.py`: `check` recusa `..` para fora, symlink na folha e em directório
  intermédio, ENOTDIR e laço; aceita destino inexistente; relativos contra `cwd`; `write` na raiz
  recusado, `read` aceite; igual sem `ALLOW_MISSING`. Gate: fora → `permission_denied` sem chamar o
  handler nem abrir operação no `MeterScope`; `{}` em `list_directory` → `cwd` da policy; argumento
  mapeado omitido sem `default` → recusa; `capability="filesystem"` sem mapa → bloqueada.
- `test_filesystem_write.py` (`tmp_path`): bytes exactos; sem `overwrite` o existente fica intacto;
  folha symlink → alvo intacto; duas escritas paralelas ao mesmo caminho novo → um `Created`, um erro;
  TOCTOU: o handler troca `root/sub` por symlink para fora antes de aprovar, ou devolve `modified_args`
  para fora → recusa, nada fora; `append_file` com hard link → erro; preview `replace` com `-`/`+`;
  `DryRunGate` → nada escrito; `search_files` ligado sem `OUTSIDE-SECRET`.
- `test_tools_group.py`: `ApprovalRequest.preview` vem do hook; hook que levanta → preview actual.
- Antes, falham pelas repros do Problema ou por a API não existir.

## Fora do âmbito

- Remoção; `edit_file` por substituição (candidato seguinte); binário e outras codificações;
  `expected_sha256`; mudar `ResourcePolicy`; âmbito por caminho para `run_command`.
- Sandbox (L10); apps/URLs, clipboard, lixo do SO, `system_info`, UI de aprovação (app, L9).
- Limite de saída das leituras e `csv_read` sem metadados: achados para o `FINDINGS.md`.

## Riscos

- Sem Windows no CI (`.github/workflows/ci.yml:45`): `dir_fd`, `isreserved`, `O_BINARY` e
  `os.replace` (`PermissionError` com o ficheiro aberto) sem prova; verificar à mão ou juntar
  `windows-latest` (decisão do dono).
- `run_command` e `csv_read` no mesmo grupo contornam o âmbito (`python_repl` bloqueia `open`, e
  `http_get` só aceita `http(s)://`, ambos verificados): dizê-lo em `docs/safety.md`.
- Inode novo: perde hard links, ACL/xattrs e dono. O preview é um retrato do ficheiro, não um lock.
- C04: `append_file` e `move_path` não são idempotentes; o audit de aprovação guarda o conteúdo.

## Registo do dono

- **Estado:** done. As notas de desenho dos agentes da C07 e da C07c perderam-se num reinício da
  sessão (estavam num directório temporário); o que segue vem dos relatórios deles, da revisão e
  das notas das correcções.
- **Gate final, no checkout principal:** 8425 passed, 118 skipped; ruff, formatação, pyright e
  `uv lock --check` limpos (2026-10-09).
- **CHANGELOG:** as linhas entraram em `[Unreleased]` (Upgrade notes, Added, Changed).

### C07a, b, d, e, f (um agente numa worktree)

- **Ficheiros novos:** `toolkit/tools/_filesystem_policy.py` (`FilesystemPolicy`,
  `FilesystemPolicyError`, `check`, `PathScopeGate`), `_filesystem_write.py` (as operações
  privadas, as quatro tools de escrita e o `filesystem_tools`), `tests/toolkit/test_filesystem_policy.py`
  e `test_filesystem_write.py`.
- **Ficheiros mudados:** `core/_tools/_result.py` e `_governance.py` (`permission_denied`, C07.9),
  `toolkit/tools/_filesystem.py` e `dangerous.py`, `tests/toolkit/tool_catalog.py` (as tools da
  factory com uma policy de teste, C07.3), `contract_cases.py`, `test_tools_exports.py`,
  `test_tool_invariants.py` (o `dangerous` exporta nomes que não são tools), as docs, e o
  `pyproject.toml` com o `uv.lock` (Python 3.13.4, C07.4).
- **Desvios:** o `filesystem_tools` vive no `_filesystem_write.py` e não no `_filesystem.py`, para
  evitar um ciclo de imports. As raízes relativas e o `cwd` da policy resolvem-se contra o
  directório do processo; os caminhos relativos do agente, contra o `cwd` da policy.
- **Prova:** os testes novos falhavam antes por `ImportError`; dez mutações das garantias (o link
  atómico, a verificação de novo na tool, a caminhada sem seguir links, a recusa de hard links,
  entre outras) fizeram falhar pelo menos um teste cada.

### C07c (o hook de preview, a pedido do dono a 2026-10-08)

- `ToolDefinition.preview`, posto por `@tool(preview=)` e por `tool_from_schema(preview=)`; lido
  pelo pedido de aprovação e pelo `DryRunGate`, com uma cópia dos argumentos validados e
  mudados pelos gates; o preview das quatro tools de escrita (acção, caminho canónico, tamanhos e
  um diff cortado a 80 linhas / 8 KB).
- **Prova:** 45 testes novos, cada um visto a falhar; 16 mutações, todas apanhadas.

### Revisão independente (2026-10-08)

Nada saiu das raízes. Três médios: as recusas do gate diziam o que havia fora das raízes (M1); dois
`move_path` em paralelo podiam apagar um ficheiro (M2); uma raiz de escrita que não é de leitura
lia-se por um move (M3). Cinco baixos e um pormenor; e, da C07c, o preview em JSON sem limite e
um hook sem prazo.


Fonte: `notes/C07-review.md`. Cada teste foi escrito primeiro e falhou pela razão certa. Depois
veio o código. Nada foi commitado.

### Correcções da revisão (2026-10-09)

#### Gate final

- `uv run pytest -q`: 8425 passados, 118 saltados.
- `uv run ruff check src tests examples` e `ruff format --check`: limpos.
- `uv run pyright src`: 0 erros.

#### Médios

##### M1 · O gate não diz nada sobre o que está fora das raízes

- **Alteração** (`_filesystem_policy.py`):
  - O `check` testa as raízes antes de olhar para a folha. O `lstat` da folha (`_is_link`) só
    corre quando o pai já está dentro.
  - Toda a recusa de um caminho de fora sai de uma só função, `_outside`. Diz o caminho como foi
    dado e as raízes. Nunca diz o caminho resolvido.
  - Se o `realpath` falhar (ENOTDIR, ELOOP, EACCES, nome longo), o `error.filename` indica a parte
    que estava a resolver. Fora das raízes, a falha dá a mesma recusa `_outside`. Dentro delas é
    erro do próprio caminho (ver L1).
  - As mensagens do `_protects` e do link na folha passam a nomear o caminho dado.
- **Testes** (`test_filesystem_policy.py`, classe `TestNothingIsLearntOutside`):
  - nove caminhos de fora (ficheiro, nada, pasta, através de um ficheiro, através de nada, link,
    através de um link, laço, laço na folha) dão a mesma recusa, em `read` e em `write`;
  - uma pasta de fora sem permissão de pesquisa dá a mesma recusa;
  - um espião no `os.lstat` prova que a folha de um caminho de fora nunca é vista;
  - um link de dentro para fora nunca revela o alvo;
  - pelo gate e pela tool, a recusa tem o mesmo tipo e a mesma forma, e o audit guarda o caminho
    dado.
  - Dois testes antigos esperavam o caminho resolvido na mensagem. Agora esperam o contrário.

##### M2 · Um move já não apaga um ficheiro que não lhe pediram

- **Alteração** (`_filesystem_write.py`):
  - As quatro tools de escrita passam a ter vez, uma de cada vez por processo (`_one_write`, um
    `threading.Lock`). Fecha a corrida da revisão entre chamadas paralelas do ReAct.
  - A espera máxima é de 30 s (`_WAIT_S`). Depois disso, a escrita falha com `upstream`,
    retryable. Assim não corre depois de a chamada ter dado timeout.
  - Em `_link_then_unlink`, o nome antigo só é removido se ainda aponta para o inode ligado
    (`samestat`). Se outro processo lhe deu outro ficheiro, esse ficheiro fica e a resposta diz:
    "… no longer names the file that moved, and is left as it is".
  - Não se desfaz o link, ao contrário do que a revisão sugeria. Desfazê-lo apagaria o original, e
    a nova tentativa moveria o ficheiro do outro. Manter os dois nomes equivale a uma ordem serial
    válida: primeiro o move, depois a escrita do outro processo.
  - A janela entre a comparação e o `unlink` fica, entre processos. Está documentada.
- **Testes** (`test_filesystem_write.py`):
  - `test_a_source_replaced_between_link_and_unlink_is_left_in_place`: um `os.link` interceptado
    faz o `replace` de outro "processo" logo a seguir ao link. É determinístico.
  - `test_two_moves_of_one_process_never_interleave`: o move B arranca dentro da janela do move A
    e tem de esperar. Sem o lock, B acabava na janela.
  - `test_a_write_that_waits_too_long_for_another_fails_and_writes_nothing`.
  - O teste antigo das duas escritas paralelas usava uma `Barrier` dentro do `os.link`, o que com o
    lock dá deadlock. Agora corre sem barreira, e um teste novo cobre o `os.link` que recusa um
    nome tomado por outro processo (`test_a_name_another_process_takes_before_the_link_is_never_replaced`).

##### M3 · As raízes de escrita não passam a ser lidas por um move

- **Regra escolhida:** um move cujo destino a policy lê e cuja origem não lê é recusado com
  `permission_denied`, no gate e na tool. Esta regra não recusa mais nada.
- **Porquê:** das quatro escritas, só o move traz bytes que o agente não enviou. O `write_file` e o
  `append_file` escrevem o texto do agente, e o `make_directory` não escreve nada. Logo, só o move
  alarga o que se lê, e só quando o destino é lido e a origem não é. Um move que sai de uma raiz de
  leitura, ou que fica entre pastas não lidas, deixa as leituras como estavam.
- **Caixa:** o destino conta como lido com caixa e forma Unicode dobradas, como no `_protects`. A
  origem conta só como está escrita. Num disco insensível à caixa, o erro possível é uma recusa
  falsa, nunca uma aceitação falsa.
- **Código:** `check_move` em `_filesystem_policy.py`, com a justificação na docstring.
  - O gate aplica-a aos tools de `_MOVES` (o `move_path`).
  - A tool e o preview aplicam-na em `_move_paths`.
  - A justificação também está em `docs/safety.md`, "Limits to know".
- **Testes** (`TestMovesNeverWidenReads`):
  - `read_roots=(home/project,)`, `write_roots=(home,)`: `home/.secret` não se move para o projecto,
    nem por cima de um ficheiro lido, nem com `PROJECT` noutra caixa;
  - o gate recusa antes do aprovador;
  - o preview diz `will fail (permission_denied)` com as mesmas palavras;
  - os moves que não alargam as leituras continuam a funcionar.

#### Baixos

##### L1 · Um só classificador

- **Alteração:**
  - `checked` passa para `_filesystem_policy.py` e é a única leitura da resposta do `check`, no
    gate e na tool.
  - Dentro das raízes, um ENOTDIR ou ELOOP passa a ser `ValueError` e, portanto,
    `validation_error`. Antes era `permission_denied`.
  - O nulo é testado à entrada do `check`, dentro e fora.
  - Para um argumento que não é caminho (não é texto, é malformado, passa por um ficheiro), o gate
    faz o seguinte:
    - se a tool verifica os seus caminhos com a mesma policy (marca `__filesystem_policy__`, posta
      por `checks_paths` em `filesystem_tools`), o gate deixa-o passar. A tool falha então com
      `validation_error`, nas mesmas palavras, e o preview avisa o aprovador;
    - se não, o gate bloqueia com `permission_denied`. Fecha por omissão, porque uma tool da app
      podia receber `5` e abrir o descritor 5.
  - O `GovernanceOutcome` não tem `validation_error`, e mudá-lo estava fora do âmbito.
- **Testes:**
  - `TestOneWordForOnePath`: seis casos (nulo, `5`, através de um ficheiro, em `read_file`,
    `list_directory`, `write_file`, `make_directory` e `move_path`) dão o mesmo tipo e as mesmas
    palavras com e sem gate;
  - uma tool ligada a outra policy fica bloqueada;
  - a recusa do padrão é igual no gate e na tool;
  - em `test_filesystem_policy.py`, o `read_file` do módulo (sem verificação própria) é bloqueado
    para nome longo, nulo, `5` e caminho através de um ficheiro;
  - nos testes do `check`, ENOTDIR e laço dentro das raízes passam a dar `ValueError`, que não é
    `FilesystemPolicyError`.

##### L2 · Um fsync de pasta que falha é uma nota

- **Alteração:** `_synced(*folders)` corre depois de o nome estar no sítio, em `write_file`,
  `make_directory` e `move_path`. Se o fsync falhar, a resposta acaba em
  `; the folder could not be synced to disk (…): the change is made, but may not survive a crash`.
  O fsync do ficheiro temporário continua a ser falha, porque acontece antes de o nome existir.
- **Teste:** `test_a_folder_that_cannot_be_synced_is_a_note_not_a_failure` dá EINVAL só a pastas.
  Cobre create, replace, mkdir e move, e verifica os bytes finais.

##### L3 · Move entre dois hard links do mesmo ficheiro

- **Alteração:** `_refuse_two_names` corre quando origem e destino têm o mesmo inode:
  - em pastas diferentes, ou com os dois nomes listados na mesma pasta, é recusado com
    `validation_error`, "the same file, under two names";
  - com um só nome na pasta (mudança de caixa num disco insensível), o rename continua.
  - Escolhi recusar e não fazer `unlink`: nenhuma tool remove (C07.7), e um `unlink` aqui seria uma
    remoção disfarçada. O preview diz o mesmo.
- **Testes:** `test_a_move_onto_another_hard_link_of_the_same_file_is_refused` cobre outra pasta e
  a mesma pasta. `test_a_rename_that_changes_only_the_case_still_works` corre no APFS e é saltado
  num disco sensível à caixa.

##### L4 · O gate verifica o padrão

- **Alteração:** `climbs(pattern)` em `_filesystem_policy.py` é usada pelo gate (`_GLOBS`,
  `list_directory.pattern`) e pela tool ligada.
- **Testes:** `test_a_pattern_that_climbs_out_is_refused_before_the_approver` (três padrões, sem
  aprovador, audit com `argument="pattern"`) e o teste de palavras iguais no gate e na tool.

##### L5 · `docs/safety.md`

- Uma tool MCP só é bloqueada se a app lhe declarar `capability="filesystem"`
  (`tool_from_schema` não põe nenhuma). Sem isso, passa intacta.
- Numa listagem ou pesquisa ligada, uma pasta trocada depois do check dá uma resposta sem o que
  está fora (vazia, ou com os ficheiros que não leu). Não dá falha.
- A garantia é "dentro das raízes", não "exactamente estes caminhos". O aprovador vê os caminhos
  que a chamada nomeia, não um retrato dos ficheiros.
- Em "Limits to know" entram a regra do M3 (com a justificação), a janela entre processos do M2 e
  os restos de um move interrompido.
- Ficam também descritas a recusa única de fora (M1), a vez das escritas (M2), a nota de sync (L2),
  os hard links (L3) e a recusa do padrão no gate (L4).

##### Nit · A mensagem de uma tool sem mapa

- **Alteração:** `_unmapped` lista os argumentos do schema da tool e dá um exemplo com o primeiro,
  por exemplo `PathScopeGate(paths={'touch': {'target': 'read'}})`.
- **Testes:** `test_the_block_of_an_unmapped_tool_names_its_own_arguments`. O `csv_read` passa a
  ter a sugestão exacta `{'path': 'read'}`.

#### Seguimentos da C07c

##### a · `governed(..., preview=)`

- `governed(reason, *, preview=None)` em `_filesystem.py` passa o hook ao `@tool`.
- O `_governed` local, que fazia `replace` na definição depois de decorada, saiu. No lugar dele
  ficou `_previewed(picture, policy)`, que só constrói o hook.
- Os testes de preview que já existiam cobrem o comportamento, sem mudança.

##### b · O preview JSON também é cortado

- `_arguments_preview` passa pelo `_bounded`: no máximo 16 000 caracteres, com nota.
  `request.arguments` fica inteiro.
- Teste do agente anterior: `test_the_arguments_preview_is_cut_too` (sync e async), mantido.

##### c · Um hook que nunca volta

- **Alteração** (`_approval.py`):
  - o hook corre numa thread daemon própria nos dois caminhos (`_started`), com uma cópia do
    contexto de quem chama;
  - espera no máximo `PREVIEW_TIMEOUT_S = 10.0` (`concurrent.futures.wait`, ou `asyncio.wait`
    sobre o `wrap_future`);
  - depois disso o preview é o JSON e o hook fica a acabar sozinho;
  - o resultado lê-se do estado do future (`_answer`), não do tipo da excepção. Assim, um
    `TimeoutError` lançado pelo próprio hook não se confunde com o prazo.
- **Porque é sólido no caminho sync:**
  - a thread é daemon, por isso não segura o `asyncio.run` nem o fim do processo;
  - o contexto é copiado, e um teste novo prova-o;
  - no caminho async o hook já corria noutra thread (`to_thread`), por isso um hook preso à thread
    de quem chama já não era suportado. Os dois caminhos ficam iguais;
  - a thread que fica para trás é uma por chamada, como as tools da D30;
  - medi o preview mais pesado (diff de 256 KB): menos de 0,05 s. Os 10 s são largos.
- **Testes:**
  - `test_a_hook_that_never_returns_gives_way_to_the_arguments_preview` (do agente anterior) foi
    mantido. Tirei o `raising=False`, porque a constante agora existe;
  - novo: `test_the_hook_runs_in_the_callers_context` (sync e async). Falha se se tirar o
    `context.run`.
- Docstrings do `DryRunGate`, do `ApprovalGate`, do `ToolDefinition` e do `check_preview`
  actualizadas. Também `docs/safety.md` ("Human approval") e `docs/tools.md` (`preview=`).

#### Fora do âmbito, para decidir

- Um outcome `validation_error` no `GovernanceOutcome` deixaria o gate bloquear antes do aprovador
  o que hoje passa para a tool ligada (L1). É uma mudança no core.
- `_approval._started` repete o `_executor._in_thread`. Partilhá-lo obriga a mexer no
  `_executor.py`.
- No gate, a regra do move vale só para o `move_path` (`_MOVES`). Uma tool de move da app,
  mapeada em `paths`, só tem as verificações argumento a argumento.
