# C07 · Tools de escrita tipadas e `FilesystemPolicy`

- **Dono:** por atribuir · **Estado:** todo · **Depende de:** nada (coordenar `core/_tools` com C02)
- **Origem:** `docs/internal/agentes-app-toolkit-review.md` — L9 (`:360-379`), §10 (`:184-190`), §4
  `ResourcePolicy.check_path` para âmbitos (`:460-462`), ponto C (`:739-757`)
- **Decisões:** por fixar (ver abaixo); respeita D4 (aprovação) e D7 (validar antes dos gates)

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
symlinks, com preview legível e dry-run; stdlib, erros como strings, leituras actuais intactas.

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

- Estado: todo
- Ficheiros tocados:
- Testes novos:
- Verificações:
- CHANGELOG proposto:
- Desvios ao plano:
