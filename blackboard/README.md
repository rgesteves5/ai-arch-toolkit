# Blackboard

Memória de trabalho partilhada entre os agentes (e as pessoas) que trabalham neste repositório ao
mesmo tempo ou em sessões diferentes. Lê isto antes de começares; escreve aqui enquanto trabalhas.

## Ficheiros

| Ficheiro | Para quê | Quem escreve |
|---|---|---|
| `BOARD.md` | Frentes de trabalho activas e estado de cada tarefa | coordenador |
| `tasks/<ID>-<slug>.md` | Uma tarefa: problema, decisão, ficheiros, prova, registo | só o dono da tarefa |
| `DECISIONS.md` | Decisões de desenho, com contexto e consequência (só acrescentar) | coordenador; os outros propõem na sua ficha |
| `FINDINGS.md` | Bugs ou riscos descobertos que nenhuma tarefa cobre (só acrescentar) | qualquer agente |
| `LOG.md` | Diário datado: o que aconteceu e o que vem a seguir | coordenador |

## Protocolo

1. **Orientar:** `AGENTS.md` → este ficheiro → `BOARD.md` → a tua ficha → as decisões que ela cita.
2. **Reclamar:** uma tarefa é tua quando o `BOARD.md` te dá como dono. Muda o estado na tua ficha
   para `in-progress` ao começar.
3. **Ficar no âmbito:** edita só os ficheiros que a ficha lista. Se precisares de outro, escreve em
   "Bloqueios" na ficha em vez de o editar.
4. **Provar:** escreve primeiro o teste comportamental e vê-o falhar pela razão certa; depois corrige.
5. **Registar:** mantém o "Registo do dono" da ficha actual — estado, ficheiros tocados, testes
   novos, resultado das verificações, linhas propostas para o `CHANGELOG`, desvios ao plano.
6. **Nunca:** fazer commit, push, merge, stash ou rebase, criar ou apagar branches, editar
   `CHANGELOG.md`. O dono do repositório revê o diff e faz os commits; o coordenador escreve o changelog.
7. **Trabalho paralelo** corre em worktrees git criadas a partir de `main`. As alterações ficam por
   commitar na worktree e o coordenador aplica-as ao checkout principal. Os ficheiros deste
   blackboard só existem no checkout principal: lê-os e escreve na tua ficha pelo caminho absoluto.
8. **Fora do âmbito:** acrescenta a `FINDINGS.md`, com a reprodução.

## Definição de pronto

- Um teste comportamental que falha antes e passa depois — nunca um teste a um getter.
- `uv run pytest -q` verde.
- `uv run ruff check src tests examples` e `uv run ruff format --check <ficheiros tocados>` limpos.
- `uv run pyright src` sem erros.
- Documentação actualizada quando o comportamento é visível; linhas propostas para `[Unreleased]`.

## Estados

`todo` · `in-progress` · `review` (feito na worktree, à espera de ser aplicado) · `done` (aplicado e
verificado no checkout principal) · `blocked`

## Comandos

```bash
uv sync --extra dev                          # uma vez, numa worktree nova
uv run pytest -q                             # suite inteira (~12 s)
uv run pytest tests/flow -q                  # um directório
uv run ruff check --fix src tests examples
uv run ruff format <ficheiros>
uv run pyright src                           # ~5 s
```

Testes com tempos (`asyncio.sleep`, timeouts) podem falhar sob carga quando vários agentes correm a
suite ao mesmo tempo: repete o teste isolado antes de concluir que partiu.
