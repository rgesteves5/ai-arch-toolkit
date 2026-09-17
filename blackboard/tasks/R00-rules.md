# R00 · Regras da frente de robustez (valem para R01, R02 e R03)

Lê isto inteiro antes de começares a tua fase. A tua ficha (`R01`, `R02` ou `R03`) diz o quê; este
ficheiro diz como.

## Contexto

Muitas aplicações dependem deste toolkit. O dono quer correcções estruturais, não remendos: o código
tem de sair de cada fase mais simples do que entrou. As causas, as costuras a redesenhar e as provas
estão em `docs/internal/hardening-plan.md`; os achados, com reproduções, em `blackboard/FINDINGS.md`
(entradas de 2026-09-15 e 2026-09-17); os scripts em `blackboard/prototypes/2026-09-hardening/`.

## Ler primeiro, por esta ordem

`AGENTS.md` → `blackboard/README.md` → `blackboard/BOARD.md` → este ficheiro → a tua ficha →
`blackboard/DECISIONS.md` (todas; D15–D20 são desta frente) → `docs/internal/hardening-plan.md` →
as entradas de `FINDINGS.md` que a tua ficha cita → o `README.md` dos protótipos.

## Regras de desenho

- **O critério.** Uma correcção é estrutural quando reduz o número de sítios que conhecem um assunto;
  é remendo quando acrescenta um caso especial a um sítio que já existe. Para cada passo, escreve na
  ficha, antes do código, uma nota de desenho curta: a interface (assinaturas e docstrings), que
  código desaparece e que testes o provam. Não esperes aprovação; a nota fica para quem revê.
- **Um assunto, uma casa.** Garante-o com testes de arquitectura por AST em `tests/test_architecture.py`.
- **Sem caminhos duplos.** Nenhum `if novo … else antigo`, nenhum shim de compatibilidade interno,
  nenhum `getattr`/`hasattr` para descobrir capacidades. Uma migração acaba na fase em que começa.
- **Tipos em vez de strings e flags:** `Literal`, dataclasses `frozen=True, slots=True` (`kw_only` a
  partir de 3 campos), `Protocol`, `TypedDict` do SDK. Nenhum `Any` novo em assinaturas públicas,
  nenhum `# type: ignore` nem `# noqa` novo.
- **Orçamento de complexidade.** Código novo ou tocado: complexidade ciclomática ≤ 10 (ruff `C901`),
  funções ≤ 60 linhas. A linha de base do repositório só pode descer (R01 cria o teste que o impõe).
- **Saldo de linhas.** Regista na ficha as linhas antes e depois dos módulos tocados. Nos módulos
  redesenhados o saldo esperado é negativo.
- **Testes primeiro.** Cada mudança de comportamento entra com um teste que falha antes pela razão
  certa. Um teste que afirma um bug corrige-se e lista-se na ficha; nunca se apaga nem se enfraquece
  um teste para o fazer passar.
- **Andaimes com saída.** Listas `xfail`, de legado ou de tolerância só encolhem e têm de estar vazias
  e apagadas no fim da fase que as criou.
- **API pública:** só muda o que a ficha diz. Cada mudança visível leva entrada no `CHANGELOG.md`
  (`Added`/`Changed`/`Fixed`, em inglês, no estilo das existentes) e, se for quebra, começa por
  "**Breaking:**".

## Proibido

- `git commit`, `push`, `merge`, `rebase`, `stash`, criar ou apagar branches e worktrees. O dono revê
  o diff e faz os commits. No fim, propõe na ficha a divisão em commits por área.
- Chamadas a fornecedores pagos ou a qualquer host que não seja `127.0.0.1`. Não leias o `.env`. Para
  exercitar os SDKs reais usa o servidor falso dos protótipos (`adapter-phases/fakeserver.py`).
- Tocar em `src/ai_arch_toolkit/nanope/`, `board/`, `.claude/worktrees/`, `research/`.
- Dependências obrigatórias novas. Extras novos só se a ficha os pedir; depois de mexer em
  dependências corre `uv lock`.
- Inventar formas de API. Confirma no SDK instalado (`uv run python`, `inspect`) e na documentação
  oficial, e escreve na ficha o URL e o que confirmaste.
- Alargar o âmbito. O que encontrares a mais vai para `FINDINGS.md`, com reprodução.

## Dúvidas e bloqueios

Se uma decisão não estiver em `DECISIONS.md` nem na ficha, escolhe a opção mais simples que respeite
estas regras, regista-a em `DECISIONS.md` (a partir de D21: contexto, decisão, consequência) e segue.
Se algo te bloquear de verdade, escreve em "Bloqueios" na ficha e avança para o passo seguinte que
não dependa dele.

## Verificação (corre no fim de cada passo; tudo verde antes de passares ao seguinte)

```bash
uv run pytest -m "not live_api" -q
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv run pyright src
uv lock --check
```

## Registo

Nesta frente és o coordenador da tua fase: manténs o "Registo" da tua ficha (estado, notas de
desenho, ficheiros tocados, testes novos, testes corrigidos, verificações, saldo de linhas, desvios),
o `CHANGELOG.md`, `DECISIONS.md`, `LOG.md` e a linha da tua fase no `BOARD.md`. O blackboard
escreve-se em português europeu com a grafia anterior a 1990 (actual, correcção, excepção, objectivo);
código, docstrings, docs de utilizador e `CHANGELOG` em inglês.

## Relatório final (na ficha e na tua última mensagem)

O que ficou feito por passo; o que não ficou e porquê; números (testes, complexidade antes e depois,
saldo de linhas); mudanças visíveis para quem depende do toolkit; o que não pôde ser verificado sem
chamadas ao vivo, com o comando exacto para o dono correr; divisão proposta em commits.
