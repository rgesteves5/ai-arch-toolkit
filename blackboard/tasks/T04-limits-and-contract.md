# T04 · Limites na assinatura e invariante de contrato

- **Dono:** T04a Claude; T04b Claude (2026-10-05) · **Estado:** T04a done (PR #71, `84c0ee2`); T04b done
- **Depende de:** T04a de nada (em série com a C02, que também mexe em `core/_tools`); T04b de T01,
  T02, T03 e T04a
- **Origem:** plano, secções 3.4 e 4 · **Decisões:** D37 a D42 · **Regras:** `T00-rules.md`

## T04a · Limites declarados

### Problema

- Os limites vivem nas docstrings ("(1-25)") e aplicam-se com `max(…, min(…))`, sem o dizer.
- O schema que o modelo recebe não os mostra.
- O validador (`core/_tools/_validation.py`) só verifica presença e tipo.
- Há limites que nem a docstring diz (`_eonet.py:89`, `_eurostat.py:167`).

### Desenho (confirma-o na nota de desenho)

- Um marcador do core, sem dependências, por exemplo `Annotated[int, Range(1, 25)]` (o nome
  decide-se na nota).
- O `_schema.py` escreve `minimum` e `maximum`, e também `minLength`/`maxLength` se uma string
  precisar.
- O validador recusa um valor fora do intervalo com `validation_error`, e a mensagem diz o intervalo
  aceite.
- O `Literal` já cobre as enumerações.
- Público em `ai_arch_toolkit.core`.

### Ficheiros

`core/_tools/_schema.py`, `core/_tools/_validation.py`, `core/_tools/__init__.py`, `core/__init__.py`,
`tests/test_tools_schema.py`, `tests/test_tools_validation.py`, `tests/test_core_exports.py`,
`docs/tools.md`.

### Prova

- O schema traz `minimum` e `maximum`.
- Um valor fora do intervalo dá `validation_error` com o intervalo; um valor dentro passa.
- O `Literal` e os tipos sem marcador ficam iguais.

## T04b · Invariante de contrato e lista de dívida

### Desenho

- **`tests/toolkit/test_tool_contract.py`** reutiliza o `TOOLS`, o `NETWORK`, o `_benign` e o
  `_answering` de `test_tool_invariants.py`; se for preciso, extrai-os para um módulo partilhado.
  Verifica, em todas as tools:
  1. um corpo maior do que a janela dá um rodapé, e a chamada do rodapé devolve a janela seguinte
     (nas tools com parâmetro de continuação);
  2. cada corpo de erro da tabela `tests/toolkit/error_bodies.py` (um ou mais por fonte, tirados da
     documentação, com o URL) dá `ToolFailure` com o código da fonte;
  3. um recurso que não existe dá `not_found`; zero resultados dá `ok=True` e uma frase que o diz;
  4. cada inteiro limitado tem o marcador da T04a, e o schema e o comportamento coincidem.
- **`tests/toolkit/contract_debt.py`** guarda o nome de cada tool e os pontos que ainda não cumpre.
  O teste falha se a lista crescer ou se uma tool listada já cumprir um ponto seu. Arranca com todas
  as tools que falham hoje.
- **Arquitectura:** as definições de `_truncate`/`_trim` nos módulos de `toolkit/tools` só podem
  diminuir. A contagem vive na lista de dívida e termina em zero.

### Ficheiros

`tests/toolkit/test_tool_contract.py`, `tests/toolkit/contract_debt.py` e
`tests/toolkit/error_bodies.py` (novos), `tests/toolkit/test_tool_invariants.py`.

### Prova

A invariante passa com a lista inicial. Um módulo de exemplo migrado de propósito, numa cópia de
trabalho, faz o teste pedir que se apague a sua linha da lista. Regista na ficha o tamanho inicial
da lista, por ponto.

### T04b · Nota de desenho (2026-10-05)

- **Partilhado:** a descoberta das tools, os argumentos benignos e o `_answer` saem de
  `test_tool_invariants.py` para `tests/toolkit/tool_catalog.py`; o `sandbox` passa para o
  `tests/conftest.py` (o pytest 9.1 perde as fixtures de um `conftest` de pasta quando a linha de
  comando mistura pastas). A invariante e o contrato importam-nos.
- **Dívida = o que não está provado.** Um ponto cumpre-se com um caso declarado que passa; sem caso,
  é dívida. O teste, por tool, calcula os pontos que falham e compara com `contract_debt.py`: um
  ponto novo a falhar ("the debt list only shrinks") ou um ponto listado que já passa ("delete it
  from contract_debt.py") falham o teste.
- **O que cada tool faz** (`contract_cases.py`, `KINDS`, todas as tools, um teste exige-o):
  `lookup` (um recurso: ponto `not_found`), `search` (uma lista: ponto `zero`) ou `other`. `WHOLE`:
  as 38 tools que nunca cortam (plano, anexo A), a que o ponto `window` não se aplica.
- **Os pontos:**
  - `window`: um `Case` (argumentos, respostas da fonte e ficheiros): o texto acaba num rodapé da
    janela da T03, e a chamada com os argumentos do `next:` devolve outra parte, numerada a partir
    de onde a primeira acabou. Uma fonte que pagina por cursor leva também a posição no `next:`.
  - `errors` (tools de módulos com `Api`): cada corpo de erro de `error_bodies.py` que se aplica à
    tool (estado, cabeçalhos, corpo, o URL da documentação, o tipo e o texto esperados) dá
    `ToolFailure` desse tipo, com esse texto. Sem corpo que se aplique, é dívida.
  - `not_found` (lookups): um caso cuja resposta dá `not_found`.
  - `zero` (searches): um caso de zero resultados dá texto que diz que nada se achou e contém a
    consulta (T00, ponto 2).
  - `limits` (automático): cada parâmetro inteiro tem um limite no schema (o `Range` da T04a), salvo
    os de `UNBOUNDED` com a razão; o intervalo da docstring, se o escreve, é o do schema; e nem a
    tool nem um helper do módulo a que ela o passe o aperta (`*clamp*(…)`, ou `min`/`max` com uma
    constante).
- **Cortes copiados:** `CUT_HELPERS` em `contract_debt.py` é o número de definições `_truncate` e
  `_trim` em `toolkit/tools`; o teste exige a igualdade, e quem apaga uma baixa o número.
- **Sementes:** corpos de erro já provados na T02 (família MediaWiki, Eurostat, UniProt, …) e os
  `not_found` dos lookups com `missing=`. O resto fica na dívida, para as T05 a T09.

## Fora do âmbito

Migrar as tools (T05 a T09).

## Registo do dono

- **Estado:** T04a done — revista no branch `feat/tools-contract-wave1` e aplicada em `main` pelo
  PR #71 (`84c0ee2`, 2026-09-30); T04b done (Claude, 2026-10-05), pela nota de desenho.

### T04a · Nota de desenho

- **Interface:** `Range(minimum=None, maximum=None)`, dataclass `frozen`/`slots` em
  `core/_tools/_schema.py`, pública em `ai_arch_toolkit.core` e no topo. Uso:
  `max_results: Annotated[int, Range(1, 25)] = 10`. Recusa com `ValueError`, ao construir, um
  `Range` sem limites, com limites que não sejam números finitos (`bool` incluído) ou com
  `minimum > maximum`.
- **Schema:** o `infer_schema` lê as anotações com `include_extras=True`; o `_hint_to_json_schema`
  trata `Annotated`: o schema do tipo de base, mais `minimum`/`maximum` de um `Range`. Um `Range`
  sobre um tipo sem números (nem `integer`, nem `number`, nem um ramo `anyOf` numérico) levanta
  `ValueError` ao decorar, e não cai no `string` de recurso, que só apanha `TypeError`. Outros
  metadados de `Annotated` ignoram-se, como antes.
- **Validador:** depois da coerção e do `enum`, um número fora de `minimum`/`maximum` dá
  `ArgumentError` ("argument 'max_results': expected integer from 1 to 25, got int 40"), que o
  executor já transforma em `validation_error`. Vale também para o caminho `anyOf`.
- **Âmbito:** só os parâmetros de topo; os campos de dataclasses e `TypedDict` aninhados não mudam,
  porque o validador não desce a objectos. `minLength`/`maxLength` ficam de fora: nenhuma tool os
  pediu ainda.
- **O que desaparece:** nada nesta ficha; os `max(…, min(…))` das tools saem quando cada módulo
  adoptar o `Range` (T05 a T09).
- **Testes:** `tests/test_tools_schema.py` (limites, um só limite, `Optional`, tipo sem números,
  metadados alheios), `tests/test_tools_validation.py` (dentro, fora, string coercida para fora,
  `anyOf`), `tests/test_tools_executor.py` (o `validation_error` com o intervalo),
  `tests/test_core_exports.py` (`Range` público).

### T04a · Registo

- **Feito:** `Range` e a leitura de `Annotated` no `_schema.py`; o limite no `_validation.py`;
  exports no `core._tools`, no `core` e no topo; docs em `docs/tools.md` (como declarar),
  `docs/safety.md` (tabela da validação) e `docs/api.md`; entrada `Added` no `CHANGELOG`.
- **Desvio da nota:** o teste do `validation_error` ficou em `tests/test_tools_validation.py`, que
  já corre pelo `ToolGroup` (sync e async); o `test_tools_executor.py` não mudou.
- **Estrutura:** para não subir a dívida, o `infer_schema` passou a usar helpers pequenos
  (`_type_hints`, `_annotated_hints`, `_in_schema`, `_parameter_schema`, `_range_of`,
  `_numeric`) e o `validate_arguments` também (`_check_required`, `_check_declared`, `_checked`).
  Saem da `tests/quality_baseline.json` três entradas: `infer_schema` (C901 14, PLR0912 15) e
  `validate_arguments` (C901 13). Também desapareceu um `# type: ignore[arg-type]` dos overrides.
- **Testes novos:** 23 casos: 13 em `test_tools_schema.py` (6 de limites inválidos,
  parametrizados), 7 em `test_tools_validation.py` (dois correm em sync e async), 3 em
  `test_core_exports.py`. Falharam antes por `ImportError: Range`.
- **Verificação:** 5665 passed, 42 deselected; ruff, formatação e pyright (0 erros, com
  `--pythonpath .venv/bin/python`, porque o pyright da worktree não acha o `.venv` sozinho) limpos;
  a baseline de qualidade passa.
- **Linhas:** `_schema.py` 456 → 543 (o `Range` e os helpers); `_validation.py` 315 → 351.

### T04a · Revisão (a pedido do dono, antes do push)

- **Consequência que faltava documentar:** o validador verifica `minimum`/`maximum` no topo de um
  parâmetro, venha de onde vier, e não só os que um `Range` pôs lá. Um `@tool(schema=...)` com
  limites, que antes só chegavam ao modelo, passa a ser aplicado. No repositório nenhuma tool os
  tem. Ficou escrito no `CHANGELOG`, em `docs/tools.md`, em `docs/safety.md` e na docstring do
  `_validation.py`, e há um teste.
- **Limite do âmbito, documentado:** limites dentro de um ramo `anyOf` escrito à mão não são
  verificados; só os do topo.
- **Testes novos:** 2: um `Range` num membro de uma união (`Annotated[int, Range(1, 5)] | None`), um
  caminho do `_range_of` que não estava coberto, e um `schema=` com `minimum`.

### T04b · Registo

- **Feito:** `tests/toolkit/test_tool_contract.py`, `contract_cases.py`, `error_bodies.py`,
  `contract_debt.py` e `tool_catalog.py` (o partilhado com a invariante); o `sandbox` no
  `tests/conftest.py`; o passo 8 do `CONTRIBUTING.md` e a secção de testes do `AGENTS.md` (e o
  passo 6 do `CONTRIBUTING.md`, que ainda mandava apanhar um 404).
- **Tamanho inicial da dívida** (134 tools): 121 com dívida; `window` 96, `errors` 91, `limits`
  70, `zero` 42, `not_found` 22. `CUT_HELPERS` 15.
- **Sementes:** os corpos de erro da família MediaWiki (`readonly`, `ratelimited`), do Eurostat
  (413), da UniProt (400) e do Open-Meteo (400); os `not_found` dos lookups com `missing=` (T02),
  o `missingtitle` da MediaWiki e os ficheiros que não existem.
- **Prova:** com o corpo de erro do Open-Meteo acrescentado de propósito, o teste pediu
  "air_quality_current now keeps ['errors']: delete it from tests/toolkit/contract_debt.py" (e o
  mesmo para o `air_quality_forecast`); a semente ficou, e as linhas saíram. As próprias
  verificações têm testes com tools sintéticas, nos dois sentidos.
- **Revisão independente:** sem falhas graves. Corrigi as médias:
  - os apertos escondidos num helper, ou de um só lado;
  - zero resultados que não o dizem;
  - uma janela que repete o corpo;
  - dívida que não se podia pagar (o `distance_between`, os argumentos benignos do `who_series` e
    do `weather_units`, ficheiros nos casos);
  - um 429 com `Retry-After` que fazia descansar o host nas verificações seguintes (um throttle
    novo por verificação);
  - o `sandbox` num `conftest` de pasta.
  Corrigi também as baixas: um rodapé estranho, casos que nunca correm, um nome estragado na
  extracção.
- **Para a T09:** as `youtube_*` não passam pelo `_http`; os seus pontos esperam uma costura
  própria.
- Gate: 7045 passed, 42 skipped; ruff, formatação e pyright limpos.

