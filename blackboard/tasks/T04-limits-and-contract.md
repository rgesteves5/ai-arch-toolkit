# T04 · Limites na assinatura e invariante de contrato

- **Dono:** por atribuir · **Estado:** todo
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

## Fora do âmbito

Migrar as tools (T05 a T09).

## Registo do dono

- **Estado:** todo.
