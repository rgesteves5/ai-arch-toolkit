# T03 · A janela: nenhum corte é beco sem saída

- **Dono:** por atribuir · **Estado:** todo · **Depende de:** nada; aplicar depois da T01 (ambas
  mexem no `_executor.py`)
- **Origem:** plano, secção 3.3 e anexo A · **Decisões:** D39 · **Regras:** `T00-rules.md`

## Problema

- 94 das 132 tools cortam alguma coisa, e 60 não deixam ler o resto.
- Há cerca de 72 cortes `[:N]` silenciosos. O `_truncate` está copiado em 12 módulos e o `_trim` em
  3, e nenhum diz o tamanho original.
- O `_bounded` do executor diz "kept X of Y", mas não diz como continuar e pode cortar a meio de uma
  linha.

## Objectivo

Uma só primitiva para todo o corte, com um rodapé que diz o que foi mostrado, o total e a chamada
exacta para o resto, e uma `metadata` que a aplicação pode ler. Esta ficha cria a primitiva e fixa o
formato. As tools adoptam-na nas fichas T05 a T09.

## Desenho (confirma-o na nota de desenho)

- **`toolkit/tools/_window.py`:**
  - **Texto:** uma janela por caracteres (`offset`, `limit`) que, quando pode, não corta a meio de
    uma linha; e um `find`, que devolve a janela à volta de cada ocorrência, com a posição.
  - **Listas:** uma página por `offset` ou pelo `cursor` da fonte, com o total quando se sabe e
    "more available" quando não.
  - **Rodapé:** um só formato, em inglês, que nomeia o parâmetro e o valor da chamada seguinte. Por
    exemplo: `[chars 0–4000 of 34,651 · next: offset=4000]` ou `[results 1–20 of 1,234 · next:
    offset=20]`.
  - **`metadata`:** `{"truncated": …, "total": …, "next": {…}}`. Decide na nota de desenho como a
    tool a entrega ao executor; o `_coerce_result` já aceita um `ToolResult.success(text,
    metadata=…)` devolvido pela tool.
- **Executor:** o `_bounded` passa a usar o vocabulário do rodapé (sem continuação, que não conhece)
  e a cortar numa fronteira de linha.
- **Tectos:** ficam os actuais (D39). Nas 72 chamadas registadas no ai-network, o agente pediu sempre
  o máximo ou perto dele.

## Passos

1. Nota de desenho, com o formato exacto do rodapé para texto, listas com total, listas sem total e
   `find`.
2. `_window.py` e os seus testes, escritos primeiro.
3. `_bounded` no executor.
4. Docs: `CONTRIBUTING.md` (como uma tool corta) e `docs/tools.md` (o que o agente vê). `CHANGELOG`
   quando o texto visível do `_bounded` mudar.

## Ficheiros

`toolkit/tools/_window.py` (novo), `core/_tools/_executor.py` (só o `_bounded`),
`tests/toolkit/test_window.py` (novo), `tests/test_tool_limits.py`, `CONTRIBUTING.md`,
`docs/tools.md`.

## Prova

- Seguir os rodapés de janela em janela reconstrói o texto inteiro, sem perdas nem repetições.
- O `find` devolve todas as ocorrências, por ordem, e diz quantas são.
- O total do rodapé é exacto; uma lista sem total diz "more available" só quando há mais.
- O `_bounded` corta numa fronteira de linha e usa o vocabulário novo.

## Fora do âmbito

Migrar tools para a janela (T05 a T09) e apagar os helpers copiados, que desaparecem com cada módulo.

## Registo do dono

- **Estado:** todo.
