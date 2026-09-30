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

- **Estado:** `review` no branch `feat/tools-contract-wave1` (Claude, 2026-09-30).

### Nota de desenho

- **`toolkit/tools/_window.py`**, interno às tools:
  - `Window`, dataclass `frozen`/`slots`/`kw_only`, com `body`, `unit` (`chars`, `results` ou
    `matches`), `first`, `last`, `total`, `next_call` (os argumentos da chamada seguinte, ou
    `None`) e `label` (o termo de um `find`). Métodos `footer()`, `text()` (corpo e rodapé) e
    `result()`, que devolve `ToolResult.success(text, metadata={"window": {...}})`. O
    `_coerce_result` do executor já aceita um `ToolResult` devolvido pela tool.
  - `text_window(text, offset=0, limit=…)`: a janela acaba numa quebra de linha quando há uma na
    segunda metade, para uma linha de tabela ou um parágrafo não ficar cortado; a janela seguinte
    começa exactamente onde esta acaba.
  - `find_window(text, needle, offset=0, limit=…, context=400)`: sem distinguir maiúsculas; cada
    bloco é feito de linhas inteiras à volta da ocorrência, com a posição à cabeça (`[at char
    13860]`); blocos que se tocam fundem-se; a continuação procura a partir do fim do último bloco.
  - `list_window(lines, first=1, total=None, next_call=None)`, para uma página que a fonte já
    cortou; `page_window(items, offset=0, limit=…, param="offset")`, para uma lista que a tool tem
    inteira.
- **Rodapé**, em ASCII para não haver dúvidas de tokens, com os números sem separadores, para o
  valor do rodapé ser o que o modelo passa:
  - `[chars 0-4000 of 34651 | next: offset=4000]` e, na última janela, `| end]`;
  - `[results 21-40 of 1234 | next: offset=40]`, `[results 1-20 | next: page_token="abc"]`;
  - `[matches 1-3 of 5 for "1960" | next: find="1960", offset=16500]`, `[no matches for "1960"]`;
  - os argumentos vão como `nome=valor`, com o valor em JSON.
  - Só aparece quando falta alguma coisa ou a janela não começa no início; o de um `find` aparece
    sempre, porque diz quantas ocorrências há.
- **`metadata`:** fica em `metadata["window"]` (`unit`, `first`, `last`, `total`, `next_call`), e
  não em `truncated`, que é a chave do corte do executor (um dict com `chars` e `kept`).
  Desvio consciente da ficha, que sugeria `truncated`/`total`/`next`.
- **Um só sítio sabe cortar numa linha:** `line_cut(text, start, end)` em `core/_tools/_result.py`,
  usado pelo `_bounded` do executor e pela janela; o core não importa o toolkit.
- **Executor:** o `_bounded` corta numa linha e usa o mesmo vocabulário:
  `[chars 0-200000 of 5000000 | cut at the output limit; ask for less]`.
- **O que desaparece:** nada nesta ficha; os `_truncate`/`_trim` saem com cada módulo (T05 a T09).
- **Testes:** `tests/toolkit/test_window.py` (reconstrução seguindo os rodapés, fim de linha, `find`
  com todas as ocorrências por ordem e contagem, blocos fundidos, listas com e sem total, página
  local, `result()` com a `metadata`) e `tests/test_tool_limits.py` (corte numa linha e o texto
  novo).

### Registo

- **Feito:** `toolkit/tools/_window.py` (novo); `line_cut` em `core/_tools/_result.py`; `_bounded`
  no `_executor.py`; `CONTRIBUTING.md` (o passo 5 do guia de tools, "Never cut without a way on",
  e os limites com `Range`); `docs/tools.md` (a nota de corte e o rodapé); entrada `Changed` no
  `CHANGELOG`.
- **Desvio da nota:** a nota do executor ficou mais curta do que a primeira versão ("cut at the
  output limit; ask for less"). Com 82 caracteres, um erro cortado passava o limite de 200 100 do
  `test_a_long_error_message_is_cut`; em vez de afrouxar o teste, encurtou-se o texto.
- **Testes novos:** 20 em `tests/toolkit/test_window.py`; em `tests/test_tool_limits.py`, um novo
  (o corte numa linha) e dois corrigidos (o texto da nota). Contra o código de antes, numa cópia
  em scratch: `test_window.py` falha no import (`_window` não existe) e os três de
  `test_tool_limits.py` falham no texto e no corte.
- **Verificação:** 5686 passed, 42 deselected; ruff, formatação e pyright limpos; a baseline de
  qualidade passa sem mudanças (as funções novas ficam abaixo de 10).
- **Linhas:** `_window.py` 0 → 241; `_result.py` 94 → 108; `_executor.py` 574 → 578.
- **Para as fichas seguintes:** uma tool devolve `window.result()`, que o `_coerce_result` do
  executor aceita; a chamada crua passa a devolver um `ToolResult` em vez de uma `str`, e os testes
  das tools lêem `result.value`.

### Revisão (a pedido do dono, antes do push)

- **Corrigido:**
  - **`find` sem tecto.** Num termo frequente, as passagens encostadas fundiam-se sem limite, e uma
    só passagem podia ser o documento inteiro. Agora uma passagem só cresce enquanto cabe no
    `limit`, o contexto fica em metade do `limit` no máximo, e uma ocorrência que já está à vista
    conta nessa passagem. Continua a valer que cada ocorrência conta uma vez e que a continuação
    começa no fim da última passagem.
  - **Termos com acentos.** O rodapé escapava-os (`"émile"`), e o modelo devolveria o termo
    errado. Os valores vão agora em JSON sem escapes: o enquadramento do rodapé é ASCII, e o termo
    vai como foi escrito.
  - **Termo vazio.** Contava uma ocorrência vazia por carácter; agora não encontra nada.
- **Testes novos:** 3 em `test_window.py`: um termo frequente dentro do limite e com cada
  ocorrência uma vez, o termo vazio e os acentos. Falham na versão de `4d5b7cd`, verificado numa
  cópia em scratch.
- **Fica como estava:** num erro cortado pelo executor, os números da nota contam o texto do
  modelo (prefixo `Tool error [tipo]: ` incluído), mas o corte é feito na mensagem; a diferença é o
  tamanho do prefixo. Já era assim antes.
- **Verificação:** 5691 passed, 42 deselected; ruff, formatação e pyright limpos.
