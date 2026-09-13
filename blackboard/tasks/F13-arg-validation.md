# F13 · Validar e coagir argumentos antes dos gates

- **Dono:** coordenador · **Estado:** done · **Depende de:** F07 e F12 aplicados
- **Plano:** M, F13 · **Decisões:** D7 (revisto)

## Problema

Só os nomes dos argumentos são verificados (via `TypeError` da chamada). `soma(a: int, b: int)` com
`{"a": "1", "b": "2"}` → `ok=True`, `"12"`. Importa sobretudo com modelos locais.

## Mudança

- Antes dos gates, validar e coagir contra `definition.schema.input_schema`; o `ExecutionContext`
  recebe um `ToolCall` com os valores coagidos. Depois de um `GateModify`, validar outra vez.
- Regras: `integer` (int; float inteiro; string `^[+-]?\d+$`; bool rejeitado); `number` (int/float;
  string numérica finita); `boolean` (bool; `"true"`/`"false"`); `string` (números → str; resto
  intacto); `enum` verificado depois da coerção; `anyOf` mantém o valor se já corresponder a uma
  variante e senão tenta coagir por ordem; `None` passa; `array`/`object`/sem tipo intactos.
- Obrigatório em falta e chave desconhecida sem `**kwargs` → `validation_error` com o nome.
- Validação falhada não gasta `max_calls` nem abre operação no meter.
- Docs: pipeline em `docs/safety.md`.

## Prova

- `{"a": "1", "b": "2"}` → 3; `{"a": "x"}` → `validation_error` a nomear `a`; chave desconhecida →
  lista as permitidas; `enum` fora do conjunto → erro.
- O `approval_handler` recebe `{"a": 1}` quando o modelo mandou `{"a": "1"}`; chamada inválida nunca
  chega ao handler; `modified_args` inválidos → `validation_error`.
- Validação falhada não conta para `max_calls` nem para o meter.

## Registo

- Estado: done (2026-09-13)
- Âmbito alargado com os achados: encadeamento de `GateModify` (worker-B), `TypeError` dentro da tool
  (worker-B), C2 (aliases PEP 695) e C3 (`**kwargs`, `anyOf`).
- Ficheiros tocados: novo `core/_tools/_validation.py` (`validate_arguments`, `bind_arguments`,
  `ArgumentError`); `core/_tools/_executor.py` (validação antes dos gates; cada gate recebe um
  `ExecutionContext` com os argumentos correntes e cada `GateModify` é revalidado; binding antes de
  `max_calls`/meter; `_split_arguments`/`_invoke` do worker-B substituídos por `bind_arguments`; um
  `TypeError` da tool passa a `runtime_error`); `core/_tools/_schema.py` (desembrulha
  `typing.TypeAliasType`); `docs/safety.md` (pipeline, tipos de erro, secção "Argument validation",
  parágrafo dos gates próprios).
- Desvio a D7: `string` não é coagido nem recusado — o gerador de schemas mapeia tipos desconhecidos e
  `Any` para `string` (C2), e recusar não-strings partiria chamadas válidas. `Any`/`object` continuam
  `string` para não arriscar o Gemini com propriedades sem tipo.
- Testes novos: `tests/test_tools_validation.py` (30, síncrono e assíncrono). Antes: 27 falhavam.
- Verificações: 2855 passed (antes do F15); com o F15, 2867 passed; ruff limpo; pyright 0 erros.
- CHANGELOG (Added): validação e coerção de argumentos contra o schema antes dos gates.
- CHANGELOG (Fixed): `GateModify` de um gate chega aos gates seguintes e ao pedido de aprovação; um
  `TypeError` levantado dentro de uma tool deixa de ser `validation_error`; aliases PEP 695 geram o
  schema do tipo de destino.
- CHANGELOG (Changed): argumentos de tipos incompatíveis passam a `validation_error` antes de a tool
  correr (antes chegavam à função, p.ex. `"1" + "2"`).
