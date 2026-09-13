# F19 · Gemini: schemas que `types.Schema` rejeita

- **Dono:** coordenador · **Estado:** done · **Depende de:** F12 aplicado
- **Plano:** achado C1 do worker-C (`FINDINGS.md`)

## Problema

No `google-genai` 2.11.0, `FunctionDeclaration(parameters=...)` valida contra `types.Schema`
(subconjunto OpenAPI, `extra="forbid"`). `tuple[str, int]` (`prefixItems`) e schemas com
`$defs`/`$ref` rebentam no cliente com `pydantic.ValidationError` antes de qualquer pedido. Com o F12,
`tuple[int, int] | str` (antes `string`) também rebenta.

## Mudança

`_tool_to_sdk` tenta `parameters=`; se o SDK recusar (`ValueError`, de que o `ValidationError` do
pydantic é subclasse), usa `parameters_json_schema=` com o schema intacto. Os dois campos são
mutuamente exclusivos (docstring do SDK). Schemas que já funcionavam seguem o caminho de antes.

## Prova

- Schema dentro do subconjunto: `parameters` preenchido, `parameters_json_schema is None`.
- `def f(point: tuple[float, float])`: `parameters is None`, `prefixItems` no JSON Schema.
- Schema com `$defs`/`$ref`: vai por `parameters_json_schema`.
- `GeminiProvider.complete` com essa tool constrói o pedido sem rebentar.

## Registo

- Estado: done (2026-09-13)
- Ficheiros tocados: `core/_providers/_gemini.py` (`_tool_to_sdk`), `docs/tools.md` (nota a seguir à
  assinatura do `@tool`).
- Testes novos: 4 em `tests/test_gemini_provider.py::TestToolToSdk`. Antes: os 3 do recurso falhavam
  com `ValidationError`.
- Não verificado: se a API do Gemini aceita todos os JSON Schemas que o cliente agora deixa passar
  (sem teste `live_api`). O recurso só muda o que antes nem chegava a sair do cliente.
- CHANGELOG (Fixed): tools com parâmetros `tuple` ou schemas com `$defs`/`$ref` deixam de rebentar no
  adaptador Gemini; seguem como JSON Schema em `parameters_json_schema`.
