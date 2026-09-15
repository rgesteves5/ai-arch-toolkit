# C02 · API pública para tools dinâmicas

- **Dono:** por atribuir · **Estado:** todo · **Depende de:** nada
- **Origem:** `agentes-app-toolkit-review.md` §2.6, L1 (último parágrafo), "Dívida de desenho" 15;
  `toolkit-fix-plan.md` §4.10; `docs/internal/core_audit.md` SILENT-5
- **Decisões:** por fixar (ver abaixo)

## Problema

Não há forma pública de criar uma tool governada a partir de nome, descrição, JSON Schema completo e
política, sem assinatura Python. O MCP (C03), integrações tipo OpenAPI e tools da app precisam dela.

- `@tool(schema=)` funde overrides por parâmetro (`core/_tools/_decorator.py:53`, `_schema.py:422-431`),
  mas `docs/tools.md:53` chama-lhe "override the inferred JSON Schema". `ToolGroup.add()` só lê
  `fn.__tool_definition__` (`_group.py:64-88`, `_executor.py:63-72`); prendê-lo à mão não é API.
- O executor liga os argumentos à assinatura e chama `fn(**nomeados)` (`_executor.py:95`, `108-113`;
  `_validation.py:97-132`). A D7 só coage na raiz, com `type` string e `anyOf` (`_validation.py:143-173`);
  chaves desconhecidas decidem-se pela assinatura, não pelo schema (`_validation.py:60-67`).
- Nomes não são validados; duplicados avisam e sobrescrevem (`_group.py:83-88`).
- Um grupo vazio é falso (`_group.py:148-149`): o `Agent` troca-o por outro (`agents/_agent.py:56`) e
  um `executor_tools`, `solver_tools` ou `rollout_tools` vazio cai nas tools principais (`_lats.py:114`,
  `_plan_execute.py:60`, `_reflexion.py:63`, `_llm_compiler.py:97`, `_self_discovery.py:85`).
- Os adaptadores enviam o schema como vem (`_anthropic.py:128-134`; `_openai.py:114-123`, sem `strict`;
  `_xai.py:150-157`; `_meta.py:250-257`); o Gemini cai para `parameters_json_schema` com `$defs`,
  `oneOf` ou `type` em lista (`_gemini.py:227-245`, verificado). O strict do F23 (`_openai.py:196-273`)
  só se aplica a `output_schema`.

Repro (definição presa à mão a `async def call(**arguments)`):

```
{"count": "3"}, count = {"$ref": "#/$defs/Count"}     -> ok; a função recebe "3"
{"mode": "7"} (oneOf), {"limit": "5"} (["integer", "null"]), {"x": 1} (additionalProperties: false) -> passam intactos
handler(arguments: dict)                              -> validation_error: argument mismatch: missing a required argument: 'arguments'
@tool(schema={"type": "object", "properties": {...}}) -> properties = {"type": ..., "properties": ...}
Agent(spec, llm, ToolGroup(approval_handler=h)); group.add(ping) -> Tool error [unknown_tool]: Unknown tool: 'ping'
plan_execute, deps={"executor_tools": ToolGroup()}    -> executor liga ToolGroup(delete_everything)
```

Fontes: `platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools` (`^[a-zA-Z0-9_-]{1,128}$`);
`openai` 2.45.0 `FunctionDefinition.name` (`a-z A-Z 0-9 _ -`, máx. 64); `google-genai` 2.11.0
`FunctionDeclaration.name` (começa por letra ou `_`, aceita `. : -`, máx. 128); xAI sem regra;
`modelcontextprotocol.io/specification/2026-07-28/server/tools` (nomes `[A-Za-z0-9_.-]{1,128}`,
`inputSchema` com `type: "object"`) e `.../basic/index` (nunca desreferenciar `$ref` de rede).

## Objectivo

Uma factory pública cria, a partir de um JSON Schema completo e de um handler que recebe um `dict`,
uma tool que passa pelo mesmo executor governado que `@tool` (validação, gates, aprovação, `max_calls`,
metering) e chega aos fornecedores pelos mesmos adaptadores. O `ToolGroup` muda em execução sem
sombrear tools em silêncio, e um grupo vazio chega ao agente com a sua identidade.

## API proposta

```python
async def search(arguments: dict[str, Any]) -> str | ToolResult: ...   # ou síncrona

search_tool = tool_from_schema(      # -> callable com __tool_definition__, como @tool
    search, name="github_search", description="Search issues.",
    input_schema={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
    policy=ToolRuntimePolicy(capability="github", risk_level="high", requires_approval=True),
)
group = ToolGroup(search_tool, approval_handler=ask)
group.add(new_version, replace=True)  # sem replace, nome de outra definição -> ValueError
group.remove("github_search")         # -> ToolDefinition; KeyError se não existir
```

- `ValueError` na criação: nome fora de `^[A-Za-z_][A-Za-z0-9_-]{0,63}$`, raiz sem `type: "object"`,
  não serializável em JSON, `$ref` não local, expansão acima do orçamento. Guarda uma cópia com os
  `$ref` locais não recursivos embutidos (`_inline_local_refs`, `_schema.py:138-170`, com orçamento).
- O callable tem assinatura `(**arguments)` e chama `handler(dict(arguments))`, sem `__wrapped__`
  (`inspect.signature` seguiria o handler). `async def` corre no loop, síncrono numa thread; um
  `ToolResult` passa intacto; uma excepção vira `runtime_error` redigido. Governança, metering e caminho
  síncrono não mudam (`_executor.py:255-365`; cobrança só em `_meter_tool_open`).
- Validação (propõe D15): `oneOf` e `type` em lista coagem como `anyOf`; `additionalProperties: false`
  na raiz recusa chaves desconhecidas mesmo com `**kwargs`; aninhados intactos (D7).
- `ToolSchema.__post_init__` aplica a regra de nomes (as 133 tools decoradas do pacote cumprem; uma
  `lambda` passa a `ValueError`); `@tool(schema=)` com valor que não é mapping → `TypeError`.
- `ToolGroup` com cópia na escrita: a mudança vale na próxima leitura; uma chamada em curso acaba com a
  definição que tinha; `max_calls` não reinicia. O ReAct lê o grupo em cada volta (`_react.py:80-86`);
  `{tools}` e o mapa do ReWOO ficam como na compilação (`_rewoo.py:72-75`, `_plan_execute.py:65`).

## Decisões a fixar antes de codificar

1. **Forma.** (a) factory que devolve callable; (b) `ToolGroup.add_definition()`; (c)
   `@tool(input_schema=)`. Recomendo (a), porque `ToolGroup`, `prepare_tools`, `run_tools` e
   `execute_tool` já aceitam callables com definição; (b) mudaria os quatro e (c) não tem consumidor.
2. **Handler.** (a) um `dict`; (b) `**kwargs`. Recomendo (a), porque chaves como `x-api-version`,
   `from` ou `self` deixam de importar e é a forma de `call_tool(name, arguments)` do MCP.
3. **Validação.** (a) D7 como está; (b) D7 estendida, stdlib; (c) `jsonschema` opcional. Recomendo (b),
   porque cobre as strings numéricas dos modelos locais sem dependências e o servidor valida o resto
   (spec MCP: "Servers MUST validate all tool inputs").
4. **Nomes.** (a) regra portátil em `ToolSchema`; (b) só na factory; (c) em cada adaptador. Recomendo
   (a): falha cedo num só ponto, e `LLM(fallback=)` pode mudar de fornecedor a meio de um run.
5. **Colisões.** (a) aviso e sobrescrita; (b) `ValueError` salvo `replace=True`, mesmo objecto no-op.
   Recomendo (b), porque uma tool remota não pode sombrear uma local sem ninguém saber.
6. **Grupo vazio.** Recomendo `is None` em vez de `or`, porque um grupo que começa vazio (MCP) tem de
   chegar ao agente e um grupo de fase vazio quer dizer "sem tools".
7. **Chaves extra** (`title`, `$schema`, `x-mcp-header`). (a) enviar como vêm; (b) limpar. Recomendo
   (a) com prova ao vivo local (C02f), porque limpar muda o schema que a app mostra e valida.

As decisões 4–6 são quebras visíveis: entrada `Changed` e aviso antes de actualizar a app.

## Sub-tarefas, por ordem

- **C02a** Grupo vazio mantém a identidade no `Agent` e nas cinco estratégias.
- **C02b** Regra de nomes em `ToolSchema`; `TypeError` em `@tool(schema=)`; corrigir `docs/tools.md:53`.
- **C02c** `ToolGroup.add(replace=)`, `remove()`, colisões, cópia na escrita.
- **C02d** Validação D15. **C02e** `tool_from_schema` com normalização e orçamento; exports; docs.
- **C02f** (só local, `live_api`) schema com a forma MCP (`$defs`/`$ref`, `title`, `default`,
  `additionalProperties`, `oneOf`) aceite e chamado por Anthropic, OpenAI, Gemini, xAI e Meta.

## Ficheiros

- `src/ai_arch_toolkit/core/_tools/_dynamic.py` (novo), `_schema.py`, `_validation.py`, `_group.py`;
  `_definition.py`, `_decorator.py` (partilhados com C07)
- Exports: `core/_tools/__init__.py`, `core/__init__.py`, `ai_arch_toolkit/__init__.py` (partilhados com
  C01, C04, C06)
- `toolkit/agents/_agent.py` (partilhado com C01, C04); `toolkit/agents/flows/_plan_execute.py`,
  `_reflexion.py`, `_llm_compiler.py`, `_self_discovery.py` (partilhados com C01, C05), `_lats.py`
  (partilhado com C01, C04, C05)
- Testes: `tests/test_tools_dynamic.py` (novo), `test_tools_validation.py`, `test_core_exports.py`
  (partilhado com C04, C06), `test_tools_decorator.py`, `test_tools_group.py` (partilhados com C07),
  `tests/agents/test_configurable.py`, `tests/agents/test_phase_overrides.py`,
  `tests/integration/test_provider_contracts_live.py`
- Docs: `docs/tools.md` (partilhado com C03–C05, C07, C08), `docs/safety.md` (partilhado com C04, C07,
  C08), `docs/agents.md` (partilhado com C01, C04), `docs/api.md` (partilhado); D15 pelo coordenador

## Prova

- `test_tools_dynamic.py`: o handler recebe `{"x-api-version": "2", "from": 1}`; um síncrono corre fora
  do loop; `{"count": "3"}` com `$ref` local chega como `3` e o approval handler vê `3`; sem handler,
  `approval_denied` e o handler não corre; `MeterScope` conta 1 executada e 0 bloqueadas; `ValueError`
  para raiz sem `object`, `$ref` `https://…`, NaN, nomes `a.b`/`1a`/65 caracteres e a bomba de refs
  (< 1 s); o schema chega igual aos `_tool_to_sdk` de Anthropic, OpenAI e Gemini.
- `test_tools_validation.py`: coerção em `oneOf` e `["integer", "null"]`; `additionalProperties: false`
  com `**kwargs` → `validation_error` a nomear a chave. `test_tools_decorator.py`: `@tool(schema=<schema
  completo>)` → `TypeError`; `name="a.b"` → `ValueError`.
- `test_tools_group.py`: nome repetido → `ValueError`; `replace=True` troca; mesmo objecto no-op; depois
  de `remove`, `unknown_tool`; chamada bloqueada num `asyncio.Event` termina `ok` após `remove`.
- `agents/test_configurable.py`: repro do grupo vazio → a tool corre. `agents/test_phase_overrides.py`:
  grupo de fase vazio → a fase fica sem tools e `{tools}` rende `(none)`, nas cinco estratégias.

## Fora do âmbito

- `agent_as_tool` (a delegação é uma tool da app); `@tool(input_schema=)`; grupos com várias fontes.
- Validação completa ou aninhada (`minimum`, `pattern`, objectos dentro de objectos).
- `{tools}` avaliado em cada run; resultados multimodais; `strict` em tools; importador OpenAPI.
- Manifestos: `tools.factory`/`tools.manifest` continuam strings da app (`_manifest.py:97`, `517-521`).

## Riscos

- `$defs`/`$ref`, `title`, `$schema` e `x-*` em tools não foram verificados ao vivo em nenhum
  fornecedor (só o cliente Gemini, F19); a Meta recebe tools sem `strict` e o M01 só testou schemas
  inferidos. C02f cobre.
- `_inline_local_refs` é exponencial: 1,9 KB com 18 níveis de refs duplas → 17,5 MB em 1,36 s. Sem
  orçamento, um servidor MCP hostil bloqueia o processo. `oneOf` coage pelo primeiro ramo, como `anyOf`.

## Registo do dono

- Estado: todo
- Ficheiros tocados:
- Testes novos:
- Verificações:
- CHANGELOG proposto:
- Desvios ao plano:
