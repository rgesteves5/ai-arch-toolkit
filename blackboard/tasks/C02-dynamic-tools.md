# C02 · API pública para tools dinâmicas

- **Dono:** Claude (um agente: C02b a C02f, 2026-10-08) · **Estado:** done · **Depende de:** nada
- **Origem:** `agentes-app-toolkit-review.md` §2.6, L1 (último parágrafo), "Dívida de desenho" 15;
  `toolkit-fix-plan.md` §4.10; `docs/internal/core_audit.md` SILENT-5
- **Decisões:** fixadas pelo dono a 2026-10-08, D62
- **Actualização (2026-09-30):** a C02a (grupo vazio mantém a identidade) saiu na F24, e com ela a
  decisão 6. A validação (C02d) leva o número de decisão que o coordenador der, a partir de D43. As
  referências a ficheiros e linhas são de 2026-09-15, anteriores às frentes R e T: relê-as contra o
  `main` antes de começar.

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
- Validação (propõe uma decisão nova): `oneOf` e `type` em lista coagem como `anyOf`;
  `additionalProperties: false` na raiz recusa chaves desconhecidas mesmo com `**kwargs`; aninhados
  intactos (D7).
- `ToolSchema.__post_init__` aplica a regra de nomes (as 133 tools decoradas do pacote cumprem; uma
  `lambda` passa a `ValueError`); `@tool(schema=)` com valor que não é mapping → `TypeError`.
- `ToolGroup` com cópia na escrita: a mudança vale na próxima leitura; uma chamada em curso acaba com a
  definição que tinha; `max_calls` não reinicia. O ReAct lê o grupo em cada volta (`_react.py:80-86`);
  `{tools}` e o mapa do ReWOO ficam como na compilação (`_rewoo.py:72-75`, `_plan_execute.py:65`).

## Decisões a fixar antes de codificar

**Fixadas a 2026-10-08 (D62), todas na opção recomendada; a lista abaixo fica como estava.**

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
- **C02b** Regra de nomes em `ToolSchema`; `TypeError` em `@tool(schema=)`; corrigir `docs/tools.md`.
- **C02c** `ToolGroup.add(replace=)`, `remove()`, colisões, cópia na escrita.
- **C02d** Validação (a decisão nova). **C02e** `tool_from_schema` com normalização e orçamento;
  exports; docs.
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
- Docs: `docs/tools.md` (partilhado com C03–C05, C07, C08), `docs/safety.md` (partilhado com C04,
  C07, C08), `docs/agents.md` (partilhado com C01, C04), `docs/api.md` (partilhado); a decisão nova
  pelo coordenador

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

- **Estado:** done. Feito por agentes em worktrees, juntado e revisto pelo coordenador; uma revisão independente por parte, e as correcções dela (secção "Correcções da revisão").
- **Gate final, no checkout principal com todas as fichas da vaga:** 8195 passed, 118 skipped (os ao vivo e os 59 do nanope à espera); ruff, formatação, pyright e `uv lock --check` limpos (2026-10-08).
- **CHANGELOG:** as linhas entraram em `[Unreleased]` (Upgrade notes, Added, Changed, Fixed).

- Worktree: `/Users/rge/Documents/dev/pessoal/ai-arch-toolkit/ai-arch-toolkit/.claude/worktrees/agent-afc91aac2e6529bb9`
  (branch `worktree-agent-afc91aac2e6529bb9`, a partir de `601e38c`). Tudo por commitar.
- Estado: `review`. Feitas a C02b, C02c, C02d, C02e e a C02.8 (D62); a C02f está escrita e não
  correu. A C02a estava feita (F24) e foi só verificada.

### Nota de desenho

- **A factory** (`core/_tools/_dynamic.py`, novo):
  `tool_from_schema(handler, /, *, name, description="", input_schema, policy=None) -> Callable[..., object]`.
  - Devolve uma função `(**arguments)` que chama `handler(dict(arguments))`. Para um handler
    `async def`, a função é uma corrotina e corre no loop. Para um síncrono, é uma função simples,
    que o executor corre numa thread.
  - O `__tool_definition__` entra pelo `__dict__` da função, sem `# type: ignore`. Não há
    `__wrapped__`, e o `__name__`/`__qualname__` é o nome da tool.
  - O executor, os gates, a aprovação, o `max_calls` e o metering ficam como estavam. Uma
    excepção passa pelo mesmo `_result_from_exception`: o `ToolFailure` guarda o tipo, e outra
    qualquer dá um `runtime_error` redigido.
  - Na assinatura pública não há `Any`: o handler recebe `dict[str, object]`.
- **A normalização** (`_standalone`):
  - uma ida e volta por JSON, que faz a cópia e recusa `NaN` e sets;
  - uma volta sem recursão, que recusa um `$ref` não local (o que não começa por `#`) e um
    aninhamento acima de 100 níveis;
  - o `_inline_local_refs`, com orçamento;
  - por fim, a raiz tem de ser `"type": "object"`.
  - Todas as falhas são `ValueError("input_schema of tool '…': …")`.
- **O orçamento** (`_schema.py`): a closure recursiva deu lugar a uma classe pequena,
  `_Inlining`, com o que ainda pode gastar.
  - O limite é de cerca de 1 000 000 de unidades, uma por carácter de JSON, e de 100 níveis (o
    mesmo número da D59).
  - Vale também para os modelos Pydantic de um `@tool`.
  - Medido: a bomba da ficha (18 níveis de refs duplas) levava 1,44 s e dava 17,3 MB; agora dá
    `ValueError` em 0,06 s. Com 30 níveis, também 0,06 s.
- **A regra de nomes:** `check_tool_name`, em `_definition.py`, é a casa única. Chamam-na o
  `ToolSchema.__post_init__` (que abrange o `@tool`, o `tool_schema`, os callables nus e a
  factory) e o `prepare_tools`, para os dicts.
- **Uma tool por nome (C02.8):** `one_per_name`, em `_definition.py`, serve o `prepare_tools` e o
  `_resolve_fn` do executor.
  - A identidade compara-se por igualdade, como no grupo (um método lido duas vezes é uma tool).
  - A mesma tool repetida conta uma vez e vai uma vez para o fornecedor.
  - O grupo mantém a sua verificação, com a mesma mensagem (`name_clash`).
- **O `ToolGroup`:**
  - `add(fn, *, replace=False)` e `remove(name) -> ToolDefinition`; o `remove` levanta
    `KeyError` se o nome não existir;
  - o `_defs` passou a `Mapping`, e por isso o tipo já não deixa editar a tabela no sítio. Cada
    escrita põe lá uma tabela nova (cópia na escrita);
  - um lock ao nível do módulo serializa os escritores. Um lock por instância partia o
    `copy.deepcopy` de um grupo, que no `HEAD` funciona, e há um teste que o guarda.
- **A validação (C02.3):**
  - `_branches` lê `anyOf`, depois `oneOf`, depois uma lista em `type` (cada membro com as outras
    chaves do schema);
  - `_declared` e `_closed` tratam a raiz fechada;
  - `_admits_none` e `_admits_null` decidem o `null` de um argumento que chega pelo `**kwargs`;
  - `_none_allowed` devolve `None` quando a assinatura não se lê, e salta os variádicos;
  - o `_coerce` dividiu-se em `_coerce_to_a_branch` e numa tabela, `_CONVERTERS`.
- **O que desapareceu:**
  - o `_resolve_fn` em duas passagens;
  - o ciclo de 13 ramos do `prepare_tools`, com o seu `hasattr` e o seu `# type: ignore`;
  - quatro cópias da compreensão de `anyOf` em `_validation.py`;
  - três entradas da dívida de complexidade.
- **Saldo de linhas** em `core/_tools/`: 2006 antes, 2434 depois (+428). Disso, 148 são do módulo
  novo, e o resto é sobretudo docstrings.

### Mudanças em ficheiros partilhados

- **Fora da lista que me deram:** `core/_tools/_executor.py`. O `_resolve_fn` é a única costura
  por onde passam o `execute_tool`, o `async_execute_tool` e o `run_tools` (através do
  `_require_known_tools`). Sem lhe mexer, a metade "antes de correr" da C02.8 ficava por fazer.
  - A mudança: um `_tool_name` novo; o `_resolve_fn` passa a usar o `one_per_name`; e as
    docstrings do `execute_tool` e do `async_execute_tool` ganham `Raises: ValueError`.
  - Nenhum agente da vaga mexe neste ficheiro. O coordenador decide se fica.
- `core/_tools/__init__.py`: o `prepare_tools` foi reescrito, passou a exportar
  `tool_from_schema` e a importar `_definition_for`.
- `core/__init__.py` e `ai_arch_toolkit/__init__.py`: só o export de `tool_from_schema`.
- `_definition.py`, `_decorator.py`, `_group.py`, `_schema.py` e `_validation.py`: como descrito
  na nota de desenho.
- `tests/quality_baseline.json`: só desceu. Saíram as entradas `prepare_tools:C901` (13),
  `prepare_tools:PLR0912` (14) e `_coerce:C901` (11). Os agentes das tools mexem noutras linhas
  do mesmo ficheiro.
- Testes:
  - novo: `tests/test_tools_dynamic.py`;
  - mexidos: `test_tools_validation.py`, `test_tools_decorator.py`, `test_tools_group.py`,
    `test_core_exports.py`, `tests/integration/test_provider_contracts_live.py`.
- Docs: `docs/tools.md`, `docs/safety.md`, `docs/agents.md` e `docs/api.md`.
- Não toquei em `toolkit/agents/*`, porque a C02a não precisava de nada.

### Prova

- **C02e** (`tests/test_tools_dynamic.py`, 43 testes):

  | Item da ficha | Teste |
  |---|---|
  | o handler recebe `{"x-api-version": "2", "from": 1}` | `test_the_handler_receives_the_arguments_as_one_dict` (sync e async) |
  | um síncrono corre fora do loop | `test_a_sync_handler_runs_in_a_thread_off_the_loop` |
  | `{"count": "3"}` com `$ref` local chega como `3`, e o approval handler vê `3` | `test_a_local_reference_is_inlined_and_its_value_coerced_before_approval` (sync e async) |
  | sem handler, `approval_denied`, e o handler não corre | `test_without_an_approval_handler_the_call_is_denied_and_the_handler_never_runs` |
  | o `MeterScope` conta 1 executada e 0 bloqueadas | `test_the_meter_counts_the_call_that_ran_and_not_the_one_denied` |
  | `ValueError` para uma raiz sem `object`, um `$ref` `https://…` e `NaN` | `test_a_schema_the_tool_cannot_send_is_a_value_error` (mais um ref para outro ficheiro e um set) |
  | `ValueError` para os nomes `a.b`, `1a` e 65 caracteres | `test_a_name_outside_the_portable_rule_is_a_value_error` |
  | a bomba de refs falha em menos de 1 s | `test_a_hostile_schema_fails_in_under_a_second` (refs duplas com 30 níveis, uma definição longa 50 vezes, uma cadeia de 5000, `anyOf` com 300 níveis) |
  | o schema chega igual a Anthropic, OpenAI e Gemini | `test_the_schema_reaches_{anthropic,openai,gemini}_unchanged`, pelo `prepare` de cada adaptador e com o `wire_log` |

  - Extra:
    - um handler async corre no loop, e também no caminho síncrono;
    - um `ToolResult` passa intacto; o `ToolFailure` guarda o tipo, e outra excepção dá
      `runtime_error`;
    - a assinatura é `(**arguments)` e não há `__wrapped__`;
    - a tool passa pelas quatro portas (`prepare_tools`, `execute_tool`, `async_execute_tool`,
      `run_tools`);
    - a tool guarda uma cópia do schema;
    - o schema MCP chega como veio;
    - um ref recursivo fica, com a tabela.
- **C02d** (`tests/test_tools_validation.py`, +13):
  - `test_one_of_coerces_like_any_of`;
  - `test_a_type_list_coerces_like_any_of` e `test_a_type_list_keeps_a_value_that_already_matches_a_member`;
  - `test_a_closed_root_refuses_an_unknown_key_even_with_kwargs`: o `validation_error` nomeia a
    chave, também em `details["argument"]`;
  - `test_a_closed_root_without_properties_takes_no_arguments` e
    `test_an_open_root_with_kwargs_lets_unknown_keys_through`;
  - `test_null_through_kwargs_passes_only_where_the_schema_admits_it`;
  - `test_a_keyword_declared_only_in_the_schema_admits_null_only_if_the_schema_does`, que mostra
    a mudança num `@tool`;
  - `test_nested_values_are_left_as_they_are`.
- **C02b** (`tests/test_tools_decorator.py`):
  - `TestSchemaOverrides`: um schema completo dá `TypeError`, que nomeia `'type'` e aponta o
    `tool_from_schema`; o que não é um mapping de mappings também dá `TypeError`;
  - `TestPortableNames`: `@tool(name="a.b")` e outros dão `ValueError`; o `ToolSchema` verifica o
    nome; uma `lambda` no `ToolGroup` e no `prepare_tools` dá `ValueError`, e o nome de um dict
    também; todas as tools do pacote têm um nome portável.
- **C02c** (`tests/test_tools_group.py`):
  - em `TestOneToolPerName` (que já existia): um nome repetido dá `ValueError`, e o mesmo objecto
    não muda nada;
  - em `TestChangingTheGroup`, o `replace=True` troca a tool, e também acrescenta uma que o grupo
    não tem;
  - depois do `remove`, a chamada dá `unknown_tool`; um nome ausente dá `KeyError`;
  - uma chamada parada num `asyncio.Event` acaba `ok` depois do `remove`;
  - o `max_calls` não reinicia;
  - as definições lidas antes da mudança ficam como estavam;
  - um grupo copiado muda por si;
  - 64 threads a acrescentar ao mesmo tempo não perdem nenhuma tool.
- **C02.8** (`tests/test_tools_group.py::TestOneToolPerNameAcrossListsAndGroups`):
  - o `prepare_tools` recusa um nome repetido numa lista, em dois grupos, ou entre um dict e uma
    tool;
  - a mesma tool vai uma vez;
  - o `execute_tool` recusa antes de correr;
  - o `run_tools` recusa uma função nua com o nome de uma tool, sem correr nada;
  - a mesma tool duas vezes na lista corre uma vez.
- **Exports:** `test_core_exports.py::test_tool_from_schema_is_exported`, nos três módulos.
- **C02a:** feita na F24, com os testes em `tests/agents/test_empty_tool_groups.py` (verdes). A
  ficha cita `test_configurable.py` e `test_phase_overrides.py`, mas a F24 pô-los noutro ficheiro.
  Verifiquei ainda com um script, sem teste novo: com um grupo de fase vazio, as cinco
  estratégias não oferecem tools, e o `{tools}` rende `(none)` no `plan_execute` e no
  `llm_compiler`, as que têm o token no planner.
- Antes de corrigir, cada teste novo falhou pela razão certa:
  - o `ImportError` de `tool_from_schema`;
  - `DID NOT RAISE` (16 `ValueError` e 5 `TypeError`);
  - o `AttributeError` do `remove` e o `TypeError` do `replace`;
  - a lista com nomes repetidos;
  - na validação, os valores por coagir (`'7'`, `'5'`) e as chaves e os `null` que passavam.
  - A excepção é o teste das 64 threads: é uma guarda, e já passava antes.

### Ao vivo, pelo dono (C02f)

```bash
set -a && source .env && set +a
uv run pytest -m live_api tests/integration/test_provider_contracts_live.py \
  -k "mcp_shaped or recursive_reference or not_python_names" -v
```

- São 15 chamadas, uma por teste e por fornecedor, com `max_tokens=1024`. Os modelos:
  - Anthropic: `claude-haiku-4-5`;
  - OpenAI: `gpt-4.1-mini`;
  - Gemini: `gemini-2.5-flash`;
  - xAI: `grok-4.3`;
  - Meta: `muse-spark-1.3`, com `temperature=1.0`.
- `test_an_mcp_shaped_schema_is_accepted_and_called`: um schema com `$schema`, `title`,
  `default`, `additionalProperties: false`, `oneOf`, `x-mcp-header` e um `$ref` local (que vai
  embutido).
- `test_a_recursive_reference_with_its_table_is_accepted_and_called`: o `$defs` e o `$ref` chegam
  ao fornecedor.
- `test_argument_names_that_are_not_python_names_are_accepted_and_called`: as chaves `from` e
  `x-request-id`. No Gemini este schema vai por `parameters`, cujo nome de parâmetro o
  `google-genai` documenta como `[A-Za-z_][A-Za-z0-9_]{0,63}`: é provável que o Gemini o recuse
  aqui.
- Pela D62 (C02.7), cada recusa vira regra no adaptador desse fornecedor, com a fonte.

### Bloqueios e achados

- **`_executor.py` fora da lista:** ver "Mudanças em ficheiros partilhados".
- **Docstring desactualizada:** a do `run_tools`/`run_tools_sync` (`toolkit/_runner.py`) diz
  `Raises: KeyError…`, mas agora também levanta `ValueError` (C02.8). O ficheiro está fora do
  âmbito, e não lhe mexi. A linha proposta: "ValueError: Two different tools in ``tools`` share a
  name; no call runs."
- **Os nomes das tools:** nenhuma tool do pacote quebra a regra, e um teste guarda-o
  (`test_every_tool_the_package_ships_has_a_portable_name`, com mais de 100 tools). Se uma tool
  nova dos agentes das tools tiver um nome que não é portável, falha ao importar.
- **Fontes da regra de nomes:**
  - confirmadas no SDK instalado: a do `openai` (`FunctionDefinition.name`: a-z, A-Z, 0-9, `_`,
    `-`, até 64) e a do `google-genai` (`FunctionDeclaration.name`: começa por letra ou `_`,
    aceita `. : -`, até 128);
  - a da Anthropic (`{1,128}`) vem da ficha, porque a docstring do SDK não a diz. A intersecção
    dá o mesmo, porque o limite de 64 vem da OpenAI.
- **Refs draft-07:** um `$ref` para `#/definitions/...` (o draft-07, que alguns servidores MCP
  ainda mandam) é local, por isso passa, mas não é embutido: só o `#/$defs/` o é. Fica para a C03
  decidir, se os servidores reais o mandarem.
- **`@tool(name="")`:** um nome vazio continua a cair para o nome da função (o `name or …` que
  já existia). Não mudei isto.
- **Uma decisão minha:** o lock dos escritores do `ToolGroup` é ao nível do módulo, para não
  partir o `copy.deepcopy`. Um `State` faz deep copy dos seus valores, e isto funcionava no
  `HEAD`.
- **Fora do âmbito:** o `strict_schema` (`_providers/_base.py`) também embute refs, com
  `deepcopy` e sem orçamento. Só serve `output_schema`, que vem da app e não de um servidor
  remoto, por isso o risco é baixo; fica registado para quando a C03 tratar dos output schemas
  do MCP.
- **Divisão proposta em commits:**
  1. `feat(tools): portable tool names and one tool per name` (C02b, C02.8);
  2. `feat(tools): ToolGroup.add(replace=) and remove(), copy-on-write` (C02c);
  3. `feat(tools): oneOf, type lists and a closed root in argument validation` (C02d);
  4. `feat(tools): tool_from_schema, with bounded $ref inlining` (C02e, exports, docs);
  5. `test(integration): MCP-shaped schemas live on five providers` (C02f).

### Correcções da revisão

Feitas no checkout principal, a 2026-10-08, sobre o trabalho por commitar. Cada teste novo
falhou primeiro pela razão certa: corri-os contra uma cópia de `HEAD` com o `C02.patch`
aplicado (`scratchpad/c02fix/prefix`, pelo `PYTHONPATH`), onde falharam 17. As excepções são as
guardas, que já passavam e o dizem abaixo.

#### 1. (Alta) Os ramos de `anyOf`/`oneOf` lêem-se com as chaves do pai

- **O que mudou** (`core/_tools/_validation.py`):
  - o `_branches` funde cada ramo com as outras chaves do pai (`{**pai_sem_anyOf_oneOf,
    **ramo}`; ganham as do ramo), como já fazia a lista em `type`. Um schema com `anyOf` e
    `oneOf` lê-se pelo `anyOf`;
  - os ramos achatam-se numa só lista de alternativas sem ramos (`_alternatives`, por um
    gerador, `_leaves`). O `_coerce`, o `_matches`, o `_admits_null`, o `_describe` e o
    `_describe_bounds` lêem essa lista, e a recursão em cada nível desapareceu. A ordem é a
    mesma (primeiro o valor que já cabe, depois a primeira alternativa que o converte);
  - fundir multiplica: uma lista em `type` sobre n ramos dá n × m alternativas. Por isso o
    percurso pára em 1 000 (`_ALTERNATIVES_LIMIT`). Acima disso o valor passa como veio, como
    num schema sem tipo, e o servidor valida-o. Cada alternativa conta uma vez por tipo e por
    `enum` (por identidade: os ramos de um pai partilham o `enum` dele), e assim um `enum` longo
    lê-se uma vez só;
  - um `anyOf` que não é uma lista (`"anyOf": 5`) já não dá `TypeError` na validação.
- **Testes** (`tests/test_tools_validation.py`):
  - `test_branches_without_a_type_keep_the_type_of_their_parameter`: `@tool` com `anyOf` e com
    `oneOf`, nos dois modos. `"5"` chega como `5`; `"abc"` e `[1]` dão `validation_error`.
    Antes falhava com `"str '5'"`;
  - `test_a_titled_enum_coerces_to_the_type_of_its_parent`: o enum com títulos do MCP num
    `tool_from_schema`. `"2"` chega como `2`. Antes chegava `'2'`;
  - `test_null_through_kwargs_is_refused_where_the_type_of_the_branches_parent_is`: o caso do
    `_admits_null`. Antes o `null` passava;
  - `test_a_value_is_walked_against_a_bounded_number_of_alternatives`: 3 001 × 3 000
    alternativas, em menos de 0,5 s. É uma guarda: antes passava (os ramos não se fundiam), e
    uma fusão sem limite levaria minutos.

#### 2. (Média) Um só nome para cada entrada, no `prepare_tools` e no executor

- **O que mudou:**
  - `_entry_name` (`core/_tools/_executor.py`) substitui o `_tool_name`. Devolve `None` para um
    `ServerTool`, um dict já na forma de envio, um `ToolGroup` e o resto que não é callable. Para
    um dict de tool, devolve o seu `"name"`; para uma tool, o nome da definição ou o da função;
  - o `_resolve_fn` dá nome às entradas com ele, e só resolve callables: uma chamada ao nome de
    um dict de tool continua a ser `unknown_tool`;
  - o `_wire_entries` do `prepare_tools` (`core/_tools/__init__.py`) dá nome às entradas com a
    mesma função. As tools de um grupo continuam com o nome da sua definição. Assim, uma lista
    que o `prepare_tools` aceita, o executor também aceita.
- **Testes** (`tests/test_tools_group.py::TestOneToolPerNameAcrossListsAndGroups`):
  - `test_a_list_llm_complete_takes_runs_without_a_false_clash`: uma lista com dois server
    tools, dois dicts na forma de envio e dois grupos passa pelo `prepare_tools`, pelo
    `execute_tool`, pelo `async_execute_tool`, pelo `run_tools_sync` e pelo `run_tools`. Antes
    falhava com `Two different tools are named 'ServerTool'`;
  - `test_the_executor_refuses_the_dict_and_tool_with_one_name_that_prepare_tools_refuses`.
    Antes o `execute_tool` corria o `search` (`DID NOT RAISE`);
  - `test_a_call_to_a_tool_dict_is_an_unknown_tool`: é uma guarda, e já passava.

#### 3. (Média) Um schema hostil dá sempre `ValueError`

- **O que mudou** (`core/_tools/_schema.py`, `_dynamic.py`):
  - seguir um `$ref` conta como um nível (`depth + 1`). Uma cadeia feita só de referências pára
    nos 100 níveis;
  - um `$ref` para uma definição que não é um objecto (`true`, `false`, uma lista) dá
    `ValueError`, que nomeia a referência. Antes dava `TypeError`;
  - no `_standalone`, um `RecursionError` passa a `ValueError`. É só um último recurso, porque o
    percurso já é limitado: serve para quem chama do fundo da sua própria pilha;
  - a docstring do `tool_from_schema` e o `docs/tools.md` dizem-no ("no other exception for a
    schema that is JSON").
- **Decisão:** a ordem que recebi pedia `ValueError` para qualquer alvo que não seja um dict, e
  foi o que fiz. O revisor lembrava que `true`/`false` são schemas válidos em 2020-12. Se um
  servidor MCP real os mandar, a C03 pode trocar a recusa por embutir `true` como `{}` e
  `false` como `{"not": {}}`.
- **Testes** (`tests/test_tools_dynamic.py`):
  - em `test_a_schema_the_tool_cannot_send_is_a_value_error`, três casos novos: uma definição
    booleana, uma lista, e uma raiz que aponta para `false`. Antes davam `TypeError`;
  - em `test_a_hostile_schema_fails_in_under_a_second`, dois casos novos:
    - `_ref_chain(2_000)`, que antes dava `RecursionError`;
    - `_doubling_any_of(12)`, que antes era aceite.

#### 4. (Baixa) Um orçamento mais apertado e uma recusa curta

- **Os números:**
  - `_INLINE_SIZE_LIMIT = 100_000`. São cerca de 25 000 tokens em cada pedido, sessenta vezes o
    maior schema das 123 tools do pacote (1 688 caracteres, `earthquake_search`, medido hoje).
    A justificação está no comentário. Vale também para os modelos Pydantic de um `@tool`;
  - `_ALTERNATIVES_LIMIT = 1_000`: cobre um enum com títulos de algumas centenas de valores
    (países, moedas, fusos);
  - `_EXPECTED_LIMIT = 300`: o que se esperava mostra-se até 300 caracteres, e assim o "got …"
    aparece sempre;
  - `_MESSAGE_LIMIT = 1_000`: a mensagem inteira, cortada no `ArgumentError`, com cerca de 250
    tokens;
  - o `_describe` nomeia cada tipo uma vez ("integer", e não "integer or integer or …").
- **O que se mediu** (`scratchpad/c02fix/bench.py`):
  - o `anyOf` da revisão que duplica é recusado ao criar a partir de 11 níveis, em 7 ms;
  - com 10 níveis (1 024 alternativas) o valor passa como veio, em 1,7 ms;
  - com 9 níveis, a recusa leva 2 ms e tem 45 caracteres;
  - 1 000 ramos a partilhar um `enum` de 20 000 valores levam 2 ms; antes da deduplicação
    levavam 190 ms;
  - um enum de 20 000 valores dá uma mensagem de 339 caracteres; antes eram 69 000;
  - antes, o caso de 14 níveis levava 0,56 s e dava 180 KB.
- **Testes:**
  - `test_a_bad_call_against_many_alternatives_is_refused_quickly_and_briefly` (`tests/test_tools_dynamic.py`):
    em menos de 0,1 s, com a mensagem exacta. Antes era "integer or integer …";
  - `test_past_the_alternatives_the_walk_takes_a_value_passes_as_it_came` (`tests/test_tools_dynamic.py`):
    usa o maior schema que o orçamento deixa passar, e responde em menos de 0,1 s. Antes levava
    0,59 s;
  - `test_a_refusal_stays_brief_however_large_the_schema` (`tests/test_tools_validation.py`):
    fica abaixo de 1 100 caracteres e acaba em `got str 'nope'`. Antes tinha 68 954.

#### 5. (Baixa) As docstrings do `toolkit/_runner.py`

- O `run_tools` e o `run_tools_sync` ganham, em `Raises: ValueError`, o caso de duas tools com
  um nome (D62), em que nenhuma chamada corre. O `_require_known_tools` di-lo também.
- Só mudaram docstrings, por isso não há teste.

#### 6. (Baixa) `{tools}` rende `(none)` no `llm_compiler`

- `tests/agents/test_phase_overrides.py::TestToolsPlaceholder::test_an_empty_phase_group_renders_none`
  corre com o `plan_execute` e o `llm_compiler`. As tools principais são `ToolGroup(lookup)` e
  o `executor_tools` é `ToolGroup()`. O planner vê `Available tools:\n(none)` e nunca vê
  `lookup`.
- O comportamento existe desde a F24, e o teste passou à primeira. Para ver que o teste
  distingue os dois casos, corri uma cópia sem o `executor_tools`, que falhou nas duas
  estratégias e depois apaguei.

#### Verificações

- O comando dirigido (`tests/test_tools_dynamic.py`, `test_tools_validation.py`,
  `test_tools_group.py`, `test_tools_decorator.py`, `test_tools_schema.py`, `tests/agents`)
  deu **640 passed**.
- `uv run pytest -m "not live_api"` deu 8143 passed, 59 skipped e 9 failed.
  - Os 9 falhados estão todos em `tests/toolkit/test_eurostat.py`, que outro agente está a
    editar (`_eurostat.py` e o seu teste estão `MM`).
  - Não têm que ver com isto: são verificações dos próprios argumentos da tool, chamada
    directamente, sem o executor.
  - O `test_quality_budget.py` passa, porque nenhuma função nova excede os limites de
    complexidade.
- `uv run pyright src` deu 0 erros.
- `ruff check` e `ruff format --check` estão limpos nos ficheiros que mexi.
- **Ficheiros tocados:**
  - `core/_tools/_validation.py`, `_schema.py`, `_dynamic.py`, `_executor.py` e `__init__.py`;
  - `toolkit/_runner.py`, só nas docstrings;
  - `tests/test_tools_validation.py`, `test_tools_dynamic.py` e `test_tools_group.py`;
  - `tests/agents/test_phase_overrides.py`;
  - `docs/tools.md`, só nos parágrafos da C02: "Tools from a JSON Schema" e o parágrafo "One
    name, one tool".

#### Por fazer, fora do meu âmbito

- **`docs/safety.md` ("Argument validation"):** a tabela e o parágrafo seguinte deviam dizer
  três coisas:
  - um ramo lê-se com as chaves do pai;
  - acima de 1 000 alternativas, o valor passa como veio;
  - uma recusa tem no máximo cerca de 1 000 caracteres.
- **`const`:** não se verifica, e nunca se verificou. Num enum com títulos, um `"7"` passa a
  `7` mesmo sem ramo `const: 7`, e é o servidor que o recusa. Tratar `const` como um `enum` de
  um valor seria uma linha, mas é uma extensão da D7 que cabe ao dono decidir.

#### Mudanças às linhas propostas para o CHANGELOG

- **Added** (`tool_from_schema`): troca o fim da linha por
  "A schema that is not JSON, whose root is not an object, that refers outside itself or to a
  definition that is not an object schema (`true`, a list), that nests deeper than 100 levels, or
  whose references would inline to more than about 100,000 characters or 100 levels (each
  reference followed counts as one) raises `ValueError`, and nothing else does."
- **Changed** (validação): acrescenta, depois de "coerce like `anyOf`",
  "; a branch is read with its parent's keywords, so `{"type": "integer", "oneOf":
  [{"minimum": 1}, ...]}` and MCP's titled enums coerce `"5"`; a value whose branches flatten
  to more than 1,000 alternatives passes as it came; a refusal stays under about 1,000
  characters".
- **Changed** (uma tool por nome): acrescenta "Server tools, dicts in wire form and groups in
  the list never clash by their type's name: a list `llm.complete` takes, `execute_tool` and
  `run_tools` take too."
- **Fixed** (o orçamento): troca por
  "Inlining a schema's `$ref` references (a Pydantic parameter's, or a `tool_from_schema`
  schema's) is bounded at about 100,000 characters and 100 levels, a reference followed
  counting as one: a schema whose references multiply (1.9 KB that became 17.5 MB in 1.4 s), or
  a chain of 2,000 references alone, now raises `ValueError` in milliseconds."
