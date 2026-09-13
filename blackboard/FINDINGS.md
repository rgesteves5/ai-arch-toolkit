# Achados

Só acrescentar. Cada entrada: o que é, como reproduzir, e a tarefa que o cobre (ou "sem tarefa").

## 2026-09-13 · coordenador

### N8 · `Policy(max_cost=)` por step não vê o gasto de flows aninhados → F08 (8g)

Um flow que herda o meter volta a ligá-lo ao span da raiz (`bind_meter(scope)` sem span), por isso
o span do step pai não contabiliza as chamadas do flow interno.

```
step que chama o LLM directamente   -> error='Cost exceeded limit 0.0001: cost 1.05'
step que corre um flow aninhado     -> error=None   (meter do run: 1.05)
```

### N9 · Campos de uso `null` na Anthropic rebentam o parse → F01

`_extract_usage` deixa passar `cache_creation_input_tokens=None`; `_parse_sdk_response` levanta
`TypeError` em `estimate_cost`, e `complete()` falha mesmo sem meter.

### N10 · Parâmetro positional-only torna a tool impossível de chamar → F07

`def square(x: int, /)` → schema `required=['x']`; execução →
`validation_error: got some positional-only arguments passed as keyword arguments: 'x'`.

### N3 (detalhe) · Negação a meio de uma vaga paralela → F08 (8f)

Se o step negado não for o primeiro da vaga, os irmãos anteriores entram no trace mas os seus
artefactos não entram no estado: trace e estado discordam.

## 2026-09-13 · worker-B

### `GateModify` de um gate anterior é descartado em tools com aprovação → sem tarefa (junto do F13)

O `ApprovalGate` corre sempre em último e devolve
`GateModify(args=decision.modified_args or dict(ctx.tool_call.input))`, ou seja, os argumentos
originais do modelo; o `ExecutionContext` é construído uma única vez, por isso nenhum gate vê as
modificações dos anteriores. Numa tool com `requires_approval=True`, um gate que estreita argumentos
é ignorado e o handler aprova os argumentos originais — precisamente nas tools perigosas.

```
gate ForceSafeTarget -> GateModify(target="staging"); handler aprova tudo
tool sem aprovação   -> 'deployed to staging'
tool com aprovação   -> 'deployed to prod'      (o handler viu {'target': 'prod'})
```

Proposta: encadear — cada gate recebe um `ExecutionContext` cujo `tool_call.input` são os argumentos
correntes (depois do último `GateModify`), e o `ApprovalGate` pede aprovação sobre esses.
`docs/safety.md` ("Custom gates", F17) descreve hoje o comportamento actual; mudar essa frase com a
correcção.

### `run_tools` levanta `KeyError` depois de já ter executado chamadas anteriores → sem tarefa

A verificação de tool desconhecida é feita chamada a chamada, logo antes de a executar. Numa
resposta `[send_email, does_not_exist]`, `send_email` corre (efeito lateral) e só depois sai o
`KeyError`; o `tool_result` da primeira perde-se. Igual com lista e com `ToolGroup` (o F04 manteve
a ordem).

```
list      -> KeyError "Unknown tool: 'does_not_exist'" | efeitos já feitos: ['a@example.com']
ToolGroup -> KeyError "Unknown tool: 'does_not_exist'" | efeitos já feitos: ['a@example.com']
```

Proposta: verificar todos os nomes da resposta antes de executar qualquer chamada.

### `TypeError` dentro do corpo de uma tool é classificado como argumento errado → F13

`_result_from_exception` converte qualquer `TypeError` em `validation_error` ("argument mismatch",
`retryable=False`), incluindo bugs internos da tool:

```
return "value: " + x  -> validation_error | Tool 'buggy' argument mismatch: can only concatenate str (not "int") to str
```

Proposta: com a validação do F13 antes da chamada, separar o erro de ligação de argumentos (p. ex.
`inspect.signature(fn).bind(...)` antes de chamar) de um `TypeError` levantado pela própria tool,
que passaria a `runtime_error`.

## 2026-09-13 · worker-C (transcrito pelo coordenador)

### C1 · Gemini rejeita `prefixItems` e `$defs`/`$ref` nos schemas de tools → F19

No `google-genai` 2.11.0, `types.Schema` tem `extra="forbid"`: `_tool_to_sdk` rebenta no cliente com
`pydantic.ValidationError` para `tuple[str, int]` (`prefixItems`) e modelos Pydantic aninhados
(`$defs`/`$ref`). Com o F12, `tuple[int, int] | str` (antes `string`) também rebenta.
Reprodução: `def f(pair: tuple[str, int])` → `_tool_to_sdk(prepare_tools([f])[0])`.

### C2 · Tipos desconhecidos viram `{"type": "string"}` → F13

Aliases PEP 695 (incluindo `type IdOrName = int | str`), `Any`, `object`, `date`, `Path` e classes
próprias passam a `string`. Com a validação do F13 isso seria imposto (`n=5` falharia; `Any`
rejeitaria objectos). Proposta: desembrulhar `TypeAliasType` (`__value__`); `Any`/`object` → `{}`; o
validador ignora schemas sem `type`/`anyOf`.

### C3 · Validação e `**kwargs` → F13

Com `**kwargs` fora do schema, "chave desconhecida → `validation_error`" partiria tools com `**kwargs`:
o validador tem de ver `VAR_KEYWORD` na assinatura. Em `anyOf`, manter o valor se já satisfaz um
ramo; coagir só se nenhum satisfizer.

### C4 · nanope: `allow_dangerous` não chega → sem tarefa (WIP)

`resolve_tools_with_limits` constrói `ToolGroup` sem `approval_handler`; `run_command`/`python_repl`
já davam `approval_denied` e, com o F11, as outras cinco também. `nanope/bbeh/_solvers.py` usa
`ToolGroup(..., python_repl)` sem handler. Nenhum teste de `tests/nanope` usa estas tools.

## 2026-09-13 · worker-E (transcrito pelo coordenador)

### O OpenAI não converte conteúdo não-string de mensagens `system()` → sem tarefa

`_openai._messages_to_sdk` envia cru o conteúdo de uma mensagem `system()` que seja lista (p.ex.
`CachePart`) e o SDK falha. Antes do F05 isto só acontecia sem `system=`; agora também com `system=`
(antes a mensagem era descartada em silêncio). Reprodução:
`_messages_to_sdk([{"role": "system", "content": [CachePart(content="A")]}, {"role": "user", "content": "x"}], system="B")`.


## 2026-09-13 · worker-D

### `_stream_sync` levanta itens que são excepções, mesmo produzidos como valor → sem tarefa

O consumidor trata qualquer item que seja instância de `BaseException` como erro da fonte e
levanta-o, por isso uma fonte que *produz* uma excepção como valor (sem a levantar) interrompe o
stream do lado síncrono; a via async entrega-a normalmente. Anterior a F15 e mantido tal como estava
(o canal de erros não tem envelope próprio). Nenhuma fonte actual (`LLM`, `Flow.iter`) produz
excepções como valor, daí a severidade baixa.

```
async def gen():
    yield ValueError("valor")
    yield "depois"

list(_stream_sync(lambda: gen()))  # -> ValueError('valor'); esperado [ValueError('valor'), 'depois']
```

## 2026-09-13 · coordenador (fecho)

### `_run_before`/`_run_after` ficaram só com testes → sem tarefa

Desde o F03, `core/_llm.py` já não usa os hooks síncronos de `core/_middleware.py`; só
`tests/test_middleware.py` chama `_run_before` e `_run_after`. Código morto privado, sem efeito para
utilizadores. Remover junto com os testes quando se mexer no middleware.

### `Any`/`object` continuam a gerar `{"type": "string"}` → sem tarefa (ver F13)

A validação do F13 não coage nem recusa `string`, por isso não parte chamadas. O schema continua a
dizer ao modelo que um parâmetro `Any` é texto. Mudar para `{}` exige confirmar que o Gemini aceita
propriedades sem tipo; com o F19 o recurso a `parameters_json_schema` torna isso mais seguro.

## 2026-09-13 · coordenador (resolução)

Resolvidos pela F21: C4 (nanope, incluindo o `research_center`, que o C4 não referia), o conteúdo
não-string de mensagens `system()` (worker-E), as excepções como valores no `_stream_sync`
(worker-D), `Any`/`object` descritos como `string`, e `_run_before`/`_run_after` mortos.

### `cache()` enviado como `repr` fora do Anthropic → F21

Em mensagens de utilizador, OpenAI, Gemini e xAI convertiam `CachePart` com `str(part)`, e o modelo
recebia `CachePart(content='…', ttl='ephemeral')`. O xAI fazia o mesmo com `DocumentPart`, bytes
incluídos. Reprodução: `_openai._messages_to_sdk([user(["a", cache("LONG")])])`.

## 2026-09-13 · revisão pós-implementação (F22)

### Parâmetro `Optional` sem default não pode ser omitido → sem tarefa

`infer_schema` tira de `required` um parâmetro `X | None` mesmo sem default; o modelo omite-o e a
chamada falha no binding com `validation_error "argument mismatch: missing a required argument"`.
Anterior ao plano (antes era `TypeError` na chamada). Opções: exigir parâmetros sem default, ou passar
`None` aos opcionais omitidos. Reprodução: `def find(q: int | None) -> str` chamado com `{}`.

### `functools.partial` em `ToolGroup` → sem tarefa

`ToolGroup(functools.partial(f, 1))` rebenta com `AttributeError: __name__` em `infer_schema`.

### `ApprovalDecision.approve(modified_args={})` ignorado → sem tarefa

`ApprovalGate._outcome` usa `decision.modified_args or ...`, logo um dict vazio mantém os argumentos
originais. Anterior ao plano.

### Gemini com declarações mistas numa `Tool` → por verificar ao vivo

Uma lista de tools em que umas vão por `parameters` e outras por `parameters_json_schema` gera uma
única `types.Tool`; não foi possível confirmar offline que a API aceita a mistura.


## 2026-09-13 · fornecedor Meta (M01)

### Chave da OpenAI enviada a um `base_url` remoto → sem tarefa

`LLM("modelo", base_url="https://gateway.example/v1")` sem `api_key=` cai no adaptador OpenAI e
`_resolve_key("OPENAI_API_KEY", ...)` envia a chave da OpenAI do ambiente para esse host. Com
`localhost` isso já não acontece. O `MetaProvider` usa só `MODEL_API_KEY`, mas o risco continua
para qualquer outro endpoint compatível com OpenAI. Anterior ao M01.

### `output_schema` Pydantic com `strict: true` no adaptador OpenAI → por verificar ao vivo

`_resolve_output_schema` gera `OutputSchema(strict=True)` com `model_json_schema()`, que não tem
`additionalProperties: false`; o adaptador OpenAI envia-o assim. A Meta recusa exactamente este caso
com 400 (verificado); a OpenAI documenta a mesma regra para `strict`, mas não foi confirmado ao vivo.

### `openai.APIConnectionError` não dá retry nem fallback → sem tarefa

Os adaptadores OpenAI e Meta deixam passar `openai.APIConnectionError`/`APITimeoutError`, que não
são `APIError`, `ConnectionError`, `OSError` nem `TimeoutError` (`PROVIDER_ERRORS` em `_llm.py`) e
não têm `status_code` para o `_retry.py`. Uma falha de rede não faz retry nem passa ao fallback.

### `@pytest.mark.timeout` não faz nada → sem tarefa

O marcador está registado no `pyproject.toml`, mas o `pytest-timeout` não está instalado; os
timeouts dos testes `live_api` são decorativos. O job de integração passou a ter 30 minutos.
