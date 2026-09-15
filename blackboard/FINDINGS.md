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

### Seguimento (2026-09-13): sem testes reais no CI

A pedido do dono, o workflow `integration.yml` foi removido e o `ci.yml` corre `pytest -m "not
live_api"`; a nota acima sobre o job de integração com 30 minutos deixa de se aplicar.

## 2026-09-15 · coordenador (resolução, F23)

Resolvidos pela F23, todos com testes: `X | None` sem default omitido, `functools.partial` em
`ToolGroup`, `approve(modified_args={})`, chave da OpenAI em `base_url` remoto (agora exige
`api_key=`), `output_schema` Pydantic com `strict` no OpenAI (400 confirmado ao vivo e corrigido),
`APIConnectionError` sem retry nem fallback (OpenAI, Meta, Anthropic e Gemini) e
`@pytest.mark.timeout` sem efeito. Gemini com declarações mistas numa `Tool` foi verificado ao vivo
e funciona: não precisava de correcção.

## 2026-09-15 · abertura da frente C (coordenador)

Encontrados pelos agentes que escreveram as fichas C01–C09. O coordenador voltou a correr as
reproduções (sem rede, com SDKs simulados) ou confirmou no código e na documentação oficial, como se
indica em cada grupo.

**Reproduzidos pelo coordenador:**

### `ToolGroup` vazio é substituído → C02a

Um grupo vazio é falso (`__len__`), e `tools or ToolGroup()` troca-o por outro
(`toolkit/agents/_agent.py:56`). O mesmo `or` escolhe as tools de fase (`_plan_execute.py:60`,
`_reflexion.py:63`, `_llm_compiler.py:97`, `_self_discovery.py:85`, `_lats.py:114`): um
`executor_tools` vazio fica com as tools principais.

```
group = ToolGroup(approval_handler=h); Agent(ReasoningSpec(strategy="react"), llm, group); group.add(ping)
-> o LLM recebe outro grupo (len 0) e o modelo lê "Tool error [unknown_tool]: Unknown tool: 'ping'"
```

### `@tool(schema=)` com um JSON Schema completo corrompe a tool → C02b

`schema=` são overrides por parâmetro, mas `docs/tools.md:53` chama-lhe "override the inferred JSON
Schema".

```
@tool(schema={"type": "object", "properties": {"q": {"type": "string"}}})
-> properties = {"type": "object", "properties": {...}}, required = []
```

### Erro do adaptador a montar o pedido envenena o budget → sem tarefa

A operação de metering abre antes de o adaptador montar o pedido; um `ValueError` do adaptador
fica como chamada de custo desconhecido e, sob `max_cost`, a chamada seguinte é negada. O C05a só
move para antes do meter a verificação das server tools.

```
with budget_scope(BudgetPolicy(max_cost=5.0)):
    LLM("muse-spark-1.3").complete("q", tools=[t], tool_choice="required")  # ValueError
    LLM("muse-spark-1.3").complete("q")  # BudgetExceeded: a prior call could not be priced
# chamadas de rede: 0
```

### `reserve="strict"` não reserva o custo de tools com preço → C08c

`HeuristicEstimator` devolve uma reserva vazia para operações que não são LLM
(`toolkit/budget/_estimator.py:42-43`); `docs/safety.md:282` promete tectos rígidos com `strict`.

```
Pricer da app: 0,016 USD por chamada de tool; BudgetPolicy(max_cost=0.02, reserve="strict")
sequencial: executam 2 (0,032 USD) e só a 3.ª é negada · paralelo: executam 3 (0,048 USD)
```

### Server tools no fio: config descartada e formas inválidas → C05a

`web_search(max_uses=3, allowed_domains=[...])` e `code_execution()` saem assim:

```
anthropic [{"type": "web_search_20250305"}, {"type": "code_execution_20250522"}]
openai    [{"type": "web_search"}, {"type": "code_interpreter"}]
gemini    [{"google_search": {}}, {"code_execution": {}}]
meta      [{"type": "web_search"}] + UserWarning (code_execution descartado)
xai       NotImplementedError
```

Na Anthropic falta `name` (o SDK 0.116 marca-o `Required`); o `tools` do Chat Completions só aceita
`function`/`custom` no SDK `openai` 2.45 e a OpenAI documenta web search no Chat Completions só com
modelos de pesquisa. Os 400 esperados estão por confirmar ao vivo; `examples/25_server_tools.py` usa
este caminho.

### Com `max_cost`, uma server tool é negada antes de correr → C05a

```
BudgetPolicy(max_cost=5.0) + complete(tools=[web_search()])
-> BudgetExceeded "server tools have unmetered cost"; chamadas ao provider: 0
```

`docs/safety.md:275` diz que só o trabalho seguinte é negado. Na mesma reprodução, a Anthropic cola o
preâmbulo ao texto da resposta (`"I'll search.Claude Shannon…"`) e ignora os blocos de servidor.

### `stream_events()` perde `parsed` e `response_id` → C01a

```
OpenAI, output_schema: complete      parsed=Answer(answer=3) response_id='chatcmpl-1'
                       stream_events parsed=None             response_id=''
```

### Gemini em stream reenvia só o último chunk → C01a

`_gemini.py:634` guarda como `raw` o último chunk, e é esse que o histórico reenvia.

```
2 function calls em 2 chunks: complete reenvia [get_weather + assinatura, get_time]
                              stream_events reenvia [get_time]
```

Que o Gemini 3 recusa uma `functionCall` sem assinatura vem da documentação; não confirmado ao vivo.

### Stream sem usage fica com custo 0 conhecido → C01a

```
MeterScope, stream sem usage (gpt-4o): llm_calls=1 cost=0.0 unknown_cost_count=0
```

### Gemini separa resultados de tools consecutivos → sem tarefa

`_messages_to_sdk` põe cada `tool_result` num `Content` `user` próprio (`_gemini.py:153-154`), ao
contrário do que diz a docstring. Se a API aceita, está por confirmar ao vivo.

```
dois tool_result seguidos -> contents [('user', 1), ('model', 2), ('user', 1), ('user', 1)]
```

### `step_end` em falta quando `max_wall_s` é excedido → sem tarefa

O check de budget sai antes de emitir o evento (`toolkit/flow/_executor.py:356-359` sequencial,
`:434-437` vaga de um step); as vagas paralelas não são afectadas.

```
Flow(slow 0,1 s, next).iter(State(), budget_policy=BudgetPolicy(max_wall_s=0.05))
eventos: flow_start, step_start slow, policy_decision budget_exceeded, flow_end   (sem step_end)
trace: ['slow', 'budget_exceeded']; estado slow_done=True
```

### `csv_read` lê qualquer ficheiro sem aprovação → sem tarefa

`toolkit.tools.csv_read` é `@tool` nu (`toolkit/tools/_json.py:63-78`): `risk_level="low"`, sem
aprovação e fora de `dangerous`, contra "safe-by-default" do `AGENTS.md`.

```
ToolGroup(csv_read), sem approval_handler, path=<ficheiro de chave> -> ok=True e o conteúdo da chave
```

### Leituras sem limite em linhas longas → sem tarefa

```
read_file(f), ficheiro de 2 MB numa só linha      -> 2 000 000 caracteres
search_files(dir, "var a", max_results=1)          -> 2 000 041 caracteres
```

`read_file` também carrega o ficheiro inteiro antes de cortar linhas.

### `list_directory` levanta com certos `pattern` → sem tarefa (menor)

`pattern="/etc/*"` → `NotImplementedError`; `pattern=""` → `ValueError`. Pelo executor viram
`runtime_error`, mas a convenção das tools é devolver strings.

### Risco: `_inline_local_refs` é exponencial → C02e

Um schema de 1 926 bytes com 18 níveis de `$ref` duplos expande para 17,3 MB em 1,39 s. Hoje só
chega lá quem escreve o schema; com schemas de servidores MCP (C03) passa a ser entrada não confiável.

**Confirmados no código e na documentação oficial, sem chamada ao vivo:**

### `thinking=True` no Anthropic falha nos modelos actuais → sem tarefa

`_build_thinking_param` envia sempre `{"type": "enabled", "budget_tokens": N}`
(`core/_providers/_anthropic.py:238-252`) e nunca `adaptive`. A documentação da Anthropic diz que
`budget_tokens` é recusado com 400 em Fable 5/5.1, Opus 5/4.8/4.7 e Sonnet 5 (usar
`{"type": "adaptive"}` com `output_config.effort`). Nos modelos anteriores exige
`budget_tokens < max_tokens`, e o toolkit põe 10 000 (`_providers/_base.py:27`) contra
`max_tokens=4096` por omissão (`core/_llm.py:506`). A matriz de compatibilidade nunca testou thinking
na Anthropic.

### `tool_choice="required"` recusado no Fable 5.1 → sem tarefa

O adaptador envia `{"type": "any"}` (`_anthropic.py:544`); o Fable 5.1 e o Mythos 5.1 recusam `any` e
`tool` com 400, segundo a documentação da Anthropic.

### Preço do `claude-fable-5-1` herdado por prefixo → sem tarefa

`_default_pricing.toml` não tem `[claude-fable-5-1]`; o prefixo dá `[claude-fable-5]` com
`cache_read = 1.0`, e a tarifa publicada do Fable 5.1 é 0,25 USD/MTok. O `claude-mythos-5-1` herda do
Mythos 5; a documentação ainda não fixa a tarifa de cache da 5.1.

### Deriva do `AGENTS.md` → sem tarefa

`AGENTS.md:78` diz que o structured output da Anthropic cai para o prompt automaticamente; o código
só o faz com `structured_output_mode="prompt"` (`_anthropic.py:520-525`). A lista de prefixos do
`AGENTS.md` não tem `chat-` (`core/_providers/__init__.py:21`).

### Comentário do xAI desactualizado → sem tarefa (fora do âmbito do C05)

`_xai.py:408-416` diz que o `xai-sdk` não tem server tools; a 1.17.0 instalada tem
`xai_sdk.tools.web_search`, `x_search` e `code_execution`.
