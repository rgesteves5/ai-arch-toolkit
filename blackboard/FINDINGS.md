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

## 2026-09-17 · coordenador

### Qualquer chamada LLM falhada fica com custo desconhecido e, sob tecto de custo, mata o run → sem tarefa (impeditivo)

Alarga o achado "Erro do adaptador a montar o pedido envenena o budget" (2026-09-15): não é só o erro
do adaptador. `MeterStore.fail()` atribui `Cost.unknown("operation did not settle")` a toda a
operação `llm` que não liquida (`core/_metering/_store.py:156-160`), seja qual for a causa: resposta
de erro do fornecedor (429, 5xx, 4xx — não facturada), falha de rede, timeout, cancelamento ou erro
local antes de qualquer I/O. Os dois consumidores de `unknown_cost_count` fecham a porta a seguir:
`BudgetController._exceeds` com `unpriced="fail_closed"`, que é o valor por omissão
(`toolkit/budget/_controller.py:102-108`), e o tecto por step (`core/_step_engine.py:212`). Como cada
tentativa é uma operação, a tentativa seguinte do retry e o primeiro fallback já são negados.

Reproduções do coordenador (provider falso, sem rede, `BudgetPolicy(max_cost=5.0)`):

```
503 e depois ok, retry=2           -> BudgetExceeded após 1 chamada ao provider (o retry é negado)
429 e depois ok, retry=2           -> idem
primário 503, fallback configurado -> BudgetExceeded; o fallback nunca é chamado
ConnectionError e depois ok        -> BudgetExceeded
timeout de quem chama, nova chamada -> BudgetExceeded
ValueError do adaptador, nova chamada -> BudgetExceeded
sem tecto, ou unpriced="allow"     -> o retry funciona (unknown_cost_count=1)
Step com Policy(max_cost=1.0), 503 e retry com sucesso (custo real 0,0006 USD)
                                   -> "Cost exceeded limit 1.0: a call could not be priced (fail-closed)"
```

O desenho original (`docs/internal/metering-plan.md:135-137`) fixou "llm falhado → Unknown" sem
distinguir tipos de falha, e a matriz de testes prevista cruza `max_cost` × Known/Unknown ×
fail_closed/allow, mas nunca falha × retry/fallback × tecto. Contorno: `unpriced="allow"`, que também
deixa passar modelos sem preço e server tools. Afecta qualquer app com um tecto de custo: um 429
passageiro termina o run com `BudgetExceeded`.

## 2026-09-17 · investigação das causas (coordenador e cinco agentes de leitura)

Base de `docs/internal/hardening-plan.md`, que agrupa estes achados e os de 2026-09-15 por causa.
Os scripts estão em `blackboard/prototypes/2026-09-hardening/`. Nenhum tem tarefa ainda.

**Verificados pelo coordenador (código lido ou reprodução corrida):**

### Gemini nunca junta os resultados de tools e não devolve o `id`

`_messages_to_sdk` chama `_flush_fn_responses()` à entrada de cada `tool_result`
(`core/_providers/_gemini.py:156`), por isso cada resultado sai num `Content` `user` próprio, ao
contrário da docstring (`:138-140`). A `FunctionResponse` é construída sem `id` (`:175-178`); o
Gemini 3 dá um `id` por `functionCall` e pede o mesmo `id` na resposta. A única forma documentada é um
`Content` com todas as respostas; o 400 "number of function response parts…" está por confirmar ao
vivo (um pedido num modelo 2.5).

### O 529 da Anthropic não é repetido

`RetryConfig.retry_on_status` é `(429, 500, 502, 503, 504)` (`core/_retry.py:23`); `overloaded_error`
(529) é o erro transitório típico da Anthropic e propaga à primeira.

### `mediawiki_*` deixam o modelo escolher o host, no namespace seguro

`_valid_api_url` só exige `https`, um netloc e um caminho acabado em `api.php`
(`toolkit/tools/_mediawiki.py:276-278`). Com `ToolGroup(mediawiki_search)` sem handler,
`api_url="https://169.254.169.254/api.php"` → o pedido é tentado; a política é a por omissão
(`capability=None`, sem aprovação).

### `ip_lookup` usa `http://` e revela o IP da máquina

`ip=""` → pedido a `http://ip-api.com/json/?fields=…` (`toolkit/tools/_geo.py:192-205`): devolve IP
público, ISP e localização do anfitrião. O `ip` é interpolado sem `quote`.

### `math_eval("9**9**9")` não termina

Ainda a correr ao fim de 4 s (morto pelo teste). O executor não tem timeout por tool
(`core/_tools/_executor.py:101-113`); uma tool síncrona presa ocupa uma thread de `to_thread` sem fim.
`regex_search` com `(a+)+$` tem o mesmo efeito (reprodução do agente).

### Bloco `document` da Anthropic leva `name`; o SDK só tem `title`

`core/_providers/_anthropic.py:112-113` envia `block["name"]`; `DocumentBlockParam` tem
`cache_control, citations, context, source, title, type`. `tests/test_anthropic_provider.py:1117`
afirma `name`: o teste fixa o erro.

### O preço por prefixo dá preços errados a modelos sem entrada

Não é só o Fable 5.1. `pricing.get()` casa pelo prefixo mais longo (`core/_pricing.py:107-119`):

```
o3-pro -> [o3] 2/8 USD (o publicado é 20/80)      gpt-4o-audio-preview -> [gpt-4o]
o3-deep-research -> [o3]                          gemini-2.5-flash-image -> [gemini-2.5-flash]
gpt-5-codex -> [gpt-5]                            grok-4.6-mini -> [grok-4.6]
```

Um custo "conhecido" errado passa por baixo do `fail_closed`.

### Regras por modelo: o ramo por omissão é o antigo

Anthropic `_TEMPERATURE_DEPRECATED_PREFIXES` e thinking (`_anthropic.py:54-62`, `:238-252`), OpenAI
`_MAX_COMPLETION_TOKEN_PREFIXES` (`_openai.py:60-61`), Gemini `model.startswith("gemini-3")`
(`_gemini.py:256`): os modelos novos entram por lista e tudo o resto recebe a forma antiga. Cada
geração nova parte até alguém editar a lista (foi o que aconteceu ao thinking do Claude).

### O Gemini muda de pilha HTTP conforme os extras instalados

O `google-genai` usa `aiohttp` sempre que é importável (`_api_client.py:74-78`, `_use_aiohttp`), e o
`xai-sdk` instala-o. Nessa pilha o SDK reenvia o pedido por conta própria em erros de ligação
(reprodução do agente: um `complete()` → 2 pedidos), fora do `retry_options` e do meter.
`HttpOptions.httpx_async_client` permite fixar a pilha.

### O xAI ignora `timeout`, mas o SDK aceita-o

`_xai.py:321-325` avisa que não é suportado; `xai_sdk.AsyncClient.__init__` tem `timeout` e o valor
por omissão do SDK são 27 minutos.

### Cancelar à espera do `inference_limit` conta como chamada iniciada

`op.mark_started()` corre antes de `async with inference_slot()` (`core/_llm.py:677,682`): uma
chamada cancelada na fila fica iniciada e com custo desconhecido sem nunca ter saído.

**Reproduzidos pelos agentes (não repetidos pelo coordenador):**

### Erros de transporte a meio de um stream escapam sem mapeamento

Anthropic, OpenAI e Meta deixam sair `httpx.RemoteProtocolError`/`ReadError`/`ReadTimeout` crus (os
SDKs só embrulham erros de `send()`); não são `OSError`, por isso ficam fora de `PROVIDER_ERRORS`:
sem retry nem fallback. No OpenAI, `data: {error}` dentro do stream deixa sair `openai.APIError` cru.
Na Anthropic, o evento `error` vira `APIError(status_code=200)`. Meta e xAI inventam 5xx depois de um
HTTP 200: o código de estado não chega para classificar uma falha.

### Validação local do SDK dentro da chamada aguardada

`anthropic`: `ValueError("Streaming is required…")` com `max_tokens` grande; `anthropic`/`openai`:
`TypeError` de JSON com um `set` no input; `google-genai`: `ValueError('contents are required.')`.
Nada saiu para a rede, mas a operação já está iniciada. Um `ValueError` cru tanto pode querer dizer
"nada foi enviado" como "a resposta cobrada não se leu" (HTTP 200 com corpo não-JSON).

### xAI: `tool_choice` com nome de tool rebenta no SDK real

O adaptador envia um dict ao estilo OpenAI; `xai_sdk…chat.create()` levanta `ValueError: Protocol
message ToolChoice has no "type" field` (o SDK quer `required_tool(name)`). O mock dos testes esconde
o `create()` real.

### Gemini 3: `thinking_effort="xhigh"`/`"max"` só dá aviso

O SDK avisa `xhigh is not a valid ThinkingLevel` e o pedido segue.

### Anthropic: um `user` por `tool_result`

A documentação de parallel tool use chama a esta forma "Wrong": não dá erro (a API junta turnos do
mesmo papel), mas reduz as chamadas paralelas nos turnos seguintes.

### Tools: nenhum limite central

0 de 49 leituras de rede têm limite de bytes; o `urllib` segue redirects para outro host (https →
http incluído); não há helper HTTP comum (36 helpers privados em 32 módulos, 11 `urlopen` inline);
nenhum limite de saída no executor nem nos flows (5 MB chegam ao modelo); tectos sem grampo
(`http_get(max_chars=-1)` → 3 MB; `wikipedia_article(max_chars=-1)` → 2 MB); 12 tools levantam com
argumentos hostis e 90 com corpos JSON de forma inesperada; `..` chega ao caminho em `country_info`,
`define_word` e `europe_pmc_citations`; as 125 tools seguras têm todas `capability=None`; as três
`youtube_*` usam `youtube_transcript_api` e `requests` (não são stdlib-only).

### Testes: só a Meta cobre tool calls paralelas com reenvio

Nenhum teste de Anthropic, OpenAI, Gemini ou xAI reenvia um histórico com duas chamadas paralelas.
`test_stream_retry_meters_every_physical_attempt` (`tests/test_llm_metering.py:279-319`) afirma
`unknown_cost_count == 1` depois de um 500: fixa o comportamento do achado impeditivo.

## 2026-09-17 · revisões externas verificadas pelo coordenador

Duas revisões de outros modelos, pedidas pelo dono. As métricas e as duplicações que apontam
reproduzem-se todas (174 ficheiros, 40 859 linhas, 1929 funções, radon: `_eval_expr` 84, `_run_dag`
48, `_validate_manifest` 36, `_run_attempts` 33; 54 janelas de 8 linhas duplicadas entre
`core/graph/_store.py` e `toolkit/memory/graph/_store.py`; 2979 testes e 89% de cobertura sem
`nanope`). Uma delas chegou sozinha ao achado impeditivo (tecto de custo × retry). Os quatro bugs
abaixo são novos; reprodução em `blackboard/prototypes/2026-09-hardening/external_review_probes.py`.
A dívida de manutenção que apontam está na secção 10 de `docs/internal/hardening-plan.md`.

### `LLM(fallback=outro)` esvazia a cadeia de fallbacks de `outro` → sem tarefa

`_normalize_fallbacks` achata a cadeia aninhada e limpa `_fallbacks` e `_owned_fallbacks` do objecto
recebido (`core/_llm.py:121-127`; a docstring avisa, mas o efeito é num objecto do utilizador).

```
b = LLM("claude-sonnet-4-6", fallback=c)   -> b tem 1 fallback
a = LLM("claude-opus-5", fallback=b)       -> b passa a ter 0; usado sozinho, b já não recua para c
```

### A validação aceita `None` num parâmetro obrigatório e não valida elementos de listas → sem tarefa

```
def typed(value: int, items: list[int])
{"value": None, "items": ["wrong"]} -> ok=True, a função recebe value=None items=['wrong']
{"value": "x",  "items": [1]}       -> validation_error (como esperado)
```

D7 deixou arrays e objectos intactos de propósito; o `None` num `integer` não anulável não foi
decisão (`core/_tools/_validation.py:143-173`).

### `ReasoningSpec.from_mapping` descarta em silêncio o que não reconhece → sem tarefa

API documentada (`docs/agents.md:284`). `{"policy": {"timeout": 1}}` → `policy=None`;
`{"output_schema": 123}` → `None`; chave desconhecida → ignorada (`toolkit/agents/_spec.py:42-70`).
É o mesmo género do achado G da primeira revisão: configuração declarada que fica inerte sem aviso.

### Imutabilidade só à superfície → sem tarefa (decisão de contrato)

`State.snapshot()` diz "Immutable copy" mas partilha os valores (`core/_state.py:159-166`): um step
que faça `snapshot["items"].append(...)` altera o estado sem passar por `Result.artifacts` nem pelo
merge. O mesmo step comporta-se de duas maneiras (verificado): numa vaga paralela a escrita in-place
perde-se sem aviso, porque cada irmão corre sobre um `fork()` com deep copy; em modo sequencial e nas
vagas de um step fica no estado e o step seguinte vê-a. `ReasoningSpec(frozen=True).knobs` é um
`dict` mutável. Cópias profundas por
step trariam de volta o custo quadrático medido em N7: a correcção é de contrato (vista só de
leitura documentada, congelar à entrada onde é barato), não cópias defensivas em todo o lado.

## 2026-09-17 · coordenador (F24)

### Tentativas falhadas de fallbacks intermédios não ficam em `Response.attempts` → sem tarefa

No caminho `complete`, `_try_fallbacks` só junta as tentativas do fallback que teve sucesso
(`attempts.extend(response.attempts)`); as de um fallback que falhou perdem-se com a excepção. Com
A → B (falha) → C (ok), `attempts` dá `[A, C]`. O caminho de stream regista-as. Pertence à pipeline
de tentativa única do plano de robustez.

## 2026-09-18 · R01 (F25.2)

### `ip_lookup`: transporte HTTP gratuito continua pendente

Recusados IPs vazios e inválidos antes de I/O. Mantém-se o endpoint HTTP explicitamente previsto pela F25: a passagem a HTTPS exige escolher outro serviço ou plano e pertence ao refactor de tools. Não houve rede nem verificação externa; o estado do endpoint não foi inferido.

## 2026-09-18 · SDK Anthropic 1.6.0 já não aceita temperature (R02)

- **Reprodução local:** `uv run python -c 'import inspect, anthropic; print(anthropic.__version__); print(inspect.signature(anthropic.resources.messages.AsyncMessages.create))'` mostra 1.6.0 sem temperature. LLM injecta por omissão temperature=0.0; complete e ambos os streams dão `TypeError: AsyncMessages.create() got an unexpected keyword argument 'temperature'`, antes de sair qualquer pedido ao servidor falso dos protótipos. Os mocks de SDK antigos aceitam kwargs arbitrários e ocultam o desvio.
- **Âmbito:** migração de parâmetros/capacidades do adaptador, explicitamente R02; não se alterou o adaptador nesta fase. Os 7 casos locais Anthropic da R01 retiram o default apenas na fixture, para medir os erros de transporte/HTTP de um pedido válido. Os 7 casos OpenAI não precisam dessa adaptação.
- **Consequência:** não afirmar que o percurso real Anthropic com defaults já funciona; R02 deve acrescentar prova de pedido por omissão no SDK instalado e corrigir a preparação.

## 2026-09-18 · R01, verificação independente (Claude)

### Um tecto por step sem `BudgetPolicy` continua a falhar depois de um 5xx → decisão do dono (R02)

O protótipo `meter/step_cap.py`, que é a reprodução da linha "Step com `Policy(max_cost=1.0)`" do
achado impeditivo de 2026-09-17, corre sem `BudgetPolicy` e continua vermelho depois da R01:

```
uv run python blackboard/prototypes/2026-09-hardening/meter/step_cap.py
provider calls: 2 | step error: Cost exceeded limit 1.0: a call could not be priced (fail-closed)
```

Com `flow.run(State(), budget_policy=BudgetPolicy(max_cost=5.0))`, soft ou strict, o step passa
(`cost_at_most=0.062067`). É o contrato que a ficha R01 fixou ("sem controller não há tecto") e que
`docs/safety.md` documenta. A D20 diz, no entanto, "o resto fica incerto com tecto", sem condição.
O tecto de uma falha indeterminada é um facto do pedido (preço do modelo × entrada + `max_tokens`),
não uma opinião do controller: com a D16 (sob um meter, todo o modelo tem preço), o core pode
calculá-lo sempre, e o desconhecido sem tecto fica reduzido às server tools. Proposta para a R02: o
estimador do pior caso passa para o core, é a casa única do tecto de falha e da reserva estrita, e
desaparece o `Protocol` `FailureBoundController` (`core/_metering/_admission.py`).

## 2026-09-18 · resolução antes do push (Claude, a pedido do dono)

### `anthropic` 1.6.0 e `temperature` → resolvido

`temperature`, `top_p` e `top_k` vão em `extra_body`, a via documentada para o SDK 1.x, e uma só
função (`_sampling_body` em `core/_providers/_anthropic.py`) decide o que segue: tira a `temperature`
nos modelos que a recusam e com thinking ligado, como antes. Prova com o SDK real no servidor falso
em loopback: os sete ensaios Anthropic de `tests/integration/test_attempts_local_sdk.py` correm com
o `temperature=0.0` por omissão, sem o contorno da fixture, e afirmam que ele chega ao corpo do
pedido; antes da correcção falhavam com `TypeError`. Os testes unitários passam a ligar os kwargs à
assinatura instalada do SDK. Falta a verificação ao vivo: não há créditos Anthropic.
