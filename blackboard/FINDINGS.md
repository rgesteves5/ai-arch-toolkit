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

## 2026-09-18 · R02 (Claude)

### Os resultados de batch são preçados à tarifa normal → sem tarefa

`OpenAIProvider._parse_batch_response` e `AnthropicProvider.batch_results` calculam
`Response.cost` com `_estimate_response_cost(model, usage)`, que nunca passa `is_batch=True`: o custo
de um resultado de batch sai com a tarifa normal, o dobro da de batch nos dois fornecedores
(`batch_input`/`batch_output` estão na tabela). Reprodução: um resultado de batch do `gpt-4o` com
1000 tokens de entrada dá `cost == 0.0025` em vez de `0.00125`. O meter não é afectado (o batch não
passa pelo meter). Não corrigido na R02 para não alargar o âmbito.

### Um `LLM` para um modelo Grok não se constrói depois de um `asyncio.run` → sem tarefa

`XAIProvider` constrói o `xai_sdk.AsyncClient` no construtor (`LoopAwareClientCache._install_client`
chama a fábrica logo), e o canal `grpc.aio` pede `asyncio.get_event_loop()`. Num programa síncrono
que já correu um `asyncio.run(...)`, a política ficou sem loop e `LLM("grok-4.6", api_key=...)`
levanta `RuntimeError: There is no current event loop in thread 'MainThread'` (reproduzido: num
processo novo constrói; depois de `asyncio.run(asyncio.sleep(0))`, rebenta). Os outros adaptadores
não sofrem (os clientes `httpx`/`httpx2` não pedem loop). A correcção natural é construir o cliente
só no primeiro acesso ao `_client`, mas muda o sítio onde aparecem os erros de construção dos cinco
adaptadores: fica para decisão, fora da R02.

### O adaptador xAI larga imagens que os modelos Grok aceitam → resolvido na A13 (G-39)

`_xai._user_text` avisa "xAI does not support image input" e larga a imagem, mas as páginas dos
modelos dão `text, image → text` ao `grok-4.6`, `grok-4.5`, `grok-4.3`, `grok-4.20-*` e
`grok-build-0.1` (https://docs.x.ai/developers/models/grok-4.6, 2026-09-18), e o SDK tem
`chat.image(...)`. É uma funcionalidade em falta, não um erro de fio; o aviso diz mais do que é
verdade. Fora do âmbito da R02 (catálogo e capacidades: C06).

### OpenAI: `thinking_effort` sem `thinking=True` perde-se em silêncio → sem tarefa

Nos outros quatro adaptadores o esforço aplica-se sozinho (D13, D25; a Anthropic manda-o em
`output_config.effort`, o Gemini como nível ou orçamento), mas o `OpenAIProvider._reasoning` sai
logo quando `thinking` é falso: `LLM("gpt-5.4").complete(..., thinking_effort="low")` envia o pedido
sem `reasoning_effort` e sem aviso, e o modelo raciocina com o esforço por omissão
(https://developers.openai.com/api/docs/models/gpt-5.2: os GPT-5 raciocinam sem pedir).
Reprodução: `prepare(OpenAIProvider("gpt-5.4", "k"), [user("hi")], thinking_effort="low").params`
não tem `reasoning_effort`. Encontrado ao documentar o thinking por fornecedor (passo 6); o passo
do OpenAI já estava fechado, e alinhar muda o fio de quem hoje passa um esforço sem `thinking`:
fica para decisão do dono. Documentado em `docs/llm.md` tal como está.

## 2026-09-18 · R02 concluída (Claude): achados fechados

Cada um com a prova na secção da ficha `tasks/R02-providers.md` indicada; nada foi verificado ao
vivo (comandos no relatório final da ficha).

- "Erro do adaptador a montar o pedido envenena o budget" → resolvido no passo 2 (o `prepare`
  corre antes de qualquer admissão).
- "`stream_events()` perde `parsed` e `response_id`", "Gemini em stream reenvia só o último chunk"
  e "Stream sem usage fica com custo 0 conhecido" → resolvidos nos passos 2 e 3 (uma montagem;
  `raw` com todas as partes; sem usage, custo desconhecido). O resto do C01a não mudou.
- "Server tools no fio: config descartada e formas inválidas" → as formas ficam certas (Anthropic
  com `name` e versões actuais; OpenAI e xAI recusam) e a config levanta `RequestError` em vez de
  se perder; suportá-la continua no C05a.
- "Gemini separa resultados de tools consecutivos" e "Gemini nunca junta os resultados de tools e
  não devolve o `id`" → resolvidos no passo 3 (Gemini); a nota "Known issue" espera a prova ao vivo.
- "`thinking=True` no Anthropic falha nos modelos actuais", "`tool_choice="required"` recusado no
  Fable 5.1" e "Anthropic: um `user` por `tool_result`" → resolvidos no passo 3 (Anthropic).
- "Preço do `claude-fable-5-1` herdado por prefixo", "O preço por prefixo dá preços errados a
  modelos sem entrada" e "Regras por modelo: o ramo por omissão é o antigo" → resolvidos nos
  passos 1 e 3 (preço por id exacto; perfis com a geração actual por omissão).
- "Comentário do xAI desactualizado" e "xAI: `tool_choice` com nome de tool rebenta no SDK real" →
  resolvidos no passo 3 (xAI).
- "O Gemini muda de pilha HTTP conforme os extras instalados" e "Gemini 3:
  `thinking_effort="xhigh"`/`"max"` só dá aviso" → resolvidos no passo 3 (Gemini).
- "Erros de transporte a meio de um stream escapam sem mapeamento" e "Validação local do SDK dentro
  da chamada aguardada" → resolvidos nos passos 2 e 3 (mapeador por passo do stream; marcador de
  despacho, D24).
- "Testes: só a Meta cobre tool calls paralelas com reenvio" → os cinco adaptadores têm o contrato
  de conversa (`tests/test_provider_conversations.py`) e testes ao vivo preparados.
- Continua aberto: "Um tecto por step sem `BudgetPolicy` continua a falhar depois de um 5xx" espera
  a decisão do dono; a ficha da R02 não o incluía e não foi tocado.

## 2026-09-18 · R03 (Claude)

### `regex_search`: o backtracking polinomial continua possível → decisão do dono

Um match de regex corre em C com o GIL preso: nem o timeout do executor nem o `pytest-timeout`
o param (medido em `scratchpad/gil_probe.py`: `re.search("(a+)+$", "a"*32+"!")` numa thread
congelou o event loop 60 s sem um tick). A R03 recusa à entrada as formas exponenciais (grupo
repetido com quantificador ou alternativa lá dentro, referências para trás), o padrão acima de 500
caracteres e o texto acima de 20 000. Fica o custo polinomial: um quantificador sem limite é
quadrático (`\w+x` em 20 000 caracteres: 0,83 s) e dois seguidos que apanham os mesmos caracteres
são cúbicos (`.*.*x`: 1000 caracteres 0,15 s, 2000 caracteres 1,07 s; em 20 000, por extrapolação,
cerca de 18 min com o processo parado). Reprodução: `regex_search("a" * 20_000, ".*.*x")`. Com o
`re` da stdlib não há regra estática simples que o evite sem recusar padrões comuns
(`\w+@\w+\.\w+`). Opções: passar a tool para `dangerous` (aprovação), baixar o texto para uns 2000
caracteres, ou um motor linear (dependência nova). A ficha pedia guardas de tamanho; ficou isso e o
risco fica aqui para o dono decidir.

### `pdb_ligands` faz um pedido por entidade, sem tecto → sem tarefa

`pdb_ligands` pede a entrada e depois um `nonpolymer_entity` por cada id que ela lista
(`toolkit/tools/_pdb.py:84-88`), em série e sem limite: uma entrada com cem ligandos são cento e um
pedidos (cada um com prazo de 20 s; o executor corta a chamada aos 120 s). Já era assim antes da
migração para `_http`; o agente que migrou o módulo assinalou-o. Correcção provável: um tecto de
entidades, com nota no texto.

### arXiv: o intervalo de 3 s que a API pede não é respeitado → sem tarefa

Os termos de uso pedem "no more than one request every three seconds", numa só ligação de cada
vez (https://info.arxiv.org/help/api/tou.html, 2026-09-18). O módulo
`_arxiv` nunca teve throttle; com `_http` basta `min_interval_s=3.0` no seu `Api`. Não mudado na
R03 para não alargar o âmbito (a ficha pedia migrar os throttles que existiam).

### `uniprot_search` talvez peça um caminho que não é o de pesquisa → por verificar ao vivo

A pesquisa vai para `https://rest.uniprot.org/uniprotkb?query=…` (já antes da R03; a migração
manteve o fio). O agente que migrou o módulo lembra que o endpoint documentado é
`/uniprotkb/search?query=…`. Não confirmado: as páginas de ajuda e o Swagger da UniProt só
carregam com JavaScript e a R03 não faz pedidos à API. Verificação do dono:
`curl -s "https://rest.uniprot.org/uniprotkb?query=insulin&size=1" | head -c 300` contra o mesmo
pedido com `/uniprotkb/search`.

### `python_repl` só limita o expoente de `**`; o tamanho do resultado fica livre → sem tarefa

O avaliador recusa um expoente acima de 1000, mas não o tamanho do que calcula: `(10**1000)**1000`
passa o guarda e calcula um inteiro de 3,3 milhões de bits; mais um `**1000` e são 3,3 mil milhões,
com o GIL preso (o timeout do executor não o pára, D30 e D32). `'x' * 10**9` reserva 1 GB. Medido
em escala pequena: `(10**1000)**100` calcula-se num instante e depois esbarra no limite de 4300
dígitos do `str()`; `'x' * 10**7` também é instantâneo. Os casos grandes não se correram, porque
prenderiam a máquina. A tool está em `dangerous` e pede aprovação, por isso fica para o dono. O
`math_eval` já tem o guarda certo (a estimativa do tamanho do resultado, D32); partilhá-lo aqui
fecha a porta para `**` e `*`.

## 2026-09-30 · revisão dos docs contra o código (Claude, a pedido do dono)

Quatro passagens de agentes compararam cada página dos docs com o código e correram os exemplos
com sondas offline (FakeProvider, `prepare()` dos adaptadores, sockets bloqueados). Os docs passaram
a dizer o que o código faz; os defeitos do próprio código ficam aqui, todos sem tarefa salvo
indicação. Já estavam registados, e não se repetem: o `LLM("grok-…")` depois de um `asyncio.run`,
o batch à tarifa normal, as imagens que o xAI larga, o `thinking_effort` que o OpenAI ignora, o
`@tool(schema=)` com schema completo (C02b) e o `_inline_local_refs` (C02e).

### Núcleo e fornecedores

- **`inference_limit` e `RateLimitMiddleware` presos ao primeiro loop.** O `asyncio.Semaphore`
  (`core/_concurrency.py:46`) e o `asyncio.Lock` (`core/_rate_limit.py:36`) nascem fora do loop:
  um segundo `asyncio.run`/`*_sync` com contenção dentro do mesmo `with` dá `RuntimeError: … bound
  to a different event loop`. Com `requests_per_minute < 1` e sem `burst`, `burst=int(rpm)=0` e o
  `abefore` espera para sempre (`_rate_limit.py:31`).
- **`TracingMiddleware` deixa chaves no pedido.** Põe `_otel_span`/`_otel_start` em
  `request.kwargs` (`core/_telemetry.py:40-44`); com OpenTelemetry instalado, cada chamada avisa
  "Unknown parameter(s) ignored" (`_base.py:352-358`), e uma chamada que falha nunca fecha o span.
- **Batch.** `batch_submit` ignora os defaults do `LLM` (manda `max_tokens` 4096 e nenhuma
  `temperature`: `_openai.py:707`, `_anthropic.py:746`), o middleware e o retry; no batch da
  Anthropic o `parsed` nunca se preenche (`_anthropic.py:767`).
- **Fallbacks em string não herdam o `timeout=`**: `_normalize_fallbacks` só passa `api_key`,
  `base_url` e `provider`.
- **`claude-sonnet-5-5` não está nos perfis nem nos preços**: um `tool_choice` forçado passa no
  `prepare()` e daria 400; numa chamada medida dá `UnpricedModelError`.
- **Gemini, `count_tokens`** (`_gemini.py:454-469`): um prompt de sistema dá `RequestError` (o
  google-genai recusa `system_instruction` na Developer API) e as tools não se contam.
- **`thinking_budget=-1`** é recusado em `_llm.py:199`, embora o adaptador Gemini o aceite e o
  sugira na sua mensagem de erro (`_gemini.py:536-543`).
- **Partes e parâmetros.** Os adaptadores mandam `str(part)` para partes que não conhecem, em vez de
  `RequestError` (`_openai.py:209`, `_anthropic.py:202`). `stop`/`stop_sequences` não se traduzem
  entre fornecedores, e um `response_format` explícito é trocado em silêncio pelo `output_schema`
  (`_openai.py:604-611`). A cache explícita do Gemini (`cached_content`) e do OpenAI
  (`prompt_cache_key`, `prompt_cache_retention`) e o `safety_settings` do Gemini caem em "Unknown
  parameter". O `stop_reason` vem cru de cada fornecedor. O xAI devolve em `Response.model` o id
  pedido, não o que respondeu (`_xai.py:305`).
- **Estimativa do `reserve="strict"`.** `_request_size` (`core/_attempts.py:84-97`) só conta como
  não-texto as partes `dict`; `ImagePart`/`DocumentPart` são dataclasses, por isso a margem de
  4000 tokens do `HeuristicEstimator` nunca se aplica. A reserva também fica curta com o thinking
  da Anthropic (`_attempts.py:128` contra `_anthropic.py:617`).
- **Tecto brando com um a mais.** `limit_denial` (`core/_metering/_admission.py:207-219`) só nega
  quando o gasto passa o tecto: ao chegar exactamente a ele ainda deixa passar uma chamada
  (`max_total_tokens=15` → 2 chamadas, 30 tokens), e o `BudgetReport` marca-o em `>=`
  (`toolkit/budget/_report.py:14-27`).
- **Política de step.** `on_low_confidence="escalate"` (e o fallback por baixa confiança) salta o
  `max_cost` (`core/_step_engine.py:118-131`: custo 1.8 com tecto 1e-6, sem `cost_exceeded`).
  `on_timeout="halt"` não pára o flow com `on_exhausted="continue"`
  (`toolkit/flow/_executor.py:621-624`). A `Policy` aceita qualquer valor em `on_timeout`,
  `on_exhausted` e `on_low_confidence`.
- **`PricingRegistry` sem lock** — por confirmar: um `get()` em paralelo com `reset()`/`load()`
  pode fixar `None` no memo `_cache`; decide-o um teste com threads.
- **Coroutines públicas sem `_sync`**, contra a regra do `AGENTS.md`: `GraphStore` e as vistas,
  `Graph.to_dict`, `execute_step`, `execute_flow`, `Agent.iter`, `LLMModerator.moderate`,
  `OpenAIModerator.moderate`, `MemoryPreset.consolidate`, os métodos de `BruteForceIndex` e
  `VectorIndex`.
- **Docstrings desactualizadas:** `APIError` ("An HTTP error response"), o `todo` do `ServerTool`
  (diz que a config é ignorada; os cinco adaptadores levantam `RequestError`) e
  `toolkit/flow/_executor.py:790` ("enforced precisely").

### Tools

- **O validador não vê os elementos das listas**: `list[int]` aceita `["wrong"]`
  (`core/_tools/_validation.py:257` só verifica `isinstance(value, list)`).
- **`@tool` num método**: `ToolGroup(obj.m)` corre o wrapper sem `self` e dá `validation_error`
  "missing a required argument: 'self'", embora o schema omita o `self`.
- **`bind_arguments` corre depois das gates**: uma entrada de `schema=` para um nome que não é
  parâmetro passa a validação e só falha depois de o approval handler ser consultado.
- **`_failed`** (`core/_tools/_executor.py:451-468`) larga o `admitted.audit` (aprovação ou
  `GateModify`) quando a tool levanta ou esgota o prazo.
- **`_bounded` num erro** (`_executor.py:188-198`): a nota conta o texto com o prefixo "Tool error
  [tipo]: ", mas só a mensagem é cortada; com limite 40 o modelo recebe 68 caracteres e a nota diz
  "chars 0-40 of 128".
- **`DangerousToolGate`** manda usar `--allow-dangerous-tools`, uma flag que não existe
  (`core/_tools/_governance.py:131-133`).
- **`run_command`** aceita até 600 s, mas o `timeout_s` fica em 120: o executor desiste e o comando
  continua.
- **Redacção.** O fragmento "token" mascara `input_tokens`/`output_tokens` nos traces; escapam
  `export X=`, `access_token:`, camelCase, `bearer` em minúsculas, `mongodb+srv://` e `rediss://`.
- **Uma tool cujo pricer falha fica de graça** (`_executor.py:338-345`).
- **`max_calls` no caminho síncrono** conta sem lock (`_run_tool_sync` → `_spent`); com threads
  não se reproduziu (30 ensaios × 40 threads).
- **Pistas de instalação pelo PyPI**, onde o pacote não está: `_youtube.py`,
  `_providers/_imports.py:19`, `_tokens.py:49`, `graph/_networkx.py:15`, `resources/_codecs.py:89`,
  `resources/_serializers.py:67`, `agents/_manifest.py:353`, `prompts/_variables.py:84`,
  `prompts/_template_engines.py:65`.

### Agentes e flows

- **Tarefa multimodal em texto.** Fora de `react` e `completion`, as oito estratégias fazem
  `str(task)`: um `ImagePart`/`DocumentPart` entra no prompt como repr Python, com os bytes
  (`_plan_execute.py:143`, `_reflexion.py:146`, `_rewoo.py:163`, `_tot.py:192`, `_lats.py:263`,
  `_self_discovery.py:162`, `_llm_compiler.py:201`, `_generate_review.py:148`). Saídas: recusar
  tarefas que não são texto, ou propagar as partes.
- **ReWOO.** Ignora o `system` (nem a spec nem o parâmetro da factory chegam a uma chamada);
  substitui `#E1` como substring, o que estraga `#E10` (`_rewoo.py:97-102`; o `llm_compiler` já foi
  corrigido); ordena as evidências como strings (`#E10` antes de `#E2`).
- **Iterações a zero.** `Flow(max_iterations=0)` é aceite (`_flow.py:166`), e o `reflexion` aceita
  `max_retries: 0`: zero tentativas, zero chamadas, resposta vazia.
- **LATS.** `best_score` começa em 0.0 com `>` (`_lats.py:179-183`): com todas as notas a 0.0, o
  solver recebe "Best answer: " vazio.
- **Replan.** O `plan_execute` e o `llm_compiler` voltam a chamar o planner com as mesmas entradas,
  sem o feedback da falha; o `llm_compiler` detecta "REPLAN" como substring ("No need to replan"
  dispara um replan).
- **Chamadas sem uso:** o `reflexion` corre o `reflect` depois da última tentativa falhada, e o
  `lats` no último rollout.
- **`ReasoningSpec.from_mapping`** (`_spec.py:50-73`) aceita `timeout: "30"` (falha no build com
  `TypeError`), dá `TypeError` com `knobs: 5`, aceita `max_iterations: 0`, transforma `system: null`
  e `strategy: null` na string "None", e o mapping de `output_schema` ignora chaves desconhecidas. A
  spec não aceita `deepcopy`, `pickle` nem `asdict` (`MappingProxyType`) e não tem `to_dict`.
- **`Flow.as_step()`** põe o `scope` do flow aninhado no step que o envolve (`_flow.py:367`) e a
  execução aninhada aplica-o outra vez: `transform` e `enrich` correm duas vezes.
- **`State.merge(strategy="collect")`** (`_state.py:209-214`) estende a lista no sítio (a mudança
  aparece num snapshot anterior), perde um valor escalar que já lá estava e, com um só escritor,
  substitui em vez de acrescentar.
- **`ai-arch agent validate`** corre noutro processo: um manifesto com estratégia custom dá
  "unknown strategy".
- **`generate_review_flow`/`generate_review_initial_state`** não estão no topo nem em
  `ai_arch_toolkit.toolkit`, ao contrário das outras oito factories. Via `Agent`, o
  `generate_review` corre sempre ReAct interno, porque recebe sempre um `ToolGroup` (talvez
  intencional).

### Memória, grafo, prompts e recursos

- **Conteúdo em lista.** `_extract_text` (`toolkit/moderation/_middleware.py:96-106`) e
  `_extract_query` (`toolkit/memory/_middleware.py:90-100`) só lêem strings e dicts com `"text"`:
  com `user(["texto"])`, `user(["x", image(…)])` ou `user([cache(…)])` nada é moderado, e o
  `MemoryMiddleware` não injecta memórias. O teste `tests/moderation/test_middleware.py:142-146` só
  cobre dicts.
- **`MemoryPreset.consolidate()` apaga o que não é duplicado**: os nós sem valores escalares
  colidem na chave `""`, também entre tipos (`toolkit/memory/_presets.py`); cinco nós diferentes
  deram três removidos.
- **Grafo.** O índice de tipos fica velho depois de `get_subgraph()` + `add`, ou com um backend
  partilhado (`_facade.py:112-129`); `Graph.copy()` diz "deep copy" mas partilha metadados e
  conteúdo (`_store.py:210-221`); `NetworkXBackend.find_all_paths` repete o caminho por cada aresta
  paralela.
- **Serializers.** `TextSerializer` e `MarkdownSerializer` passam folhas não escalares ao
  `json.dumps` (`_serializers.py:84-93`): uma `date` de TOML ou YAML dá `TypeError` nu em vez de
  `ResourceSerializationError`.
- **`load_resources(extensions={"md"})`**, sem o ponto, não carrega nada e não avisa
  (`_resolver.py:152`), enquanto `register_codec(extensions=("md",))` aceita sem ponto.
- **Fingerprint com caminhos absolutos.** `PromptTemplate.fingerprint` inclui o
  `metadata["manifest"]` e a proveniência das secções (`_manifest.py:260-265`,
  `_templates.py:153-166`, `:361-368`): depende da máquina e, num manifesto `package://` em zip,
  muda a cada carga e aponta para um directório temporário já apagado.
- **`_infer_manifest_variables`** (`_manifest.py:685`) salta as secções `template:` com `select`:
  as variáveis do excerto nunca se inferem e o render falha sempre, a menos que estejam em
  `variables:`.
- **Por confirmar:** um selector com `${var}` grava o valor resolvido na proveniência
  (`_sources.py:123-135`), contra a regra "nomes, não valores"; no layout de texto, uma secção que
  só tem subsecções duplica o separador (`_layouts.py:176-189`).

### Exemplos

29 dos 47 exemplos não têm `from __future__ import annotations`; o `24` lista o `csv_read` como
seguro e escreve "All tools available" com 19 nomes; a docstring do `32` promete passos com `when`
que o código não usa; a do `36` fala de chaves da Anthropic e do xAI que o script não usa.

## 2026-10-02 · Frente O (Claude): achados fechados

- "OpenAI: `thinking_effort` sem `thinking=True` perde-se em silêncio" → resolvido pela D45: o
  esforço aplica-se sozinho no host oficial.
- "Os resultados de batch são preçados à tarifa normal" → resolvido: `_estimate_response_cost` e o
  `BaseProvider._answer` recebem `batch`, e os caminhos de batch do OpenAI (Responses e Chat
  Completions), dos servidores compatíveis e da Anthropic preçam à tarifa de batch. Testes que
  falharam antes com o dobro do custo.
- Encontrados e corrigidos na O04, sem entrada anterior:
  - a tabela de regras do OpenAI supunha que um modelo da geração actual não raciocina sem
    esforço enviado; o gpt-5, os 5.5, os 5.6, o o3 e os GPT-6 raciocinam e recusavam a
    `temperature` que o `LLM` manda sempre (400). Os esforços e o esforço por omissão de cada
    modelo passaram a vir de medições ao vivo;
  - o teste ao vivo do 4xx usava `max_tokens=10_000_000`, que a Responses aceita; passou a
    `top_logprobs=50`.
- `research/agent-strategies/02-react.md` (linhas 13 e 144) e `00-index.md` (linha 65) diziam que
  o OpenAI não reenvia o raciocínio → corrigidos com autorização do dono (2026-10-02), que levantou
  para isto a regra da R00 sobre `research/`.

## 2026-10-04 · Frente A (Claude): visto na A03

- **`~utilizador` que não existe faz `read_file`, `list_directory` e `search_files` levantar.**
  O `Path(path).expanduser()` de cada uma fica fora do `try`, e um `~nome` de um utilizador que o
  sistema não tem levanta `RuntimeError` ("Could not determine home directory"), contra a regra das
  tools do toolkit de devolver uma frase de erro. O executor apanha-o, mas a pessoa recebe "Tool
  execution failed". Reprodução: `read_file("~nao_existe/x")`. O `cwd` novo do `run_command`
  (A03) já devolve "Not a directory".
- **Um nome, uma tool, só no `ToolGroup`** (da revisão independente da A03). O `prepare_tools`
  manda nomes repetidos para `tools=[a, b]` ou `tools=[grupo1, grupo2]`, que os fornecedores
  recusam com um 400, e o `execute_tool`/`run_tools` com uma lista corre em silêncio a primeira
  que encontrar. Os fluxos dos agentes recebem um `ToolGroup` e estão protegidos.

## 2026-10-08 · Frente T (T06 a T09) e vaga 1 da frente C (C02, C06, C08): visto pelo caminho

Resolvidos nesta vaga, de entradas anteriores:

- "Um nome, uma tool, só no `ToolGroup`" (2026-10-04) → resolvido pela C02 (D62): dois nomes
  iguais numa lista ou entre grupos levantam `ValueError` no `prepare_tools`, no `execute_tool` e
  no `run_tools`, antes de qualquer envio.

Por decidir ou por confirmar, sem tarefa:

- **Brave, quota mensal.** O 429 da quota do mês (`QUOTA_LIMITED`) sai como `rate_limited`
  repetível, como o do segundo; a Brave não documenta os valores de `code`. Depois de um 429 sem
  `Retry-After`, o host descansa 1 s (`cooldown_s`), o que não chega para a quota. Precisa de uma
  verificação ao vivo do corpo e, talvez, de um leitor que distinga os dois.
- **GBIF, o `match` v1 está obsoleto.** O `gbif_species_match` usa `/v1/species/match`, que o GBIF
  marca como obsoleto a favor do serviço de matching v2. Pede uma ficha própria.
- **WHO GHO, a API OData foi anunciada como obsoleta** perto do fim de 2025, e não documenta
  `$count` (as listas pedem uma linha a mais para saber se há mais). Os comandos ao vivo da T08
  verificam se ainda responde.
- **USGS, `min_magnitude=0.0` por omissão** deixa de fora os sismos de magnitude negativa; e cada
  página do `earthquake_search` pede o `count` de novo.
- **World Bank, a pesquisa de indicadores** varre 10 a 30 páginas do catálogo por chamada; uma
  cache pedia expiração e um reset nos testes, fora do âmbito.
- **Fontes cujos corpos de erro vêm do código-fonte ou de gravações, não da documentação:** RCSB
  PDB e ChEMBL (T07b), Europe PMC `errMsg` e o cabeçalho `message` da NVD (T06; as páginas da
  Europe PMC, NCBI, NVD e Semantic Scholar não se leram daqui). Os comandos ao vivo das fichas
  confirmam-nos.
- **C06, o catálogo:**
  - a tabela de perfis da Anthropic não diz que modelos adaptativos pensam sem pedir, por isso o
    `thinking_mode` deles fica `None`;
  - o `claude-sonnet-5-5` e o `claude-haiku-5-5` não têm preço na tabela;
  - vários modelos da semente acabam entre Outubro e Dezembro de 2026 (gpt-image-1 a 23/10,
    Sonnet 4.5 a 30/11, e outros a 1/12);
  - a documentação do xAI contradiz-se sobre function calling no modelo multi-agente;
  - a Meta documenta 1 a 10 imagens de entrada numa edição; o adaptador aceita uma.
- **C02:** o `const` de um schema não é verificado (decisão do dono, se o quiser); o Gemini pode
  recusar `x-request-id` como nome de parâmetro (a C02f, ao vivo, di-lo).
- **`python_repl` e o `re`:** o mesmo congelamento do `regex_search` (D66) acontece numa regex que
  recua dentro do código que o `python_repl` avalia; o sandbox corre no processo.
- **O custo de ler um ficheiro enorme às janelas:** cada janela do `read_file` descodifica desde o
  início até ao `offset`, por isso ler um ficheiro de centenas de MB inteiro custa O(N²). Fica
  documentado na docstring; um salto por bytes em UTF-8 puro resolvia-o.
