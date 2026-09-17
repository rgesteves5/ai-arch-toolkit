# Decisões

Só acrescentar. Uma decisão revista ganha uma nova entrada que diz qual substitui.

## D1 · `Flow(policy=)` é a policy por omissão de cada step; `Flow(timeout=)` limita o run

- **Contexto:** a policy de um `Flow` só era aplicada quando o flow corria aninhado (`as_step()`),
  tornando inertes `ReasoningSpec.policy`, `ReasoningSpec.timeout` e `limits.timeout_seconds`.
  `docs/agents.md` já descrevia `policy` como "per-step" e `timeout` como "wall-clock".
- **Decisão:** policy efectiva de um step = `step.policy or flow.policy` (também para decidir se um
  erro pára o flow). Novo `Flow(timeout=)` limita o run inteiro. `ReasoningSpec.policy` → policy;
  `ReasoningSpec.timeout` e `limits.timeout_seconds` → timeout.
- **Consequência:** timeouts declarados passam a ser impostos (quebra visível para a app).

## D1a · Flows internos das estratégias não herdam a policy

- **Contexto:** reflexion, plan_execute, llm_compiler, lats, self_discovery e generate_review (com
  tools) correm `react_flow(...).run()` dentro de um step, sem policy.
- **Decisão:** não propagar. A policy aplica-se aos steps do flow da estratégia; um timeout por step
  limita o loop interno inteiro desse step. Limites por chamada LLM: `LLM(timeout=, retry=)`.
  Limite do run: `ReasoningSpec.timeout`.
- **Porquê:** propagar aplicaria o mesmo timeout com dois significados e duplicaria o retry de que o
  `LLM` já é dono. Documentado em `docs/agents.md`.

## D2 · O wrapper de `Flow.as_step()` não leva a policy do flow

- **Decisão:** o flow aninhado aplica a sua própria policy e timeout quando corre; o `Step` wrapper
  fica com `policy=None`. Evita aplicar a policy duas vezes.

## D3 · Prompts de sistema fundidos nos adaptadores (revisto)

- **Contexto:** os adaptadores escolhiam entre `system=` e as mensagens `system()`, perdendo uma.
  A versão inicial (fundir na camada `LLM`) moveria para o topo as mensagens `system()` a meio da
  conversa no OpenAI, Ollama e vLLM.
- **Decisão:** juntar sempre, dentro de cada adaptador: `system=` primeiro, depois as mensagens
  `system()` pela ordem. Anthropic, Gemini e xAI juntam no parâmetro nativo; o OpenAI põe `system=` à
  cabeça e mantém as mensagens `system()` na sua posição. Batch incluído.
- **Consequência:** quem usava `system=` para substituir a mensagem passa a enviar as duas.

## D4 · Tools de `toolkit.tools.dangerous` exigem aprovação

- **Decisão:** as cinco sem metadados ganham `capability` (`filesystem` ou `network`),
  `risk_level="high"` e `requires_approval=True`.
- **Consequência:** sem `approval_handler` passam a dar `approval_denied`.

## D5 · `iter()` mantém o nome e devolve um objecto de execução

- **Decisão:** `Flow.iter()` e `Agent.iter()` devolvem um objecto que é `AsyncIterator[FlowEvent]` e
  expõe `.result` (`FlowResult` / `AgentResult`) depois de drenado. Os `async for` existentes
  continuam a funcionar.

## D6 · Captura do trace por omissão: `"keys"`

- **Decisão:** `Flow(trace_capture="keys" | "full" | "none")`, por omissão `"keys"` — chaves em vez de
  valores. `"full"` faz cópias profundas.
- **Consequência:** quem lia valores de `StepTrace.input_state` tem de pedir `"full"`.

## D7 · Argumentos de tools validados e coagidos antes dos gates (revisto)

- **Contexto:** os tipos não eram verificados (`"1" + "2" == "12"`). Validar depois dos gates faria um
  humano aprovar chamadas que depois falham ou valores que depois mudam.
- **Decisão:** antes dos gates, validar e coagir contra o schema (strings numéricas e booleanas para o
  tipo declarado; `enum` verificado; arrays e objectos intactos); os gates vêem os valores coagidos.
  Depois de um `GateModify`, validar outra vez.

## D8 · Operação de metering de um stream: reservada na criação, iniciada na primeira tentativa

- **Contexto:** hoje é iniciada logo na criação. Com o middleware async a correr antes do provider,
  uma rejeição da moderação contaria como chamada com custo desconhecido e bloquearia um run com
  `max_cost`.
- **Decisão:** a admissão continua na criação; `mark_started()` só quando a primeira tentativa
  começa. Stream rejeitado pelo middleware, ou nunca iterado, liberta a reserva.
- **Consequência:** `tests/test_llm_metering.py` deixa de ver `llm_calls == 1` antes de iterar.

## D9 · Motor de execução: um gerador, steps em tasks, avanço a pedido

- **Contexto:** um motor numa `asyncio.Task` com fila sem limite continuaria a correr depois de um
  `break` sem `aclose()`. Hoje o flow só avança quando alguém pede o evento seguinte.
- **Decisão:** um único gerador assíncrono. Cada step corre numa task; os eventos dessas tasks saem
  em tempo real; entre steps nada avança sem o consumidor pedir; o `finally` cancela e aguarda as
  tasks pendentes. `run()` drena o gerador. O timeout do run usa esperas limitadas, nunca um cancel
  scope atravessado por `yield`.

## D10 · `run_tools` com `ToolGroup` usa só a governança do grupo

- **Decisão:** executa por `group.async_execute` / `group.execute`. Passar `approval_handler=` junto com
  um `ToolGroup` levanta `ValueError`, porque o handler pertence ao grupo.

## D11 · Meta pela Responses API do SDK `openai`

- **Contexto:** a Meta não tem SDK; manda usar o `openai` (Responses, Chat Completions) ou o
  `anthropic` (Messages). No Chat Completions a Meta apaga o raciocínio e não há `web_search`; a
  Messages exige bearer (o adaptador Anthropic manda `x-api-key`) e esse adaptador não reenvia o
  raciocínio.
- **Decisão:** `MetaProvider` próprio em `_meta.py`, sobre `client.responses.create`, com o extra
  `meta = ["openai>=2.6.0"]` (2.6.0 trouxe `responses.input_tokens.count`).
- **Consequência:** o adaptador OpenAI continua só com Chat Completions; a Meta não depende dele.

## D12 · Pedidos sem estado; o raciocínio viaja no `_raw`

- **Decisão:** `store: false` e `include: ["reasoning.encrypted_content"]` em todos os pedidos. Uma
  mensagem de assistente cujo `_raw` é uma resposta da Responses API e ainda coincide com o texto e
  as tool calls da mensagem é reenviada item a item (mesmo padrão do Gemini com as thought
  signatures). Se não coincide, ou o `_raw` é de outro fornecedor, é reconstruída sem raciocínio, com
  `phase: "commentary"` no texto que antecede tool calls. Itens de raciocínio no fim de um turno
  incompleto não são reenviados.
- **Consequência:** sem `previous_response_id` nem conversas guardadas na Meta; retries e
  fallbacks do `LLM` continuam a funcionar sobre o histórico local.

## D13 · Semântica exposta da Muse Spark

- **Decisão:** a Muse Spark raciocina sempre: `thinking_effort` aplica-se sozinho e `thinking=True`
  pede resumos (`reasoning.summary: "auto"`). Só existe `tool_choice="auto"`: `"none"` envia o
  pedido sem tools; forçar uma tool levanta `ValueError` antes da chamada. `output_schema` vai com
  `strict: false` (a Meta restringe a descodificação na mesma; `strict: true` recusa schemas
  Pydantic simples). O texto junta as mensagens com uma linha em branco, comentário incluído
  (como os preâmbulos do Anthropic); `parsed` vem da última mensagem que não é comentário.
  `stop_reason` é o da API: `completed`, ou o motivo de `incomplete_details`, ou `refusal`.

## D14 · Encaminhamento, chave e cabeçalhos

- **Decisão:** só o prefixo `muse-spark-` vai para `meta` (o Muse Glimmer é self-hosted e tem de
  cair no fallback de `base_url`). A chave vem só de `MODEL_API_KEY`, nunca de `OPENAI_API_KEY`;
  `base_url` é respeitado. O cliente limpa `organization`/`project` para os cabeçalhos
  `OpenAI-Organization`/`OpenAI-Project` lidos do ambiente não chegarem à Meta. A `temperature` passa
  tal como vem; a documentação recomenda `temperature=1.0` (a Meta afina o modelo para 1.0).

## D15 · Dependências: mínimos testados, tecto na versão maior seguinte

- **Contexto:** os mínimos declarados nunca foram testados e eram falsos (`pyyaml` 6.0 nem compila em
  Python 3.13). `anthropic` 1.x e `openai` 3.x trocaram o transporte para `httpx2`.
- **Decisão (2026-09-17):** cada mínimo é a versão mais baixa que o job `floors` do CI instala e
  testa (`--resolution lowest-direct`); cada SDK de fornecedor tem tecto na versão maior seguinte. O
  extra `dev` instala `all`, para cada mínimo ficar declarado uma vez.
- **Consequência:** apps presas a `anthropic` 0.x ou `openai` 1.x/2.x têm de actualizar esses SDKs.

## D16 · Um modelo sem preço não corre (decisão do dono, 2026-09-17)

- **Decisão:** se o toolkit não tem preço para o modelo pedido, a chamada falha com um erro que manda
  registar o preço (`pricing.register(...)`); um modelo local regista preço zero de forma explícita
  (nunca se infere do URL: um loopback pode ser proxy de um modelo pago). O projecto mantém os preços
  dos modelos novos dos fornecedores que suporta.
- **Por fixar na ficha:** se a regra vale sempre ou só com um `MeterScope` ligado. A correspondência
  passa a ser por id exacto ou sufixo de snapshot (plano de robustez, causa 3).

## D17 · Limites por omissão para qualquer tool, configuráveis por tool (decisão do dono, 2026-09-17)

- **Decisão:** o executor governado impõe limites de saída e de tempo a todas as tools, as do toolkit
  e as de quem usa a framework, com valores por omissão e override por tool. A falta de limites é da
  framework, não de cada tool. As tools perigosas que estão no namespace seguro passam para
  `dangerous`.

## D18 · Ordem dos adaptadores (decisão do dono, 2026-09-17)

- **Decisão:** OpenAI → xAI → Gemini → Meta → Anthropic, porque não há créditos Anthropic para a
  verificação ao vivo. (O dono escreveu "xai" duas vezes; assume-se que a segunda é a Meta.)

## D19 · `null` só onde o parâmetro admite `None`

- **Contexto:** a validação aceitava `None` em qualquer parâmetro, porque o schema não regista a
  opcionalidade (`width: int` recebia `None`).
- **Decisão (F24):** lê-se da assinatura: default `None`, anotação com `None`, ou sem anotação
  utilizável. Fora disso, `null` dá `validation_error` antes dos gates, também num parâmetro opcional
  com default diferente de `None`. Arrays e objectos continuam intactos (D7).

## D20 · As recomendações R1–R15 do plano de robustez valem como decididas

- **Contexto:** o dono aprovou a lista de refactors e pediu as três fases para agentes com contexto
  limpo; decisões em aberto seriam improviso garantido. O dono pode rever qualquer uma.
- **Decisão (2026-09-17):** valem as recomendações de `docs/internal/hardening-plan.md` §8, com estes
  acertos: R2 — 429 é `unbilled` em todos os fornecedores, as outras respostas de erro só onde o
  fornecedor o documenta, o resto fica incerto com tecto; R6 é a D16, e o erro por falta de preço só
  vale com um `MeterScope` ligado (sem scope a chamada corre e `Response.cost` fica `None`); R7 —
  somar (`max_tokens = budget + max_tokens`); R9 é a D17, com `max_output_chars=200_000` e
  `timeout_s=120` por omissão; R10 — os agentes nunca chamam fornecedores, o dono corre a verificação
  ao vivo; R11 — as tags são do dono; R14 — em vez de desenho aprovado antes do código, cada passo
  leva uma nota de desenho na ficha, sem esperar aprovação.
