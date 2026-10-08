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

## D21 · F25: IP explícito; a escolha de transporte de `ip_lookup` fica pendente

- **Contexto:** F25.2 pede validação local e deixa HTTPS em aberto, porque o serviço gratuito escolhido usa HTTP; as regras R00 proíbem rede externa nesta execução.
- **Decisão (2026-09-18):** exigir um endereço válido explícito antes de construir o URL e preservar o endpoint previsto pela ficha. A escolha de outro serviço/transporte pertence à R03 e fica registada em FINDINGS; não se afirma que o endpoint foi verificado hoje.
- **Consequência:** desaparece a divulgação implícita do IP da máquina; o risco do transporte HTTP continua conhecido e pendente.

## D22 · Construção na matriz R01 e fronteira de preparação R02

- **Contexto:** a interface BaseProvider actual junta construção/envio em complete() assíncrono; a R01 mantém essa interface e a R02 introduz prepare()+send(). O primeiro duplo da matriz lançava ValueError depois de contar um envio, embora a célula exigisse nenhum envio/operação: não representava construção pura.
- **Decisão:** a célula RequestError exercita a fronteira pura de preparação da fachada; afirma zero chamadas ao provider e zero operações, sem baixar o contrato. Um erro normalizado not_sent numa operação já iniciada mantém a contagem, como exige a disposição do meter. A função única dispatch é a costura onde a R02 separará a preparação específica dos adaptadores.
- **Consequência:** a R01 não afirma ter separado os interiores dos SDKs. A matriz mantém o cenário de construção e as recuperações seguintes/steps; a R02 amplia a prova aos erros de preparação dos seis adaptadores, sem mudar a pipeline.

## D23 · Vaga de inferência e delegação de streams

- **Contexto:** a pipeline exige adquirir vaga antes de marcar início, mas uma vaga mantida através de yields controlados pelo consumidor permite a um stream abandonado bloquear complete() e os steps seguintes.
- **Decisão:** complete mantém a vaga durante a chamada inteira; streams mantêm-na apenas no despacho e espera pelo primeiro item. A vaga é libertada antes de entregar esse item. Cada gerador que delega assume o fecho explícito do gerador delegado, por uma única ferramenta `_managed`.
- **Consequência:** cancelamento na fila não inicia operação; consumo lento não prende a vaga; abandono fecha imediatamente o transporte em vez de depender de GC. O teste existente de concorrência confirma ambos os contratos.

## D24 · Despacho e entrega dos erros dos fornecedores (R02, passo 3)

- **Contexto:** o tipo da excepção não diz se o pedido saiu. Um 200 ilegível (cobrado) e uma recusa
  local do SDK (nada enviado) saem ambos como `ValueError`/`TypeError` (prova em loopback, passo 2).
- **Decisão:** cada chamada tem um marcador de despacho num `ContextVar`. O hook `request` do cliente
  HTTP do SDK marca-o quando o pedido é entregue ao transporte; o xAI marca-o ao entrar no
  `send`/`open_stream`, porque todo o trabalho local está no `prepare`. `map_error(exc, *, sent)` é o
  único sítio que conhece as excepções do SDK: uma excepção desconhecida é `RequestError` antes do
  despacho e `ResponseError` depois. As falhas da fase de ligação (`ConnectError`, `ConnectTimeout`,
  `PoolTimeout`) são `not_sent`; o 429 é `unbilled` em todos (D20); o resto segue a página de cada
  fornecedor, citada no mapeador; sem documento, `indeterminate`. No gRPC, `UNAVAILABLE` cobre
  também uma ligação que nunca se fez, mas o gRPC não diz se o pedido saiu, por isso fica
  `indeterminate`.
- **Consequência:** nenhuma excepção de SDK, `httpx`/`httpx2`, `aiohttp` ou gRPC sai crua (regra de
  arquitectura com canário); o meter lê a entrega do erro e não o seu tipo.

## D25 · xAI: esforço por modelo e mínimo do SDK (R02, passo 3)

- **Contexto:** o adaptador ignorava com aviso todo o `thinking`/`thinking_effort`, mas o `grok-4.6`,
  o `grok-4.5` e o `grok-4.3` documentam `reasoning_effort`; o `xai-sdk` 1.7 declarado não tinha
  `medium`, `xhigh`, `agent_count` nem `cost_usd`.
- **Decisão:** os modelos Grok de raciocínio raciocinam sem pedir, por isso `thinking_effort`
  aplica-se sozinho (como a Muse Spark, D13) e é validado contra os esforços da página do modelo;
  `thinking=True` não pede nada a mais, salvo num modelo que não raciocina, onde levanta
  `RequestError`, como o esforço num modelo que não o aceita. Um modelo novo recebe as regras da
  geração actual; os ids retirados seguem o modelo que os serve. O extra `xai` passa a
  `xai-sdk>=1.18,<2`, sem ramos por versão (D15).
- **Consequência:** `thinking_effort` num `grok-4.20-reasoning` ou num `grok-build-0.1`, que antes
  era ignorado, passa a falhar antes do pedido; quem usa `xai-sdk` < 1.18 tem de actualizar.

## D26 · Uma falha com usage reportado é liquidada com esse usage (R02, passo 3)

- **Contexto:** a Meta devolve o usage de uma resposta que falhou (`response.failed`), e o meter
  liquidava toda a falha pela entrega: custo desconhecido com tecto.
- **Decisão:** `ProviderError.usage` leva o usage que o fornecedor reportou para o pedido falhado;
  a pipeline regista-o na tentativa (`Response.attempts`) e chama
  `MeterOperation.fail(delivery, usage=, cost=)` com o custo do pricer do scope, calculado fora do
  lock como na liquidação normal. Sem usage, nada muda.
- **Consequência:** uma falha com usage conta tokens e custo conhecido; `fail` recusa um custo
  estimado, como `settle`.

## D27 · Anthropic: thinking pelas tabelas documentadas (R02, passo 3)

- **Decisão:** nos modelos adaptativos, `thinking=True` envia `{type: "adaptive", display:
  "summarized"}` (o `display` por omissão esconde o texto nos modelos novos) e `thinking=False` não
  envia nada, por isso um modelo que pensa por omissão continua a pensar. `thinking_effort` vai em
  `output_config.effort`, sozinho e fundido com o `format`, validado pela tabela de esforços. Nas
  famílias antigas, que só aceitam `budget_tokens`, um esforço ou um orçamento liga o thinking, com
  `max_tokens = orçamento + max_tokens` (D20). O `tool_choice` forçado é recusado onde a API o
  recusa (Fable 5.1, Mythos 5.1). Os pedidos falhados são `unbilled`, como a Anthropic documenta; um
  erro dentro do stream fica `indeterminate`.
- **Consequência:** o `claude-sonnet-4-6` passa de `budget_tokens` a adaptativo; um `top_p`/`top_k`
  explícito num modelo sem amostragem falha antes do pedido.

## D28 · `ip_lookup` passa para o ipwho.is, em https (R03, passo 1; resolve a pendência da D21)

- **Contexto:** a D21 deixou o transporte para a R03. O endpoint gratuito do ip-api só serve `http`
  e proíbe uso comercial (https://ip-api.com/docs/api:json, lido a 2026-09-18).
- **Decisão:** `https://ipwho.is/{ip}`: gratuito, sem chave, uso comercial permitido, 1000 pedidos
  por dia por IP de cliente (https://ipwhois.io/documentation, lido a 2026-09-18). A tool mantém as
  mesmas linhas de saída; os campos lidos são os da página (`success`, `message`, `city`, `region`,
  `country`, `latitude`, `longitude`, `timezone.id`, `connection.isp`, `connection.org`).
- **Consequência:** os IPs consultados vão para outro serviço; não foi verificado ao vivo (comando no
  relatório final da ficha).

## D29 · Limites das tools: por tool, e o grupo só aperta (R03, passo 1)

- **Decisão:** `ToolRuntimePolicy.max_output_chars` (200 000) e `timeout_s` (120), declarados com
  `@tool(...)`; `None` desliga (D17, D20). `ToolGroup(max_output_chars=, timeout_s=)` é um tecto
  para todas as tools do grupo: vale o mais estrito dos dois; `None` no grupo quer dizer sem tecto.
- **Consequência:** a app aperta um grupo inteiro de uma vez; para alargar, declara-o na própria
  tool. O corte fica no texto e em `ToolResult.metadata["truncated"]`.

## D30 · O executor corre as tools síncronas numa thread daemon própria (R03, passo 1)

- **Contexto:** só se impõe prazo a uma tool síncrona se ela correr noutra thread. O
  `asyncio.to_thread` usa o executor por omissão, cujas threads o `asyncio.run` espera até 300 s e o
  fim do processo espera sem prazo.
- **Decisão:** um só caminho, `_invoke`: uma coroutine é cancelada no prazo; uma função síncrona
  corre numa thread daemon com o contexto copiado, e o executor deixa de esperar. O caminho síncrono
  (`execute`, `run_tools_sync`) corre `_invoke` num loop seu.
- **Consequência:** uma tool síncrona chamada pelo caminho síncrono deixa de correr na thread de quem
  chama. Uma tool que passou do prazo corre até ao fim sem prender o processo.

## D31 · Cada resposta é lida dentro do `_http` (R03, passo 1)

- **Contexto:** a escada de `except` de cada tool protegia o parse com `TypeError`, mas não com
  `AttributeError`: 93 tools levantavam com corpos inesperados.
- **Decisão:** todo o pedido do `_http` recebe a função que lê a resposta (`parse=`). O que ela
  levantar por uma forma inesperada (`AttributeError`, `LookupError`, `TypeError`, `ValueError`,
  `ArithmeticError`, `RecursionError`, `SyntaxError`) vira `HttpError("could not parse API
  response: …")`. Nenhum dado cru sai do `_http`.
- **Consequência:** uma tool não rebenta com um corpo que não previu; o teste de invariantes prova-o
  com corpos construídos a partir das chaves que cada módulo lê. Um erro que não é de forma continua
  a propagar.

## D32 · Tools de cálculo: o que prende o GIL recusa-se à entrada (R03, passo 1)

- **Contexto:** medido que um regex catastrófico e a aritmética de inteiros enormes prendem o GIL;
  o timeout do executor não os pára.
- **Decisão:** `math_eval` estima o resultado de `**`, `pow`, `*`, `factorial` e `round` antes de o
  calcular (até 15 000 bits; expressão até 1000 caracteres). `regex_search` aceita padrão até 500
  caracteres e texto até 20 000, recusa as formas exponenciais e as referências para trás, e mostra
  até 1000 correspondências.
- **Consequência:** `9**9**9`, `factorial(10**7)`, `(a+)+$` e `(a|aa)+$` dão erro logo. O custo
  polinomial de um regex continua possível: fica em `FINDINGS.md` para o dono decidir.

## D33 · As vagas do motor lêem o snapshot da vaga, sem `fork()` (R03, passo 2)

- **Contexto:** a vaga de um step lia o estado vivo e a vaga paralela um `fork()` (cópia profunda)
  por irmão: o mesmo step comportava-se de duas maneiras com uma escrita in-place, e a cópia por
  step é o custo que a R15 manda não trazer.
- **Decisão:** um só executor de vagas; todas as vagas lêem o `state.snapshot()` tirado à entrada
  (camadas copiadas, valores partilhados). Os irmãos continuam sem ver os artefactos uns dos outros,
  porque a fusão é no fim da vaga, pela ordem de declaração. O trace regista cada step quando acaba,
  pela ordem dos `step_end`.
- **Consequência:** uma escrita in-place num valor lido passa a ficar no estado em todos os modos
  (antes perdia-se numa vaga paralela). O passo 3 escreve o contrato (vista só de leitura) e testa-o.
  Mudança visível no `CHANGELOG`.

## D34 · `budget_policy` fica nas factories, como opção do `Flow` (R03, passo 4.2)

- **Contexto:** as nove factories aceitam `budget_policy` e os builders nunca o passam. O plano
  (secção 10) chama-lhe um segundo caminho para o mesmo budget.
- **Decisão:** fica. É uma das quatro opções do `Flow` (`FlowOptions`) que a factory passa sem lhes
  tocar. É o budget por omissão das corridas desse fluxo, e o `run(budget_policy=)` substitui-o;
  num fluxo encaixado vale o scope de fora. Não é um segundo mecanismo: é o parâmetro do `Flow`,
  agora declarado uma vez. Os builders não o passam porque o `ReasoningSpec` não tem budget (o
  `Agent` recebe-o por corrida).
- **Consequência:** nenhuma quebra: tirá-lo quebrava a API pública das factories sem ganho, e
  `docs/agents.md` e `docs/safety.md` já descrevem os dois sítios (construção e corrida).

## D35 · A resposta de um agente está em `answer`, ou é o valor do último step (R03, passo 4.3)

- **Contexto:** o `extract_text` adivinhava a resposta por quatro chaves seguidas (`answer`,
  `response`/`last_response`, `last_answer`) e depois o valor do último step, porque cada estratégia
  deixava a resposta à sua maneira.
- **Decisão:**
  - Cada estratégia do toolkit deixa a resposta em `answer` (texto) e a `Response` de onde veio em
    `response`, também quando esgota as voltas.
  - O runner lê `answer`. Sem `answer`, vale o valor do último step, como o valor de um fluxo
    encaixado (`as_step()`).
  - As chaves partilhadas vivem num só módulo (`flows/_keys.py`).
- **Consequência:**
  - Um `answer` vazio conta como resposta.
  - Um fluxo feito à mão que queira dar a sua resposta escreve `answer`; sem ela, responde o último
    step.
  - O `AgentResult.response` vem da mesma fonte que o texto.
  - Nenhuma chave de estado desaparece. Mudança visível no `CHANGELOG`.

## D36 · Cada manifesto tem uma declaração, e o JSON Schema gera-se dela (R03, passo 4.4)

- **Contexto:** os manifestos de agentes e de prompts validavam-se à mão, em código de leitura, e
  o JSON Schema empacotado dos prompts era um terceiro texto, mais largo do que o loader, que
  ninguém verificava.
- **Decisão:**
  - A fonte da forma de cada manifesto é uma declaração em Python (`toolkit/_shape.py`), da qual
    saem a verificação do loader e o JSON Schema publicado.
  - O loader não precisa do `jsonschema`, que é opcional; os testes usam-no para provar que o
    schema publicado diz a verdade.
  - O manifesto de agentes ganha um JSON Schema empacotado.
- **Consequência:**
  - As folgas medidas fecham-se:
    - `version: true`;
    - um booleano como número;
    - campos ignorados em silêncio;
    - um objecto onde se espera uma lista;
    - o crash do `trace_capture`.
  - As mensagens de forma ganham um só formato. Mudanças visíveis no `CHANGELOG`.

## D37 · As falhas das tools são `ToolError` tipados, e o executor trata-as como falhas (contrato das tools)

- **Contexto:** as tools devolvem strings de erro, com seis ou mais formulações, e o executor
  embrulha qualquer valor em `ToolResult.success`. O meter liquida a chamada como sucesso, nenhum
  fornecedor recebe o sinal de erro e o ai-network mostrou `ok: true` para páginas que não existiam
  (`docs/internal/tools-contract-plan.md`, 3.1).
- **Decisão:**
  - Uma tool que não consegue responder lança `ToolError` com um tipo de um conjunto fechado
    (`not_found`, `invalid_argument`, `upstream`, `rate_limited`), uma mensagem com o motivo da
    fonte e o passo seguinte, e o `retryable` certo.
  - O executor, que já converte excepções em falhas sem parar o agente, usa esse tipo em vez de
    `runtime_error`; o `tool_result()` leva o erro, e cada adaptador passa-o como o fornecedor o
    aceita (o `is_error` do Anthropic).
  - O `HttpError` passa a ser um `ToolError`.
  - Zero resultados não é falha: é um sucesso que o diz, com a consulta.
- **Consequência:**
  - Desaparecem as strings de erro feitas à mão e o `try/except HttpError` de cada tool.
  - A regra do `AGENTS.md` ("Toolkit tools return error strings instead of raising") muda quando a
    costura entrar.
  - Quem chama a função crua, fora do executor, passa a ver excepções. Mudança visível no
    `CHANGELOG`, com aviso ao ai-network.

## D38 · A porta HTTP valida a resposta de cada fonte (contrato das tools)

- **Contexto:** o `_http` trata qualquer 2xx como sucesso e não entrega estado nem cabeçalhos às
  tools; o corpo dos erros perde-se; 12 módulos traduzem qualquer 404 por "no matching records
  found."; os erros dentro de um 200 (MediaWiki `error`, World Bank `message`, Overpass `remark`,
  texto do GDELT, 204 da RCSB) viram "sem resultados" (plano, 3.2 e anexo C).
- **Decisão:**
  - Cada `Api` declara uma vez como a sua fonte sinaliza erro: uma função sobre estado, cabeçalhos
    e corpo, corrida antes do `parse=`.
  - O que um 404 quer dizer é declarado por endpoint: num recurso é `not_found`; numa pesquisa ou
    listagem é `upstream`, porque o endpoint mudou.
  - A mensagem leva o código e o texto da fonte.
- **Consequência:** desaparecem os `status_messages={404: …}` e as guardas `isinstance` que nunca
  disparam. Um endpoint mudado deixa de parecer "sem resultados", como o `uniprot_search` antes de
  28/09.

## D39 · Nenhum corte é beco sem saída: uma primitiva de janela para texto e listas (contrato das tools)

- **Contexto:** 94 das 132 tools cortam e 60 não deixam ler o resto; cerca de 72 cortes `[:N]` são
  silenciosos, e os helpers de corte estão copiados em 15 módulos sem dizer o tamanho original
  (plano, 3.3 e anexo A).
- **Decisão:**
  - Todo o corte passa por uma primitiva: texto por caracteres, listas por itens. Ela escreve um
    rodapé com o que foi mostrado, o total e a chamada exacta para o resto, e preenche `metadata`
    (`truncated`, `total`, `next`).
  - Documentos navegam-se por `section`, `offset` e `find` (a janela à volta de cada ocorrência); o
    índice de secções dá o tamanho de cada uma.
  - Listas paginam por `offset` ou `cursor`, como a fonte; com total quando a fonte o dá, e "há mais"
    quando não dá.
  - Os tectos actuais ficam. Nas 72 chamadas a `mediawiki_page` registadas no ai-network, o agente
    pediu sempre 3000 ou 4000 caracteres, nunca o valor por omissão: um tecto maior seria contexto
    reenviado a cada volta. A navegação substitui o tamanho.
  - Um campo de um registo (um resumo, uma descrição) não se corta abaixo da janela; as listas
    dentro de um registo dizem quantas faltam.
- **Consequência:** desaparecem os helpers copiados e os cortes soltos. A invariante de contrato
  prova que a continuação de cada rodapé devolve a janela seguinte.

## D40 · A MediaWiki lê-se pelo HTML que o servidor renderiza (contrato das tools)

- **Contexto:** o `_clean_wikitext` limpa wikitexto com uma regex de uma só passagem: ficam as
  predefinições exteriores, a marcação das tabelas (369 linhas `|-` e 253 `rowspan` na página dos
  Nobel da Física) e os parâmetros das imagens (`80px`). O TextExtracts do `wikipedia_article` deita
  fora as tabelas: o texto completo dessa página não tem nenhum laureado.
- **Decisão:** `action=parse&prop=text`, com as predefinições já expandidas e `section=N`,
  convertido para texto pelo `html.parser` da stdlib. As tabelas saem em linhas, com `rowspan` e
  `colspan` resolvidos; referências, caixas de navegação, `<style>` e ligações de edição ficam de
  fora.
- **Consequência:** desaparece o `_clean_wikitext`. O mesmo conversor serve a Wikipedia, o
  Wiktionary e os outros wikis da Wikimedia.

## D41 · Tools repetidas sobre a mesma fonte fundem-se, sem aliases (contrato das tools)

- **Contexto:** dez grupos de tools fazem o mesmo trabalho sobre a mesma fonte com limites,
  validações e saídas diferentes; as tools MediaWiki usam o Wiktionary por omissão (plano, anexo E).
- **Decisão:** uma família por fonte, com o mesmo prefixo e um trabalho por tool; a família wiki usa
  a Wikipedia por omissão. As tools substituídas saem sem aliases: o pacote é pré-1.0 e não se
  mantêm dois caminhos.
- **Consequência:** mudança incompatível. O `CHANGELOG` traz a tabela de migração, e o ai-network,
  que chama as tools pelo nome, recebe o aviso antes.

## D42 · A excepção das tools chama-se `ToolFailure` e leva o `ToolError` (refina D37)

- **Contexto:** a D37 escreveu "as tools lançam `ToolError`", mas `ToolError` já é o registo público
  (dataclass congelada, exportada no topo) que o `ToolResult` leva no campo `error`. Uma dataclass
  congelada não pode ser uma excepção (o Python escreve `__traceback__` ao lançar), e dar dois
  significados ao mesmo nome seria pior.
- **Decisão:**
  - A excepção chama-se `ToolFailure(Exception)` e leva um `ToolError` em `.error`; o executor
    devolve `ToolResult(ok=False, error=exc.error)`, com a mensagem redigida como hoje.
  - Os tipos que as tools lançam são `not_found`, `validation_error`, `upstream` e `rate_limited`.
    Um argumento inválido usa o `validation_error` que o executor já dá quando o schema falha: para
    o agente é o mesmo assunto.
  - O `tool_result()` ganha `is_error`, e o adaptador Anthropic manda-o no bloco `tool_result`. Nos
    outros fornecedores, o `Tool error [type]: …` do `to_model_text()` já o diz no texto.
  - O `HttpError` da porta é uma subclasse de `ToolFailure`.
- **Consequência:** a D37 lê-se com estes nomes; o `ToolError` não muda.
- **Adenda (T01, 2026-10-05):** o Gemini também tem onde o pôr. O `FunctionResponse.response` lê
  a chave `error` como "error details" (docstring do `google-genai`), e o adaptador põe lá o
  resultado de uma chamada que falhou. A Responses API, a Chat Completions e o xAI continuam só com
  o texto.
- **Adenda (D64, C07.9, 2026-10-08):** o `ToolFailureType` ganha `permission_denied`, para um
  caminho que a `FilesystemPolicy` recusa: a mesma palavra no gate e na tool. Entra com a C07.

## D43 · OpenAI pela Responses API no host oficial; Chat Completions só para servidores compatíveis (frente O)

- **Contexto:** desde o GPT-5.4 a Chat Completions só aceita tool calls com `reasoning_effort` a
  `none` (https://developers.openai.com/api/docs/guides/migrate-to-responses). O GPT-6 Astra e o
  GPT-6.1 Sol não têm `none`, e a Chat Completions "is supported without tool calling" para o
  6.1 Sol (https://developers.openai.com/api/docs/models/gpt-6.1-sol). A GPT-6 Luna e o GPT-6 Sol
  só chamam tools sem raciocinar. Por isso, dos três modelos de topo do catálogo de 2026-10-01
  (Astra, 6.1 Sol, Luna), nenhum corre aqui um agente com tools a raciocinar. A OpenAI mantém a
  Chat Completions ("remains supported"), mas recomenda a Responses "for all new projects". Só a
  Responses dá resumos do raciocínio, raciocínio cifrado entre voltas e as tools alojadas. O
  adaptador da Meta já é Responses sobre o mesmo SDK (D11, D12). O LOG de 2026-09-28 tinha deixado
  a porta fora de âmbito pelos custos relatados; a O01 mede-os.
- **Decisão:**
  - Os modelos no host oficial (sem `base_url`, ou com `api.openai.com`) vão pela Responses, sem
    estado (`store: false`), com o raciocínio cifrado reenviado do `_raw`, como a Meta (D12).
  - A Chat Completions fica para os servidores compatíveis (um `base_url` de outro host: Ollama,
    LM Studio, vLLM…), sem as regras dos modelos OpenAI, como hoje.
  - A API pública não ganha escolha de endpoint, nem como nome de fornecedor nem como opção: o host
    decide. O endpoint é um pormenor de implementação (`docs/internal/api-semantics-audit.md`, §6).
  - O que é da Responses e não da Meta passa para um núcleo partilhado, e OpenAI e Meta ficam dois
    perfis sobre ele.
  - Um `_raw` só se reenvia ao mesmo fornecedor e à mesma família de modelos ("Persisted reasoning
    can be reused only within the same model family",
    https://developers.openai.com/api/docs/guides/reasoning). Fora disso, a mensagem reconstrói-se
    sem raciocínio.
  - No host oficial, as kwargs que a Responses não tem (`stop`, `seed`, `frequency_penalty`,
    `presence_penalty`) levantam `RequestError`; os servidores compatíveis continuam a recebê-las.
- **Alternativas rejeitadas:**
  - Ficar só com a Chat Completions: os agentes não raciocinam com tools em nenhum modelo de topo
    da OpenAI.
  - As duas APIs para os mesmos modelos, à escolha: duplica regras e testes para ganhar quatro
    parâmetros, e mete o endpoint na API pública.
  - Tudo pela Responses, servidores compatíveis incluídos: a maioria fala sobretudo Chat
    Completions.
- **Consequência:**
  - Quebra visível no host oficial: as quatro kwargs.
  - Passa a funcionar: `thinking=True` com tools do GPT-5.4 em diante; o Astra e o 6.1 Sol chamam
    tools; o OpenAI dá resumos do raciocínio.
  - Substitui a consequência da D11 ("o adaptador OpenAI continua só com Chat Completions") e a
    linha do OpenAI no `AGENTS.md`.
  - Nada sai antes da sonda da O01. Se a latência medida for claramente pior, decide o dono, com os
    números à frente.

## D44 · As três escolhas da O03 (frente O)

- **Contexto:** a ficha O03 deixava três escolhas ao dono. Em 2026-10-02 o dono pediu a frente O
  implementada até ao fim e deixou as outras frentes paradas; o coordenador fixou-as por
  delegação, com as propostas da ficha.
- **Decisão:**
  1. No host oficial, um `response_format` cru levanta `RequestError`: a maneira é o
     `output_schema` ou o `json_mode`.
  2. Na Responses, o OpenAI leva a `web_search()` sem config, pelo mesmo mapa da Meta
     (`hosted_tools`). Uma config, ou outra server tool, continua a levantar. A proposta era
     deixar tudo para a C05, mas a C05 não corre agora.
  3. Os servidores compatíveis ficam na Chat Completions, mesmo os que já falam Responses, até
     alguém pedir.
- **Consequência:** a O03 fica sem escolhas abertas; a config tipada das server tools continua na
  C05.

## D45 · No OpenAI, o `thinking_effort` aplica-se sozinho (frente O)

- **Contexto:** o achado de 2026-09-18 ("o OpenAI ignora o `thinking_effort` sem `thinking=True`,
  sem aviso") esperava decisão do dono. Com a Responses ficou claro que é um erro: os GPT-5, 5.5,
  5.6, o3 e GPT-6 raciocinam a `medium` sem esforço enviado (medido a 2026-10-02), e quem passava
  `thinking_effort="none"` para os desligar ficava a raciocinar sem saber. Os outros quatro
  fornecedores já aplicam o esforço sozinho (D13, D25). O dono pediu, a 2026-10-02, que os bugs
  encontrados pelo caminho fossem corrigidos.
- **Decisão:** no host oficial, o `thinking_effort` vai como `reasoning.effort` com ou sem
  `thinking`, verificado contra os esforços do modelo; um modelo que não raciocina levanta
  `RequestError`. O `thinking=True` acrescenta o resumo (`summary: "auto"`), com o esforço dado
  ou `high`. Os servidores compatíveis não mudam.
- **Consequência:** muda o fio de quem passava um esforço sem `thinking` (fica no `CHANGELOG`,
  Fixed). Verificado ao vivo: `thinking_effort="none"` sozinho baixa a saída do `gpt-5.5` de 35
  para 5 tokens.

## D46 · Gerar imagens: `LLM.generate_image()` e `Response.images` (frente I)

- **Contexto:**
  - O ai-network pede ao toolkit que um modelo de imagem devolva a imagem em bytes, com o mime, e
    com o custo no meter (a G-36 dele; a E06-09 espera por isto).
  - Hoje:
    - o `LLM` não tem chamada para modelos de imagem;
    - o Gemini deita fora as partes `inline_data` (`_gemini.py`, `_parse_sdk_response`);
    - o núcleo da Responses ignora os itens `image_generation_call`;
    - a `Response` não tem onde guardar uma imagem.
  - Os fornecedores geram imagens de maneiras diferentes (documentação oficial lida a 2026-10-02):
    - **OpenAI:**
      - Images API: `/v1/images/generations` e `/edits`, com `gpt-image-2.5-sunburst`/`-flare`,
        `gpt-image-2`, `1.5`, `1` e `1-mini`. Devolve sempre base64 e um `usage` em tokens de
        imagem (https://developers.openai.com/api/docs/guides/image-generation,
        https://developers.openai.com/api/reference/resources/images).
      - A ferramenta `image_generation` da Responses, que o modelo chama a meio do turno
        (https://developers.openai.com/api/docs/guides/tools-image-generation).
    - **Gemini:**
      - `generate_content` num modelo de imagem (`gemini-3.1-flash-image`,
        `-flash-lite-image`, `gemini-3-pro-image`), com as imagens em partes `inline_data`.
      - Os modelos de imagem não chamam funções, e os de texto não geram imagens.
      - O Imagen foi desligado a 2026-08-17, e o `gemini-2.5-flash-image` a 2026-10-02
        (https://ai.google.dev/gemini-api/docs/generate-content/image-generation,
        https://ai.google.dev/gemini-api/docs/deprecations).
    - **xAI:** o `image.sample` do `xai-sdk` (`grok-imagine-image-2.0`), com o custo exacto na
      resposta, e uma server tool no turno
      (https://docs.x.ai/developers/model-capabilities/images/generation,
      https://docs.x.ai/developers/tools/image-generation).
    - **Meta:** o modelo `muse-image-1.0`, pela Responses API, a $0.01 por imagem. O Muse Spark
      só dá texto (https://dev.meta.ai/docs/image-generation).
    - **Anthropic:** o Claude não gera imagens
      (https://platform.claude.com/docs/en/build-with-claude/vision).
- **Decisão** (o dono aceitou as recomendações a 2026-10-03):
  - `LLM.generate_image()`, com `_sync`, para os modelos de imagem. O modelo do `LLM` é o modelo
    de imagem: `LLM("gpt-image-2.5-flare")`.
  - Devolve a `Response` de sempre, com um campo novo `images: tuple[GeneratedImage, ...]`
    (`data: bytes`, `media_type`, `revised_prompt`).
  - Corre pelo mesmo `Execution` do `complete`. O charge site é o mesmo, e o meter, os retries,
    os fallbacks, o middleware e os attempts ficam como estão.
  - Cada adaptador traduz o pedido no `prepare` e escolhe o endpoint no `send`:
    - o OpenAI pela Images API, ou pelo `edit` quando há `images=`;
    - o Gemini por `generate_content` com `response_modalities`;
    - o xAI por `image.sample`;
    - a Meta pela Responses, no `muse-image`.
  - A Anthropic e os servidores compatíveis recusam antes de qualquer charge, como no
    `count_tokens`.
  - O custo:
    - o `Usage` ganha contadores de tokens de imagem, disjuntos como os outros;
    - o `ModelPricing` ganha tarifas de imagem;
    - onde o fornecedor dá o custo (xAI), vale o `provider_cost`;
    - onde o preço é por imagem (Meta), há uma tarifa por imagem.
  - A seguir, as imagens também numa resposta normal:
    - o `Response.images` preenchido no `complete` e no `stream` (o Gemini, e os
      `image_generation_call` no núcleo da Responses);
    - uma server tool `image_generation()` para o OpenAI;
    - um `StreamEvent` `image` para as imagens parciais.
  - Uma só frente. Primeiro a sonda e o `generate_image` nos quatro fornecedores, que desbloqueia
    o ai-network; depois as imagens no turno.
- **Alternativas rejeitadas:**
  - Uma `ImageResponse` à parte: duplicava o caminho do meter, dos retries e dos attempts.
  - Só as imagens no turno: o Gemini e a Meta não geram imagens num turno de agente, e a E06-09
    quer uma capacidade que a app chama.
  - O Imagen: está desligado.
- **Consequência:**
  - O `AGENTS.md` passa a dizer que há charges também no `generate_image`, no mesmo sítio.
  - O `StreamEvent.kind` ganha `"image"` (I04): quem faz `match` exaustivo sobre ele vê um caso
    novo.
  - O resto é aditivo.

## D47 · Os parâmetros portáveis do `generate_image` (frente I)

- **Contexto:**
  - Cada fornecedor mede o tamanho à sua maneira:
    - o OpenAI com `size` em `WxH` (arbitrário, em múltiplos de 16, no gpt-image-2 e no 2.5; três
      tamanhos fixos nos anteriores);
    - o Gemini com `aspect_ratio` + `image_size` (`512`, `1K`, `2K`, `4K`);
    - o xAI com `aspect_ratio` + `resolution` (`1k`, `2k`);
    - a Meta com `size`, que só fixa a proporção.
  - A qualidade só existe no OpenAI (`low` a `max`) e no xAI (`low`, `medium`). As fontes estão na
    D46.
- **Decisão** (o dono aceitou as recomendações a 2026-10-03):
  - A assinatura:
    `generate_image(prompt, *, images=(), n=1, aspect_ratio=None, resolution=None, quality=None, output_format=None)`.
  - `images` são `ImagePart` (os de `image()`), para editar.
  - Os valores:
    - `aspect_ratio` escreve-se `"16:9"`;
    - `resolution` é `"512"`, `"1K"`, `"2K"` ou `"4K"`;
    - no OpenAI, os dois juntos dão o `size` em `WxH`;
    - `quality` leva os valores do fornecedor;
    - `output_format` é `png`, `jpeg` ou `webp`.
  - Cada modelo valida no `prepare`, pela sua tabela de regras (resolvida pelo `_model_id.py`), e
    levanta `RequestError` para o que não aceita. Por exemplo: uma proporção que o modelo não tem,
    `quality` no Gemini, `n > 1` onde não existe, ou mais imagens de entrada do que o limite.
  - Não há passagem crua de parâmetros do fornecedor (como no `response_format` cru, D44).
- **Alternativas rejeitadas:**
  - O `size` do OpenAI como vocabulário comum: não diz nada ao Gemini nem ao xAI.
  - `**kwargs` por fornecedor: cada app escreveria quatro dialectos.
- **Consequência:** `background`, `mask`, `moderation` e os outros parâmetros de um só fornecedor
  ficam de fora até alguém os pedir.

## D48 · Nenhum endereço lido do ambiente (frente A, G-16)

- **Contexto:**
  - Sem `base_url`, os adaptadores da OpenAI e da Anthropic não passavam endereço nenhum ao SDK,
    e o SDK lia-o do ambiente: o `openai` lê `OPENAI_BASE_URL` (`openai/_client.py`) e o
    `anthropic` lê `ANTHROPIC_BASE_URL` (`anthropic/_client.py`).
  - O `google-genai` lê `GOOGLE_GEMINI_BASE_URL`. Com `GOOGLE_GENAI_USE_VERTEXAI` muda para o
    Vertex, e então lê `GOOGLE_VERTEX_BASE_URL` (`google/genai/_base_url.py`, `_api_client.py`).
  - O `OpenAIModerator` também deixava o SDK escolher.
  - O registo trata `base_url=None` como o servidor do fornecedor e manda-lhe a chave do ambiente.
    Uma variável esquecida mandava essa chave para onde ela apontasse, contra a regra do
    `AGENTS.md`: "environment keys are never sent there". No OpenAI aplicava ainda as regras do
    host oficial a outro host.
  - O ai-network contorna-o passando sempre o endereço oficial (G-16, no briefing dele de 29/09).
- **Decisão:**
  - O toolkit não lê endereço nenhum do ambiente. Os endereços oficiais vivem numa casa só
    (`OWN_BASE_URLS` em `core/_providers/__init__.py`, de onde sai o `_OWN_HOSTS`).
  - Sem `base_url`, cada adaptador passa o seu ao SDK explicitamente: OpenAI, Anthropic, Gemini,
    Meta e o `OpenAIModerator`.
  - O Gemini fica na Gemini Developer API (`vertexai=False`), que é a API das regras do adaptador.
  - O xAI não muda: o `xai-sdk` usa um host fixo e não lê variáveis.
  - Pela mesma razão, o adaptador para servidores compatíveis tira os cabeçalhos da conta OpenAI
    (`OPENAI_ORG_ID`, `OPENAI_PROJECT_ID`) que o SDK lê do ambiente, como a Meta já fazia.
- **Alternativas rejeitadas:**
  - Seguir a variável do ambiente como se fosse um `base_url` dado, com a guarda das chaves.
    Assim, quem a usa para um gateway passaria a precisar de `api_key=` na mesma, e o toolkit
    ficaria com duas maneiras de dizer o endereço.
  - Levantar um erro quando a variável existe: castigaria quem a tem por outra razão, por exemplo
    para outra ferramenta na mesma máquina.
- **Consequência:**
  - Quem usava `OPENAI_BASE_URL` ou `ANTHROPIC_BASE_URL` para um gateway passa a dar `base_url=`
    (com `api_key=`, se não for loopback). Fica no `CHANGELOG`, com a migração.

## D49 · Toda a falha tem tecto, com ou sem budget (frente A, G-29)

- **Contexto:**
  - Uma chamada que falha depois de enviada (um 5xx, um corte a meio) pode ter sido cobrada. Com
    um `BudgetController`, ela fica incerta com tecto (D20): a reserva estrita, ou a estimativa do
    controller (`FailureBoundController`). Verificado a 2026-10-04 no `6d1a989`: até $0.0615, e a
    nova tentativa num modelo com preço é admitida.
  - Sem controller, só a medir, a falha fica desconhecida, sem tecto. Um `Policy(max_cost=...)`
    por passo falha então mesmo quando a nova tentativa deu certo (achado de 2026-09-18), e o
    ledger do ai-network mostra a chamada "sem preço" (a G-29 dele).
  - O tecto é um facto do pedido (o preço do modelo, a entrada, o `max_tokens`), não uma opinião
    do controller. Com a D16, todo o modelo sob um meter tem preço.
  - A proposta estava no BOARD desde 2026-09-18, e o dono escolheu-a a 2026-10-04.
- **Decisão:**
  - O pior caso de uma operação passa para o core (`core/_metering/_worst_case.py`). É a casa
    única do tecto de uma falha e da estimativa por omissão da reserva estrita.
  - O meter calcula sempre o tecto de uma falha: a reserva estrita quando existe, senão o pior
    caso dos factos do pedido, ao preço do pricer da execução.
  - Sai o `Protocol` `FailureBoundController`. O `HeuristicEstimator` do toolkit passa a delegar no
    core.
  - Fica desconhecido sem tecto só o que não tem preço: uma server tool, ou um pricer que falha.
- **Alternativas rejeitadas:**
  - Deixar como está: um tecto por passo sem budget continua a falhar depois de um retry com
    êxito, e a G-29 do ai-network fica meia fechada.
  - O tecto só nos runs com budget, mas com um controller "medidor" por omissão: seria uma
    segunda maneira de dizer o mesmo.
- **Consequência:**
  - Muda o contrato da R01 ("sem controller não há tecto").
  - Num run só a medir, uma falha passa de `unknown_cost_count` a `uncertain_cost` com tecto. Os
    testes que afirmavam o contrato antigo corrigem-se e listam-se na ficha.
  - As constantes do pior caso (quatro caracteres por token, a folga por imagem e por documento)
    passam do toolkit para o core: são limites, não estimativas.

## D50 · Um preço pode ter data de fim (frente A, G-20)

- **Contexto:**
  - A tabela tem preços promocionais com data de fim: o `gpt-5.6-sol` até 2026-11-21, e o
    `gemini-3.8-flash`, o `-3.7-flash` e o `-3.6-flash` até 2026-12-31. A data está só num
    comentário.
  - Depois dela, o meter cobraria o preço antigo até alguém mudar a tabela e publicar uma versão.
    Uma app que não actualize o toolkit, como o ai-network, conta a menos sem saber.
- **Decisão** (o dono escolheu-a a 2026-10-04):
  - Um `ModelPricing` pode dizer até quando vale (`until`, o último dia, inclusive) e o preço que
    vale a seguir (`then`, outro `ModelPricing`).
  - O registo escolhe, em cada consulta, o preço do dia (UTC). O `get` aceita um dia para
    consultar outro.
  - No TOML, `until` é uma data e o preço seguinte é uma subtabela `then`.
- **Alternativas rejeitadas:** só a tabela, com o comentário e uma versão nova no dia. Depende de
  quem a usa actualizar a tempo.
- **Consequência:**
  - O custo de uma chamada passa a depender do dia em que se calcula: os testes fixam o dia.
  - O catálogo de uma app pode ler `until` e `then`, para mostrar a data de fim e o preço a seguir.

## D51 · As tools verificam o TLS com as autoridades do sistema, por um extra opcional (frente A, G-30)

- **Contexto:**
  - O Python standalone do uv no macOS, o que o ai-network usa, lê as autoridades de
    `/etc/ssl/cert.pem`: 128 raízes, sem a `GlobalSign Root R46`. O Eurostat (`ec.europa.eu`)
    falha nesse Python com `CERTIFICATE_VERIFY_FAILED`.
  - A mesma raiz está no Keychain do sistema e no ficheiro do Homebrew (192 raízes), onde o
    pedido passa. Medido a 2026-10-04.
  - Não é só o Eurostat: qualquer site com uma raiz recente falha nesse Python.
- **Decisão** (o dono escolheu-a a 2026-10-04):
  - Com o pacote `truststore` instalado (extra `truststore`), o `_http.py` verifica com o
    armazém de certificados do sistema;
  - sem ele, fica o contexto da biblioteca padrão, e o erro de certificado diz como resolver;
  - o `truststore` tem licença MIT, não tem dependências, e é o que o pip usa por omissão desde a
    24.2.
- **Alternativas rejeitadas:**
  - uma dependência obrigatória, que tira ao toolkit as zero dependências obrigatórias;
  - deixar à app o `truststore.inject_into_ssl()`, que muda o `ssl` de todo o processo.
- **Consequência:** as tools continuam só com a biblioteca padrão por omissão; a app que corre
  num Python destes instala o extra.

## D52 · Uma tool lê do ambiente a chave opcional do seu serviço (frente A, G-30)

- **Contexto:** sem chave, o Semantic Scholar partilha um limite por todos os anónimos, esgotado
  a 2026-10-04 (429 logo ao primeiro pedido). Uma chave gratuita dá 1 pedido por segundo a quem a
  tem (https://www.semanticscholar.org/product/api/tutorial).
- **Decisão** (o dono escolheu-a a 2026-10-04):
  - um `Api` pode declarar a variável de ambiente da chave (`key_env`), o cabeçalho que a leva
    (`key_header`) e onde se pede (`key_url`). A chave lê-se em cada pedido;
  - sem chave, um 429 diz que variável definir e onde a pedir;
  - a primeira é a `SEMANTIC_SCHOLAR_API_KEY`, no cabeçalho `x-api-key`.
- **Alternativas rejeitadas:** só melhorar a mensagem do 429, que deixa a tool sem resposta
  enquanto o limite partilhado estiver esgotado.
- **Consequência:** é a primeira tool que lê uma chave do ambiente. A chave nunca entra no texto
  que a tool devolve.

## D53 · Um 429 fecha o host durante o tempo que a API pede, e o ritmo conta do fim do pedido (frente A, G-30)

- **Contexto:**
  - O GDELT respondeu 429 a todos os pedidos de 2026-10-04, com qualquer User-Agent, mesmo
    depois de 150 s de pausa.
  - Relatos de 2026-07 e 2026-09 medem uma porta que fica fechada um minuto ou mais depois de
    um 429, sem `Retry-After` (https://github.com/cyanheads/gdelt-mcp-server/issues/44).
  - O toolkit espaçava os pedidos pelo início de cada um e voltava a bater logo a seguir a um
    429, o que prolonga o castigo.
- **Decisão** (o dono escolheu-a a 2026-10-04):
  - depois de um 429, o `_http.py` não manda mais pedidos a esse host durante o `Retry-After`,
    ou o `cooldown_s` que o `Api` declara: responde logo que o serviço pediu para abrandar e
    quando voltar a tentar;
  - o `min_interval_s` passa a contar do fim de cada pedido;
  - o GDELT declara 60 s.
- **Alternativas rejeitadas:**
  - tirar as tools do GDELT do catálogo;
  - deixar como estava e só o escrever nos docs.
- **Consequência:** o GDELT anónimo continua a aguentar poucos pedidos por minuto e por IP; a tool
  deixa de o agravar e diz quanto esperar.

## D54 · As chamadas ao LLM de um flow iterado correm em stream e chegam como eventos do flow (frente A, G-22)

- **Contexto:**
  - As dez estratégias chamam `llm.complete` em 18 sítios, e o texto do modelo só chega no
    `step_end` do passo.
  - Para mostrar os tokens, o ai-network copia o `react_flow` e corre ele próprio o ciclo:
    o texto da última volta, as omissões e as chamadas em paralelo.
- **Decisão** (o dono escolheu-a a 2026-10-04):
  - o core ganha um canal (`llm_events_to(sink)`). Enquanto está ligado, o `LLM.complete`
    corre pelo caminho de stream e entrega cada `StreamEvent` ao `sink`, com o id da chamada. A
    `Response` é a mesma;
  - cada passo de um flow que se itera (`iter()`, sempre: o dono preferiu-o a uma flag) liga o
    canal. Cada evento chega como `FlowEvent(type="llm_event")`, com o passo, o evento e a
    chamada;
  - o `run()` não liga o canal;
  - um flow aninhado herda o canal do passo de fora, e as chamadas de dentro das tools também
    passam por ele.
- **Alternativas rejeitadas:**
  - um helper em cada estratégia: obriga a mudar as 18 chamadas, e um flow da app teria de o
    usar também;
  - os tokens só com uma flag no `iter()`.
- **Consequência:**
  - quem consome um `iter()` passa a receber `llm_event`s;
  - dentro de um flow iterado, uma chamada que falhe depois do primeiro evento já não se repete
    (a semântica do stream); antes dele, o retry e o fallback são os de sempre;
  - o `run()` fica como estava.

## D55 · Duas tools de pesquisa na web, Brave e Tavily, com chave do ambiente (frente A, G-13)

- **Contexto:**
  - "Procura na web" é o pedido mais comum a um assistente, e o toolkit só tem a server tool
    `web_search()`, que o fornecedor do modelo corre e cobra;
  - essa server tool não serve modelos locais nem compatíveis, ignora o `ServerTool.config`, e
    o meter não lhe conhece o custo;
  - o ai-network decidiu (D-51 dele) uma tool do toolkit, com a chave de uma API de pesquisa.
- **Decisão** (o dono escolheu-a a 2026-10-05: "as duas"):
  - `brave_search(query)` usa a Brave Search API ($5 por 1000 pedidos, $5 de crédito grátis por
    mês) e lê a chave de `BRAVE_SEARCH_API_KEY`;
  - `tavily_search(query)` usa a Tavily (pesquisa básica, um crédito de $0,008, 1000 créditos
    grátis por mês) e lê a chave de `TAVILY_API_KEY`;
  - cada uma lê a sua chave do ambiente (D52), e sem ela diz onde a pedir, sem enviar pedido;
  - os nomes são os dos serviços: `web_search` já é a server tool do core.
- **Alternativas rejeitadas:** só uma das duas.
- **Consequência:** a app oferece a que tiver chave. Servem qualquer modelo, também os locais.

## D56 · O preço de uma tool paga vive na tabela de preços, e o meter cobra o que o serviço cobrou (frente A, G-13)

- **Contexto:** o meter contava toda a tool como gratuita, a não ser com um pricer próprio. Uma
  tool de pesquisa custa por pedido.
- **Decisão** (o dono escolheu-a a 2026-10-05):
  - a tabela de preços ganha uma secção `[tools]`. Cada entrada, pelo nome da tool, dá o preço
    de uma unidade (`per_unit`: um pedido na Brave, um crédito na Tavily), com `until` e `then`
    (D50). `pricing.register_tool(...)` muda-o para o plano de cada um;
  - o meter reserva uma unidade antes da chamada (o tecto da D49);
  - a porta HTTP das tools regista cada pedido que o serviço aceitou, com as unidades que ele
    diz ter gasto (`Api(billed_as=..., bill_units=...)`). O executor cobra no fim a soma
    registada;
  - um pedido recusado (sem chave, 401, 429, um erro de rede) não custa nada;
  - uma tool fora da tabela continua gratuita, ou com o preço do pricer próprio.
- **Alternativas rejeitadas:** um preço declarado na tool, que só se muda no código; e um preço
  por chamada, que cobraria uma chamada sem chave ou recusada.
- **Consequência:** o custo de uma tool paga entra no meter, nos budgets e no relatório, como o
  das chamadas ao LLM.

## D57 · Um tecto partilhado por várias execuções: o `SharedMeter`, sob um lock seu (frente A, G-28)

- **Contexto:**
  - cada `MeterScope` tem o seu `MeterStore`, e um `BudgetPolicy` só conta dentro dele;
  - duas execuções a correr ao mesmo tempo não gastam do mesmo tecto. O ai-network dá a cada
    uma o que restava quando começou, e juntas podem passar dele;
  - o ai-network decidiu (D-55 dele) um orçamento partilhado no toolkit, com a admissão e o
    acerto sob um lock, que a app semeia com o que o ledger já gastou.
- **Decisão** (o resultado vem da D-55 do ai-network; o desenho é do coordenador, a 2026-10-05):
  - o core ganha o `SharedMeter(limits, spent=...)`, com um lock e contadores seus, semeados
    com o gasto que a app já tem;
  - liga-se a cada execução pelo `RunConfig(shared=...)`, que todas as entradas já aceitam;
  - cada operação de uma execução ligada é admitida também contra ele, sob o seu lock, depois do
    da execução (a ordem é sempre execução → partilhado, sem risco de deadlock). A reserva e o
    acerto ficam nos dois contadores;
  - para que juntas nunca passem do tecto de custo, cada operação reserva nele o seu pior caso
    (D49), mesmo numa execução sem budget ou com um budget sem reserva estrita. Uma operação
    sem preço é recusada sob um tecto de custo;
  - o `SharedBudget(policy, spent=...)` do toolkit constrói-o a partir de um `BudgetPolicy`:
    `max_cost`, `max_llm_calls` e `max_tool_calls` (o tempo e os tokens não se partilham);
  - um custo desconhecido numa execução fecha o tecto partilhado, com o `unpriced="fail_closed"`.
- **Alternativas rejeitadas:**
  - dar a cada execução o que resta quando começa (o contorno de hoje);
  - um tecto partilhado sem reservas, que deixa passar o que estiver em curso.
- **Consequência:**
  - perto do tecto, uma chamada pode ser recusada por o seu pior caso não caber, mesmo que o
    custo real coubesse;
  - a app lê `shared.snapshot()` (o gasto com a semente) e guarda no ledger o que cada execução
    gastou.

## D58 · O gasto de cada passo, os spans públicos e as dependências fracas de um DAG (frente A, G-32, G-33)

- **Contexto:**
  - o `execute_step` só abria um span do meter com `Policy.max_cost`, e o `StepTrace` não dizia o
    que o meter mediu no passo. A app media cada passo, cada corrida aninhada e cada delegação
    com o `open_span` e o `current_meter` internos;
  - um passo de um DAG corria só se todas as dependências tivessem corrido bem. Os caminhos que um
    `when` separava nunca se juntavam, e um passo opcional não se exprimia;
  - a razão de um salto vinha só em texto, e a app refazia a regra para dizer que dependência o
    saltou.
- **Decisão** (do coordenador, a 2026-10-05, pela forma que o briefing do ai-network pede):
  - sob um meter, cada passo corre num span seu. O `StepTrace.metered` (um `MeterSnapshot`) é o
    que lá se mediu, também para um passo que a corrida cortou, até ao corte;
  - `open_span`, `current_meter`, `current_span_id` e `bind_meter` passam a ser públicos, em
    `ai_arch_toolkit.core`;
  - o id de um span é o caminho desde a raiz (`run/3/7`). Uma operação que começa depois de o seu
    span fechar (a thread de uma tool que um passo deixou a correr) conta no antepassado aberto
    mais próximo, em vez de falhar. Um id que não pode ser do store é recusado; um id bem formado
    de outro store não se distingue, como já acontecia com os `span-N`;
  - uma corrida aninhada tem também um span seu, e a sua entrada nos `children` do passo traz o
    gasto dela;
  - o `FlowStep` ganha `after_any` (corre se pelo menos uma correu bem) e `after_optional` (espera
    por ela e nunca a exige). O passo espera sempre que todas acabem, e cada dependência
    nomeia-se uma só vez;
  - o `StepTrace.blocked_by` diz que dependências saltaram um passo e como acabaram (`"failed"`
    ou `"skipped"`). O texto continua no `skip_reason`.
- **Alternativas rejeitadas:**
  - só exportar os spans, sem o gasto no trace: cada app voltava a abrir um span à volta de cada
    passo;
  - um `after_any` que corre logo que a primeira acaba bem: o passo leria o estado antes de os
    outros ramos se fundirem, e as ondas do DAG não o permitem;
  - um passo opcional como propriedade do passo (`optional=True`): a dependência opcional é mais
    geral (um dependente pode exigi-lo, outro não);
  - guardar os spans fechados para as operações tardias: a memória crescia com as iterações, o
    que o `close_span` existe para impedir.
- **Consequência:**
  - todo o passo de um flow medido abre um span, e cada corrida aninhada também. Por passo, o
    lock do store toma-se três vezes, e fechar o span percorre os spans e as operações vivas.
    Medido: 4000 passos em leque, todos abertos ao mesmo tempo, em 0,4 s;
  - um span que não fecha por ter uma operação viva (um stream por drenar, a thread de uma tool
    que passou do prazo) fica até ao fim da corrida. Antes era só nos passos com `max_cost`; o
    que fica é proporcional às operações vivas, que o store já guarda;
  - um `FlowStep` com uma dependência repetida passa a levantar `ValueError`;
  - as razões dos saltos nomeiam todas as dependências que não correram bem, e o "all
    dependencies skipped" desaparece.

## D59 · Os manifestos e recursos que o toolkit carrega: os aliases do YAML, o aninhamento e as heranças com limite (frente A, G-34)

- **Contexto:**
  - o `yaml.safe_load` partilha o nó de um alias, mas os percursos do toolkit (o
    `_canonical_config` e a impressão digital dos manifestos de agente) andam pela árvore já
    expandida. 412 bytes com seis níveis de dez aliases levam 2,6 s e 170 MB, e cada nível
    multiplica por dez;
  - o `YamlCodec` dos recursos, por onde passam os manifestos de prompt e o conhecimento, faz o
    mesmo `safe_load` sem limite;
  - um ficheiro de 4 KB com 2000 níveis de aninhamento levanta um `RecursionError` cru, em YAML,
    JSON e TOML;
  - o `extends` dos manifestos de agente e o `extends`/`include` dos de prompt relêem um pai
    por cada caminho até ele. Dez manifestos de agente com 757 bytes, cada um a herdar quatro
    vezes do seguinte, dão 349 525 leituras;
  - a app recusa, antes do toolkit, âncoras, aliases e mais de 32 níveis (E02-05 do ai-network).
- **Decisão** (do coordenador, a 2026-10-05, pela forma que o briefing pede: "recusa aliases, ou
  limita o que eles expandem"):
  - um só módulo, `toolkit/_safe_data.py`, lê o YAML, o JSON e o TOML dos manifestos de agente e
    dos codecs dos recursos. Um teste de arquitectura só deixa importar o PyYAML nos módulos que
    lista, e os seus loaders só no `_safe_data`;
  - os aliases limitam-se, não se proíbem, como faz também o go-yaml (com outra regra, uma razão
    entre os aliases e o resto). Antes de construir os dados, um percurso iterativo do grafo de
    nós mede a árvore expandida. Os aliases podem acrescentar até 10 000 nós e 1 000 000 de
    caracteres, ou tantos quantos o documento tem, se for mais. Contar os caracteres apanha um
    alias que copia muitas vezes um texto longo (139 KB davam 3,3 s e 2 GB só a contar nós);
  - as chaves de um mapa fundido (`<<`) contam ao lado das do mapa, não como um nível. Uma cadeia
    de merge keys tem no máximo 100 elos, porque o PyYAML desce uma vez por elo;
  - um alias dentro da sua própria âncora é recusado;
  - nenhum documento passa de 100 níveis de mapas e listas, em nenhum formato. O YAML conta os
    níveis enquanto se compõe e pára no 101.º. No JSON e no TOML, o parser que fica sem pilha é a
    mesma recusa, e o resto vê-se num percurso nível a nível;
  - um override de um manifesto de agente também não passa de 100 níveis, contando os do seu
    caminho com pontos;
  - um manifesto que vários herdam ou incluem é lido e construído uma só vez por carregamento. O
    limite de herança conta-se na mesma por todos os caminhos;
  - a recusa é um `UnsafeDataError`, que cada sítio converte no seu erro (`AgentManifestError`,
    `AgentOverrideError`, `ResourceDecodeError`, e daí `PromptLoadError`).
- **Alternativas rejeitadas:**
  - proibir as âncoras e os aliases: são YAML normal, e as merge keys servem para não repetir
    configuração. A app pode continuar a proibi-los do seu lado;
  - um tecto absoluto de nós: recusaria um ficheiro de dados grande sem aliases nenhuns;
  - confiar no `RecursionError`: chega tarde (0,8 s para 2000 níveis em YAML) e depende do limite
    de recursão do processo;
  - passar pelo mesmo módulo os ficheiros que são dados da própria app (um grafo guardado, uma
    tabela de preços): não são manifestos importados, e ficam como estão.
- **Consequência:**
  - um manifesto ou recurso com mais de 100 níveis passa a ser recusado, também num recurso JSON
    que antes carregava;
  - os percursos recursivos do toolkit sobre manifestos, recursos e overrides não passam de 100
    níveis;
  - um manifesto ou recurso custa em proporção ao seu tamanho. Ler um YAML grande fica 14 a 17%
    mais lento, e um JSON de 15 MB passa de 0,18 s para 0,30 s.

## D60 · Os argumentos de uma chamada a uma ferramenta, a chegar em stream (frente A, G-37)

- **Contexto:**
  - um `StreamEvent` era `text`, `thinking`, `tool_call` ou `image`. Os adaptadores juntavam os
    pedaços de uma chamada e só a entregavam completa, depois do fim do stream;
  - a Anthropic manda o id e o nome de um bloco `tool_use` no início, e o input em pedaços
    `input_json_delta`. A Responses API manda um item `function_call` e os deltas dos argumentos
    pelo mesmo `output_index`. A Chat Completions manda deltas por `index`, com o id e o nome no
    primeiro;
  - a Gemini API manda cada chamada inteira numa parte: o `stream_function_call_arguments` é só
    da Vertex AI. O xAI manda cada chamada "in whole in a single chunk";
  - a app só mostra um artefacto quando ele está guardado.
- **Decisão** (do coordenador, a 2026-10-05, pela forma que o briefing pede):
  - um tipo novo de evento, `tool_call_delta`, sempre `partial`, com um `ToolCallDelta`:
    - `index`, o lugar da chamada entre as chamadas da resposta, o mesmo que em
      `Response.tool_calls` e na ordem dos eventos `tool_call` finais;
    - o `id` e o `name` do fornecedor, em cada pedaço. O `id` fica vazio quando o fornecedor não
      dá nenhum, e a chamada final leva então um id seu;
    - `input_json`, o pedaço seguinte do input em texto JSON. Os pedaços de uma chamada, juntos
      pela ordem, são o input inteiro, e um texto vazio é um input vazio. Na Responses API, o
      `function_call_arguments.done` dá o que os deltas não escreveram, para que a regra valha
      num servidor que não manda deltas (a Meta partilha o núcleo, e nada mostra que os mande);
  - o primeiro pedaço sai logo que o nome se sabe, mesmo vazio;
  - na Gemini e no xAI, cada chamada é um pedaço só, com o input inteiro, enviado logo que o
    chunk chega;
  - os eventos `tool_call` finais não mudam: vêm depois do stream, com o input já lido;
  - um só ajudante, `CallPieces` em `core/_providers/_base.py`, numera as chamadas e constrói os
    eventos. Cada adaptador só começa as chamadas que o seu `assemble` faz chamadas, pela mesma
    ordem.
- **Alternativas rejeitadas:**
  - `kind="tool_call"` com `partial=True`: um consumidor que lê `event.tool_call` em cada
    `tool_call` recebia `None`. Um tipo novo não muda o que já se lia;
  - casar os pedaços com a chamada final pelo id: o id que o toolkit dá a uma chamada sem id é
    aleatório, e uma chamada pode chegar sem id. O lugar na resposta é estável;
  - entregar o input parcial já lido como dicionário: o JSON a meio não se lê, e cada fornecedor
    corta-o noutro sítio. Quem quiser mostrar o parcial junta o texto;
  - deixar os pedaços fora da regra do D54 (não repetir depois do primeiro evento), para que uma
    resposta só com chamadas continue a repetir-se num flow iterado: a nova tentativa mandava os
    seus pedaços com o lugar a recomeçar em 0, misturados com os da que falhou, e o consumidor
    não as distinguia.
- **Consequência:**
  - quem compara todos os `StreamEvent.kind` encontra um novo, e quem conta eventos vê mais;
  - os pedaços contam como entregues, como o texto (D54): no `stream_events()` e no `complete`
    de um flow iterado, uma resposta só com chamadas que falhe depois do primeiro pedaço já não
    se repete nem passa ao fallback. Antes, nada se mostrava até ao fim do stream, e repetia-se.
    O `complete` sem canal e o `stream()` (que só mostra texto) repetem-se como antes;
  - um stream abandonado a meio de uma chamada não deixa uma chamada a meio na `Response`;
  - os eventos chegam também aos `llm_event` de um flow iterado (D54).

## D61 · A reserva de uma imagem pelo modelo, pela qualidade e pelo tamanho (frente A, G-40)

- **Contexto:**
  - o pior caso (`core/_metering/_worst_case.py`, D49) reservava 16 000 tokens de imagem por
    imagem pedida, olhando só para o número de imagens. Uma imagem em baixa no Flare, a 1024x1024,
    custa 196 tokens: a app contorna com a sua própria estimativa por qualidade (D-81 dela);
  - os 16 000 também não eram um tecto: o gpt-image-2 em `high`, e os 2.5 em `max`, chegam a
    23 719 tokens a 2880x2880;
  - a OpenAI publica a tabela dos modelos antes do gpt-image-2 (por qualidade e pelos três
    tamanhos) e a calculadora do gpt-image-2 e dos 2.5 (uma grelha de mosaicos por qualidade, com
    o custo de cada mosaico a crescer com os píxeis). O gpt-image-1-mini só tem preços por
    imagem. O Gemini publica os tokens por tamanho. O xAI e a Meta cobram por imagem;
  - a fórmula confere com o que a app mediu ao vivo a 4 Out: 1 167 no gpt-image-2 em média a 2:3
    de 1K, 292 no Flare na mesma, 5 488 no gpt-image-2 em alta a 1024x1536, e 196 no Flare em
    baixa a 1024x1024 (a corrida da frente I).
- **Decisão** (do coordenador, a 2026-10-05, pela forma que o briefing pede):
  - as contagens são regras por modelo, por isso vivem nas tabelas de cada adaptador
    (`ImageModel.tokens` no OpenAI, `_ImageProfile.sizes` no Gemini). O
    `BaseProvider.image_token_bound(image)` dá o tecto de uma imagem do pedido, ou `None`;
  - o `request_facts` leva-o ao meter como facto, `OperationRequest.declared_image_tokens`. O
    `worst_case` reserva-o por imagem, e o metering continua neutro: não conhece modelos;
  - o tamanho é o que o adaptador manda (o mesmo `image_size`). Uma qualidade ou um tamanho que
    ficam ao modelo reservam o mais caro que ele pode escolher: o `auto` não tem contagem
    publicada;
  - o gpt-image-1-mini usa os seus preços por imagem a $8 por milhão: o maior número de tokens que
    cada preço de três casas decimais admite;
  - um modelo cobrado por imagem (Grok Imagine, Muse Image) dá 0 tokens de imagem;
  - os modelos de imagem do Gemini pensam sempre ("thinking cannot be disabled in the API"), ao
    preço do texto: o `image_text_token_bound` dá o limite de saída de cada um (32 768, e 4 096
    no Flash Lite), que o `request_facts` leva como `declared_max_output_tokens` de uma geração
    que não declara nenhum. A revisão mostrou que, sem ele, 1 500 tokens de pensamento passavam
    a reserva exacta da imagem;
  - um modelo sem contagens publicadas reserva 24 000 tokens, acima da imagem publicada mais cara.
- **Alternativas rejeitadas:**
  - uma tabela no core, ao lado do metering: as regras por modelo vivem nos adaptadores (o
    `AGENTS.md`), e o tamanho que se manda é do adaptador;
  - passar o `ImageRequest` ao `OperationRequest` e calcular no `worst_case`: o metering teria de
    conhecer os modelos de imagem;
  - medir ao vivo: o dono não quer chamadas pagas, e as contagens publicadas conferem com as
    medidas da app.
- **Consequência:**
  - um orçamento estrito reserva perto do que a imagem custa: duas imagens em baixa no Flare
    reservam 392 tokens, onde reservavam 32 000;
  - um modelo de imagem sem contagens reserva mais do que antes (24 000 em vez de 16 000), e uma
    chamada ao gpt-image-2 ou aos 2.5 sem qualidade nem tamanho também (23 719);
  - o Gemini não promete o número de imagens pedido ("won't always follow the exact number"):
    imagens a mais do que as pedidas ficam fora da reserva, como antes;
  - a app pode tirar a estimativa da D-81 quando passa a qualidade.

## D62 · As tools dinâmicas: `tool_from_schema` (frente C, C02)

- **Contexto:** só o `@tool` cria definições; o cliente MCP (C03) precisa de tools feitas de um
  JSON Schema que chega em execução, com nomes que podem não ser identificadores Python. O
  `ToolGroup.add`, o `prepare_tools`, o `execute_tool` e o `run_tools` já aceitam qualquer callable
  que traga o seu `__tool_definition__`.
- **Decisão** (do dono, a 2026-10-08, na página das decisões da vaga 1; as recomendações do
  coordenador, revistas contra o código de hoje):
  - **C02.1:** uma factory pública, `tool_from_schema(handler, name=, description=, input_schema=,
    policy=)`, devolve um callable com `__tool_definition__` (e sem `__wrapped__`), que entra nos
    quatro caminhos sem lhes mexer;
  - **C02.2:** o handler recebe os argumentos num `dict`: a factory faz o callable `(**arguments)`,
    que chama `handler(dict(arguments))`; o handler levanta `ToolFailure` (D42) para dar o tipo;
  - **C02.3:** a validação da D7 estende-se, só com a biblioteca padrão. `oneOf` e `type` em lista
    coagem como `anyOf`. `additionalProperties: false` na raiz recusa chaves desconhecidas, mesmo
    com `**kwargs`. Um parâmetro que chega pelo `**kwargs` só aceita `null` se o schema o admitir.
    Os aninhados não mudam, e o `jsonschema` fica nos testes (D36);
  - **C02.4:** a regra de nomes portátil (`^[A-Za-z_][A-Za-z0-9_-]{0,63}$`, a intersecção das dos
    fornecedores e do MCP) vive na `ToolSchema`. O `prepare_tools` passa a usá-la também para
    callables nus e dicts, e uma `lambda` passa a `ValueError`;
  - **C02.7:** as chaves extra de um schema MCP vão como vêm, só com os `$ref` locais embutidos.
    Uma recusa provada ao vivo (C02f) vira regra no adaptador desse fornecedor, com a fonte;
  - **C02.8:** nomes repetidos numa lista ou em vários grupos levantam `ValueError` antes de enviar
    ou de correr; a mesma tool repetida não conta. Fecha o achado de 2026-10-04;
  - **já decididas:** a C02.5 (nomes repetidos num grupo) pela A03 (G-21), e a C02.6 (o grupo
    vazio) pela F24.
- **Alternativas rejeitadas:**
  - `ToolGroup.add_definition()` ou `@tool(input_schema=)`;
  - `**kwargs` no handler;
  - a D7 como está, ou o `jsonschema` como extra: dois comportamentos para a mesma tool (R00);
  - a regra de nomes só na factory, ou por adaptador;
  - limpar as chaves extra à partida;
  - deixar as listas sem verificação.

## D63 · O catálogo técnico de modelos no core (frente C, C06)

- **Contexto:** nada no `src/` guarda os limites e as capacidades de cada modelo com fonte e data.
  Desde a R02 as regras por modelo vivem nas tabelas de perfis de cada adaptador, resolvidas pelo
  `_model_id.lookup`.
- **Decisão** (do dono, a 2026-10-08, na página das decisões da vaga 1):
  - **C06.1** (confirmada depois da resposta abaixo): um facto diz o que funciona através do adaptador, com
    `kind="adapter"` quando o limite é do adaptador e não do modelo;
  - **C06.3:** cada facto é `T | None` (`None` é desconhecido, nunca "não"), com `sources` por
    campo, e os três limites como os publicam (janela, entrada, saída);
  - **C06.4:** o catálogo tira os factos do adaptador das próprias tabelas de cada um, por um
    método de classe puro (como o `image_token_bound` da D61). O TOML guarda só o que o adaptador
    não sabe (limites e modalidades publicados, com fonte e data). Sem o extra do SDK, esses campos
    ficam `None`. Os adaptadores nunca lêem o catálogo;
  - **C06.5:** `LLM.describe_model()` (e o `_sync`) para a Anthropic, o Gemini e o xAI; a OpenAI e a
    Meta levantam `NotImplementedError`. Não escreve no catálogo. É a C06e, fora da vaga 1;
  - **C06.6:** a semente é a linha actual com página oficial: os 30 modelos do inventário que ainda
    servem, e os de imagem;
  - **C06.7:** um campo `output_modalities` (`text`, `image`), e os modelos de imagem na semente,
    com `tools=False`;
  - **já decidida:** a C06.2 (como um id casa com a sua entrada) pela R02 (`_model_id.lookup`, D16).
- **A dúvida do dono na C06.1** ("isto não limita quem usa a framework, se ela não for
  actualizada?") e a resposta: o catálogo não limita nada, porque nenhum adaptador o lê. Quem
  limita é o adaptador, com catálogo ou sem ele, e o catálogo só o diz, e de quem é o limite. Um
  modelo que a framework não conhece dá `None`; um facto errado corrige-se na app com `register()`
  (`kind="override"`); e, pela C06.4, corrigir o adaptador corrige o catálogo.
- **Alternativas rejeitadas:**
  - o modelo publicado (a app mandaria o que o adaptador perde);
  - `bool` com uma fonte por entrada;
  - os adaptadores a ler o catálogo;
  - o TOML a repetir os factos do adaptador, ligado por um teste;
  - a descoberta em execução na app;
  - tudo o que tem preço na semente.

## D64 · As tools de escrita tipadas e a `FilesystemPolicy` (frente C, C07)

- **Contexto:** as tools de ficheiros só lêem e não têm raízes; a C07 traz a escrita, sob uma
  policy com raízes por acção.
- **Decisão** (do dono, a 2026-10-08, na página das decisões da vaga 1):
  - **C07.1:** a policy verifica-se no gate e na tool, com o mesmo `check`: o gate poupa o humano e
    o budget, e a tool, logo antes do syscall, é a garantia;
  - **C07.2:** o `PathScopeGate` recebe no construtor o mapa `paths={tool: {argumento: acção}}`,
    com o das sete tools por omissão; uma tool `capability="filesystem"` sem mapa fica bloqueada;
  - **C07.3:** uma factory, `filesystem_tools(policy)`. As leituras actuais não mudam, as ligadas
    recusam `..` no `pattern` e saltam alvos fora das raízes, e sem policy não há escrita. O
    `tests/toolkit/tool_catalog.py` constrói as tools da factory com uma policy de teste, para a
    invariante e o contrato as verem. A C07e vem depois da parte de ficheiros da T09; a C07a, b e d
    podem começar já;
  - **C07.4:** o Python mínimo sobe para o 3.13.4 (`requires-python = ">=3.13.4"`, com `uv lock`), e
    a resolução de um caminho que ainda não existe usa só
    `os.path.realpath(strict=os.path.ALLOW_MISSING)`. A garantia continua a ser a caminhada da raiz
    com `O_DIRECTORY|O_NOFOLLOW` e a operação relativa ao `dir_fd` do pai;
  - **C07.5:** a escrita é atómica. Um temporário no mesmo directório
    (`O_CREAT|O_EXCL|O_NOFOLLOW`) e `fsync`. Sem `overwrite`, `os.link` e `unlink`; com
    `overwrite`, `os.replace` com o modo antigo e `fsync` do directório. O `append_file` só em
    ficheiro regular com `st_nlink == 1`; o `move_path` sem cópia entre volumes. UTF-8 estrito; um
    `content` que não é texto é `validation_error`;
  - **C07.6:** um hook `@tool(preview=fn)` no core (`ToolDefinition.preview`), lido pelo pedido de
    aprovação e pelo `DryRunGate`, e um outcome novo, `permission_denied`. É a C07c, na vaga 2;
  - **C07.7:** nenhuma tool de remoção na v1;
  - **C07.8:** só POSIX. Em Windows, `filesystem_tools` com `write_roots` levanta
    `NotImplementedError` ao construir; as leituras ficam;
  - **C07.9:** `permission_denied` entra no `ToolFailureType` (adenda à D42): a mesma palavra no gate
    e na tool.
- **Alternativas rejeitadas:**
  - a policy só no gate, ou só na tool;
  - `path_args` na `ToolRuntimePolicy`;
  - a policy num parâmetro ou num contextvar;
  - o `ALLOW_MISSING` onde existe e outra coisa antes (dois caminhos, contra a R00);
  - a escrita directa;
  - um helper de preview para a app;
  - um `delete_path` com quarentena;
  - o Windows sem prova no CI;
  - `validation_error` ou `upstream` para a recusa da policy.

## D65 · A pesquisa web local: o que fica da C08 (frente C, C08)

- **Contexto:** a A07 já fez as tools `brave_search` e `tavily_search` (D55, D56), de outra maneira
  do que a ficha propunha; ficam a dívida de contrato das duas e o fim da C08d.
- **Decisão** (do dono, a 2026-10-08, na página das decisões da vaga 1):
  - **C08.3:** sem aprovação obrigatória, risco `low`, como as outras tools de rede. O custo fica
    sob o budget (D56), e a app que queira aprovar re-decora a tool ou põe um gate seu;
  - **C08.4:** os URLs dos resultados filtram-se a `http(s)`, sem rótulo de conteúdo de terceiros;
  - **C08.7:** a ficha C08 reescreve-se com o que falta. A dívida de contrato das duas tools
    (`Range` no `max_results`, os casos de `errors` e de `zero`, a janela). O contraste com o
    `web_search()` em `docs/tools.md`, o `docs/safety.md`, o `CANDIDATE_TOOLS.md`. O exemplo com
    modelo local e a verificação ao vivo;
  - **já decididas:** a C08.1, a C08.2 e a C08.6 pela D55 (com a D52), e a C08.5 pela D56.
- **Alternativas rejeitadas:**
  - aprovação obrigatória com risco `medium` (quebrava a app, que as usa sem handler);
  - o rótulo "untrusted" só nestas duas tools;
  - passar a C08 para a T09, ou deixar a dívida sem dono.


## D66 · Comandos e regex que se podem parar: grupo de processos e processo filho (frente T, T09)

- **Contexto:** a revisão da T09a achou duas formas de uma tool prender o programa:
  - um processo que o `run_command` deixa em segundo plano (`yes X &`, `tail -f`) sobrevivia à
    chamada, e duas threads ficavam a ler a saída dele sem fim;
  - uma regex que recua em tempo polinomial (`a*a*b` em 20 000 caracteres, cerca de 18 minutos)
    congelava o processo inteiro no `regex_search`, porque o motor do `re` segura o GIL, e nem o
    prazo do executor a parava.
- **Decisão** (do coordenador, a 2026-10-08, na ronda de correcções; o dono pode revê-la):
  - **`run_command`:**
    - corre o comando num grupo de processos próprio, sem entrada (`/dev/null`);
    - uma só thread lê os dois pipes, e o grupo morre quando a chamada acaba, ou meio segundo
      depois de a shell acabar, com uma nota que manda acabar o comando com `wait`;
    - fica só POSIX: em Windows dá uma falha `upstream` tipada, pelo critério da C07.8 (sem
      Windows no CI, sem prova);
  - **`regex_search`:**
    - cada procura corre num Python filho (`-I -S`, um script fixo que só procura), morto aos 5 s;
    - uma regex que recua é recusada quando o tempo acaba, e o programa nunca congela; custa cerca
      de 20 ms por chamada;
    - um programa congelado ou embebido (sem um Python em `sys.executable`) recebe uma falha
      `upstream` que o diz;
    - o teste de invariantes aceita-o por nome: a tool computa e não corre nada de quem chama.
- **Alternativas rejeitadas:**
  - **uma verificação estática da regex**, porque, para apanhar `a*a*b`, recusa padrões comuns
    como `\d+-\d+`, e não se prova completa;
  - **threads para a regex**, que não se podem parar enquanto o `re` segura o GIL;
  - **manter o Windows no `run_command`**, sem grupos de processos nem `select` sobre pipes.
- **Consequência:**
  - quem usava o `run_command` em Windows deixa de o ter: é uma quebra (`CHANGELOG`);
  - o `python_repl` tem o mesmo risco com o `re` e fica nos achados.
