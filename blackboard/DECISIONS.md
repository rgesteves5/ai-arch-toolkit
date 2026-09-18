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
