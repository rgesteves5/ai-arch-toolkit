# Revisão das limitações do AI Arch Toolkit para a app Agentes

**Data da revisão:** 12 de setembro de 2026

**Código revisto:** `main` em `48a43ac`

**Versão instalada na app:** `f210d6f`

## Conclusão executiva

O diagnóstico original é, no essencial, bom: identifica corretamente a falta de MCP, de
agendamento, de uma representação declarativa de workflows/máquinas de estados, de streaming de
tokens através de `Agent`, de checkpoint/retoma, de routing automático e de delegação
agente-como-ferramenta. Porém, há quatro correções importantes:

1. **L13 está formulada de forma demasiado absoluta:** já existe pesquisa web genérica através da
   ferramenta *server-side* `core.web_search()`. O que não existe é uma pesquisa web local,
   independente do fornecedor, do tipo Brave/Tavily/DuckDuckGo.
2. **L9 confunde ausência de ferramentas tipadas com ausência de escrita:** `run_command` pode
   escrever, editar, apagar e abrir aplicações. Faltam operações dedicadas e governáveis, mas o
   toolkit não é “só de leitura”.
3. **As propostas de L3 e L6 não são semanticamente suficientes:** `Flow.max_iterations` conta
   passagens pelo conjunto de passos, não transições de estado, e o executor não aceita um cursor de
   retoma. Guardar artefactos por passo não permite retomar corretamente um DAG ou ciclo.
4. **Algumas capacidades assumidas são mais estreitas do que o documento diz:** `Agent.iter()` só
   emite eventos de passos; `Flow.iter()` perde o paralelismo de DAG; `agent_from_manifest()` exige
   que a aplicação forneça o LLM principal; `ToolError.type` é aberto (`str`); e
   `ResourcePolicy` não governa automaticamente ferramentas de filesystem.

Divisão recomendada:

- **Toolkit:** contrato de eventos/streaming de agente, registo seguro de ferramentas dinâmicas,
  adaptador MCP, agente-como-ferramenta, checkpoint/cursor do executor e, depois de estabilizado o
  modelo, specs genéricas de flow/máquina de estados.
- **Aplicação:** scheduler e catch-up, cofre e configuração de fornecedores, descoberta de modelos
  instalados, routing `Auto`, escolha de arquitetura, persistência de conversas/execuções,
  notificações e integrações Tauri/OS.
- **Dividido:** catálogo técnico de capacidades dos modelos, permissões de filesystem e
  serialização de workflows. O toolkit deve fornecer contratos/mecanismos; a aplicação mantém
  política, inventário, UX e persistência de produto.

O `f210d6f` está apenas um commit atrás de `48a43ac`. A diferença relevante é a adição do GPT-6
Astra (preço, tratamento de parâmetros no adaptador OpenAI e inventário de probes). Nenhuma das
conclusões L1–L13 muda para a versão instalada; nela há 71 entradas de preço em vez de 72.

## Correções às premissas da secção 2

### 1. Cliente LLM — confirmado

`LLM.__init__` aceita `provider`, `api_key`, `base_url` e `fallback`
(`src/ai_arch_toolkit/core/_llm.py:361-410`). `complete()` expõe ferramentas, thinking,
`output_schema`, `tool_choice` e `json_mode` (`_llm.py:671-699`); `stream_events()` expõe os mesmos
controlos (`_llm.py:1150-1181`). `StreamEvent.kind` é exatamente `text | thinking | tool_call`, com
`partial` (`core/_response.py:349-365`), e a resposta final só fica disponível depois de consumir ou
fechar o stream (`_response.py:395-425`).

Nuance: `stream_events()` não é uma coroutine; devolve imediatamente um `RichStreamResponse`, que é
um iterador assíncrono. Um stream abandonado deve ser usado como context manager ou fechado com
`aclose()` para a contabilização ficar terminal de imediato.

### 2. Modelos locais — confirmado

Um modelo desconhecido com `base_url` é encaminhado para o adaptador OpenAI; endpoints loopback
recebem uma chave placeholder e não leem uma chave cloud do ambiente
(`core/_providers/__init__.py:38-42`, `70-109`, `112-171`). O adaptador OpenAI extrai
`reasoning_content` e `reasoning` (`core/_providers/_openai.py:218-231`) e emite fragmentos de
thinking com `partial=True` (`_openai.py:544-545`).

### 3. Agent e estratégias — confirmado com limites

`Agent(ReasoningSpec, llm, tools, deps=...)`, `run`, `iter`, `as_step` e `from_flow` existem
(`toolkit/agents/_agent.py:37-133`). Há nove factories públicas e a décima estratégia,
`completion`, é construída inline no registo (`agents/_builders.py:236-280`). Modelos, ferramentas e
prompts por fase existem para as fases declaradas por cada estratégia; não constituem uma
configuração arbitrária por passo.

Duas lacunas relevantes para a app:

- `Agent.run()` e `Agent.iter()` aceitam apenas `budget_policy`, não `RunConfig`; para instalar
  sinks/redactor/pricer é necessário envolver a chamada num `MeterScope(RunConfig(...))` público ou
  descer para `Flow` (`_agent.py:88-129`; `core/_metering/_scope.py:56-68`).
- `Agent.iter()` não é streaming de resposta: apenas encaminha `FlowEvent`s.

### 4. Manifestos de agentes — confirmado com uma correção importante

Existem YAML, JSON e TOML, não apenas `*.agent.yaml`; há herança, perfis, overrides governados,
fingerprints, restrição de raízes e rejeição de segredos (`agents/_manifest.py:34-117`, `286-378`,
`381-422`). `limits` produz `BudgetPolicy` (`_manifest.py:249-269`).

Contudo, `agent_from_manifest()` **não constrói o agente inteiro a partir do ficheiro**. Exige um
`llm` principal já criado; `llm_factory` só resolve modelos de fases (`agents/_assemble.py:19-69`).
O modelo principal, ferramentas, factories/manifests de ferramentas, rendering de prompts e ids de
schemas continuam a ser resolvidos pela aplicação, como o próprio manifesto documenta
(`_manifest.py:160-177`).

### 5. Flow — parcialmente confirmado

`Flow` suporta sequência, ciclos condicionais e DAGs; valida dependências e ciclos de DAG, exige
`max_iterations` em ciclos condicionais e expõe `max_parallelism`
(`toolkit/flow/_flow.py:115-192`). `Flow.run()` aceita `RunConfig`, que substitui por inteiro a
configuração de budget (`_flow.py:231-248`). Flows aninhados herdam o meter
(`flow/_executor.py:600-627`).

Há dois limites omitidos:

- `Flow.run()` executa os nós prontos de um DAG em paralelo e respeita `max_parallelism`
  (`_executor.py:383-445`), mas `Flow.iter()` usa `_iter_dag`, que percorre os nós prontos
  sequencialmente e não consulta `max_parallelism` (`_executor.py:449-537`). Usar `iter()` muda,
  portanto, a semântica temporal/performance de um DAG.
- `StepTrace.children` existe, mas o executor nunca o preenche; `Flow.as_step()` devolve apenas o
  resultado/artefactos do flow filho (`_flow.py:283-318`). A árvore de subflows/subagentes não fica
  ligada na `Trace`, apesar de o tipo permitir filhos.

`State.to_dict()` e `Trace.to_dict()` produzem representações em dicionário, mas isso não significa
que qualquer estado seja JSON serializável. `Result.value` e `artifacts` aceitam `Any`
(`core/_step.py:18-65`), e os agentes guardam objetos `Response` no estado. Em modo redacted, o
`Redactor` converte dataclasses e representa objetos desconhecidos como texto; em `full_debug`
devolve-os intactos (`core/_redaction.py:37-70`).

### 6. Ferramentas governadas — parcialmente confirmado

A separação `ToolSchema` / `ToolRuntimePolicy` / `ToolDefinition` e o binding em
`fn.__tool_definition__` estão implementados (`core/_tools/_definition.py:22-55`,
`_decorator.py:45-79`). `ToolGroup` resolve o binding, aplica gates, aprovação e limite de chamadas,
e executa sync/async (`_group.py:26-127`). Gates podem alterar argumentos; portanto, a função recebe
`definition.fn(**args)` depois das transformações, não necessariamente o input original
(`_executor.py:193-286`).

Correções:

- `ToolError.type` é `str`, logo a taxonomia de erros não é fechada
  (`core/_tools/_result.py:10-18`). Só `GovernanceOutcome` é um literal fechado.
- `ToolGroup` aceita callables, não `ToolDefinition` diretamente. Prender manualmente uma definição
  ao callable funciona, mas falta uma factory/API pública ergonómica para ferramentas com schema
  dinâmico.
- `ToolGate` e `ExecutionContext` são usados na assinatura pública de `ToolGroup`, mas não são
  reexportados por `core._tools`/`core` (`core/_tools/__init__.py:26-59`). Isto torna gates próprios
  uma extensão por *duck typing*, não uma superfície pública completa.
- `ToolGroup.max_calls` é estado da instância e só reinicia com `group.reset()`
  (`_group.py:44-61`, `89-91`). Como `Agent` reutiliza o flow compilado, a aplicação tem de decidir
  se o limite é por execução e reiniciá-lo sem criar corridas entre execuções concorrentes. Para
  limites verdadeiramente por run, `BudgetPolicy.max_tool_calls` é a opção mais segura.

### 7. Metering e budgets — confirmado com ressalvas

Os três modos de meter estão explícitos (`core/_metering/_scope.py:1-12`). `RunConfig` fornece
controller, sinks, redactor, pricer e retenção de eventos (`_scope.py:56-68`). LLMs e ferramentas
abrem operações no ponto de cobrança (`core/_llm.py:583-642`;
`core/_tools/_executor.py:161-190`); o store fecha operações pendentes/iniciadas como
`aborted`/`incomplete` (`core/_metering/_store.py:469-485`). Um `UsageEvent` terminal é criado quando
há sinks (`_store.py:487-508`).

Ressalvas para uma UI de inspeção:

- operações bloqueadas antes da execução e dry-runs não geram operações de metering, por desenho;
- batch é explicitamente não medido por tentativa e é recusado sob budget, salvo opt-in
  (`core/_llm.py:1227-1239`);
- `OperationRequest` e `UsageEvent` têm campo `provider`, mas o pedido criado por `LLM` não o
  preenche (`_llm.py:607-618`), logo o sink não fornece hoje o fornecedor efetivo;
- ferramentas custam zero por omissão; custos externos exigem um `Pricer` próprio;
- fechar cedo um stream requer `aclose()`/context manager para não depender do fecho do scope.

### 8. Preços — confirmado

Há 72 prefixos no TOML em `48a43ac`; `ModelPricing` suporta preços standard, cache, batch, contexto
longo e fast (`core/_pricing.py:25-56`). Isso não significa que todas as 72 entradas tenham todas as
variantes. O registo usa longest-prefix match e é extensível por `register()`/`load()`
(`_pricing.py:64-127`, `257-296`).

### 9. Memória — confirmado com uma correção

`GraphStore` tem CRUD, relações, pesquisa, serialização, `save`/`load`
(`toolkit/memory/graph/_store.py:37-395`); há views, presets, `memory_tools` e
`MemoryMiddleware`.

Sem embedder não resta apenas a vista temporal: `GraphStore.search()` cai para pesquisa por
palavras-chave no backend (`_store.py:209-243`). O `BruteForceIndex` é efetivamente cosseno sobre
vetores (`memory/graph/_index.py:29-64`).

Não se deve confundir esta memória com a base de dados da app. `save()` é um snapshot JSON atómico,
mas não há transações, locks de domínio, histórico, rollback, isolamento por conversa ou store de
execuções. Conversas e runs continuam a precisar de persistência de produto.

### 10. Prompts, recursos e conhecimento — confirmado com limite de âmbito

As APIs existem e `ResourcePolicy.check_path()` canonicaliza e restringe raízes
(`toolkit/resources/_policy.py:14-75`). Porém, essa política é aplicada pelo sistema de recursos;
`read_file`, `list_directory` e `search_files` usam `Path` diretamente e não recebem
`ResourcePolicy` (`toolkit/tools/_filesystem.py:13-114`). Assim, `check_path()` pode ser reutilizado
por código da app, mas não constitui por si só uma política de permissões das ferramentas.

### 11. Matriz de compatibilidade — parcialmente confirmado

Há inventário e uma matriz baseada em probes reais. Não deve, porém, ser apresentada como uma
matriz atual e completamente verificada: o último full run registado é de 28 de abril de 2026, há
cenários `Not probed`, `Fail` e `Unsupported`, e o GPT-6 Astra foi adicionado sem probe live
(`docs/model-compatibility.md:13-22`, `47-58`, `67-139`). Além disso, estes dados não são exportados
pelo pacote.

## Revisão L1–L13

### L1 · Cliente MCP — confirmo

**Evidência:** não há cliente, transportes, descoberta nem conversão MCP em `core/` ou `toolkit/`.
Também não existe extra `mcp` em `pyproject.toml`. As classes públicas tornam o adaptador possível,
mas não pronto: um callable com `ToolDefinition` é suficiente para `ToolGroup`, e o executor trata
de gates, aprovação e metering.

**Onde pertence:** ao toolkit, porque a conversão de uma ferramenta remota no contrato canónico e
o seu lifecycle são independentes da app. Whitelists, credenciais, política “sempre/perguntar/nunca”
e UI pertencem à app.

**API recomendada:** não recuperar a antiga classe imaginária `Tools`; alinhar com `ToolGroup`.
Um cliente MCP tem lifecycle, pelo que uma simples `Tools.from_mcp(...) -> list` é insuficiente:

```python
from ai_arch_toolkit.toolkit.mcp import MCPClient, StdioTransport

async with MCPClient(StdioTransport(command, args=args, env=env)) as mcp:
    group = await mcp.tool_group(
        include={"search", "read"},
        default_policy=ToolRuntimePolicy(
            capability="mcp:github",
            risk_level="high",
            requires_approval=True,
        ),
    )
```

O módulo sugerido é `toolkit.mcp`, com extra opcional `mcp` sobre o SDK oficial, e transportes
stdio e streamable HTTP. O `ToolGroup` deve ganhar `add_definition()` ou uma factory pública
`tool_from_callable(...)`.

`@tool(schema=...)` não aceita hoje um schema completo: `schema` é passado como `overrides` por
parâmetro (`core/_tools/_decorator.py:23-53`; `_schema.py:310-345`). Para não quebrar esse contrato,
é preferível acrescentar `input_schema=` (schema JSON completo) e validar que não é usado ao mesmo
tempo que `schema=`.

### L2 · Agendador — confirmo

Não existe scheduler em `core/`/`toolkit/`. Deve ficar na app. Horários, timezone, persistência,
catch-up, execução com a janela fechada, notificações, concorrência entre jobs e política de missed
runs são decisões de produto/runtime, não de raciocínio de agentes. O toolkit só tem de expor uma
operação executável e cancelável; não deve possuir o relógio do produto.

### L3 · Máquinas de estados declarativas — confirmo, mas refuto a compilação proposta tal como está

Existe mecanismo de ciclo condicional, não uma máquina de estados. `Flow` não modela transições,
prioridade, estado terminal ou caminho como conceitos (`toolkit/flow/_flow.py:31-38`, `129-192`).

Compilar “um Step por estado” pode funcionar para casos simples, mas
`Flow.max_iterations` **não é** um limite de passos/transições. Numa passagem, cada condição é
reavaliada sobre o estado já alterado pelo passo anterior; vários estados podem executar na mesma
iteração (`flow/_executor.py:132-183`). Além disso, erros numa condição de um flow cíclico são
registados como skips e a execução continua (`_executor.py:143-165`), o que pode não ser aceitável
numa máquina de estados.

**Onde pertence:** a DSL/biblioteca criada pelo utilizador começa na app; o mecanismo genérico,
quando os casos reais estabilizarem, faz sentido em `toolkit.flow`, não como estratégia de agente.
Só se cada “estado” for especificamente uma fase de raciocínio LLM é que `register_strategy("machine",
...)` é a camada certa.

**API futura:** `StateMachineSpec`, `TransitionSpec(priority, condition_ref, target)`,
`max_transitions`, `terminal_states`, `compile_state_machine(spec, actions, conditions) -> Flow` e
um caminho visitado explícito no resultado. Condições declarativas devem ser ids resolvidos por um
registo, nunca código arbitrário serializado.

### L4 · Workflows declarativos/serializáveis — confirmo

`Flow` contém callables (`Step.fn`, `when`, scopes) e não tem `to_dict/from_dict`. Os manifestos de
agente só validam `tools.manifest` como string e deixam a sua resolução para a app
(`agents/_manifest.py:95`, `160-177`, `511-515`).

**Onde pertence:** a primeira versão deve ficar na app, compilada para `Flow`, como proposto. O
formato é parte do modelo de produto (ids de agentes, permissões, biblioteca, versões). Se surgir um
subconjunto realmente genérico, o toolkit pode depois oferecer `toolkit.flow.manifest`.

**Forma futura:** um `FlowSpec` versionado contém apenas referências (`step_ref`, `agent_ref`,
`condition_ref`, `workflow_ref`), `after`, parâmetros e política. `compile_flow(spec, registry)`
resolve referências runtime e rejeita ciclos/ids desconhecidos. Não se devem tentar serializar
callables Python.

### L5 · Streaming de tokens nas estratégias — confirmo

`react_flow` chama `llm.complete()` (`agents/flows/_react.py:57-99`); as restantes arquiteturas
também acabam em `complete()`. `FlowEvent` não tem eventos de token/tool (`flow/_flow.py:92-112`).
Não há uma via escondida em `react_flow`.

**Onde pertence:** toolkit e com prioridade alta para uma app de chat. Um `on_delta` serve como
remendo, mas um iterador estruturado lida melhor com backpressure, cancelamento, múltiplas fases e
correlação:

```python
stream = agent.stream(task, config=run_config)
async for event in stream:
    # text_delta, thinking_delta, tool_call, tool_result,
    # step_start/end, policy_decision, flow_end
    render(event)
result = stream.result
```

Os eventos devem levar `run_id`, `step/phase`, índice da iteração e ids de correlação. Sem nova
dependência.

O loop temporário da app é viável, mas deve usar `ToolGroup.async_execute()` (não a função raw),
acrescentar `response.to_message()` e mensagens `tool_result`, manter o `MeterScope`, e fechar o
stream explicitamente. Isso duplica semântica ReAct e deve ser temporário.

### L6 · Checkpoint e retoma — confirmo; a proposta da app é insuficiente

`State.from_trace()` reconstrói deliberadamente o estado **inicial**
(`core/_state.py:230-237`). O executor guarda `completed/failed/skipped` apenas em variáveis locais
para DAGs (`flow/_executor.py:321-325`) e `iteration` apenas localmente para ciclos
(`_executor.py:132-183`). Não recebe cursor nem conjunto de passos concluídos.

Logo, persistir `Result.artifacts` e “reentrar no passo seguinte” só é seguro num flow sequencial
muito simples recompilado pela app. Não preserva scheduling de DAG, skips/falhas, iteração cíclica,
histórico de resultados, chamadas de ferramenta parcialmente concluídas nem budget consumido.

**Onde pertence:** dividido. O toolkit deve definir o cursor/checkpoint e a semântica idempotente de
retoma; a app fornece o `CheckpointStore`, durabilidade, migrações, política após crash e UI.

**API futura:** `FlowRun`/`AgentSession` com `step()`, `stop()`, `inject()` e
`checkpoint()`, mais `Flow.resume(checkpoint, config=...)`. Um checkpoint precisa de versão e
fingerprint do flow, estado JSON com codecs para `Response`/`Result`, status de cada nó, cursor de
iteração/transição, resultados, operações externas idempotentes e snapshot/continuação do budget.

O `Session` do briefing não pode ser considerado plano atual: `docs/internal/README.md:3-13`
classifica `from_claude_chat/` como sketches históricos mantidos para rastreabilidade, fora da
documentação pública.

### L7 · Router e metadados de modelos — confirmo, com divisão mais fina

`ModelPricing` só contém tarifas e thresholds de preço (`core/_pricing.py:25-56`); não há catálogo
runtime nem router. A matriz de probes vive fora do pacote e está desatualizada para decisões
automáticas em runtime.

**Onde pertence:**

- no **core**, um catálogo técnico extensível pode expor factos operacionais que o próprio cliente
  precisa: context window, max output, tools, structured output, JSON, streaming, thinking e server
  tools, com `source`/`verified_at` e overrides;
- na **app**, ficam qualidade/inteligência, benchmarks, latência observada, região/data residency,
  pesos abertos, modelos permitidos e a função de utilidade do `Auto`;
- um router opinativo genérico poderia existir mais tarde em `toolkit.routing`, mas não deve ser
  pré-requisito do `LLM`.

Não convém misturar todo o catálogo com `ModelPricing`: disponibilidade/capacidade varia por API,
deployment e versão, enquanto preços têm outra cadência. Uma forma possível é
`ModelCatalog.get(model, provider=...) -> ModelCapabilities` e `catalog.load(...)`.

### L8 · Escolha automática de arquitetura — confirmo

O registo apenas armazena/resolve estratégias (`agents/_builders.py:145-166`); não seleciona uma.
Deve ficar na app, porque a escolha e a explicação fazem parte do comportamento do `Auto` e podem
usar telemetria real. O toolkit já fornece a fronteira certa: devolver um `ReasoningSpec` com um
nome registado. Mais tarde, um protocolo opcional `StrategySelector` pode padronizar a integração,
sem impor heurística.

### L9 · Capacidades do computador “só de leitura” — refuto a formulação; confirmo a lacuna tipada

As ferramentas dedicadas de filesystem são de leitura
(`toolkit/tools/_filesystem.py:13-114`), mas `run_command` usa `shell=True` e pode modificar ou
destruir o sistema (`toolkit/tools/_shell.py:13-55`). Portanto, capacidade de escrita existe de
forma genérica e crítica; o que falta são operações estruturadas com schemas e políticas próprias.

**Onde pertence:**

- `write_text`, escrita atómica, append, move e eventualmente remoção recuperável são bons
  candidatos a `toolkit.tools.dangerous`;
- abrir aplicações/URLs, clipboard e integrações de aplicações devem ficar na camada Tauri/app,
  porque são específicos do OS, sessão gráfica e modelo de permissões;
- `system_info` pode ser uma ferramenta cross-platform, mas a app deve decidir que dados privados
  expõe.

Não usar `ResourcePolicy` sem adaptação como política completa de escrita. É preferível introduzir
`FilesystemPolicy`/`PathScopeGate`, com ações `read/write/delete`, tratamento de destinos ainda
inexistentes, symlinks e verificação imediatamente antes da operação. Para extensibilidade real, o
toolkit deve também reexportar `ToolGate` e `ExecutionContext`.

### L10 · Sandbox para código runtime — confirmo

`python_repl` é um interpretador AST allowlistado em processo, com limites de statements/range/etc.,
sem `eval/exec`, imports ou ficheiros (`toolkit/tools/_python.py:1-46`, `277-340`, `716-765`). É útil
para pequenos cálculos/código produzido pelo modelo, mas não é isolamento de processo, memória, CPU
ou syscalls. O próprio módulo `dangerous` manda acrescentar sandboxing externo antes de o expor
(`toolkit/tools/dangerous.py:1-6`).

Gerar e instalar novas ferramentas Python em runtime é outra capacidade e não deve usar
`_SafeEvaluator`. O spike deve ficar na app/plataforma. Se estabilizar, o toolkit pode expor apenas
um protocolo de `CodeRunner`/`ToolLoader`; a sandbox efetiva deve ser um processo/container/VM com
limites e filesystem/network explicitamente montados. Não há evidência de um plano atual para isto.

### L11 · Agente como ferramenta — confirmo

Não há adapter direto. `Agent.as_step()` é composição de flows, não uma ferramenta. A parte boa da
proposta está correta: um agente async executado dentro do callable herdará o meter atual, porque o
flow aninhado lê `current_meter()` (`flow/_executor.py:600-619`).

**Onde pertence:** toolkit, por exemplo em `toolkit.agents.tools`:

```python
delegate = agent_as_tool(
    researcher,
    name="delegate_research",
    description="Delegate a bounded research task.",
    policy=ToolRuntimePolicy(...),
    max_depth=3,
)
```

Deve devolver um callable compatível com `ToolGroup`, preservar `AgentResult`/erro em metadata e
impor profundidade/ciclos com `ContextVar`. Aprovação, catálogo de agentes disponíveis e gravação da
execução filha pertencem à app. Para a inspeção ficar completa, é também necessário resolver a
lacuna atual de traces filhos/correlação, não apenas devolver texto.

### L12 · Fornecedores, cofre e descoberta local — confirmo

As chaves são resolvidas por `api_key` ou ambiente; loopback evita reutilizar a chave cloud
(`core/_providers/__init__.py:92-109`). Não há armazenamento de credenciais nem listagem de modelos
Ollama/LM Studio. Tudo isto deve ficar na app: keychain/credential vault do OS, perfis de endpoint,
health checks e descoberta. O toolkit deve continuar a receber `LLM(model, base_url, api_key, ...)`.

### L13 · Pesquisa web genérica — refuto em parte

O `core` exporta `web_search()` como `ServerTool` (`core/_server_tools.py:9-28`;
`core/__init__.py:71`, `244`). Anthropic mapeia-o para a sua ferramenta versionada
(`core/_providers/_anthropic.py:43-47`, `452-461`), OpenAI para `{"type": "web_search"}`
(`_openai.py:407-417`) e Gemini para Google Search (`_gemini.py:443-459`). xAI rejeita-o nesta via
do SDK (`_xai.py:394-407`).

Limites reais:

- é pesquisa executada e faturada pelo fornecedor do modelo, não um tool client-side portável;
- `ServerTool.config` é atualmente ignorado pelos providers, como o próprio TODO diz
  (`core/_server_tools.py:16-19`);
- não funciona para modelos locais/OpenAI-compatible sem server tool;
- o custo do pedido torna-se `unknown` porque a cobrança da ferramenta não está nos tokens
  (`core/_pricing.py:229-252`).

O inventário do próprio toolkit explica por que DuckDuckGo Instant Answer não foi tratado como
pesquisa geral e lista Brave/Tavily como candidatos com chave
(`toolkit/tools/CANDIDATE_TOOLS.md:134-142`, `234-284`). Não aconselho scraping de HTML de motores de
pesquisa. Para independência de fornecedor, acrescentaria integrações explícitas
`brave_web_search`/`tavily_search`, configuradas pela app, sem fingir que DuckDuckGo Instant Answer
tem cobertura equivalente.

## Revisão da secção 4 (“não são limitações”)

- **Aprovação humana assíncrona:** confirmado. O handler pode devolver awaitable e a ausência de
  handler nega (`core/_tools/_approval.py:132-157`). “Sempre/perguntar/nunca” e a persistência da
  decisão são política da app. A via sync não pode esperar por aprovação assíncrona.
- **Budgets aninhados:** confirmado quando o agente filho corre no mesmo contexto async/bound meter.
  O flow filho ignora o seu budget próprio e partilha o cumulativo do pai
  (`flow/_executor.py:606-625`).
- **`RunConfig` não se mistura com `budget_policy`:** confirmado
  (`flow/_flow.py:238-245`). O controller tem de ser incluído no config.
- **Manifestos sem segredos e fingerprint:** confirmado, com a ressalva de que são config de agente,
  não snapshot completo do runtime nem checkpoint.
- **`ResourcePolicy.check_path` para âmbitos:** parcialmente confirmado. É uma boa primitiva de
  validação de leitura, mas não está ligada às ferramentas nem implementa a política de permissões
  da app.
- **Memória sem embeddings:** corrigir. Sem embedder há pesquisa keyword em `store.search`, além das
  vistas temporal/propriedades/relações; não há semelhança semântica vetorial.
- **`nanope/`:** confirmado no sentido importante: é WIP, não API pública, e está excluído de ruff e
  pyright. Não deve ser usado como contrato pela app.

## O que escapou ao diagnóstico original

Estas lacunas são especialmente relevantes para a app descrita:

1. **`Flow.iter()` serializa DAGs que `Flow.run()` executa em paralelo.** Uma UI que observa a
   execução altera hoje a concorrência do workflow.
2. **Não há árvore de traces de subflows/subagentes.** `StepTrace.children` existe mas não é
   preenchido, impedindo inspeção hierárquica pronta a usar.
3. **`Agent` não recebe `RunConfig`.** É possível contornar com um `MeterScope` externo, mas a API de
   alto nível não deixa configurar sinks/retention diretamente.
4. **A telemetria não identifica o provider efetivo.** `UsageEvent.provider` existe mas LLM não o
   preenche; isto também complica fallbacks entre fornecedores.
5. **`run_tools(response, ToolGroup)` não preserva os gates/max_calls do grupo.** `_normalize_tools`
   extrai `group.tools` e chama o executor livre, que só instala `ApprovalGate`
   (`toolkit/_runner.py:25-60`; `core/_tools/_executor.py:317-339`). Para a app, usar sempre
   `ToolGroup.async_execute()` no loop manual.
6. **`ToolGroup.max_calls` é por instância, não automaticamente por run.** Reutilização e runs
   concorrentes precisam de cuidado; preferir o meter para caps por execução.
7. **`State.to_dict()` não é um checkpoint JSON.** Objetos runtime e curso do executor não são
   serializados.
8. **Batch não entra no meter por tentativa.** Não serve para runs que exijam auditoria/budget forte
   sem reconciliação própria.
9. **A memória não é o event store da aplicação.** Não oferece concorrência, histórico/rollback de
   alterações nem persistência de conversas/runs.
10. **Server-tool config é descartado.** Allowlist de domínios/max uses passada a `web_search(...)`
    não chega hoje ao wire.
11. **O pacote está em `0.1.0.dev0`.** Não há símbolos efetivamente marcados como deprecated em
    `core/`/`toolkit/`, mas a versão é pré-1.0; importar módulos `_...` ou tratar docs internos como
    contrato aumenta muito o risco de quebra.

## Respostas às perguntas de intenção

### 1. O briefing antigo ainda é o rumo?

Não há base documental para o tratar como roadmap atual. `docs/internal/README.md` diz que são
auditorias/planos históricos e que `from_claude_chat/` contém sketches mantidos para
rastreabilidade. A implementação também escolheu contratos diferentes (`ToolGroup`,
`ReasoningSpec`, `Flow`, graph memory). `from_mcp`, `from_openapi`, `from_agent`, `Session`,
`allow_spawn` e `Memory.lock/history/rollback` devem ser considerados ideias não implementadas, não
promessas.

Recomendação de prioridade atual:

1. contrato unificado de eventos, `Agent.stream`, `RunConfig` no Agent e correlação/trace de filhos;
2. API pública para ferramentas dinâmicas + `agent_as_tool`;
3. cliente MCP com lifecycle;
4. cursor/checkpoint/retoma do executor;
5. specs declarativas de flow/máquina de estados, depois de validadas na app;
6. catálogo técnico de capacidades dos modelos.

`from_openapi`, self-modification/spawn genérico e memória mutável com rollback são posteriores e
devem ser reavaliados em vez de implementados literalmente a partir do sketch.

### 2. E a porta Rust?

Não há referência a uma porta Rust no código ou documentação de arquitetura revistos. Portanto,
não se pode confirmar o compromisso nem a ordem. Se for uma intenção real, convém estabilizar
primeiro em Python os contratos neutros de linguagem: eventos, schemas de tool, checkpoint,
`FlowSpec` e erros. Scheduler, vault, routing `Auto`, descoberta local, integração com o desktop e
persistência de produto não devem ser portados como toolkit.

### 3. O que não usar como está?

- Não importar módulos `_...` nem copiar APIs de `docs/internal/from_claude_chat`.
- Não usar `Flow.iter()` se o workflow depende de paralelismo de DAG até a divergência ser corrigida.
- Não usar `run_tools(..., group)` quando são necessários gates/max_calls próprios do grupo.
- Não persistir `State.to_dict()` como se fosse checkpoint.
- Não depender de `ServerTool.config` até os adapters o encaminharem.
- Não usar a matriz de probes como catálogo runtime atual sem data/proveniência e novo probe.
- Não usar batch sob requisitos de budget/auditoria por chamada.

Não há deprecações ativas encontradas em `core/`/`toolkit/`; o risco é sobretudo API pré-1.0 e uso de
superfícies internas.

### 4. O que a app não deve reconstruir

A app deve reutilizar diretamente:

- `LLM`, routing por prefixo/base URL, SDKs oficiais, retry/fallback e streams ricos;
- `OutputSchema`, JSON mode, thinking e normalização provider-agnostic de mensagens/respostas;
- `ToolDefinition`/`ToolGroup`, aprovação async, alteração de argumentos, gates e executor com
  metering;
- `MeterScope`/`RunConfig`/`UsageSink`, `BudgetPolicy` e redaction;
- `Flow`, `Step`, policies, DAG/sequence/cycles e composição `Flow.as_step`/`Agent.as_step`, dentro das
  ressalvas acima;
- as dez estratégias via `Agent(ReasoningSpec(...))` e o registo de estratégias;
- manifestos de agente, incluindo herança, perfis, fingerprints, limites e modelos/prompts por fase;
- prompts, resources, knowledge e respetivos loaders/policies;
- graph memory, keyword/vector search, views, presets, middleware e memory tools;
- `core.web_search()` e `core.code_execution()` quando o fornecedor/modelo os suporta;
- `inference_limit` para limitar concorrência de chamadas LLM.

A app deve construir a camada de produto que falta: identidade e permissões persistentes,
conversas/runs, árvore de execução enquanto o toolkit não a produzir, scheduler, notificações,
catálogo/router `Auto`, cofre, descoberta local, integrações Tauri/OS e o formato inicial da sua
biblioteca de workflows.

---


---

# Segunda revisão — verificação independente

**Data:** 12 de setembro de 2026. **Código:** `main` em `48a43ac`.
Feita contra os documentos de arquitectura (`framework-overview.md`, `flow-architecture.md`,
`safety.md`, `tools.md`, `moderation.md`, `configuring-agents.md`), não apenas contra o código, e
com cada afirmação executada como teste. Inclui as correcções de uma terceira leitura crítica.

## Veredicto sobre o documento da app

O diagnóstico está globalmente certo. **L1, L2, L4, L6, L7, L8, L10, L11 e L12 confirmam-se sem
reservas.** L5 confirma-se e é pior do que o documento diz (ponto A). L3 confirma-se no facto mas a
compilação proposta não funciona como descrita. L9 e L13 estão mal formuladas — certas no facto,
erradas na palavra:

- **L9** diz "só de leitura". `run_command` corre `shell=True`, logo escreve, apaga e abre
  aplicações. O que falta são operações **tipadas e governáveis**, não capacidade de escrita.
- **L13** diz "nenhuma pesquisa web". Existe `core.web_search()` como server tool em Anthropic,
  OpenAI e Gemini. O que falta é pesquisa client-side, portável, que funcione com modelos locais e
  dentro do canal de tools das estratégias standard (ponto D).

## Confirmações da primeira revisão

Reverifiquei os pontos que a sustentam. Todos se confirmam:

| Afirmação | Evidência |
|---|---|
| `_iter_dag` serializa o que `_execute_dag` paraleliza | `flow/_executor.py:449-537` vs `:383-445` |
| `StepTrace.children` nunca é preenchido | `Flow.as_step` descarta `flow_result.trace` (`flow/_flow.py:283-318`) |
| `run_tools(..., group)` perde gates e `max_calls` | `toolkit/_runner.py:25-29`; `core/_tools/_executor.py:317-339` |
| `UsageEvent.provider` nunca é preenchido pelo `LLM` | `OperationRequest(...)` em `_llm.py:607-618` |
| `ToolGate`/`ExecutionContext` não são públicos | ausentes de `core/_tools/__init__.py` e `core/__init__.py` |
| `ServerTool.config` é descartado | os adapters só leem `st["type"]` |
| xAI recusa server tools | `NotImplementedError` em `_providers/_xai.py:396-407` |
| Batch fora do meter, recusado sob budget | `_llm.py:1227-1239` |
| Custo fica `unknown` com server tools | `_pricing.py:236-238` |
| `resolve_approval_sync` nega handlers async | `core/_tools/_approval.py:146-157` |
| `State.to_dict()` não é checkpoint JSON | o `world` guarda o `MeterScope` |
| Matriz de compatibilidade de 2026-04-28 | `docs/model-compatibility.md:13-22` |

## Os achados

Oito pontos, cada um verificado com código executado.

O padrão que os liga é **erosão de contratos entre APIs supostamente equivalentes**: capacidades
declaradas que mudam, desaparecem ou deixam de funcionar consoante o caminho escolhido. `run()` vs
`iter()`, `complete()` vs `stream_events()`, `ToolGroup.async_execute()` vs `run_tools()`, flow
directo vs flow aninhado. Não são funcionalidades em falta — são promessas já feitas que se perdem
ao atravessar a fronteira errada.

**G é o mais grave e é novo:** uma `Policy` declarada num `Flow` não tem efeito nenhum quando esse
flow corre directamente, o que torna `ReasoningSpec(timeout=...)` silenciosamente inerte.

A, B, E e F são o mesmo problema visto de quatro ângulos — o toolkit não tem um contrato de execução
observável: quem quer ver o que se passa perde middleware (A), perde o isolamento do DAG (B), não
recebe os eventos que o tipo promete (E) e não obtém resultado no fim (F).

### A · Bug de contrato: middleware assíncrono é ignorado em streaming

**Este é o achado importante, e é um bug, não uma lacuna de desenho.**

O caminho de stream corre middleware pelos hooks **síncronos** (`_run_before` em `_llm.py:820-829`,
`_run_after` em `:964-966`); os hooks async só correm em `complete()` (`_llm.py:702-757`). Nos dois
middlewares do toolkit o `before`/`after` síncrono é um no-op declarado — o trabalho está todo no
`abefore`/`aafter` (`toolkit/moderation/_middleware.py:50-56`; `toolkit/memory/_middleware.py:43-49`).

Medido com um middleware-espia dos quatro hooks e com um moderador que sinaliza sempre:

```
complete()      -> ['abefore(async)', 'aafter(async)']   -> BLOQUEADO por ModerationError
stream_events() -> ['before(sync)', 'after(sync)']       -> PASSOU, texto entregue
```

Porque é contrato e não desenho:

- `docs/middleware.md:3` — "Middleware hooks into **every** LLM call". A única excepção documentada
  é o rate limiter (`:115`).
- `docs/moderation.md:62` — "Output moderation runs after stream finalization, so streamed text may
  reach the user before the output check completes — **prefer `input` screening** (or non-streaming
  calls) when you must block before display."

Ou seja, a documentação promete que a moderação de output corre depois da finalização — não corre de
todo — e a mitigação que recomenda em alternativa, o *input screening*, **também não corre em
streaming**. As duas metades da promessa estão partidas.

**A correcção não é "hooks síncronos reais".** Moderação e memória fazem trabalho assíncrono
(chamar um classificador, consultar o grafo); um hook síncrono não o consegue executar. O stream
precisa de uma pipeline assíncrona lazy:

1. correr `abefore()` **antes** de abrir o stream do provider;
2. entregar os eventos;
3. construir a resposta final;
4. correr e **esperar** por `aafter()`;
5. fechar metering e recursos.

E fica por decidir uma questão de produto que o toolkit não pode resolver sozinho: moderar o output
depois da finalização não impede que os tokens já tenham sido mostrados. Bloquear antes da
apresentação exige buffering ou moderação incremental — decisão da app, mas o toolkit tem de dar o
ponto de intercepção.

Para a app: até isto estar resolvido, um loop de streaming próprio corre **sem moderação e sem
memória**, sem erro nem aviso. O mesmo se aplica a um `Agent.from_flow()` cujo step chame
`LLM.stream_events()` — a via que existe hoje para ter tokens dentro de um `Agent` é exactamente a
via que perde o middleware. "Streaming possível num step próprio" lê-se, em rigor, "possível, mas
sem middleware".

### B · `Flow.iter()` sobre um DAG viola o contrato de isolamento — abrir como bug

`_execute_dag` faz `state.fork()` por passo e só junta no fim da vaga; `_iter_dag` corre os passos
prontos um a um sobre o `State` partilhado e faz `state.merge()` a seguir a cada um. Dois passos
irmãos `a` e `b` sem dependência entre si, com `b` a ler uma chave que `a` escreve:

```
run()  -> b viu: '<NADA>'
iter() -> b viu: 'escrito por A'
```

`flow-architecture.md` garante que passos paralelos "cannot see each other's writes" e que o merge
acontece depois de todos terminarem. `iter()` quebra as duas garantias, e pode produzir resultados
**funcionalmente diferentes** de `run()` — não é só perda de paralelismo. Uma UI que observe a
execução muda o resultado do workflow.

`Scope` não resolve isto: filtra ou transforma o snapshot entregue a cada step, e como o snapshot do
segundo irmão é construído **depois** do merge do primeiro, as escrituras continuam visíveis se
estiverem incluídas no scope.

**Quem é afectado.** Nenhuma das dez estratégias standard constrói o seu flow principal com `after=`
(verificado: zero ocorrências em `toolkit/agents/`), portanto nenhuma delas passa pelo caminho do
DAG. Quando precisam de paralelismo, fazem-no dentro de um step — `llm_compiler` monta um DAG
próprio com `_DAGTask` + `asyncio.gather` dentro do único step `compile` (`_llm_compiler.py:111-172`).
O bug atinge, portanto, flows construídos à mão. Para o toolkit em geral é baixa prioridade; **para
esta app não é**: os planos L3 (máquinas de estados) e L4 (workflows) são exactamente compilar specs
para `Flow` com `after=` e observá-los numa UI — a combinação que dispara o bug.

**A correcção estrutural: um só motor.** A causa não é o DAG, é haver dois executores
(`run() → _execute_dag`, `iter() → _iter_dag`) que divergiram. As duas APIs públicas justificam-se —
`run()` para testes, batch, flows aninhados e agentes-como-step; `iter()` para progresso na UI,
decisões de policy, cancelamento e observabilidade — mas não justificam dois motores. O desenho
correcto é um motor canónico que emite eventos, com `run()` a drenar o iterador:

```python
execution = flow.start(state, config=config)
async for event in execution:
    update_ui(event)
result = execution.result

# run() passa a ser conveniência:
async def run(self, state, **kw):
    execution = self.start(state, **kw)
    async for _ in execution:
        pass
    return execution.result
```

Ganha-se uma única semântica de DAG, `run()` e `iter()` com o mesmo estado e resultado, budgets,
retries e traces por um só caminho, e o iterador passa a expor o resultado final (ver ponto E).

**A dificuldade real, que a proposta subestima:** um gerador assíncrono simples não consegue emitir
eventos de dentro de um `asyncio.gather`. Foi por isso que `_iter_dag` acabou sequencial — é a forma
ingénua de escrever um gerador sobre uma vaga de passos. O motor unificado tem de empurrar eventos
para uma fila que o consumidor drena, com as tarefas paralelas a publicar nela à medida que acabam.
Não é uma refactorização de meia hora, e há semântica a definir: o que vale `execution.result` num
iterador abandonado a meio (hoje o `finally: scope.close()` já finaliza o meter em `GeneratorExit`).

**Nomes.** `iter()` confunde-se com streaming de tokens, que é outra coisa. Vale a pena separar:
`Flow.iter_events()` para eventos de execução, `Agent.stream()` para texto/thinking/tools mais
eventos, `run()` para o resultado final. O pacote está em `0.1.0.dev0` e sem deprecações activas,
portanto renomear custa pouco — toca em `run`/`run_sync`/`iter`/`iter_sync`, `execute_flow`/
`iter_flow` e `Agent.iter`.

### C · Falha de hardening nos metadados de `tools.dangerous`

Duas das sete ferramentas declaram risco (`run_command`: `capability="shell"`,
`risk_level="critical"`, `requires_approval=True`; `python_repl`: `"python"`, `"high"`, `True`). As
outras cinco — `read_file`, `list_directory`, `search_files`, `http_get`, `scrape_text` — usam
`@tool` nu, logo `capability=None`, `risk_level="low"`, `requires_approval=False`.

**Não é vulnerabilidade por omissão do toolkit:** o módulo é opt-in, `docs/safety.md` manda instalar
`DangerousToolGate(blocked=[...])`, que bloqueia por nome, e o docstring do módulo põe sandboxing e
permissões do lado da aplicação. Esse contrato cumpre-se.

**Mas também não é cosmético.** `docs/safety.md` apresenta um mecanismo conjunto — metadados de
risco em cada tool, gates que os consultam, e negação por omissão para as que exigem aprovação.
Ferramentas oficiais do módulo `dangerous` a aparecerem como `low`, sem capability e sem aprovação
podem induzir uma aplicação a confiar numa classificação errada. O veredicto certo é: **falha de
hardening e de coerência dos metadados públicos**, não vulnerabilidade.

Para a app: aplicar permissões próprias por nome/capability e por caminho, e não derivar o nível de
risco destas cinco do `risk_level` delas.

### D · As estratégias standard não combinam tools client-side e server-side

`ToolGroup(web_search())` rebenta com `AttributeError: 'ServerTool' object has no attribute
'__name__'`, e `react_flow(llm, tools: ToolGroup)` / `Agent(..., tools: ToolGroup | None)` só
aceitam um `ToolGroup`. `ReasoningSpec` não tem campo para server tools, e passar `tools=` por
`llm_kwargs` colidiria com o argumento que a estratégia já fornece.

**Mas não são inalcançáveis dentro de um `Agent`.** `Agent.from_flow()` envolve um flow próprio cujo
step chame directamente o LLM. Verificado:

```
Agent.from_flow(Flow(Step(name="call", fn=step_fn))).run("ola")
  step_fn: await llm.complete(messages, tools=[group, web_search()])
  -> chegou ao provider: [{'name': 'read_note', ...}, {'_server_tool': True, 'type': 'web_search'}]
```

A lacuna precisa é: **as estratégias standard não aceitam uma combinação de tools client-side e
server-side.**

E a correcção que eu tinha sugerido — aceitar `ServerTool` dentro de `ToolGroup` — está errada. O
grupo executa funções localmente, com gates, aprovação e metering; server tools são executadas pelo
fornecedor e nada disso se lhes aplica. Um canal separado é mais correcto: `server_tools=` nas
factories e no `Agent`, ou um `ToolSet(client=..., server=...)`. Independentemente disso, o
`AttributeError` devia ser substituído por um erro de validação explícito.

### E · O canal de eventos é mais pobre do que o tipo promete

`FlowEvent.type` declara nove valores (`_flow.py:96-106`): `flow_start`, `flow_end`, `step_start`,
`step_end`, `step_skipped`, `retry`, `fallback`, `timeout`, `policy_decision`. **Três nunca são
emitidos**: não há uma única ocorrência de `type="retry"`, `type="fallback"` ou `type="timeout"` em
todo o pacote. E `policy_decision`, o único que sobra para decisões de política, só é emitido com
`policy_decision="budget_exceeded"`.

Ou seja, as decisões que a `Trace` regista em `StepTrace.policy_decisions` — retry, fallback,
timeout, low-confidence — **não chegam ao stream de eventos**. Uma UI que faça `match` sobre esses
três tipos nunca recebe nada, e não tem como mostrar "o passo falhou e vai tentar outra vez".

Isto compõe-se com a granularidade grossa das estratégias. Seis delas chamam `react_flow(...).run(state)`
dentro de um step (`_reflexion.py:82`, `_plan_execute.py:100`, `_llm_compiler.py:161`,
`_self_discovery.py:152`, `_lats.py:144`, `_generate_review.py:82`, `:113`) — com `run()`, não
`as_step()`, logo o flow filho não produz eventos nem fica ligado em `StepTrace.children`. Para uma
UI, `Agent.iter()` sobre `llm_compiler` mostra literalmente:

```
step_start: compile
   ... todo o planeamento, o DAG interno e as sub-tarefas ReAct, invisíveis ...
step_end: compile
```

Faltam eventos uniformes por fase, chamada LLM, tool call, sub-tarefa, subagente e iteração interna.
É a mesma lacuna que o ponto A, vista do outro lado: o toolkit não tem hoje um contrato de eventos
que sirva uma interface.

### F · `Agent.iter()` não devolve resultado nem dá acesso ao estado

`Agent.run()` devolve um `AgentResult` com texto, `Response`, `FlowResult`, usage, custo, relatório
de budget e erros. `Agent.iter()` cria o `State` internamente (`_agent.py:127`, variável local) e só
entrega `FlowEvent`s. No fim, quem consumiu o stream tem a `Trace` do `flow_end` e mais nada: não
recebe `AgentResult`, não alcança o `State`, e tem de inferir a resposta a partir dos eventos.

Para uma UI de chat — que é precisamente o caso da app — isto obriga a escolher entre observar a
execução e obter um resultado estruturado. É a mesma correcção do ponto B vista do lado do `Agent`:
um stream devia expor `stream.result` depois de consumido.

Relacionado, duas limitações menores do `Agent`:

- **`RunConfig` não é aceite por execução.** `run`/`iter` só levam `budget_policy`. Sinks, redactor,
  pricer e retenção de eventos exigem envolver a chamada num `MeterScope(RunConfig(...))` — que
  funciona, porque `_open_meter_scope` herda o scope ambiente. É limitação ergonómica, não ausência
  de capacidade.
- **`from_flow(init_state=...)` só semeia a camada `operational`.** Não há como, pelo `Agent`,
  fornecer um `State` completo ou inicializar `current`/`persistent`/`world`. Para isso é preciso
  descer a `flow.run(State(...))`.

### G · `Flow.policy` não é aplicada em execução directa — timeouts silenciosamente inertes

**O achado mais grave de toda a revisão.**

`Flow` guarda uma `Policy` e expõe-a em `flow.policy`, mas o executor nunca a usa. `execute_step()`
lê `step.policy or Policy()` (`core/_step_engine.py:27`) e `_execute_flow_step` entrega-lhe o step
tal e qual — a policy do flow não desce para lado nenhum. O único sítio onde é transferida é
`Flow.as_step()`, que a coloca no `Step` resultante (`_flow.py:318`).

Resultado: **o mesmo flow comporta-se de maneira diferente consoante seja corrido directamente ou
composto dentro de outro.** Medido, com um step de 50 ms e `Policy(timeout=0.001)`:

```
policy no Flow, corrido directo   -> erro=None              decisões=()
policy no Flow, aninhado noutro   -> erro='Step timed out'  decisões=('timeout',)
policy no Step                    -> erro='Step timed out'  decisões=('timeout',)
```

A assimetria é clara quando se compara com `Scope`, que **cascateia**:
`_resolve_and_apply_scope` resolve `FlowStep.scope or Step.scope or flow.scope`
(`_executor.py:564-567`). Para a policy não existe equivalente. Duas propriedades irmãs do mesmo
objecto, uma cascateia e a outra não.

**A consequência atinge as estratégias standard.** `react_flow(timeout=...)` constrói
`Flow(..., policy=Policy(timeout=timeout))` (`_react.py:180-190`), e `ReasoningSpec.policy` /
`ReasoningSpec.timeout` acabam no mesmo sítio. `Agent.run()` corre esse flow directamente. Medido:

```
Agent(ReasoningSpec(strategy="react", timeout=0.001), llm, tools).run("ola")
  Flow compilado tem policy? Policy(..., timeout=0.001, ...)
  provider de 50 ms -> text='resposta lenta'  errors=()
```

O timeout declarado não interrompeu nada. É violação de contrato documentado: `docs/agents.md:53-54`
lista `policy` e `timeout` como campos do `ReasoningSpec` que fazem retry/timeout/confidence, e a
docstring de `react_flow` diz "timeout: Overall timeout in seconds".

Para a app: qualquer limite de tempo por agente declarado em `ReasoningSpec` ou num manifesto
(`strategy.timeout`) não está a ser imposto. Só `BudgetPolicy(max_wall_s=...)` está — e essa é
verificada entre passos, não durante um passo.

**A correcção não é óbvia — e não é "fazer a policy cascatear como o `Scope`".** Essa formulação
esconde uma decisão semântica por tomar:

| Leitura | `timeout=10` significa | `retry` significa | `fallback` significa |
|---|---|---|---|
| **Default por step** (`step.policy or flow.policy`) | 10 s **por step** — um flow de 5 steps pode levar 50 s | cada step tenta outra vez | substitui **um** step |
| **Envelope da execução** | 10 s para o **flow inteiro** | repete o flow todo | substitui o flow inteiro |

A segunda leitura levanta problemas próprios: repetir um flow exige restaurar estado e não repetir
efeitos externos já cometidos. O código actual sugere essa leitura — `as_step()` transforma o flow
inteiro num `Step` com a policy —, mas a documentação sugere a primeira: `docs/agents.md:53`
descreve `ReasoningSpec.policy` como "**Per-step** retry/timeout/confidence" e `:54` descreve
`ReasoningSpec.timeout` como "**Wall-clock** timeout (seconds)". **Dois campos com semânticas
diferentes que hoje aterram no mesmo `Flow(policy=...)`.**

A separação correcta é explicitar os dois eixos, por exemplo
`Flow(..., default_step_policy=Policy(...), timeout=60.0)`, com `ReasoningSpec.policy` a alimentar o
primeiro e `ReasoningSpec.timeout` o segundo. Quem fizer a correcção tem também de **retirar a
policy do wrapper de `as_step()`**, ou ela passa a ser aplicada duas vezes num flow aninhado.

**Bug adicional, independente deste:** quando se passam `policy` e `timeout` ao mesmo tempo, o
timeout é descartado em silêncio. `react_flow` faz `if timeout is not None and flow_policy is None`
(`_react.py:180-182`), portanto uma `policy` explícita engole o `timeout`:

```
só timeout=30        -> flow.policy.timeout=30.0
só policy(retry=2)   -> flow.policy.timeout=None  retries=2
policy + timeout=30  -> flow.policy.timeout=None  retries=2   <- timeout perdido
```

Mesmo depois de G ser corrigido, esta combinação continua a perder o timeout.

**Porque é que passou despercebido:** a suite tem 260 testes verdes em `tests/flow` e
`tests/agents`, e o único teste que toca em `Flow(policy=...)` é `test_properties`
(`tests/flow/test_flow.py:67-74`), que faz `assert flow.policy is p` — testa o getter, nunca o
efeito. O atributo está coberto; o contrato não.

### H · O modo do `Flow` é inferido, e a inferência desarma uma validação de segurança

`Flow` decide o modo por inspecção: `is_dag = any(fs.after for fs in steps)`. Consequências
verificadas:

**1. Um `after=` num step muda a semântica do `when=` de outro.** O mesmo `when` significa "repete
enquanto for verdade" num flow cíclico e "executa uma vez se for verdade" num DAG:

```
só when=            -> is_dag=False   n final=3   (ciclou até à condição falhar)
when= + um after=   -> is_dag=True     n final=1   (when virou guarda de uma passagem)
```

**2. A validação de `max_iterations` desaparece — mas isso *não* é um risco de segurança.**

```
when= sem max_iterations          -> ValueError: "You must set max_iterations to limit the loop."
when= + after= sem max_iterations -> aceite (is_dag=True)
```

Uma versão anterior desta revisão chamou-lhe "guarda que se desarma sozinha". **Estava errado.** O
`Flow` valida a aciclicidade do DAG na construção — `ValueError: Flow contains a cycle in step
dependencies` — e valida referências `after` desconhecidas. Em modo DAG cada nó corre no máximo uma
vez, portanto não há ciclo infinito contra o qual proteger e `max_iterations` não faz falta. A
validação não desapareceu: deixou de ser aplicável.

O problema é de **previsibilidade**, não de segurança: a mesma estrutura de código muda de regime
por causa de um `after=` acrescentado noutro sítio.

**3. `Flow.run()` tem um caminho de excepção não documentado.** Um `when` que levanta é tratado como
skip em modo cíclico (`_executor.py:143-165`, com `try/except`) mas **propaga** em modo DAG
(`_executor.py:338-341`, sem `try/except`):

```
step que levanta            -> FlowResult, erro no trace='step rebentou'
condição que levanta (DAG)  -> ESCAPA: ValueError: condição rebentou
```

Invocar aqui o contrato "erros são dados" seria esticá-lo: esse contrato está associado a *tools* e
à execução de *steps*, e uma `ConditionFn` é código de orquestração, não um `Step` — tratar a sua
excepção como erro de programação é defensável. O problema é a **inconsistência**, e a metade mais
perigosa não é a que levanta: é a cíclica, que converte um bug na condição num skip de aparência
perfeitamente normal, com o flow a terminar "bem".

A regra uniforme que faz sentido:

- **condição falsa** → `step_skipped`, como hoje;
- **condição que levanta** → nunca um skip silencioso. Deve aparecer como `StepTrace.error`, um
  `FlowEvent(type="condition_error", ...)` e terminação do flow, salvo policy explícita em contrário
  — nos dois modos.

Declarar o modo explicitamente (`Flow(..., mode="dag")`) ou separar construtores eliminaria (1) e
(2); (3) precisa da regra acima, independentemente do modo.

## Duas coisas que verifiquei e **não** são lacunas

**`requires_approval` estático não impede "sempre / perguntar / nunca".** O `approval_handler` é por
`ToolGroup`, e a app constrói um grupo por agente. Para uma tool com `requires_approval=True` as
três políticas exprimem-se todas:

```
politica 'sempre'      -> ok=True   ENVIADO para a@b.c          (handler devolve approve())
politica 'perguntar'   -> ok=True   ENVIADO para a@b.c          (handler async espera pela pessoa)
politica 'nunca'       -> ok=False  approval_denied             (handler devolve deny())
sem handler            -> ok=False  approval_denied             (nega por omissão)
```

O problema real é só o **sentido inverso**: uma tool com `requires_approval=False` nunca chega ao
handler (verificado: o handler não é sequer chamado), portanto a app não a consegue tornar *mais*
restritiva pelo `ApprovalGate` standard. Falta um override ergonómico ou um `PermissionGate`
público — mas o grosso do encaixe já existe.

**`inference_limit` não se aplicar a streaming não é um bug.** É o comportamento documentado
(`core/_concurrency.py:43-45`), e por uma razão defensável: um stream pode ficar aberto muito tempo
ou ser abandonado, e seguraria o semáforo. Medido:

```
complete()      com inference_limit(1) -> pico 1
stream_events() com inference_limit(1) -> pico 4
```

Como as estratégias actuais usam `complete()`, continuam protegidas. O problema só aparece no loop
de streaming directo da app ou num futuro `Agent.stream()`. Um `stream_limit(n)` para limitar
sessões activas seria funcionalidade nova, não correcção desta. `docs/concurrency.md` devia ao menos
mencionar a exclusão.

## Resultados negativos — onde procurei e não encontrei

Para não inflacionar a tese da "pipeline duplicada", registo o que testei e está bem:

- **`complete_sync()` corre exactamente os mesmos hooks async que `complete()`**
  (`['abefore', 'aafter']` nos dois). A divergência de middleware é específica do streaming, não do
  eixo sync/async.
- **O middleware do LLM primário corre quando é o fallback a responder.** Testado com um primário a
  levantar `APIError(500)` e um fallback a responder: `abefore`/`aafter` dispararam e as duas
  tentativas ficaram registadas em `response.attempts`.

## Deriva de documentação: o total propaga, a atribuição não

`docs/flow-architecture.md` diz, sobre `Flow.as_step()`: "Cost, usage, confidence propagate up
automatically". Dizer simplesmente que isso é falso seria incorrecto. Medido, com um flow aninhado a
fazer uma chamada LLM medida:

```
meter do pai      -> custo=0.00010500  in_tokens=10  out=5     <- correcto
Result do wrapper -> cost=0.0  usage=0                          <- sem atribuição ao nó
StepTrace.children do wrapper -> ()
trace.flow('inner')           -> None
```

O consumo **propaga para o meter do run**, porque flows aninhados partilham o mesmo `MeterScope`; o
`FlowResult` do pai vê o total certo. O que não existe é **atribuição hierárquica**: o `Result` do
wrapper leva zero (de propósito — `_flow.py` comenta que anotá-lo duplicaria a contagem no trace
cru), e a trace do filho é descartada, pelo que não há forma de dizer "este subagente gastou X".

A doc devia distinguir as duas coisas: *total do run* propaga, *atribuição ao nó* não existe. Para
uma UI que queira mostrar custo por subagente, falta a segunda — que é o mesmo buraco do
`StepTrace.children` (ponto E).

## Nota sobre `Scope`

`Scope` é uma das seis primitivas do Flow e controla **visibilidade de chaves do `State`**:
`include`/`exclude`, `transform` de valores, `enrich` com valores computados; resolução
`FlowStep.scope > Step.scope > Flow.scope`.

Não é fronteira de segurança: não restringe caminhos de ficheiros, não governa argumentos de tools,
não impede acesso directo a recursos externos, não substitui `ResourcePolicy` nem gates, e não cria
isolamento entre irmãos em `Flow.iter()`. Serve para reduzir o contexto entregue a um step — não
para implementar permissões do computador.

## Correcções às assunções da secção 2 do documento da app

**2.3 — `Agent` não expõe `RunConfig`.** `run`/`iter` só aceitam `budget_policy`
(`agents/_agent.py:88-129`). Sinks, redactor e pricer exigem um `MeterScope(RunConfig(...))` à
volta, ou descer a `Flow.run(config=...)`.

**2.6 — o executor chama `definition.fn(**args)`, não `**tool_call.input`.** Os gates podem
reescrever argumentos via `GateModify` antes da chamada (`core/_tools/_executor.py:224`, `:271-274`).
É o ponto certo para injectar raízes permitidas — mas o input que chega à função não é
necessariamente o que o modelo pediu.

**Sobre os manifestos — não há limitação aqui.** Os manifestos permitem configurar cada fase; o que
não permitem é lá pôr objectos runtime, e isso é desenho explícito. `configuring-agents.md:101-102`
define os dois baldes: **deps** para objectos runtime (`planner_llm`, `executor_tools`, …) e
**knobs** para configuração serializável. O manifesto é "the serializable half" — leva prompts por
fase (`strategy.phases.*.system`/`system_file`) e modelos por fase como *dados*, que
`agent_from_manifest(..., llm_factory=)` converte em LLMs. As ferramentas por fase são deps de
código pela mesma razão que o LLM principal também é: um `ToolGroup` com callables, gates e handlers
não é serializável nem deve ser. Para a app: um agente é sempre manifesto + factory, nunca só
manifesto — que é o que `agent_from_manifest` já assume.

## Prioridade que isto sugere para o toolkit

Separei o que é **contrato partido** (uma promessa já feita que não se cumpre) do que é **âmbito**
(funcionalidade que nunca existiu). MCP, scheduler, router, agente-como-ferramenta e workflows
declarativos são âmbito e não entram aqui — a decisão de os fazer é independente. O que está abaixo
é tudo coisa que o toolkit já diz fazer.

### Bugs — corrigir antes de construir por cima

1. **`Flow.policy` ignorada em execução directa** (ponto G). Torna `ReasoningSpec.timeout` e
   `ReasoningSpec.policy` inertes em todas as estratégias. **Não é uma correcção mecânica:** exige
   primeiro decidir se a policy do flow é default por step ou envelope da execução — hoje a doc diz
   uma coisa (`policy` "per-step", `timeout` "wall-clock") e o código sugere outra. Separar os dois
   eixos, corrigir a colisão que descarta o `timeout` quando há `policy`, e retirar a policy do
   wrapper de `as_step()` para não a aplicar duas vezes.
2. **`Flow.iter()` altera a semântica dos DAGs** (ponto B). Observar muda o resultado.
3. **Middleware assíncrono ignorado em streaming** (ponto A). Moderação e memória não correm, e a
   mitigação que `docs/moderation.md:62` recomenda também não.
4. **`run_tools(..., ToolGroup)` perde a governance do grupo.** Delegar em `ToolGroup.async_execute()`
   quando recebe um grupo resolve-o.
5. **Condição que levanta escapa de `Flow.run()` em modo DAG** (ponto H.3), ao contrário do modo
   cíclico e do resto do sistema.

### Arquitectura — a raiz dos quatro primeiros

6. **Um motor de execução único, com eventos, e `run()` a drená-lo** (pontos B, E, F). Elimina a
   duplicação que produziu (2), dá resultado ao iterador e abre caminho aos eventos granulares.
   Ressalva de implementação: um gerador assíncrono simples não emite de dentro de um
   `asyncio.gather` — é preciso uma fila que o consumidor drena.
7. **Preparação/finalização async canónica partilhada por `complete()` e streaming** — a raiz de (3).
8. **Eventos emitidos por quem toma a decisão.** Hoje `execute_step` conhece retries, timeouts e
   fallbacks e o motor de eventos não: `retry`/`fallback`/`timeout` estão no `Literal` e nunca são
   emitidos (ponto E).
9. **Ligar `StepTrace.children`** nos flows filhos, hoje sempre vazio — sem isto não há inspecção
   hierárquica de subagentes.
10. **Separar definições de estado por execução no `ToolGroup`** — `max_calls` acumula entre runs de
    um `Agent` compilado uma vez.
11. **`RunConfig` aceite por `Agent.run`/`iter`** — a fachada expõe hoje menos do que o `Flow` que
    encapsula.

### Dívida de desenho

12. **Modo do `Flow` declarado em vez de inferido** (ponto H) — resolve a semântica dupla do `when=`
    e a validação de `max_iterations` que se desarma sozinha.
13. **Canal `server_tools=`** no `Agent` e nas factories (ponto D), e erro de validação em vez de
    `AttributeError`.
14. **Fronteira entre estado de domínio e estado de execução** — `_meter_scope` vive na camada
    `world` do `State`, o que faz de `to_dict()` algo que pode não ser JSON e não serve de checkpoint.
15. **Superfície pública de gates** (`ToolGate`, `ExecutionContext`, `GateBlock`/`GateModify`/
    `GateDryRun`, factory de `ToolDefinition`, `PermissionGate`) — desbloqueia MCP, agente-como-
    ferramenta, ferramentas dinâmicas e o sentido restritivo das permissões por agente.
16. Classificar as cinco ferramentas de `dangerous` sem metadados (ponto C); `provider` preenchido no
    `OperationRequest`; `docs/concurrency.md` a mencionar a exclusão do streaming; corrigir
    `flow-architecture.md` quanto à propagação de custo em `as_step()`.

Nota de sequência: (1) é independente e barato. (6) e (7) antes de qualquer `Agent.stream`, porque
construir a funcionalidade mais visível da app sobre dois motores divergentes e sobre um caminho que
ignora middleware async seria assentá-la nas duas fundações partidas.

---

# Terceira revisão — varredura do código ainda não lido

**Data:** 12 de setembro de 2026. **Código:** `main` em `48a43ac`.

As rondas anteriores concentraram-se em `Flow`, `Agent`, tools e metering. Esta leu o que faltava —
`core/_sync.py`, `core/_step_engine.py`, `core/_policy.py`, `core/_redaction.py`,
`toolkit/flow/_scope.py`, `core/_tools/_schema.py`, `core/_metering/_scope.py`,
`toolkit/budget/*`, `toolkit/resources/_policy.py`, `toolkit/memory/graph/*` e a conversão de
mensagens dos adapters — e **testou** cada hipótese em vez de a inferir. Todos os números abaixo
saem de código executado.

**Contexto para quem vai planear as correcções:** a suite completa passa —
`2608 passed, 7 skipped` — com todos os achados deste documento presentes no código. Nenhum deles é
apanhado por um teste existente. Cada correcção precisa de um teste **comportamental** novo; a
armadilha a evitar está ilustrada pelo ponto G da revisão anterior, onde o único teste que tocava em
`Flow(policy=...)` verificava o getter (`assert flow.policy is p`) e nunca o efeito.

## Resultados negativos primeiro

Para calibrar: estas áreas foram testadas e **estão bem**. Não precisam de plano.

| Verificado | Resultado |
|---|---|
| Dois `Agent.run()` concorrentes sobre o mesmo `Agent` | Sem contaminação; relatórios de budget separados |
| `complete_sync()` chamado repetidamente | Estável; o cache de cliente por loop aguenta |
| `GraphStore.save()` — 8 gravações concorrentes | Ficheiro sempre válido (escrita atómica) |
| `GraphStore.save()` durante mutação concorrente | Ficheiro válido, snapshot coerente |
| `Trace.to_dict()` → `from_dict()` | Round-trip fiel; `json.dumps` OK no modo por omissão |
| Abandonar um `flow.iter()` a meio | Meter fecha, `has_live_ops`=False, sem ops penduradas |
| Mensagens `system()` nos caminhos principais | Anthropic, Gemini e xAI içam-nas para o parâmetro nativo |
| Nomes de ferramenta duplicados num `ToolGroup` | Avisa e fica a última (comportamento razoável) |
| `core/_metering` e `toolkit/budget` | Desenho cuidado: `Money` exacto, fail-closed em custo desconhecido, revalidação no store sob lock, rejeição de NaN/inf |

O metering é a parte mais bem construída do pacote. Não mexer sem necessidade.

## Achados novos

### I · A memória do trace cresce com o quadrado do número de passos

`execute_step` guarda `input_state=snapshot.to_dict()` em **todos** os `StepTrace`
(`core/_step_engine.py:44`). `to_dict()` faz `dict(layer)` — cópia rasa — mas um loop ReAct cria uma
**lista nova** de mensagens a cada turno (`_react.py:162`, `updated_messages = [*messages, ...]`),
por isso cada trace fica com a sua própria cópia do histórico inteiro.

Medido com um `react_flow` real, 12 turnos com ferramenta e observações de 2 KB:

```
histórico final de mensagens :  26 190 bytes
soma dos input_state do trace: 368 472 bytes
amplificação: 14.1x
```

É O(n²) no número de turnos. Um agente de 50 turnos com 200 KB de contexto produz na ordem de
10 MB de trace por execução, tudo vivo em memória e devolvido dentro do `FlowResult`. Para uma app
que corre agentes longos e guarda execuções, é o problema de escala mais concreto que encontrei.

`TraceMode` não ajuda: só afecta `to_dict()` na serialização, não o que é **guardado**. Faria
sentido um modo de captura (referência, diff, ou nada) decidido na construção do trace.

### J · Há três caminhos por onde uma excepção escapa de `Flow.run()`, não um

A revisão anterior identificou o `when` em modo DAG. Testando, são três — e os outros dois escapam
em **todos** os modos, porque `apply_scope` corre fora do `execute_step`
(`_executor.py:548`, antes de `_run_step_with_scope`):

```
step que levanta                 -> FlowResult, erro no trace   (convertido)
condição when que levanta (DAG)  -> ESCAPA: ValueError
Scope.transform que levanta      -> ESCAPA: ZeroDivisionError
Scope.enrich que levanta         -> ESCAPA: ZeroDivisionError
```

Qualquer callable de orquestração fornecido pelo utilizador — `when`, `transform`, `enrich` — pode
derrubar `flow.run()` com uma excepção crua. A regra uniforme proposta no ponto H.3 tem de cobrir os
três, não só o `when`.

### K · `Scope` não compõe: `enrich` contorna o `exclude`, `transform` corre uma vez por camada

`apply_scope` filtra cada camada e só depois corre o `enrich`, passando-lhe o **snapshot original**
(`_scope.py:57-58`). Logo:

```python
Scope(exclude={"api_key"}, enrich={"vazou": lambda snap: snap["api_key"]})
# o step vê: {'api_key': None, 'vazou': 'sk-SEGREDO'}
```

Medido. `Scope` não é fronteira de segurança (já registado), mas isto significa que as duas metades
da mesma `Scope` não são coerentes entre si.

E `transform` é aplicado **por camada**: uma chave presente em `current`, `operational` e `world`
faz a função correr três vezes (medido: 3 chamadas, `['C','O','W']`). Com uma transform cara ou com
efeitos, é surpresa.

### L · A inferência de schema produz schemas que mentem — dois casos

**1. Uma união multi-tipo torna um parâmetro obrigatório em opcional.**
`_hint_to_json_schema` devolve `is_optional=True` para qualquer `UnionType` com mais de um tipo não-
`None` (`_schema.py:36-48`), não só para `X | None`. Resultado:

```python
@tool
def procura(consulta: int | str) -> str: ...      # sem default: é obrigatório

schema required = []                               # <- 'consulta' não aparece
modelo omite o argumento (legal segundo o schema)
  -> ok=False validation_error: procura() missing 1 required positional argument
```

Uma ferramenta que **não pode ser chamada correctamente** se o modelo seguir o schema que lhe foi
dado. O aviso que é emitido fala só do colapso para `string`, não da perda da obrigatoriedade.

**2. `*args`/`**kwargs` viram propriedades escalares obrigatórias.**

```python
@tool
def flexivel(a: int, *args: int, **kwargs: str) -> str: ...

required = ['a', 'args', 'kwargs']
modelo obedece ao schema ({"a":1,"args":2,"kwargs":"x"})
  -> ok=True, mas a função recebe args=() kwargs={'args': 2, 'kwargs': 'x'}
```

O schema exige dois parâmetros que não existem, e o que chega à função é lixo. Varargs deviam ser
ignorados na inferência (ou rejeitados com erro claro no `@tool`).

### M · Os argumentos das ferramentas não são validados contra o schema — só os nomes

```
soma(a: int, b: int)
  {"a": 1,   "b": 2  } -> ok=True  "3"
  {"a": "1", "b": "2"} -> ok=True  "12"        <- concatenação de strings, resposta errada
  {"a": 1, "b": 2, "c": 9} -> ok=False validation_error
  {"a": 1}                 -> ok=False validation_error
```

Nomes em falta ou a mais dão `validation_error` (é o `TypeError` da chamada Python a ser apanhado).
Tipos não são verificados: o schema diz `integer`, o modelo manda `"1"`, e a ferramenta devolve
silenciosamente a resposta errada. Importa sobretudo com modelos locais e servidores
OpenAI-compatible, que emitem strings com muito mais frequência — exactamente o caso de uso da app.

Para um pipeline que se apresenta como *governed tool execution*, falta o passo de coerção/validação
entre o gate e a chamada.

### N · Um timeout nunca é retentado, mesmo com retries configurados

```
Policy(timeout=0.01, retry=RetryConfig(max_retries=3))  sobre um step de 50 ms
  -> tentativas reais = 1   decisões = ('timeout',)
```

`_run_attempts` faz `break` incondicional no `except TimeoutError` (`_step_engine.py:78-92`). É
defensável por desenho (`on_timeout` é `halt`/`fallback`), mas a interacção com `retry` não está
documentada em lado nenhum e contraria a expectativa natural de quem configura os dois.

### O · Um `Result(error=...)` de domínio é retentado com backoff

```
step que devolve Result(error="utilizador não encontrado"), max_retries=2, base_delay=0.05
  -> 3 chamadas à função, decisões ('retry','retry','halt'), 0.17 s de espera
```

`Result.error` significa ao mesmo tempo "este passo falhou" e "volta a tentar". Isto colide com o
estilo que o próprio toolkit recomenda — *"Toolkit tools return error strings instead of raising, so
agents can keep going"* (AGENTS.md). Um passo que devolve erros de domínio como dados passa a ser
retentado como se fossem falhas transitórias. Faltaria distinguir erro retryable de erro terminal no
`Result` (o `ToolError` já tem `retryable`; o `Result` não).

### P · Um timeout do wrapper síncrono não cancela o trabalho

`_run_sync`, quando já há um loop a correr, lança uma thread daemon e faz `thread.join(timeout=...)`.
Ao esgotar, levanta `TimeoutError` — **mas a thread continua**:

```
TimeoutError: Sync wrapper timed out after 0.05s
logo a seguir     -> coroutine já acabou? False
passados 0.4 s    -> a coroutine continuou e terminou? True
```

Quem chamou recebe um timeout; a chamada ao modelo, ou a ferramenta com efeitos, completa-se à mesma
mais tarde. Sob um meter, a operação liquida depois de o chamador já ter seguido em frente.

Relacionado, em `_stream_sync`: a fila é ilimitada (sem backpressure — um consumidor lento faz o
produtor encher a memória com o stream todo) e, se o consumidor abandonar o gerador a meio,
`drained` fica `False` e a thread **nunca é aguardada**.

### Q · O `MeterScope` vive dentro do `input_state` de cada `StepTrace`

O executor guarda o scope na camada `world` do `State` (`_executor.py:624`), e `execute_step` copia o
snapshot inteiro para o trace. Consequência medida:

```
redacted       json.dumps=OK       api_key='[REDACTED]'   _meter_scope=str (repr)
metadata_only  json.dumps=OK       api_key=None           _meter_scope=None
full_debug     json.dumps=FALHA    api_key='sk-SEGREDO'   _meter_scope=MeterScope
```

O modo por omissão safa-se porque o redactor converte o objecto em `repr`. Mas **`full_debug` produz
um trace que não é serializável** (`TypeError`), e `docs/safety.md` apresenta esse modo como "local
debugging" — que é normalmente escrever para ficheiro. Mais de fundo: um objecto de runtime com um
`threading.Lock` não devia estar no contentor de estado de domínio (é o ponto 11 da lista anterior,
aqui com consequência concreta).

### R · O caminho de batch da Anthropic descarta a mensagem `system()`

Os caminhos principais (`complete`, `stream`, `stream_events`, `count_tokens`) fazem
`effective_system = system if system is not None else msg_system` e içam correctamente a mensagem.
O `batch_submit` não:

```python
_, wire = _messages_to_sdk(messages)      # _anthropic.py:775 — msg_system deitado fora
...
if req_system:                            # só req["system"] é usado
    params["system"] = req_system
```

`batch_submit([{"messages": [system("A"), user("x")]}])` perde silenciosamente o prompt de sistema.

### S · `system=` e mensagem `system()` não se fundem — a mensagem desaparece

Em todos os adapters a regra é `system if system is not None else msg_system`. Com os dois
presentes, o que está nas mensagens **evapora-se sem aviso**:

```
extraído das mensagens : 'REGRA A (da mensagem)'
system='REGRA B'       -> enviado: 'REGRA B'      # REGRA A perdida
```

É relevante para as estratégias: `llm_call` passa sempre `system=system or None`, portanto uma app
que construa a lista de mensagens com um `system()` próprio e configure também `ReasoningSpec.system`
fica só com o segundo. Merge (ou erro explícito) seria mais seguro que precedência silenciosa.

## Como isto se encaixa na lista de prioridades

Nada aqui invalida a ordem anterior. Acrescenta:

**Bugs — juntar aos cinco já listados**

6. **Schema mentiroso em uniões multi-tipo e varargs** (ponto L). Afecta qualquer app que defina
   ferramentas; uma delas produz ferramentas impossíveis de chamar.
7. **Batch da Anthropic perde o `system()`** (ponto R).
8. **`transform`/`enrich` do `Scope` escapam de `Flow.run()`** (ponto J) — a mesma correcção do `when`.

**Arquitectura — juntar**

9. **Política de captura do trace** (ponto I). O trace guarda o estado completo por passo; é
   O(n²) e é o limite prático de execuções longas. Decidir o que capturar (referência, diff,
   metadata) em vez de copiar tudo.
10. **Coerção/validação de argumentos no executor de ferramentas** (ponto M), entre o gate e a
    chamada.
11. **Tirar o `_meter_scope` do `State`** (ponto Q) — resolve ao mesmo tempo o trace não
    serializável em `full_debug` e o checkpoint do ponto 14 anterior.

**Dívida de desenho — juntar**

12. **Semântica de retry**: timeout nunca retentado (N) e `Result(error=)` sempre retentado (O).
    As duas decisões são defensáveis mas nenhuma está documentada, e a segunda colide com o estilo
    de erros-como-dados que o toolkit recomenda.
13. **Precedência de `system`** (ponto S): fundir ou recusar, não descartar.
14. **`_run_sync`/`_stream_sync`**: um timeout que não cancela (P), uma fila sem backpressure e uma
    thread não aguardada quando o consumidor desiste.
15. **Coerência do `Scope`** (ponto K): `enrich` a ver o snapshot original e `transform` a correr
    uma vez por camada.

## Nota de método para quem pegar nisto

Três das conclusões desta revisão contradizem versões anteriores dela própria — `dangerous`,
manifestos, `max_iterations`. Em todos os casos a causa foi a mesma: inferir a partir do código sem
confrontar com o modelo declarado, ou sem executar. As conclusões que se aguentaram foram as que
tinham um teste por trás. Os scripts que produziram os números deste documento são triviais de
reconstruir a partir dos excertos citados; vale a pena reproduzi-los antes de agir sobre qualquer
ponto, sobretudo os que implicam mudar contratos públicos.
