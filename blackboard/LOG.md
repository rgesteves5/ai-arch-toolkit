# Diário

## 2026-09-13

- Revisão `docs/internal/agentes-app-toolkit-review.md` verificada por execução; plano escrito em
  `docs/internal/toolkit-fix-plan.md`.
- Revisão cruzada de outro modelo integrada no plano (secção 0): D1a, D3 e D7 revistos; D8 e D9 novos.
- Confirmados por execução N8, N9 e N10 (ver `FINDINGS.md`).
- Baseline em `main` @ `48a43ac`: 2608 passed, 7 skipped (11 s); pyright 0 erros; ruff limpo.
- Blackboard criado. Próximo: workers A–E nas worktrees (F01, F04/F06/F07/F17, F11/F12, F15,
  F05/F10); coordenador começa F02, depois F09 e F14.
- F02, F09, F14 feitos pelo coordenador (2650 passed). F08 implementado no checkout principal: motor
  único (gerador + task por step), `FlowExecution`/`AgentExecution`, eventos de policy, children,
  irmãos preservados, span em flows aninhados (2691 passed, testes do motor estáveis em 3 repetições).
- Workers A, B, C e E terminaram. Diffs revistos e aplicados sem conflitos: F01, F04, F05, F06, F07,
  F10, F11, F12, F17. Suite integrada: 2812 passed, 7 skipped; ruff e pyright limpos. Um teste do F12
  dependia da ordem do cache do `typing` em 3.13 e foi corrigido.
- Workers E e C não conseguiram escrever no checkout principal (a harness recusou); registos
  transcritos pelo coordenador. Achados novos: gates não encadeiam `GateModify` (worker-B) e C2/C3
  entram no F13; C1 vira F19; `run_tools` com `KeyError` depois de efeitos vira F20.
- Falta: F03, F13, F15 (worker-D em curso), F16, F19, F20, F18, e a documentação do F08.
- F15 (worker-D) aplicado: wrappers síncronos cancelam no timeout e têm backpressure (2867 passed).
- F03 e F13 feitos pelo coordenador; ver os registos nas fichas (D8 mudou quatro testes de metering).
- F20 feito: `run_tools` verifica os nomes antes de executar. F19 feito: `_tool_to_sdk` recorre a
  `parameters_json_schema` quando o SDK recusa o schema.
- F16 feito: `Flow(trace_capture=)`, default `"keys"`; ligado a `ReasoningSpec` e manifesto
  (2901 passed, 7 skipped).
- F18 feito: docs (`tools.md`, `api.md`, `memory.md`, docstring do `RateLimitMiddleware`) e
  `CHANGELOG` `[Unreleased]` com F01–F20.
- Plano completo. Verificação final: 2901 passed, 7 skipped; ruff e format limpos; pyright 0 erros.
  Nada commitado. Worktrees dos workers A–E ficam em `.claude/worktrees/agent-*` para o dono remover.
- F21 feita a pedido do dono: aprovações no nanope (agente configurável, solvers BBEH e o gestor do
  `research_center`, regressão do F11 que nenhum teste apanhava), conteúdo de sistema com partes de
  texto nos quatro adaptadores e `cache()` como texto fora do Anthropic, envelope de erros no
  `_stream_sync`, `Any`/`object` com schema vazio (verificado live em Anthropic, OpenAI e Gemini; xAI
  sem créditos), helpers de middleware removidos. 2927 passed, 7 skipped; ruff e pyright limpos.
- Revisão final a pedido do dono: três revisores independentes sobre o diff; testes ao vivo em
  Anthropic, OpenAI e Gemini (ver F21); dois bugs encontrados ao vivo e corrigidos (`$defs` de
  pydantic aninhados; Gemini sem `parts`); teste `live_api` antigo actualizado; 7 testes `live_api`
  novos. xAI fica pendente (créditos).
- A pedido do dono: commits por área (providers, tools, flow, docs, internal) verificados um a um em
  worktrees temporárias, push para `main` (`48a43ac..5e29759`). Worktrees dos agentes removidas e
  os seus branches apagados (0 commits próprios; conteúdo confirmado em `main`).
- F22: os três revisores confirmaram bugs; corrigidos em `2411831` (tools), `7559d5e` (llm) e
  `cc83cb9` (flow), com testes que falham antes. 2956 passed; `live_api` Anthropic/OpenAI 11 passed.
  Achados não corrigidos em `FINDINGS.md`. xAI continua pendente (créditos).

## 2026-09-13 · fornecedor Meta

- A pedido do dono: investigação da Meta Model API (sem SDK da Meta; `openai` ou `anthropic`) e
  integração "da melhor forma", com autorização para chamadas ao vivo com `MODEL_API_KEY`.
- M01 feito: `MetaProvider` sobre a Responses API do SDK `openai` (D11–D14), com reenvio do
  raciocínio encriptado. Revisão adversarial: 3 bugs de streaming confirmados e corrigidos.
  3020 passed, 20 skipped; ruff e pyright limpos; `live_api` Meta 6 passed; matriz de probes 6/6.
- Achados fora do âmbito em `FINDINGS.md` (chave OpenAI em `base_url` remoto, `strict` no adaptador
  OpenAI, `APIConnectionError` sem retry, `pytest-timeout` ausente).
- A pedido do dono: commits `9588f31` (código, testes, scripts, CI), `7967287` (docs) e o registo no
  blackboard, com push para `main`.
- O dono não quer testes reais no CI do GitHub (não quer gastar dinheiro no CI): removido
  `.github/workflows/integration.yml` (corria `pytest -m live_api` todos os dias às 06:17 UTC e a
  pedido) e o job de testes do `ci.yml` passa a `-m "not live_api"`. AGENTS.md e CONTRIBUTING.md
  dizem que os testes `live_api` só correm localmente. O secret `MODEL_API_KEY` no GitHub deixa de
  ser preciso.
- Auditoria do `CHANGELOG`: tudo o que foi feito nesta sessão estava lá; faltava a correcção do
  preço do `grok-4.6` (`8b7299a`, anterior à sessão) e a linha do CI dizia que os testes ao vivo
  corriam num cron diário. Ambas corrigidas.
- A pedido do dono (2026-09-15): commits `c679ac3` (CI sem testes reais), `44885a3` (changelog do
  `grok-4.6`) e o registo no blackboard, com push para `main`.

## 2026-09-15 · achados em aberto

- A pedido do dono: F23 corrige os oito achados que ficaram em `FINDINGS.md` (ver a ficha). Dois
  verificados ao vivo: o OpenAI recusava `output_schema` Pydantic (corrigido e confirmado) e o Gemini
  aceita declarações mistas (sem alteração). 3045 passed, 22 skipped; ruff, pyright e lock limpos.
- Commits `b504998` (tools), `dfaca5c` (providers), `19a3905` (pytest-timeout), `4509bc9` (docs) e o
  registo no blackboard; `b504998` e `dfaca5c` verificados em worktrees temporárias (3026 e 3045
  passed). Push para `main`.

## 2026-09-15 · frente de capacidades em falta

- A pedido do dono: abrir uma frente para tudo o que a revisão da app Agentes pediu ao toolkit e o
  plano de correcção deixou de fora (§4, itens 3 e 10). Nove fichas: C01 `Agent.stream()`, C02 tools
  dinâmicas, C03 cliente MCP, C04 checkpoint e retoma, C05 server tools, C06 catálogo de modelos, C07
  escrita tipada com `FilesystemPolicy`, C08 pesquisa web local, C09 `FlowSpec` (blocked até a app
  validar as formas).
- Sete agentes, só leitura, escreveram as fichas com evidência `ficheiro:linha`, reproduções sem rede
  e a documentação oficial (SDK `mcp`, Anthropic, OpenAI, Gemini, Meta, xAI, Brave, Tavily). Nenhum
  ficheiro de código, teste ou doc mudou.
- Revisão do coordenador: números de exemplo passam a ser atribuídos ao aplicar; nota no C05 sobre os
  tipos actuais de server tools da Anthropic; ordem por vagas e decisões que mudam contrato no quadro.
- Achados: o coordenador voltou a correr as reproduções e confirmou no código e na doc; vinte entradas
  em `FINDINGS.md`, onze sem tarefa. Os mais graves: `thinking=True` no Anthropic recusado nos modelos
  actuais, `csv_read` sem aprovação, budget envenenado por erro do adaptador.
- A seguir: o dono fixa as decisões da vaga 1 (C02, C06, C07, C08) e decide se os achados sem tarefa
  abrem uma frente de correcção antes. Nada commitado.

## 2026-09-17 · causas dos achados e plano de robustez

- O dono perguntou se algum achado era impeditivo. Um é, e é mais largo do que o registado a 15: toda
  a chamada LLM falhada fica com custo desconhecido e, sob `max_cost`, o retry, o fallback e o resto
  do run são negados (também o tecto por step). Registado em `FINDINGS.md` com as reproduções.
- A pedido do dono ("corrigir a causa, não o sintoma"): cinco agentes de leitura investigaram a
  facturação de falhas nos fornecedores (documentação oficial), a validação de pedidos contra os
  tipos dos SDKs, as fases e o mapeamento de erros dos cinco adaptadores, os resultados de tools
  paralelas por fornecedor e os invariantes das ~130 tools. Achados novos em `FINDINGS.md`; os que o
  coordenador verificou estão marcados.
- Um dos agentes fez, por engano, um pedido ao endpoint real da Google com a chave literal "dummy"
  (400, sem credenciais, sem custo). O script não foi guardado.
- Resultado: `docs/internal/hardening-plan.md` — seis causas, correcção estrutural de cada uma, como
  se garante, ordem por vagas e onze decisões por fixar (R1–R11). Protótipos reutilizáveis guardados
  em `blackboard/prototypes/2026-09-hardening/` (o scratchpad da sessão foi limpo entre dias e os
  scripts de 15 de setembro perderam-se).
- A seguir: o dono fixa R1–R11; depois abre-se a frente com fichas. Nada commitado.
- Revisão do plano no mesmo dia, a pedido do dono ("remendos ou correcções estruturais?"): cinco
  mecanismos da primeira versão eram remendos e foram substituídos por quatro costuras redesenhadas
  (modelo de erros tipado, contrato de três fases como caminho único, pipeline de tentativa única,
  gramática única de ids de modelo), com regras de desenho e orçamentos de qualidade no CI. Linha de
  base medida: 35 funções em `core/` com complexidade acima de 10, 34 acima de 60 linhas.
- Dependências: ensaio numa cópia descartável com `uv lock --upgrade` (95 pacotes; `anthropic` 0.116 →
  1.6 e `openai` 2.45 → 3.14, ambos para `httpx2`). Tudo verde (3045 passed, pyright 0, um `RUF036`),
  o que confirma que a suite não vê o SDK real. A actualização passa a ser a vaga 0a. O repositório
  não foi alterado. Decisões agora R1–R14.
- Duas revisões externas (outros modelos), verificadas pelo coordenador a pedido do dono: métricas e
  duplicações reproduzem-se todas; uma chegou sozinha ao achado impeditivo. Quatro bugs novos em
  `FINDINGS.md` (fallback que altera o `LLM` do utilizador, validação que aceita `None` e não vê
  elementos de listas, `from_mapping` que descarta em silêncio, imutabilidade só à superfície). O plano
  adopta a unificação dos caminhos de `_run_dag` e ganha as secções 9 (configuração estrita e
  imutabilidade, R15) e 10 (dívida de manutenção confirmada, para frente própria).
- O `uv.lock` do checkout principal aparece actualizado às 20:17 e o `.venv` já tem `anthropic` 1.6.0 e
  `openai` 3.14.1. Não foi o coordenador (o ensaio correu numa cópia em scratch); fica como está, à
  espera de indicação do dono. `pyproject.toml`, hooks e CI continuam por actualizar.

## 2026-09-17 · dependências actualizadas

- A pedido do dono ("fecha a actualização de dependências"). O `uv.lock` já estava actualizado (95
  pacotes; `anthropic` 1.6.0 e `openai` 3.14.1, ambos sobre `httpx2`; `google-genai` 2.24.0; `xai-sdk`
  1.19.0; `ruff` 0.16.8; `pyright` 1.1.414). Fechado agora: mínimos verdadeiros e tectos na versão
  maior seguinte no `pyproject.toml` (`anthropic>=1.0,<2`, `openai>=3.0,<4`, `google-genai>=2.0,<3`,
  `xai-sdk>=1.7,<2`, `pyyaml>=6.0.2`, `tiktoken>=0.11`); o extra `dev` passa a instalar `all`, para
  cada mínimo ficar declarado uma vez; job `floors` no CI (`--resolution lowest-direct`); actions
  (`checkout@v7`, `setup-uv@v10.1.0`, `setup-python@v7`); hooks (`ruff-pre-commit` v0.16.8,
  `pre-commit-hooks` v6.0.0); `RUF036` corrigido; README, CONTRIBUTING e CHANGELOG.
- Os mínimos antigos eram falsos: `uv sync --resolution lowest-direct` falhava logo no `pyyaml` 6.0,
  que não compila em Python 3.13.
- Verificação: 3045 passed, ruff e formatação limpos, pyright 0, `uv lock --check` OK, na versão
  actual; nos mínimos (cópia em scratch): 3045 passed e pyright actual 0. Ensaios sem rede pelos SDKs
  reais contra servidor em loopback, nas duas pontas: 73 cenários de falha sem erros de forma de API e
  caminho de sucesso (`smoke_success.py`) em Anthropic, OpenAI, Meta e Gemini. Por fazer: o CI não
  correu (só no push) e nenhuma chamada ao vivo foi feita com os SDKs novos. Nada commitado.

## 2026-09-17 · contratos pequenos e correcções locais

- A pedido do dono: F24 feita (grupo de tools vazio, fallback que alterava o `LLM` do utilizador,
  `null` em parâmetros que não admitem `None`, `ReasoningSpec.from_mapping` estrito). 3070 passed;
  ruff, formatação e pyright limpos. Nota do problema conhecido do Gemini com tools em
  `docs/model-compatibility.md` e no `AGENTS.md`.
- Decisões do dono registadas: D15 (dependências), D16 (modelo sem preço não corre), D17 (limites
  por omissão para qualquer tool), D18 (ordem dos adaptadores: OpenAI, xAI, Gemini, Meta, Anthropic),
  D19 (`null`).
- F25 aberta: nove correcções locais independentes, prontas para um agente. Os refactors do plano de
  robustez continuam sem fichas. Nada commitado.

## 2026-09-17 · frente de robustez em três fases

- A pedido do dono: commits por área do que havia (`7c9973b` dependências, `7bdfad3` F24, `5a210ba`
  nota do Gemini, `538fcd0` plano, achados, protótipos e quadro), sem push.
- Tudo o que ficou combinado (correcções, refactors e dívida de manutenção) dividido em três fases
  para agentes com contexto limpo: `R01` núcleo de chamadas (começa pela F25), `R02` fornecedores,
  `R03` tools, motor e manutenção. Regras comuns em `R00-rules.md`. D20 fixa as recomendações R1–R15
  como decididas, com os acertos do dono. A frente C fica em espera até ao fim da R03.

## 2026-09-18 · R01 (Codex)

Leitura obrigatória concluída; R01 e F25 reclamadas. Sem operações git, fornecedores ou leitura do `.env`. Começa a F25, com vermelho primeiro e verificações completas por item.

- F25 itens 1–8 aplicados com vermelho primeiro (item 8 só prova documental por leitura). 31 casos novos. Todas as oito verificações R00: lint/formato/pyright/lock limpos; final 3084 passed, 17 failed, 22 deselected. Bloqueio: import csv_read no nanope, árvore proibida pela R00; pedida autorização de um único import. F25.9 bloqueada: preço oficial exige rede externa, proibida. R01 blocked antes dos instrumentos; relatório completo na ficha. D21 regista a escolha de transporte IP pendente. Nada commitado ou chamado ao vivo.

- O dono autorizou a migração do import nanope e a consulta gratuita de páginas oficiais de preços. Import migrado; retomada a R01 pela ordem. As restantes proibições mantêm-se.

- R01 concluída (passos 0–5), F25 também done. Gate final: 3884 passed (+814 sobre 3070), 22 live_api deselected, zero xfail; lint/formato/pyright/lock limpos. Matriz 720/720, conservação/replay, AST e qualidade; 14 ensaios SDK oficiais exclusivamente no servidor falso dos protótipos em 127.0.0.1. Fachada 1561 → 532, fachada+pipeline+ciclo de vida −467 linhas, fonte tocada −263; dívida ruff 172 → 161, C901 core 35 → 29/toolkit 64 → 64. Dois vermelhos da revisão final corrigiram texto não consumido na resposta parcial síncrona. D22 regista construção pura vs futura preparação R02; D23 fixa posse/fecho de streams e vaga inicial. Achado novo: anthropic 1.6.0 rejeita temperature por omissão, reprodução em FINDINGS para R02. Relatório, testes corrigidos, gates e commits propostos na ficha. Sem .env, fornecedores pagos, dependências novas ou operações git. Próximo: dono revê diff e faz commits; R02/R03 continuam todo.

## 2026-09-18 · R01, continuação (Claude)

O Codex ficou sem créditos depois de fechar a ficha, o BOARD e o LOG, e antes da mensagem final.
Verificação independente: gate R00 repetido, critérios de aceitação medidos e protótipos originais
`meter/` corridos. Três acabamentos, com nota de desenho na ficha: o servidor falso dos ensaios SDK
passa de `blackboard/prototypes/` para `tests/integration/fakeserver.py`; entrada `Changed` para o
`fallback_on` por omissão; uma tentativa abandonada antes de começar levanta `StreamAbandoned` em vez
de rebentar num `assert` (vermelho primeiro). Gate: 3886 passed, 22 deselected; ruff, formato,
pyright e lock limpos. Achado novo para decisão do dono: um tecto por step sem `BudgetPolicy` continua
a falhar depois de um 5xx (`meter/step_cap.py`), por contrato da ficha; proposta para a R02. Por
verificar: job `floors` do CI com os ensaios SDK novos. Próximo: o dono revê o diff e faz os commits;
a R02 começa depois, com contexto limpo. Antes de publicar `main`: com o `anthropic` 1.6.0 do lock,
`claude-haiku-4-5` e `claude-sonnet-4-6` levantam `TypeError` (`temperature`) antes de enviar; só os
modelos da lista sem `temperature` (Opus 4.7+, família 5) funcionam. A D18 põe a Anthropic em último
na R02: o dono decide se isto passa à frente.

- A pedido do dono, a R01 ficou commitada em `main` em quatro commits: `a16e3f9` tools, `f95c81f`
  F25, `5a0ab3b` núcleo, e o registo. Não publicada. Cada árvore intermédia passa o gate sozinha.

## 2026-09-18 · `temperature` da Anthropic antes do push (Claude, a pedido do dono)

O `anthropic` 1.6.0 do lock tirou `temperature`, `top_p` e `top_k` das assinaturas, e todas as
chamadas a modelos que ainda os aceitam (Haiku 4.5, Sonnet e Opus 4.5–4.6) davam `TypeError` antes
de enviar. Passam a ir em `extra_body`, decididos por uma só função; a regra de quando a
`temperature` cai não mudou. Vermelho primeiro: sete ensaios do SDK real em loopback, sem o contorno
da fixture, e o teste unitário que afirmava `temperature` como argumento do SDK, corrigido. Gate: 3886
passed; ruff, formato, pyright e lock limpos. Verificação ao vivo pendente (sem créditos): comando no
BOARD. Documentação do SDK 1.x confirmada no guia de migração oficial incluído na skill `claude-api`
(passo 6, "Removed request parameters").

## 2026-09-18 · R02 (Claude)

R01 publicada em `main` (`b3dae3f`); a R02 começa com a leitura obrigatória da R00. Linha de base:
3886 passed, 22 deselected. Sem commits, fornecedores (só páginas públicas de documentação) nem
`.env`.

- Passo 1 (costura D) feito: gramática única de ids em `core/_model_id.py`; preços por id exacto ou
  snapshot, com `aliases` e `match="prefix"` explícito; tokenizer e encaminhamento pela gramática;
  D16 — sob um scope, um modelo sem preço levanta `UnpricedModelError` antes de abrir a operação;
  `scripts/audit_models.py` para o dono. Oito testes que afirmavam a herança por prefixo ou o
  contrato antigo da D16 corrigidos e listados na ficha. Gate: 3954 passed; ruff, formato, pyright
  e lock limpos.
- Passo 2 (costura B) feito: `BaseProvider` em seis peças com o algoritmo na base; `prepare` antes
  de qualquer admissão; paridade por construção (uma montagem); os cinco adaptadores reorganizados
  (−730 linhas, 44 blocos `except` de SDK → 0, dívida de complexidade 161 → 137). Um duplo de teste
  (`tests/fake_provider.py`) substitui os duplos duck-typed e os providers `AsyncMock` em 33
  ficheiros, migrados por três agentes com regras escritas e revistos. Gate: 3963 passed; ruff,
  formato, pyright e lock limpos; os 14 ensaios SDK em loopback verdes.

- Passo 3 · OpenAI feito: marcador de despacho comum na base (D24); pedido tipado pelo SDK;
  perfis pela gramática; `max_completion_tokens` no host oficial; server tools recusadas; batch
  pelo `prepare`; transporte em loopback com o SDK real (13 casos). Gate: 3998 passed.
- Passo 3 · xAI feito: o `prepare` constrói o pedido com o `chat.create` do SDK (local), a partir
  de um `TypedDict` verificado pelo pyright; perfis pelos esforços documentados de cada modelo
  (D25); `required_tool` para o `tool_choice` com nome; mapeador pelo `google.rpc.Code`; o usage de
  um stream sem usage deixa de valer zero; servidor gRPC falso em loopback (11 casos de
  transporte); `xai-sdk>=1.18`. Comum: a guarda de import dos SDKs passa a gestor de contexto e
  saem os 18 `# noqa: E402` (sete tinham entrado nesta fase, contra a R00). Dívida 134 → 129.
  Gate: 4100 passed; ruff, formato, pyright e lock limpos.
- Passo 3 · Gemini feito: `GenerateContentConfig` tipada num `TypedDict` verificado pelo pyright;
  perfis pelos níveis e orçamentos documentados; resultados de um turno num só `Content` com o
  `id` do Gemini; pilha fixa em `httpx` por um transporte próprio (acaba o reenvio escondido do
  `aiohttp`, provado em loopback: 2 pedidos → 1); server tools com config ou desconhecidas
  recusadas; entrega documentada (400 e 500 `unbilled`); transporte em loopback (15 casos).
  Dívida 129 → 124. Gate: 4178 passed; ruff, formato, pyright e lock limpos.
- Passo 3 · Meta feito: pedido tipado com três desvios listados (cada um com a prova do M01); um
  mapeador; códigos pela tabela documentada da Meta (código nulo ou desconhecido → `ResponseError`);
  o usage de um `response.failed` vai até ao meter (D26: `ProviderError.usage`,
  `MeterOperation.fail(..., usage=, cost=)`); esforços por modelo; `timeout` sem `httpx`. Desvio
  assumido: o adaptador foi escrito antes dos seus testes; o vermelho foi visto contra uma
  reconstrução do passo 2. Dívida 124 → 123. Gate: 4215 passed; ruff, formato, pyright e lock limpos.
- Passo 3 · Anthropic feito: thinking e esforço pelas tabelas documentadas (D27); resultados de um
  turno numa só mensagem; turno reenviado do `_raw` com as assinaturas; `tool_choice` forçado
  recusado no Fable 5.1/Mythos 5.1; server tools com `name`; pedidos falhados `unbilled`; erro
  dentro do stream pelo tipo; batch pelo `prepare`. Sai o `network_error` e a lista
  `_PENDING_ADAPTERS` (vazia). Os cinco adaptadores: 3752 → 3280 linhas (o pacote
  `core/_providers/`: 4301 → 4018). Dívida 123 → 121. Gate:
  4305 passed; ruff, formato, pyright e lock limpos.
- Passo 4 feito: rede do fio em `tests/` (`wire_contract.py` e a fixture `autouse` `wire_log`):
  cada pedido que um `prepare` constrói na suite é validado contra o contrato do SDK (Stainless
  estrito com escalares estritos, Meta com os três desvios preenchidos, `extra_body` da Anthropic
  pelos campos de amostragem, Gemini pelo pydantic e pelo conversor offline, xAI pelos enums do
  `proto`). 490 pedidos em 402 testes; só os três testes que enviam de propósito um pedido mau
  falharam, e declaram-no com `wire_contract(tolerate=...)`. 36 canários. Gate: 4341 passed;
  ruff, formato, pyright e lock limpos.
- Passo 5: os 20 testes `live_api` (quatro por fornecedor) ficam preparados; não correram.
- Passo 6 feito: `docs/llm.md`, `pricing.md`, `model-compatibility.md`, `safety.md` e `AGENTS.md`
  com as regras da R02; corrigidos também `tools.md`, `getting-started.md`,
  `framework-overview.md`, o quadro do `README.md`, o "Adding a provider" do `CONTRIBUTING.md` e o
  exemplo 25 (server tool no OpenAI, agora recusada). A nota "Known issue" do Gemini fica até à
  prova ao vivo, reescrita. Duas entradas do `CHANGELOG` que a R02 contradizia foram acertadas.
  Achado novo: o `thinking_effort` do OpenAI sem `thinking` perde-se. Gate: 4341 passed.
- R02 concluída, sem commits. 3886 → 4341 passed (42 live_api deselected); dívida 161 → 121;
  `src/` −249 linhas (Python −123, TOML −126); `# noqa` em `src/` 14 → 4. Relatório final, comandos
  ao vivo e três commits propostos no fim da ficha. Próximo: o dono revê, commita e corre as
  verificações ao vivo; a R03 continua `todo`.
- R02 commitada a pedido do dono, sem push: `cf2aa43` código e testes (passos 1 a 3), `a8c3eea` rede
  do fio (passo 4, com os três marcadores do transporte) e o registo (docs, CHANGELOG, blackboard).
  Cada árvore passa o gate sozinha (índice exportado): 4305, 4341 e 4341 passed. A decisão sobre o
  tecto de uma falha sem `BudgetPolicy` ficou em "Por fazer" no BOARD, com pistas para o dono
  investigar. A seguir: a R03, nesta sessão, depois de o dono a compactar.

## 2026-09-18 · R03 (Claude)

- Início: base `main` @ `4a1fa4e`, 4341 passed. Ficha e BOARD em `in-progress`.
- Passo 1 feito (tools). `toolkit/tools/_http.py` é a única porta para a rede (regra por AST, sem
  lista de legado): HTTPS para os hosts de cada módulo, redirects só no mesmo host, leitura limitada
  em bytes e em tempo, segmentos citados um a um, throttle partilhado entre threads, uma só forma de
  erro, e a leitura de cada resposta dentro de uma fronteira (`parse=`, D31). Os 47 sítios
  migraram: quatro módulos de referência e o `_web` por mim, 31 por cinco agentes em paralelo, com
  revisão. Executor: `max_output_chars` e `timeout_s` para qualquer tool, tecto do grupo que só
  aperta (D29), tools síncronas numa thread daemon nos dois caminhos (D30). Facto medido: um regex
  catastrófico ou um inteiro enorme prendem o GIL e nenhum timeout os pára; `math_eval` e
  `regex_search` recusam-nos à entrada (D32). `ip_lookup` passa ao ipwho.is em HTTPS (D28).
  Invariantes executáveis: 739 casos, sem `xfail`. Gate: 5208 passed, 42 deselected; ruff,
  formatação, pyright e lock limpos. `toolkit/tools` −270 linhas; dívida 121 → 73. Quatro achados
  novos em `FINDINGS.md` (o risco polinomial do regex para o dono decidir).
- Passo 2 feito (motor). Um só executor de vagas (o sequencial corre vagas de um), um só sítio onde
  um step acaba (`_ended`), e as vagas lêem o snapshot da vaga em vez de um `fork()` por irmão (D33:
  mudança visível, escrita no `CHANGELOG`). Verificador de gramática de eventos com 16 cenários,
  reutilizado pelos testes do motor; apanhou o stream e o trace a contarem ordens diferentes numa
  vaga paralela. `_run_attempts` com um só bloco de fallback, fixado antes por 20 caminhos. Gate:
  5245 passed. Dívida 73 → 66.
- Passo 3 feito (imutabilidade). `StateSnapshot` documentado como vista só de leitura (camadas
  copiadas, valores partilhados) e testado igual nos dois modos; `ReasoningSpec.knobs` e
  `llm_kwargs` congelados à entrada (`MappingProxyType`), quebra visível no `CHANGELOG`. O
  `from_mapping` estrito já estava feito. Gate: 5250 passed.
- Passo 4.1 feito (stores de grafo). Uma base genérica partilhada, `GraphFacade[N]`, com nós,
  índice por tipo, arestas, lotes e a forma do JSON guardado; o `Graph` e o `GraphStore` herdam-na
  e o `GraphStore` fica só com o que é da memória. Desvio à nota (compor o `Graph`): o pyright
  recusa um `MemoryBackend` como `GraphBackend`, e um `cast` escondia-o. API pública igual (por AST
  e `inspect`). Janelas duplicadas 60 → 0, −110 linhas, dívida 66 → 61.
- Passo 4.2 feito (opções das factories). `FlowOptions` declara uma vez as quatro opções que cada
  factory passa ao `Flow`; as factories recebem-nas em `**options`, os builders lêem-nas do spec
  num só sítio, e o `nested()` diz o que um ReAct interno herda (só o `trace_capture`). A
  `budget_policy` fica como opção do `Flow` (D34). Chamadas iguais, verificadas pelo pyright. Gate:
  5295 passed. −70 linhas.
- Passo 4.3 feito (chaves de estado). As quatro chaves que as estratégias e o runner partilham
  vivem em `flows/_keys.py` (regra por AST: 66 grafias → 0). O `extract_text` deixou a cadeia de
  quatro chaves: lê `answer`, ou o valor do último step (D35), e a `Response` vem da mesma fonte.
  As dez estratégias deixam `answer` e `response` também quando esgotam. Medido antes: um Reflexion
  que passava com resposta vazia respondia o score, e um ReAct que acabava em tools respondia a
  lista dos resultados. Gate: 5316 passed.
- Passo 4.4 feito (manifestos).
  - Cada manifesto tem uma declaração (`toolkit/_shape.py` e um `_manifest_shape.py` por
    manifesto). O loader verifica cada ficheiro com ela, e o JSON Schema empacotado gera-se dela;
    o de prompts era mais largo do que o loader, e o de agentes é novo.
  - A sonda diferencial (1323 e 1166 casos) mostrou só as mudanças de propósito.
  - Achado e corrigido: os leitores do manifesto de agentes partiam com `null` que o loader
    aceitava, e um `trace_capture` lista dava `TypeError`.
  - Testes de conformidade com o `jsonschema` do ambiente de desenvolvimento, e do loader contra
    a declaração.
  - Gate: 5380 passed. Dívida 61 → 53. Saldo +441 linhas de Python (desvio registado).
- Passo 4.5 feito (`_eval_expr`).
  - Duas tabelas de despacho (18 expressões, 7 instruções), um método por nó, e um teste por cada
    nó do `ast` em execução, permitido ou recusado.
  - Saíram o `hasattr` e o `getattr` do avaliador.
  - Cinco semânticas erradas em silêncio corrigidas (desempacotar dicionários, `**kwargs`, curto
    circuito, `del` de atributo, `**=` sem guarda).
  - Achado na infra de testes: o pytest deixava cair as fixtures de `tests/toolkit` numa linha de
    comando intercalada, e os testes das tools corriam com rede e throttle a sério. As fixtures
    foram para o conftest da raiz, com um canário.
  - Gate: 5460 passed. Dívida 53 → 48.
- Passo 4.6 feito (ReAct interno). O `run_react` devolve a resposta, a `Response` e se algum step
  falhou. Os sete sítios de seis estratégias usam-no, com uma regra por AST: só `_react` constrói
  um ReAct. Gate: 5465 passed. Dívida 48 → 46.
- Passo 4.7 feito: o levantamento do `__all__` de topo está na ficha (204 nomes, 81 usados nas docs
  e nos exemplos, 25 nunca citados). Há duas incoerências: `Agent`/`ReasoningSpec` não estão no
  topo, e falta lá a nona factory. Nenhum export mudou.
- Passo 5 feito, e a R03 concluída.
  - `AGENTS.md` e `CONTRIBUTING.md` sem o `urllib` e com as regras novas.
  - "Upgrade notes" no `CHANGELOG`, com as quebras das três fases.
  - Linha de base 121 → 46.
  - Relatório final na ficha, com dois commits propostos. A árvore do primeiro foi verificada
    sozinha, numa cópia: 5465 passed, pyright 0.
  - O BOARD devolve a vez à frente C, que começa depois dos commits da R03.
  - Gate final: 5465 passed, 42 deselected; ruff, formatação, pyright e lock limpos. Sem commits,
    pushes, fornecedores nem `.env`.
- A pedido do dono: os dois commits propostos (`6ceb516` código e testes, e o registo), e o push de
  `main`, que publicou também os três commits da R02.
