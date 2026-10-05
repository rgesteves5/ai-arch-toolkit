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

## 2026-09-28 · Seis tools gratuitas ao vivo (Claude, a pedido do dono)

- Um agente disse que arXiv, REST Countries, UniProt, PDB, Eurostat e Free Dictionary não
  funcionavam. Verificação ao vivo, chamada a chamada: o arXiv funciona; falhavam `country_info`,
  `uniprot_search`, `pdb_search`, `eurostat_dataset_search` e `define_word`.
- Corrigidos, por commitar:
  - `country_info` passa a ler o Wikidata (pesquisa + uma consulta SPARQL), sem chave: a REST
    Countries desligou as v1–v4 e a v5 pede chave. O dono pode acrescentar a v5 se arranjar uma.
  - `uniprot_search` pede `/uniprotkb/search` (fecha o achado da UniProt em `FINDINGS.md`).
  - `pdb_search` usa o serviço `full_text` (o `text` sem atributo dava 400).
  - `eurostat_dataset_search` lê só os stubs do catálogo (1,5 MB em vez de 20 MB, acima do tecto
    de 10 MB); o período e as observações ficam no `eurostat_dataset`.
- Por resolver: `define_word` depende do dictionaryapi.dev, que responde 522 da Cloudflare ao fim
  de ~19 s (o nosso prazo é 10 s). Não é código nosso; o `wiktionary_entry` cobre as definições.
- Gate: 5549 passed, 42 skipped; ruff, formatação e pyright limpos.

## 2026-09-29 · Commits pendentes e contrato das tools (Claude, a pedido do dono)

- Por fazer push, nada. Por commitar: as correcções de 28/09 (checkout principal) e quatro
  correcções de estratégias de 24/09 numa worktree (`exciting-germain-a41255`), nunca aplicadas nem
  registadas: ToT, notas do avaliador em `tot`/`lats`, `ACCEPT` do `generate_review`, `$N` do
  `llm_compiler`.
- As duas passaram o gate num export do `main`; commitadas como `d892cfd` (tools) e `1433016`
  (estratégias, com o `CHANGELOG` fundido à mão) e enviadas. Gate: 5566 passed, 42 skipped; ruff e
  pyright limpos. A worktree antiga fica como estava.
- Levantamento das 132 tools, depois de duas conversas do ai-network em que o agente não chegou ao
  que as páginas tinham: `docs/internal/tools-contract-plan.md`. Decisões D37–D41: falhas tipadas,
  porta HTTP que valida, janela sem becos sem saída, MediaWiki pelo HTML renderizado, fusão das tools
  repetidas sem aliases.
- Em paralelo, numa sessão própria: `mediawiki_page` e `wiktionary_entry` deixam de devolver sucesso
  vazio para páginas que não existem (ainda com strings de erro; converte-se no passo 3 do plano).
- Fornecedor, fora do plano: a GPT-6 Luna nunca raciocina num agente com tools, porque o adaptador
  OpenAI só usa a Chat Completions, que desde o GPT-5.4 só aceita tools com `reasoning_effort` a
  `none`. A porta para a Responses API tem custos relatados por terceiros: latência 2–3× (medição de
  2025 no Azure), histórico frágil com o raciocínio cifrado (400 quando um item de raciocínio perde o
  seu par), raciocínio preso ao modelo e à organização (o `_replayable_output` da Meta verifica o
  tipo do `_raw`, não o modelo) e schemas das funções normalizados para strict. Continua fora de
  âmbito.
- Próximo: abrir a frente do contrato das tools no `BOARD.md`, quando o dono disser.

## 2026-09-29 · Erros que as fontes mandam com sucesso (Claude, a pedido do dono)

- A sessão paralela do `missingtitle`, revista e commitada a pedido do dono. A correcção está na
  porta: o `Api` ganha `body_error`, uma função que lê a resposta antes do `parse=` e devolve o erro
  que a fonte lá pôs; um corpo que não é JSON chega-lhe como texto (os primeiros 2000 caracteres).
  Ainda com strings de erro.
- Leitores: `mediawiki_error` em todos os `api.php` da MediaWiki (um teste verifica), a `message` do
  World Bank, o `ERROR` do ESearch, o `error` do Internet Archive, o `remark` "runtime error" do
  Overpass e a linha de texto do GDELT. A Wikipedia diz o `invalidreason` de um título inválido.
- Anexo C do plano: resolvidas as confirmadas das tools MediaWiki, `wikipedia_*`, `wikidata_search`,
  `country_info`, `world_bank_*` e `overpass_*`, e as suspeitas das duas tools do GDELT, do
  `pubmed_search` e do `internet_archive_search`. A forma do `gdelt_timeline` confirmou-se (o
  cliente `gdeltdoc` lê `timeline[0].data` e os seus testes correm contra a API): nenhuma resposta
  dava pontos; corrigido. Continuam: a troca do `wikipedia_related` para a pesquisa quando a página
  não existe (documentada), o `hacker_news`, o 204 da RCSB e as outras suspeitas.
- Para o D38: o `body_error` só vê o corpo de um 2xx. A função do D38 lê também o estado e os
  cabeçalhos, e o D37 troca a string por um erro tipado; os leitores passam para lá.
- Gate: 5642 passed, 42 deselected; ruff e pyright limpos. Ao vivo: as correcções da MediaWiki, da
  Wikipedia, do World Bank, do PubMed, do Internet Archive, do Overpass e o erro de texto do GDELT
  (`Invalid/Unsupported Country.`); o timeline do GDELT não, porque a API respondeu 429 a todas as
  tentativas.

## 2026-09-30 · Frente do contrato das tools aberta (Claude, a pedido do dono)

- Apagadas, a pedido do dono, as 33 branches locais cujo remoto já não existe, depois de verificar
  que estavam todas integradas no `main` e que nenhuma estava numa worktree.
- `main` avançou para `c259e0b` (a sessão paralela do `missingtitle`: leitores de erros num 200,
  ainda com strings). As fichas partem daí.
- Frente T aberta no `BOARD.md`: regras em `tasks/T00-rules.md` e fichas T01 a T09. As costuras vêm
  primeiro (falhas tipadas, porta HTTP, janela, limites); depois a invariante de contrato com a lista
  de dívida; depois a família wiki, que fica como modelo; e, por fim, os quatro grupos de módulos em
  paralelo.
- D42 refina a D37: a excepção chama-se `ToolFailure` e leva o `ToolError` público, que já existia
  como registo; os argumentos inválidos usam o `validation_error` do executor.
- Com a frente C: a C07 e a C08 esperam pela T01 e pela T03; a C02 e a T04a aplicam-se em série; o
  resto segue em paralelo.
- Próximo: o dono escolhe quem pega na vaga 1 (T01, T03, T04a).

## 2026-09-30 · O resto do anexo C (Claude, a pedido do dono)

- O que ficou do anexo C depois de `c259e0b`, com cada suspeita vista ao vivo antes de mexer.
- Porta: o `body_error` lê também o corpo de um 4xx ou 5xx e o seu texto toma o lugar da razão do
  estado. Um pedido cuja fonte responde "nada encontrado" com `204 No Content` ou corpo vazio
  declara-o (`allow_empty=True`, por chamada, como a T02 prevê) e lê essa resposta como vazia; sem
  a declaração continua a ser erro de leitura. Leitores novos: `label` do Eurostat, entrada de erro
  do arXiv (também num 200) e `messages` da UniProt. O do GDELT recusa HTML, que agora também lhe
  chega.
- Por tool: UniProt inactivo (fundido, separado ou apagado) diz o destino; Open Library segue um
  registo fundido (até três) e diz que um apagado foi apagado; `wikidata_entity` segue um QID
  fundido; `eonet_event` explica o 500 que o EONET dá a um ID que não conhece e recusa uma resposta
  sem evento; `hacker_news` numera por posição e diz quais não carregou; `wikipedia_related` diz
  porque pesquisa.
- Visto ao vivo: 204 da RCSB, 404 e 413 do Eurostat com `{"error": [...]}`, 400 do arXiv com a
  entrada de erro, `entryType: Inactive` da UniProt, `/type/redirect` e `/type/delete` da Open
  Library, o redirect que o Special:EntityData segue, `null` do HN. Não se confirmou: o evento em
  branco do EONET (um ID desconhecido dá 500) e a entrada de erro do arXiv com 200 (hoje vem com
  400); ficam as guardas.
- Fica para a frente T: os `status_messages={404: ...}` de 12 módulos e o 404 por endpoint (T02),
  o texto de erro em HTML do Overpass (400) e os erros tipados (T01). As fichas T02 e T05 a T09
  dizem o que já está feito.

## 2026-09-30 · Vaga 1 da frente T: T04a e T03 num branch (Claude, a pedido do dono)

- O dono deu a vaga 1, se não interferisse com a sessão paralela e num branch novo. Essa sessão
  (a do `missingtitle`, a trabalhar na worktree `beautiful-feistel-20d1c5`) mexe na porta
  `_http.py` (o 204 e o texto dos erros) e em nove módulos (arXiv, Eurostat, GDELT, UniProt, PDB,
  EONET, Wikidata, Open Library, Hacker News): a T01 esperaria por ela. A T03 e a T04a não tocam
  nesses ficheiros.
- Branch `feat/tools-contract-wave1`, numa worktree própria criada a partir de `origin/main`
  (`b9d7524`), sem upstream, para o checkout principal não mudar de branch.
- T04a (`aa54ba1`): `Range` no schema e no validador; o `infer_schema` e o `validate_arguments`
  repartidos em funções pequenas, e três entradas saem da baseline de complexidade.
- T03: `toolkit/tools/_window.py` (texto, `find`, listas), `line_cut` no core e o `_bounded` do
  executor no mesmo vocabulário. Nenhuma tool a usa ainda.
- Revisão antes do push, a pedido do dono:
  - o `find` fundia passagens sem tecto, e um termo frequente podia devolver o texto todo;
  - os termos com acentos saíam escapados no rodapé;
  - um termo vazio contava ocorrências vazias.

  Os três foram corrigidos, com testes que falham na versão anterior. Ficou documentado que o
  validador aplica os `minimum`/`maximum` de um `schema=` escrito à mão.
- Gate no branch: 5691 passed, 42 deselected; ruff, formatação e pyright limpos.
- O `main` recebeu entretanto o resto do anexo C (`6a34668`, entrada acima); o branch fez merge
  dele para o PR #71 não ficar em conflito (só o `LOG.md` chocou).
- Próximo: o dono faz o merge do PR #71; a T01 já não tem a sessão paralela à frente.

## 2026-09-30 · Worktrees e branches locais arrumados (Claude, a pedido do dono)

- Com o PR #71 em `main` (`84c0ee2`) não ficou nenhum PR aberto nem branch remoto além de `main`.
- `exciting-germain-a41255` (branch `claude/elegant-euclid-2e5020`, de 2026-09-24): as alterações
  por commitar eram, linha a linha, o `1433016`. Só o `tests/agents/flows/test_common.py` (o
  `parse_score` sozinho) tinha ficado de fora; entrou em `main` no `81c3699`.
- `beautiful-feistel-20d1c5` (a sessão do `missingtitle`) estava limpa, com tudo em `main`.
- O dono removeu as duas worktrees e apagou os três branches `claude/*`; o `BOARD.md` deixa de
  pedir para não tocar na primeira.
- Gate em `main` com o teste novo: 5744 passed, 42 skipped.

## 2026-09-30 · Docs contra o código (Claude, a pedido do dono)

- O dono pediu todos os docs actualizados contra o código real, uma nova revisão, e só depois
  commit e push.
- Quatro passagens de agentes com sondas offline (FakeProvider, `prepare()` dos adaptadores): a 1.ª
  corrigiu a deriva conhecida e o resto de cada página; a 2.ª reviu em modo adversarial as mudanças
  e as páginas; a 3.ª e a 4.ª só factos, a 4.ª nas páginas onde a 3.ª ainda corrigira erros. A 4.ª
  ainda achou casos-limite (o `count_tokens` do Gemini com `system`, `tuple[int, ...]`, decisões
  `continue` que o trace não regista), já corrigidos.
- Nos docs públicos, entre outros: os tectos de tokens e custo são brandos por omissão;
  `@tool(schema=)` é por parâmetro; só `react` e `completion` mandam partes multimodais; como o
  `Flow` escolhe o modo; os defaults do `LLM` e o que o batch salta; de onde se importam `Agent` e
  `ReasoningSpec`; a instalação por git; os 34 modelos do inventário; as contagens das tools;
  exemplos que não corriam. Entrada no `CHANGELOG`.
- Internos e blackboard: índice de `docs/internal/`, estado do desenho de prompts e caminhos do
  briefing; T03 e T04a done; as fichas T06–T09 sem o que o `6a34668` fechou; notas nas fichas C;
  "Por fazer" e o plano de robustez actualizados.
- Os defeitos do código encontrados pelo caminho estão em `FINDINGS.md` (2026-09-30).
- Uma sonda de um agente da 3.ª passagem saiu para a Anthropic com uma chave falsa (`APIError`, sem
  custo); a 4.ª correu com sockets bloqueados.
- Gate: 5744 passed, 42 skipped; `mkdocs build --strict` com validação de links e âncoras limpo.

## 2026-10-01 · GPT-6.1 Sol e frente O: OpenAI pela Responses (Claude, a pedido do dono)

- O catálogo da OpenAI tem agora como modelos de topo o `gpt-6-astra`, o `gpt-6.1-sol` e o
  `gpt-6-luna`. O `gpt-6.1-sol` não estava registado e recebia as regras da geração actual, que
  supõem um `none`: o sampling e as tool calls saíam sem mudança e o `thinking_effort="none"` era
  aceite. Segundo a página do modelo, não tem `none` nem `minimal`, raciocina a `medium` por
  omissão, e a Chat Completions "is supported without tool calling".
- Registado como o Astra: perfil `_SOL_6_1` no `_openai.py`, preços (os do GPT-6 Sol, com a cache a
  5% da entrada), inventário de probes sem tools, docs, `AGENTS.md` e `CHANGELOG`. Os testes
  (`TestAlwaysReasoning` parametrizado com os dois modelos e `test_gpt61_sol_prices`) falharam antes
  pela razão certa. Gate: 5763 passed, 42 skipped; ruff, formatação e pyright limpos. Ao vivo
  ainda não correu: o comando está no "Por fazer".
- O dono aceitou a recomendação e pediu a frente. Ficaram a D43 e as fichas O01 a O04: uma sonda
  ao vivo antes de tudo, um núcleo Responses tirado do `_meta.py`, o OpenAI no host oficial pela
  Responses, e no fim os docs e a verificação.
- A D43 responde aos custos que o LOG de 2026-09-28 registava:
  - a latência mede-a a O01;
  - os pares de raciocínio já os trata o código da Meta;
  - o raciocínio preso à família resolve-se no reenvio, que passa a verificar fornecedor e família
    (O02);
  - o `strict` vai explícito a `false`.
- Commitado e publicado em `main` a pedido do dono: `e18fcb5` (o `gpt-6.1-sol`) e o registo da
  frente O.
- Próximo: a O02 pode começar; a O01 precisa do script e de o dono o correr.

## 2026-10-02 · O01: a sonda da Responses (Claude, a pedido do dono)

- O dono aceitou a licença do Xcode, que bloqueava o `git`, e pediu que o Claude escrevesse e
  corresse a sonda. As chamadas pagas à OpenAI ficaram autorizadas, com tecto.
- `scripts/probe_openai_responses.py` com 8 testes sem rede. Duas corridas exploratórias
  corrigiram o script, e a final é a `openai-responses-20261002T022338Z`. Custo total: $0.03.
- A latência da Responses não fica atrás: os p50 são iguais ou melhores, e a cache rende o mesmo.
  A porta da D43 passa.
- Os factos que fixam o pedido da O03 estão na ficha da O01:
  - o raciocínio vem cifrado sem `include`;
  - sem `strict`, a OpenAI reescreve o schema em modo strict;
  - `frequency_penalty` e `presence_penalty` dão 500 ao fim de ~90 s;
  - as regras de sampling são as da Chat Completions;
  - a OpenAI tolera reenvios que a Meta recusa.
- Encontrado e corrigido: a Chat Completions recusa o `max` em todos os GPT-6, embora as páginas
  o listem. Verificado ao vivo nos quatro, com o teste a falhar antes; ficou no `CHANGELOG`
  (Fixed), no `AGENTS.md` e nos docs.
- Commitado e publicado em `main` a pedido do dono: `7d72003` (a correcção do `max`) e o commit da
  sonda, com as notas e o blackboard.
- Próximo: a O02, num agente com contexto limpo.

## 2026-10-02 · Frente O concluída (Claude, a pedido do dono)

- O dono pediu só a frente O, até ao fim, com os bugs encontrados corrigidos pelo caminho.
- O02 (`afdf4a6`): o núcleo da Responses API sai do `_meta.py` para o `_responses.py`. O reenvio
  passa a ser só para o mesmo fornecedor e família, e os itens vão com os nomes do fio.
- D44 e O03 (`d97273d`): o host oficial da OpenAI vai pela Responses, e os outros hosts pela
  Chat Completions (`_openai_compatible.py`), com o mesmo pedido de antes.
- O04:
  - a verificação ao vivo achou a tabela de regras errada (`temperature` recusada pelos modelos
    que raciocinam sem esforço enviado);
  - esforços medidos modelo a modelo, `thinking_effort` sozinho (D45) e batch à tarifa de batch
    (`17e0938`);
  - docs (`2e73faa`).
- Ao vivo: 77 de 77 probes em 13 modelos OpenAI, os testes `live_api -k openai` (6 de 6, três
  vezes), um turno reconstruído e um batch em `/v1/responses` lido à tarifa de batch.
- Gate: 6086 passed, 42 skipped; ruff, formatação e pyright limpos.
- Por publicar: os commits desde `c1cfdfb` estão só no `main` local, à espera da revisão do dono.

## 2026-10-03 · Frente I aberta: geração de imagens (Claude, a pedido do dono)

- O dono quer remover as limitações que impedem o ai-network de usar o toolkit, a começar pelas
  imagens geradas (a G-36 do ai-network; a E06-09 espera por isto).
- Investigação:
  - o código: o `LLM` não chama modelos de imagem, o Gemini deita fora as `inline_data`, e o
    núcleo da Responses ignora os `image_generation_call`;
  - os SDKs instalados já trazem tudo: `images.generate`/`edit`, `image_config`, `image.sample`;
  - a documentação oficial dos cinco fornecedores, com as fontes na D46.
- D46 e D47: o dono aceitou as recomendações.
  - O `LLM.generate_image()` devolve a `Response` com `images` e corre pelo mesmo `Execution` do
    `complete`.
  - Os parâmetros portáveis são `aspect_ratio` + `resolution`, mais `quality`, `n`, `images` e
    `output_format`, validados por modelo.
  - Uma só frente: primeiro a sonda e o `generate_image`, depois as imagens no turno.
- Fichas I01 a I05.
- No `BOARD.md`:
  - a frente O passou a "anterior" (publicada, `be64062`);
  - as decisões da frente C começam agora na D48;
  - a linha "Modelos novos ao vivo" foi actualizada com a corrida de 2026-10-02.
- Próximo: a I01 (paga, menos de $1), quando o dono a autorizar. A I02 pode começar já.

## 2026-10-03 · Frente I concluída (Claude, a pedido do dono)

- O dono autorizou as chamadas pagas e pediu a I01 a I05 por ordem, com os bugs corrigidos pelo
  caminho.
- I01: a sonda (`scripts/probe_images.py`) fixou o `usage` da Images API, o `tool_usage.image_gen`
  da tool, a edição sem estado (o item dá 404; uma `input_image` funciona) e os tamanhos. O
  Gemini e o xAI ficaram sem resposta: faturação e créditos.
- I02: `LLM.generate_image()`, `Response.images`, os contadores e as tarifas de imagem, e a
  reserva por imagem. Bug corrigido: o tamanho de um pedido com imagens contava os bytes como texto.
- I03: os quatro adaptadores, com a Images API partilhada pelo OpenAI e pela Meta
  (`_openai_images.py`). Bugs corrigidos: o `data:` URL no Gemini; a edição da Meta (`image[]`);
  o custo zero do `muse-image` num `complete`.
- I04: a `image_generation(model=...)`, o evento `image`, o reenvio por fornecedor e o custo da
  imagem no turno. Bug corrigido: `partial_images` sem stream dá 400.
- I05: os docs, o `AGENTS.md`, o exemplo 48 e o `CHANGELOG`.
- Gate: 6217 passed, 42 skipped. Ao vivo, cerca de $0.45.
- Commitado e publicado em `main` a pedido do dono: `07c16c7` (a sonda), `f8d25a7` (o código e
  os testes, cuja árvore passa o gate sozinha), `23c4ed7` (os docs) e `9edbe66` (o blackboard).
- Por fazer: o Gemini e o xAI ao vivo, depois da faturação e dos créditos.

## 2026-10-03 · Frente A aberta, A01 feita (Claude, a pedido do dono)

- O dono pediu a próxima lacuna do toolkit para o ai-network. Pela ordem do briefing dele (D-54),
  é o grupo 1, a segurança: G-16 e G-19, as duas confirmadas abertas no `104af8f`.
- D48: o toolkit não lê endereço nenhum do ambiente. Os endereços oficiais estão em
  `OWN_BASE_URLS`, e o Gemini fica na Developer API.
- A01:
  - os quatro adaptadores e o `OpenAIModerator` passam o endereço oficial ao SDK;
  - o adaptador compatível deixa de mandar os cabeçalhos da conta OpenAI a outros hosts;
  - o `Redactor` apaga `xai-…`, `gsk_…` e `AIza…`.
- Dois testes que afirmavam o bug foram corrigidos (listados na ficha).
- Gate: 6232 passed, 42 skipped.
- Próximo: o grupo 2 do briefing, o custo (G-20, G-29).

## 2026-10-04 · A02 feita: o custo (Claude, a pedido do dono)

- O dono escolheu as duas recomendações: toda a falha tem tecto, com ou sem budget (D49), e um
  preço pode ter data de fim (D50).
- G-29: o pior caso passou para o core (`_worst_case.py`) e o meter usa-o sempre. Sai o
  `FailureBoundController`. Um passo com `max_cost` passa depois de um retry com êxito: era o
  achado de 2026-09-18, pendente no BOARD. Os testes que afirmavam o contrato antigo (150 casos da
  matriz e 14 outros) foram corrigidos e estão listados na ficha.
- G-20:
  - `ModelPricing.until`/`then`, e o registo lê o preço do dia;
  - as promoções do `gpt-5.6-sol` (até 2026-11-21) e dos Gemini 3.8, 3.7 e 3.6 Flash (até
    2026-12-31) mudam sozinhas;
  - preços do DeepSeek, do Mistral e do Poolside, das páginas oficiais;
  - os testes leem os preços num dia fixo.
- Gate: 6256 passed, 42 skipped.
- Próximo: o grupo 3 do briefing, a correcção.

## 2026-10-04 · A03 feita: a correcção (Claude, a pedido do dono)

- A A02 foi commitada e publicada (`4bb8cc0`, `565fca4`).
- O grupo 3 do briefing, verificado no `main`, tinha seis lacunas abertas; a G-15 já tinha
  fechado na frente O.
- Resolvidas:
  - G-21: um nome repetido num `ToolGroup` é um erro; uma tool embrulhada corre o invólucro;
    o bloqueio de tools perigosas fala para a pessoa;
  - G-23: os ids das chamadas de ferramenta nascem num sítio só (`named_calls`, no `_answer`);
  - G-17: os clientes dos SDKs nascem no primeiro uso, dentro do loop. O `prepare()` do xAI
    deixou de mexer no cliente;
  - G-18: um pedido longo da Anthropic vai por stream quando o SDK recusa mandá-lo sem;
  - G-26: `run_command(cwd=...)`;
  - G-27: o `search_files` e o `list_directory` ficam dentro da pasta.
- Uma revisão independente encontrou cinco problemas, e quatro foram corrigidos com testes:
  - o `close()` depois de chamadas sync (uma regressão);
  - os streams sync do xAI (a G-17 estava só em parte);
  - o limite do `list_directory`;
  - o `OpenAIModerator`.

  O quinto, nomes repetidos fora do `ToolGroup`, ficou em `FINDINGS.md`.
- Gate: 6298 passed, 42 skipped.
- Próximo: o grupo 4 do briefing, as seis fontes que falham (G-30).

## 2026-10-04 · A04 feita: as fontes que falham (Claude, a pedido do dono)

- A A03 foi commitada e publicada (`87c7e35`, `0619e1a`).
- Uma chamada a sério por fonte, gratuita:
  - responderam o arXiv, os países, o UniProt, o PDB, o Eurostat e o `define_word`;
  - o Eurostat falha no Python do ai-network, porque o ficheiro de autoridades dele não tem a
    raiz `GlobalSign Root R46`;
  - o GDELT e o Semantic Scholar dão 429.
- O dono escolheu as três recomendações:
  - D51: o TLS das tools verifica com as autoridades do sistema quando o extra `truststore` está
    instalado;
  - D52: a chave opcional de uma tool vem do ambiente; a primeira é a
    `SEMANTIC_SCHOLAR_API_KEY`;
  - D53: um 429 deixa o host em espera pelo `Retry-After` ou pelo tempo declarado (GDELT 60 s), e
    o ritmo conta do fim do pedido.
- O User-Agent passou a ter a versão real e o repositório.
- Provado ao vivo: o Eurostat responde no Python do ai-network com o `truststore`.
- Gate: 6317 passed, 42 skipped.
- Próximo: o grupo 5 do briefing, o streaming dentro das estratégias (G-22).

## 2026-10-05 · A05 feita: o streaming dentro das estratégias (Claude, a pedido do dono)

- A A04 foi commitada e publicada (`331786d`, `ff6e8ba`).
- G-22, parte 1: o `step_end` traz o `StepTrace`, e todo o passo que começa acaba, também os
  cortados por um timeout, por um budget negado ou por uma excepção do motor.
- G-22, parte 2 (D54): o dono escolheu o canal no LLM e os tokens sempre no `iter()`. Cada
  `llm.complete` de um passo de um flow iterado vai em stream e chega como `llm_event`; o `run()`
  não muda.
- Uma revisão independente encontrou quatro defeitos, todos corrigidos com testes:
  - eventos de chamadas órfãs depois do passo, e uma falha no `iter_sync` com o loop fechado;
  - eventos em fila perdidos num corte;
  - o `inference_limit` que deixava de limitar;
  - docs.

  Corrigido também um bug antigo do `aclose()` logo a seguir ao `flow_start`.
- Gate: 6333 passed, 42 skipped.
- Próximo: o grupo 6 do briefing, as exportações (G-14, G-24).

## 2026-10-05 · A06 feita: as exportações (Claude, a pedido do dono)

- A A05 foi commitada e publicada (`31bf368`, `6bc7038`).
- G-14: o encaminhamento de modelos é público (`resolve_provider_name`, `MODEL_PREFIXES`,
  `MODEL_IDS`, `is_local_url`).
- G-24: o `NetworkXBackend` é público, e o do core é genérico no tipo de nó, por isso
  `GraphStore(NetworkXBackend())` passa no pyright.
- À parte: o `virtualenv` dos três alertas altos do Dependabot.
- Gate: 6352 passed, 42 skipped.
- Próximo: o grupo 7 do briefing, a pesquisa na web (G-13).

## 2026-10-05 · A07 feita: a pesquisa na web (Claude, a pedido do dono)

- A A06 foi commitada e publicada (`dffcfe9`, `3ebbc6a`), com o `virtualenv` dos alertas do
  Dependabot à parte (`834f735`).
- O dono escolheu as duas tools e o preço na tabela:
  - D55: o `brave_search` e o `tavily_search`, com a chave do ambiente;
  - D56: uma secção `[tools]` na tabela de preços. O meter cobra as unidades que o serviço
    cobrou, que a porta HTTP regista.
- Sem chaves no `.env`, a verificação ao vivo fica para o dono (o comando está na ficha).
- Gate: 6396 passed, 42 skipped.
- Próximo: o grupo 8 do briefing, o tecto partilhado (G-28).

## 2026-10-05 · A08 feita: o tecto partilhado (Claude, a pedido do dono)

- A A07 foi commitada e publicada (`67dc63e`, `86f2033`). Os alertas do Dependabot estão
  fechados.
- D57: o `SharedMeter` do core e o `SharedBudget` do toolkit. As execuções ligadas por
  `RunConfig(shared=...)` são admitidas e acertadas contra ele sob um lock seu, e cada operação
  reserva lá o seu pior caso. Assim, juntas não passam do tecto.
- Provado com execuções em tasks e em threads.
- Gate: 6407 passed, 42 skipped.
- Próximo: o grupo 9 do briefing, os workflows (G-32, G-33).

## 2026-10-05 · A09 feita: os workflows (Claude, a pedido do dono)

- A A08 foi commitada e publicada (`8de0d02`, `cf8df2f`).
- D58:
  - G-32: sob um meter, cada passo corre num span seu, e o `StepTrace.metered` diz o que lá se
    mediu, também num passo cortado. As corridas aninhadas têm também um span seu. Os spans são
    públicos;
  - G-33: `FlowStep(after_any=..., after_optional=...)` e o `StepTrace.blocked_by`.
- Ao abrir um span por passo, apareceu uma falha que já existia com o `max_cost`: uma operação
  que começava depois de o seu span fechar levantava `ValueError`. Os ids dos spans são agora
  caminhos desde a raiz, e a operação conta no antepassado aberto mais próximo.
- Uma revisão independente não encontrou erros de correcção; os seus sete reparos menores estão
  tratados (a ficha lista-os).
- Gate: 6441 passed, 42 skipped.
- Próximo: o grupo 10 do briefing, os manifestos importados (G-34).

## 2026-10-05 · A10 feita: os manifestos importados (Claude, a pedido do dono)

- A A09 foi commitada e publicada (`6bea5be`, `e199cc8`).
- D59: os manifestos de agente e os codecs dos recursos (prompts e conhecimento) lêem o YAML, o
  JSON e o TOML por `toolkit/_safe_data.py`:
  - os aliases do YAML podem acrescentar até 10 000 nós e 1 000 000 de caracteres, ou o tamanho
    do documento;
  - nada passa de 100 níveis, nem um override com o seu caminho;
  - um manifesto que vários herdam ou incluem lê-se uma vez.
- A revisão independente apanhou o que a primeira versão deixava passar: um alias para um texto
  longo (139 KB davam 2 GB) e as heranças multiplicadas (349 525 leituras para dez ficheiros).
  Os sete reparos estão tratados (a ficha lista-os).
- Gate: 6477 passed, 42 skipped.
- Publicada (`34da992`). O dono pediu para não fazer o grupo 12 do briefing (os argumentos das
  ferramentas a chegar, G-37): a frente pára até o dono dizer o que se segue. A G-36 (o grupo 11)
  fechou na frente I.
