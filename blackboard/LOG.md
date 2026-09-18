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
