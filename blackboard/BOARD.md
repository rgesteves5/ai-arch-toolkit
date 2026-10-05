# Quadro

## Frente activa: as lacunas que o ai-network contorna

- **Estado:** aberta em 2026-10-03, a pedido do dono, pela ordem do briefing do ai-network
  (`ai-network/board/toolkit-brief.md`, D-54 dele). A A01, o grupo 1 (a segurança), está feita
  a 2026-10-03 e publicada (`002642c`). A A02, o grupo 2 (o custo), está feita a 2026-10-04 e
  publicada (`4bb8cc0`). A A03, o grupo 3 (a correcção), está feita a 2026-10-04 e
  publicada (`87c7e35`); a G-15 já tinha fechado na frente O. A A04, o grupo 4 (as fontes que falham, G-30), está feita a 2026-10-04 e
  publicada (`331786d`). A A05, o grupo 5 (o streaming dentro das estratégias, G-22), está feita a
  2026-10-05 e publicada (`31bf368`). A A06, o grupo 6 (as exportações), está feita a 2026-10-05 e publicada
  (`dffcfe9`).
  A A07, o grupo 7 (a pesquisa na web, G-13), está feita a 2026-10-05 e
  publicada (`67dc63e`). A A08, o grupo 8 (o tecto partilhado, G-28), está feita a 2026-10-05 e
  publicada (`8de0d02`). A A09, o grupo 9 (os workflows, G-32 e G-33), está feita a
  2026-10-05 e publicada (`6bea5be`). A A10, o grupo 10 (os manifestos importados, G-34), está feita a
  2026-10-05 e publicada (`34da992`). A G-36, o grupo 11, fechou na frente I. Depois de pedir uma
  pausa, o dono pediu para terminar a frente: as A11 a A14 (os grupos 12 a 15: G-37, G-38, G-39,
  G-40) estão feitas a 2026-10-05 e publicadas (`1e1e223`). **Frente A concluída** com elas: o briefing não
  tem mais grupos (a G-25 fica de fora, pelo próprio briefing).
- **Origem:** o briefing de 29/09, onde a app lista o que contorna no toolkit. Cada lacuna diz o
  que falta, a evidência e quando fica feita. A G-36 (imagens) já fechou na frente I.
- **Decisões:** D48 (nenhum endereço lido do ambiente), D49 (toda a falha tem tecto), D50 (um
  preço pode ter data de fim), D51 (o TLS das tools com as autoridades do sistema), D52 (a chave
  opcional de uma tool vem do ambiente), D53 (um 429 fecha o host durante a espera), D54 (as chamadas ao LLM de um flow
  iterado correm em stream), D55 (as tools de pesquisa Brave e Tavily), D56 (o preço das tools pagas
  na tabela), D57 (o tecto partilhado), D58 (o gasto de cada passo, os spans públicos e as
  dependências fracas), D59 (os aliases do YAML, o aninhamento e as heranças com limite), D60 (os
  argumentos das ferramentas a chegar), D61 (a reserva de uma imagem pela qualidade e pelo
  tamanho).
- **Fora do âmbito:** o ai-network. Não se edita daqui; o dono leva-lhe a nota de cada grupo
  fechado.

| ID | Tarefa | Dono | Estado | Depende de |
|---|---|---|---|---|
| A01 | Segurança: nenhum endereço do ambiente (G-16) e as chaves `xai-`, `gsk_` e `AIza` no `Redactor` (G-19) | Claude | done | nada |
| A02 | Custo: toda a falha tem tecto (G-29), preços com data e os que faltam (G-20) | Claude | done | nada |
| A03 | Correcção: tools com nome repetido e embrulhadas (G-21), ids das chamadas (G-23), o cliente xAI sem loop (G-17), pedidos longos da Anthropic (G-18), `cwd` no `run_command` (G-26), ligações no `search_files` (G-27) | Claude | done | nada |
| A04 | As fontes que falham (G-30): o TLS com as autoridades do sistema (Eurostat), a chave do Semantic Scholar, a espera do GDELT depois de um 429, e o User-Agent | Claude | done | nada |
| A05 | O streaming dentro das estratégias (G-22): o `StepTrace` no fim de cada passo, todo o passo que começa acaba, e os tokens das chamadas ao LLM num flow iterado | Claude | done | nada |
| A06 | As exportações: o encaminhamento de modelos e a regra de loopback (G-14), o backend de memória público e com o tipo certo (G-24) | Claude | done | nada |
| A07 | A pesquisa na web (G-13): `brave_search` e `tavily_search`, com a chave do ambiente e o custo no meter pela tabela de preços | Claude | done | nada |
| A08 | O tecto partilhado (G-28): um `SharedMeter`/`SharedBudget` que várias execuções gastam ao mesmo tempo, sob um lock seu, semeado pela app | Claude | done | nada |
| A09 | Os workflows: o gasto medido de cada passo e os spans públicos (G-32), as dependências fracas e a razão de um salto (G-33) | Claude | done | nada |
| A10 | Os manifestos importados (G-34): os aliases do YAML, o aninhamento e as heranças com limite, num só sítio para os manifestos e os recursos | Claude | done | nada |
| A11 | Os argumentos das ferramentas a chegar (G-37): eventos `tool_call_delta` em todos os adaptadores | Claude | done | nada |
| A12 | A memória num pedido com imagens (G-38): o `MemoryMiddleware` procura pelo texto das partes | Claude | done | nada |
| A13 | Que modelos vêem imagens (G-39): a matriz pelas páginas dos fornecedores, o cenário `vision`, e as imagens no adaptador xAI | Claude | done | nada |
| A14 | A reserva de uma imagem pela qualidade e pelo tamanho (G-40): as contagens publicadas por modelo, nos adaptadores | Claude | done | nada |

## Frente activa: geração de imagens

- **Estado:** aberta em 2026-10-03, a pedido do dono. **Frente I concluída** a 2026-10-03: as cinco
  fichas estão feitas, commitadas e publicadas em `main` a pedido do dono (`07c16c7` sonda,
  `f8d25a7` código, `23c4ed7` docs, `9edbe66` blackboard; a árvore do código passa o gate
  sozinha).
  - Gate: 6217 passed, 42 skipped.
  - Ao vivo: OpenAI e Meta (cerca de $0.45 no total). O Gemini e o xAI esperam pela faturação e
    pelos créditos ("Por fazer").
- **Decisões:** D46 (a forma: `LLM.generate_image()` e `Response.images`) e D47 (os parâmetros
  portáveis). O dono aceitou as recomendações a 2026-10-03.
- **Origem:** o ai-network precisa de gerar imagens (a G-36 dele; a E06-09 espera por isto).
  - Hoje o toolkit não tem chamada para modelos de imagem.
  - As imagens de uma resposta perdem-se: o Gemini deita fora as partes `inline_data`, e o
    núcleo da Responses ignora os `image_generation_call`.
- **Base:** `main` @ `be64062`; gate 6086 passed, 42 skipped.
- **Com as outras frentes:**
  - A C05 (server tools): a `image_generation()` da I04 é a primeira server tool com config
    tipada, e a C05 generaliza a partir dela.
  - A C06 (catálogo): os modelos de imagem entram nos factos quando a C06 correr.
  - A C01 (`Agent.stream()`): os eventos `image` passam pelo `on_event` dela.
  - A frente T não toca em adaptadores.
- **Como correr:**
  - A I01 escreve-a e corre-a o Claude, depois de o dono autorizar as chamadas pagas (menos de
    $1).
  - A I02 pode começar já, em paralelo, num agente com contexto limpo.
  - A I03 espera pela I01 e pela I02, a I04 pela I03, e a I05 fecha a frente.
  - Os agentes não fazem commits nem chamadas a fornecedores; o dono revê e commita.

| ID | Tarefa | Dono | Estado | Depende de |
|---|---|---|---|---|
| I01 | Sonda ao vivo: as APIs de imagem (custo no `usage`, edição sem estado, assinaturas do Gemini, tamanhos) | Claude (script e execução, com autorização do dono) | done | nada |
| I02 | Tipos, preços e o charge site: `GeneratedImage`, `Response.images`, `LLM.generate_image()`, tokens e tarifas de imagem | Claude | done | nada |
| I03 | Adaptadores: gerar e editar no OpenAI (Images API), no Gemini, no xAI e na Meta; preços e regras por modelo | Claude | done | I01, I02 |
| I04 | Imagens no turno: a server tool `image_generation()` do OpenAI, o evento `image` no stream, o reenvio | Claude | done | I01, I03 |
| I05 | Documentação, exemplo e verificação ao vivo final | Claude | done | I03, I04 |

### Quebras visíveis

- **I04:** o `StreamEvent.kind` ganha `"image"`. O resto é aditivo.

## Frente activa: contrato das tools

- **Estado:** aberta em 2026-09-30. Fichas T00 (regras) e T01 a T09 escritas. A T01 está feita a
  2026-10-05 (Claude, a pedido do dono, com cinco agentes nos módulos) e publicada (`8cd6c3c`); a T02 também (`afe6674`); a T04b também (`7532ba4`); a seguir, a T05. A T03 e a T04a
  estão feitas: PR #71, em `main` desde 2026-09-30 (`84c0ee2`). A T01 já pode começar: a sessão
  paralela, que mexia no `_http.py` e em nove módulos, terminou (`6a34668`).
- **Plano:** `docs/internal/tools-contract-plan.md`. **Regras:** `tasks/T00-rules.md`, que remete
  para `tasks/R00-rules.md`. **Decisões:** D37 a D42, tomadas pelo coordenador por delegação do dono.
- **Origem:** duas conversas do ai-network em que o agente não chegou ao que as páginas tinham, e o
  levantamento das 132 tools (29/09).
- **Base:** `main` @ `6a34668`; gate 5690 passed, 42 deselected.
- **Já feito antes da abertura:** `c259e0b` fez a parte da D38 sobre os erros dentro de um 200,
  ainda com strings. Resolveu as confirmadas do anexo C nas tools MediaWiki, `wikipedia_*`,
  `wikidata_search`, `country_info`, `world_bank_*` e `overpass_*`, e corrigiu o `gdelt_timeline`.
- **Já feito depois da abertura:** `6a34668`, a pedido do dono, fez o resto do anexo C, ainda com
  strings: a porta lê o erro que a fonte explica num 4xx ou 5xx e aceita, por chamada, uma resposta
  vazia (`allow_empty`); erros do Eurostat, do arXiv e da UniProt, entradas inactivas da UniProt,
  registos fundidos ou apagados da Open Library, QID fundido, 500 do EONET, histórias do HN que
  falham e a pesquisa do `wikipedia_related`. Cada ficha diz o que já tem.
- **Com a frente C:** C01, C04, C06 e C09 não tocam em tools e podem correr em paralelo. A C02 e a
  T04a partilham `core/_tools`, por isso aplicam-se em série. A C07 e a C08 criam tools: esperam
  pela T01 e pela T03 e nascem com o contrato. As fichas delas ainda dizem "erros → string"; vale a
  D37.
- **Como correr:** cada ficha num agente com contexto limpo; a T06, a T07, a T08 e a T09 em
  worktrees, em paralelo. Os agentes não fazem commits nem chamadas a fornecedores; o dono revê e
  commita.

| ID | Tarefa | Dono | Estado | Depende de |
|---|---|---|---|---|
| T01 | Falhas tipadas: `ToolFailure`, executor, `is_error`, os 44 módulos sem strings de erro | Claude | done | nada |
| T02 | Porta HTTP: um leitor de erros por fonte, 404 por endpoint, erro da fonte na mensagem | Claude | done | T01 |
| T03 | Janela: primitiva de corte com rodapé e continuação | Claude | done | nada |
| T04a | Limites na assinatura: marcador no schema e no validador | Claude | done | nada; em série com a C02 |
| T04b | Invariante de contrato e lista de dívida | Claude | done | T01, T02, T03, T04a |
| T05 | Família wiki: HTML, navegação e fusão (8 tools) | — | todo | T04b |
| T06 | Literatura e identificadores (8 módulos, 18 tools) | — | todo | T05 |
| T07 | Vida e saúde (9 módulos, 34 tools) | — | todo | T05 |
| T08 | Dados, geo e notícias (13 módulos, 44 tools) | — | todo | T05 |
| T09 | Ficheiros, web e o resto (11 módulos, 28 tools) | — | todo | T05 |

### Ordem de aplicação

| Vaga | Tarefas | Porquê |
|---|---|---|
| 1 | T01, T03, T04a | Ficheiros quase disjuntos: a T01 mexe no núcleo das tools, na porta e em todos os módulos; a T03 cria um módulo novo (e toca no `_bounded` depois da T01); a T04a mexe no `_schema.py` e no `_validation.py`. |
| 2 | T02 | Precisa do `ToolFailure`; mexe na porta e nos módulos com leitores ou `status_messages`. |
| 3 | T04b | A invariante verifica o que as costuras dão. |
| 4 | T05 | A primeira família a sair da lista de dívida; fica como modelo. |
| 5 | T06, T07, T08, T09 | Módulos disjuntos, em paralelo; o coordenador junta a lista de dívida, o `CHANGELOG` e os docs. |

### Quebras visíveis

- **T01:** uma tool que falha, chamada crua, lança `ToolFailure`; pelo executor dá `ok=False` com o
  tipo. `ToolFailure` é pública.
- **T03:** os textos de corte mudam para o formato do rodapé.
- **T04a:** um argumento fora dos limites passa a ser recusado com `validation_error`, em vez de
  ajustado em silêncio. O marcador de limites é público.
- **T05 a T09:** tools fundidas ou renomeadas (sem aliases, com tabela de migração) e saídas
  formatadas de outra maneira. O `nanope` importa tools pelo nome: pergunta-se ao dono antes de tirar
  um nome que ele use.

## Frente activa: capacidades em falta

- **Com a frente T (30/09):** C01, C04, C06 e C09 seguem em paralelo. A C02 aplica-se em série com a
  T04a. A C07 e a C08 esperam pela T01 e pela T03, e as tools delas seguem a D37 (falhas tipadas),
  não o "erros → string" das fichas.
- **A vez:** volta a esta frente com o fim da R03 (2026-09-18). A base é o `main` publicado com a
  R03 (`6ceb516` e o registo; gate 5465 passed, 42 deselected). As fichas
  foram escritas antes da frente de robustez: quem pegar numa relê a sua secção de ficheiros contra
  o `main` novo (a porta `_http.py`, o `FlowOptions`, as chaves `answer`/`response` e as
  declarações dos manifestos mudaram o terreno de C02, C05, C07 e C08).
- **Estado:** aberta em 2026-09-15. As nove fichas estão escritas; nenhuma tarefa começou.
- **Antes de codificar:** o dono fixa as "Decisões a fixar" de cada ficha. Cada decisão tomada entra
  em `DECISIONS.md` a partir de D62 (as D15 a D42 foram para as frentes R e T, as D43 a D45
  para a frente O, as D46 e D47 para a frente I e as D48 a D61 para a frente A), com o número dado pelo coordenador.
- **Origem:** o que `docs/internal/agentes-app-toolkit-review.md` pediu ao toolkit (L1, L3–L7, L9, L13
  e o ponto D) e que `docs/internal/toolkit-fix-plan.md` §4 (itens 3 e 10) deixou de fora por ser
  âmbito, não contrato partido.
- **Base:** `main` @ `7ebf7ef`; baseline 3045 passed, 22 skipped. **Coordenador:** sessão principal.
- **Fora da frente:** `agent_as_tool` (a delegação é uma tool da app); scheduler, cofre, descoberta
  local, router `Auto` e escolha de arquitectura (app: L2, L8, L12); sandbox de código (L10).
- **Exemplos novos:** levam o próximo número livre (hoje 49), atribuído pelo coordenador ao aplicar.

| ID | Tarefa | Dono | Estado | Depende de |
|---|---|---|---|---|
| C01 | `Agent.stream()`: texto, thinking e tools através das estratégias | — | todo | nada |
| C02 | API pública para tools dinâmicas (`tool_from_schema`) | — | todo | nada |
| C03 | Cliente MCP (`toolkit.mcp`, extra `mcp`) | — | todo | C02 |
| C04 | Checkpoint e retoma de runs | — | todo | nada; aplicar depois do C01 |
| C05 | Server tools: config no fio e `server_tools=` nas estratégias | — | todo | nada; aplicar depois do C01 |
| C06 | Catálogo técnico de modelos no core | — | todo | nada; C06e depois do C01 e do C05 |
| C07 | Tools de escrita tipadas e `FilesystemPolicy` | — | todo | nada; C07c depois do C02 |
| C08 | Pesquisa web local (Brave e Tavily) | — | todo | nada |
| C09 | `FlowSpec` e máquinas de estados | — | blocked | C04a, C04b; formas validadas na app |

### Ordem de aplicação

Há poucas dependências lógicas; a ordem vem sobretudo dos ficheiros partilhados.

| Vaga | Tarefas | Porquê |
|---|---|---|
| 1 | C02, C06a–d, C07 sem C07c, C08 | Ficheiros quase disjuntos: `core/_tools`, módulos novos, `toolkit/tools`. |
| 2 | C01, C03, C07c | C01 mexe no motor, no `_llm.py`, nos adaptadores e nas estratégias (depois do C02a); C03 precisa do C02; C07c toca em `_definition.py` e `_decorator.py` depois do C02. |
| 3 | C04, C05 | Ambas depois do C01: C04 no motor; C05 nos adaptadores, no `_llm.py`, nas estratégias e no `budget/_estimator.py` (depois do C08c). Partilham `_lats.py` e `_manifest.py`: o coordenador aplica em série. |
| depois | C06e, C09 | C06e mexe nos adaptadores; C09 espera pela app. |

Quase todas tocam em `core/__init__.py`, `ai_arch_toolkit/__init__.py`, `docs/tools.md`,
`docs/safety.md`, `docs/agents.md` e `docs/api.md`; o coordenador junta-os ao aplicar.

### Decisões que mudam o contrato público

As primeiras a fixar. As alternativas e as razões estão nas fichas.

- **C01:** os deltas nascem em `LLM.complete(on_event=)`; `FlowEvent` ganha tipos e campos;
  `stream()` é novo e `iter()` não muda.
- **C02:** o handler recebe um `dict`; regra de nomes portátil em `ToolSchema`; nome repetido levanta
  `ValueError` (as duas últimas são quebras visíveis).
- **C03:** `mcp>=2.2,<3`; a ligação vive numa task própria; sem wrappers `_sync` (excepção à regra do
  `AGENTS.md`).
- **C04:** checkpoints só em fronteiras do motor; journal de tools no executor governado;
  `MeterScope(baseline=)` para o budget continuar.
- **C05:** config tipada que falha quando o fornecedor não a aplica; `ReasoningSpec.server_tools`;
  custo por uso na tabela de preços; o OpenAI passa a levantar (quebra visível).
- **C06:** um facto descreve o que funciona através do adaptador; ids exactos e aliases, nunca
  prefixo; os adaptadores não lêem o catálogo.
- **C07:** a policy é verificada no gate e outra vez na tool; hook `@tool(preview=)` e outcome
  `permission_denied` no core.
- **C08:** uma factory por fornecedor, com chave explícita; `capability="web_search"`, aprovação
  obrigatória, em `toolkit.tools`.

### Achados da abertura

Vinte entradas em `FINDINGS.md` (2026-09-15): catorze reproduzidas pelo coordenador, um risco medido e
cinco confirmadas no código e na documentação oficial. Onze não tinham tarefa, por serem correcções
e não capacidades: o plano de robustez agrupou-as nas suas seis causas, que a frente R corrigiu. O
que ficou aberto está em "Por fazer".

## Frente anterior: OpenAI pela Responses API

- **Estado:** aberta em 2026-10-01, a pedido do dono. A O01 está feita (2026-10-02): a latência
  da Responses não fica atrás, e o pedido da O03 ficou fixado. A O02 está feita (2026-10-02):
  núcleo em `_responses.py`, Meta sobre ele. A O03 também (2026-10-02): o host oficial vai pela
  Responses e os servidores compatíveis pela Chat Completions. A O04 fechou a frente
  (2026-10-02): docs, verificação ao vivo (77 de 77 em 13 modelos OpenAI, batch e turno
  reconstruído) e as correcções que ela encontrou (D45). **Frente O concluída** e publicada
  em `main` (`be64062`).
- **Decisão:** D43. O host oficial passa à Responses, a Chat Completions fica para os servidores
  compatíveis, e a API pública não ganha escolha de endpoint.
- **Origem:** desde o GPT-5.4 a Chat Completions só aceita tools com effort `none`. Dos três
  modelos de topo do catálogo da OpenAI (`gpt-6-astra`, `gpt-6.1-sol`, `gpt-6-luna`), nenhum corre
  aqui um agente com tools a raciocinar. O LOG de 2026-09-28 tinha deixado a porta fora de âmbito,
  pelos custos relatados; a O01 mede-os.
- **Já feito na abertura:** o `gpt-6.1-sol` registado (perfil como o do Astra, preços e inventário
  de probes), commitado e publicado em `main` a pedido do dono (`e18fcb5`). Gate: 5763 passed, 42
  skipped.
- **Base:** `main` @ `e18fcb5`.
- **Com as outras frentes:**
  - A C01, a C05 e a C06 mexem nos mesmos adaptadores: a O02 e a O03 aplicam-se em série com elas.
  - Na C05, a decisão 6 (server tools do OpenAI) ganha a Responses como caminho: é a opção (c).
  - A C06 descreve o que funciona através do adaptador: o "Astra sem tools" deixa de valer depois
    da O03.
  - A frente T não toca em adaptadores.
- **Como correr:**
  - A O01 escreveu-a e correu-a o Claude, com autorização do dono para as chamadas pagas.
  - A O02 pode começar já, num agente com contexto limpo.
  - A O03 espera pela O02.
  - Os agentes não fazem commits nem chamadas a fornecedores; o dono revê e commita.

| ID | Tarefa | Dono | Estado | Depende de |
|---|---|---|---|---|
| O01 | Sonda ao vivo: a Responses do OpenAI contra a Chat Completions (latência, reenvio, `strict`) | Claude (script e execução, a pedido do dono) | done | nada |
| O02 | Núcleo Responses partilhado, extraído do `_meta.py`; reenvio só ao mesmo fornecedor e família | Claude (agente) | done | nada |
| O03 | OpenAI pela Responses no host oficial; Chat Completions só para servidores compatíveis | Claude (agente) | done | O02 |
| O04 | Documentação, quebras visíveis e verificação ao vivo final | Claude (agente nos docs; coordenador ao vivo) | done | O03 |

### Quebras visíveis

- **O03:**
  - No host oficial, `stop`, `seed`, `frequency_penalty` e `presence_penalty` levantam
    `RequestError`.
  - `thinking=True` com tools deixa de levantar do GPT-5.4 em diante, e o Astra e o 6.1 Sol passam
    a chamar tools.
  - O OpenAI passa a dar resumos do raciocínio e um `_raw` reenviável.

## Frente anterior: robustez (três fases)

- **Estado:** aberta em 2026-09-17. **Base:** `main` a partir do commit que abre esta frente.
  Baseline: 3070 passed, 22 deselected; ruff, formatação e pyright limpos.
- **Plano:** `docs/internal/hardening-plan.md`. **Regras comuns:** `tasks/R00-rules.md`.
  **Decisões:** D15–D36.
- **R01:** commitada e publicada em `main` a pedido do dono (`a16e3f9` tools, `f95c81f` F25,
  `5a0ab3b` núcleo, `6206df1` registo, `b3dae3f` `temperature` da Anthropic). Cada commit passa o gate
  sozinho (3097, 3107, 3886 e 3886 passed).
- **R02:** concluída (Claude) e commitada em `main` a pedido do dono (`cf2aa43` código e testes,
  `a8c3eea` rede do fio, e o registo); publicada com a R03. Cada árvore passa o gate sozinha (4305, 4341 e 4341
  passed); 42 live_api deselected; dívida de complexidade 161 → 121. Falta o dono correr as
  verificações ao vivo (comandos no relatório final da ficha).
- **R03:** concluída (Claude) em 2026-09-18; commitada e publicada em `main` a pedido do dono
  (`6ceb516` código e testes, e o registo). A árvore do primeiro commit foi verificada sozinha.
  Gate: 5465 passed, 42 deselected, zero `xfail`; ruff, formatação, pyright e lock limpos. Dívida de
  complexidade 121 → 46. As verificações ao vivo das tools gratuitas e sem chave (comandos no
  relatório) estão feitas só em parte: a 28/09 viram-se seis tools e cinco estavam partidas; quatro
  foram corrigidas em `d892cfd`, e o `define_word` depende do dictionaryapi.dev, que não responde.
  Ficam para decisão do dono: o risco polinomial do regex, o `pdb_ligands`, o arXiv e o guarda de
  tamanho do `python_repl` (em `FINDINGS.md`; a UniProt ficou resolvida em `d892cfd`), o
  levantamento do `__all__` de topo (ficha, passo 4.7) e o saldo positivo da dívida de manutenção,
  por causa dos +593 do `_shape.py` (ficha, passo 4.4).
- **Como correr:** uma fase de cada vez, por ordem, cada uma num agente com contexto limpo. A fase
  seguinte só começa depois de o dono rever e commitar a anterior. Os agentes não fazem commits nem
  chamadas a fornecedores.

| ID | Fase | Dono | Estado | Depende de |
|---|---|---|---|---|
| F24 | Quatro contratos pequenos | coordenador | done | — |
| R01 | Núcleo de chamadas: F25 (nove correcções locais), erros tipados, meter com disposições e tecto incerto, pipeline de tentativa única | Codex; Claude (continuação) | done | nada |
| R02 | Fornecedores: ids e preços, contrato de três fases, um adaptador de cada vez (OpenAI, xAI, Gemini, Meta, Anthropic) | Claude | done | R01 |
| R03 | Tools, motor de flows e dívida de manutenção | Claude | done | R01, R02 |

## Frente anterior: achados em aberto

- **Estado:** concluída em 2026-09-15 (F23 done), commitada e publicada em `main` a pedido do dono
  (`b504998` tools, `dfaca5c` providers, `19a3905` pytest-timeout, `4509bc9` docs, e o registo).
- **Ficha:** `tasks/F23-open-findings.md`. **Base:** `main` @ `1334fa8`.

| ID | Tarefa | Dono | Estado |
|---|---|---|---|
| F23 | Corrigir os achados que ficaram em `FINDINGS.md` | coordenador | done |

## Frente anterior: fornecedor Meta (Muse Spark)

- **Estado:** concluída em 2026-09-13, commitada e publicada em `main` a pedido do dono (`9588f31` código e testes, `7967287` docs, e o registo).
- **Regra do dono:** nada de testes reais (`live_api`) no CI do GitHub; correm só localmente. O workflow de integração foi removido.
- **Ficha:** `tasks/M01-meta-provider.md`. **Decisões:** D11–D14.
- **Base:** `main` @ `a0807cc`. **Coordenador:** sessão principal (sem workers).

| ID | Tarefa | Dono | Estado |
|---|---|---|---|
| M01 | `MetaProvider` sobre a Responses API do SDK `openai` | coordenador | done |

## Frente anterior: plano de correcção do toolkit

- **Estado:** concluída em 2026-09-13 (F01–F22 done), commitada e publicada em `main`.
- **Plano:** `docs/internal/toolkit-fix-plan.md` — a secção 0 tem os ajustes da revisão cruzada.
- **Origem:** `docs/internal/agentes-app-toolkit-review.md`.
- **Base:** `main` @ `48a43ac`. Baseline: 2608 passed, 7 skipped; pyright e ruff limpos.
- **Commits:** publicados em `main` a pedido do dono (`14e623d`..`cc83cb9` e o registo).
- **Coordenador:** sessão principal. Aplica os diffs das worktrees, corre a suite, escreve o `CHANGELOG`.

### Vaga 1 — independentes (workers em worktrees; coordenador no checkout principal)

| ID | Tarefa | Dono | Estado |
|---|---|---|---|
| F01 | Usage Anthropic: deltas cumulativos e campos `null` | worker-A | done |
| F02 | `Flow.policy` por step e `Flow(timeout=)` | coordenador | done |
| F04 | `run_tools` usa a governança do `ToolGroup` | worker-B | done |
| F05 | Fundir prompts de sistema nos adaptadores | worker-E | done |
| F06 | `ToolGroup` rejeita `ServerTool` e não-callables | worker-B | done |
| F07 | Tools async no caminho síncrono; positional-only | worker-B | done |
| F09 | `RunConfig` por execução no `Agent` | coordenador | done |
| F10 | `provider` nos pedidos de metering | worker-E | done |
| F11 | Metadados de risco em `tools.dangerous` | worker-C | done |
| F12 | Schema: uniões multi-tipo e varargs | worker-C | done |
| F14 | `Scope.enrich` sobre o snapshot filtrado | coordenador | done |
| F15 | Wrappers síncronos: cancelamento e backpressure | worker-D | done |
| F17 | Exportar a superfície de gates | worker-B | done |

### Vagas seguintes — coordenador, no checkout principal

| ID | Tarefa | Estado | Depende de |
|---|---|---|---|
| F03 | Middleware async em streaming | done | F05, F10 (aplicados) |
| F13 | Validar e coagir argumentos antes dos gates; encadear gates; C2/C3 | done | F07, F12 (aplicados) |
| F08 | Motor único de execução (8a–8g) | done | F02, F09 |
| F16 | Política de captura do trace | done | F08 |
| F19 | Gemini: schemas que `types.Schema` rejeita (C1) | done | F12 (aplicado) |
| F20 | `run_tools` verifica todos os nomes antes de executar | done | F04 (aplicado) |
| F18 | Deriva de documentação e `CHANGELOG` | done | todas |
| F21 | Achados restantes: nanope, conteúdo de sistema, `_stream_sync`, `Any`, middleware | done | F01–F20 |
| F22 | Revisão adversarial pós-implementação: 3 revisores, correcções confirmadas | done | F21 |

## Por fazer (dono do repositório)

- **Anthropic:** quando houver créditos, correr
  `uv run pytest tests/integration/test_provider_contracts_live.py -m live_api -k anthropic -q`
  (usa o `claude-haiku-4-5`, que leva `temperature` no corpo do pedido). Confirma a correcção do
  `temperature` com o `anthropic` 1.x, que só foi provada com o SDK real em loopback.
- **xAI:** repor créditos na conta e depois correr `uv run pytest -m live_api -k xai` (custo por
  pedido) e um probe com uma tool cujo parâmetro seja `Any` (schema sem tipo) e com `system=` +
  `system()` ao mesmo tempo — únicas mudanças desta frente que o xAI ainda não confirmou.
- **Imagens no Gemini e no xAI, ao vivo:** activar a faturação no projecto da chave Gemini (no free
  tier, os modelos de imagem têm quota 0) e repor os créditos xAI. Depois:
  `uv run python scripts/probe_images.py --provider gemini --provider xai --max-cost 1`.
- **Modelos novos ao vivo:** o Grok 4.7 e o Opus 5.5 ainda não correram ao vivo
  (`docs/model-compatibility.md`). Os 13 modelos OpenAI do inventário, com o Astra e o 6.1 Sol,
  correram a 2026-10-02 (77 de 77, O04).
- **Frente C, decisões:** fixar as da vaga 1 (C02, C06, C07, C08) antes de atribuir donos; a C07 e
  a C08 só começam depois da T01 e da T03.
- **Decidir, achado de 2026-09-18 sem tarefa (`FINDINGS.md`):** o xAI larga os documentos de um
  pedido (o SDK tem `file(...)`). As imagens passaram a ir na A13, e o `LLM("grok-…")` depois de
  um `asyncio.run` já não levanta desde a A03 (G-17). (O `thinking_effort` do OpenAI e o preço do
  batch ficaram resolvidos na frente O, D45.)
- **Frente A, ao vivo e no ai-network:** correr o cenário `vision`
  (`uv run python scripts/probe_models.py --suite full --scenario vision`, uma imagem pequena
  por modelo), e levar ao briefing do ai-network as linhas do Registo das fichas A01 a A14.
- ~~O tecto de uma falha sem `BudgetPolicy`~~: decidido a 2026-10-04 (D49) e feito na A02.
- **R02, verificação ao vivo:** comandos no relatório final de `tasks/R02-providers.md` (o do Gemini
  decide se sai a nota "Known issue").
- **Plano de robustez (`docs/internal/hardening-plan.md`):** as decisões R1–R15 foram fixadas na D20
  e a frente R (R01–R03) está concluída. Ficam por cumprir a R10 (as verificações ao vivo acima) e a
  R11 (uma tag de pré-release por vaga; ainda não há nenhuma). Do que o plano absorveu da frente C,
  a C01a (R02), a C02a (F24), a C08c (R01) e a validade do fio do C05a (R02) estão feitas; o
  `@tool(schema=)` com um schema completo (C02b) continua por fazer. Protótipos em
  `blackboard/prototypes/2026-09-hardening/`.
