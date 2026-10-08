# C08 · Pesquisa web local (Brave e Tavily): o que falta

- **Dono:** Claude (um agente, 2026-10-08) · **Estado:** done · **Depende de:** T05 (o modelo das pesquisas)
- **Origem:** L13 (`docs/internal/agentes-app-toolkit-review.md:424-446`, `:586-588`);
  `toolkit/tools/CANDIDATE_TOOLS.md` (Brave #22, Tavily #23)
- **Decisões:** D55 e D56 (as tools, a chave e o preço, na A07), D65 (o que fica desta ficha).
- **Reescrita a 2026-10-08 (D65, C08.7).** A A07 fez as tools `brave_search` e `tavily_search` de
  outra maneira do que esta ficha propunha (chave do ambiente, `capability="network"`, preço na
  tabela `[tools]`). A proposta original (factories `*_tool(api_key)`, aprovação obrigatória, rótulo
  "untrusted") está no histórico do git, até `601e38c`. A C08c (o estimador reserva o preço de uma
  tool) saiu na R01.

## O que já existe

- `toolkit/tools/_web_search.py`: as duas tools, cada uma com a sua `Api` (`key_env`,
  `key_required`, `billed_as`; a Tavily com `bill_units` pelos créditos que diz ter gasto), leitores
  de erro (401/403 da chave; 432/433 da Tavily), testes em `tests/toolkit/test_web_search.py`.
- O preço das duas em `[tools]` da tabela de preços (D56), com testes em `tests/test_pricing.py`.

## O que falta

1. **A dívida de contrato** (`tests/toolkit/contract_debt.py`: as duas devem `window`, `errors`,
   `zero` e `limits`):
   - `max_results` com `Range` na assinatura, e os limites de cada API citados (Brave `count`
     1–20; Tavily `max_results` 0–20);
   - zero resultados é sucesso que diz a consulta (já o diz; o contrato tem de o reconhecer);
   - os casos de erro de cada fonte em `error_bodies.py`, com o corpo que a documentação mostra;
   - a janela: a lista de resultados diz quantos mostra e, onde a API pagina (Brave `offset`),
     a chamada exacta para os seguintes; onde não pagina (Tavily), di-lo. Os snippets cortados aos
     500 caracteres deixam de ser um corte sem saída: ou o snippet vem inteiro, ou o rodapé diz o
     corte e o caminho (por exemplo, `scrape_text` no URL).
2. **C08.3:** sem aprovação obrigatória, risco `low`, como as outras tools de rede (já é assim no
   código; fica afirmado por teste e nas docs).
3. **C08.4:** os URLs dos resultados filtram-se a `http(s)` (um `javascript:` ou `data:` sai);
   sem rótulo de conteúdo de terceiros.
4. **Docs:**
   - `docs/tools.md`: o contraste com a server tool `web_search()` (quem corre, quem cobra, que
     modelos serve);
   - `docs/safety.md`: a query leva o que o contexto lá puser, e os resultados são texto de
     terceiros; a app que queira aprovar re-decora a tool ou põe um gate seu;
   - `CANDIDATE_TOOLS.md`: #22 e #23 em parte feitos (o que falta: news, LLM context, extract).
5. **Exemplo** `examples/49_local_web_search.py`: um modelo local (Ollama, `base_url` loopback) com
   `brave_search` ou `tavily_search`, a que tiver chave; `examples/README.md` e `docs/examples.md`.
6. **Ao vivo, só local:** `tests/integration/test_web_search_live.py`, `integration` + `live_api`,
   skip sem chave: uma pesquisa real em cada, com pelo menos um resultado e o custo no meter.

## Ficheiros

`toolkit/tools/_web_search.py`, `tests/toolkit/test_web_search.py`, as linhas destas duas tools em
`tests/toolkit/contract_debt.py`, `contract_cases.py` e `error_bodies.py`,
`tests/integration/test_web_search_live.py` (novo), `docs/tools.md`, `docs/safety.md`,
`docs/tools-catalog.md` (a secção destas duas), `toolkit/tools/CANDIDATE_TOOLS.md`,
`examples/49_local_web_search.py` (novo), `examples/README.md`, `docs/examples.md`.

## Prova

- As duas saem da lista de dívida; a invariante de contrato passa.
- Brave: a página seguinte pela continuação do rodapé (`offset`); `count` fora do `Range` é
  `validation_error` do schema.
- Um resultado com `javascript:` não aparece; um snippet longo não se perde sem rodapé.
- A política das duas, afirmada campo a campo: `capability="network"`, risco `low`, sem aprovação.
- **Ao vivo, pelo dono:** `uv run pytest tests/integration/test_web_search_live.py -m live_api`
  com `BRAVE_SEARCH_API_KEY` e `TAVILY_API_KEY` (cada pesquisa custa: Brave $0,005, Tavily $0,008).

## Fora do âmbito

Brave news/LLM context/imagens/local; Tavily extract/crawl/map/research; `include_raw_content`;
factories com chave explícita; `web_search()` e `server_tools=` (C05).

## Registo do dono

- **Estado:** done. Feito por agentes em worktrees, juntado e revisto pelo coordenador; uma revisão independente por parte, e as correcções dela (secção "Correcções da revisão").
- **Gate final, no checkout principal com todas as fichas da vaga:** 8195 passed, 118 skipped (os ao vivo e os 59 do nanope à espera); ruff, formatação, pyright e `uv lock --check` limpos (2026-10-08).
- **CHANGELOG:** as linhas entraram em `[Unreleased]` (Upgrade notes, Added, Changed, Fixed).

Worktree: `/Users/rge/Documents/dev/pessoal/ai-arch-toolkit/ai-arch-toolkit/.claude/worktrees/agent-ad3149162f411e215`.
Tudo por commitar. Estado proposto: `review`.

### Nota de desenho

Fontes, só documentação oficial (WebFetch e WebSearch de páginas de docs). Nenhum pedido às APIs,
nenhuma chave, nenhum dado pessoal em pedido algum.

- Brave, referência da pesquisa web:
  https://api-dashboard.search.brave.com/api-reference/web/search/get
  - `count` inteiro de 1 a 20, por omissão 20; vale só para os resultados web.
  - `offset` inteiro de 0 a 9, por omissão 0: "the zero based offset that indicates number of
    search result pages (count) to skip". Conta páginas, não resultados.
  - `q` até 600 caracteres e 75 palavras.
  - Erros documentados: 404, 422 e 429, todos com o esquema `ErrorResponse`: `type`, `error`
    (`id`, `status`, `code`, `detail`, `meta`) e `time`. O `code` não tem lista de valores.
- Brave, início e paginação:
  https://api-dashboard.search.brave.com/app/documentation/web-search/get-started
  (`query.more_results_available`: só pedir a página seguinte quando é `true`).
- Brave, limites de ritmo:
  https://api-dashboard.search.brave.com/documentation/guides/rate-limiting
  (um pedido falhado não conta nem se cobra; o 429 traz `X-RateLimit-Reset`, não `Retry-After`).
- Tavily, referência da pesquisa:
  https://docs.tavily.com/documentation/api-reference/endpoint/search
  - `max_results` inteiro de 0 a 20 (por omissão 10 na API); sem `offset`, sem página.
  - Corpos de erro `{"detail": {"error": "…"}}`: 400, 401, 429 (com `Retry-After` possível),
    432, 433, 500. O 422 traz `detail` em lista de `loc`, `msg`, `type`, `input`.
  - Pesquisa `basic` custa um crédito; `include_usage` devolve os créditos.

Decisões, dentro de D39, D42, D55, D56 e D65:

- **Limites na assinatura.** `brave_search(max_results: Range(1, 20) = 10, offset: Range(0, 9))`;
  `tavily_search(max_results: Range(0, 20) = 5)`. Sai o `max(1, min(…, 20))` das duas. O
  `offset` da Brave entra no fim da assinatura, para não partir chamadas posicionais.
- **Janela da Brave.** Os resultados numeram-se pela posição na Brave (`offset * count + i`).
  Enquanto `more_results_available` é `true` e `offset < 9`, o rodapé dá
  `next: offset=N+1, max_results=M`: o `offset` conta páginas de `count`, por isso a continuação
  leva também o `max_results`. Na página 10, com mais resultados, o corpo diz que a Brave não
  serve página depois do `offset` 9. A tool devolve `Window.result()`, com `metadata["window"]`.
- **Tavily sem página.** Mostra tudo o que recebe, e quando a página vem cheia diz que a Tavily
  só tem uma página e como pedir mais (subir o `max_results` até 20, ou refinar a consulta). Não
  corta nada: entra no `WHOLE` do contrato. `max_results=0` só faz sentido com
  `include_answer=True`; sem ele é `validation_error` antes do pedido (não gasta um crédito).
- **Snippets inteiros.** Sai o corte aos 500 caracteres (`_SNIPPET_CHARS`). Um campo de um registo
  não se corta abaixo da janela (D39); o tecto do executor continua a valer.
- **C08.4.** Só passam resultados com URL `http(s)` e anfitrião (`urlsplit`); um `javascript:` ou
  `data:` sai, e o texto diz quantos saíram, para a numeração não parecer errada. Sem rótulo de
  conteúdo de terceiros.
- **C08.3.** A política fica como estava (`network`, `low`, sem aprovação), agora afirmada por
  teste campo a campo. Um teste prova também o caminho que as docs dão à app: re-decorar com
  `tool(requires_approval=True)(brave_search)` mantém o nome, o schema e o preço.
- **Erros.** O leitor lê só respostas de erro (um sucesso nunca é erro). Lê o `error.detail` da
  Brave, o `detail.error` da Tavily e a lista do 422 da Tavily (`query: Input should be a valid
  string`). Brave 422, Tavily 400 e 422 passam a `validation_error` (os argumentos são de quem
  chama, D42). O 429 das duas leva as palavras do serviço: antes perdiam-se, porque o leitor
  devolvia `None` no 429. 401/403, 432/433 como estavam.
- **D55 e D56 intactas:** chaves do ambiente, preços em `[tools]`, créditos da Tavily cobrados,
  nomes iguais.

Saldo de linhas: `_web_search.py` 250 → 416. Cresce com a janela, a paginação, o filtro, os erros
documentados e as docstrings.

### Mudanças em ficheiros partilhados

- `tests/toolkit/contract_debt.py`: saem só as linhas de `brave_search` e `tavily_search`. O
  `CUT_HELPERS` fica em 14 (o módulo não tinha `_truncate` nem `_trim`).
- `tests/toolkit/contract_cases.py`: `tavily_search` entra no `_WHOLE`, com a razão no
  comentário; `WINDOW_CASES["brave_search"]` e `ZERO_CASES` das duas; um helper `_brave_page`.
- `tests/toolkit/error_bodies.py`: a entrada `"_web_search"` (três da Brave, sete da Tavily),
  com os helpers `_brave` e `_tavily`. Pus a entrada no topo do dicionário, para colidir menos com
  as outras migrações.
- `tests/quality_baseline.json`: não muda (nenhuma entrada do módulo).
- Nenhuma mudança em `core/`, `_http.py`, `_window.py` nem noutros módulos de tools.
- `examples/README.md`: a contagem dizia "Forty-seven" com 48 ficheiros; passa a "Forty-nine".

### Prova

- As duas saem da lista de dívida, e a invariante de contrato passa para as duas (janela da Brave,
  erros documentados, zero resultados, limites).
- `tests/toolkit/test_web_search.py`: 49 testes (eram 20). Escritos antes do código; 22 falharam
  pela razão certa (o `javascript:` aparecia, o snippet vinha com "…", o 429 perdia as palavras,
  não havia `offset` nem `Range`, o 400 era `upstream`).
  - Brave: a página seguinte pela continuação do rodapé (`offset=1, max_results=2`, depois
    `[results 3-4 | end]`); a página 10 diz que não há mais; uma página vazia diz de onde;
    `count` e `offset` fora do `Range` são `validation_error` do schema, sem pedido.
  - Um resultado com `javascript:` ou `data:` não aparece; um snippet de 2000 caracteres vem
    inteiro.
  - A chave não aparece no URL nem em nenhuma mensagem (401, 403, 422, 429, 500), nem no corpo
    do pedido da Tavily.
  - A política das duas, campo a campo; um grupo sem handler corre-as; a re-decoração com
    aprovação é negada sem handler e, aprovada, corre e custa o preço da `brave_search`.
  - Um pedido recusado não custa nada (401, 422, 429, 432, 500, nas duas), nem um recusado antes
    de sair (`max_results=0`).
- Testes corrigidos (afirmavam o comportamento antigo): o 422 da Brave e o 400 da Tavily eram
  `upstream` ("HTTP error 422: …"); passam a `validation_error`. As comparações com a string
  passam a ler `.value`.
- Exemplo 49: verificado sem chave e com chave falsa apontado a uma porta sem servidor: as duas
  saídas dizem o que falta e nenhum pedido chega à Brave (o `_open` da porta estava a falhar de
  propósito). Não corri com servidor e chave.

### Ao vivo, pelo dono

O teste ao vivo foi escrito e nunca corrido. Cada pesquisa custa: Brave $0,005, Tavily $0,008.

```bash
set -a && source .env && set +a
uv run pytest tests/integration/test_web_search_live.py -m live_api
```

Confirma: pelo menos um resultado em cada; o custo no meter ($0,005 e $0,008); na Brave, que uma
consulta popular traz `more_results_available` e o rodapé dá `next: offset=1, max_results=3`.

O exemplo (gratuito do lado do modelo, uma a três pesquisas pagas):

```bash
ollama pull qwen3:8b
set -a && source .env && set +a
uv run python examples/49_local_web_search.py
```

### Bloqueios e achados

- Sem bloqueios.
- **Achado (fora do âmbito, para o `FINDINGS.md`):** a Brave documenta `X-RateLimit-Reset` (os
  segundos até cada janela reabrir) e não `Retry-After`. A porta só lê o `Retry-After`, por isso
  depois de um 429 da Brave o host não descansa. E o `code` do erro (`RATE_LIMITED`,
  `QUOTA_LIMITED`) não tem valores documentados: um 429 da quota mensal sai como `rate_limited`
  repetível. Correcção possível: um `Api(retry_after=...)` que leia o cabeçalho de cada fonte.
- **Para o coordenador:** a frase de `docs/tools.md` "The other tools still keep their numeric
  arguments within their limits themselves…" e a introdução de `docs/tools-catalog.md` ("The wiki
  family declares its limits as `Range` bounds…") já não excluem estas duas. Não lhes toquei,
  porque as migrações em paralelo também lhes mexem; convém juntá-las ao aplicar.
- A Tavily documenta hoje o tópico `finance` e as formas curtas de `time_range` (`d`, `w`, `m`,
  `y`); a tool aceita só `general`/`news` e as formas longas. A Brave aceita `freshness` com um
  intervalo de datas. Fica fora do âmbito, como na ficha.
- O número 49 estava livre nesta worktree e em nenhuma outra ficha; confirma ao aplicar.

### Correcções da revisão

Revisão: `notes/T09b-C08-review.md` (2026-10-08). Na checkout principal, por commitar. Cada teste
foi escrito antes da correcção e falhou pela razão certa. Nenhuma mudança em `_http.py`,
`_window.py` nem `core/`. Não houve nenhum pedido a uma API. Li só documentação, com WebFetch,
sem dados pessoais.

#### M1 · `docs/tools.md`: o exemplo não pesquisava

- **Mudança:** o exemplo corre um agente ReAct:
  - `Agent(ReasoningSpec(strategy="react"), llm, ToolGroup(brave_search)).run_sync(...)`;
  - com `pricing.register("qwen3:8b", ModelPricing())`;
  - e uma frase que diz que o `llm.complete_sync(..., tools=...)` sozinho só devolve a chamada,
    e que o `run_tools_sync` a corre num ciclo próprio (ligação para `#run_tools-helper`).
- **Verificado contra a API real:** `scratchpad/c08fix/verify_snippet.py` corre o exemplo tal
  como está, contra um servidor OpenAI-compatível em 127.0.0.1. O modelo pede o `brave_search`,
  e a Brave é a resposta preparada na costura `_http._open`. Resultado:
  - 2 pedidos ao modelo e 1 à Brave;
  - o resultado da tool volta ao modelo;
  - `result.text` é a resposta, sem erros, com `report.tool_calls == 1`.
- **O registo do preço é preciso.** Sem ele o run não chega a pedir nada: "No price for model
  'qwen3:8b' ... a local model registers zero".
- **Uma célula da tabela (`docs/tools.md`, mesma secção):** `country` passou a dizer "a code Brave
  lists: `GB`, not `UK`".

#### M3 · Brave 422: escolhi validar o `country` antes de pedir

- **A escolha:**
  - A referência da Brave dá o `country` como uma lista fechada, de 38 códigos com `ALL`, `US` por
    omissão (https://api-dashboard.search.brave.com/api-reference/web/search/get, lida a
    2026-10-08). Está em `_BRAVE_COUNTRIES`, com o URL e a data.
  - Um código fora da lista é `validation_error` antes do pedido, com a lista inteira: "invalid
    country 'UK'; use AR, AU, …, GB, US, ALL, or ''".
  - O agente fica a saber qual é o argumento errado, sem uma volta à rede.
- **Porque não só a outra opção:** o `error.meta` é, na referência, "non-standard
  meta-information", sem forma documentada. Sem ele, um 422 não diz qual argumento recusou.
- **O que fiz também, por ser barato:**
  - O passo do 422 nomeia todos os argumentos: "check the arguments: query (at most 600
    characters and 75 words), max_results, offset, country and freshness". Cobre uma recusa que
    não previ, como um código que a Brave deixe de servir.
  - Leio o `error.meta.errors` só quando traz `loc`/`msg` (a forma de validação que a Tavily
    documenta no seu 422). Os campos saem entre parênteses: "Unable to validate request
    parameter(s) (q: String should have at most 400 characters)". Qualquer outra forma é
    ignorada.
- **O risco da lista:** um país que a Brave junte é recusado até a lista mudar. Por isso a lista
  diz a data em que foi lida.
- **`_field_error`:** tira o primeiro elemento do `loc` só quando é a parte do pedido (`body`,
  `query`, `path`, `header`, `cookie`). O `["body","query"]` da Tavily continua a dar `query`, e
  o `["query","q"]` da Brave passa a dar `q`.
- **Testes:**
  - `test_a_country_brave_does_not_serve_is_refused_before_asking`: `uk`, sem pedido.
  - `test_the_fields_brave_names_in_its_meta_are_said`.
  - `test_arguments_brave_cannot_validate_are_a_validation_error_in_its_words`: actualizado, o
    passo nomeia todos os argumentos.
  - Guarda: `test_a_country_brave_serves_is_sent_in_its_form` (`gb`, ` pt `, `all`).
- **O probe do revisor:** o `brave_422.py` pára agora antes de enviar.

#### L8 · O rodapé da Brave: precisa do `_window.py`

- **O `rest=` não chega** sem mudar o `_window.py`, por duas razões:
  - o `Window.footer` escreve "end" sempre que `last < first`, seja qual for a chamada seguinte;
  - o `_onward` só mostra o `rest` quando o total se conhece, e a Brave não dá total.
- **O que fiz no `_web_search.py`:**
  - Uma página vazia com mais diz a chamada no corpo: "(No web results on this page, but Brave
    has more: offset=1, max_results=10 reads the next page.)".
  - A décima página leva `rest=` "Brave serves no page past offset 9: refine the query for other
    results", pronto para quando o `_window.py` o mostrar. A linha do corpo com o mesmo texto
    fica.
- **Testes:**
  - `test_an_empty_page_brave_has_more_after_names_the_next_call`: falhou.
  - Dois `xfail(strict=True)`, com o motivo, para o rodapé que deviam dar:
    `test_the_footer_of_an_empty_page_brave_has_more_after_names_the_call` e
    `test_the_footer_of_the_tenth_page_says_brave_has_more_it_does_not_serve`.
- **A mudança precisa no `_window.py`, para o seu dono:**
  - no `footer`: `[no {unit} from {first}{of} | {_onward(self)}]`;
  - no `_onward`: devolver `window.rest` quando não há `next_call`, também sem total.
- **Quando essa mudança entrar:** os dois `xfail` estritos passam a XPASS e falham. Tiram-se as
  marcas e as duas linhas do corpo.

#### L10 · Os detalhes

- **`docs/safety.md`:** o exemplo usava `approve_handler` antes de o definir. Agora define o seu,
  `review`, que pergunta com `input()` e aprova com "y". Uma frase aponta para "Human approval".
  Verificado com o `input` simulado: "y" corre a pesquisa, Enter dá `approval_denied`.
- **`cooldown_s` (D53).** Os dois documentos dão um número:
  - Brave: 1 s. "Rate limits are enforced using a 1-second sliding window"
    (https://api-dashboard.search.brave.com/documentation/guides/rate-limiting).
  - Tavily: 60 s. Os limites são em pedidos por minuto, e o 429 traz `Retry-After` só às vezes
    (https://docs.tavily.com/documentation/rate-limits).
  - Quando há `Retry-After`, ganha ele (a porta usa `retry_after_s or cooldown_s`).
  - **Fica por fazer, na porta:** um 429 da quota mensal da Brave (o `X-RateLimit-Reset` vai até
    cerca de 30 dias) continua a descansar só 1 s. Ler esse cabeçalho é coisa do `_http.py`, e o
    achado fica para o seu dono.
- **Testes:**
  - `test_after_a_429_brave_rests_for_its_one_second_window` e
    `test_after_a_429_without_retry_after_tavily_rests_a_minute` (relógio parado, sem pedido na
    segunda chamada): falharam.
  - Como o host agora descansa depois de um 429, separei o teste da Tavily que juntava o 429 e o
    500. No ciclo que procura a chave nas mensagens da Brave, o 429 passou para o fim.
- **`examples/49_local_web_search.py`:** sem mudanças. Já usava o `Agent` e registava o preço.

#### Linhas do CHANGELOG (o que muda nas propostas)

- **Changed (nova):** "`brave_search` refuses a `country` Brave does not list (`UK`: Brave's code
  is `GB`) with a `validation_error` that lists the codes, before any request; Brave's 422 names
  the fields its `error.meta` lists, and every argument to check."
- **Changed (nova):** "After a 429 without `Retry-After`, Brave's host rests 1 s (its 1-second
  window) and Tavily's 60 s (its per-minute limits) (D53)."
- **Fixed (nova):** "A Brave page with no web results while Brave has more says the call that
  reads the next page."
- **Docs (se o coordenador as quiser):** "the web-search example in `docs/tools.md` runs a ReAct
  agent (it returned the model's tool call, never run)", e "`docs/safety.md` defines the approval
  handler its example uses".

#### Verificação

- `uv run pytest -q` de `test_web_search.py` e dos ficheiros da T09b, mais
  `test_tool_contract.py`, `test_tool_invariants.py` e `test_tools_exports.py`: **1054 passed, 2
  xfailed**. Os dois `xfail` são os do L8.
- `tests/test_architecture.py` e `tests/test_quality_budget.py`: 34 passed.
- `uv run pyright src`: 0 erros.
- `ruff check` e `ruff format --check` dos ficheiros tocados: limpos (o `contract_cases.py` só
  com `ruff check`).
