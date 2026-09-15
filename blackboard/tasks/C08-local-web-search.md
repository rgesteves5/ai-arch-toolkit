# C08 · Pesquisa web local (Brave e Tavily)

- **Dono:** por atribuir · **Estado:** todo · **Depende de:** nada (C05 é o par server-side)
- **Origem:** L13 (`docs/internal/agentes-app-toolkit-review.md:424-446`, `:586-588`);
  `toolkit/tools/CANDIDATE_TOOLS.md:134-142` (DuckDuckGo), `:234-284` (Brave #22, Tavily #23)
- **Decisões:** por fixar (ver abaixo)

## Problema

- Só há `core.web_search()`, `ServerTool` do fornecedor (`core/_server_tools.py:9-28`, `config`
  ignorado `:16-19`): xAI levanta (`_xai.py:410-416`), o OpenAI manda `{"type": "web_search"}` a
  qualquer `base_url` (`_openai.py:492-493`), o custo fica `unknown` (`core/_pricing.py:239-240`).
  Nada client-side em `toolkit.tools` nem `dangerous` (`tools/__init__.py:163-289`,
  `dangerous.py:15-23`); nenhuma tool recebe chave (grep `api_key|getenv|Authorization`: zero).
- `_web.py:17-22` usa `Request(headers=…)`. Verificado em loopback (Python 3.13.7): um 302 para outro
  host recebe `X-Subscription-Token`/`Authorization` postos assim; com `add_unredirected_header`, não.
- O metering de tools leva só `metadata={"tool": nome}`, preça antes da chamada e liquida qualquer
  retorno (`core/_tools/_executor.py:233-252`, `:350-358`). Verificado: um `Pricer` da app (0,016 USD
  por chamada) entra no `MeterScope` e `max_cost=0.03` nega a 3.ª chamada, mas a que devolveu "HTTP
  429" foi cobrada; sob `reserve="strict"` nada se reserva para tools (`budget/_estimator.py:42-43`):
  tecto 0,02 e 3 chamadas paralelas → 0,048 USD, contra `docs/safety.md:282`.

## Objectivo

Tools de pesquisa web client-side e stdlib-only para qualquer modelo (locais incluídos), no canal de
tools das estratégias standard: chave explícita da app, aprovação obrigatória, saída limitada e
normalizada, custo levado ao `MeterScope` pelo `Pricer` da app, com tecto rígido sob `strict`.

## API proposta

```python
from ai_arch_toolkit.toolkit.tools import brave_web_search_tool, tavily_search_tool
brave_web_search_tool(api_key, *, name="brave_web_search", country="", search_lang="",
                      safesearch="moderate", timeout=15.0, max_chars=8000) -> Callable[..., str]
tavily_search_tool(api_key, *, name="tavily_search", search_depth="basic", topic="general",
                   include_domains=(), exclude_domains=(), timeout=15.0, max_chars=8000)
# schema do modelo, igual nas duas: (query: str, max_results: int = 5,
#                                    time_range: Literal["", "day", "week", "month", "year"] = "")
```

- **Chave:** vazia → `ValueError`; ambiente nunca lido; host fixo; `add_unredirected_header`. Fica na
  closure, fora de schema, argumentos, `ApprovalRequest`, audit e texto; o texto do fornecedor passa
  por `replace(api_key, "[REDACTED]")` (o redactor não apanha `tvly-…` nem chaves Brave soltas).
- **Pedido:** Brave `GET https://api.search.brave.com/res/v1/web/search` (`q`, `count`, `freshness`
  `pd|pw|pm|py`, `result_filter=web`, `text_decorations=false`, `safesearch`, país, língua); Tavily
  `POST https://api.tavily.com/search` (`query`, `max_results`, `search_depth`, `topic`, `time_range`,
  domínios, `include_published_date: true`, `auto_parameters: false`). Sem `Accept-Encoding: gzip`.
- **Saída:** `Results from <Brave Search|Tavily> for '<query>' (untrusted third-party content):` e por
  hit título, URL, data, snippet; sem HTML nem controlo, espaços colapsados (um snippet não forja
  hits), só `http(s)`, título ≤ 200 e snippet ≤ 500 caracteres, total ≤ `max_chars`, 1–10 hits.
- **Erros → string** (`"Brave web search failed: …"`): 401/403; 400/422 com `detail`; 429 com a
  espera (`X-RateLimit-Reset` e `RATE_LIMITED`/`QUOTA_LIMITED`; `retry-after`); Tavily 432/433;
  timeout, `URLError`, `OSError`, `http.client.HTTPException`; JSON inválido; zero hits.
- **Governança e metering:** `@tool(name=name, capability="web_search", risk_level="medium",
  requires_approval=True, approval_reason=…)`, num `ToolGroup` com o handler da app. Custo 0 por
  omissão; a app preça por `request.metadata["tool"]` e delega o resto em `pricing` (receita nas
  docs). Só `search_depth` muda o preço e é do factory (argumento extra → `validation_error`).
- **Docs oficiais (2026-09-15).** Brave `api-dashboard.search.brave.com/api-reference/web/search/get`:
  `x-subscription-token`, `q` ≤ 600 car./75 palavras, `count` 1–20, `text_decorations` true por
  omissão, `web.results[]`, `ErrorResponse.error.code`; `…/documentation/guides/rate-limiting`:
  `X-RateLimit-*` por segundo e por mês, 429; `…/documentation/pricing`: 5 USD/1000 pedidos, 5 USD
  grátis/mês, 50/s; `brave.com/search/api`: guardar resultados exige plano próprio. Tavily
  `docs.tavily.com/documentation/api-reference/endpoint/search`: `Authorization: Bearer tvly-…`,
  advanced 2 créditos e os outros 1, `max_results` 0–20, `auto_parameters` pode passar a advanced,
  `results[]` com `published_date`, erros 400/401/429/432/433 em `{"detail": {"error": …}}`;
  `…/rate-limits`: 100/1000 RPM, `retry-after`; `…/api-credits`: 1000 créditos/mês, 0,008 USD PAYG.

## Decisões a fixar antes de codificar

1. **Tool por fornecedor ou genérica** — (a) factory por fornecedor, schema e saída comuns; (b)
   `web_search_tool(provider=…)` com registo; (c) tools de módulo com chave do ambiente. Recomendo (a),
   porque não cria abstracção, mantém os erros de cada API e dá nomes distintos no trace, na aprovação
   e no `Pricer`; trocar de fornecedor não muda prompts.
2. **Chave** — (a) só explícita; (b) fallback `BRAVE_SEARCH_API_KEY`/`TAVILY_API_KEY`. Recomendo (a),
   porque a chave vem do cofre da app (L12), um fallback gastaria uma chave de ambiente sem ninguém
   pedir, e host fixo com cabeçalho não reenviado já dá a garantia de `_resolve_key`
   (`core/_providers/__init__.py:103`): a chave só chega ao seu host.
3. **Governança** — `network` (como `http_get`, D4) ou `web_search`; `toolkit.tools` ou `dangerous`.
   Recomendo `web_search`, `medium`, aprovação, em `toolkit.tools`, porque o risco não é SSRF (o host
   é fixo) mas privacidade (a query leva o que o contexto lá puser) e custo; só o `ApprovalRequest` lê
   `capability` (`_approval.py:125`), a app separa pesquisa de fetch, e a aprovação segue D4.
4. **Conteúdo e falhas** — Recomendo snippets curtos, rótulo "untrusted" em vez de instruções, sem
   conteúdo bruto (`scrape_text` lê páginas), sem retries nem throttle, porque cada tentativa é
   facturada, a concorrência é da app e a defesa contra injecção é aprovar as tools com efeitos.
5. **Metering** — (a) sem hook no core, `Pricer` por nome; (b) preço declarado em `ToolRuntimePolicy`
   até ao `OperationRequest`; (c) preço no settle com `usage.credits`, falhas grátis. Recomendo (a) +
   C08c, porque (a) já funciona, C08c fica em `toolkit/budget` ("o pricer da estimativa é o do
   settle"), e (b)/(c) mexem no executor por um ganho que só compensa com C03. A documentar: uma
   chamada que devolve erro em string é cobrada como feita (erro do lado seguro do tecto).
6. **Nomes** — Recomendo tools `brave_web_search`/`tavily_search`, factories com `_tool`, nunca
   `web_search`, porque `ai_arch_toolkit.web_search` é a `ServerTool` e a server tool da Anthropic se
   chama `web_search` no wire; `docs/tools.md` contrasta as duas.

## Sub-tarefas, por ordem

- **C08a** `_web_search.py`: núcleo (`_SearchHit` frozen/slots/kw_only, limpeza, formato, limites,
  pedido, erros) e `brave_web_search_tool`. **C08b** `tavily_search_tool` no mesmo núcleo.
- **C08c** `HeuristicEstimator` reserva `pricer.price(request, Usage())` para `kind="tool"` sob
  `strict`; preço desconhecido ou pricer que levanta → nega.
- **C08d** Exports; docs (receita do `Pricer`, `safety.md`, contraste em `tools.md`); `CANDIDATE_TOOLS.md`
  (#22/#23 em parte feitos); exemplo com modelo local.

## Ficheiros

- `src/ai_arch_toolkit/toolkit/tools/_web_search.py` (novo), `toolkit/tools/__init__.py`,
  `toolkit/tools/CANDIDATE_TOOLS.md`, `toolkit/budget/_estimator.py`
- `tests/toolkit/test_web_search.py` (novo), `tests/toolkit/test_tools_exports.py` (partilhado com
  C07), `tests/budget/test_budget.py`, `tests/integration/test_web_search_live.py` (novo)
- `docs/tools.md` (partilhado com C02–C05, C07), `docs/tools-catalog.md` (partilhado com C07),
  `docs/safety.md` (partilhado com C02, C04, C07), `docs/agents-and-capabilities.md`;
  `examples/<n>_local_web_search.py`, `examples/README.md`, `docs/examples.md` (o número é o próximo
  livre, atribuído pelo coordenador ao aplicar)

## Prova

Herméticos, `urllib.request.urlopen` mockado em `…tools._web_search`; hoje falham no import.
- Sem handler: `approval_denied` e `urlopen` nunca chamado; política afirmada campo a campo.
- Brave: `time_range="week"` → `freshness=pw`, `count` ≤ 10; chave só em `req.unredirected_hdrs`
  (nunca na URL, em `req.headers`, no texto, no audit); em loopback (`http.server`, URL do módulo
  trocada) o 302 para outro host chega sem ela. Tavily: `search_depth` do factory no corpo.
- Fixtures das duas APIs → mesmo formato; snippet com `\n2. …` não cria hit; `javascript:` fora.
- Nunca levanta: 401, 422, 429 (espera no texto), `QUOTA_LIMITED`, 432, 433, timeout, `URLError`,
  JSON inválido, zero hits; chave na mensagem → `[REDACTED]`; `api_key=""` → `ValueError` mesmo com
  `TAVILY_API_KEY` no ambiente.
- `Agent` react com `LLM` mockado a pedir `tavily_search` → o resultado chega à 2.ª chamada.
- C08c: `budget_scope(BudgetPolicy(max_cost=0.02, reserve="strict"), pricer=P)` a 0,016 USD por
  chamada, 3 paralelas → hoje 3 execuções e 0,048; depois 1 execução e 2 `BudgetExceeded`.
- `integration` + `live_api`, só local, skip sem chaves: uma pesquisa real em cada, ≥ 1 hit.

## Fora do âmbito

Brave LLM Context/news/imagens/local; Tavily extract/crawl/map/research, `include_answer`,
`include_raw_content`; Serper; DuckDuckGo Instant Answer e scraping de motores; chave do ambiente,
`base_url`, cofre; `web_search()` e `server_tools=` (C05); opções 5b/5c; resultado estruturado, cache.

## Riscos

- Injecção pelos snippets e exfiltração pela query: a limpeza não chega; a app aprova (o
  `ApprovalRequest` mostra a query) e governa as tools com efeitos.
- Custo: erros contam (facturação de falhas não documentada); sem `strict`, chamadas paralelas passam
  o tecto; cancelar o flow não pára a thread nem o pedido (`_executor.py:108-113`).
- Deriva: a Tavily acrescenta parâmetros; as docs antigas da Brave dão 403. Guardar resultados Brave
  exige plano próprio, e a app persiste conversas.

## Registo do dono

- Estado: todo
- Ficheiros tocados:
- Testes novos:
- Verificações:
- CHANGELOG proposto:
- Desvios ao plano:
