# A07 · A pesquisa na web: `brave_search` e `tavily_search`, com o custo no meter (G-13)

- **Dono:** Claude (2026-10-05) · **Estado:** done · **Depende de:** nada
- **Origem:** o briefing do ai-network, grupo 7 (G-13, a D-51 dele) · **Decisões:** D55, D56 ·
  **Regras:** `R00-rules.md`

## Problema (verificado a 2026-10-05 no `3ebbc6a`)

- Não há pesquisa web do lado do cliente. Há pesquisas por domínio, o `http_get` e o
  `scrape_text` (em `dangerous`), e a server tool `web_search()`, que não serve modelos locais e
  cujo custo o meter não conhece.
- O meter conta toda a tool como gratuita (`_meter_tool_open`), a não ser com um pricer próprio.

## Nota de desenho (antes do código)

- **Core:**
  - `ToolPricing(per_unit, until, then)` e a secção `[tools]` da tabela no `_pricing.py`, com
    `register_tool`, `unregister_tool`, `get_tool(name, on=)` e `list_tools`;
  - o `price()` de uma operação de tool dá o preço de uma unidade, ou zero fora da tabela. O
    `worst_case` reserva-a sem mudança;
  - `core/_tools/_billing.py` (novo): o executor abre um registo por chamada (um `ContextVar`,
    que segue para a thread de uma tool síncrona), e `bill(key, units)` acrescenta-lhe as
    unidades;
  - o `_finished` cobra a soma das unidades ao preço da tabela; com um pricer próprio, fica como
    estava.
- **Toolkit:**
  - o `Api` ganha `billed_as` (a entrada da tabela), `bill_units` (lê do corpo as unidades
    gastas; senão, 1) e `key_prefix` (o `Bearer ` da Tavily). O `_send` regista um pedido aceite
    (estado abaixo de 400 e corpo inteiro);
  - `toolkit/tools/_web_search.py` (novo): `brave_search` (GET, `X-Subscription-Token`, até 20
    resultados) e `tavily_search` (POST, `search_depth: basic`, `Authorization: Bearer`, e a
    resposta da Tavily se for pedida). Ambas com `capability="network"`, no catálogo seguro;
  - os preços na tabela, com os URLs: Brave $0,005 por pedido, Tavily $0,008 por crédito.
- **Provas:**
  - a tabela: o preço de uma tool lê-se pelo dia, carrega do TOML e muda com `register_tool`;
  - o meter: uma tool que regista duas unidades custa duas; uma que não regista nada custa zero;
    um budget estrito abaixo de uma unidade recusa antes de correr; uma tool que levanta não
    custa nada;
  - a porta HTTP regista um pedido aceite com as suas unidades, e não regista um 4xx, um 429 ou
    um erro de rede;
  - as tools: o pedido certo a cada serviço e o texto dos resultados; sem chave, a frase e
    nenhum pedido; o 401 e o 429 explicados;
  - de ponta a ponta: num `MeterScope`, uma `brave_search` aceite custa $0,005 e uma sem chave
    $0.
- **Ao vivo:** só com as chaves do dono, se as tiver (são gratuitas até ao limite do mês).

## Ficheiros

- `src/ai_arch_toolkit/core/_pricing.py`, `core/_default_pricing.toml`, `core/__init__.py`,
  `core/_tools/_billing.py` (novo), `core/_tools/_executor.py`
- `src/ai_arch_toolkit/toolkit/tools/_http.py`, `_web_search.py` (novo), `toolkit/tools/__init__.py`
- os testes de pricing, do executor, da porta HTTP e das tools; os novos
- `docs/tools-catalog.md`, `docs/tools.md`, `docs/pricing.md`, `.env.example`, `AGENTS.md`,
  `CHANGELOG.md`

## Registo do dono

- Estado: done (Claude, 2026-10-05), pela nota de desenho. O dono escolheu as duas tools e o
  preço na tabela (D55, D56).
- **Core:**
  - `ToolPricing`, exportado, com a secção `[tools]` da tabela (`register_tool`,
    `unregister_tool`, `get_tool`, `list_tools`; o `reset` limpa-a);
  - o `price()` de uma operação de tool dá uma unidade, que é o que o `worst_case` reserva;
  - `core/_tools/_billing.py` (`bill`, `billing`);
  - o executor abre o registo à volta da chamada, e o `_finished` e o `_failed` cobram as
    unidades registadas. Com um pricer próprio, a chamada fica como estava.
- **Toolkit:**
  - o `Api` ganhou `billed_as`, `bill_units`, `key_prefix` e `key_required`. Sem a chave
    obrigatória, o pedido nem sai;
  - `_web_search.py`: o `brave_search` (o markup que a Brave põe nos excertos sai) e o
    `tavily_search` (`search_depth: basic`, `include_usage`, a resposta se for pedida); ambas no
    catálogo seguro, com `capability="network"`;
  - na tabela: `brave_search` $0,005 por pedido e `tavily_search` $0,008 por crédito, com as
    fontes.
- **Testes novos (35):**
  - `tests/test_tool_billing.py` (7): duas unidades custam duas; nenhuma custa zero; o caminho
    async; a thread de uma tool síncrona; uma tool que falha depois do pedido paga-o; um budget
    estrito abaixo de uma unidade recusa antes de correr; e sem meter;
  - `tests/test_pricing.py::TestToolPrices` (5);
  - `tests/toolkit/test_http.py::TestBilling` (8, com três estados recusados e o prefixo da
    chave);
  - `tests/toolkit/test_web_search.py` (12 de tool, 3 do meter de ponta a ponta).
- **Fica por verificar ao vivo:** o `.env` não tem `BRAVE_SEARCH_API_KEY` nem `TAVILY_API_KEY`.
  Com elas (o plano grátis chega), basta uma chamada de cada:
  `uv run python -c "from ai_arch_toolkit.toolkit.tools import brave_search, tavily_search; print(brave_search('lisbon weather')); print(tavily_search('lisbon weather'))"`.
- **Limite conhecido:** se o executor desistir de uma tool síncrona por tempo, um pedido que a
  thread ainda acabe depois não é cobrado (conta a menos, num caso raro).
- Gate: 6396 passed, 42 skipped; ruff, formatação e pyright limpos.
- **Linha para o Registo do briefing do ai-network** (o dono leva-a): "o grupo 7 fechou:
  - há duas tools de pesquisa na web do lado do toolkit: `brave_search` (com
    `BRAVE_SEARCH_API_KEY`) e `tavily_search` (com `TAVILY_API_KEY`), em
    `ai_arch_toolkit.toolkit.tools`. Servem qualquer modelo, também os locais;
  - o meter conta o custo delas pela secção `[tools]` da tabela de preços: Brave $0,005 por
    pesquisa, Tavily $0,008 por crédito, que se mudam com `pricing.register_tool` para o plano de
    cada um. Um pedido recusado ou sem chave não custa nada;
  - a server tool `web_search()` continua sem custo conhecido (D-44 da app).

  Na app, falta oferecer a que tiver chave."
