# R03 · Fase 3 — Tools, motor de flows e dívida de manutenção

- **Dono:** Claude · **Estado:** done (commitada e publicada em `main` a pedido do dono) · **Depende de:** R01 e R02 · **Regras:** `R00-rules.md`
- **Plano:** `docs/internal/hardening-plan.md` — causas 5 e 6, secções 9 e 10
- **Achados (`FINDINGS.md`):** "Tools: nenhum limite central", "`mediawiki_*`…", "`ip_lookup`…",
  "`math_eval("9**9**9")` não termina", "Leituras sem limite em linhas longas", "`list_directory`
  levanta…", "Imutabilidade só à superfície", "`step_end` em falta…", e a dívida confirmada das
  revisões externas (2026-09-17)
- **Decisões em vigor:** D4, D7, D17, D19, D20

## Objectivo

As tools passam a ser seguras por construção e não por convenção: um só ponto de saída HTTP, limites
de saída e de tempo impostos pelo executor a qualquer tool, capacidades declaradas e verificadas, e
invariantes executáveis que apanham a próxima tool mal feita. O motor de flows perde os caminhos
duplicados. E sai a dívida de manutenção que duas revisões externas confirmaram.

## Passos, por ordem

### 1. Tools — infra-estrutura comum

- **`toolkit/tools/_http.py`, o único sítio que toca em `urllib`, `http.client` ou `socket`:** só
  `https` (excepções explícitas e justificadas), host numa lista dada pelo módulo chamador, sem
  redirects para outro host, leitura com `max_bytes`, prazo total, segmentos de caminho com
  `quote(safe="")`, erros de rede, de charset e de forma do JSON devolvidos como string. Migra os 47
  sítios (35 helpers privados e 11 `urlopen` inline) nesta fase; não fica lista de legado.
  `prototypes/…/tool-invariants/hosts_by_tool.json` é a semente da lista de hosts.
- **Limites no executor governado (D17), para qualquer tool — do toolkit, de quem usa a framework,
  dinâmicas ou de MCP:** `ToolRuntimePolicy.max_output_chars` (por omissão 200 000) e `timeout_s` (por
  omissão 120; `None` desliga), com override em `@tool(...)` e no `ToolGroup`. O corte fica marcado no
  texto e em `ToolResult.metadata`; o timeout dá `ToolResult.failure("timeout")`. Uma tool síncrona
  numa thread não se mata: o executor deixa de esperar, e as tools de cálculo ganham guardas à entrada
  (`math_eval`: tamanho de expoentes e de resultados; `regex_search`: tamanho do padrão e do texto).
  Grampos nos parâmetros de tamanho (`max_chars=-1` e afins).
- **Metadados verdadeiros:** toda a tool declara `capability` (`network`, `filesystem`, `compute`, …).
  Um teste por AST confirma que o declarado bate com o que o código alcança
  (`prototypes/…/tool-invariants/ast_survey.py`). Tools com efeitos ou acesso local ficam em
  `dangerous` com os metadados de D4. `youtube_*` ficam como excepção declarada (extra `youtube`) mas
  cumprem os mesmos invariantes (nunca levantam, saída limitada).
- **Invariantes executáveis** em `tests/toolkit/test_tool_invariants.py`, com descoberta automática
  por `pkgutil` e `__tool_definition__`: partição de exports; capacidades; host fixo (argumentos
  hostis não mudam esquema, host nem caminho); nunca levanta, com argumentos hostis e com corpos de
  resposta hostis (sockets bloqueados, `time.sleep` substituído, `timeout(10)`); saída dentro do
  limite. `prototypes/…/tool-invariants/harness*.py` é o ponto de partida. No fim da fase não há
  `xfail`: as 12 tools que levantam com argumentos hostis e as 90 que levantam com corpos inesperados
  ficam corrigidas pela migração para `_http.py`.

### 2. Motor de flows

- `_run_dag` (`toolkit/flow/_executor.py`, a função mais complexa do motor): uma vaga de um step é
  uma vaga; fica um só caminho, com os checks de budget, prazo e erro de orquestração num só sítio e um
  único ponto de emissão de `step_end`.
- `_run_attempts` (`core/_step_engine.py`): o bloco de fallback escrito várias vezes passa a um.
- Verificador de gramática de eventos, reutilizado pelos testes do motor: cada `step_start` fecha com
  exactamente um evento terminal e o stream concorda com o trace, em todos os modos e causas de paragem.

### 3. Contrato de imutabilidade (D20)

`State.snapshot()` documenta-se como vista só de leitura; o mesmo step não pode comportar-se de forma
diferente em vaga paralela e em modo sequencial sem que isso esteja escrito e testado.
`ReasoningSpec.knobs` e `llm_kwargs` congelam à entrada (`MappingProxyType`). Sem cópias profundas por
step: trariam de volta o custo quadrático medido em N7.

### 4. Dívida de manutenção (secção 10 do plano), por esta ordem

1. `toolkit/memory/graph/_store.py` deixa de reimplementar a fachada de `core/graph/_store.py` (20
   métodos com o mesmo nome, 54 janelas duplicadas): compõe ou herda, e fica só com o que é da
   memória (embeddings, índice, ciclo de vida). Decide com uma nota de desenho; a API pública de
   `GraphStore` não muda.
2. Flow factories e builders: cada knob comum declara-se uma vez (hoje repete-se nas nove factories,
   dez vezes em `_builders.py`, no schema do manifesto e nas docs). `budget_policy` nas factories:
   decide se fica como caminho documentado ou sai, e regista.
3. Chaves de estado: as strings que as estratégias e o runner partilham (`"task"`, `"response"`,
   `"answer"`, `"last_answer"`, …) passam a constantes tipadas num só módulo, e `extract_text` deixa
   de adivinhar por quedas sucessivas.
4. Validadores de manifesto: os dois escritos à mão (agentes e prompts) e o JSON Schema à parte
   passam a ter uma só fonte de verdade por manifesto.
5. `_eval_expr` (`toolkit/tools/_python.py`): tabela de despacho por tipo de nó, com o conjunto
   permitido enumerável e um teste por nó permitido e por nó recusado.
6. O padrão "criar um ReAct interno, correr, extrair a resposta, inspeccionar o trace" repetido em
   seis estratégias passa a uma primitiva.
7. `__all__` de topo com 198 nomes: só um levantamento para o dono decidir antes da 1.0; não mudes
   exports.

### 5. Fecho da frente

Deriva de documentação (`AGENTS.md`, `CONTRIBUTING.md`, `docs/`), secção "Upgrade notes" no
`CHANGELOG.md` com todas as quebras das três fases, linha de base de complexidade actualizada, e o
`BOARD.md` a devolver a vez à frente C.

## Aceitação

- Regras de arquitectura verdes: `urllib`, `http.client` e `socket` só em `toolkit/tools/_http.py`;
  capacidades declaradas iguais às alcançadas; nenhuma tool com acesso local fora de `dangerous`.
- `tests/toolkit/test_tool_invariants.py` verde sem `xfail`; um resultado de 5 MB sai cortado ao
  limite; `math_eval("9**9**9")` e o regex com backtracking devolvem erro dentro do prazo.
- `_run_dag` e `_run_attempts` dentro do orçamento de complexidade; gramática de eventos verde.
- Duplicação entre os dois stores de grafo eliminada; `_eval_expr` dentro do orçamento.
- Saldo de linhas negativo em `toolkit/tools/` e nos módulos da dívida.

## Fora do âmbito

Tools novas (escrita tipada `C07`, pesquisa web `C08`), MCP (`C03`), tools dinâmicas (`C02`),
checkpoint (`C04`), `FlowSpec` (`C09`); mudar a API pública de `Flow`, `Agent` ou `GraphStore`.

## Registo do dono

- Estado: done (2026-09-18, Claude); commitada e publicada em `main` a pedido do dono.
- Notas de desenho: uma por passo, abaixo. Decisões D28 a D36.
- Ficheiros tocados: 160 caminhos (140 em `src/` e `tests/`, 20 de docs e registo); a lista está
  no `git status`.
- Testes novos e corrigidos: "Números" no relatório final; as mensagens mudadas no passo 4.4.
- Verificações: gate final 5465 passed, 42 deselected; ruff, formatação, pyright e lock limpos.
- Saldo de linhas e complexidade: tabela no relatório final; dívida 121 → 46.
- CHANGELOG: Added, Changed e Fixed por passo, e "Upgrade notes" no fecho.
- Bloqueios: nenhum.
- Desvios ao plano:
  - 4.1 foi herança de uma base partilhada, não composição;
  - 4.4 deixou saldo positivo;
  - 4.7 contou 204 nomes, não 198;
  - a infra de testes foi corrigida em passagem (passo 4.5).
- Commits: os dois propostos no relatório final, feitos a pedido do dono (`6ceb516` e o registo).

### Início (2026-09-18)

Linha de base no checkout principal (`main` @ `4a1fa4e`): 4341 passed, 42 deselected; ruff,
formatação, pyright e lock limpos. Linhas: `toolkit/tools` 13 936, `core/_tools` 2073,
`tests/toolkit` 5459. Dívida de complexidade: 121 entradas (69 em `toolkit/tools`).

Levantamento com os protótipos (`tool-invariants/`, copiados para o scratchpad): 130 tools; 93
levantam com corpos hostis (quase todas porque o JSON do topo não é um objecto; 20 por
`UnicodeDecodeError`; 8 por um `null` a meio), 11 levantam com argumentos hostis, e duas não terminam
(`math_eval("9**9**9")`, `regex_search` com `(a+)+$`). 47 chamadas a `urlopen` em 36 módulos; cinco
módulos com um throttle próprio (o Nominatim é chamado de dois módulos com dois relógios, um deles sem
throttle); 22 cópias de `_http_error`; a escada de cinco `except` repetida em cerca de 100 sítios.

**Facto novo que muda o desenho.** Um regex catastrófico ou um inteiro enorme numa thread prendem o
GIL dentro de código C: o event loop pára. Medido (`scratchpad/gil_probe.py`): com
`asyncio.wait_for(asyncio.to_thread(...), 1.0)`, `re.search("(a+)+$", "a"*32+"!")` congelou o loop
60 s sem um único tick (processo morto), e `9**9**7` segurou-o 3 s (o timeout de 1 s só disparou no
fim). O timeout do executor protege das tools que esperam por E/S ou correm bytecode; não protege
de uma única chamada C longa. Para essas só há guardas à entrada.

### Passo 1 · Nota de desenho (tools: infra-estrutura comum)

**1a. `toolkit/tools/_http.py`, a única porta para a rede.** Único módulo do pacote que importa
`urllib.request`, `urllib.error`, `http.client`, `socket` ou `ssl` (regra por AST; o `urllib.parse`
fora das tools é parsing puro e fica).

```python
class HttpError(Exception):
    """A request that produced no usable body; str() is the reason shown to the model."""
    status: int | None   # HTTP status, None when no response arrived
    body: str            # start of an error response, for APIs that explain errors there

@dataclass(frozen=True, slots=True, kw_only=True)
class Api:
    """One upstream HTTPS API. Requests go to its host and under its base path, nowhere else."""
    base: str                    # "https://api.crossref.org/works", validated at construction
    name: str                    # "Crossref", for the rate-limit message
    timeout_s: float = 10.0      # deadline of one request, checked between reads
    max_bytes: int = 10_000_000
    min_interval_s: float = 0.0  # one clock per (host, interval), shared across modules
    params: Mapping[str, str] = {}    # sent with every request
    segment_safe: str = ""       # kept raw in path segments (World Bank ";", Semantic Scholar ":")
    query_safe: str = ""         # kept raw in the query (WHO OData)

    def get_json(self, *segments: str, params=None, expect: type = dict) -> Any: ...
    def get_text(self, *segments: str, params=None) -> str: ...
    def post_json(self, *segments: str, payload, expect: type = dict) -> Any: ...
    def post_form(self, *segments: str, form, expect: type = dict) -> Any: ...

def fetch_page(url: str, *, max_bytes: int, timeout_s: float = 10.0) -> Page: ...
```

- Só `https`. A única excepção é `fetch_page`, das tools `dangerous` `http_get`/`scrape_text`, que
  aceita `http://` porque o URL foi aprovado por uma pessoa. Nenhum redirect muda de host nem desce a
  `http`: o redirect para outro host é um `HttpError` que dá o destino (o modelo pode pedir esse URL,
  com nova aprovação). Os segmentos vão com `quote(safe=segment_safe)`; `""`, `"."` e `".."` são
  recusados antes de qualquer pedido. Um `Api` cujo host vem do modelo (MediaWiki) constrói-se com
  `Api.within(url, domains, name=...)`, que recusa o que não está na lista de domínios do módulo.
- A leitura é por blocos até `max_bytes`; a mais dá `HttpError`. Rede (`URLError`, `OSError`,
  `TimeoutError`, `http.client.HTTPException`), charset (`LookupError`), JSON inválido ou fundo
  demais (`ValueError`, `RecursionError`) e um topo do tipo errado (`expect`) dão `HttpError`. As
  mensagens são as de hoje: "HTTP error 500: Internal Server Error", "rate limited by ChEMBL (HTTP
  429). Try again later.", "URL error: …", "request timed out.", "could not parse API response: …".
- O throttle (`min_interval_s`) reserva a vaga sob um lock e dorme fora dele: seguro com tools em
  threads paralelas, o que os cinco throttles de hoje não são.
- **Desaparece:** os 35 helpers `_fetch_*`/`_post_*`, os 11 `urlopen` inline, os cinco `_throttle`,
  as 22 cópias de `_http_error` e `_read_error_body`, e a escada de `except` em cada tool (fica um
  `except HttpError`). A migração acaba neste passo, sem lista de legado.
- **Testes.** `tests/toolkit/test_http.py` com um `OpenerDirector` real e um transporte falso (sem
  sockets): redirect para outro host e para `http`, `max_bytes`, prazo, charset, JSON, `expect`,
  segmentos, throttle partilhado entre threads. Os testes das tools passam a simular um só ponto,
  `_http._open(request, timeout)`, que recebe o mesmo `Request` que o `urlopen` recebia; um fixture
  autouse em `tests/toolkit/conftest.py` bloqueia os sockets e o `sleep` do throttle em todos.
- **`ip_lookup` (D21 → D28).** Passa do `http://ip-api.com` para `https://ipwho.is/{ip}`. Confirmado
  na documentação oficial: `https://ip-api.com/docs/api:json` ("256-bit SSL encryption is not
  available for this free API"; "We do not allow commercial use of this endpoint");
  `https://ipwhois.io/documentation` (endpoint gratuito `https://ipwho.is/{IP}`, sem chave, "Commercial
  use allowed", 1000 pedidos/dia por IP; campos `success`, `message`, `ip`, `city`, `region`,
  `country`, `latitude`, `longitude`, `timezone.id`, `connection.isp`, `connection.org`). A saída da
  tool mantém as mesmas linhas. Por verificar ao vivo (comando no relatório final).

**1b. Limites no executor governado (D17), para qualquer tool.**

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class ToolRuntimePolicy:
    ...
    max_output_chars: int | None = 200_000   # None: no limit
    timeout_s: float | None = 120.0          # None: no deadline

def tool(..., max_output_chars: int | None = 200_000, timeout_s: float | None = 120.0)
class ToolGroup:
    def __init__(..., max_output_chars: int | None = None, timeout_s: float | None = None)
```

- No `ToolGroup` os dois valores são um tecto para todas as tools do grupo: vale o mais estrito dos
  dois (`None` no grupo quer dizer "sem tecto"). Assim a app aperta, nunca alarga o que a tool
  declarou (D29).
- **Corte:** o texto que o modelo recebe (`to_model_text()`) fica nos primeiros `max_output_chars`
  caracteres, seguidos de `[Output truncated: kept N of M characters.]`; `metadata["truncated"] =
  {"chars": M, "kept": N}`. Vale para sucessos e para a mensagem de um erro.
- **Timeout:** `ToolResult.failure("timeout", ...)` (retryable). Uma tool `async` é cancelada. Uma
  tool síncrona corre numa thread daemon própria, com o contexto copiado, e o executor deixa de
  esperar. Não se mata, mas também não prende o fim do processo, ao contrário do executor por omissão
  do `asyncio.to_thread`, que o `asyncio.run` espera até 300 s. O caminho síncrono (`execute`,
  `run_tools_sync`) passa a correr a tool nessa thread: é a única forma de lhe impor um prazo
  (mudança visível: recursos presos à thread de quem chama deixam de servir).
- **Uma só casa:** a invocação limitada (`_invoke`) e o corte (`_bounded`) servem os dois caminhos;
  os dois pipelines (`_run_tool_sync`, `_arun_tool`) só diferem nos gates `check_sync`/`check`.
- **Testes:** um resultado de 5 MB sai cortado ao limite e marcado; uma tool lenta dá `timeout` no
  prazo (async e sync, a sync sem esperar pela thread); o tecto do grupo aperta e não alarga;
  `@tool(timeout_s=None)` desliga; o meter liberta a operação num timeout.

**1c. Guardas nas tools de cálculo** (o facto novo acima: só elas protegem). `math_eval`: expressão
até 1000 caracteres; cada `**`, `pow`, `factorial` e `*` inteiro estima o tamanho do resultado antes
de o calcular e recusa acima de 15 000 bits (o `str()` de um inteiro já recusa mais de 4300
dígitos); `RecursionError` e `MemoryError` passam a erro. `regex_search`: padrão até 500 caracteres
e texto até 100 000; recusa as formas de backtracking exponencial (grupo repetido com quantificador
ou alternativa lá dentro, referências para trás); no máximo 1000 correspondências. Grampos nos
parâmetros de tamanho (os 11 sem grampo do `caps_static.py`: `max_chars=-1`, `max_lines`,
`max_rows`, `max_results`, `days`). `OSError` nas tools de ficheiros, `OverflowError` nas datas,
`RecursionError` no `json_extract`, padrões inválidos no `list_directory`, erros de rede das
`youtube_*` (o `requests` levanta `OSError`): todos passam a string de erro.

**1d. Metadados verdadeiros.** Cada tool do toolkit declara `capability`: `network` (111),
`compute` (13), e as `dangerous` como estão (`filesystem`, `network`, `python`, `shell`). O tipo
público continua `str | None` (as apps usam os seus próprios valores); o teste fixa os valores do
toolkit.

**1e. Invariantes executáveis** em `tests/toolkit/test_tool_invariants.py`, com descoberta por
`pkgutil` e `__tool_definition__`: (1) partição: cada tool definida num módulo das tools é exportada
por exactamente um dos dois namespaces, com o nome do schema; (2) capacidade declarada igual à
alcançada pelo grafo de chamadas por AST (`_http` ou um `Api` do módulo → `network`; `pathlib`/`os` →
`filesystem`; `subprocess` → `shell`; nada disso → `compute`, e `python` só para o `python_repl`), e
nada com `filesystem`, `shell` ou `python` fora de `dangerous`; (3) host fixo: com argumentos hostis,
cada pedido é `https`, vai a um host do módulo e fica sob a base do seu `Api`, sem segmentos `.` nem
`..`; (4) nunca levanta com argumentos hostis e com os 16 corpos hostis do protótipo (sockets
bloqueados, `sleep` substituído, `timeout(10)`); (5) saída dentro do limite pelo executor. Sem
`xfail`.

**Regras de arquitectura** (`tests/test_architecture.py`): `urllib.request`, `urllib.error`,
`http.client`, `socket` e `ssl` só em `toolkit/tools/_http.py` (com canário).

**Ordem de trabalho:** 1a com os seus testes → migração módulo a módulo (o teste do módulo muda de
ponto de simulação e fica verde antes do seguinte) → 1b → 1c → 1d → 1e. O gate completo corre no
fim do passo.

### Passo 1 · Resultado

- **1a · `_http.py`.** Feito como na nota, com um acrescento que a migração do Crossref mostrou
  necessário (D31): cada pedido recebe a função que lê a resposta (`parse=`) e o que ela levantar
  por forma inesperada vira `HttpError`. Sem isso, tirar a escada de `except` deixava o parse sem
  rede (o `except TypeError` antigo protegia-o). Os 47 sítios migraram todos: `urllib.request`,
  `urllib.error`, `http.client`, `socket` e `ssl` só em `_http.py` (regra por AST com canário,
  verde, sem lista de legado). O `Api.within` recebe `timeout_s`. Migração: quatro módulos de
  referência por mim (Crossref, ChEMBL, OSM, geo) e o `_web`; os outros 31 por cinco agentes em
  paralelo, com ficheiros disjuntos e um brief comum (`scratchpad/r03/BRIEF.md`). Revi os diffs;
  três agentes compararam o código antigo e o novo nos mesmos pedidos (185, 139 e as amostras do
  grupo A) e os URLs saíram iguais, salvo o que está em "Desvios".
- **1b · Limites no executor.** Como na nota. Uma só invocação limitada (`_invoke`): o caminho
  síncrono corre-a num loop seu, o que tirou o `_call_blocking` e o `_as_coroutine` (D30). Os dois
  pipelines partiram-se em passos partilhados (`_admit`/`_admit_sync`, `_apply`, `_bind`, `_spent`,
  `_finished`, `_failed`, `_bounded`) e saem da dívida (eram 14 e 14).
- **1c · Guardas.** `math_eval` e `regex_search` como na nota, mas o texto do regex fica em 20 000
  caracteres, não 100 000: medido, `\w+x` em 20 000 é 0,83 s e em 50 000 é 5,3 s, com o GIL preso
  (D32). O custo polinomial que sobra vai para `FINDINGS.md`. Grampos, `OSError`, `OverflowError`,
  `RecursionError`, padrões inválidos, byte nulo e rede das `youtube_*`: feitos. A verificação
  dinâmica dos grampos (-1 e 10⁹ com um corpo de 5 MB) apanhou o `wikipedia_article`
  (`max_chars=-1` → 2 000 012 caracteres), agora com grampo e teste.
- **1d · Metadados.** 111 tools `network`, 13 `compute`; as `dangerous` como estavam.
- **1e · Invariantes.** `tests/toolkit/test_tool_invariants.py`, 739 casos, sem `xfail`: partição,
  capacidade por AST, host fixo, nunca levanta (argumentos hostis com a rede a falhar no `_http` e
  com o opener real e sockets bloqueados; os 16 corpos do protótipo; e corpos construídos a partir
  das chaves que cada módulo lê, que apanham uma tool que leia a resposta fora do `parse`),
  saída limitada pelo executor, e seis cálculos que prendem o GIL, num subprocesso com prazo duro.
  O `run_command` não é chamado com argumentos hostis (correria comandos nesta máquina): é a única
  excepção, declarada no teste, e tem os seus testes próprios.
- **Testes novos:** `tests/toolkit/test_http.py` (60), `tests/test_tool_limits.py` (19), os 739
  invariantes, e casos novos em `test_geo.py`, `test_web.py`, `test_text.py`, `test_math.py`,
  `test_filesystem.py`, `test_json.py`, `test_datetime.py`, `test_shell.py`, `test_youtube.py`,
  `test_wikipedia.py`. Todos vistos a falhar primeiro, excepto o do relógio partilhado do
  Nominatim, escrito depois da migração do geo (antes, o `reverse_geocode` não tinha throttle).
- **Testes corrigidos:** os do `ip_lookup` passam à forma do ipwho.is (D28); o 429 do Wikidata
  afirma agora o texto uniforme ("rate limited by Wikidata Query Service (HTTP 429)"); o teste do
  nanope que fazia patch de `_web._fetch` passa a simular o `_http._open`, com as mesmas
  asserções. Os testes das tools mudaram de ponto de simulação (`_http._open` via
  `tests/toolkit/http_fakes.py`) e perderam os patches de `_throttle`; nenhuma asserção foi
  enfraquecida.
- **Verificações (fim do passo):** 5208 passed, 42 deselected; ruff, formatação, pyright e lock
  limpos.
- **Saldo de linhas:** `toolkit/tools` 13 936 → 13 666 (−270, com o `_http.py` novo de 439; os
  módulos das tools −709); `core/_tools` 2073 → 2277 (+204: timeouts, cortes e o pipeline em
  passos); `tests/toolkit` 5459 → 6361. Dívida de complexidade 121 → 73 (48 entradas saíram,
  nenhuma entrou).
- **Desvios:** o `parse=` (D31); o texto do regex em 20 000 (D32); o caminho síncrono numa thread
  (D30, mudança visível); agentes em paralelo na migração. No fio, fora do que as decisões dizem:
  onde o URL era montado à mão, espaços passam de `%20` a `+` (Wikipedia, tempo, geo); o
  User-Agent é um só; um 429 diz "rate limited by …" em todos os módulos; o `tool=` do PubMed
  passa a primeiro parâmetro; segmentos `.`/`..` são recusados (antes subiam no caminho em vários
  módulos).
- **Achados novos (`FINDINGS.md`):** risco polinomial do `regex_search`; `pdb_ligands` sem tecto de
  pedidos; intervalo do arXiv por respeitar; endpoint de pesquisa da UniProt por verificar.

### Passo 2 · Nota de desenho (motor de flows)

- **Uma vaga é uma vaga.** Um só executor de vagas, `_run_wave(wave)`: o modo sequencial corre cada
  step como uma vaga de um, o DAG corre cada conjunto pronto (um ou vários). Os checks ficam num só
  sítio, à entrada da vaga (orçamento de tempo, prazo, e o `Scope` e o `when` de cada step, cujo
  erro de orquestração regista o step e pára o run); o orçamento volta a ver-se no fim da vaga. O
  DAG guarda o seu estado de escalonamento num `_Graph` (espera, dependentes, acabados, falhados,
  saltados) e só decide o que fica pronto e o que se salta por dependência.
- **Um só sítio onde um step acaba:** `_ended(fs, result, trace)` regista o trace e o resultado e
  devolve o `step_end`. A vaga chama-o à medida que cada step acaba (o stream e o trace contam a
  mesma ordem), e o erro de orquestração também. O último step de uma vaga funde a vaga antes do
  seu `step_end`, pela ordem de lançamento: um step sozinho continua fundido quando se ouve o seu
  `step_end` (como hoje) e, numa vaga paralela, cada irmão anuncia-se quando acaba (como a
  documentação diz). Desaparecem o caminho rápido da vaga de um step, o `announce_end`, o segundo
  tratamento do timeout e o `fork()` de cada irmão.
- **Sem `fork()` (D33):** todas as vagas lêem o `state.snapshot()` tirado à entrada. As camadas são
  copiadas, por isso nenhum irmão vê os artefactos de outro (a fusão é no fim da vaga); os valores
  aninhados são partilhados, como já eram no modo sequencial. Hoje uma escrita in-place perde-se numa
  vaga paralela e fica no estado nos outros modos; passa a ficar sempre. O passo 3 documenta a vista
  só de leitura e testa que o mesmo step se comporta igual nos dois modos. Sem cópias profundas por
  step (R15).
- **`_run_attempts`:** um só `_fallback()` para os três usos (timeout, confiança baixa,
  esgotamento); o ciclo parte-se em `_attempt` (uma chamada: resultado, ou `None` no timeout) e
  decisões pequenas. A ordem das decisões registadas não muda: fixada antes por uma tabela de 20
  caminhos em `tests/test_step_engine.py`.
- **Testes:** verificador de gramática em `tests/flow/event_grammar.py`, com 16 cenários (sequencial,
  cíclico e DAG, cada um nas causas de paragem: fim, erro, `when`/`Scope` que levanta, tempo,
  negação do meter, timeout do run, e decisões dentro dos steps). Hoje falha um, os irmãos que
  acabam fora de ordem (o stream diz `fast, slow`, o trace diz `slow, fast`).

### Passo 2 · Resultado

- **Motor.** Como na nota. `_run_wave` é o único executor de vagas (o sequencial corre vagas de um,
  o DAG os conjuntos prontos, com o `_Graph`); `_ended` é o único sítio que regista o fim de um
  step e faz o `step_end`; o `_wait` assenta cada step quando acaba (e drena outra vez antes de
  esperar, para nenhum evento ficar preso depois dos `yield`). Saíram o caminho rápido da vaga de
  um step, o `announce_end`, o segundo tratamento do timeout, o `_step_end` solto e o `fork()` por
  irmão (D33). `_run_dag` (34/31/88) e `_run_sequential` (12) saem da dívida; nenhuma função do
  motor passa de 10.
- **`_run_attempts`** (22/21/70) sai da dívida: `_attempt`, `_retry`, `_fallback` (o único bloco de
  fallback, para os três usos), `_exhausted`, `_low_confidence`, `_within_cost`. A tabela de 20
  caminhos passou antes e depois, sem mudar uma decisão registada.
- **Testes novos:** `tests/flow/event_grammar.py` (verificador) e `tests/flow/test_event_grammar.py`
  (16 cenários e um canário do verificador); os testes do motor que recolhem eventos passam a
  verificar a gramática (`_checked_events`, 16 casos em `test_engine.py`); a tabela dos 20 caminhos e
  a precedência do fallback do step em `tests/test_step_engine.py`. Vermelho antes: os irmãos fora
  de ordem (stream `fast, slow`, trace `slow, fast`).
- **Verificações:** 5245 passed, 42 deselected; ruff, formatação, pyright e lock limpos.
- **Saldo de linhas:** `flow/_executor.py` 831 → 841 (+10: as classes `_Wave` e `_Graph`);
  `_step_engine.py` 246 → 272 (+26: as decisões em funções com nome). Dívida 73 → 66.
- **Documentação:** `docs/flow-architecture.md` (fusão da vaga, sem `fork()`); `CHANGELOG`
  (quebra das vagas paralelas, ordem do trace).

### Passo 3 · Nota de desenho e resultado (contrato de imutabilidade)

A nota ficou escrita junto com o resultado: o desenho é o da ficha e da D20 (R15), sem escolha nova
além da D33 do passo 2.

- **Snapshot:** o `StateSnapshot` e o `State.snapshot()` passam a dizer o que são: uma vista só de
  leitura, com as camadas copiadas e os valores partilhados; um step devolve o que muda como
  artefacto, e um valor mutado in-place muda o estado para todos os steps seguintes, irmãos
  incluídos. Com a D33 o comportamento já é o mesmo em todos os modos; o teste
  `test_a_step_behaves_the_same_in_a_parallel_wave_and_in_sequence` fixa-o (sequencial e vaga
  paralela), e `test_the_snapshot_cannot_be_written_through_and_misses_later_writes` fixa o resto do
  contrato. Sem cópias profundas por step.
- **`ReasoningSpec.knobs` e `llm_kwargs`:** copiados e congelados à entrada (`MappingProxyType`
  sobre uma cópia rasa), também pelo `from_mapping`; a igualdade por valor mantém-se. Consequência
  (no `CHANGELOG` e em `docs/agents.md`): um spec deixa de aceitar `deepcopy` e pickle; ninguém no
  repositório o fazia.
- **Secção 9, primeiro ponto (configuração estrita):** o `from_mapping` já recusava chaves e valores
  que não reconhece, com a chave no erro (feito antes da R03); nada a fazer.
- **Testes novos:** dois em `tests/agents/test_configurable.py` (vermelhos antes), dois em
  `tests/flow/test_engine.py` (verdes desde o passo 2, que trouxe a mudança).
- **Verificações:** 5250 passed, 42 deselected; ruff, formatação, pyright e lock limpos.
- **Documentação:** `docs/flow-architecture.md` (StateSnapshot), `docs/agents.md`, `CHANGELOG`.

### Passo 4.1 · Nota de desenho (os dois stores de grafo)

- **Medido:** 60 janelas de 8 linhas iguais entre `core/graph/_store.py` e
  `toolkit/memory/graph/_store.py`, em sete sítios: o índice por tipo (`update`, `list`, `count`),
  `connect`/`edges`/`disconnect`, `add_many`/`remove_many`, a serialização e a leitura das arestas, e
  `save`/`load`.
- **Compõe, não herda.** Herdar do `Graph` daria ao `GraphStore` uns sessenta métodos públicos (os
  algoritmos e os `_sync`) e estreitaria o tipo do parâmetro de `add` num override (o pyright
  recusa). O `GraphStore` passa a ter um `Graph` sobre o mesmo backend (o `Node` da memória é
  subclasse do do core e o `MemoryBackend` cumpre o `GraphBackend`) e delega nele o que não é da
  memória: adicionar, actualizar e remover no backend com o índice por tipo, listar e contar,
  limpar, e a superfície das arestas. Fica só com o que é da memória: o embedding no `add` e no
  `update`, o contador de acessos no `get`, o índice vectorial, a pesquisa, e os campos do seu
  `Node` na persistência.
- **Partes partilhadas:** a forma do JSON de um grafo guardado (arestas, versão, `nodes`/`edges`)
  passa a funções do módulo do core que os dois usam; `add_many`/`remove_many` a duas funções
  (`add_all`, `remove_all`) sobre o `add`/`remove` de cada fachada.
- **Tipos:** o `Graph` devolve `Node[Any]`; nos dois sítios em que o `GraphStore` precisa do seu
  `Node` (`list`, `update`) há um `cast` comentado: o backend é um `MemoryBackend`, logo cada nó
  que devolve é um `Node` da memória. A API pública do `GraphStore` não muda.
- **Custo:** o `clear` com índice vectorial lista os ids pelo backend antes de limpar (o índice por
  tipo passou a ser do `Graph`); o `update` deixa de ir buscar o nó antigo (o embedding antigo está
  no nó actualizado).
- **Prova:** teste de arquitectura que conta as janelas iguais entre os dois ficheiros (60 hoje;
  tem de dar 0) e os testes de memória existentes, que não mudam.

### Passo 4.1 · Resultado

- **Desvio à nota: herança de uma base partilhada, não composição.** Compor o `Graph` pedia que o
  `MemoryBackend` passasse por `GraphBackend`, e o pyright recusa-o: o `add_node` do backend da
  memória aceita só o `Node` da memória, mais estreito do que o do core (os parâmetros não se
  estreitam). Um `cast` escondia isso em vez de o resolver. Ficou uma base abstracta genérica,
  `GraphFacade[N]` (`core/graph/_facade.py`), sobre um protocolo `NodeStore[N]`: nós e índice por
  tipo, arestas, operações em lote, `to_dict`/`save`, tipados pelos nós de cada fachada. O `Graph` e
  o `GraphStore` herdam dela; nenhum herda o outro, por isso o `GraphStore` não ganha os algoritmos
  nem os `_sync` do `Graph`, e não há `cast` nenhum. A forma do JSON guardado passa a
  `check_payload` e `edges_from`, no mesmo módulo; cada fachada dá só a verificação e a leitura dos
  seus nós.
- **O que ficou no `GraphStore`:** o embedding no `add` e no `update` (`_reembed`), o contador de
  acessos no `get`, o índice vectorial no `remove` e no `clear`, a `search`, e os campos do seu `Node`
  na persistência. O `from_dict` põe os nós como foram guardados pelo `add` da base (sem embeddings
  novos).
- **Custo (correcção à nota):** igual ao de antes. O `clear` tira os ids ao índice por tipo antes de
  limpar, como fazia; o `update` continua com um só `get` do nó antigo (o `_update` da base devolve
  o antigo e o actualizado).
- **API pública:** os métodos públicos do `Graph` e do `GraphStore`, e os seus parâmetros, são os
  mesmos do HEAD (comparados por AST e `inspect`); as mensagens de erro do payload também.
- **Testes novos:** `test_the_memory_graph_store_does_not_reimplement_the_graph_facade` (60 janelas
  iguais antes; 0 agora) e o canário do contador em `tests/test_architecture.py`. Os testes de
  memória e de grafo não mudaram (206 passam).
- **Saldo de linhas:** 1084 → 974 (−110): `core/graph/_store.py` 635 → 436, `_facade.py` 275 novo,
  `toolkit/memory/graph/_store.py` 449 → 263. Dívida 66 → 61 (`_validate_graph_payload`,
  `_validate_memory_payload`, `GraphStore.update`).
- **CHANGELOG:** nada visível.

### Passo 4.2 · Nota de desenho (opções comuns das factories e dos builders)

- **Medido:** as nove factories declaram as quatro opções do `Flow` (`timeout`, `trace_capture`,
  `policy`, `budget_policy`) em três sítios cada uma: na assinatura, na docstring (em sete; o
  `plan_execute_flow` e o `llm_compiler_flow` não as documentam) e na chamada ao `Flow`. O
  `_builders.py` passa `timeout=s.timeout, trace_capture=s.trace_capture, policy=s.policy` dez
  vezes (as nove factories e o `completion`). Seis factories passam o `trace_capture` ao ReAct
  interno.
- **Desenho:** um `TypedDict` `FlowOptions` (`total=False`) em `flows/_common.py`, com as quatro
  chaves tipadas como no `Flow` e documentadas uma vez. Cada factory recebe-as em
  `**options: Unpack[FlowOptions]` (PEP 692) e passa-as tal e qual ao `Flow`. As chamadas não
  mudam: os mesmos nomes, os mesmos tipos (o pyright verifica-os), e uma chave desconhecida continua
  a dar `TypeError` (no `Flow`). O `FlowOptions` passa a ser exportado por
  `ai_arch_toolkit.toolkit.agents.flows`, porque está nas assinaturas públicas (quem embrulha uma
  factory precisa dele); o `__all__` de topo não muda.
- **Fluxo interno:** `nested(options)` diz num só sítio o que um ReAct interno herda do fluxo de
  fora: o `trace_capture`, para os filhos do trace ficarem registados como o pai. O prazo, o budget
  e a policy ficam com o fluxo de fora: o prazo e o budget já cobrem o interno (o scope é
  partilhado), e a policy é dos steps do de fora (`docs/agents.md` já o diz).
- **Builders:** `_flow_options(spec)` devolve as três opções que o spec tem (`timeout`,
  `trace_capture`, `policy`); cada builder passa `**_flow_options(s)`.
- **`system` e `llm_kwargs`** ficam explícitos em cada factory: não vão para o `Flow`, cada factory
  usa-os à sua maneira, e o `generate_review_flow` tem um par por fase.
- **`budget_policy` (D34):** fica, como uma das quatro opções que a factory passa ao `Flow` sem lhes
  tocar.
- **Manifesto e docs:** a validação de `timeout` e `trace_capture` no manifesto de agentes, que
  repete a do `ReasoningSpec`, é do 4.4 (uma só fonte de verdade por manifesto). As docs mostram
  chamadas às factories, que continuam válidas.
- **Prova:** um teste de arquitectura por AST, em que nenhuma factory declara uma das quatro opções
  como parâmetro seu nem as passa uma a uma ao `Flow`, e o `_builders.py` não lê
  `s.timeout`/`s.trace_capture`/`s.policy` fora de `_flow_options`. Mais um teste por factory de que
  as quatro opções chegam ao `Flow`, e de que o ReAct interno herda o `trace_capture`.

### Passo 4.2 · Resultado

- **Como na nota.** `FlowOptions` e `nested()` em `flows/_common.py`; as nove factories recebem as
  quatro opções em `**options: Unpack[FlowOptions]` e passam-nas ao `Flow`, e os seis ReAct
  internos recebem `**nested(options)`. O `_builders.py` lê as opções do spec só em
  `_flow_options`. O `plan_execute_flow` e o `llm_compiler_flow` passam a documentá-las. Em
  passagem, o `substitute_tools` deixou o `hasattr(tools, "definitions")`: recebe sempre um
  `ToolGroup`.
- **Tipos:** confirmado com o pyright numa sonda fora do repositório: um `timeout` que não é número,
  um `trace_capture` fora dos três e uma opção desconhecida dão erro na chamada. Em execução, uma
  opção desconhecida dá `TypeError` no `Flow`.
- **Testes novos:** três regras por AST e o seu canário em `tests/test_architecture.py` (vermelhas
  antes: 30 sítios no `_builders.py` e as nove factories);
  `tests/agents/flows/test_flow_options.py`, com 40 casos: as quatro opções chegam ao `Flow` das
  nove factories, os valores por omissão são os do `Flow`, uma opção desconhecida é recusada, os
  seis ReAct internos gravam como o fluxo de fora (`"full"` e `"none"`), e o `nested` só passa o
  `trace_capture`. Os 39 primeiros passaram antes da mudança (correram contra o código antigo).
- **Verificações:** 5295 passed, 42 deselected; ruff, formatação, pyright e lock limpos.
- **Saldo de linhas:** `toolkit/agents/flows/` e `_builders.py` +109 −179 (−70). Dívida sem
  mudança (61).
- **Documentação:** `docs/flow-architecture.md` (secção "Flow options"); `CHANGELOG` (Added:
  `FlowOptions`; Changed: a assinatura das factories, sem quebra de chamadas).

### Passo 4.3 · Nota de desenho (chaves de estado)

- **Medido:** as quatro chaves que as estratégias e o runner partilham (`task`, `messages`,
  `answer`, `response`) aparecem como texto 66 vezes em doze módulos de `toolkit/agents`. O
  `extract_text` tenta `answer`, depois `response` ou `last_response`, depois `last_answer`, e por
  fim o valor do último step. O `AgentResult.response` tenta `response` e depois `last_response`. A
  cadeia existe porque as estratégias não deixam a resposta da mesma maneira: o ReAct só escreve
  `response`; o Reflexion escreve `answer` e `response` só quando uma tentativa passa (ao esgotar,
  a resposta fica em `last_answer` e `last_response`); o Generate-Review escreve `answer` em todas
  as revisões, mas `response` só quando aceita.
- **Custo da adivinha, medido nos casos-limite:** um `answer` vazio salta para a chave seguinte e
  acaba no valor do último step (um Reflexion que passa com resposta vazia devolve o score, "1.0";
  um ReAct que acaba numa volta de tools devolve a lista de resultados das tools como texto).
- **Desenho:** um módulo, `flows/_keys.py`, com as quatro chaves como constantes `Final` e o
  contrato escrito:
  - cada estratégia deixa em `ANSWER` o texto da sua resposta e em `RESPONSE` a resposta do modelo
    de onde veio;
  - o ReAct passa a escrever `ANSWER` em cada volta;
  - o Reflexion passa a escrevê-las em cada avaliação, e o Generate-Review a escrever `RESPONSE`
    também quando rejeita.
  - As chaves privadas de cada estratégia (`feedback`, `plan_text`, `last_answer`, …) ficam no seu
    módulo. Nenhuma chave desaparece; as novas só se somam.
- **O runner lê uma fonte, não quatro.** Uma função decide o texto e a resposta a partir da mesma
  fonte:
  - com `ANSWER` no estado, o texto é o `ANSWER`, e a resposta o `RESPONSE`;
  - sem `ANSWER` (um fluxo feito à mão no `Agent.from_flow`, ou uma corrida que acabou antes de a
    estratégia responder), vale o valor do último step, a mesma regra que o `as_step()` já usa
    para o valor de um fluxo encaixado. Esse valor dá o texto (o `.text` de uma `Response`, ou o
    `str` do valor) e, se for uma `Response`, a resposta.
- **Visível (`CHANGELOG`, D35):**
  - um `answer` vazio é a resposta, em vez de saltar para outra chave;
  - um fluxo feito à mão que guarde a sua `Response` em `response` sem escrever `answer` passa a
    responder com o valor do último step (com o ReAct não muda nada, porque passa a escrever
    `answer`);
  - o `AgentResult.response` de um fluxo sem `answer` passa a ser a `Response` do último step
    (antes, `None`).
- **Prova:**
  - uma regra por AST: as quatro chaves só se escrevem como texto em `flows/_keys.py`, dentro de
    `toolkit/agents`;
  - as dez estratégias correm até ao fim e deixam `ANSWER` e `RESPONSE` iguais ao
    `AgentResult.text` e ao `AgentResult.response`, também quando esgotam (Reflexion sem passar,
    Generate-Review sem aceitar, ReAct no limite de voltas com tools);
  - o `extract_text` sem adivinha: um estado com `response` e sem `answer`, e um `answer` vazio.

### Passo 4.3 · Resultado

- **Como na nota.** `flows/_keys.py` tem as quatro chaves (`TASK`, `MESSAGES`, `ANSWER`,
  `RESPONSE`) e o contrato escrito. As 66 grafias passam às constantes. O `read_answer` do
  `_compile.py` é a única leitura da resposta: dá o texto ao `extract_text` e o texto e a resposta
  ao `AgentResult`, da mesma fonte. O ReAct escreve `ANSWER` em cada volta; o Reflexion escreve
  `ANSWER` e `RESPONSE` em cada avaliação; o Generate-Review escreve-as em cada revisão (saiu o
  ramo que só as escrevia ao aceitar).
- **Medido antes (código antigo):** um Reflexion que passa com resposta vazia respondia `"1.0"` (o
  score); um ReAct que acabava numa volta de tools respondia a lista dos resultados das tools como
  texto. Agora os dois respondem `""`.
- **Testes novos:**
  - em `tests/test_architecture.py`, a regra das chaves e o seu canário. Vermelha antes: as 66
    grafias.
  - `tests/agents/test_answer_contract.py`, com 19 casos:
    - as dez estratégias deixam `ANSWER` e `RESPONSE` iguais ao `AgentResult.text` e ao
      `AgentResult.response`;
    - as mesmas duas chaves nos três fins por esgotamento;
    - três regras do `extract_text`;
    - dois fluxos feitos à mão, e os nomes das quatro chaves.
  - Vermelhos antes: 7 dos 19 (o ReAct no fim normal, os três esgotamentos, o `answer` vazio, a
    queda para `response`, e a `Response` do fluxo feito à mão).
- **Verificações:** 5316 passed, 42 deselected; ruff, formatação, pyright e lock limpos.
- **Saldo de linhas:** `toolkit/agents` +14 neste passo (o módulo das chaves tem 26 linhas);
  `toolkit/agents` desde o início da R03 −56. Dívida sem mudança (61).
- **Documentação:** `docs/agents.md` (a resposta de um fluxo feito à mão, o contrato para
  estratégias próprias, o `extract_text`); `CHANGELOG` (quebra: a resposta vem de `answer` ou do
  último step).

### Passo 4.4 · Nota de desenho (validadores de manifesto)

- **Medido:**
  - O manifesto de agentes valida-se à mão em `_validate_manifest` (72 instruções, C901 29) e em
    doze tabelas de campos.
  - O de prompts valida-se nas funções de leitura (`_load_prompt` 77 instruções, `_parse_sections`,
    seis parsers de objectos) e num JSON Schema empacotado que o loader nunca usa e que é mais
    largo do que ele: `source`, `template`, `layout` e os selectores são lá `object` sem campos.
- **Sonda diferencial:** cada campo dos dois manifestos recebe um conjunto fixo de valores (nulo,
  texto vazio, 0, −1, 0.5, `true`, listas, objectos, `inf`, `nan`, e os valores válidos de cada
  enum); regista-se aceite, recusado ou crash. Deu 1323 casos nos agentes e 1166 nos prompts,
  guardados antes de mudar código.
  - Um crash: `strategy.trace_capture: []` dá `TypeError` em vez de `AgentManifestError`.
  - Folgas no loader de prompts:
    - `version: true` passa (`True == 1`), e um booleano passa como `order`;
    - um template com `content` aceita qualquer `select` e `serialize_as`, e ignora-os;
    - `metadata_attributes` aceita um objecto (fica com as chaves).
- **Desenho:** uma declaração por manifesto é a fonte.
  - `toolkit/_shape.py` tem as formas: texto, booleano, inteiro com mínimo, número finito com
    limites, escolha, constante, objecto livre, lista, campos fechados, nomes, alternativas, objecto
    etiquetado por `type`, e "pode ser nulo".
  - Cada forma sabe verificar um valor (a mensagem diz o caminho e o que falta) e escrever-se em
    JSON Schema. As sugestões "did you mean" passam a valer para os dois manifestos.
  - `agents/_manifest_shape.py` e `prompts/_manifest_shape.py` declaram cada manifesto. O loader
    verifica cada documento lido com a declaração, uma vez, e o código só guarda o que tem
    significado: ciclos, raízes, ficheiros, a exclusão entre campos, as regras de
    `remove`/`replace`/`merge`, e os construtores.
  - O JSON Schema de prompts passa a ser gerado da declaração; o de agentes é novo, gerado da mesma
    maneira. Um teste compara cada ficheiro com o gerado.
- **Prova:**
  - testes de cada forma, com um canário;
  - a sonda de antes e depois, com cada diferença justificada na ficha;
  - um teste de conformidade: com o `jsonschema`, que já vem no extra `prompts` do ambiente de
    desenvolvimento, cada schema gerado aceita e recusa o mesmo que a declaração, num corpus tirado
    da sonda;
  - uma regra por AST: os módulos de manifesto não têm tabelas de campos nem verificações de forma.
- **Mensagens:** mudam de redacção onde a forma passa a falar por si (dizem o mesmo campo e o
  mesmo problema). Os testes que as fixavam mudam de texto, nunca de força; a lista vai no
  resultado.

### Passo 4.4 · Resultado

- **Como na nota.**
  - `toolkit/_shape.py` tem as formas; cada uma sabe verificar, descrever-se e escrever-se em JSON
    Schema.
  - Um `Ref` dá nome a uma forma partilhada ou recursiva, e escreve-se uma vez em `$defs`.
  - `agents/_manifest_shape.py` e `prompts/_manifest_shape.py` são as declarações, que tiram as
    escolhas dos tipos de execução:
    - `TraceCapture`, `Reserve`, `Unpriced`, `PromptVariableType` e `PromptStability`, dos
      `Literal`;
    - a regra de tag XML e os modos do layout JSON, que passam a ter nome em `_layouts.py`.
  - Os dois loaders verificam cada ficheiro com a declaração e só guardam as regras entre campos.
    Saíram as doze tabelas de campos dos agentes, os `_reject_unknown`, os
    `_optional_*`/`_positive_*`, o `_validate_profile`, as duas tabelas e o `difflib` dos prompts,
    e as verificações de forma de nove parsers.
  - O `_load_prompt` partiu-se em `_manifest_resource`, `_inherited_sections` e `_template`; o
    `_parse_sections` ganhou `_section_action`.
- **Nulos nos agentes:** o loader já deixava `null` em quase todos os campos opcionais, mas os
  leitores não. `_section` e `_given` dão "não definido" a um `null` em `reasoning_spec`,
  `phase_models`, `budget_policy` e `_apply_overrides`.
- **Sonda diferencial, antes e depois:**
  - agentes: 21 de 1323 veredictos mudaram, todos de propósito:
    - o `trace_capture` com lista ou objecto deixa de dar crash;
    - um `strategy.system` que não é texto passa a ser recusado;
    - `override_policy: null` passa a ser aceite.
  - prompts: 38 de 1166 mudaram, todos de propósito, e nenhum crash:
    - `version: true` e um `order` booleano passam a ser recusados;
    - `metadata_attributes` que não é lista passa a ser recusado;
    - `select` e `serialize_as` num template com `content` passam a ser recusados.
- **Medido antes, nos leitores (código antigo):**
  - `strategy: null` falhava no `reasoning_spec()`;
  - `name: null` dava a estratégia `"None"`;
  - `max_iterations: null` dava `TypeError`;
  - `system: null` dava o prompt `"None"`;
  - `limits.reserve: null` falhava no `budget_policy()`.
- **Testes novos:**
  - `tests/toolkit/test_shape.py` (43):
    - cada forma, com os caminhos nas mensagens;
    - sugestões, alternativas, etiquetas e referências;
    - a concordância entre verificação e JSON Schema.
  - `tests/test_manifest_declarations.py` (9) e o gerador `tests/manifest_corpus.py`:
    - os dois ficheiros de schema são os gerados;
    - schema e declaração concordam em todos os documentos do corpus (mais de 1000 por manifesto,
      tirados da própria declaração);
    - o loader recusa tudo o que a declaração recusa, com a mesma mensagem (655 documentos nos
      agentes, 447 nos prompts), e só acrescenta regras de uma lista escrita;
    - uma regra por AST: nos dois loaders não há texto de forma ("must be a", "unknown fields",
      "did you mean").
  - Dez testes nos agentes (nulos, `system`, `trace_capture`), vermelhos contra o código do HEAD,
    corrido de uma cópia feita com `git archive`.
  - Mutação: tirar a verificação do loader de prompts põe o teste do loader vermelho.
- **Testes mudados, de texto e nunca de força:**
  - dois nos agentes (o campo desconhecido, agora com a sugestão; `version must be 1`);
  - 38 casos nos prompts, que agora fixam o caminho inteiro do campo.
  - Um caso de selector `block` sem `end_marker` passou a mostrar essa falta primeiro; ficou, e
    juntou-se outro que isola o `start_marker` vazio.
  - Um caso novo para o template com `content` e `select`.
- **Verificações:** 5380 passed, 42 deselected; ruff, formatação, pyright e lock limpos.
- **Saldo de linhas:**
  - Positivo: +441 de Python.
    - Os loaders: `agents/_manifest.py` 1006 → 733, `prompts/_manifest.py` 931 → 700.
    - Novos: `_shape.py` 593, declarações 148 + 203.
  - É o preço de uma declaração por manifesto e de um verificador partilhado com todas as funções
    dentro do orçamento de complexidade. Desvio à aceitação ("saldo negativo nos módulos da
    dívida"), para o dono pesar.
  - Os schemas são gerados: prompts 91 → 606 linhas, agentes 609 novas.
  - Dívida 61 → 53: saíram `_validate_manifest` (29/28/72), `_load_prompt` (24/25/77) e
    `_parse_sections` (14/13).
- **Documentação e CHANGELOG:**
  - `docs/agents.md` (a forma declarada, o nulo, o schema empacotado); `docs/prompt-manifests.md`
    (o schema gerado, as mensagens com caminho, o template em linha).
  - `CHANGELOG`: Added (schema de agentes); Changed, com quebra (mensagens e valores que deixam de
    passar); Fixed (nulos e o crash do `trace_capture`).

### Passo 4.5 · Nota de desenho e resultado (`_eval_expr`)

- **Desenho, como na ficha:**
  - Duas tabelas de despacho, `_EXPRESSIONS` (18 nós) e `_STATEMENTS` (7), com um método por nó;
    um nó fora das tabelas é recusado.
  - O `_eval_expr` (C901 50, 98 instruções) e o `_exec_stmt` (22) passam a três linhas cada.
  - O atributo lê de uma tabela de métodos por tipo (`_METHODS`, com o descritor ligado ao valor)
    e das funções do `re`, pelo mesmo `_method`.
  - A chamada verifica um callable pelo mesmo `_method` (um método ligado traz o seu `__self__` e
    `__name__`), e saem o `hasattr` e o `getattr` que adivinhavam capacidades.
  - As compreensões partilham um só gerador; o `**` e o `**=` passam pelo mesmo `_binary`.
  - Saiu o `_run_eval`, que ninguém chamava.
- **Corrigido em passagem (o comportamento antigo, medido pelos testes vermelhos):**
  - `{**a}` dava `{None: a}`;
  - `f(**d)` largava o `d` em silêncio;
  - `and`/`or` avaliavam todos os operandos (`0 and 1/0` dava erro);
  - `del s.upper` era ignorado;
  - `x **= 5000` saltava o guarda do expoente.
- **Achado em passagem, na infra de testes:**
  - Com o pytest 9.1, uma linha de comando que intercala ficheiros de `tests/toolkit` com os de
    outra pasta deixa cair as fixtures de `tests/toolkit/conftest.py`. Os testes das tools correram
    então com sockets livres e com o throttle a dormir de verdade; foi assim que os invariantes
    pararam.
  - O gate não o via, porque colhe a árvore inteira.
  - As duas fixtures passaram para o `tests/conftest.py`, pelo caminho do teste, e o conftest da
    pasta saiu.
  - Um canário em `test_http.py` (`getaddrinfo("localhost")` tem de dar o erro de rede bloqueada)
    ficou vermelho na ordem intercalada antes da mudança, e verde depois.
- **Testes novos:**
  - `tests/toolkit/test_python_nodes.py` (73):
    - as tabelas contêm exactamente os 25 nós com amostra;
    - cada nó permitido corre a sua amostra;
    - cada nó do `ast` fora das tabelas é recusado (expressões e instruções, enumeradas do `ast` em
      execução, por isso um Python novo obriga a decidir);
    - 17 programas com nós recusados.
  - Cinco testes de semântica em `tests/test_python_eval.py`.
  - A regra de descoberta de capacidades estendida a `_python.py`, em `tests/test_architecture.py`.
  - Todos vermelhos antes.
- **Verificações:** 5460 passed, 42 deselected; ruff, formatação, pyright e lock limpos.
- **Saldo de linhas:** `_python.py` 765 → 729 (−36). Dívida 53 → 48 (`_eval_expr` 50/49/98,
  `_exec_stmt` 22/21).
- **Achado para o dono (`FINDINGS`):** o `python_repl` só limita o expoente de `**`, e o
  `math_eval` já tem o guarda de tamanho que lhe falta.

### Passo 4.6 · Nota de desenho e resultado (a primitiva do ReAct interno)

- **Medido:** sete sítios em seis estratégias construíam um `react_flow`, semeavam o estado,
  corriam-no, liam a resposta do `RESPONSE`, e dois deles percorriam o trace à procura de erros.
  Os sítios eram o Generate-Review duas vezes, o Plan-Execute, o LLMCompiler, o LATS, o Reflexion
  e o Self-Discovery.
- **Desenho:** `run_react(llm, tools, task, *, system, max_iterations, llm_kwargs, **options)` em
  `flows/_react.py` devolve um `ReactRun(answer, response, failed)`.
  - Recebe as opções do fluxo de fora e aplica-lhes o `nested` num só sítio, que assim deixa de
    aparecer nas estratégias.
  - A resposta vem do `ANSWER` que o ReAct já deixa (D35), e `failed` é o "algum step acabou em
    erro" que o Plan-Execute e o LLMCompiler escreviam à mão.
  - A docstring diz o que valia nos comentários repetidos: o scope do meter é o de fora (um só
    budget) e o prazo de fora cobre o ciclo.
  - O LLMCompiler guarda o seu escalonamento próprio dentro do step: está fora deste padrão.
- **Testes novos:**
  - `tests/agents/flows/test_run_react.py` (3: a resposta e a `Response`, o erro que marca
    `failed`, o sistema e os kwargs que chegam ao modelo);
  - uma regra por AST e o seu canário em `tests/test_architecture.py`: só `_react.py` constrói ou
    semeia um ReAct.
  - Vermelhos antes: os três (o import) e a regra (seis módulos). Os testes de cada estratégia e
    os do passo 4.2 (o ReAct interno grava como o de fora) passaram sem mudança.
- **Verificações:** 5465 passed, 42 deselected; ruff, formatação, pyright e lock limpos.
- **Saldo de linhas:** `toolkit/agents/flows/` 1825 → 1813 no acumulado dos passos 4.2, 4.3 e 4.6
  (o `_keys.py` incluído).
- **Dívida:** 48 → 46. Saíram `lats_flow` (54 instruções) e `llm_compiler_flow` (51); o
  `llm_compiler_flow` passou de 17 para 16 e o seu `compile` de 16 para 15.
- **CHANGELOG:** nada visível.

### Passo 4.7 · Levantamento do `__all__` de topo (para o dono decidir; nenhum export mudou)

- **Contagem:** 204 nomes. Eram 198 na medição do plano: a R01 juntou cinco erros de fornecedor
  (`ProviderError`, `RequestError`, `TransportError`, `ProviderTimeout`, `ResponseError`) e a R02 o
  `UnpricedModelError`. A R03 não mudou nenhum (comparado com o HEAD).
- **Tipos:** 113 classes, 52 funções, 23 aliases de tipo, 14 excepções, a instância `pricing` e o
  `__version__`. Todos, menos o `__version__`, são também exportados por um subpacote
  (`ai_arch_toolkit.core` ou `ai_arch_toolkit.toolkit.*`): o topo é um espelho de conveniência.
- **Por área (módulo de origem):**

  | Área | Nomes | Exemplos |
  |---|---:|---|
  | `core._tools` | 27 | `tool`, `ToolGroup`, `ToolResult`, portões (`ApprovalGate`, `DryRunGate`, …), `infer_schema` |
  | `toolkit.prompts` | 20 | `load_prompt`, `PromptTemplate`, os quatro layouts, `SectionSpan` |
  | `toolkit.agents` | 16 | oito das nove factories e os seus `*_initial_state` |
  | `core._metering` | 15 | `MeterScope`, `RunConfig`, `Money`, `AdmissionController`, `Reservation` |
  | `toolkit.memory` | 14 | `GraphStore`, as quatro vistas, `MemoryPreset`, `memory_tools` |
  | `core._content` | 12 | `user`, `assistant`, `system`, `image`, `document`, `cache` |
  | `toolkit.flow` | 11 | `Flow`, `FlowStep`, `Scope`, `execute_flow`, `iter_flow` |
  | `core._response` | 10 | `Response`, `Usage`, `ToolCall`, `StreamEvent` |
  | `toolkit.budget` | 9 | `BudgetPolicy`, `BudgetController`, `budget_scope` |
  | `core._exceptions` | 8 | `APIError`, `ProviderError` e as suas quatro filhas |
  | outras 23 áreas | 52 | grafo, redacção, estado, política, trace, tokens, preços, moderação, … |
- **Uso medido:**
  - Só 81 dos 204 nomes são importados de `ai_arch_toolkit` nos exemplos, nas docs ou no README.
  - Não aparecem uma única vez nesses textos 25 nomes, entre eles: `AdmissionController`,
    `AdmissionDecision`, `ConditionFn`, `CostKind`, `Estimator`, `GovernanceOutcome`,
    `HeuristicEstimator`, `KnowledgeSearchResult`, `NotMeteredOperationError`, `OnExhausted`,
    `OnLowConfidence`, `OnTimeout`, `OperationRequest`, `PromptMessageContent`,
    `PromptMessageRole`, `PromptStability`, `Reservation`, `Reserve`, `RunState`, `ToolSchema`,
    `TraceMode`, `UsageSink`, `configure_sync_timeouts`, `deprecated`, `tool_schema`.
- **Incoerências para o dono ver:**
  - `Agent` e `ReasoningSpec` (a entrada recomendada no `AGENTS.md`) não estão no topo, mas oito
    das nove factories do nível de baixo estão.
  - O `generate_review_flow` (a nona) e o seu `*_initial_state` também faltam no topo.
  - Os 23 aliases de tipo são quase todos de anotação de subsistemas (`OnExhausted`,
    `PromptMessageRole`, `NodeID`, …).
- **Opções, sem recomendação forte:**
  1. Um topo pequeno, com os nomes de um primeiro programa (os 81 usados, mais `Agent` e
     `ReasoningSpec`), e o resto só nos subpacotes, com avisos de descontinuação num ciclo (um
     `__getattr__` no `__init__` que avisa e devolve o nome).
  2. Manter o espelho, mas documentar dois níveis ("primeiro programa" e "extensão") e fechar as
     incoerências (juntar `Agent`/`ReasoningSpec` e a nona factory, ou tirar as factories).
  3. Manter tudo como está até haver dados de uso de fora.

### Passo 5 · Fecho da frente

- **Deriva de documentação:**
  - O `AGENTS.md` e o `CONTRIBUTING.md` ainda mandavam usar e simular o `urllib` nas tools.
    Passaram a descrever a porta única `_http.py` e a sua simulação (`HTTP_OPEN`), a capacidade
    declarada, os limites por chamada, as opções dos fluxos e as chaves da resposta, o
    `run_react`, e as declarações dos manifestos com os schemas gerados.
  - As `docs/` foram actualizadas em cada passo (listas nos resultados).
- **`CHANGELOG`:** uma secção "Upgrade notes" no topo do `[Unreleased]`, com as quebras das três
  fases.
  - Entram as marcadas e as que, sem a marca, pedem uma mudança a quem usa: o `fallback_on`, o
    `RequestError` antes de enviar, e o `Response.cost` a `None` sem usage.
  - As quebras marcadas anteriores à frente (13, 15 e 17 de Setembro) não entram: o `git log -S`
    mostrou a data de cada uma.
- **Linha de base de complexidade:** 121 → 46 entradas, actualizada em cada passo e verificada pelo
  `test_quality_budget`.
- **BOARD:** a frente de robustez fica à espera da revisão e dos commits do dono; a vez passa à
  frente C.

## Relatório final (R00)

**Por passo**

1. **Tools: feito.**
   - `_http.py` é a porta única (regra por AST, sem lista de legado): os 47 sítios em 36 módulos
     migraram.
   - Limites de saída e de tempo em qualquer tool (D29); as tools síncronas correm numa thread
     própria (D30); a leitura de cada resposta faz-se dentro de uma fronteira (D31).
   - Guardas à entrada no `math_eval` e no `regex_search` (D32); o `ip_lookup` passou ao ipwho.is
     (D28).
   - Invariantes executáveis (739 casos, sem `xfail`).
   - Não feito, fica em `FINDINGS` para o dono:
     - o risco polinomial do regex;
     - o tecto do `pdb_ligands`;
     - o intervalo de 3 s do arXiv;
     - o endpoint da UniProt, por verificar ao vivo.
2. **Motor: feito.**
   - Um só executor de vagas e um só sítio onde um step acaba.
   - As vagas lêem o snapshot, sem `fork()` (D33).
   - Gramática de eventos verificada nos testes do motor.
   - O `_run_attempts` ficou com um só bloco de fallback.
3. **Imutabilidade: feito.** O `StateSnapshot` está documentado e testado igual nos dois modos, e os
   `knobs` e `llm_kwargs` congelam à entrada. A configuração estrita já estava feita.
4. **Dívida: feito.**
   - 4.1: uma base partilhada para os dois stores de grafo; é herança, não composição, e o desvio
     está na nota.
   - 4.2: `FlowOptions`, com a `budget_policy` como opção do `Flow` (D34).
   - 4.3: chaves num só módulo e uma só leitura da resposta (D35).
   - 4.4: uma declaração por manifesto, com os JSON Schema gerados dela (D36).
   - 4.5: tabelas de despacho no `_eval_expr`.
   - 4.6: `run_react`.
   - 4.7: só o levantamento, sem mudar exports, como a ficha pedia.
5. **Fecho: feito** (acima).

**Números**

- **Testes:** 4341 → 5465 passed (+1124), 42 `live_api` deselected, zero `xfail`.
  - Ficheiros novos: `tests/toolkit/test_http.py`, `test_tool_invariants.py`, `http_fakes.py`,
    `test_shape.py`, `test_python_nodes.py`, `tests/test_tool_limits.py`,
    `tests/flow/event_grammar.py`, `test_event_grammar.py`, `tests/agents/test_answer_contract.py`,
    `tests/agents/flows/test_flow_options.py`, `test_run_react.py`,
    `tests/test_manifest_declarations.py`, `tests/manifest_corpus.py`.
  - As mensagens mudadas estão listadas no passo 4.4; nenhum teste perdeu força.
- **Dívida de complexidade:** 121 → 46 entradas.
- **Linhas de Python, contra o HEAD:**

  | Área | Antes | Depois | Saldo |
  |---|---:|---:|---:|
  | `toolkit/tools` | 13 936 | 13 630 | −306 |
  | `core/_tools` (limites, thread por chamada) | 2073 | 2277 | +204 |
  | `flow/_executor.py` | 831 | 841 | +10 |
  | `core/_step_engine.py` | 246 | 272 | +26 |
  | stores de grafo (com `_facade.py`) | 1763 | 1653 | −110 |
  | `toolkit/agents` (com a declaração do manifesto) | 4075 | 3932 | −143 |
  | `toolkit/prompts` (com a declaração do manifesto) | 3349 | 3322 | −27 |
  | `toolkit/_shape.py` (novo) | 0 | 593 | +593 |
  | `src/ai_arch_toolkit`, tudo | 49 960 | 50 749 | +789 |
  | `tests` | 42 865 | 46 002 | +3137 |

  Os JSON Schema gerados não entram na tabela: prompts 91 → 606 linhas; agentes 609, novo.
- **Aceitação:**
  - Cumpridos:
    - regras de arquitectura verdes (rede só em `_http.py`, capacidades declaradas iguais às
      alcançadas, nenhuma tool local fora de `dangerous`);
    - invariantes sem `xfail`;
    - 5 MB cortados ao limite;
    - `9**9**9` e `(a+)+$` recusados dentro do prazo;
    - `_run_dag` e `_run_attempts` dentro do orçamento, e gramática verde;
    - duplicação entre os stores eliminada (0 janelas);
    - `_eval_expr` dentro do orçamento;
    - saldo negativo em `toolkit/tools`.
  - Em falta: o saldo negativo nos módulos da dívida. Cada um desce (stores −110, agentes −143,
    prompts −27, `_python.py` −36), mas o `_shape.py` novo (+593) deixa a soma positiva. É o desvio
    do passo 4.4, para o dono pesar.

**Mudanças visíveis** (todas no `CHANGELOG`, e as quebras nas "Upgrade notes")

- **Tools:**
  - 120 s de prazo e 200 000 caracteres de saída por omissão;
  - tools síncronas numa thread própria no caminho síncrono;
  - `ip_lookup` no ipwho.is;
  - só HTTPS e só os hosts de cada módulo, com um User-Agent, 10 MB por resposta e uma mensagem
    de 429;
  - `http_get` e `scrape_text` mais apertados;
  - guardas no regex e na matemática, e limites nas tools de ficheiros;
  - capacidades declaradas, e o relógio do Nominatim partilhado.
- **Motor:** as vagas paralelas lêem o snapshot, e o trace regista os steps pela ordem em que
  acabam.
- **Agentes:**
  - `knobs` e `llm_kwargs` congelados;
  - `FlowOptions` exportado de `agents.flows`, e as factories com `**options`;
  - a resposta vem de `"answer"`, ou do último step.
- **Manifestos:** uma forma declarada, mensagens com caminho, alguns valores recusados, e o
  schema de agentes novo. Os nulos dos agentes passam a funcionar.
- **`python_repl`:** recusa `{**a}`, `f(**d)` e `del x.attr`, o `and`/`or` faz curto-circuito, e
  o `**=` tem o guarda de expoente.

**O que não se verificou ao vivo** (a R00 proíbe-me chamar APIs externas; estas são gratuitas e
sem chave)

1. `ip_lookup` no ipwho.is:
   `uv run python -c "from ai_arch_toolkit.toolkit.tools import ip_lookup; print(ip_lookup('8.8.8.8'))"`
2. O fio das tools que mudou (HTTPS, espaços como `+`, `;` e `:` nos segmentos, o `tool=` do
   PubMed, os dois relógios do Open Food Facts). Cada linha deve mostrar dados, não
   `HTTP error`/`URL error`:

   ```bash
   uv run python - <<'EOF'
   from ai_arch_toolkit.toolkit import tools as t
   checks = {
       "gdelt": lambda: t.gdelt_news_search("climate policy", max_results=2),
       "world bank": lambda: t.world_bank_series("PRT;ESP", "NY.GDP.MKTP.CD", "2020", "2021"),
       "who": lambda: t.who_series("WHOSIS_000001", country="PRT"),
       "pubmed": lambda: t.pubmed_search("insulin resistance", max_results=2),
       "open food facts": lambda: t.open_food_facts_search(product_name="nutella", max_results=2),
       "osm": lambda: t.osm_search_place("Lisbon", max_results=1),
       "reverse geocode": lambda: t.reverse_geocode(38.72, -9.14),
       "wikipedia": lambda: t.wikipedia_search("Fernando Pessoa"),
       "weather": lambda: t.get_weather("Lisbon"),
       "semantic scholar": lambda: t.semantic_scholar_paper("DOI:10.1038/nature14539"),
   }
   for name, call in checks.items():
       print(f"== {name}\n{call()[:300]}\n")
   EOF
   ```

3. UniProt: o `curl` em `FINDINGS.md`, contra o mesmo pedido com `/uniprotkb/search`.
4. Opcional, com chave e um custo pequeno: um agente com tools de ponta a ponta, para ver o
   `AgentResult.text` do contrato da resposta com um modelo real.
   `set -a && source .env && set +a && uv run python examples/09_react_agent.py`

**Divisão proposta em commits** (o dono faz os commits; nenhuma operação git de escrita nesta
sessão)

1. `refactor: one HTTP door for tools, one wave runner, declared manifests, maintenance debt`,
   com todo o `src/` e o `tests/`. A árvore foi verificada sozinha: uma cópia com as docs e o
   registo no estado do HEAD, criada com `git ls-files`, `tar` e `git show` (só leitura), e o
   `PYTHONPATH` na cópia. Deu 5465 passed, 42 deselected, e ruff, formatação e pyright (0 erros)
   limpos. O `pyproject.toml` e o `uv.lock` não mudam.
2. `docs: upgrade notes, contributor guides and the R03 record`, com `docs/`, `AGENTS.md`,
   `CONTRIBUTING.md`, `CHANGELOG.md` e `blackboard/`. Não muda código nem testes.

Não proponho um commit por passo pela mesma razão da R02: a linha de base de complexidade (um
teste de igualdade exacta) e o `tests/test_architecture.py` mudam em todos os passos. Commits por
passo pediam variantes intermédias desses ficheiros, que ninguém correu.

**Feitos a pedido do dono (2026-09-18) e publicados em `main`:**

- `6ceb516` (1), cuja árvore é a verificada acima;
- o commit do registo (2).

O push levou também os três commits da R02 (`cf2aa43`, `a8c3eea`, `4a1fa4e`), que estavam por
publicar.

**Regras da R00 nesta sessão:**

- Sem commits, pushes, branches nem stash.
- Sem fornecedores nem `.env`.
- Os testes das tools correm com os sockets bloqueados, e agora também numa linha de comando
  intercalada (o canário do passo 4.5).
- Sem dependências novas: o `jsonschema` dos testes já vinha no extra `prompts` do ambiente de
  desenvolvimento.
- Sem `# type: ignore` nem `# noqa` novos. Dois escaparam num rascunho e foram tirados antes do
  gate.
- Operações git só de leitura: `status`, `diff`, `show`, `log`, `ls-files` e `archive`. Houve
  também três `git stash list`, sem efeito, que não deviam ter sido corridos.
