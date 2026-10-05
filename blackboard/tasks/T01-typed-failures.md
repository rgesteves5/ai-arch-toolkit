# T01 · Falhas tipadas: `ToolFailure`, o executor e o `is_error`

- **Dono:** Claude (2026-10-05) · **Estado:** done · **Depende de:** nada
- **Origem:** plano, secção 3.1 e anexo F · **Decisões:** D37, D42 · **Regras:** `T00-rules.md`

## Problema

- As tools devolvem strings de erro, com seis ou mais formulações ("X failed: …", "No … found",
  "not found").
- O executor embrulha qualquer valor em `ToolResult.success` (`core/_tools/_executor.py:240`), e o
  meter liquida a chamada como sucesso.
- O `tool_result()` (`core/_content.py`) não tem campo de erro, e o adaptador Anthropic nunca manda
  `is_error`.
- O ai-network mostrou `ok: true` para páginas que não existiam.

Uma excepção lançada por uma tool já dá `ToolResult(ok=False)`: o `_failed` passa-a ao
`_result_from_exception`, que devolve `runtime_error` com `retryable=True`, sem distinguir "não
existe" de "a fonte caiu".

## Objectivo

Uma tool que não consegue responder lança `ToolFailure`. O executor devolve `ok=False` com o tipo e
o `retryable` da falha, e o erro chega ao fornecedor marcado como erro. Nenhuma tool devolve strings
de erro.

## Desenho (confirma-o na nota de desenho, antes do código)

- **`ToolFailure(Exception)`**, em `core/_tools/_result.py`, pública em `ai_arch_toolkit.core` e no
  topo: `ToolFailure(type, message, *, retryable=False, details=None)`, com `.error: ToolError`. Os
  tipos das tools são um `Literal` com `not_found`, `validation_error`, `upstream` e `rate_limited`;
  os do runtime (`unknown_tool`, `runtime_error`, `timeout`, os dos gates) ficam como estão.
- **Executor:** no `_failed`, um `ToolFailure` vira `ToolResult(ok=False, error=exc.error)`, com a
  mensagem redigida pelo `Redactor`; o meter continua a fazer `op.fail("unbilled")`. As outras
  excepções continuam `runtime_error`.
- **`tool_result(content, *, tool_use_id, name=None, is_error=False)`.** O `_react.py` e o
  `toolkit/_runner.py` passam `is_error=not result.ok`. O adaptador Anthropic manda `is_error: true`
  no bloco `tool_result` (confirma no `ToolResultBlockParam` do SDK instalado). Para os outros
  fornecedores, confirma na documentação se algum tem campo próprio (o Gemini, por exemplo) e
  regista o que encontrares; sem campo, basta o texto `Tool error [type]: …`.
- **`HttpError` é uma subclasse de `ToolFailure`.** 429 dá `rate_limited` (retryable); 5xx, falha de
  rede e timeout dão `upstream` (retryable); os outros 4xx dão `upstream`. O 404 fica `upstream` até
  à T02, que o declara por endpoint.
- **Nas tools:** desaparece o `try/except HttpError: return f"… failed: {e}"` de cada uma. As
  validações próprias ("… failed: invalid title.") passam a `raise ToolFailure("validation_error",
  …)`, e os "não existe" a `not_found`. Zero resultados continua a ser uma string de sucesso.

## Nota de desenho (2026-10-05, antes da migração dos módulos)

- **Core (feito):**
  - `ToolFailure(type, message, *, retryable=False, details=None)` e o tipo `ToolFailureType`
    (`not_found`, `validation_error`, `upstream`, `rate_limited`), em `core/_tools/_result.py`,
    públicos em `ai_arch_toolkit.core` e no topo;
  - o executor (`_result_from_exception`) devolve o `ToolError` da falha, com a mensagem e os
    `details` redigidos e o `tool_name` acrescentado. O meter faz `op.fail("unbilled")`, como para
    qualquer excepção;
  - `tool_result(..., is_error=)`. O runner (`run_tools`, `run_tools_sync`) e o ReAct passam
    `is_error=not result.ok`. A Anthropic manda `is_error: true` no bloco `tool_result`
    (https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls). O Gemini
    põe o resultado sob `error` no `FunctionResponse.response`, a chave que a API lê como "error
    details" (docstring do `google-genai`). A Responses API, a Chat Completions e o xAI não têm
    campo próprio: o texto `Tool error [type]: …` diz-lo;
  - o executor não repete sozinho uma falha `retryable`: nada no core ou no toolkit lê o campo.
- **A porta:** o `HttpError` é um `ToolFailure`:
  - um 429 é `rate_limited` e retryable, um 5xx é `upstream` e retryable, outro 4xx é `upstream`;
  - um timeout ou um erro de rede é `upstream` e retryable;
  - um URL ou um segmento que o módulo não pode pedir é `validation_error`;
  - os `details` levam o `status`.
- **Os módulos, regra a regra:**
  1. Desaparece todo o `try/except HttpError` que só devolve uma string: o `HttpError` sobe.
     Se o módulo distingue um estado (`e.status == 404` → "not found"), fica
     `raise ToolFailure("not_found", …) from e`, e os outros estados voltam a subir (`raise`).
  2. Um argumento inválido levanta `ToolFailure("validation_error", …)`, e um "não existe" (o
     recurso pedido não existe na fonte) levanta `ToolFailure("not_found", …)`. Um erro que a fonte
     explica no corpo de um 200 levanta `ToolFailure("upstream", …)`.
  3. **A mensagem** diz o motivo e o passo seguinte, sem o prefixo "X failed:" (o tipo já o diz):
     "invalid DOI '10.x/'; a DOI looks like 10.1000/xyz", "no Crossref work with DOI 10.1/abc;
     search with crossref_search". O nome da fonte entra quando ajuda.
  4. **Zero resultados continua a ser uma string de sucesso** ("No Crossref results for: 'x'").
  5. Um helper que devolvia uma string de erro como sentinela passa a levantar; o chamador deixa de
     a testar.
  6. O texto de uma execução (a saída de um programa, um traceback do código que o agente pediu
     para correr) é resultado, não falha.
- **Testes:** os testes de cada módulo verificam as falhas com
  `pytest.raises(ToolFailure) as caught` e `caught.value.error.type`, ou pelo executor. Nenhum
  teste lê uma string de erro devolvida.

## Passos

1. Nota de desenho na ficha: assinaturas, o que desaparece, os testes que o provam.
2. **Core.** `ToolFailure`, executor, `tool_result(is_error=)`, react, runner, adaptador Anthropic e
   exports. Testes primeiro:
   - uma tool que lança `ToolFailure("not_found", …)` dá `ok=False`, `error.type == "not_found"`,
     `retryable` falso e meter `unbilled`;
   - o pedido ao Anthropic leva `is_error`, e a rede do fio (`wire_log`) aceita-o.
3. **`_http.py`:** o `HttpError` como `ToolFailure`, com o mapeamento dos estados.
4. **Os 44 módulos de `toolkit/tools`:** nenhum `except HttpError` que devolva uma string;
   validações e "não existe" tipados. Os testes das tools passam a verificar pelo executor
   (`execute_tool`) ou com `pytest.raises(ToolFailure)`, sempre com o `type`.
5. **`test_tool_invariants.py`:** "never raise" passa a "a chamada crua só lança `ToolFailure`" e
   "pelo executor, sempre um `ToolResult`".
6. **Docs:** `AGENTS.md` (a regra "Toolkit tools return error strings instead of raising" muda),
   `CONTRIBUTING.md` (o guia de tools), `docs/tools.md` e `docs/safety.md` onde falarem disto.
   `CHANGELOG`: **Breaking:** uma tool que falha, chamada crua, lança `ToolFailure`; pelo executor dá
   `ok=False` com o tipo.

## Ficheiros

`core/_tools/_result.py`, `core/_tools/_executor.py`, `core/_tools/__init__.py`, `core/__init__.py`,
`ai_arch_toolkit/__init__.py`, `core/_content.py`, `core/_providers/_anthropic.py`,
`toolkit/_runner.py`, `toolkit/agents/flows/_react.py`, `toolkit/tools/_http.py`, os 44 módulos
`toolkit/tools/_*.py` e os seus testes em `tests/toolkit/`, `tests/toolkit/test_tool_invariants.py`,
`tests/test_tools_executor.py`, `tests/test_anthropic_provider.py`, `tests/test_core_exports.py`,
`AGENTS.md`, `CONTRIBUTING.md`, `docs/tools.md`, `docs/safety.md`.

## Prova

- Os testes do passo 2, a falhar antes pela razão certa.
- O `retryable` por estado HTTP, na porta.
- Zero resultados continua `ok=True`.
- **Arquitectura (AST):** nenhum módulo de `toolkit/tools`, fora do `_http.py`, tem `except
  HttpError`.
- Gate completo verde.

## Fora do âmbito

O 404 por endpoint e os leitores de erro da porta (T02); a janela (T03); os limites (T04a).

## Riscos

- É uma quebra para quem chama as tools cruas (testes de aplicações, scripts). Mitigação: a entrada
  **Breaking** e os exemplos actualizados.
- `retryable=True` em `upstream` pode levar um agente a repetir chamadas. Confirma que o executor não
  repete sozinho e escreve na ficha o que encontrares.

## Registo do dono

- **Estado:** done (Claude, 2026-10-05), pela nota de desenho. A costura do core e as partes
  partilhadas fê-las o coordenador; os módulos, cinco agentes em paralelo, com módulos e testes
  disjuntos e o mesmo guia.
- **Core:**
  - `ToolFailure` e `ToolFailureType`, públicos;
  - o executor devolve o tipo, o `retryable` e a mensagem redigida, e o meter faz
    `fail("unbilled")`;
  - `tool_result(is_error=)` no runner e no ReAct;
  - o `is_error` da Anthropic (extraído para `_tool_result_block`, que mantém a complexidade
    dentro do orçamento) e a chave `error` do Gemini.
- **A porta:** o `HttpError` é um `ToolFailure`. O tipo vem do estado e do sítio onde é
  levantado: os timeouts e os erros de rede são retryable, e os URLs e segmentos recusados, o
  URL inválido do `fetch_page` incluído, são `validation_error`.
- **Os módulos:** nenhum devolve uma string de erro.
  - Um `except HttpError` só aparece para levantar uma falha mais precisa: um 404 dá `not_found`,
    e o 500 do EONET ganha uma pista. Também no openFDA, que responde 404 a uma pesquisa sem
    resultados e que passou a ser um sucesso. O teste de arquitectura recusa um handler que
    devolva.
  - Os 404 declarados em `status_messages` ficam `upstream` até à T02.
  - As saídas de execução continuam resultado: um erro do programa no `python_repl` ou um código
    de saída diferente de zero no `run_command`.
- **As invariantes:** a chamada crua dá texto ou um `ToolFailure` tipado, nunca outra excepção.
  Um 500 é `upstream` e um 429 é `rate_limited`, quando a tool chegou a fazer o pedido (com as
  chaves das APIs no ambiente). O guarda do GIL recusa com `validation_error`.
- **Testes:** `tests/test_tool_failures.py` (12), os testes da porta (tipos por estado, timeout,
  URL recusado) e os de cada módulo, que verificam o tipo e as palavras da mensagem.
- **Docs:** `AGENTS.md`, `CONTRIBUTING.md`, `docs/tools.md`, `docs/tools-catalog.md`,
  `docs/safety.md`, `docs/framework-overview.md`, `CHANGELOG.md` (Upgrade notes, Added e
  **Breaking**).
- **Revisão independente** (cinco revisores sobre os módulos, mais o core, a porta, os testes e
  as docs): nenhuma falha grave, o core está certo e nenhum módulo devolve strings de erro.
  Corrigido:
  - **médias:**
    - as tools de memória (`memory_tools`) ainda devolviam strings: passaram a `not_found`;
    - as docs desactualizadas (`docs/code-style.md`, duas frases de `docs/tools.md`) e o exemplo
      24, que chamava o `define_word` cru;
    - o `csv_read` dava tipos diferentes do `read_file` para os mesmos erros do sistema de
      ficheiros: passou a partilhar o `path_failure`;
    - o 404 do Open Food Facts v2: `not_found`, e o `compare` lista o produto em falta.
  - **baixas:**
    - na porta: o URL com caracteres de controlo é `validation_error`, um certificado inválido
      deixa de ser retryable, e o `retry_after_s` vai para os `details`;
    - o `ToolFailure` volta a fazer pickle e cópia (`__reduce__`);
    - o `is_error` do ReAct tem teste, nos dois ramos;
    - os passos seguintes errados (`get_forecast`, `nvd_cve`, as falhas aritméticas do
      `math_eval`);
    - "try again" em falhas que não são retryable (air quality, EONET, Hacker News);
    - a pesquisa vazia da Wikipedia e o `invalidtitle` da MediaWiki são `validation_error`;
    - o 403 do YouTube deixa de ser retryable;
    - as excepções cruas do `regex_search` (`OverflowError`) e do `csv_read` (`csv.Error`);
    - a sequência vazia da UniProt é `not_found`;
    - as palavras das mensagens do openFDA (os nomes dos parâmetros) e da DailyMed.
  - Ficam para a T02 as duas médias que são da porta: o 404 declarado em `status_messages`
    ("no matching records found", `upstream`, em onze módulos, e as docstrings do
    `earthquake_event` e do `rxnorm_concept` prometem `not_found`), e o `body_error`, que só sabe
    dar `upstream` (as entradas inactivas da UniProt, o erro 120 do World Bank, o `missingtitle`
    lido pela mensagem na MediaWiki, o 413 do Eurostat).
- Gate: 6769 passed, 42 skipped; ruff, formatação e pyright limpos.
- **Para a T02**, pelos agentes e pela revisão:
  - o 503 do Open Food Facts, que a fonte documenta como limite de pedidos;
  - o 404 de um produto desconhecido no OFF v2;
  - o `ratelimited` no corpo da MediaWiki, hoje `upstream`;
  - o 404 de um dataset que o Eurostat não publica;
  - uma chave em falta ("no key: set X"), hoje `upstream`.
