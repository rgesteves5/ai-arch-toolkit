# T01 · Falhas tipadas: `ToolFailure`, o executor e o `is_error`

- **Dono:** por atribuir · **Estado:** todo · **Depende de:** nada
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

- **Estado:** todo.
