# F07 · Tools async no caminho síncrono; parâmetros positional-only

- **Dono:** worker-B · **Estado:** done · **Depende de:** —
- **Plano:** N2, N10, F7

## Problema

1. `_run_tool_sync` (`src/ai_arch_toolkit/core/_tools/_executor.py`) chama `definition.fn(**args)` sem
   aguardar: uma tool `async def` devolve uma coroutine com `ok=True`; `to_model_text()` rebenta com
   `TypeError`; aparece `RuntimeWarning: coroutine was never awaited`. Afecta `ToolGroup.execute`,
   `execute_tool` e `run_tools_sync`.
2. No caminho async, uma função síncrona que devolva um awaitable fica com a coroutine como valor.
3. `def square(x: int, /)` → o schema pede `x`; a chamada `fn(x=3)` falha com `TypeError`
   (`validation_error`): a tool é impossível de chamar.

## Mudança

- Síncrono: se a função é coroutine function ou devolve um awaitable, completar com
  `ai_arch_toolkit.core._sync._run_sync` (envolver awaitables que não sejam coroutines numa coroutine).
- Async: se `asyncio.to_thread` devolver um awaitable, aguardá-lo.
- Chamada: parâmetros positional-only vão por posição, pela ordem de `inspect.signature(definition.fn)`
  (segue `__wrapped__`); os restantes por nome. Um helper único para os dois caminhos.
- Manter a alteração local: o coordenador acrescenta validação de argumentos (F13) nestas mesmas
  funções depois.

## Ficheiros

- `src/ai_arch_toolkit/core/_tools/_executor.py`
- `tests/test_tools_executor.py` (ou `tests/test_tools_group.py`)

## Prova

- `ToolGroup(async_tool).execute(tc)` → `ok=True`, valor já aguardado, sem `RuntimeWarning` (registar
  avisos e forçar `gc.collect()`).
- `run_tools_sync(resp, [async_tool])` devolve o texto certo.
- `execute()` chamado de dentro de um loop a correr (caminho da thread do `_run_sync`) funciona.
- Caminho async com uma função síncrona que devolve uma coroutine → valor aguardado.
- `square(x: int, /)` → `ok=True`, valor `9`, em `execute` e em `async_execute`.

## Registo do dono

- Estado: review (feito na worktree `.claude/worktrees/agent-a120b7265ce857942`, por commitar).
- Ficheiros tocados:
  - `src/ai_arch_toolkit/core/_tools/_executor.py` — nova secção "Invocation":
    `_split_arguments(fn, args)` (positional-only por posição, pela ordem de `inspect.signature`, que
    segue `__wrapped__`; um positional-only omitido antes de um fornecido recebe o default; o resto
    por nome), `_invoke`, `_as_coroutine`, `_call_tool_sync` (chama; se o resultado é awaitable,
    `_run_sync(_as_coroutine(valor))`) e `_call_tool_async` (coroutine function → `await`; o resto →
    `asyncio.to_thread(_invoke, ...)` e aguarda um awaitable devolvido). Em `_run_tool_sync` e
    `_arun_tool` só muda a linha da chamada: gates, `max_calls` e metering ficam intactos para o F13.
  - `tests/test_tools_executor.py`, `tests/test_runner.py`.
- Testes novos:
  - `tests/test_tools_executor.py::TestAsyncToolOnSyncPath`: `test_group_execute_awaits_async_tool`
    (regista avisos + `gc.collect()`: nenhum "never awaited"), `test_execute_tool_awaits_async_tool`,
    `test_group_execute_inside_a_running_loop` (caminho da thread do `_run_sync`),
    `test_sync_function_returning_a_coroutine`,
    `test_sync_function_returning_a_non_coroutine_awaitable`,
    `test_async_tool_that_raises_is_a_runtime_error`.
  - `tests/test_tools_executor.py::TestAwaitableOnAsyncPath`: `test_sync_function_returning_a_coroutine`,
    `test_sync_function_returning_a_non_coroutine_awaitable`.
  - `tests/test_tools_executor.py::TestPositionalOnlyParameters`: `test_decorated_tool_on_sync_path`,
    `test_decorated_tool_on_async_path`, `test_async_tool_on_both_paths`,
    `test_undecorated_callable_on_both_paths`, `test_mixed_with_keyword_parameters`,
    `test_omitted_positional_only_parameter_takes_its_default`,
    `test_missing_required_positional_only_is_a_validation_error` (este já passava; fixa o contrato).
  - `tests/test_runner.py::TestRunToolsSync::test_async_tool_is_awaited`.
  - Antes da correcção: valor = coroutine ou objecto awaitable por aguardar (`ok=True`);
    `run_tools_sync` rebentava com `TypeError: Object of type coroutine is not JSON serializable`;
    positional-only → `validation_error`.
- Verificações: suite inteira `2658 passed, 7 skipped` (Python 3.14); `tests/test_tools_group.py`,
  `test_tools_executor.py`, `test_runner.py`, `test_core_exports.py` e `test_sync.py` também em
  Python 3.13 (125 passed, venv separado no scratchpad); `ruff check` limpo; `ruff format --check`
  limpo; `pyright src` 0 erros.
- Linhas propostas para o CHANGELOG:
  - *Fixed* — `ToolGroup.execute()`, `execute_tool()`, and `run_tools_sync()` run `async def` tools
    to completion (on a private event loop, or a worker thread that carries the caller's context
    when a loop is already running) instead of returning an un-awaited coroutine as a successful
    result; both execution paths also await an awaitable returned by a sync function.
  - *Fixed* — Tools with positional-only parameters (`def square(x: int, /)`) can be called: those
    arguments are passed by position (every call used to fail with `validation_error`).
- Desvios, decisões propostas, achados:
  - Desvio: o teste de `run_tools_sync` com tool async ficou em `tests/test_runner.py` (ficheiro do
    F04, também meu), não em `tests/test_tools_executor.py`.
  - Decisão: o caminho síncrono não pergunta `iscoroutinefunction`; chama a função e testa
    `inspect.isawaitable` no resultado — cobre `async def`, objectos com `async __call__` e funções
    síncronas que devolvem awaitables. Envolver numa coroutine é obrigatório: em Python 3.13
    `asyncio.run()` rejeita awaitables que não sejam coroutines (`ValueError: a coroutine was
    expected`, verificado em 3.13.12; em 3.14 já aceita).
  - Decisão: um positional-only com default, omitido antes de um fornecido, recebe o default
    (`def span(start=0, stop=10, /)` com `{"stop": 4}` → `span(0, 4)`); se falta um obrigatório, a
    chamada levanta o `TypeError` normal (`validation_error`).
  - Nota para o F13: a validação/coerção entra antes dos gates e depois de um `GateModify`;
    `_call_tool_sync`/`_call_tool_async` recebem os `args` finais e não precisam de mudar.
  - Proposta para F18 (não editei: `docs/tools.md:86` fica fora das secções que me foram
    atribuídas): "`execute()` also runs `async def` tools, completing them on a private event loop
    (a worker thread when called from inside a running loop, which blocks that loop until the tool
    returns) — prefer `async_execute()` in async code."
  - Limitação conhecida (sem mudança): no caminho síncrono uma tool `async def` corre noutro event
    loop; recursos presos ao loop do chamador (p. ex. uma sessão HTTP async criada nesse loop) não
    funcionam lá.
  - Achado registado em FINDINGS: um `TypeError` levantado dentro do corpo de uma tool é
    classificado como `validation_error` → F13.
- Aplicado pelo coordenador ao checkout principal em 2026-09-13; suite integrada: 2812 passed, 7 skipped; ruff e pyright limpos.
