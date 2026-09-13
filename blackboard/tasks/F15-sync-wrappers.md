# F15 · Wrappers síncronos: cancelamento e backpressure

- **Dono:** worker-D · **Estado:** done · **Depende de:** —
- **Plano:** P, F15

## Problema

- `src/ai_arch_toolkit/core/_sync.py::_run_sync`, com um loop já a correr, corre a coroutine numa
  thread e faz `thread.join(timeout)`. Ao expirar levanta `TimeoutError`, mas a coroutine continua e
  termina depois (medido: acabou 0.3 s após o timeout) — a chamada ao modelo ou a tool com efeitos
  completa-se sem ninguém à espera, e o meter liquida depois de o chamador ter seguido.
- `_stream_sync` usa uma `Queue` sem limite: um consumidor lento faz o produtor acumular o stream
  inteiro em memória. Quando o consumidor desiste, a thread só pára ao chegar o item seguinte e o
  `aclose()` da fonte fica adiado até lá; não há `join`.

## Mudança

- `_run_sync`: guardar o loop e a task da thread; no timeout, `loop.call_soon_threadsafe(task.cancel)`,
  esperar a thread até `_stream_join_timeout` e levantar `TimeoutError`. Manter a cópia de contexto.
  Tolerar a corrida com o fecho do loop (`RuntimeError`).
- `_stream_sync`: fila com limite (constante de módulo, p.ex. 256). O produtor nunca bloqueia o loop de
  fundo: `put_nowait` e, se a fila estiver cheia, `await asyncio.sleep(pequeno)` em ciclo até entrar ou
  até `stop`. Quando o consumidor abandona antes de drenar, cancelar a task de drenagem no loop de
  fundo (para o `aclose()` da fonte correr logo) e fazer `join` com `_stream_join_timeout`, mantendo o
  aviso se a thread ficar viva. Excepções da fonte continuam a chegar ao consumidor.

## Ficheiros

- `src/ai_arch_toolkit/core/_sync.py`
- `tests/test_sync.py`

## Prova

- Dentro de um loop a correr: `configure_sync_timeouts(sync_timeout=0.05)`, `_run_sync` de uma coroutine
  de 0.5 s → `TimeoutError`; a coroutine vê `CancelledError` e nunca chega ao fim (confirmar 0.7 s
  depois). Repor o timeout no teardown.
- Consumidor lento com o limite da fila reduzido a 2 (monkeypatch): itens produzidos ≤ consumidos +
  limite + 1 em qualquer momento.
- Abandonar depois do primeiro item, com a fonte a dormir 5 s entre itens: o `finally` da fonte corre e
  a thread termina em menos de 1 s.
- Uma excepção da fonte chega ao consumidor.
- Os testes existentes de propagação de `ContextVar` continuam verdes.
- Margens de tempo generosas: a suite pode correr em paralelo com a de outros agentes.

## Registo do dono

- Estado: review — feito na worktree `.claude/worktrees/agent-a4533943325128dee`, por commitar.
- Ficheiros tocados: `src/ai_arch_toolkit/core/_sync.py`, `tests/test_sync.py`.
- Testes novos (12, todos em `tests/test_sync.py`; os três testes de `ContextVar` existentes ficam
  intactos e verdes):
  - `TestRunSyncTimeout::test_timeout_cancels_the_coroutine_before_raising`
  - `TestRunSyncTimeout::test_timeout_before_the_worker_starts_means_the_coroutine_never_runs`
  - `TestRunSyncTimeout::test_timeout_bounds_the_wait_for_a_coroutine_that_ignores_cancellation`
  - `TestStreamSync::test_error_reaches_a_slow_consumer_after_every_item`
  - `TestStreamSync::test_error_after_the_worker_loop_closed_still_reaches_the_consumer`
  - `TestStreamSyncBackpressure::test_producer_runs_at_most_one_item_past_a_full_queue`
  - `TestStreamSyncBackpressure::test_full_queue_does_not_block_the_worker_loop`
  - `TestStreamSyncAbandonment::test_abandoning_cancels_a_pending_source_and_joins_the_worker`
  - `TestStreamSyncAbandonment::test_abandoning_with_a_full_queue_stops_the_producer`
  - `TestStreamSyncAbandonment::test_abandoning_does_not_interrupt_a_source_already_cleaning_up`
  - `TestStreamSyncAbandonment::test_abandoning_frees_a_worker_holding_an_unread_error`
  - `TestStreamSyncAbandonment::test_stream_closed_on_its_own_worker_thread_does_not_join_itself`
- Prova vermelha (código de `48a43ac` só com a constante `_STREAM_QUEUE_MAXSIZE`, para o monkeypatch
  encontrar o atributo): 5 falhas pela razão certa — a coroutine nunca é cancelada (2 testes); o
  produtor chega 49 itens à frente (limite 3); a fila nunca enche, o produtor acaba e o heartbeat
  pára; o `finally` da fonte não corre em 1 s. Os outros testes novos guardam o código novo.
- Mutações: 10 mutações do código novo (sem `detach`, `attach` falhado ignorado, sem `join` depois
  de cancelar, `put` bloqueante no loop, `put` simples do erro no `_target`, sem guarda da thread
  corrente, stream sem cancelar, `_run_sync` sem cancelar, fila sem limite, `RuntimeError` do loop
  fechado não tolerado): cada uma faz falhar o teste que a guarda.
- Robustez: `tests/test_sync.py` 8 vezes seguidas e 8 execuções em paralelo com 36 processos a
  queimar CPU (12 núcleos): sempre verde, cerca de 3 s.
- Verificações: `uv run pytest -q` → `2620 passed, 7 skipped, 3 warnings in 13.40s` (baseline 2608 +
  12; os 3 avisos são anteriores: genai e `TestAstra`); `ruff check src tests examples` limpo;
  `ruff format --check` nos dois ficheiros limpo; `pyright src` 0 erros.
- Medido pela API pública (script descartável, `sync_timeout=0.1`, antes → depois):
  - `LLM.complete_sync()` dentro de um loop: `TimeoutError` a 0.101 s com a chamada ao provider a
    terminar na mesma e a op do meter só resolvida depois → `TimeoutError` a 0.111 s, provider
    cancelado, op já falhada (custo desconhecido) quando o chamador recebe o erro.
  - `stream_sync()` abandonado a meio: `aclose()` da fonte 5.0 s depois do `close()` → 0.0001 s.
  - `Flow.iter_sync()` abandonado durante um step de 5 s: step desenrolado aos 5.0 s → imediato.
- Linhas propostas para o CHANGELOG:
  - Fixed: Sync calls made while an event loop is already running (`LLM.complete_sync()`,
    `Flow.run_sync()`, `Agent.run_sync()`, …) now cancel their coroutine when the sync timeout
    expires and wait up to the stream join timeout for it to unwind before raising `TimeoutError`;
    the model call or tool no longer completes, and settles its meter, after the caller has moved on.
  - Fixed: Sync streams (`LLM.stream_sync()`, `LLM.stream_events_sync()`, `Flow.iter_sync()`)
    buffer at most 256 items ahead of a slow consumer instead of the whole stream, and a consumer
    that stops early cancels the pending read, so the source closes at once instead of at its next
    item.
  - Changed: `configure_sync_timeouts(stream_join_timeout=)` (`AI_ARCH_STREAM_JOIN_TIMEOUT`) now also
    bounds the wait for the worker thread of an abandoned sync stream and of a timed-out sync call,
    with a warning if it is still alive. Closing a sync stream early can block for up to that long
    (5 s by default) while its source finishes blocking work, such as a sync tool in a thread.
- Desvios e decisões:
  1. Classe privada `_TaskHandle` em vez de `nonlocal loop, task` nas closures. O pyright estreita
     as variáveis capturadas para `None` no âmbito exterior e deixava os ramos de cancelamento por
     verificar (confirmado plantando erros numa cópia). A classe dá às duas funções o mesmo aperto
     de mão: o worker faz `attach()` sob lock; um `cancel()` que chegue antes faz o `attach()` falhar
     e a coroutine nunca começa (`coro.close()`, sem aviso "never awaited").
  2. O dreno faz `detach()` antes de fechar a fonte: um cancelamento que chegue depois de o dreno ter
     visto `stop` não interrompe o `aclose()`/`finally` da fonte. O callback corre no loop do dreno,
     por isso a ordem face ao `detach()` é determinística.
  3. `join` em todas as saídas do consumidor (antes só depois de drenar), excepto quando o `finally`
     corre na própria thread do dreno (gerador finalizado pelo GC num ciclo): juntar-se a si própria
     levantaria `RuntimeError`.
  4. O `_target` deixou de pôr o `_SENTINEL` depois de uma excepção (o consumidor levanta e nunca o
     lia). A excepção entra na fila com uma espera que desiste quando o consumidor sai; com a fila
     cheia, um `put` simples prenderia a thread para sempre (há teste).
  5. A espera de `_run_sync` depois de cancelar usa `_stream_join_timeout`, como a ficha pede, com
     aviso no log se a thread continuar viva; a mensagem do `TimeoutError` não muda.
  6. Prova da ficha com `sync_timeout=0.2` (e não `0.05`) nos dois testes que precisam de a coroutine
     já ter começado quando o prazo expira: margem contra arranque lento da thread sob carga. O teste
     do prazo antes do arranque usa `0.05`.
  7. `configure_sync_timeouts` ganhou docstring com `Args` e os env vars: é API pública e o
     `stream_join_timeout` passou a cobrir mais casos.
  8. Com a fila cheia o produtor tenta de novo a cada `_STREAM_PUT_INTERVAL = 0.005` s (constante de
     módulo, ao lado de `_STREAM_QUEUE_MAXSIZE = 256`).
- Riscos para o coordenador:
  - (medido) Abandonar `Flow.iter_sync()` enquanto um step corre trabalho bloqueante em
    `asyncio.to_thread` (tools síncronas): a teardown do `asyncio.run` do dreno espera pelo executor,
    por isso `close()` bloqueia até a tool acabar, no máximo `stream_join_timeout` (5 s), com aviso.
    Antes voltava logo. O mesmo acrescenta até 5 s ao `TimeoutError` de `_run_sync`. É a consequência
    do `join` pedido; alternativa, se se preferir: esperar só pelo fim da task de drenagem (um
    `Event` no `finally` de `_drain`) e deixar a teardown do loop correr em fundo.
  - Um consumidor que abandona depois do último item mas antes do sentinel pode fazer o cancelamento
    cair no `finally` de fim de stream da própria fonte (semântica normal de cancelamento).
  - Um produtor com a fila cheia acorda a cada 5 ms enquanto o consumidor está parado.
- Achados: acrescentado a `FINDINGS.md` — `_stream_sync` levanta qualquer item que seja instância de
  `BaseException`, mesmo quando a fonte o produz como valor (anterior a F15, não alterado).
- Aplicado pelo coordenador ao checkout principal em 2026-09-13; suite integrada: 2867 passed, 7 skipped; ruff e pyright limpos; `tests/test_sync.py` estável em 3 repetições junto com os testes do motor.
