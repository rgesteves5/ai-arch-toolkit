# F03 · Middleware async em streaming

- **Dono:** coordenador · **Estado:** done · **Depende de:** F05 e F10 aplicados (mesmos ficheiros)
- **Plano:** A, N5, F3 · **Decisões:** D8

## Problema

`stream()`/`stream_events()` só correm os hooks síncronos (`_run_before`, `_run_after`); moderação e
memória não correm — nem o input screening que `docs/moderation.md` recomenda. O fallback de
`complete()` recebe as mensagens de antes do middleware e o `system` de depois.

## Mudanças

- `_single_stream` deixa de correr middleware e lê o pedido preparado de forma lazy (retries incluídos).
- Camada exterior: `abefore` antes de abrir o stream do provider; `aafter` depois de o candidato
  terminar e liquidar no meter, antes do `StopAsyncIteration`; o finalizer devolve essa resposta.
- Fallbacks recebem as mensagens e o `system` depois do middleware, em streaming e em `complete()`.
- D8: operação reservada na criação, iniciada na primeira tentativa; rejeição pelo middleware liberta.
- Docs: `docs/middleware.md`, `docs/moderation.md`, `docs/llm.md`, docstring do `ModerationMiddleware`.
- Testes que fixam o início na criação: `tests/test_llm_metering.py::test_stream_starts_on_build_and_settles_on_drain`,
  `::test_abandoned_stream_is_incomplete_at_scope_close`.

## Prova

- Espião dos quatro hooks: `stream`, `stream_events`, `stream_sync`, `stream_events_sync` →
  `['abefore', 'aafter']` uma vez; `aafter` recebe a resposta com `usage`.
- `ModerationMiddleware(input=sinaliza)` → `ModerationError` no primeiro `__anext__`, provider nunca
  chamado, meter sem chamadas nem custo desconhecido.
- Fallback a responder: `aafter` do primário corre uma vez; o fallback recebe mensagens pós-middleware.
- Retry antes do primeiro item não repete `abefore`; stream abandonado não corre `aafter`.

## Registo

- Estado: done (2026-09-13)
- Ficheiros tocados: `core/_llm.py` (`_PreparedStream`; `_prepare_stream_request` sem middleware;
  `_single_stream(prepared, events=…)` lê o pedido de forma lazy; `_stream_with_fallbacks` corre
  `abefore` antes de qualquer I/O e `aafter` sobre a resposta final, uma vez, à volta de primário e
  fallbacks; `_StreamRun.start_attempt` inicia a operação (D8); `_open_stream_op` só reserva;
  `complete()` passa `normalized` aos fallbacks), `toolkit/moderation/_middleware.py` (docstring),
  `docs/middleware.md`, `docs/moderation.md`, `docs/llm.md`.
- Testes novos: `tests/test_stream_middleware.py` (12). Antes: 12 falhavam.
- Testes alterados (D8): `tests/test_llm_metering.py` — `test_stream_starts_on_build…` passa a
  `test_stream_reserves_on_build_starts_on_iteration_and_settles_on_drain`;
  `test_abandoned_stream_is_incomplete…` divide-se em
  `test_never_iterated_stream_releases_its_reservation_at_scope_close` e
  `test_started_but_undrained_stream_is_incomplete_at_scope_close`;
  `tests/flow/test_flow_metering.py` — o step do teste de abandono passa a iniciar o stream, e
  `test_policy_max_cost_fails_closed_on_undrained_stream` continua a falhar fechado mas já não conta
  custo desconhecido para uma operação que nunca chegou ao provider.
- Verificações: 2825 passed, 7 skipped; testes de streaming estáveis em 3 repetições; ruff limpo;
  pyright 0 erros.
- CHANGELOG (Fixed): middleware async (`abefore`/`aafter`) corre em `stream()`, `stream_events()` e
  nos wrappers síncronos — a moderação, a memória e o rate limiting deixam de ser saltados em
  streaming; os fallbacks recebem as mensagens depois do middleware, em `complete()` e em streaming.
- CHANGELOG (Changed): a operação de metering de um stream é reservada na criação e só conta quando a
  primeira tentativa começa; um stream nunca iterado, ou rejeitado pelo middleware, liberta a reserva.
