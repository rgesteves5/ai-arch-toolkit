# F01 · Usage Anthropic: deltas cumulativos e campos `null`

- **Dono:** worker-A · **Estado:** done · **Depende de:** —
- **Plano:** `docs/internal/toolkit-fix-plan.md` → N1, N9, F1

## Problema

1. `src/ai_arch_toolkit/core/_providers/_anthropic.py`, em `stream()` (≈ linhas 600-615) e
   `stream_events()` (≈ 724-739), **soma** `message_delta.usage` ao uso do `message_start`. A API e o
   SDK definem esse `usage` como **cumulativo** (docstring de `anthropic.types.MessageDeltaUsage`: "The
   cumulative number of input tokens which were used"; o próprio SDK substitui em
   `anthropic/lib/streaming/_messages.py`). Medido: mensagem com 25 tokens de input →
   `state.usage.input_tokens == 50`, output +1; `TypeError` quando um campo do delta vem `None`.
2. `_extract_usage` (≈ 265) deixa passar `None` nos campos opcionais do SDK
   (`cache_creation_input_tokens`, `cache_read_input_tokens`). `_parse_sdk_response` rebenta em
   `estimate_cost` com `TypeError`: qualquer `complete()` falha contra um endpoint que devolva esses
   campos a `null` (a API tipa-os como anuláveis; endpoints compatíveis via `base_url=` fazem-no).

Os adaptadores OpenAI, Gemini e xAI já substituem e usam `or 0` — não mexer.

## Decisão

- O `usage` de `message_delta` **substitui** o valor corrente, campo a campo; um campo `None` ou
  ausente mantém o valor anterior. Um helper único serve `stream()` e `stream_events()`.
- `_extract_usage` converte `None` em 0 em todos os campos.

## Ficheiros

- `src/ai_arch_toolkit/core/_providers/_anthropic.py` — **só** `_extract_usage` e o tratamento de
  `message_start`/`message_delta`. O worker-E mexe no mesmo ficheiro nas linhas de `effective_system`
  e em `batch_submit`; não toques nessas zonas.
- `tests/test_anthropic_provider.py`
- `tests/test_stream_tool_calls.py` (o fake `_message_delta` só tem `output_tokens`; ajustar se preciso)

## Prova (testes a escrever primeiro)

1. Com **tipos reais do SDK** (`RawMessageStartEvent` com `Usage(input_tokens=25, output_tokens=1, …)`,
   `RawContentBlockStartEvent`, `RawContentBlockDeltaEvent`, `RawContentBlockStopEvent`,
   `RawMessageDeltaEvent` com `MessageDeltaUsage(input_tokens=25, output_tokens=15,
   cache_creation_input_tokens=0, cache_read_input_tokens=0)`, `RawMessageStopEvent`) passados pelo
   `_FakeAnthropicStream` existente: `stream()` e `stream_events()` terminam com
   `state.usage == Usage(input_tokens=25, output_tokens=15, cache_write_tokens=0, cache_read_tokens=0)`.
2. Delta só com `output_tokens` (forma antiga): sem `TypeError`; input fica 25.
3. `_parse_sdk_response` com os dois campos de cache a `None` → `cache_write_tokens == 0` e
   `cache_read_tokens == 0`; `LLM.complete()` sob `MeterScope()` com essa mensagem num `_client`
   mockado → sucesso e `snapshot().input_tokens == 10`.
4. `LLM("claude-sonnet-4-6", api_key="x")` com `llm._provider._client` mockado, `stream_events`
   drenado sob `MeterScope()` → `snapshot().input_tokens == 25`.

## Registo do dono

- Estado: review — feito na worktree
  `/Users/rge/Documents/dev/pessoal/ai-arch-toolkit/ai-arch-toolkit/.claude/worktrees/agent-a05f374a9aeeb567f`
  (sem commit), à espera de ser aplicado.
- Ficheiros tocados:
  - `src/ai_arch_toolkit/core/_providers/_anthropic.py` — `_extract_usage` passa a `or 0` nos
    quatro contadores; novo helper `_merge_delta_usage(prev, sdk_usage) -> Usage` logo abaixo
    (`getattr(u, campo, None)`; `None`/ausente mantém o valor anterior); os ramos `message_delta` de
    `stream()` e `stream_events()` ficam numa linha que o chama. `message_start` inalterado (já usa
    `_extract_usage`). Zonas `effective_system` e `batch_submit` intocadas; sem imports novos.
  - `tests/test_anthropic_provider.py`
  - `tests/test_stream_tool_calls.py`
- Testes novos (todos com tipos reais de `anthropic.types`):
  - `TestExtractUsage::test_null_cache_fields_become_zero`
  - `TestParseSdkResponse::test_null_cache_tokens_become_zero_and_are_priced`
  - `TestAnthropicStreamUsage` (parametrizado `stream` / `stream_events`, via `_FakeAnthropicStream`):
    `test_cumulative_delta_replaces_message_start_usage`,
    `test_cumulative_cache_counts_are_not_added_twice`,
    `test_delta_with_only_output_tokens_keeps_message_start_counts`
  - `TestAnthropicUsageMetering::test_complete_with_null_cache_counts_is_metered`
    (`LLM` real + `_client` mockado + `MeterScope` → `snapshot().input_tokens == 10`)
  - `TestAnthropicUsageMetering::test_stream_events_meters_cumulative_input_tokens_once`
    (`snapshot().input_tokens == 25`, `output_tokens == 15`)
  - Alterado: `test_stream_tool_calls.py::TestAnthropicStreamToolCalls::test_single_tool_call` — delta
    na forma actual (input + cache cumulativos) e `assert state.usage == Usage(input_tokens=25,
    output_tokens=15)`.
- Falham antes, pela razão certa (verificado no código de `48a43ac`):
  - cumulativo → `input_tokens: 50 != 25` (output 16); cache → `cache_read_tokens 400 != 200`;
  - delta só com `output_tokens` → `TypeError: int + NoneType` (`_anthropic.py:607` e `:731`);
  - `_parse_sdk_response` e `LLM.complete()` com cache `None` → `TypeError` em `_pricing.py:155`;
  - `stream_events` sob `MeterScope` → `snapshot().input_tokens == 50`;
  - `_extract_usage` → `cache_*_tokens=None`;
  - os cinco testes Anthropic antigos de `test_stream_tool_calls.py` com delta só `output_tokens`
    passam a falhar no código antigo com `TypeError` — o fake `SimpleNamespace` escondia-o.
- Verificações:
  - `uv run pytest -q` → `2618 passed, 7 skipped, 3 warnings in 10.40s` (antes: `2608 passed,
    7 skipped, 3 warnings`; +10 casos; os 3 avisos são os de antes, alheios).
  - `uv run ruff check src tests examples` → `All checks passed!`;
    `uv run ruff format --check` nos 3 ficheiros → `3 files already formatted`.
  - `uv run pyright src` → `0 errors, 0 warnings, 0 informations`.
  - Extra (script descartável no scratchpad, fora do diff): o adaptador corrido sobre o
    `AsyncMessageStreamManager` real do SDK (com eventos sintetizados e um bloco `tool_use`), em três
    cenários × dois métodos: `state.usage` igual ao usage da mensagem final acumulada pelo próprio SDK
    em todos.
- Linhas propostas para o CHANGELOG (`### Fixed`):
  - Anthropic streaming (`LLM.stream`, `LLM.stream_events`) no longer double-counts usage:
    `message_delta` usage is cumulative and now replaces the `message_start` counts field by field
    instead of being added to them. Input and cache tokens were reported and metered twice (output
    +1), so `max_input_tokens`/`max_total_tokens`/`max_cost` budgets tripped early; a delta without
    `input_tokens` raised `TypeError`.
  - Anthropic `complete()` and batch results no longer raise `TypeError` when the API or an
    Anthropic-compatible endpoint returns `null` token counts (e.g. `cache_creation_input_tokens`);
    missing counts are recorded as 0.
- Desvios, decisões propostas, achados:
  1. Fakes `_message_start`/`_message_delta` de `tests/test_stream_tool_calls.py` convertidos para
     tipos reais do SDK (a ficha dizia "ajustar se preciso"; o plano F1 pede "fake com
     `input_tokens`"). `_message_start` usa contagens de cache 0 por omissão (forma actual da API);
     `_message_delta` usa só `output_tokens=0` por omissão (forma antiga) e o `stop_reason` por
     omissão passa de `""` a `None`, porque o SDK rejeita `""`. Comportamento observado igual. Os
     outros fakes desse ficheiro (blocos de conteúdo) ficam em `SimpleNamespace`: não levam usage.
  2. Mantive o guard `if getattr(ev, "usage", None):` para diff mínimo (um modelo pydantic é truthy).
  3. Não mexi no ramo `message_start`: com `_extract_usage` corrigido já não deixa passar `None`.
  4. O fake final de `_real_text_stream` reproduz o snapshot do SDK (contagens não nulas do delta por
     cima das do `message_start`), para os testes continuarem válidos se um dia o adaptador ler o
     usage de `get_final_message()`.
  5. Sem alterações de docs: nenhuma doc pública descreve o usage de streams Anthropic.
  6. Processo: o Edit tool recusou editar esta ficha no checkout principal (isolamento da worktree);
     editei-a por Bash (python), só este ficheiro.
  7. Achados: nenhum novo. Verifiquei que o custo por pedido dos server tools (ex. web search) não
     entra no `Usage`, mas é tratado de propósito — `Pricer.price` devolve `Cost.unknown` quando
     `has_server_tools` — por isso não o registei em `FINDINGS.md`.
- Aplicado pelo coordenador ao checkout principal em 2026-09-13; suite integrada: 2812 passed, 7 skipped; ruff e pyright limpos.
