# Context — Core Layer Completion

## What We're Building

Completing the `core/` layer of ai-arch-toolkit so it reaches feature parity with `_legacy/`.
Once done, `_legacy/` can be removed and `core/` becomes the sole implementation.

## Architecture Rules

- **core/** never imports toolkit/ or _legacy/
- **toolkit/** imports core/ only
- **_legacy/** exists for backward compat — reference only, do not extend
- All dataclasses: `frozen=True, slots=True`
- Python 3.12+, `from __future__ import annotations` in every file
- Ruff line length: 99

## Package Layout

```
src/ai_arch_toolkit/
  core/           # Primitives — providers, LLM, response, tools, content
    _providers/   # Anthropic, OpenAI (+ future Gemini, compat)
    _tools/       # @tool decorator, ToolGroup, execute helpers
    _content.py   # Message builders (user, assistant, system, tool_result)
    _llm.py       # LLM class (complete, stream, stream_sync)
    _response.py  # Response, ToolCall, Usage, StreamResponse, SyncStreamResponse
    _pricing.py   # Cost estimation
    _http.py      # async_post_json, async_stream_sse
  toolkit/        # Convenience — runner, middleware, tokens, parsing, memory
    _runner.py    # run_tools, run_tools_sync
  _legacy/        # Old code (backward compat re-exports)
```

## Completed Phases

- **Phase 1**: Core foundation — LLM class, Response, Usage, ToolCall
- **Phase 2**: Two providers (Anthropic + OpenAI), tools layer (@tool, ToolGroup, execute)
- **Phase 3**: Package reorganization (core/ + toolkit/ + _legacy/), 684 tests passing
- **Phase 4**: Tool call streaming — Anthropic (content_block state machine) + OpenAI (per-index delta accumulation), tools param wired through LLM.stream/stream_sync, BaseProvider.stream() contract updated. 699 tests.

## Key Design Decisions

1. **Thinking interface**: `thinking=True|int|str` (not a ThinkingConfig dataclass) — avoids leaky abstraction over different provider semantics
2. **Anthropic structured output**: Tool trick (synthetic tool + tool_choice) — not system prompt injection
3. **Multimodal input types**: Live in `_content.py` (they're input types, not response types)
4. **Middleware**: Lives in `toolkit/`, wraps LLM via composition (`MiddlewareLLM`)
5. **Rich stream events**: Deprioritized — Phase 4 tool_calls on stream.response is enough for agent loops
6. **Tool result batching**: Already implemented in core `_messages_to_wire` for Anthropic

## Notes

- `_StreamState` is not frozen (mutable accumulator, one per call). Uses `__slots__`.
- Anthropic stream state machine: `content_block_start` → `input_json_delta` × N → `content_block_stop`. Closure locals `current_block` + `tool_args_acc` inside `_generate()`.
- OpenAI stream accumulation: `tc_acc: dict[int, dict]` keyed by index. Emitted on `finish_reason == "tool_calls"`, then cleared.
- `BaseProvider.stream()` now has explicit `tools` param (matching `complete()`).
- Malformed tool args fallback: Anthropic uses `{"_raw": raw_text}`, OpenAI reuses `_parse_tool_args()` which does the same.
