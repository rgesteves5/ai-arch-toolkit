# Phase 4: Tool Call Streaming

**Status**: Done
**Why**: Agent loops that stream need tool calls during streaming. Currently both providers silently drop tool_use events. This is the #1 blocker.

## Files to Modify

| File | Action |
|------|--------|
| `src/ai_arch_toolkit/core/_providers/_anthropic.py` | Modify `_StreamState` + `stream()` + `_generate()` |
| `src/ai_arch_toolkit/core/_providers/_openai.py` | Modify `_StreamState` + `stream()` + `_generate()` |
| `src/ai_arch_toolkit/core/_llm.py` | Modify `stream()` + `stream_sync()` signatures + `_finalize` |
| `tests/test_stream_tool_calls.py` | **New** — ~12 tests |

## Changes

### 1. Anthropic Provider

**`_StreamState`**: add `tool_calls: list[ToolCall] = []`

**`stream()` signature**: add `tools: list[dict] | None = None` parameter

**`stream()` body**: pass `tools=` to `_build_payload`

**`_generate()` inner function**: add tool call accumulation state machine:
- Closure locals: `current_block: dict | None = None`, `tool_args_acc: str = ""`
- `content_block_start` → capture block metadata (type, id, name)
- `content_block_delta` → handle `input_json_delta` (accumulate partial_json) in addition to existing `text_delta`
- `content_block_stop` → if block was `tool_use`, parse accumulated JSON args, append `ToolCall` to `state.tool_calls`

### 2. OpenAI Provider

**`_StreamState`**: add `tool_calls: list[ToolCall] = []`

**`stream()` signature**: add `tools: list[dict] | None = None` parameter

**`stream()` body**: pass `tools=` to `_build_payload`

**`_generate()` inner function**: add per-index delta accumulation:
- `tc_acc: dict[int, dict[str, str]] = {}` — index → {id, name, arguments}
- For each `delta.tool_calls` entry: accumulate id, name, arguments by index
- On `finish_reason == "tool_calls"`: emit all accumulated tool calls to `state.tool_calls`, clear accumulator

### 3. LLM Class

**`stream()` signature**: add `tools: Any | None = None`

**`stream()` body**: `wire_tools = prepare_tools(tools)`, pass to `self._provider.stream()`, include `tuple(state.tool_calls)` in `_finalize`

**`stream_sync()` signature**: same `tools` parameter

**`stream_sync()` body**: same pattern, include `tool_calls` in `_finalize`

## Tests (`tests/test_stream_tool_calls.py`)

### Anthropic Tests (mock `async_stream_sse`)
- `test_single_tool_call` — content_block_start(tool_use) → input_json_delta × N → content_block_stop → verify state.tool_calls
- `test_text_then_tool_call` — text_delta then tool_use block → verify text yields AND tool_calls
- `test_multiple_tool_calls` — two tool_use blocks in sequence → verify 2 entries
- `test_malformed_tool_args` — invalid JSON in input_json_delta → verify `{"_raw": ...}` fallback
- `test_tools_passed_in_payload` — verify tools= appears in payload sent to async_stream_sse

### OpenAI Tests (mock `async_stream_sse`)
- `test_single_tool_call` — delta.tool_calls with incremental arguments → finish_reason=tool_calls → verify state.tool_calls
- `test_multiple_tool_calls` — two indices → verify 2 entries in index order
- `test_text_then_tool_call` — delta.content then delta.tool_calls → verify both
- `test_tools_passed_in_payload` — verify tools= in payload

### LLM-level Tests
- `test_stream_with_tools` — mock provider → StreamResponse.response.tool_calls after consumption
- `test_stream_sync_with_tools` — same via stream_sync()

### SSE Mock Data

**Anthropic single tool call**:
```python
events = [
    '{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tc_1","name":"get_weather"}}',
    '{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"city\\""}}',
    '{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":": \\"NYC\\"}"}}',
    '{"type":"content_block_stop","index":0}',
    '{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":15}}',
]
```

**OpenAI single tool call**:
```python
events = [
    '{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"tc_1","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}',
    '{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"city\\""}}]},"finish_reason":null}]}',
    '{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":": \\"NYC\\"}"}}]},"finish_reason":"tool_calls"}]}',
    '{"choices":[],"usage":{"prompt_tokens":25,"completion_tokens":15}}',
]
```

## Verification

1. `uv run pytest tests/test_stream_tool_calls.py` — new tests pass
2. `uv run pytest tests/test_anthropic_provider.py tests/test_openai_provider.py` — existing stream tests still pass
3. `uv run pytest tests/test_llm.py tests/test_stream_response.py` — existing tests still pass
4. `uv run pytest` — all 684+ tests pass
5. `uv run ruff check src tests && uv run ruff format --check src tests` — clean
