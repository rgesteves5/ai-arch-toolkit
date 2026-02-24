# Phase 8: Rich Streaming Events

**Status**: Queued (after Phase 6, independent of Phase 7)
**Why**: StreamEvent enables UI frameworks to react to tool calls mid-stream with fine-grained event types. Not a blocker for agent loops (Phase 4 provides tool_calls on stream.response), but needed for real-time UI.

## StreamEvent (`core/_response.py`)

- `StreamEvent(type: str, text: str = "", tool_call: ToolCall | None = None, thinking: str = "", usage: Usage | None = None)`
- Types: `"text"`, `"tool_call"`, `"thinking"`, `"usage"`, `"done"`

## Provider Changes

- `BaseProvider` gains abstract `stream_events()` → `tuple[AsyncIterator[StreamEvent], Any]`
- **Anthropic**: full state machine (block_start → delta → block_stop)
  - `text_delta` → `StreamEvent(type="text")`
  - `thinking_delta` → `StreamEvent(type="thinking")`
  - `input_json_delta` → accumulate; on `content_block_stop` → `StreamEvent(type="tool_call")`
  - `message_delta` → `StreamEvent(type="usage")`
  - `message_stop` → `StreamEvent(type="done")`
- **OpenAI**: delta accumulation, emit on finish_reason
  - `delta.content` → `StreamEvent(type="text")`
  - `finish_reason == "tool_calls"` → `StreamEvent(type="tool_call")` per accumulated call
  - usage chunk → `StreamEvent(type="usage")`

## EventStreamResponse (`core/_response.py`)

- Like StreamResponse but yields `StreamEvent`
- Accumulates final `Response` from events
- Context manager support (same pattern as StreamResponse)

## LLM Integration

- `LLM.stream_events()` → returns EventStreamResponse
- `LLM.stream_events_sync()` → sync variant

## Tests: `tests/test_stream_events.py`

- Event types/payloads for both providers
- EventStreamResponse accumulation → correct Response
- Context manager early exit
- Mixed text + tool_call + thinking events
