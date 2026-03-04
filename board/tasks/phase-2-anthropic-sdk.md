# Phase 2: Anthropic SDK Provider

**Status**: Complete

## What Changed

### `core/_providers/_anthropic.py` — Full rewrite
- Uses `anthropic.AsyncAnthropic` SDK client (no more raw httpx/_http.py)
- Pure adapter functions: `_messages_to_sdk`, `_tool_to_sdk`, `_parse_sdk_response`, `_extract_usage`
- New: `_build_thinking_param()` — maps thinking/effort/budget → SDK thinking config
- New: `_build_output_schema_tool()` — tool trick for structured output
- Error mapping: `anthropic.RateLimitError` → `RateLimitError`, `anthropic.APIStatusError` → `APIError`
- Streaming via `client.messages.stream()` context manager
- Thinking blocks extracted from `get_final_message()` after stream completes
- Temperature auto-removed when thinking enabled (Anthropic requirement)

### Tests
- `test_anthropic_provider.py` — rewritten to mock SDK client (not HTTP)
- `test_stream_tool_calls.py` — Anthropic tests use `_FakeStream` helper
- `TestAnthropicStreamThinking` — covers thinking accumulation via SDK stream
- `test_runner.py` — OpenAI roundtrip marked `xfail`, Anthropic import updated

### Verification
- 697 passed, 1 xfail (OpenAI roundtrip — Phase 3)
- Ruff clean
