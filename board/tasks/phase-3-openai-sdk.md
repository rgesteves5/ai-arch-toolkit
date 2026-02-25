# Phase 3: OpenAI SDK Provider

**Status**: Complete

## What Changed

### `core/_providers/_openai.py` — Full rewrite
- Uses `openai.AsyncOpenAI` SDK client (no more raw httpx/_http.py)
- Pure adapter functions: `_messages_to_sdk`, `_tool_to_sdk`, `_parse_sdk_response`, `_extract_usage`
- New: `_build_output_schema_format()` — native `response_format` with `json_schema` type
- `thinking_effort` → `reasoning_effort` (OpenAI naming)
- `thinking_budget` → logged debug warning, ignored (not supported by OpenAI)
- Tools + output_schema coexist (unlike Anthropic)
- Error mapping: `openai.RateLimitError` → `RateLimitError`, `openai.APIStatusError` → `APIError`
- Streaming via `chat.completions.create(stream=True, stream_options={"include_usage": True})`
- Also serves xAI (Grok) via `base_url` parameter
- `max_completion_tokens` in `_SDK_PARAMS` (newer OpenAI models)

### Post-phase refactoring (shared utilities)
- `StreamState` extracted to `_base.py` — both Anthropic and OpenAI import from there
- `parse_tool_args` extracted to `_base.py` — replaces local duplicates in both providers
- Anthropic stream uses `parse_tool_args()` instead of inline `json.loads`
- Anthropic `json` import removed (no longer needed)
- Removed unused `_DEFAULT_MAX_TOKENS` from OpenAI provider
- Duplicate `TestOpenAIStreamState` merged into single `TestStreamState`

### Tests
- `test_openai_provider.py` — rewritten for SDK mocks, `max_completion_tokens` test added
- `test_stream_tool_calls.py` — OpenAI stream tests with `_oai_chunk` helpers, single `TestStreamState`
- `test_runner.py` — OpenAI roundtrip xfail removed, now passes
- All test imports updated: `StreamState` and `parse_tool_args` from `_base`

### Verification
- 742 passed, 0 failures
- Ruff clean
