# Phase 4: Gemini SDK Provider

**Status**: Complete

## What Changed

### `core/_providers/_gemini.py` — New file
- Uses `google.genai.Client` SDK (`client.aio.models.generate_content` / `generate_content_stream`)
- Pure adapter functions: `_messages_to_sdk`, `_tool_to_sdk`, `_build_thinking_config`, `_extract_usage`, `_parse_sdk_response`
- Messages: `role="assistant"` → `"model"`, tool results as `FunctionResponse` parts batched into `user` Content
- System: extracted from messages, passed via `config.system_instruction`
- Tools: wrapped in `types.Tool(function_declarations=[...])` with `FunctionDeclaration`
- Thinking: `types.ThinkingConfig(include_thoughts=True, thinking_budget=N)`, thought parts have `thought=True`
- Structured output: native `response_mime_type="application/json"` + `response_json_schema`
- Error mapping: `ClientError(code=429)` → `RateLimitError`, other `ClientError` → `APIError`, `ServerError` → `APIError`
- Streaming: `generate_content_stream()` returns chunks with full candidate structure (not deltas)
- Config: all params go into single `GenerateContentConfig` object (unlike kwargs in Anthropic/OpenAI)
- SDK auto-converts parameter dicts into `types.Schema` objects

### Tests — `test_gemini_provider.py` (41 tests)
- Message conversion (7): user, system, multi-system, assistant→model, tool_calls, tool_result, non-JSON wrap
- Tool conversion (2): basic + parameters fallback (asserts Schema object, not raw dict)
- Thinking config (4): disabled, default budget, explicit budget, effort mapping
- Usage extraction (2): basic + None values
- Response parsing (8): text, tool_calls, thinking, empty candidates, structured output, cost, raw, finish_reason
- Provider complete (9): basic, system forwarded/from messages/override, tools, thinking, output_schema, unknown/known kwargs
- Error mapping (3): rate limit (429), client error (400), server error (500)
- Streaming (5): text, tool_call, thinking, usage, error mapping
- Roundtrip (1): Response → to_message → _messages_to_sdk

### Verification
- 783 passed, 0 failures
- Ruff clean
