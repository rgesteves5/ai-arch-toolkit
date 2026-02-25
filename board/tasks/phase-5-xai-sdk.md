# Phase 5 — xAI SDK Provider

**Status**: Complete
**Tests**: 36 new, 819 total (0 failures)

## Files Changed

| File | Action |
|------|--------|
| `core/_providers/_xai.py` | **New** — xAI SDK adapter over `xai-sdk` (gRPC) |
| `core/_providers/__init__.py` | Route `grok-*` to `XAIProvider` (was OpenAIProvider + base_url) |
| `pyproject.toml` | Added `xai-sdk>=1.7.0` to `[xai]` and `dev` deps, added `[xai]` to `[all]` |
| `tests/test_xai_provider.py` | **New** — 36 tests |

## Provider Architecture

- **SDK**: `xai-sdk` v1.7+ — gRPC-based (not REST)
- **Client**: `xai_sdk.AsyncClient(api_key=...)` with `client.chat.create(**kwargs)`
- **Completion**: `chat.sample()` returns `Response`
- **Streaming**: `chat.stream()` yields `(Response, Chunk)` tuples
- **Messages**: Built via `xai_chat.system()`, `user()`, `assistant()`, `tool_result()`
- **Assistant + tool_calls**: Proto `Message` built directly with `Content` and `ToolCall` protos

## Key Mappings

| Feature | xAI SDK |
|---------|---------|
| System message | `xai_chat.system(text)` in messages list |
| Tools | `xai_chat.tool(name, description, parameters)` |
| Thinking | `reasoning_effort` param on `create()` |
| Structured output | `ResponseFormat` proto with `FORMAT_TYPE_JSON_SCHEMA` |
| Server-side search | `search_parameters` kwarg passthrough |
| Rate limit | `grpc.StatusCode.RESOURCE_EXHAUSTED` → `RateLimitError` |
| Other errors | `grpc.aio.AioRpcError` → `APIError` via `_grpc_code_to_http()` |

## Fixes During Implementation

1. **Proto `Content` field**: `Message.content` is a list of `Content` protos, not a string
2. **Proto `ToolCall.type`**: Is an enum (`TOOL_CALL_TYPE_CLIENT_SIDE_TOOL`), not a string
3. **Stream `response` variable**: Captured as `final_response` since loop variable only exists inside loop body
4. **Ruff SIM105**: `try/except pass` → `contextlib.suppress()`
5. **Ruff B007**: Renamed unused loop var `response` → tracked via `final_response`
