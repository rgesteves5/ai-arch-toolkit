# Board — SDK Provider Rewrite

> Full plan: see `.claude/plans/keen-watching-horizon.md`

## Completed

**Phase 1: Foundation** → [tasks/phase-1-foundation.md](tasks/phase-1-foundation.md)
- OutputSchema, ThinkingBlock types added
- LLM class gains thinking + output_schema params
- `_http.py` deleted, SDKs as optional deps
- Provider registry updated (xAI, Gemini prefixes)
- 76 core tests pass

**Phase 2: Anthropic SDK Provider** → [tasks/phase-2-anthropic-sdk.md](tasks/phase-2-anthropic-sdk.md)
- `_anthropic.py` rewritten as thin SDK adapter
- Thinking, structured output (tool trick), error mapping
- Stream tool calls + thinking accumulation via SDK
- 697 tests pass, 1 xfail (OpenAI roundtrip)

**Phase 3: OpenAI SDK Provider** → [tasks/phase-3-openai-sdk.md](tasks/phase-3-openai-sdk.md)
- `_openai.py` rewritten as thin SDK adapter
- Native structured output via `response_format`
- `thinking_effort` → `reasoning_effort` mapping
- 742 tests pass, 0 xfail

**Phase 4: Gemini SDK Provider** → [tasks/phase-4-gemini-sdk.md](tasks/phase-4-gemini-sdk.md)
- `_gemini.py` new SDK adapter over `google-genai`
- Native thinking (`ThinkingConfig`) + structured output (`response_json_schema`)
- `assistant` → `model` role mapping, tool results as `FunctionResponse` parts
- 783 tests pass, 0 failures

**Phase 5: xAI SDK Provider** → [tasks/phase-5-xai-sdk.md](tasks/phase-5-xai-sdk.md)
- `_xai.py` new SDK adapter over `xai-sdk` (gRPC-based)
- Native reasoning (`reasoning_effort`), structured output (`ResponseFormat` proto)
- Server-side search passthrough, proto-based message construction
- 819 tests pass (36 new xAI tests), 0 failures

**Phase 6: Integration + Cleanup** → [tasks/phase-6-integration.md](tasks/phase-6-integration.md)
- Removed Mistral from scope (registry, pyproject.toml, tests)
- Fixed xAI `thinking=True` defaulting to `"high"` reasoning effort
- Added code comments (response.content asymmetry, per-chunk tool calls)
- Verified: no `_http.py`, no `requests`/`httpx` in core/, exports complete
- 817 tests pass, 0 failures

## In Progress

_(none)_

## Next

_(SDK rewrite complete)_

## Future

- Anthropic native structured output (replace tool trick)
- Rich Streaming Events (StreamEvent type, real-time thinking)
- Toolkit layer (middleware, token estimation, memory, templates)
- Batch + Fallback
