# Phase 5: Thinking/Reasoning + Structured Output

**Status**: Queued (after Phase 4)
**Why**: Extended thinking enables better reasoning. JsonSchema forces structured responses. Both affect payload construction and response parsing.

## Thinking — Simplified Interface

`thinking` accepts `True | int | str` — no ThinkingConfig dataclass.
- `thinking=True` → provider default (Anthropic: adaptive/medium, OpenAI: medium)
- `thinking=10000` → budget_tokens (Anthropic only; OpenAI ignores)
- `thinking="high"` → effort level (OpenAI: reasoning_effort, Anthropic: adaptive effort)

## Structured Output — Tool Trick for Anthropic

**OpenAI**: `json_schema` kwarg → `response_format: {"type": "json_schema", ...}` (native)

**Anthropic**: Tool trick:
1. Create synthetic tool with the schema: `{"name": json_schema.name, "input_schema": json_schema.schema}`
2. Set `tool_choice: {"type": "tool", "name": json_schema.name}` to force usage
3. Extract `tool_call.input` as structured JSON → `response.text = json.dumps(tool_call.input)`

**Pydantic support**: If `json_schema` is a Pydantic BaseModel class, call `.model_json_schema()` internally.

## New Types (`core/_response.py`)

- `ThinkingBlock(text: str)` — frozen dataclass
- `JsonSchema(name: str, schema: dict, strict: bool = True)` — frozen dataclass
- `Response` gains `thinking: str = ""`, `thinking_blocks: tuple[ThinkingBlock, ...] = ()`

## Provider Changes

### Anthropic
- `_parse_response`: handle `"thinking"` content blocks → ThinkingBlock list
- `_build_payload`: pop `thinking` kwarg → `{"type": "enabled", "budget_tokens": N}` / `{"type": "adaptive", "effort": ...}` / `{"type": "adaptive", "effort": "medium"}`
- `_build_payload`: pop `json_schema` → tool trick (synthetic tool + tool_choice)
- `_request_headers(**kwargs) -> dict`: compute beta headers (prompt-caching)
- Stream: handle `thinking_delta` → `state.thinking` accumulator
- `_StreamState` gains `thinking: str = ""`

### OpenAI
- `_build_payload`: pop `thinking` → `reasoning_effort` (str) or ignore (int with warning)
- `_build_payload`: pop `json_schema` → `response_format: {"type": "json_schema", ...}`

### HTTP (`core/_http.py`)
- `async_post_json` + `async_stream_sse` gain `extra_headers: dict | None = None`

### LLM
- `_finalize` reads `state.thinking` for Response construction

## Tests
- `tests/test_thinking.py`: thinking=True/int/str for both providers, thinking blocks in response, stream thinking
- `tests/test_json_schema.py`: Anthropic (tool trick), OpenAI (response_format), Pydantic model input
