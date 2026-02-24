# Core Layer Completion Plan

## Context

Phases 1-3 established the core/ foundation: LLM class, two providers (Anthropic + OpenAI),
tools layer, and package reorganization (core/ + toolkit/ + _legacy/). 684 tests pass.

This plan covers **all remaining gaps** between core/ and legacy, organized into phases
that build on each other. Each phase is independently testable and committable.

**Rules**: core/ never imports toolkit/ or _legacy/. toolkit/ imports core/ only.
All dataclasses frozen=True, slots=True. Python 3.12+.

**Note**: Tool result batching for Anthropic (consecutive tool_results → single user message)
is already implemented in core `_messages_to_wire`. Not a gap.

---

## Phase 4: Tool Call Streaming (DETAILED)

**Why**: Agent loops that stream need tool calls during streaming. Currently both
providers silently drop tool_use events. This is the #1 blocker.

### 1. `core/_providers/_anthropic.py` — Stream tool call accumulation

**`_StreamState`** (line 25-34): add `tool_calls` slot

```python
class _StreamState:
    __slots__ = ("model", "raw", "stop_reason", "tool_calls", "usage")

    def __init__(self) -> None:
        self.usage: Usage | None = None
        self.model: str = ""
        self.stop_reason: str = ""
        self.raw: dict[str, Any] | None = None
        self.tool_calls: list[ToolCall] = []
```

**`stream()` signature** (line 229-234): add `tools` parameter

```python
def stream(
    self,
    messages: list[dict[str, Any]],
    *,
    system: str | None = None,
    tools: list[dict[str, Any]] | None = None,  # NEW
    **kwargs: Any,
) -> tuple[AsyncIterator[str], _StreamState]:
```

**`stream()` body** (line 236-239): pass `tools=` to `_build_payload`

```python
payload = _build_payload(
    wire, model=self._model, system=effective_system, tools=tools, **kwargs
)
```

**`_generate()` inner function** (line 245-289): add tool call accumulation

State machine (closure locals inside `_generate()`):
- `current_block: dict[str, Any] | None = None`
- `tool_args_acc: str = ""`

New event handling added to existing `_generate()`:

```python
async def _generate() -> AsyncIterator[str]:
    current_block: dict[str, Any] | None = None
    tool_args_acc = ""

    async for data in async_stream_sse(...):
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type")

        if event_type == "content_block_start":
            current_block = event.get("content_block", {})
            tool_args_acc = ""

        elif event_type == "content_block_delta":
            delta = event.get("delta", {})
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                text = delta.get("text", "")
                if text:
                    yield text
            elif delta_type == "input_json_delta":
                tool_args_acc += delta.get("partial_json", "")

        elif event_type == "content_block_stop":
            if current_block and current_block.get("type") == "tool_use":
                args: dict[str, Any] = {}
                if tool_args_acc:
                    try:
                        args = json.loads(tool_args_acc)
                    except json.JSONDecodeError:
                        args = {"_raw": tool_args_acc}
                state.tool_calls.append(
                    ToolCall(
                        id=current_block.get("id", ""),
                        name=current_block.get("name", ""),
                        input=args,
                    )
                )
            current_block = None
            tool_args_acc = ""

        elif event_type == "message_start":
            # ... existing code ...

        elif event_type == "message_delta":
            # ... existing code ...
```

Key: `content_block_start` and `content_block_stop` are **new** event handlers.
`content_block_delta` is **expanded** (was only text_delta, now also input_json_delta).
`message_start` and `message_delta` handlers remain unchanged.

### 2. `core/_providers/_openai.py` — Stream tool call accumulation

**`_StreamState`** (line 33-43): add `tool_calls` slot (same pattern as Anthropic)

**`stream()` signature** (line 232-238): add `tools` parameter

```python
def stream(
    self,
    messages: list[dict[str, Any]],
    *,
    system: str | None = None,
    tools: list[dict[str, Any]] | None = None,  # NEW
    **kwargs: Any,
) -> tuple[AsyncIterator[str], _StreamState]:
```

**`stream()` body** (line 240-241): pass `tools=` to `_build_payload`

```python
payload = _build_payload(wire, model=self._model, tools=tools, **kwargs)
```

**`_generate()` inner function** (line 248-286): add tool call delta accumulation

```python
async def _generate() -> AsyncIterator[str]:
    tc_acc: dict[int, dict[str, str]] = {}  # NEW: index → {id, name, arguments}

    async for data in async_stream_sse(...):
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        # Usage chunk (unchanged)
        if raw_usage := chunk.get("usage"):
            state.usage = Usage(...)
            continue

        choices = chunk.get("choices", [])
        if not choices:
            continue

        choice = choices[0]
        delta = choice.get("delta", {})

        if text := delta.get("content"):
            yield text

        # NEW: accumulate tool call deltas
        for tc_delta in delta.get("tool_calls", []):
            idx = tc_delta.get("index", 0)
            if idx not in tc_acc:
                tc_acc[idx] = {
                    "id": tc_delta.get("id", ""),
                    "name": tc_delta.get("function", {}).get("name", ""),
                    "arguments": "",
                }
            else:
                if tc_id := tc_delta.get("id"):
                    tc_acc[idx]["id"] = tc_id
                if fn_name := tc_delta.get("function", {}).get("name"):
                    tc_acc[idx]["name"] = fn_name
            tc_acc[idx]["arguments"] += (
                tc_delta.get("function", {}).get("arguments", "")
            )

        # NEW: emit completed tool calls
        if choice.get("finish_reason") == "tool_calls":
            for _idx in sorted(tc_acc):
                acc = tc_acc[_idx]
                state.tool_calls.append(
                    ToolCall(
                        id=acc["id"],
                        name=acc["name"],
                        input=_parse_tool_args(acc["arguments"]),
                    )
                )
            tc_acc.clear()

        if finish := choice.get("finish_reason"):
            state.stop_reason = finish

        if model_name := chunk.get("model"):
            state.model = model_name
```

### 3. `core/_llm.py` — Wire tool calls through stream

**`stream()` signature** (line 103-108): add `tools` parameter

```python
def stream(
    self,
    messages: str | list[dict[str, Any]],
    *,
    system: str | None = None,
    tools: Any | None = None,  # NEW
    **kwargs: Any,
) -> StreamResponse:
```

**`stream()` body** (line 110-134): pass tools to provider, include tool_calls in _finalize

```python
normalized = self._normalize(messages)
merged = self._merge_kwargs(**kwargs)
wire_tools = prepare_tools(tools)  # NEW: same as complete()
aiter, state = self._provider.stream(
    normalized, system=system, tools=wire_tools, **merged  # NEW: tools=
)
model = self._model

def _finalize(text: str) -> Response:
    usage = state.usage or Usage()
    cost, cost_estimated = pricing.estimate_cost(...)
    return Response(
        text=text,
        tool_calls=tuple(state.tool_calls),  # NEW
        usage=usage,
        cost=cost,
        cost_estimated=cost_estimated,
        stop_reason=state.stop_reason,
        model=state.model or model,
    )

return StreamResponse(aiter, _finalize)
```

**`stream_sync()` signature** (line 159-165): add `tools` parameter (same as stream)

```python
def stream_sync(
    self,
    messages: str | list[dict[str, Any]],
    *,
    system: str | None = None,
    tools: Any | None = None,  # NEW
    **kwargs: Any,
) -> SyncStreamResponse:
```

**`stream_sync()` body** (line 167-202): pass tools, include tool_calls in _finalize

```python
normalized = self._normalize(messages)
merged = self._merge_kwargs(**kwargs)
wire_tools = prepare_tools(tools)  # NEW

state_holder: list[Any] = []
model = self._model

def _async_factory():
    aiter, state = self._provider.stream(
        normalized, system=system, tools=wire_tools, **merged  # NEW: tools=
    )
    state_holder.append(state)
    return aiter

sync_iter = _stream_sync(_async_factory)

def _finalize(text: str) -> Response:
    state = state_holder[0] if state_holder else None
    usage = (state.usage if state else None) or Usage()
    tool_calls = tuple(state.tool_calls) if state else ()  # NEW
    cost, cost_estimated = pricing.estimate_cost(...)
    return Response(
        text=text,
        tool_calls=tool_calls,  # NEW
        usage=usage,
        ...
    )

return SyncStreamResponse(sync_iter, _finalize)
```

### 4. Tests: `tests/test_stream_tool_calls.py`

**Anthropic tests (mock `async_stream_sse`)**:

```python
class TestAnthropicStreamToolCalls:
    async def test_single_tool_call():
        # SSE events: content_block_start(tool_use) → input_json_delta × N → content_block_stop
        # Assert state.tool_calls has 1 ToolCall with correct id/name/input

    async def test_text_then_tool_call():
        # SSE: text_delta → content_block_start(tool_use) → json_delta → stop
        # Assert text yields AND tool_calls populated

    async def test_multiple_tool_calls():
        # Two tool_use blocks in sequence
        # Assert state.tool_calls has 2 entries

    async def test_malformed_tool_args():
        # input_json_delta with invalid JSON
        # Assert {"_raw": ...} fallback

    async def test_tools_passed_in_payload():
        # Assert tools= appears in the payload sent to async_stream_sse

    async def test_stream_response_has_tool_calls():
        # Full LLM.stream() → consume → stream.response.tool_calls populated
```

**OpenAI tests (mock `async_stream_sse`)**:

```python
class TestOpenAIStreamToolCalls:
    async def test_single_tool_call():
        # delta.tool_calls with incremental arguments → finish_reason=tool_calls
        # Assert state.tool_calls has 1 ToolCall

    async def test_multiple_tool_calls():
        # Two tool calls with different indices
        # Assert state.tool_calls has 2 entries in index order

    async def test_text_then_tool_call():
        # delta.content first, then delta.tool_calls
        # Assert text yields AND tool_calls populated

    async def test_tools_passed_in_payload():
        # Assert tools= appears in payload

    async def test_stream_response_has_tool_calls():
        # Full LLM.stream() → consume → stream.response.tool_calls populated
```

**LLM-level integration tests**:

```python
class TestLLMStreamToolCalls:
    async def test_stream_with_tools():
        # Mock provider, return stream with tool calls in state
        # Assert StreamResponse.response.tool_calls after consumption

    def test_stream_sync_with_tools():
        # Same but via stream_sync()
        # Assert SyncStreamResponse.response.tool_calls
```

### SSE Mock Data Examples

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

### Files Summary

| File | Action |
|------|--------|
| `src/ai_arch_toolkit/core/_providers/_anthropic.py` | Modify _StreamState + stream() + _generate() |
| `src/ai_arch_toolkit/core/_providers/_openai.py` | Modify _StreamState + stream() + _generate() |
| `src/ai_arch_toolkit/core/_llm.py` | Modify stream() + stream_sync() signatures + _finalize |
| `tests/test_stream_tool_calls.py` | **New** — ~12 tests |

### Verification

1. `uv run pytest tests/test_stream_tool_calls.py` — new tests pass
2. `uv run pytest tests/test_anthropic_provider.py tests/test_openai_provider.py` — existing stream tests still pass
3. `uv run pytest tests/test_llm.py` — existing LLM stream tests still pass
4. `uv run pytest tests/test_stream_response.py` — StreamResponse tests still pass
5. `uv run pytest` — all 684+ tests pass
6. `uv run ruff check src tests && uv run ruff format --check src tests` — clean

---

## Phase 5: Thinking/Reasoning + Structured Output

**Why**: Extended thinking enables better reasoning. JsonSchema forces structured
responses. Both affect payload construction and response parsing.

### Thinking — simplified interface (no dataclass)

**User API**: `thinking` accepts `True | int | str` — no ThinkingConfig dataclass.
- `thinking=True` → provider default (Anthropic: adaptive/medium, OpenAI: medium)
- `thinking=10000` → budget_tokens (Anthropic only; OpenAI ignores)
- `thinking="high"` → effort level (OpenAI: reasoning_effort, Anthropic: adaptive effort)

This avoids a leaky abstraction trying to unify different provider semantics.

### New Types (`core/_response.py`)
- `ThinkingBlock(text: str)` — frozen dataclass
- `JsonSchema(name: str, schema: dict[str, Any], strict: bool = True)` — frozen dataclass
- `Response` gains `thinking: str = ""`, `thinking_blocks: tuple[ThinkingBlock, ...] = ()`

### Structured Output — tool trick for Anthropic

**OpenAI**: `json_schema` kwarg → `response_format: {"type": "json_schema", ...}` (native)

**Anthropic**: Use the **tool trick** (not system prompt injection, which is unreliable):
1. Create a synthetic tool with the schema: `{"name": json_schema.name, "input_schema": json_schema.schema}`
2. Set `tool_choice: {"type": "tool", "name": json_schema.name}` to force the LLM to use it
3. Extract `tool_call.input` as the structured JSON response
4. Return as `response.text = json.dumps(tool_call.input)`

**Accept Pydantic models**: If `json_schema` is a Pydantic BaseModel class, call
`.model_json_schema()` internally and use the class name as the schema name.

### Provider Changes

**Anthropic**
- `_parse_response`: handle `"thinking"` content blocks → ThinkingBlock list
- `_build_payload`: pop `thinking` kwarg; if `int` → `{"type": "enabled", "budget_tokens": N}`;
  if `str` → `{"type": "adaptive", "effort": thinking}`; if `True` → `{"type": "adaptive", "effort": "medium"}`
- `_build_payload`: pop `json_schema` → tool trick (synthetic tool + tool_choice)
- `_build_payload`: pop `cache_control` → beta header via `_request_headers()`
- `_request_headers(**kwargs) -> dict`: computes beta headers (prompt-caching, computer-use)
- Stream: handle `thinking_delta` events → `state.thinking` accumulator
- `_StreamState` gains `thinking: str = ""`

**OpenAI**
- `_build_payload`: pop `thinking`; if `int` → ignored (log warning); if `str` → `reasoning_effort`;
  if `True` → `reasoning_effort: "medium"`
- `_build_payload`: pop `json_schema` → `response_format: {"type": "json_schema", ...}`

**`core/_http.py`**
- `async_post_json` + `async_stream_sse` gain `extra_headers: dict | None = None`
- When `client` provided: merge extra_headers into per-request headers

**`core/_llm.py`**
- `_finalize` reads `state.thinking` for Response construction

### Tests
- `tests/test_thinking.py`: thinking=True/int/str for both providers, thinking blocks in response, stream thinking
- `tests/test_json_schema.py`: Anthropic (tool trick), OpenAI (response_format), Pydantic model input

---

## Phase 6: Multimodal Content

**Why**: Vision and document agents need images/PDFs in messages.

### New Types (`core/_content.py` — input types, NOT _response.py)

These are **input** types (go in messages), not output types. They live alongside
the builder functions in `_content.py`.

- `TextPart(text: str)` — frozen dataclass
- `ImagePart(url: str = "", media_type: str = "", data: str = "", detail: str = "auto")` — frozen
- `AudioPart(data: str = "", media_type: str = "", transcript: str = "")` — frozen
- `DocumentPart(data: str = "", media_type: str = "application/pdf", uri: str = "")` — frozen
- `type ContentPart = TextPart | ImagePart | AudioPart | DocumentPart`
- `type Content = str | tuple[ContentPart, ...]`

### Content Builders (`core/_content.py`)
- `image_block(url=None, data=None, media_type=None, detail="auto") -> ImagePart`
- `audio_block(data, media_type, transcript="") -> AudioPart`
- `document_block(data=None, media_type="application/pdf", uri=None) -> DocumentPart`
- `user()` signature: `str | list[str | ContentPart]` (explicit typing update)

### Provider Wire Format
- **Anthropic** `_content_to_anthropic()`: ImagePart→base64/URL source, DocumentPart→base64, AudioPart→ValueError
- **OpenAI** `_content_to_openai()`: ImagePart→image_url, AudioPart→input_audio, DocumentPart→ValueError
- Both: `_messages_to_wire` detects non-string content via `isinstance(content, (list, tuple))`

### Tests: `tests/test_multimodal.py`
- Each part type through each provider's wire converter
- Error cases (AudioPart→Anthropic, DocumentPart→OpenAI)
- Round-trip: user(ImagePart) → _messages_to_wire → correct format

---

## Phase 7: Toolkit Utilities + Middleware

**Why**: Standalone utilities for production use. Middleware is a convenience wrapper
pattern (before/after hooks) — it's opinionated pipeline structure, not a core primitive.
Both belong in toolkit/.

### Token Estimation (`toolkit/_tokens.py`)
- Correction factors per model family (Claude 3=1.12, Claude 4=1.15, Gemini=1.05, etc.)
- `raw_tiktoken_count()` → tiktoken with fallback to len//4
- `estimate_text_tokens` → `estimate_content_tokens` → `estimate_message_tokens` → `estimate_conversation_tokens`
- All have `_for_model` variants with correction factor

### Output Parsing (`toolkit/_output_parsing.py`)
- `parse_json()` — 3-stage: direct → code block → JSON snippet scan
- `parse_json_as[T]()` — generic coercion to dataclass/dict/type
- `extract_code_block()`, `extract_list()`

### Prompt Templates (`toolkit/_templates.py`) — optional convenience, not central
- `PromptTemplate(template)` — frozen, `.format(**kwargs) -> str`
- `ChatTemplate(messages)` — frozen, `.format_messages(**kwargs) -> list[dict]`

### Conversation Memory (`toolkit/_memory.py`) — optional, one approach among many
- `ConversationMemory(items)` — mutable, add/extend/history/clear/token_count
- `SlidingWindowMemory(max_tokens)` — trims oldest after each mutation
- Uses `estimate_conversation_tokens` from `_tokens.py`

### Middleware (`toolkit/_middleware.py`)
- `Request` dataclass (mutable, slots=True): operation, provider, model, messages, system, tools, json_schema, kwargs, context
- `Middleware` Protocol: `before`/`after`/`abefore`/`aafter`
- `_run_before(middlewares, request)` / `_run_after(middlewares, request, result)`
- Short-circuit via `context["_short_circuit_result"]`

### MiddlewareLLM (`toolkit/_middleware_llm.py`)
- `MiddlewareLLM(llm: LLM, middleware: list[Middleware])` — wraps an LLM with middleware pipeline
- `complete()`: Request → _run_before → short-circuit check → llm.complete → _run_after
- `stream()` / `stream_events()`: before hooks run first; after hooks wrap the response
- Does NOT modify core LLM — composition pattern

### Middleware Implementations
- `toolkit/_cost.py`: CostTracker — accumulates usage/cost via after hook; `snapshot()` → CostSnapshot
- `toolkit/_cache.py`: ResponseCache — SHA-256 key from request; InMemoryCacheBackend with TTL; short-circuits on hit
- `toolkit/_tracing.py`: TracingMiddleware — OpenTelemetry spans (optional dep, no-op if missing)
- `toolkit/_guardrails.py`: GuardrailMiddleware — regex pattern blocking + input/output validators

### Tests
- `tests/test_toolkit_tokens.py`, `tests/test_toolkit_parsing.py`
- `tests/test_toolkit_templates.py`, `tests/test_toolkit_memory.py`
- `tests/test_toolkit_middleware.py`: hook ordering, short-circuit, composition
- `tests/test_toolkit_cost.py`, `tests/test_toolkit_cache.py`, `tests/test_toolkit_guardrails.py`

---

## Phase 8: Rich Streaming Events

**Why**: StreamEvent enables UI frameworks to react to tool calls mid-stream
with fine-grained event types. Not a blocker for agent loops (Phase 4 already
provides tool_calls on stream.response), but needed for real-time UI.

Must be core (providers emit events from raw SSE — toolkit can't fabricate them).

### StreamEvent (`core/_response.py`)
- `StreamEvent(type: str, text: str = "", tool_call: ToolCall | None = None, thinking: str = "", usage: Usage | None = None)`
- Types: `"text"`, `"tool_call"`, `"thinking"`, `"usage"`, `"done"`

### Provider Changes
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

### EventStreamResponse (`core/_response.py`)
- Like StreamResponse but yields `StreamEvent`
- Accumulates final `Response` from events (text from text events, tool_calls from tool_call events, thinking from thinking events)
- Context manager support (same pattern as StreamResponse)

### LLM Integration
- `LLM.stream_events()` → returns EventStreamResponse
- `LLM.stream_events_sync()` → sync variant

### Tests: `tests/test_stream_events.py`
- Event types/payloads for both providers
- EventStreamResponse accumulation → correct Response
- Context manager early exit
- Mixed text + tool_call + thinking events in single stream

---

## Phase 9a: Additional Providers

**Why**: Multi-provider coverage validates the abstraction.

### Gemini Provider (`core/_providers/_gemini.py`)
- URL: `generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`
- `_content_to_gemini_parts()`: handles all ContentPart types
- `_items_to_contents()`: message dicts → Gemini contents/parts format
- `_build_payload()`: generationConfig for params, thinkingConfig, responseSchema
- Tool result batching as functionResponse parts
- `stream()` and `stream_events()` implementations
- `_http.py` gains `async_stream_ndjson()` if needed (Gemini SSE via `?alt=sse` may suffice)

### OpenAI-Compat Provider (`core/_providers/_openai_compat.py`)
- Registry: `{"openai": ..., "xai": ..., "mistral": ..., "groq": ...}`
- Inherits/wraps `OpenAIProvider` with configurable base_url
- Provider-specific tweaks (reasoning for xAI, include_reasoning for Groq)

### Provider Factory Update (`core/_providers/__init__.py`)
- New prefixes: `gemini-`, `grok-`, `mistral-`, `groq-` (or explicit `provider=` kwarg)

### Tests: `tests/test_gemini_provider.py`, `tests/test_openai_compat_provider.py`

---

## Phase 9b: Fallback Client

**Why**: Multi-provider resilience for production systems.

### `toolkit/_fallback.py`
- `FallbackLLM(llms: list[LLM], fallback_on=(...))` — tries in order
- Mirrors LLM interface: `complete()`, `stream()`, `stream_events()`
- Stream fallback: only before first emitted chunk
- Context manager support, `close()` closes all underlying LLMs

### Tests: `tests/test_toolkit_fallback.py`
- First LLM raises RateLimitError → second succeeds
- Stream fallback before vs after first chunk
- Context manager lifecycle

---

## Phase 9c: Batch API

**Why**: Unlocks offline/bulk workloads.

### `core/_batch.py`
- `BatchRequest`, `BatchResult`, `BatchJob` — frozen dataclasses
- `BatchClient(provider, model, api_key)`: submit/status/results
- Anthropic: direct payload → poll → NDJSON results
- OpenAI: JSONL upload → batch create → poll → download
- `AsyncBatchClient` — async variant

### Tests: `tests/test_batch.py`
- Mock HTTP for both provider flows
- NDJSON result parsing
- Error handling in batch results

---

## Phase Dependency Graph

```
Phase 4 (tool call streaming)
  └→ Phase 5 (thinking + structured output)
       └→ Phase 6 (multimodal)

Phase 7 (utilities + middleware) — independent, can start after Phase 6
Phase 8 (rich stream events) — independent, can start after Phase 6

Phase 9a (providers)  ─┐
Phase 9b (fallback)   ─┤── all independent, can start after Phase 6
Phase 9c (batch)      ─┘
```

## Verification (each phase)

1. `uv run pytest` — all existing + new tests pass
2. `uv run ruff check src tests` — clean
3. `uv run ruff format src tests` — clean
4. No imports from _legacy/ in core/ or toolkit/
