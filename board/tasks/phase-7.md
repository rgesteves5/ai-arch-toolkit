# Phase 7: Toolkit Utilities + Middleware

**Status**: Queued (after Phase 6, independent of Phase 8)
**Why**: Standalone utilities for production use. Middleware is opinionated pipeline structure — belongs in toolkit/.

## Token Estimation (`toolkit/_tokens.py`)

- Correction factors per model family (Claude 3=1.12, Claude 4=1.15, Gemini=1.05, etc.)
- `raw_tiktoken_count()` → tiktoken with fallback to len//4
- `estimate_text_tokens` → `estimate_content_tokens` → `estimate_message_tokens` → `estimate_conversation_tokens`
- All have `_for_model` variants with correction factor

## Output Parsing (`toolkit/_output_parsing.py`)

- `parse_json()` — 3-stage: direct → code block → JSON snippet scan
- `parse_json_as[T]()` — generic coercion to dataclass/dict/type
- `extract_code_block()`, `extract_list()`

## Prompt Templates (`toolkit/_templates.py`) — optional convenience

- `PromptTemplate(template)` — frozen, `.format(**kwargs) -> str`
- `ChatTemplate(messages)` — frozen, `.format_messages(**kwargs) -> list[dict]`

## Conversation Memory (`toolkit/_memory.py`) — optional, one approach among many

- `ConversationMemory(items)` — mutable, add/extend/history/clear/token_count
- `SlidingWindowMemory(max_tokens)` — trims oldest after each mutation
- Uses `estimate_conversation_tokens` from `_tokens.py`

## Middleware (`toolkit/_middleware.py`)

- `Request` dataclass (mutable, slots=True): operation, provider, model, messages, system, tools, json_schema, kwargs, context
- `Middleware` Protocol: `before`/`after`/`abefore`/`aafter`
- `_run_before(middlewares, request)` / `_run_after(middlewares, request, result)`
- Short-circuit via `context["_short_circuit_result"]`

## MiddlewareLLM (`toolkit/_middleware_llm.py`)

- `MiddlewareLLM(llm: LLM, middleware: list[Middleware])` — wraps LLM with middleware pipeline
- `complete()`: Request → _run_before → short-circuit check → llm.complete → _run_after
- `stream()` / `stream_events()`: before hooks run first; after hooks wrap the response
- Does NOT modify core LLM — composition pattern

## Middleware Implementations

- `toolkit/_cost.py`: CostTracker — accumulates usage/cost via after hook
- `toolkit/_cache.py`: ResponseCache — SHA-256 key, InMemoryCacheBackend with TTL, short-circuits on hit
- `toolkit/_tracing.py`: TracingMiddleware — OpenTelemetry spans (optional dep, no-op if missing)
- `toolkit/_guardrails.py`: GuardrailMiddleware — regex pattern blocking + input/output validators

## Tests

- `tests/test_toolkit_tokens.py`, `tests/test_toolkit_parsing.py`
- `tests/test_toolkit_templates.py`, `tests/test_toolkit_memory.py`
- `tests/test_toolkit_middleware.py`: hook ordering, short-circuit, composition
- `tests/test_toolkit_cost.py`, `tests/test_toolkit_cache.py`, `tests/test_toolkit_guardrails.py`
