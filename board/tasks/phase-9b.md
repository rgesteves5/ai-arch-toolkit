# Phase 9b: Fallback Client

**Status**: Queued (after Phase 6, independent)
**Why**: Multi-provider resilience for production systems.

## `toolkit/_fallback.py`

- `FallbackLLM(llms: list[LLM], fallback_on=(...))` — tries LLMs in order
- Mirrors LLM interface: `complete()`, `stream()`, `stream_events()`
- Stream fallback: only before first emitted chunk
- Context manager support, `close()` closes all underlying LLMs

## Tests: `tests/test_toolkit_fallback.py`

- First LLM raises RateLimitError → second succeeds
- Stream fallback before vs after first chunk
- Context manager lifecycle
