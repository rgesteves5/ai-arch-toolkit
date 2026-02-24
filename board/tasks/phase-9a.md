# Phase 9a: Additional Providers

**Status**: Queued (after Phase 6, independent)
**Why**: Multi-provider coverage validates the abstraction.

## Gemini Provider (`core/_providers/_gemini.py`)

- URL: `generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`
- `_content_to_gemini_parts()`: handles all ContentPart types
- `_items_to_contents()`: message dicts → Gemini contents/parts format
- `_build_payload()`: generationConfig for params, thinkingConfig, responseSchema
- Tool result batching as functionResponse parts
- `stream()` and `stream_events()` implementations
- `_http.py` may need `async_stream_ndjson()` (Gemini SSE via `?alt=sse` may suffice)

## OpenAI-Compat Provider (`core/_providers/_openai_compat.py`)

- Registry: `{"openai": ..., "xai": ..., "mistral": ..., "groq": ...}`
- Inherits/wraps `OpenAIProvider` with configurable base_url
- Provider-specific tweaks (reasoning for xAI, include_reasoning for Groq)

## Provider Factory Update (`core/_providers/__init__.py`)

- New prefixes: `gemini-`, `grok-`, `mistral-`, `groq-` (or explicit `provider=` kwarg)

## Tests

- `tests/test_gemini_provider.py`
- `tests/test_openai_compat_provider.py`
