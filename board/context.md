# Context — SDK Provider Rewrite

## What We're Building

Rewriting the `core/` provider layer to use official SDKs (anthropic, openai, google-genai,
mistralai) instead of raw HTTP. Providers become thin adapters: our types → SDK call → our types.

## Architecture Rules

- **core/** never imports toolkit/ or _legacy/
- **toolkit/** imports core/ only
- All dataclasses: `frozen=True, slots=True`
- Python 3.13+, `from __future__ import annotations` in every file
- Ruff line length: 99
- SDKs are optional dependencies — `pip install ai-arch-toolkit[anthropic]`

## Package Layout

```
src/ai_arch_toolkit/
  core/
    _providers/    # SDK adapters: Anthropic, OpenAI, Gemini, Mistral (xAI via OpenAI)
    _tools/        # @tool decorator, ToolGroup, execute helpers (unchanged)
    _content.py    # Message builders (unchanged)
    _llm.py        # LLM class — gains thinking + output_schema params
    _response.py   # Response, ToolCall, Usage + OutputSchema, ThinkingBlock
    _pricing.py    # Cost estimation (unchanged)
    _exceptions.py # APIError, RateLimitError (unchanged)
    _sync.py       # Async→sync bridges (unchanged)
  toolkit/
    _runner.py     # run_tools, run_tools_sync (unchanged)
```

## Completed Work (before SDK rewrite)

- LLM class, Response, Usage, ToolCall — core foundation
- Two providers (Anthropic + OpenAI) via raw HTTP — **to be rewritten**
- Tools layer (@tool, ToolGroup, execute) — stays
- Package reorganization (core/ + toolkit/) — stays
- Tool call streaming via manual SSE parsing — **replaced by SDK streaming**
- 699 tests passing

## Key Design Decisions

1. **Thinking**: Three explicit params — `thinking: bool`, `thinking_effort: str | None`, `thinking_budget: int | None`
2. **Structured output**: `output_schema` param. `OutputSchema(name, schema, strict)` dataclass.
3. **Pydantic**: Optional dependency via try/except import. Support both OutputSchema and Pydantic models.
4. **Anthropic structured output**: Tool trick (synthetic tool + forced tool_choice). Error if tools + output_schema combined.
5. **OpenAI structured output**: Native response_format. Coexists with tools.
6. **BaseProvider contract**: thinking/output_schema flow through **kwargs, not explicit on ABC.
7. **xAI**: OpenAI SDK with `base_url="https://api.x.ai/v1"`
8. **Error wrapping**: SDK exceptions → our APIError/RateLimitError

## Notes

- `_StreamState` is mutable (per-call accumulator). Uses `__slots__`. Gains `thinking` field.
- `_http.py` deleted — SDKs handle transport, retries, auth, SSE parsing.
- `requests` and `httpx` removed from core dependencies.
- Provider tests mock SDK clients instead of HTTP helpers.
