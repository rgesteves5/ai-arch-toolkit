# Phase 1: Foundation

**Status**: Queued
**Goal**: Add new types, update dependencies, update LLM class, delete raw HTTP layer.

## New Types (`core/_response.py`)

```python
@dataclass(frozen=True, slots=True)
class OutputSchema:
    name: str
    schema: dict[str, Any]
    strict: bool = True

@dataclass(frozen=True, slots=True)
class ThinkingBlock:
    text: str

def _resolve_output_schema(schema: OutputSchema | Any) -> OutputSchema:
    """Accept OutputSchema or Pydantic model class."""
    if isinstance(schema, OutputSchema):
        return schema
    try:
        from pydantic import BaseModel
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return OutputSchema(name=schema.__name__, schema=schema.model_json_schema())
    except ImportError:
        pass
    raise TypeError(f"Expected OutputSchema or Pydantic model, got {type(schema)}")
```

Response gains:
- `thinking: tuple[ThinkingBlock, ...] = ()`
- `parsed: Any = None`

## Import Guard (`core/_providers/_imports.py`)

```python
def require_sdk(package: str, extra: str) -> None:
    try:
        __import__(package)
    except ImportError:
        raise ImportError(
            f"Install the {extra} extra: pip install ai-arch-toolkit[{extra}]"
        ) from None
```

## LLM Class Updates (`core/_llm.py`)

Add params to `complete()`, `stream()`, `stream_sync()`, `complete_sync()`:
- `thinking: bool = False`
- `thinking_effort: str | None = None`
- `thinking_budget: int | None = None`
- `output_schema: OutputSchema | Any | None = None`

These flow through to the provider as kwargs. LLM normalizes `output_schema`
via `_resolve_output_schema()` before passing.

Update `_finalize` to include `thinking` from state.

## Dependencies (`pyproject.toml`)

- Remove `requests` and `httpx` from core dependencies
- Add optional dependency groups: anthropic, openai, gemini, mistral, all
- Add SDKs to dev dependencies

## Delete

- `core/_http.py` — raw HTTP helpers (SDKs replace this)
- `tests/test_http_new.py` — tests for deleted module

## Exports (`core/__init__.py`)

Add `OutputSchema`, `ThinkingBlock` to public exports.

## Tests

- `tests/test_response.py`: Add tests for OutputSchema, ThinkingBlock, _resolve_output_schema
- `tests/test_llm.py`: Update for new params (thinking, output_schema forwarding)

## Verification

1. Non-provider tests pass: `uv run pytest --ignore=tests/test_anthropic_provider.py --ignore=tests/test_openai_provider.py --ignore=tests/test_stream_tool_calls.py --ignore=tests/test_logging.py` — 182+ pass
2. Provider tests **fail with ImportError** (expected — they import deleted `_http.py`)
3. 2 roundtrip tests in `test_runner.py` fail (import provider modules that reference `_http.py`)
4. `uv run ruff check src tests && uv run ruff format --check src tests` — clean

**Note**: Provider tests are intentionally broken. Phase 2/3 rewrite them over SDKs.
