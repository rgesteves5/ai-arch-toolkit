# Core/ Audit — Production Readiness Review

**Date:** 2026-02-25
**Scope:** All 19 Python files + 1 TOML in `src/ai_arch_toolkit/core/`
**Reference:** `research/python_best_practices.md`, `research/modern_python_2015_16.md`

---

## Table of Contents

1. [Bugs](#1-bugs)
2. [Silent Failures & Missing Warnings](#2-silent-failures--missing-warnings)
3. [SDK Features Not Used](#3-sdk-features-not-used)
4. [Provider Inconsistencies](#4-provider-inconsistencies)
5. [Python Best Practices Violations](#5-python-best-practices-violations)
6. [Type Safety](#6-type-safety)
7. [Design & Architecture](#7-design--architecture)
8. [Hardcoded Values](#8-hardcoded-values)
9. [Missing Validation](#9-missing-validation)
10. [Thread Safety & Resource Management](#10-thread-safety--resource-management)
11. [Missing Features for Production](#11-missing-features-for-production)
12. [Summary & Priority Matrix](#12-summary--priority-matrix)

---

## 1. Bugs

### BUG-1: Gemini ignores `max_tokens` — no output limit applied

**File:** `_providers/_gemini.py` `_SDK_PARAMS`
**Severity:** CRITICAL

The LLM class defaults `max_tokens=4096`. Gemini's SDK expects `max_output_tokens`,
not `max_tokens`. Since `max_tokens` is not in Gemini's `_SDK_PARAMS`, it triggers
the "unknown parameter" warning and is silently dropped. Every Gemini call runs
with the provider's default max output — not the user's requested limit.

**Fix:** Translate `max_tokens` → `max_output_tokens` in `_build_config()`. This
is the only provider that uses a different parameter name for the same concept.

---

### BUG-2: `retry-after` parsing can raise ValueError, masking RateLimitError

**Severity:** HIGH

| Provider | Location | Has Issue |
|----------|----------|-----------|
| Anthropic | `_anthropic.py:285` (complete), `:386` (stream), `:494` (stream_events) | Yes — `float(retry_after)` |
| OpenAI | `_openai.py:267` (complete), `:366` (stream) | Yes — `float(retry_after)` |
| Gemini | `_gemini.py:314` (complete), `:389` (stream) | Yes — `float(retry_after)` |
| xAI | N/A (gRPC, no retry-after header) | Not applicable |

```python
retry_after=float(retry_after) if retry_after else None,
```

If the server returns `retry-after: 1.5s` or `retry-after: Thu, 01 Dec 2025 16:00:00 GMT`
(both valid HTTP formats), `float()` raises `ValueError`. This exception propagates
*instead* of the `RateLimitError`, losing the original error context entirely.

Anthropic has this in **3 separate places** (complete, stream, stream_events), OpenAI
in **2** (complete, stream), Gemini in **2** (complete, stream). All 7 occurrences need fixing.

**Fix:** Extract a shared helper in `_base.py`:
```python
def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
```

---

### BUG-3: Stream `Response.raw` is always `None`

**File:** `_llm.py:171-184` (`_finalize` lambda in `stream()`)
**Severity:** MEDIUM

The `_finalize` function that builds the `Response` after stream consumption never
sets `raw=state.raw`. All providers populate `state.raw` with the final SDK response,
but it's discarded at the LLM layer.

```python
def _finalize(text: str) -> Response:
    # ... builds Response but never includes raw=state.raw
```

**Fix:** Add `raw=state.raw` to the Response construction in both `stream()` and
`stream_sync()` finalize functions.

---

### BUG-4: Anthropic `_messages_to_sdk` raises KeyError on missing `role` / `content`

**File:** `_anthropic.py:85`
**Severity:** MEDIUM

```python
wire.append({"role": msg["role"], "content": msg["content"]})
```

| Provider | Role access | Content access |
|----------|------------|---------------|
| Anthropic | `msg["role"]` — **KeyError** | `msg["content"]` — **KeyError** |
| OpenAI | `msg.get("role", "user")` — safe | `msg.get("content", "")` — safe |
| Gemini | `msg.get("role", "user")` — safe | `msg.get("content", "")` — safe |
| xAI | `msg.get("role", "user")` — safe | `msg.get("content", "")` — safe |

Anthropic is the **only** provider that uses direct dict access for both `role` AND
`content`. All others use `.get()` with defaults.

**Fix:** Use `msg.get("role", "user")` and `msg.get("content", "")` consistently
across all providers, or validate messages at the LLM layer before they reach providers.

---

### BUG-5: Inconsistent cache token extraction across providers

**Severity:** MEDIUM

| Provider | cache_write_tokens | cache_read_tokens | Missing |
|----------|-------------------|------------------|---------|
| Anthropic | `cache_creation_input_tokens` | `cache_read_input_tokens` | None — **full support** |
| OpenAI | Not extracted | Not extracted | **Both** — `prompt_tokens_details.cached_tokens` available |
| Gemini | Not extracted | `cached_content_token_count` | `cache_write` not available in SDK |
| xAI | Not extracted | `cached_prompt_text_tokens` | `cache_write` not available in SDK |

OpenAI is the worst offender — it has both cache read data available but extracts
neither. Gemini and xAI at least extract cache_read. Only Anthropic extracts both.

**Fix:** OpenAI: extract `prompt_tokens_details.cached_tokens` into `cache_read_tokens`:
```python
def _extract_usage(sdk_usage: Any) -> Usage:
    cache_read = 0
    details = getattr(sdk_usage, "prompt_tokens_details", None)
    if details:
        cache_read = getattr(details, "cached_tokens", 0) or 0
    return Usage(
        input_tokens=getattr(sdk_usage, "prompt_tokens", 0),
        output_tokens=getattr(sdk_usage, "completion_tokens", 0),
        cache_read_tokens=cache_read,
    )
```

---

### BUG-6: `_pricing.py` truthiness check skips valid zero-token cache entries

**File:** `_pricing.py:115-118`
**Severity:** LOW

```python
if cache_write_tokens and p.cache_write is not None:
```

This uses truthiness of `cache_write_tokens`. The value `0` is falsy, so zero tokens
are correctly skipped. But the semantic intent ("if pricing exists AND there are
tokens") is expressed backwards. If someone changes this to a different check later,
the behavior breaks.

**Fix:** `if p.cache_write is not None and cache_write_tokens > 0:` — explicit intent.

---

### BUG-7: `_tool_to_sdk` uses `tool["name"]` — KeyError risk in all providers

**Severity:** MEDIUM

| Provider | Access pattern |
|----------|--------------|
| Anthropic | `tool["name"]` — **KeyError** |
| OpenAI | `tool["name"]` — **KeyError** |
| Gemini | `tool["name"]` — **KeyError** |
| xAI | `tool["name"]` — **KeyError** |

All 4 providers use direct dict access for `tool["name"]` in their `_tool_to_sdk`
functions. A tool dict missing the `name` key will raise an unhelpful `KeyError`.
Other fields (`description`, `input_schema`, `parameters`) use `.get()` safely.

**Fix:** Either validate tool dicts in `prepare_tools()` (see VAL-1) before they
reach providers, or use `tool.get("name", "")` with a warning.

---

### BUG-8: `output_schema` double extraction in OpenAI and Gemini `complete()`

**Files:** `_openai.py:256-258`, `_gemini.py:295-298`
**Severity:** LOW

```python
# In complete():
output_schema: OutputSchema | None = kwargs.get("output_schema")  # (1) reads it
sdk_kwargs = self._build_sdk_kwargs(wire, tools=tools, **kwargs)   # (2) kwargs still has it
# Inside _build_sdk_kwargs: kwargs.pop("output_schema", None)      # (3) pops it
```

`output_schema` is read via `.get()` in `complete()`, then the same kwargs dict is
passed to `_build_sdk_kwargs()` which `.pop()`s it again. This works but is fragile:
the `.get()` in step (1) leaves it in kwargs, so it flows correctly. But it means
`output_schema` appears in both the local variable AND the kwargs.

Anthropic has the same pattern (`_anthropic.py:274-276`) but the same fragility.
xAI also (`_xai.py:287-292`). **All 4 providers have this pattern.**

**Fix:** Extract once, pass explicitly. Or remove the local extraction since
`_parse_sdk_response` receives it as a parameter anyway.

---

### BUG-9: Gemini `_messages_to_sdk` tool result loses `tool_use_id` (name mapping)

**File:** `_gemini.py:77`
**Severity:** MEDIUM

```python
pending_fn_responses.append(
    types.Part(function_response=types.FunctionResponse(
        name=msg.get("name", ""),  # uses "name" key, not the tool function name
        ...
    ))
)
```

The generic message format uses `tool_use_id` to identify tool results, but Gemini's
`FunctionResponse` requires `name` (the function name). The code reads `msg.get("name", "")`
which may be empty or missing — the standard `tool_result()` helper from `_content.py`
sets `name` only if explicitly passed. If the user builds tool results without `name`,
Gemini's API will receive an empty function name.

**Fix:** Either require `name` in tool results at the LLM layer, or look up the
function name from the tool call ID in the conversation history.

---

### BUG-10: OpenAI stream never sets `state.raw`

**File:** `_openai.py:295-374`
**Severity:** MEDIUM

| Provider | Sets `state.raw` | When |
|----------|-----------------|------|
| Anthropic | Yes | `state.raw = final` after `get_final_message()` |
| OpenAI | **No** | `state.raw` stays `None` |
| Gemini | Yes | `state.raw = chunk` on every chunk (last chunk wins) |
| xAI | Yes | `state.raw = final_response` after stream ends |

OpenAI's `stream()` generator never sets `state.raw`. Combined with BUG-3 (LLM
finalizer doesn't pass `state.raw`), the raw response is completely lost for OpenAI
streams even if BUG-3 is fixed.

**Fix:** After the stream loop, set `state.raw` to the last chunk or accumulate
the full response.

---

### BUG-11: xAI `_extract_usage` reads WRONG field names — all usage silently zero

**File:** `_xai.py:117-123`
**Severity:** CRITICAL (confirmed via SDK docs research)

```python
def _extract_usage(sdk_usage: Any) -> Usage:
    return Usage(
        input_tokens=getattr(sdk_usage, "prompt_tokens", 0),      # WRONG
        output_tokens=getattr(sdk_usage, "completion_tokens", 0),  # WRONG
        cache_read_tokens=getattr(sdk_usage, "cached_prompt_text_tokens", 0),  # UNDOCUMENTED
    )
```

The xai-sdk README documents the usage fields as `input_tokens` and `output_tokens`,
NOT `prompt_tokens` and `completion_tokens` (which are OpenAI naming). Because `getattr`
falls back to 0, all xAI token usage silently returns zero. Cost estimation is broken.

Additionally, `cached_prompt_text_tokens` is not in any official documentation — it may
exist on the proto object but is unverified.

**Fix:**
```python
Usage(
    input_tokens=getattr(sdk_usage, "input_tokens", 0),
    output_tokens=getattr(sdk_usage, "output_tokens", 0),
)
```

---

## 2. Silent Failures & Missing Warnings

### SILENT-1: xAI `search_parameters` is dead code

**File:** `_xai.py:234`
**Severity:** HIGH

`_build_create_kwargs` extracts `search_parameters` from kwargs and passes it to the
SDK, but the LLM class never exposes this parameter. It only works if users
bypass the LLM class and call the provider directly. No warning is issued.

**Recommendation:** Either expose `search_parameters` at the LLM level as a
provider-specific kwarg that flows through `**kwargs`, or remove the code and document
it as not supported. Dead code in production is a liability.

---

### SILENT-2: `base_url` silently ignored for Gemini and xAI

**File:** `_providers/__init__.py:71-85`
**Severity:** HIGH

```python
if name == "xai":
    return XAIProvider(model, _resolve_key("XAI_API_KEY", api_key))
    # base_url parameter from create_provider() is silently dropped
```

`create_provider()` accepts `base_url` but only passes it to Anthropic and OpenAI.
If a user sets `base_url` for a Gemini or xAI model (e.g., for a proxy), it's
silently ignored with no error or warning.

**Fix:** Warn if `base_url` is provided but the provider doesn't support it.

---

### SILENT-3: Unknown provider kwargs warned but not actionable

**Files:** All providers' `_build_sdk_kwargs` / `_build_config` / `_build_create_kwargs`
**Severity:** MEDIUM

```python
warnings.warn(f"Unknown parameter(s) ignored for Anthropic: {sorted(unknown)}", stacklevel=4)
```

| Provider | stacklevel | Warning category |
|----------|-----------|-----------------|
| Anthropic | 4 | default (UserWarning) |
| OpenAI | 4 | default (UserWarning) |
| Gemini | 4 | default (UserWarning) |
| xAI | 4 | default (UserWarning) |

All 4 use `stacklevel=4` but the call chain depth varies (LLM → complete →
provider.complete → _build_sdk_kwargs = 4 levels). For `stream()` the call chain
is different (LLM → stream → provider.stream → _build → 4 levels, but the warning
fires during generator creation, not execution). The stacklevel may be wrong for
stream paths.

**Fix:** Include the valid parameter set in the warning message. Verify `stacklevel`
is correct for all call paths (complete vs stream vs stream_events).

---

### SILENT-4: `prepare_tools` returns `None` on empty list

**File:** `_tools/__init__.py:75`
**Severity:** MEDIUM

```python
return result or None
```

If a user passes `tools=[]`, the function returns `None` (empty list is falsy).
The user's intent was "I have tools but the list is empty" vs `None` meaning
"I don't want tools". These are semantically different.

**Fix:** Return `[]` for empty list input, `None` only for `None` input.

---

### SILENT-5: `ToolGroup.add()` silently overwrites duplicate names

**File:** `_tools/_group.py:40-47`
**Severity:** MEDIUM

```python
def add(self, fn: Callable[..., Any]) -> None:
    name = tool_def["name"]
    self._fns[name] = fn        # Silent overwrite
    self._definitions[name] = tool_def
```

If two functions have the same tool name, the second silently replaces the first.
No warning, no error. The user loses a tool without knowing.

**Fix:** Raise `ValueError` on duplicate, or warn and document the overwrite behavior.

---

### SILENT-6: `_schema.py` multi-type unions silently collapse to string

**File:** `_tools/_schema.py:35-36`
**Severity:** MEDIUM

```python
# Multi-type union (not just Optional) — fall back to string
return {"type": "string"}, True
```

`str | int | float` becomes `{"type": "string"}` with no warning. The LLM receives
an incorrect schema — it will only generate strings, never integers or floats.

**Fix:** Generate a JSON Schema `oneOf` or at minimum warn that the type was
simplified.

---

### SILENT-7: xAI `thinking_budget` silently popped with no warning

**File:** `_xai.py:232`
**Severity:** MEDIUM

```python
kwargs.pop("thinking_budget", None)  # not supported, ignored
```

The only indication this is unsupported is a code comment. No warning, no log.
Compare with OpenAI which at least logs at debug level:
```python
if thinking_budget:
    logger.debug("thinking_budget not supported by OpenAI, ignored")
```

If a user explicitly sets `thinking_budget=5000` for xAI, they have no indication
it's being ignored.

**Fix:** Emit `warnings.warn()` when `thinking_budget` is set, matching INCONSISTENT-3.

---

### SILENT-8: Anthropic `_build_output_schema_tool` error vs other providers' silent combo

**Severity:** MEDIUM

| Provider | tools + output_schema | Behavior |
|----------|----------------------|----------|
| Anthropic | Raises ValueError | Explicit error |
| OpenAI | Both applied | **Works** (native support) |
| Gemini | Both applied | **Works** (native support) |
| xAI | Both applied | **Works** (native support) |

Only Anthropic errors when both tools and output_schema are used. The error message
says "Anthropic does not support both tools and output_schema simultaneously."
This is a limitation of the tool-trick approach, not an SDK limitation.

**Recommendation:** Document this clearly. Consider using Anthropic's native JSON
mode (SDK ≥0.40) to lift this restriction (see SDK-3).

---

### SILENT-9: Empty enum produces invalid JSON Schema

**File:** `_tools/_schema.py:47-50`
**Severity:** LOW

```python
values = [m.value for m in hint]
return {"type": "string", "enum": values}
```

An empty `Enum` class produces `{"type": "string", "enum": []}` — an empty enum
array is invalid per JSON Schema spec.

**Fix:** Check `if not values: return {"type": "string"}, False` without the enum key.

---

## 3. SDK Features Not Used

### SDK-1: No multimodal content support

**Providers:** All
**Severity:** HIGH

`user()` now accepts `str | list[Any]`, but NO provider converts list content parts
to SDK format. All `_messages_to_sdk` functions assume `content` is a string:

- Anthropic: `msg["content"]` used as-is (string)
- OpenAI: `msg.get("content", "")` (string)
- Gemini: `types.Part(text=msg.get("content", ""))` (string)
- xAI: `xai_chat.user(msg.get("content", ""))` (string)

Images, audio, PDFs, documents — none are converted. Passing multimodal content
produces garbage or errors at the SDK level.

**Recommendation:** Implement content part conversion per provider, or restrict
`user()` back to `str` and add a separate `user_multimodal()` when ready. Having
the type signature accept lists that nothing handles is a trap.

---

### SDK-2: No prompt caching control (Anthropic, OpenAI, Gemini)

**Severity:** MEDIUM

All three SDKs support explicit cache control markers. Our `Usage` dataclass tracks
cache tokens (read/write), but there is no way to *request* caching. Users cannot
mark system prompts or tool definitions as cacheable.

**Recommendation:** Add optional `cache_control` parameter or document how users
can achieve caching through raw kwargs.

---

### SDK-3: Anthropic structured output uses tool trick, not native

**File:** `_anthropic.py:106-125`
**Severity:** LOW (works correctly, but SDK now has native support)

Anthropic SDK ≥0.40 has native JSON output mode. The current tool trick works
but prevents combining tools with structured output (the error on line 114-117).
Native support would lift this restriction.

**Recommendation:** Check SDK version and use native JSON mode when available,
falling back to tool trick for older SDK versions.

---

### SDK-4: OpenAI missing `stream_events()` override

**File:** `_openai.py` (no override)
**Severity:** MEDIUM

OpenAI delivers tool call deltas incrementally during streaming (index-based
accumulation). This is enough to implement real-time `stream_events()` with
tool_call events as they arrive, but only the default post-hoc wrapper is used.

**Recommendation:** Implement `stream_events()` override for OpenAI.

---

### SDK-5: Gemini server-side tools not exposed

**Severity:** LOW

`google_search`, `code_execution`, `safety_settings`, `function_calling_config` —
none are exposed. These are Gemini-specific features users may need.

**Recommendation:** Allow these through `**kwargs` passthrough to the SDK config.

---

### SDK-6: OpenAI `parallel_tool_calls` not exposed

**Severity:** LOW

OpenAI supports `parallel_tool_calls=True/False` to control whether the model can
call multiple tools in one response. Not in `_SDK_PARAMS`, not forwarded.

**Recommendation:** Add to `_SDK_PARAMS`.

---

### SDK-7: No `tool_choice` support across providers

**Severity:** MEDIUM

All providers' SDKs support `tool_choice` to force specific tool usage or control
tool behavior (`auto`, `any`, `none`, or a specific tool name). This is only used
internally for Anthropic's structured output trick. Users cannot control tool_choice.

| Provider | SDK parameter | Status |
|----------|--------------|--------|
| Anthropic | `tool_choice={"type": "tool", "name": "..."}` | Internal only |
| OpenAI | `tool_choice="auto"/"none"/"required"/{"type":"function",...}` | Not exposed |
| Gemini | `tool_config=types.ToolConfig(...)` | Not exposed |
| xAI | Not documented | N/A |

**Recommendation:** Add `tool_choice` parameter to LLM.complete() and normalize
per provider.

---

### SDK-8: Anthropic thinking blocks not streamed in `stream()`, only in `stream_events()`

**Severity:** LOW

Anthropic's `stream()` extracts thinking blocks post-hoc from `get_final_message()`,
so they're available in `state.thinking` after stream exhaustion. But the text deltas
for thinking are NOT yielded during streaming (only text content is).

In contrast, `stream_events()` provides real-time thinking events. This is correct
design — `stream()` yields only user-visible text. But it means users who want
both streaming text AND thinking must use `stream_events()`.

**Status:** By design, but should be documented.

---

## 4. Provider Inconsistencies

### INCONSISTENT-1: `max_tokens` parameter naming

| Provider | SDK Parameter | In `_SDK_PARAMS` |
|----------|--------------|-----------------|
| Anthropic | `max_tokens` | Yes |
| OpenAI | `max_tokens` + `max_completion_tokens` | Both |
| Gemini | `max_output_tokens` | `max_output_tokens` only |
| xAI | `max_tokens` | Yes |

LLM class sends `max_tokens=4096` by default. Gemini drops it. Three providers
receive it, one doesn't. This is BUG-1.

**Fix:** Normalize at the provider level. Each provider should accept `max_tokens`
and translate to its SDK's naming internally.

---

### INCONSISTENT-2: `stream_events()` real-time behavior

| Provider | stream_events() | Real-time text | Real-time thinking | Real-time tool_call |
|----------|----------------|---------------|-------------------|-------------------|
| Anthropic | Custom override | Yes | Yes (buffered per block) | Yes |
| OpenAI | Default wrapper | Yes | N/A | No (post-hoc) |
| Gemini | Default wrapper | Yes | No (post-hoc) | No (post-hoc) |
| xAI | Default wrapper | Yes | No (post-hoc) | No (post-hoc) |

Only Anthropic has a proper `stream_events()` implementation. Users get inconsistent
behavior across providers.

**Fix:** Implement `stream_events()` for at least OpenAI and xAI where the stream
data already contains the necessary deltas.

---

### INCONSISTENT-3: `thinking_budget` handling

| Provider | thinking=True, no effort/budget | thinking_budget explicitly set |
|----------|-------------------------------|------------------------------|
| Anthropic | budget_tokens=10000 | Used directly |
| OpenAI | reasoning_effort="high" | Silent debug log, ignored |
| Gemini | thinking_budget=10000 | Used directly |
| xAI | reasoning_effort="high" | `kwargs.pop("thinking_budget", None)` — silently dropped |

OpenAI and xAI silently ignore `thinking_budget`. If a user explicitly sets it,
they expect it to work. Silent dropping is wrong.

**Fix:** Warn when `thinking_budget` is set on providers that don't support it.

---

### INCONSISTENT-4: Error handling patterns vary significantly

| Provider | Error classes caught | Rate limit body | API error body | Response guard |
|----------|---------------------|----------------|---------------|---------------|
| Anthropic | `anthropic.RateLimitError`, `anthropic.APIStatusError` | `str(exc.body)` | `str(exc.body)` | None needed |
| OpenAI | `openai.RateLimitError`, `openai.APIStatusError` | `str(exc.body)` | `str(exc.body)` | `if exc.response else 429/500` |
| Gemini | `genai_errors.ClientError`, `genai_errors.ServerError` | `str(exc)` (full) | `str(exc)` (full) | `if exc.response and hasattr(...)` |
| xAI | `grpc.aio.AioRpcError` | `exc.details() or ""` | `exc.details() or ""` | Single exception type |

Issues:
1. **Error body format varies wildly** — xAI may return empty string, Gemini includes
   full traceback
2. **OpenAI guards `exc.response`** as it may be None; Anthropic assumes it's always present
3. **xAI `_grpc_code_to_http` is incomplete** — missing CANCELLED, UNKNOWN, ABORTED,
   DATA_LOSS, OUT_OF_RANGE, FAILED_PRECONDITION, ALREADY_EXISTS (all map to 500 via default)
4. **Gemini rate limit detection** uses `if exc.code == 429` on ClientError, which
   is an integer comparison on what may be a string or enum
5. **Error chaining** — all 4 correctly use `from exc` for exception chaining

**Fix:** Standardize error body extraction. Always include the original exception
type/message. Never return empty body. Add missing gRPC status codes to xAI mapping.

---

### INCONSISTENT-5: `close()` implementation

| Provider | Has close() | What it does |
|----------|------------|-------------|
| Anthropic | Yes | `await self._client.close()` |
| OpenAI | Yes | `await self._client.close()` |
| Gemini | No | Inherits no-op from BaseProvider |
| xAI | Yes | `await self._client.close()` |

Gemini's `genai.Client` may hold HTTP connections. Even if it doesn't need cleanup
today, a future SDK update might add state. Missing `close()` is a resource
management gap.

**Fix:** Add explicit `close()` to GeminiProvider, even if currently a no-op
with a comment explaining why.

---

### INCONSISTENT-6: Message role defaulting

| Provider | Missing role behavior |
|----------|---------------------|
| Anthropic | `msg["role"]` — KeyError |
| OpenAI | `msg.get("role", "user")` — defaults to user |
| Gemini | `msg.get("role", "user")` — defaults to user |
| xAI | `msg.get("role", "user")` — defaults to user |

**Fix:** Standardize. Either all providers use `.get()` with default, or validate
at the LLM layer.

---

### INCONSISTENT-7: Model name from response

| Provider | Captures model from response |
|----------|---------------------------|
| Anthropic | Yes — `message.model or model` |
| OpenAI | Yes — `completion.model or model` |
| Gemini | No — always uses `self._model` |
| xAI | No — always uses `self._model` |

**Fix:** Gemini and xAI should check if the response includes a model identifier.

---

### INCONSISTENT-8: Structured output parsing approach

| Provider | Structured output mechanism | `parsed` extraction |
|----------|---------------------------|-------------------|
| Anthropic | Tool trick (synthetic tool) | From `tool_use` block input |
| OpenAI | Native `response_format` | `json.loads(text)` |
| Gemini | Native `response_json_schema` | `json.loads(text)` |
| xAI | Native `response_format` (proto) | `json.loads(text)` |

OpenAI, Gemini, and xAI all suppress `json.JSONDecodeError` via `contextlib.suppress`.
This means if the model returns malformed JSON, `parsed` is silently `None` with no
warning. The user has no way to know the parse failed vs no structured output.

**Fix:** Log a warning when JSON parsing fails for structured output responses.

---

### INCONSISTENT-9: System message handling

| Provider | System as top-level param | System from messages | Both |
|----------|--------------------------|---------------------|------|
| Anthropic | `system=` kwarg to SDK | Extracted by `_messages_to_sdk` | `system=` wins if both |
| OpenAI | System is a message role | System in message list | `system=` prepended, inline system kept (if `system is None`) |
| Gemini | `system_instruction=` in config | Extracted by `_messages_to_sdk` | `system=` wins if both |
| xAI | System message prepended | Extracted by `_messages_to_sdk` | `system=` wins if both |

**OpenAI is the only provider that may include DUPLICATE system messages** — if
`system=` is passed AND messages contain system-role messages, OpenAI discards
inline system messages (correct). But if `system is None`, inline system messages
are kept as regular messages. This is correct but different from Anthropic/Gemini
which extract and concatenate all system messages.

**Status:** Acceptable — each provider handles its SDK's system expectations correctly.
But document the behavior difference for users who switch providers.

---

### INCONSISTENT-10: `_SDK_PARAMS` sets are not standardized

| Parameter | Anthropic | OpenAI | Gemini | xAI |
|-----------|-----------|--------|--------|-----|
| `temperature` | Yes | Yes | Yes | Yes |
| `top_p` | Yes | Yes | Yes | Yes |
| `top_k` | Yes | No | Yes | No |
| `max_tokens` | Yes | Yes | No (**BUG-1**) | Yes |
| `max_completion_tokens` | No | Yes | No | No |
| `max_output_tokens` | No | No | Yes | No |
| `stop` / `stop_sequences` | `stop_sequences` | `stop` | `stop_sequences` | `stop` |
| `frequency_penalty` | No | Yes | Yes | Yes |
| `presence_penalty` | No | Yes | Yes | Yes |
| `seed` | No | Yes | Yes | Yes |
| `response_format` | No | Yes | No | No |

Issues:
1. **`stop` vs `stop_sequences`** — Anthropic/Gemini use `stop_sequences`, OpenAI/xAI
   use `stop`. If a user passes `stop=["END"]`, it only works for OpenAI/xAI.
2. **`seed`** — Not in Anthropic's `_SDK_PARAMS`. Anthropic does not support seed.
3. **`response_format`** — In OpenAI's set but this conflicts with structured output
   when `output_schema` is also set (both try to set `response_format`).
4. **`frequency_penalty` / `presence_penalty`** — Not in Anthropic's set (not supported).

**Recommendation:** Normalize at the LLM layer:
- Map `stop` ↔ `stop_sequences` based on provider
- Warn when a parameter is not supported by the target provider (not just "unknown")

---

## 5. Python Best Practices Violations

### BP-1: Bare `except Exception` in schema inference (NEVER do this)

**File:** `_tools/_schema.py:110, 127, 257, 273`
**Rule:** "Catch specific exceptions, let unknown bubble up" / "NEVER bare except"

```python
try:
    hints = get_type_hints(fn)
except Exception:
    hints = {}
```

Four instances of broad exception catching that hide real bugs (NameError from
bad annotations, TypeError from invalid hints, etc.).

**Fix:** Catch `(NameError, AttributeError, TypeError)` specifically. Log a warning
when falling back.

---

### BP-2: `APIError` docstring says "HTTP error" — misleading for gRPC

**File:** `_exceptions.py:9`
**Rule:** Docstrings describe what and why accurately

```python
class APIError(Exception):
    """Raised when an LLM provider returns an HTTP error."""
```

xAI uses gRPC, not HTTP. The docstring is factually wrong for that provider.

**Fix:** `"""Raised when an LLM provider returns an API error."""`

---

### BP-3: Missing `__all__` in `_providers/__init__.py`

**File:** `_providers/__init__.py`
**Rule:** "Include `__all__` in library `__init__.py`"

No `__all__` defined. `create_provider` and `_detect_provider` are both importable.
Only `create_provider` should be public.

**Fix:** Add `__all__ = ["create_provider"]`.

---

### BP-4: `from __future__ import annotations` everywhere (acceptable today, removable in 3.14)

**Rule:** "PEP 649/749 eliminates the need in Python 3.14"

Currently `requires-python = ">=3.13"`. The import is harmless now but will become
unnecessary when 3.14 is the minimum. Not a violation — just noting for future cleanup.

---

### BP-5: `_sync.py` daemon thread inconsistency

**File:** `_sync.py:37, 68`
**Rule:** Code should be consistent

```python
# _run_sync — NOT daemon
thread = threading.Thread(target=_target)

# _stream_sync — IS daemon
thread = threading.Thread(target=_target, daemon=True)
```

Inconsistent. Daemon threads are killed on process exit without cleanup.
Non-daemon threads keep the process alive. Both should follow the same policy.

**Fix:** Both should be non-daemon (safer for cleanup), or both daemon (if we
accept abrupt termination). Document the choice.

---

### BP-6: No `kw_only=True` on dataclasses with 3+ fields

**Rule:** "3+ fields: add `kw_only=True`" (best practices)

`Response` has 10 fields, `Usage` has 4, `OutputSchema` has 3. None use `kw_only=True`.

```python
Response(text, tool_calls, thinking, parsed, usage, cost, ...)  # Positional = fragile
Response(text="hi", usage=Usage(), cost=0.1, ...)               # Keyword = safe
```

**Impact:** Internal code can construct `Response` positionally, which breaks if
field order changes.

**Fix:** Add `kw_only=True` to `Response`, `Usage`, `OutputSchema`. This is a
breaking change for any code constructing these positionally (check tests first).

---

### BP-7: No lazy log formatting

**File:** `_providers/_openai.py:235`
**Rule:** "NEVER eager string formatting in logs"

```python
logger.debug("thinking_budget not supported by OpenAI, ignored")
```

This specific instance is a literal string (no formatting), so it's fine. But
a grep across the codebase should verify no `f"..."` or `.format()` in log calls.

**Status:** OK — verified no eager formatting exists. Noting for awareness.

---

### BP-8: Response string-like behavior is partial and confusing

**File:** `_response.py:110-130`
**Rule:** "Duck typing — check capabilities, not types"

Response implements `__str__`, `__contains__`, `__add__`, `__radd__`, `__bool__`
but NOT `__len__`, `__getitem__`, `__iter__`, etc. It quacks like a string in some
contexts but not others. This violates the principle of least surprise.

**Recommendation:** Either commit fully to string protocol (implement all methods)
or remove `__add__`/`__radd__`/`__contains__` and just keep `__str__` and `__bool__`.
The partial implementation will confuse users.

---

## 6. Type Safety

### TYPE-1: `tools: Any | None` in LLM public API

**File:** `_llm.py:121, 144, 205, 231`

```python
async def complete(self, messages, *, tools: Any | None = None, ...):
```

`Any` means no IDE help, no type checking, no autocomplete. Users must read docs
to know what to pass.

**Fix:**
```python
tools: list[dict[str, Any]] | ToolGroup | Callable[..., Any] | None = None,
```

---

### TYPE-2: `output_schema: Any | None` in LLM public API

**File:** `_llm.py:125, 151, 209, 234`

Same issue. Should be:
```python
output_schema: OutputSchema | type | None = None,
```

---

### TYPE-3: `StreamEvent.kind` uses `Literal` but no runtime validation

**File:** `_base.py:95`

```python
kind: Literal["text", "thinking", "tool_call"]
```

`Literal` is a type-checking construct only. At runtime, nothing prevents
`StreamEvent(kind="invalid")`. Frozen dataclass doesn't validate field values.

**Recommendation:** Add `__post_init__` validation or accept this as a typing-only
constraint (document the decision).

---

### TYPE-4: `_resolve_output_schema` accepts `Any` — no type narrowing

**File:** `_response.py:48`

```python
def _resolve_output_schema(schema: OutputSchema | Any) -> OutputSchema:
```

`OutputSchema | Any` is equivalent to `Any`. The type signature communicates nothing.

**Fix:** `schema: OutputSchema | type` — then the Pydantic branch handles `type`.

---

## 7. Design & Architecture

### DESIGN-1: `_INIT_DEFAULTS` ClassVar duplicates constructor signature

**File:** `_llm.py:46`

```python
_INIT_DEFAULTS: ClassVar[dict[str, Any]] = {"temperature": 0.0, "max_tokens": 4096}
```

This duplicates the default values from `__init__`. If someone changes the
constructor defaults without updating `_INIT_DEFAULTS`, `__repr__` breaks silently.

**Fix:** Remove `_INIT_DEFAULTS`. Derive repr from the constructor signature itself,
or accept showing all non-None values.

---

### DESIGN-2: `state_holder` list hack in `stream_sync`

**File:** `_llm.py:250-258`

```python
state_holder: list[Any] = []

def _async_factory():
    aiter, state = self._provider.stream(...)
    state_holder.append(state)
    return aiter
```

Uses list mutation as a side channel to pass state between closures. If
`_async_factory` is never called, `state_holder[0]` would raise IndexError.
The `if state_holder else None` guard masks the error.

**Fix:** Use a proper mutable container (e.g., a single-element dataclass) or
restructure to avoid the side channel.

---

### DESIGN-3: `SyncStreamResponse` lacks context manager support

**File:** `_response.py:187-219`

`StreamResponse` (async) has `__aenter__`/`__aexit__`. `SyncStreamResponse` has
neither `__enter__` nor `__exit__`. Users cannot use `with llm.stream_sync(...)`.

**Fix:** Add `__enter__`/`__exit__` to `SyncStreamResponse`.

---

### DESIGN-4: `Response.to_message()` loses `parsed` and `thinking`

**File:** `_response.py:96-107`

```python
def to_message(self) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": self.text}
    if self.tool_calls:
        msg["tool_calls"] = [...]
    return msg
```

`parsed` (structured output) and `thinking` blocks are discarded. A round-tripped
response loses information.

**Fix:** Include `parsed` in the message when present. Thinking blocks are typically
not sent back to the API (provider-specific), so excluding them is acceptable but
should be documented.

---

### DESIGN-5: Cost calculated in two places for different code paths

**Files:** Provider `_parse_sdk_response` (complete), `_llm.py _finalize` (stream)
**Severity:** LOW

For `complete()`, cost comes from the provider. For `stream()`, cost is recomputed
in the LLM class's finalizer. This works but is inconsistent — the cost estimation
logic exists in two separate locations.

**Recommendation:** Always compute cost in the LLM layer for consistency, or always
in the provider layer. Not both.

---

## 8. Hardcoded Values

### HARD-1: Thinking budget defaults (10000, 2000, 5000)

**Files:** `_anthropic.py:98-102`, `_gemini.py:139-144`

```python
cfg["budget_tokens"] = {"low": 2000, "medium": 5000, "high": 10000}.get(
    thinking_effort, 10000
)
```

These magic numbers are duplicated in Anthropic and Gemini. If a provider changes
their budget ranges, we need to update multiple files.

**Fix:** Extract to a module-level constant dict:
```python
_THINKING_EFFORT_BUDGETS: dict[str, int] = {"low": 2000, "medium": 5000, "high": 10000}
_DEFAULT_THINKING_BUDGET: int = 10000
```

Place in `_base.py` so all providers share the same mapping.

---

### HARD-2: `max_tokens=4096` default in LLM

**File:** `_llm.py:33`

This is reasonable for most models but too high for some (mini models) and too
low for others (100K output models). It's a sensible default, but should be
documented that it's overridable.

**Status:** Acceptable. Just needs documentation.

---

### HARD-3: `_sync.py` timeout of 300 seconds

**File:** `_sync.py:39`

```python
thread.join(timeout=300)
```

Hardcoded 5-minute timeout with no way to configure. Long-running completions
(e.g., with thinking + large output) may exceed this.

**Fix:** Accept timeout as a parameter to `_run_sync`, or make it configurable
via a module-level constant.

---

### HARD-4: `_sync.py` stream thread join timeout of 5 seconds

**File:** `_sync.py:79`

```python
thread.join(timeout=5)
```

After stream exhaustion, only 5 seconds to clean up. No warning if the thread
is still alive.

**Fix:** Check `thread.is_alive()` after join and log a warning.

---

## 9. Missing Validation

### VAL-1: No tool name validation

**Files:** All `_tool_to_sdk` functions
**Severity:** HIGH

```python
"name": tool["name"],  # Can be "", can be missing (KeyError)
```

Empty tool names will pass to the API and fail with a cryptic provider error.
Missing `name` key raises `KeyError` with no context.

**Fix:** Validate in `prepare_tools()` before reaching providers:
```python
if not tool_def.get("name"):
    raise ValueError(f"Tool definition missing 'name': {tool_def}")
```

---

### VAL-2: No `thinking_effort` value validation

**File:** `_llm.py:104`

```python
if thinking_effort is not None:
    kwargs["thinking_effort"] = thinking_effort
```

Accepts any string. `thinking_effort="extreme"` or `thinking_effort=""` pass through
silently and produce undefined behavior at the provider level.

**Fix:** Validate against allowed values `{"low", "medium", "high"}` at the LLM layer.

---

### VAL-3: No `thinking_budget` range validation

**File:** `_llm.py:106`

Accepts any integer, including 0 and negative values. Providers will reject these
with cryptic errors.

**Fix:** `if thinking_budget is not None and thinking_budget <= 0: raise ValueError(...)`.

---

### VAL-4: `OutputSchema.name` and `OutputSchema.schema` not validated

**File:** `_response.py:36-45`

Empty name, empty schema dict — both accepted silently.

**Fix:** Add `__post_init__` validation.

---

### VAL-5: `tool_result` accepts empty `tool_use_id`

**File:** `_content.py:25-37`

```python
def tool_result(content: Any, *, tool_use_id: str, ...):
    msg = {"role": "tool", "content": content, "tool_use_id": tool_use_id}
```

Empty string `tool_use_id` is accepted. Providers will fail or ignore it.

**Fix:** Validate non-empty.

---

### VAL-6: `execute_tool` passes kwargs without signature validation

**File:** `_tools/_executor.py:49`

```python
result = fn(**tool_call.input)
```

If `tool_call.input` contains keys the function doesn't accept, raises `TypeError`.
If required params are missing, raises `TypeError`. Both with no useful context.

**Fix:** Validate against `inspect.signature(fn)` before calling, or wrap with
a try/except that produces a better error:
```python
try:
    result = fn(**tool_call.input)
except TypeError as e:
    raise TypeError(f"Tool '{tool_call.name}' call failed: {e}") from e
```

---

## 10. Thread Safety & Resource Management

### THREAD-1: `_run_sync` thread is never cancelled on timeout

**File:** `_sync.py:37-43`

```python
thread.start()
thread.join(timeout=300)
if thread.is_alive():
    raise TimeoutError("Sync wrapper timed out after 300s")
```

The thread keeps running after timeout. There's no cancellation mechanism.
The coroutine inside will complete (or hang forever) in the background, potentially
holding SDK connections and leaking memory.

**Fix:** Use `daemon=True` so the thread doesn't block process exit. For proper
cancellation, pass a cancellation event and check it in the coroutine.

---

### THREAD-2: `_stream_sync` thread not checked after join

**File:** `_sync.py:79`

```python
thread.join(timeout=5)
# Nothing happens if thread is still alive
```

**Fix:** Check and log:
```python
thread.join(timeout=5)
if thread.is_alive():
    logger.warning("Stream thread did not exit within timeout")
```

---

### THREAD-3: `PricingRegistry` mutation is not thread-safe

**File:** `_pricing.py:54-60`

`pricing.register()`, `pricing.load()`, `pricing.reset()` all mutate `self._models`
without locks. In multithreaded applications, concurrent calls can corrupt the dict.

**Impact:** Low for typical use (pricing is loaded once at startup), but a library
should not assume single-threaded usage.

**Fix:** Use `threading.Lock()` or make the registry immutable (return new instances
from mutation methods).

---

### THREAD-4: `StreamState` lists mutated during iteration

**File:** `_base.py:82-88`

`state.tool_calls` and `state.thinking` are plain lists appended to during streaming.
If a user reads these while the stream is active, they see partial state with no
synchronization.

**Impact:** Low in single-threaded async, but if users spawn tasks that inspect state,
it's a race condition.

**Recommendation:** Document that state is only reliable after stream exhaustion.

---

## 11. Missing Features for Production

### PROD-1: No connection reuse or pooling guidance

Each `LLM()` instance creates a new SDK client. In high-throughput applications,
users should reuse `LLM` instances, but this isn't documented. The SDK clients
internally manage connection pools, but users need to know not to create `LLM`
per-request.

**Fix:** Document connection reuse best practice. Consider adding a class-level
client cache.

---

### PROD-2: No retry logic

The providers map SDK exceptions to our exception types but provide no automatic
retry. `RateLimitError` includes `retry_after` but nothing uses it.

**Recommendation:** This is intentionally left to users/higher layers (correct for
core/), but should be documented. Consider adding a `RetryConfig` utility.

---

### PROD-3: No timeout support on individual requests

Neither `complete()` nor `stream()` accept a timeout parameter. Long-running
requests hang indefinitely.

**Fix:** Add `timeout` parameter that flows to SDK clients.

---

### PROD-4: No request/response logging hooks

No way to inspect what's being sent to or received from providers. Essential for
debugging in production.

**Recommendation:** Add optional callback/logging middleware at the provider level.

---

### PROD-5: `SyncStreamResponse` missing `__enter__`/`__exit__`

Covered in DESIGN-3. Users cannot use `with` statement for sync streaming.

---

## 12. Summary & Priority Matrix

### Priority 1 — Must Fix (Bugs & Safety)

| ID | Issue | Files |
|----|-------|-------|
| BUG-1 | Gemini ignores `max_tokens` | `_gemini.py` |
| BUG-2 | `retry-after` ValueError (7 occurrences across 3 providers) | `_anthropic.py`, `_openai.py`, `_gemini.py` |
| BUG-3 | Stream `Response.raw` always None | `_llm.py` |
| BUG-4 | Anthropic KeyError on missing role/content | `_anthropic.py` |
| BUG-5 | Inconsistent cache token extraction (OpenAI worst) | `_openai.py`, all providers |
| BUG-7 | `_tool_to_sdk` KeyError on missing name (all 4 providers) | All providers |
| BUG-10 | OpenAI stream never sets `state.raw` | `_openai.py` |
| BUG-11 | xAI usage reads WRONG field names (all zero) | `_xai.py` |
| VAL-1 | No tool name validation | `_tools/__init__.py` |
| VAL-2 | No thinking_effort validation | `_llm.py` |

### Priority 2 — Should Fix (Consistency & Correctness)

| ID | Issue | Files |
|----|-------|-------|
| INCONSISTENT-1 | `max_tokens` naming across providers | All providers |
| INCONSISTENT-3 | `thinking_budget` silently dropped (OpenAI, xAI) | `_openai.py`, `_xai.py` |
| INCONSISTENT-4 | Error handling patterns vary significantly | All providers |
| INCONSISTENT-6 | Message role defaulting (Anthropic only) | `_anthropic.py` |
| INCONSISTENT-10 | `_SDK_PARAMS` not standardized (`stop` vs `stop_sequences`, etc.) | All providers |
| SILENT-1 | xAI search_parameters dead code | `_xai.py` |
| SILENT-2 | base_url silently ignored (Gemini, xAI) | `_providers/__init__.py` |
| SILENT-5 | ToolGroup silent overwrite | `_tools/_group.py` |
| SILENT-6 | Union types collapse silently | `_tools/_schema.py` |
| SILENT-7 | xAI thinking_budget silently popped | `_xai.py` |
| BP-1 | Bare except Exception (4 instances) | `_tools/_schema.py` |
| BP-2 | APIError docstring says "HTTP" (wrong for gRPC) | `_exceptions.py` |
| BP-6 | No kw_only=True on dataclasses | `_response.py` |
| TYPE-1 | `tools: Any` in public API | `_llm.py` |
| TYPE-2 | `output_schema: Any` in public API | `_llm.py` |
| HARD-1 | Thinking budgets duplicated in Anthropic + Gemini | `_anthropic.py`, `_gemini.py` |
| BUG-9 | Gemini tool result loses function name (name is REQUIRED) | `_gemini.py` |
| SDK-3 | Anthropic tool trick → native structured output (lifts tools+schema restriction) | `_anthropic.py` |

### Priority 3 — Should Improve (Production Quality)

| ID | Issue | Files |
|----|-------|-------|
| SDK-1 | No multimodal support (all providers) | All providers |
| SDK-4 | OpenAI missing stream_events override | `_openai.py` |
| SDK-7 | No tool_choice support | All providers |
| INCONSISTENT-2 | stream_events real-time inconsistency | All providers |
| INCONSISTENT-5 | close() missing (Gemini) | `_gemini.py` |
| INCONSISTENT-8 | Structured output parse failure silenced | `_openai.py`, `_gemini.py`, `_xai.py` |
| DESIGN-2 | state_holder list hack | `_llm.py` |
| DESIGN-3 | SyncStreamResponse no ctx mgr | `_response.py` |
| DESIGN-4 | to_message loses parsed/thinking | `_response.py` |
| BP-3 | Missing __all__ in providers | `_providers/__init__.py` |
| BP-5 | Thread daemon inconsistency | `_sync.py` |
| THREAD-1 | Thread never cancelled on timeout | `_sync.py` |
| THREAD-2 | Stream thread not checked after join | `_sync.py` |
| PROD-3 | No timeout support on requests | `_llm.py` |

### Priority 4 — Nice to Have

| ID | Issue | Files |
|----|-------|-------|
| BUG-6 | Pricing truthiness semantics | `_pricing.py` |
| BUG-8 | output_schema double extraction (all 4 providers) | All providers |
| SILENT-4 | prepare_tools empty list → None | `_tools/__init__.py` |
| SILENT-8 | Anthropic tools+output_schema error (others work) | `_anthropic.py` |
| SILENT-9 | Empty enum invalid schema | `_tools/_schema.py` |
| SDK-2 | No prompt cache control | All providers |
| SDK-3 | Anthropic tool trick vs native JSON mode | `_anthropic.py` |
| SDK-5 | Gemini server-side tools not exposed | `_gemini.py` |
| SDK-6 | OpenAI parallel_tool_calls not exposed | `_openai.py` |
| SDK-8 | Thinking only in stream_events, not stream (by design) | `_anthropic.py` |
| DESIGN-1 | _INIT_DEFAULTS duplication | `_llm.py` |
| DESIGN-5 | Cost calculated in two places | providers + `_llm.py` |
| HARD-2 | max_tokens=4096 default (acceptable) | `_llm.py` |
| HARD-3 | Sync timeout hardcoded (300s) | `_sync.py` |
| HARD-4 | Stream thread join timeout hardcoded (5s) | `_sync.py` |
| BP-4 | `from __future__ import annotations` (removable in 3.14) | All files |
| BP-8 | Partial string-like Response | `_response.py` |
| TYPE-3 | StreamEvent.kind no runtime validation | `_base.py` |
| TYPE-4 | _resolve_output_schema accepts Any | `_response.py` |
| THREAD-3 | PricingRegistry not thread-safe | `_pricing.py` |
| THREAD-4 | StreamState lists mutated during iteration | `_base.py` |
| VAL-3 | No thinking_budget range validation | `_llm.py` |
| VAL-4 | OutputSchema not validated | `_response.py` |
| VAL-5 | tool_result accepts empty tool_use_id | `_content.py` |
| VAL-6 | execute_tool no signature check | `_tools/_executor.py` |
| INCONSISTENT-7 | Model name from response (Gemini/xAI don't) | `_gemini.py`, `_xai.py` |
| INCONSISTENT-9 | System message handling differences | All providers |
| PROD-1 | No connection reuse docs | docs |
| PROD-2 | No retry logic (by design, needs docs) | design |
| PROD-4 | No request/response logging hooks | design |
| PROD-5 | SyncStreamResponse missing enter/exit | `_response.py` |

---

**Total findings: 67**
- Priority 1 (must fix): 9
- Priority 2 (should fix): 17
- Priority 3 (should improve): 14
- Priority 4 (nice to have): 31 (includes informational)
