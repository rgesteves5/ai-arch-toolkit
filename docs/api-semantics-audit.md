# API Semantics Audit — ai-arch-toolkit

A comprehensive audit of naming inconsistencies, semantic drift, and
design-smell across the Client, Provider, Agent, and Middleware layers.

---

## 1. `chat()` vs `complete()` — Naming Mismatch at the Boundary

| Layer | Method | What it does |
|-------|--------|--------------|
| **Client** (public) | `chat()` | Send messages → get `Response` |
| **Provider** (internal) | `complete()` | Send messages → get `Response` |

They're the same operation, but named differently at the public vs internal
boundary. The public name `chat` leaks OpenAI's "Chat Completions" naming
convention into what should be a provider-agnostic abstraction.

**Worse**: `chat()` is misleading for many real use cases:

- Single-shot prompts: `client.chat("What is 2+2?")` — not a "chat"
- Tool calling: `client.chat(messages, tools=[...])` — this is tool-augmented
  completion
- Structured output: `client.chat(messages, json_schema=...)` — this is
  constrained generation
- Agent planning: `self.client.chat(messages, system="You are a planner...")`
  — every agent uses `chat()` for what are completions

The internal name `complete` is actually more accurate than the public name.

**Files:**
- `src/ai_arch_toolkit/llm/_client.py:81` — `Client.chat()`
- `src/ai_arch_toolkit/llm/_client.py:118` — calls `self._provider.complete()`
- `src/ai_arch_toolkit/llm/_providers/_base.py:25` — `BaseProvider.complete()`

---

## 2. `chat()` vs `stream()` — False Dichotomy

`chat()` and `stream()` sound like **different operations**, but they're the
**same operation** (send messages, get a response) with different delivery
modes. Streaming is *how you receive the response*, not *what you're asking
the LLM to do*.

Current surface:

```python
client.chat(...)           # → Response
client.stream(...)         # → Iterator[str]
client.stream_events(...)  # → Iterator[StreamEvent]
```

This implies three fundamentally different things. In reality they're one
operation with three return-type options.

**Files:**
- `src/ai_arch_toolkit/llm/_client.py:81` — `chat()`
- `src/ai_arch_toolkit/llm/_client.py:127` — `stream()`
- `src/ai_arch_toolkit/llm/_client.py:154` — `stream_events()`

---

## 3. Feature Asymmetry Across the Three Methods

The three client methods accept **different subsets** of parameters, creating
a confusing capability matrix:

| Feature | `chat()` | `stream()` | `stream_events()` |
|---------|:--------:|:----------:|:------------------:|
| `tools` | Yes | **No** | Yes |
| `json_schema` | Yes | **No** | **No** |
| `system` | Yes | Yes | Yes |
| `timeout` | Yes | Yes | Yes |
| `**kwargs` | Yes | Yes | Yes |

So if you want streaming + tools, you **must** use `stream_events()`, not
`stream()`. If you want streaming + structured output, there's **no method
at all** that explicitly supports it (you'd need to pass it through `**kwargs`
and hope the provider handles it).

The same asymmetry exists at the provider level:

| Feature | `complete()` | `stream()` | `stream_events()` |
|---------|:------------:|:----------:|:------------------:|
| `tools` | Yes | **No** | Yes |
| `json_schema` | Yes | **No** | **No** |

**Files:**
- `src/ai_arch_toolkit/llm/_providers/_base.py:25-52` — the three abstract
  method signatures with different parameter sets
- `src/ai_arch_toolkit/llm/_client.py:81-191` — client-side signatures mirror
  this asymmetry

---

## 4. `stream()` Returns Raw Strings — Lossy by Design

`stream()` yields `Iterator[str]` — raw text chunks. You lose:
- Usage / token counts
- Stop reason
- Tool calls
- Thinking blocks

It's a convenience shortcut that silently drops metadata, but it's presented
as a peer method alongside `chat()` and `stream_events()`. Users who start
with `stream()` hit a wall when they need tool calls or usage, and must switch
to the completely different `stream_events()` API.

**Files:**
- `src/ai_arch_toolkit/llm/_client.py:127-152` — `stream()` returns
  `Iterator[str]`
- `src/ai_arch_toolkit/llm/_client.py:154-190` — `stream_events()` returns
  `Iterator[StreamEvent]`

---

## 5. Hidden Capabilities via `**kwargs` — Non-Discoverable API

All three client methods and all provider methods accept `**kwargs` which are
forwarded to the provider payload builder. This means features like
`thinking`, `server_tools`, `temperature`, `max_tokens`, etc. **work** but
are completely invisible in the method signatures.

Examples of hidden-but-functional kwargs:
- `thinking=ThinkingConfig(...)` — works through kwargs in all three methods
- `server_tools=[ServerTool(...)]` — works through kwargs
- `temperature=0.7` — works through kwargs
- `max_output_tokens=1024` — works through kwargs

Users who read the signature of `stream()` see `(messages, *, system, timeout,
**kwargs)` and have no idea that tool-like features are available via kwargs.
The type system and IDE autocompletion cannot help.

**Files:**
- `src/ai_arch_toolkit/llm/_providers/_openai_compat.py:189+` — payload
  builder reads from kwargs
- `src/ai_arch_toolkit/llm/_providers/_openai_responses.py:159+` — same
- `src/ai_arch_toolkit/llm/_providers/_anthropic.py` — same

---

## 6. Provider Names Mix Vendor Identity with API Surface

The `provider` string conflates two orthogonal concerns:

1. **Vendor** (who): openai, xai, anthropic, gemini, mistral, groq
2. **API surface** (what protocol): Chat Completions vs Responses API

Current provider names:

| Provider string | Vendor | API surface |
|-----------------|--------|-------------|
| `"openai"` | OpenAI | Chat Completions (`/v1/chat/completions`) |
| `"openai-responses"` | OpenAI | Responses API (`/v1/responses`) |
| `"xai"` | xAI | Chat Completions (`/v1/chat/completions`) |
| `"xai-responses"` | xAI | Responses API (xAI's responses endpoint) |
| `"anthropic"` | Anthropic | Messages API |
| `"gemini"` | Google | Gemini API |
| `"mistral"` | Mistral | Chat Completions (compat) |
| `"groq"` | Groq | Chat Completions (compat) |

Problems:
- A user must know to switch from `"xai"` to `"xai-responses"` just to use
  server tools (see `examples/09_server_tools.py:9` — has a comment warning
  about this).
- The `-responses` suffix is an implementation detail. Why should users care
  which HTTP endpoint is hit?
- Some vendors have one provider string, others have two. Inconsistent.

**Files:**
- `src/ai_arch_toolkit/llm/_providers/__init__.py:48-79` — provider dispatch
- `src/ai_arch_toolkit/llm/_providers/_openai_compat.py:29-48` —
  `OPENAI_COMPAT_PROVIDERS` dict
- `examples/09_server_tools.py:9` — `# Note: must use "xai-responses"
  provider (not "xai") for ServerTool support`

---

## 7. Async Naming Inconsistency Across Classes

Three different async naming conventions coexist:

| Class | Sync | Async |
|-------|------|-------|
| **Client / AsyncClient** | `chat()`, `stream()` | `chat()`, `stream()` (separate class) |
| **FallbackClient** | `chat()`, `stream()` | `achat()`, `astream()` (prefix on same class) |
| **BaseProvider** | `complete()`, `stream()` | `acomplete()`, `astream()` (prefix on same class) |
| **BaseAgent** | `run()` | `async_run()` (different prefix style) |
| **Middleware** | `before()`, `after()` | `abefore()`, `aafter()` (prefix on same protocol) |

Three patterns:
1. **Separate class** (`Client` vs `AsyncClient`)
2. **`a`-prefix** (`complete` → `acomplete`, `before` → `abefore`)
3. **`async_`-prefix** (`run` → `async_run`)

This means users can't predict the async method name from the sync one.

**Files:**
- `src/ai_arch_toolkit/llm/_async_client.py:87` — `AsyncClient.chat()`
- `src/ai_arch_toolkit/llm/_fallback.py:66` — `FallbackClient.achat()`
- `src/ai_arch_toolkit/llm/_providers/_base.py:55` — `BaseProvider.acomplete()`
- `src/ai_arch_toolkit/agents/_base.py:245` — `BaseAgent.async_run()`
- `src/ai_arch_toolkit/llm/_middleware.py:33` — `Middleware.abefore()`

---

## 8. Agent Streaming Is Semantically Broken

### 8a. `run()` secretly accepts `stream=True` via kwargs

Every agent's `run()` pops a `stream` kwarg:

```python
def run(self, task: str, **kwargs: Any) -> AgentResult:
    stream = kwargs.pop("stream", False)
    if stream:
        return self.run_stream(task, **kwargs)  # type: ignore
```

But `run()` is typed as returning `AgentResult`, while `run_stream()` returns
`Iterator[AgentStep]`. So `run(task, stream=True)` **lies about its return
type**. The `# type: ignore` comment in the ReAct implementation confirms
this is a known type hole.

### 8b. `run_stream()` is post-hoc for most agents

The base class default:

```python
def run_stream(self, task: str, **kwargs: Any) -> Iterator[AgentStep]:
    result = self.run(task, stream=False, **kwargs)
    yield from result.steps
```

This runs the **entire** agent to completion, then yields the stored steps.
It's not streaming at all — it's "batch then replay." Only `ReActAgent`
overrides this with true incremental streaming.

### 8c. Four methods, confusing overlap

| Method | Returns | True streaming? |
|--------|---------|-----------------|
| `run()` | `AgentResult` | No |
| `run(stream=True)` | `Iterator[AgentStep]` (type lie) | Depends on agent |
| `run_stream()` | `Iterator[AgentStep]` | Only ReAct |
| `async_run()` | `AgentResult` | No |
| `async_run_stream()` | `AsyncIterator[AgentStep]` | Only ReAct |

**Files:**
- `src/ai_arch_toolkit/agents/_base.py:241-260` — base class methods
- `src/ai_arch_toolkit/agents/_react.py:155-162` — ReAct dispatches to
  `run_stream` when `stream=True`
- `src/ai_arch_toolkit/agents/_tot.py:56-58` — ToT pops `stream` kwarg
- `src/ai_arch_toolkit/agents/_plan_execute.py:78-80` — same pattern

---

## 9. Middleware Only Works Fully with `chat()`

### 9a. ResponseCache only caches `chat` operations

```python
def before(self, request: Request) -> Request:
    if request.operation != "chat":
        return request   # skip stream / stream_events
```

### 9b. CostTracker tracks `chat` and `stream_events`, but not `stream`

```python
def after(self, request: Request, result: Any) -> Any:
    if isinstance(result, Response):           # chat → works
        self._record_usage(request, result.usage)
    if request.operation == "stream_events":   # stream_events → works
        return self._wrap_stream_events(...)
    return result                              # stream → silently ignored
```

`stream()` returns `Iterator[str]` — there's no `Usage` object to extract,
so the cost tracker **cannot work** with `stream()`. This is not documented.

Users who use `stream()` with a `CostTracker` middleware will silently get
zero-cost reports.

**Files:**
- `src/ai_arch_toolkit/llm/_cache.py:74-76` — `operation != "chat"` guard
- `src/ai_arch_toolkit/llm/_cost.py:280-286` — `after()` only handles
  `Response` and `stream_events`

---

## 10. `stop_reason` Has Provider-Specific Values — Not Unified

The `Response.stop_reason` field is populated with raw, unmapped values from
each provider:

| Provider | Source field | Example values |
|----------|-------------|----------------|
| Anthropic | `stop_reason` | `"end_turn"`, `"max_tokens"`, `"tool_use"` |
| OpenAI (compat) | `finish_reason` | `"stop"`, `"length"`, `"tool_calls"` |
| OpenAI (responses) | `status` | `"completed"`, `"incomplete"` |
| Gemini | `finishReason` | `"STOP"`, `"MAX_TOKENS"`, `"SAFETY"` |

Code that checks `response.stop_reason` must handle **four different
vocabularies** depending on which provider was used. This defeats the purpose
of having a "unified" Response type.

**Files:**
- `src/ai_arch_toolkit/llm/_providers/_anthropic.py:120` —
  `stop_reason=raw.get("stop_reason", "")`
- `src/ai_arch_toolkit/llm/_providers/_openai_compat.py:133` —
  `stop_reason=choice.get("finish_reason", "")`
- `src/ai_arch_toolkit/llm/_providers/_openai_responses.py:135` —
  `stop_reason=raw.get("status", "")`
- `src/ai_arch_toolkit/llm/_providers/_gemini.py:107` —
  `stop_reason=candidate.get("finishReason", "")`

---

## 11. Gemini Tool Calls Use Empty IDs

Gemini's API doesn't return tool call IDs, so the provider generates empty
string IDs:

```python
ToolCall(id="", name=..., arguments=...)
```

Code that relies on `tool_call.id` for matching results back to calls (which
is how OpenAI and Anthropic work) will silently break with Gemini.

**Files:**
- `src/ai_arch_toolkit/llm/_providers/_gemini.py:90` — empty ID

---

## 12. `FallbackClient` Has Protocol Detection Smell

The `FallbackClient._get_async_stream_events` method tries two different
method names:

```python
stream_fn = getattr(client, "astream_events", None)
if stream_fn is None:
    stream_fn = getattr(client, "stream_events", None)
```

This exists because `FallbackClient` doesn't know if it's wrapping an
`AsyncClient` (which has `stream_events`) or another `FallbackClient`
(which has `astream_events`). This is a symptom of the inconsistent async
naming (issue #7).

**Files:**
- `src/ai_arch_toolkit/llm/_fallback.py:213-230` — protocol sniffing

---

## 13. `BaseAgent` Types `client` and `tools` as `Any`

```python
class BaseAgent(ABC):
    def __init__(
        self,
        client: Any,    # no protocol/ABC
        tools: Any,     # no protocol/ABC
        *,
        config: AgentConfig | None = None,
    ) -> None:
```

There's no protocol or ABC defining what a "client" must look like from the
agent's perspective. Any object is accepted; agents just call `.chat()` on it
by convention. Typos or wrong objects fail at runtime, not at type-check time.
Same for `tools` — it could be a `ToolRegistry`, a list, or anything.

**Files:**
- `src/ai_arch_toolkit/agents/_base.py:187-188` — `client: Any, tools: Any`

---

## 14. `GuardrailMiddleware` Checks Stream Chunks Individually — Not Accumulated

```python
def _wrap_stream(self, stream: Iterator[str]) -> Iterator[str]:
    for chunk in stream:
        self._check_text(chunk, stage="output")  # checks each chunk alone
        yield chunk
```

A blocked pattern like `"password"` split across two chunks (`"pass"` +
`"word"`) would **never be caught**. The `stream_events` path has the same
problem with `event.text` fragments. Only the `chat()` path (which gets the
full `Response.text`) is reliable for guardrail checking.

**Files:**
- `src/ai_arch_toolkit/llm/_guardrails.py:87-90` — per-chunk checking
- `src/ai_arch_toolkit/llm/_guardrails.py:92-96` — per-event checking

---

## 15. `ConversationItem` Not Exported in Public API

`_types.py:169` defines `type ConversationItem = Message | ToolResult` but it
is **not** re-exported through `llm/__init__.py` or the top-level
`__init__.py`. Yet it appears in public-facing method signatures:

- `Client._normalize_input()` returns `list[ConversationItem]`
- `preview_conversation_usage_and_cost()` accepts `list[ConversationItem]`
- All provider methods accept `list[ConversationItem]`

Users who want to type-annotate their own code can't import it cleanly from
the public package.

**Files:**
- `src/ai_arch_toolkit/llm/_types.py:169` — definition
- `src/ai_arch_toolkit/__init__.py` — not listed in exports
- `src/ai_arch_toolkit/llm/__init__.py` — not listed in exports

---

## 16. `BatchClient` / `AsyncBatchClient` Are Disconnected from `Client`

`BatchClient` and `AsyncBatchClient` don't share middleware, retry config,
or provider resolution with `Client`/`AsyncClient`. A user who sets up a
`Client` with middleware (guardrails, cost tracking, caching, tracing) gets
**none of that** when they use `BatchClient` for the same provider. Two
completely parallel initialization paths for the same vendor.

**Files:**
- `src/ai_arch_toolkit/llm/_batch.py` — standalone `BatchClient`
- `src/ai_arch_toolkit/llm/_async_batch.py` — standalone `AsyncBatchClient`
- `src/ai_arch_toolkit/llm/_client.py` — `Client` with middleware pipeline

---

## 17. `ToolRegistry` Uses a Fourth Async Naming Pattern

`ToolRegistry` uses `async_execute` (full `async_` prefix with underscore):

```python
class ToolRegistry:
    def execute(self, tool_call: ToolCall) -> str: ...
    async def async_execute(self, tool_call: ToolCall) -> str: ...
```

This is yet another async convention alongside the three already identified
in issue #7:

| Layer | Pattern |
|-------|---------|
| Client | Separate class (`AsyncClient`) |
| Provider / Middleware | `a`-prefix (`acomplete`, `abefore`) |
| Agent | `async_`-prefix (`async_run`) |
| **ToolRegistry** | **`async_`-prefix (`async_execute`)** |

Four distinct conventions in one project.

**Files:**
- `src/ai_arch_toolkit/tools/_registry.py:87` — `async_execute`
- `src/ai_arch_toolkit/tools/_registry.py:169` — `ToolRegistryView.async_execute`

---

## 18. `Middleware` Protocol Says All Four Methods Required — Runtime Says Otherwise

The `Middleware` Protocol declares all four methods as required:

```python
class Middleware(Protocol):
    def before(self, request: Request) -> Request: ...
    def after(self, request: Request, result: Any) -> Any: ...
    async def abefore(self, request: Request) -> Request: ...
    async def aafter(self, request: Request, result: Any) -> Any: ...
```

But `AsyncClient._run_before` does runtime `hasattr` checks:

```python
async def _run_before(self, request: Request) -> Request:
    for m in self._middleware:
        if hasattr(m, "abefore"):       # optional at runtime!
            request = await m.abefore(request)
        else:
            request = m.before(request)  # sync fallback
```

The Protocol *says* all four are required but the runtime *treats* them as
optional. Anyone implementing custom middleware based on the Protocol type
will implement all four; anyone reading the runtime code will think they can
skip `abefore`/`aafter`. The contract is contradictory.

**Files:**
- `src/ai_arch_toolkit/llm/_middleware.py:26-35` — Protocol with all four
- `src/ai_arch_toolkit/llm/_async_client.py:55-56` — `hasattr` fallback

---

## 19. `Request.operation` Is a Bare `str`, Not a Constrained Type

```python
@dataclass(slots=True)
class Request:
    operation: str   # could be anything
```

Middleware implementations hardcode string comparisons:

```python
if request.operation != "chat":          # ResponseCache
if request.operation == "stream_events": # CostTracker
```

If the client method names change, these strings silently break. No
type-checker will catch `"chat"` vs `"complete"` if the method is renamed.
Should be `Literal["chat", "stream", "stream_events"]` at minimum.

**Files:**
- `src/ai_arch_toolkit/llm/_middleware.py:15` — `operation: str`
- `src/ai_arch_toolkit/llm/_cache.py:75` — string comparison
- `src/ai_arch_toolkit/llm/_cost.py:284` — string comparison
- `src/ai_arch_toolkit/llm/_tracing.py:78-80` — string comparison

---

## 20. Top-Level `__init__.py` Exports ~100 Symbols — Flat Namespace Pollution

The top-level package re-exports everything: correction factors, cost helpers,
token estimators, cache backends, middleware, types, agents, parsing utilities.
`from ai_arch_toolkit import *` dumps ~100 names.

Internal implementation types like `Request`, `InMemoryCacheBackend`, and
`CLAUDE_3_CORRECTION_FACTOR` sit alongside primary user-facing types like
`Client` and `Response`. No hierarchy of importance — everything is equally
prominent. New users can't tell which 5-10 symbols they actually need.

**Files:**
- `src/ai_arch_toolkit/__init__.py:115-215` — 100-entry `__all__`

---

## 21. Middleware Short-Circuit Only Works for `chat()`, Not Streaming

The short-circuit mechanism (used by `ResponseCache` to return cached results)
is only checked in `chat()`:

```python
# chat() — checks short-circuit
if _SHORT_CIRCUIT_RESULT_KEY in request.context:
    return self._run_after(request, request.context[_SHORT_CIRCUIT_RESULT_KEY])

# stream() — no check, goes straight to provider
stream = self._provider.stream(request.messages, ...)

# stream_events() — no check, goes straight to provider
events = self._provider.stream_events(request.messages, ...)
```

Any middleware that sets `_SHORT_CIRCUIT_RESULT_KEY` in `before()` is silently
ignored by `stream()` and `stream_events()`. This makes it impossible to
build caching, circuit-breaking, or mock middleware that works across all
delivery modes.

**Files:**
- `src/ai_arch_toolkit/llm/_client.py:115-117` — short-circuit in `chat()`
- `src/ai_arch_toolkit/llm/_client.py:151` — `stream()` skips it
- `src/ai_arch_toolkit/llm/_client.py:184` — `stream_events()` skips it

---

## 22. `Request` Metadata Is Inconsistent Across Operations

The `Request` object constructed by the client populates different fields
depending on which method is called:

| Field | `chat()` | `stream()` | `stream_events()` |
|-------|:--------:|:----------:|:------------------:|
| `messages` | Yes | Yes | Yes |
| `system` | Yes | Yes | Yes |
| `tools` | Yes | **No** | Yes |
| `json_schema` | Yes | **No** | **No** |

This affects middleware observability. For example, `TracingMiddleware` logs
tool count via `len(request.tools or ())` — but for `stream()` calls,
`request.tools` is always `None` even if tools were passed via `**kwargs`.
Middleware sees an incomplete picture of the actual request.

**Files:**
- `src/ai_arch_toolkit/llm/_client.py:103-113` — `chat()` builds full Request
- `src/ai_arch_toolkit/llm/_client.py:141-150` — `stream()` omits tools/schema
- `src/ai_arch_toolkit/llm/_client.py:173-189` — `stream_events()` omits schema
- `src/ai_arch_toolkit/llm/_tracing.py:108` — reads `request.tools`

---

## 23. Gemini Silently Drops Unknown `**kwargs`

Issue #5 states kwargs are forwarded to providers. This is true for
OpenAI-compat, Anthropic, and OpenAI-responses — but **Gemini effectively
drops unknown kwargs**. Features that work through kwargs on other providers
(e.g. custom parameters) silently do nothing on Gemini. No error, no warning.

**Files:**
- `src/ai_arch_toolkit/llm/_providers/_openai_compat.py:207` — forwards kwargs
- `src/ai_arch_toolkit/llm/_providers/_anthropic.py:208` — forwards kwargs
- `src/ai_arch_toolkit/llm/_providers/_openai_responses.py:215` — forwards kwargs
- `src/ai_arch_toolkit/llm/_providers/_gemini.py:233` — drops unknown kwargs

---

## 24. "Unified" Content/Message Types Accept Everything — Providers Reject at Runtime

The type system allows constructing any `Message` with any `Content` for any
provider. But providers have hard restrictions:

| Provider | Runtime rejection |
|----------|-------------------|
| Gemini | Rejects non-user/non-assistant roles |
| OpenAI (compat) | Rejects `DocumentPart` |
| Anthropic | Rejects `AudioPart` |

A user can write `Message(role="system", content=AudioPart(...))` and the type
checker is happy. The error surfaces only at HTTP call time, deep in a provider
implementation — with a provider-specific error message, not a framework-level
one.

**Files:**
- `src/ai_arch_toolkit/llm/_providers/_gemini.py:155` — role rejection
- `src/ai_arch_toolkit/llm/_providers/_openai_compat.py:98` — DocumentPart rejection
- `src/ai_arch_toolkit/llm/_providers/_anthropic.py:82` — AudioPart rejection

---

## 25. Streaming `Usage` Events Have Inconsistent Shape Across Providers

The `StreamEvent(type="usage")` emitted during streaming has different
completeness depending on the provider:

| Provider | `input_tokens` | `output_tokens` | `total_tokens` |
|----------|:--------------:|:---------------:|:--------------:|
| Anthropic | **No** (streaming usage has output only) | Yes | **No** |
| OpenAI (compat) | Yes | Yes | Yes |
| OpenAI (responses) | Yes | Yes | Yes |
| Gemini | Yes | Yes | Yes |

`CostTracker` and `TracingMiddleware` both read `event.usage.input_tokens`
and `event.usage.total_tokens` from streaming events — but Anthropic's
streaming usage only reports output tokens. This means cost/tracing reports
from Anthropic streaming are systematically incomplete compared to other
providers.

**Files:**
- `src/ai_arch_toolkit/llm/_providers/_anthropic.py:370` — partial usage
- `src/ai_arch_toolkit/llm/_cost.py:346-348` — reads all usage fields
- `src/ai_arch_toolkit/llm/_tracing.py:130-133` — reads all usage fields

---

## 26. `FallbackClient` Does Not Fallback After Partial Stream Emission

```python
def _stream_with_fallback(self, ...):
    for index, client in enumerate(self._clients, start=1):
        emitted = False
        try:
            for chunk in stream:
                emitted = True
                yield chunk
            return
        except self._fallback_on as exc:
            if emitted:
                raise   # no fallback once chunks have been yielded
```

Once any chunk has been yielded to the consumer, `FallbackClient` re-raises
instead of falling back. This is a deliberate design choice (you can't
un-yield data), but it's an undocumented behavioral contract. Users might
expect fallback to work transparently for streaming, but it only protects
against connection-time failures, not mid-stream failures.

**Files:**
- `src/ai_arch_toolkit/llm/_fallback.py:120-121` — `stream` path
- `src/ai_arch_toolkit/llm/_fallback.py:146-147` — `stream_events` path

---

---

## 27. Only Anthropic Returns Thinking Blocks — Others Silently Ignore

`Response` has `thinking` and `thinking_blocks` fields, and all providers
accept `thinking=ThinkingConfig(...)` in kwargs. But only Anthropic populates
these fields and emits `"thinking"` stream events:

| Provider | Sets `Response.thinking`? | Emits `"thinking"` events? |
|----------|:------------------------:|:--------------------------:|
| Anthropic | Yes | Yes |
| OpenAI (compat) | **No** (empty string) | **No** |
| OpenAI (responses) | **No** (empty string) | **No** |
| Gemini | **No** (empty string) | **No** (even though API supports it) |

Users who pass `thinking=ThinkingConfig(effort="high")` to a non-Anthropic
provider will get an empty `response.thinking` — no error, no warning. The
framework silently degrades without telling you.

**Files:**
- `src/ai_arch_toolkit/llm/_providers/_anthropic.py` — full thinking support
- `src/ai_arch_toolkit/llm/_providers/_gemini.py` — sends `thinkingConfig`
  but doesn't parse thinking from response
- `src/ai_arch_toolkit/llm/_providers/_openai_compat.py` — sets
  `reasoning_effort` but `thinking=""` in Response

---

## 28. Cancellation Token Is Non-Functional Dead Code in All Agents

Every agent accepts and checks a `cancellation_token` kwarg, but the base
class implementation is fully stubbed:

```python
def _resolve_cancellation_token(self, override: object | None) -> None:
    """Cancellation is currently disabled."""
    _ = override
    return None

def _is_cancelled(self, token: object | None) -> bool:
    """Cancellation is currently disabled."""
    _ = token
    return False
```

All agent implementations in scope of this audit pop `cancellation_token` from kwargs, pass it through
`_resolve_cancellation_token`, and check `_is_cancelled` in every loop
iteration — but it always returns `False`. This is ~50 lines of dead code
per agent that misleads users into thinking cancellation works.

**Files:**
- `src/ai_arch_toolkit/agents/_base.py:202-210` — stubs
- Every agent file — dead `_is_cancelled()` checks in loops

---

## 29. Inner Agents Don't Propagate `on_event` Callback

Agents that spawn inner agents (`ReflexionAgent`, `PlanExecuteAgent`,
`LATSAgent`) create inner `ReActAgent` instances with a new `AgentConfig`
that **does not** include the outer agent's `on_event` callback:

```python
# PlanExecuteAgent — inner agent loses event handler
inner_agent = ReActAgent(
    self.client,
    self.tools,
    config=AgentConfig(
        max_iterations=3,
        system=step_system,
        max_tokens=self.config.max_tokens,
        # on_event=??? — NOT PASSED
    ),
)
```

Users who set `on_event` on the outer agent won't see tool calls, tool
results, or step events from inner agent execution. Major observability gap.

**Files:**
- `src/ai_arch_toolkit/agents/_plan_execute.py:66` — inner AgentConfig
- `src/ai_arch_toolkit/agents/_reflexion.py:45` — inner AgentConfig
- `src/ai_arch_toolkit/agents/_lats.py:227` — inner AgentConfig

---

## 30. Agent Custom Kwargs Are Invisible — Popped from `**kwargs`

Several agents accept configuration through `**kwargs` that are silently
popped with defaults:

```python
# ToT
max_depth = kwargs.pop("max_depth", 3)
branching_factor = kwargs.pop("branching_factor", 3)
beam_width = kwargs.pop("beam_width", 2)
search_strategy = kwargs.pop("search_strategy", "bfs")

# LATS
exploration_weight = kwargs.pop("exploration_weight", 1.41)
num_expansions = kwargs.pop("num_expansions", 2)
evaluator = kwargs.pop("evaluator", None)

# Reflexion
evaluator = kwargs.pop("evaluator", None)
threshold = kwargs.pop("threshold", 0.8)
```

These are not in `AgentConfig`, not in `__init__`, and not in type hints.
IDEs can't autocomplete them, type checkers can't validate them, and a typo
like `serch_strategy="bfs"` is silently ignored (it stays in kwargs and gets
forwarded to `client.chat()`).

**Files:**
- `src/ai_arch_toolkit/agents/_tot.py:68-71`
- `src/ai_arch_toolkit/agents/_lats.py:114-116`
- `src/ai_arch_toolkit/agents/_reflexion.py:71-75`
- `src/ai_arch_toolkit/agents/_compiler.py:292`

---

## 31. Provider-Specific Features Not in `BaseProvider` Contract

Several providers have methods or capabilities with no base class equivalent:

| Provider | Extra feature | In `BaseProvider`? |
|----------|---------------|:------------------:|
| Gemini | `create_cache()`, `complete_with_cache()` | **No** |
| Anthropic | `cache_control`, `computer_use` beta headers | **No** (kwargs) |
| OpenAI Responses | `previous_response_id` continuation | **No** (kwargs) |
| OpenAI Responses | `server_tools` | **No** (kwargs) |

Gemini is the worst case — `create_cache()` and `complete_with_cache()` are
entirely separate methods that only exist on `GeminiProvider`. There's no
way to use them through the `Client` facade at all.

**Files:**
- `src/ai_arch_toolkit/llm/_providers/_gemini.py` — cache methods
- `src/ai_arch_toolkit/llm/_providers/_base.py` — no cache in ABC

---

## 32. Gemini Doesn't Defensively Parse Tool Call Arguments

Anthropic, OpenAI-compat, and OpenAI-responses all wrap tool argument JSON
parsing in try-catch with a `{"_raw": ...}` fallback:

```python
# OpenAI-compat pattern (Anthropic similar)
try:
    arguments = json.loads(raw_args)
except (json.JSONDecodeError, TypeError):
    arguments = {"_raw": raw_args}
```

Gemini does **not** do this — it passes `functionCall.get("args", {})`
directly. If Gemini returns malformed args, the provider will crash instead
of gracefully degrading.

**Files:**
- `src/ai_arch_toolkit/llm/_providers/_openai_compat.py` — defensive parsing
- `src/ai_arch_toolkit/llm/_providers/_gemini.py:90` — no defensive parsing

---

## 33. Content-Type Error Semantics Differ: Raise vs Silent Skip

When a provider encounters an unsupported content type, the behavior varies:

| Provider | Unsupported type | Behavior |
|----------|------------------|----------|
| Anthropic | `AudioPart` | **Raises** exception |
| OpenAI (compat) | `DocumentPart` | **Raises** exception |
| OpenAI (responses) | `AudioPart` | **Silently skips** |
| xAI (responses) | `AudioPart` | **Silently skips** |

Some providers crash, others silently drop content. A user sending an
`AudioPart` gets an error from Anthropic but silently loses the audio from
OpenAI Responses. Neither behavior is documented.

**Files:**
- `src/ai_arch_toolkit/llm/_providers/_anthropic.py:82` — raises
- `src/ai_arch_toolkit/llm/_providers/_openai_compat.py:98` — raises
- `src/ai_arch_toolkit/llm/_providers/_openai_responses.py` — silent skip

---

## 34. `ConversationMemory` Breaks the `frozen=True` Convention

Every dataclass in the project uses `@dataclass(frozen=True, slots=True)`
except `ConversationMemory` and `SlidingWindowMemory`:

```python
@dataclass(slots=True)  # NOT frozen — mutable
class ConversationMemory:
    items: list[ConversationItem] = field(default_factory=list)
```

This is intentional (memory must be mutable), but it violates the project's
universal convention. The `_BudgetTracker` in `_base.py` is also mutable but
it's private. `ConversationMemory` is a public export and users may assume
the same immutability guarantees as every other dataclass in the package.

**Files:**
- `src/ai_arch_toolkit/llm/_memory.py:12-13` — mutable dataclass
- `src/ai_arch_toolkit/llm/_types.py` — every other dataclass is frozen

---

## 35. `@tool` Decorator Silently Suppresses Type Hint Errors

The `@tool` decorator catches **all** exceptions when resolving type hints:

```python
try:
    hints = get_type_hints(fn)
except Exception:
    hints = {}
```

Real errors (circular imports, `NameError` from forward references, missing
imports) are silently swallowed. The tool schema defaults all parameters to
`{"type": "string"}` when hints fail. Users get a working but incorrect tool
definition with no warning.

**Files:**
- `src/ai_arch_toolkit/tools/_decorator.py` — broad exception catch

---

## 36. Inner Agent `max_iterations` — Hardcoded vs Inherited

Agents that spawn inner `ReActAgent` instances use different strategies for
the inner agent's iteration limit:

| Outer Agent | Inner `max_iterations` | Source |
|-------------|:----------------------:|--------|
| Reflexion | `self.config.max_iterations` | Inherited from outer |
| PlanExecute | `3` | **Hardcoded** |
| LATS | `3` | **Hardcoded** |

This is inconsistent. A user setting `max_iterations=20` on a
`PlanExecuteAgent` would expect the inner tool-execution agent to have
proportional capacity, but it's always capped at 3.

**Files:**
- `src/ai_arch_toolkit/agents/_reflexion.py:45` — inherits
- `src/ai_arch_toolkit/agents/_plan_execute.py:66` — hardcoded `3`
- `src/ai_arch_toolkit/agents/_lats.py:227` — hardcoded `3`

---

## Summary Table

| # | Problem | Severity | Layer |
|---|---------|----------|-------|
| 1 | `chat()` vs `complete()` naming mismatch | Medium | Client ↔ Provider |
| 2 | `chat()` / `stream()` / `stream_events()` — false dichotomy | High | Client |
| 3 | Feature asymmetry (tools/json_schema) across methods | High | Client + Provider |
| 4 | `stream()` silently drops all metadata | Medium | Client |
| 5 | Hidden kwargs make API non-discoverable | Medium | Client + Provider |
| 6 | Provider names mix vendor + API surface | Medium | Provider factory |
| 7 | Three different async naming conventions | Medium | All layers |
| 8 | Agent streaming is type-unsafe and mostly fake | High | Agents |
| 9 | Middleware only fully works with `chat()` | High | Middleware |
| 10 | `stop_reason` not unified across providers | Medium | Provider |
| 11 | Gemini tool calls use empty IDs | Low | Gemini provider |
| 12 | FallbackClient protocol detection smell | Low | Fallback |
| 13 | `BaseAgent` types `client`/`tools` as `Any` | Medium | Agents |
| 14 | Guardrail stream checks per-chunk, not accumulated | Medium | Middleware |
| 15 | `ConversationItem` not exported in public API | Low | Types / Exports |
| 16 | BatchClient disconnected from Client middleware | Medium | Batch |
| 17 | ToolRegistry uses a fourth async naming pattern | Low | Tools |
| 18 | Middleware Protocol vs runtime `hasattr` contradiction | Medium | Middleware |
| 19 | `Request.operation` is bare `str`, not constrained | Low | Middleware |
| 20 | Top-level exports ~100 symbols, no hierarchy | Low | Package |
| 21 | Middleware short-circuit only works for `chat()` | High | Client + Middleware |
| 22 | `Request` metadata inconsistent across operations | Medium | Client + Middleware |
| 23 | Gemini silently drops unknown kwargs | Medium | Gemini provider |
| 24 | Unified types accept all content — providers reject at runtime | Medium | Types + Providers |
| 25 | Streaming `Usage` shape inconsistent across providers | Medium | Providers |
| 26 | FallbackClient no fallback after partial stream emission | Low | Fallback |
| 27 | Only Anthropic returns thinking blocks; others silently ignore | Medium | Providers |
| 28 | Cancellation token is non-functional dead code | Medium | Agents |
| 29 | Inner agents don't propagate `on_event` callback | High | Agents |
| 30 | Agent custom kwargs invisible, not type-safe | Medium | Agents |
| 31 | Provider-specific features not in BaseProvider contract | Medium | Providers |
| 32 | Gemini doesn't defensively parse tool call arguments | Low | Gemini provider |
| 33 | Content-type error semantics differ: raise vs silent skip | Medium | Providers |
| 34 | `ConversationMemory` breaks `frozen=True` convention | Low | Memory |
| 35 | `@tool` decorator silently suppresses type hint errors | Medium | Tools |
| 36 | Inner agent `max_iterations` hardcoded vs inherited | Low | Agents |

---

## Next Steps

These 36 issues cluster into design decisions:

1. **Client method surface** — How many methods, what names, what parameters?
2. **Provider naming** — Separate vendor from API surface?
3. **Async convention** — Pick one pattern and apply everywhere?
4. **stop_reason normalization** — Define a canonical vocabulary?
5. **Agent streaming** — Fix the type lies and fake streaming?
6. **Middleware coverage** — Should all middleware work with all delivery
   modes?
7. **Type safety** — Agent client protocol, `Request.operation` as Literal,
   content-type validation at framework level?
8. **Streaming consistency** — Usage shape, guardrail accumulation, fallback
   contracts?
9. **Export organization** — Tiered public API (core vs advanced)?
10. **Provider parity** — Thinking blocks, tool call IDs, defensive parsing,
    error semantics — normalize or document?
11. **Agent configuration** — Move hidden kwargs to AgentConfig or __init__?
    Propagate on_event to inner agents? Remove dead cancellation code?
