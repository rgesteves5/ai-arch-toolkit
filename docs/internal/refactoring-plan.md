# Refactoring Plan — First Principles Alignment

A refactoring plan that restructures `ai-arch-toolkit` to align with the four
primitives from `first_principles_llms.md`: **Content**, **Transform**,
**Identity**, **Memory** — and the five composition operators.

Each section maps audit findings to first-principles violations and prescribes
concrete changes.

---

## Guiding Principle

> An LLM is a stateless function: content in, content out.

Every layer of this toolkit wraps that function. The public API should reflect
the Transform shape (content in → content out) with consistent Identity
(naming, schemas, discovery) and clean Memory semantics (mutable vs immutable,
scoped access). Anything that obscures these primitives is a design smell.

---

## Phase 1: Unify the Transform Surface

**Audit issues addressed: #1, #2, #3, #4, #5, #21, #22**

The core problem: the Client exposes three methods (`chat`, `stream`,
`stream_events`) that are the *same Transform* with different delivery modes.
This violates the Transform primitive — one operation should have one name.

### 1.1 Rename `chat()` → `complete()` everywhere

The operation is a completion. `chat` leaks OpenAI vendor naming into what
should be a provider-agnostic Transform. The internal provider already uses
`complete()` — align the public surface to match.

```python
# Before
response = client.chat("What is 2+2?")

# After
response = client.complete("What is 2+2?")
```

Keep `chat` as a deprecated alias for one release cycle.

**Files:**
- `llm/_client.py` — rename `chat()` → `complete()`
- `llm/_async_client.py` — rename `chat()` → `complete()`
- `llm/_fallback.py` — rename `chat()`/`achat()` → `complete()`/`acomplete()`
- All agent files — update `self.client.chat(...)` → `self.client.complete(...)`
- All examples — update calls
- All tests — update calls and patches

### 1.2 Unify into one method with a `stream` parameter

Streaming is a delivery mode, not a different operation. Collapse three methods
into one:

```python
class Client:
    def complete(
        self,
        prompt_or_messages: str | Sequence[Message | ToolResult],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        json_schema: JsonSchema | None = None,
        thinking: ThinkingConfig | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        timeout: int | float | None = None,
        stream: Literal[False] = False,       # default: no stream
        **kwargs: Any,
    ) -> Response: ...

    @overload
    def complete(
        self,
        prompt_or_messages: ...,
        *,
        stream: Literal[True],
        **kwargs: Any,
    ) -> Iterator[StreamEvent]: ...
```

When `stream=False` (default): returns `Response`.
When `stream=True`: returns `Iterator[StreamEvent]`.

The old `stream()` (raw text chunks) becomes a helper, not a peer method:

```python
# Convenience — just yields event.text from stream events
def stream_text(
    self,
    prompt_or_messages: ...,
    **kwargs: Any,
) -> Iterator[str]:
    for event in self.complete(..., stream=True, **kwargs):
        if event.text:
            yield event.text
```

This eliminates the false dichotomy and the feature asymmetry — every call
gets the full parameter set regardless of delivery mode.

### 1.3 Unify the Provider ABC to match

```python
class BaseProvider(ABC):
    @abstractmethod
    def complete(
        self,
        messages: list[ConversationItem],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        json_schema: JsonSchema | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Response | Iterator[StreamEvent]: ...

    @abstractmethod
    async def acomplete(
        self,
        messages: list[ConversationItem],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        json_schema: JsonSchema | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Response | AsyncIterator[StreamEvent]: ...
```

Two abstract methods instead of six. Every provider implements the full
parameter set for both delivery modes.

### 1.4 Surface hidden kwargs as explicit parameters

The most-used kwargs become explicit in the signature (thinking, temperature,
max_output_tokens). This makes the Transform's schema (its Identity)
discoverable through IDE autocompletion and type checkers. Provider-specific
or rarely-used params remain in `**kwargs`.

### 1.5 Make middleware work with all delivery modes

With one `complete()` method, the `Request` object always carries the full
parameter set. Short-circuit, caching, cost tracking, and guardrails all
operate on the same code path. The `Request.operation` field becomes:

```python
operation: Literal["complete", "complete_stream"]
```

---

## Phase 2: Normalize Identity

**Audit issues addressed: #6, #10, #11, #13, #15, #17, #24, #31**

Identity = Name + Schema + Trust. Names should be consistent and
self-describing. Schemas should be discoverable. Trust should propagate.

### 2.1 Normalize `stop_reason` to a canonical vocabulary

Define a framework-level `StopReason` type (the Transform's output Identity):

```python
type StopReason = Literal[
    "stop",          # natural completion
    "max_tokens",    # hit token limit
    "tool_use",      # model wants to call tools
    "safety",        # blocked by safety filter
    "incomplete",    # partial/interrupted
]
```

Each provider maps its raw values to this canonical set. Code that checks
`response.stop_reason` works identically across providers.

**Files:**
- `llm/_types.py` — add `StopReason` literal type, change `Response.stop_reason: StopReason`
- Each provider — add mapping function `_normalize_stop_reason(raw: str) -> StopReason`

### 2.2 Separate vendor from API surface in provider names

The provider string should identify the vendor. The framework auto-detects
or the user explicitly selects the API surface:

```python
# Before — user must know internal API surface
Client("xai-responses", model="grok-3")

# After — framework picks the right surface, or user overrides
Client("xai", model="grok-3")
Client("xai", model="grok-3", api_surface="responses")
```

Internally, the factory resolves the best API surface for the given model.
The `-responses` suffix disappears from the public API.

**Files:**
- `llm/_providers/__init__.py` — update `create_provider()` factory
- Examples — simplify provider strings

### 2.3 Generate Gemini tool call IDs

When the API doesn't provide IDs, generate deterministic UUIDs:

```python
tool_call_id = f"call_{uuid4().hex[:24]}"
```

This gives every `ToolCall` a unique Identity, making tool result matching
work consistently across providers.

**Files:**
- `llm/_providers/_gemini.py` — generate IDs

### 2.4 Type the agent's `client` and `tools` dependencies

Define protocols that describe the Transform schemas agents depend on:

```python
class CompletionProvider(Protocol):
    def complete(
        self,
        prompt_or_messages: str | Sequence[Message | ToolResult],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> Response: ...

class ToolExecutor(Protocol):
    def execute(self, tool_call: ToolCall) -> str: ...
    async def async_execute(self, tool_call: ToolCall) -> str: ...
    @property
    def definitions(self) -> list[Tool]: ...
```

```python
class BaseAgent(ABC):
    def __init__(
        self,
        client: CompletionProvider,
        tools: ToolExecutor,
        *,
        config: AgentConfig | None = None,
    ) -> None: ...
```

This makes the Identity (schema) of agent dependencies explicit. Type checkers
catch mismatches at development time, not runtime.

### 2.5 Export `ConversationItem` in public API

Add to `llm/__init__.py` and top-level `__init__.py`. It appears in public
method signatures — its Identity should be discoverable.

### 2.6 Normalize content-type error semantics

All providers should **raise** on unsupported content types, never silently
skip. This makes the Transform's contract (Identity) honest:

```python
raise UnsupportedContentError(
    f"{provider_name} does not support {type(part).__name__}"
)
```

**Files:**
- `llm/_providers/_openai_responses.py` — change silent skip to raise
- `llm/_types.py` — add `UnsupportedContentError`

---

## Phase 3: Consistent Async Convention

**Audit issues addressed: #7, #12, #17, #18**

Pick one pattern and apply everywhere. The `a`-prefix pattern is the most
Pythonic (matches `aiohttp`, `asyncpg`, `httpx`) and the most concise.

### 3.1 Standardize on `a`-prefix for all async methods

| Layer | Sync | Async |
|-------|------|-------|
| Client | `complete()` | `acomplete()` |
| Provider | `complete()` | `acomplete()` |
| Agent | `run()` | `arun()` |
| ToolRegistry | `execute()` | `aexecute()` |
| Middleware | `before()` / `after()` | `abefore()` / `aafter()` |
| FallbackClient | `complete()` | `acomplete()` |

### 3.2 Merge `Client` and `AsyncClient` into one class

The client becomes a single class with both sync and async methods. This is
how `httpx.Client` vs `httpx.AsyncClient` works, but since our Client is
lightweight (no connection pool), a single class is simpler:

```python
class Client:
    def complete(self, ...) -> Response: ...
    async def acomplete(self, ...) -> Response: ...
    def stream_text(self, ...) -> Iterator[str]: ...
    async def astream_text(self, ...) -> AsyncIterator[str]: ...
```

Keep `AsyncClient` as a deprecated alias for one release cycle.

### 3.3 Fix the Middleware Protocol

Make the Protocol honest by making async methods optional via
`runtime_checkable` and separate protocols:

```python
class SyncMiddleware(Protocol):
    def before(self, request: Request) -> Request: ...
    def after(self, request: Request, result: Any) -> Any: ...

class AsyncMiddleware(Protocol):
    async def abefore(self, request: Request) -> Request: ...
    async def aafter(self, request: Request, result: Any) -> Any: ...

type Middleware = SyncMiddleware | AsyncMiddleware
```

The client checks which protocol the middleware satisfies at registration
time, not via `hasattr` at call time. Sync-only middleware works in async
context by wrapping sync methods in `asyncio.to_thread`.

### 3.4 FallbackClient protocol detection resolves naturally

With consistent naming (`acomplete` everywhere), the `getattr` sniffing in
`FallbackClient` disappears. One name, one lookup.

---

## Phase 4: Fix Agent Transforms

**Audit issues addressed: #8, #13, #28, #29, #30, #36**

Agents are Transforms: content in → content out. Their internal loop is an
implementation detail. The public API should reflect the Transform shape.

### 4.1 Remove `stream=True` from `run()` — fix the type lie

`run()` returns `AgentResult`. `run_stream()` returns `Iterator[AgentStep]`.
These are different return types — don't pretend they're the same method:

```python
class BaseAgent(ABC):
    @abstractmethod
    def run(self, task: str, **kwargs: Any) -> AgentResult: ...

    def run_stream(self, task: str, **kwargs: Any) -> Iterator[AgentStep]:
        """Default: run to completion, then yield steps."""
        result = self.run(task, **kwargs)
        yield from result.steps

    async def arun(self, task: str, **kwargs: Any) -> AgentResult: ...

    async def arun_stream(self, task: str, **kwargs: Any) -> AsyncIterator[AgentStep]:
        result = await self.arun(task, **kwargs)
        for step in result.steps:
            yield step
```

Remove the `stream = kwargs.pop("stream", False)` pattern from every agent.

### 4.2 Remove dead cancellation code

Delete `_resolve_cancellation_token`, `_is_cancelled`, and all
`cancellation_token` pops from agents. This is ~50 lines of dead code per
agent that lies about the system's capabilities. When cancellation is
implemented for real, add it properly with `asyncio.CancelledError` or a
typed `CancellationToken`.

### 4.3 Propagate `on_event` to inner agents

Inner agents created by `PlanExecuteAgent`, `ReflexionAgent`, and
`LATSAgent` must inherit the outer agent's `on_event` callback. Prefix
inner events with the outer agent's context:

```python
inner_config = AgentConfig(
    max_iterations=self.config.max_iterations,  # inherit, don't hardcode
    system=step_system,
    max_tokens=self.config.max_tokens,
    on_event=self.config.on_event,  # propagate
)
```

### 4.4 Move hidden agent kwargs to `AgentConfig` or `__init__`

Agent-specific configuration belongs in typed, discoverable parameters:

```python
@dataclass(frozen=True, slots=True)
class ToTConfig(AgentConfig):
    max_depth: int = 3
    branching_factor: int = 3
    beam_width: int = 2
    search_strategy: Literal["bfs", "dfs"] = "bfs"

@dataclass(frozen=True, slots=True)
class LATSConfig(AgentConfig):
    exploration_weight: float = 1.41
    num_expansions: int = 2

@dataclass(frozen=True, slots=True)
class ReflexionConfig(AgentConfig):
    threshold: float = 0.8
```

Agents that take evaluators accept them in `__init__`:

```python
class ReflexionAgent(BaseAgent):
    def __init__(
        self,
        client: CompletionProvider,
        tools: ToolExecutor,
        *,
        config: ReflexionConfig | None = None,
        evaluator: Callable[[str], float] | None = None,
    ) -> None: ...
```

### 4.5 Normalize inner agent `max_iterations`

All inner agents inherit `max_iterations` from the outer config (like
Reflexion does), not hardcoded `3`. If a user sets `max_iterations=20`, inner
execution should have proportional capacity.

---

## Phase 5: Middleware as Transform

**Audit issues addressed: #9, #14, #19, #21, #22, #34**

Middleware is a Transform wrapper: content flows through `before` → provider
Transform → `after`. It should work identically regardless of delivery mode.

### 5.1 Constrain `Request.operation` to a Literal type

```python
type Operation = Literal["complete", "complete_stream"]

@dataclass(slots=True)
class Request:
    operation: Operation
    provider: str
    model: str
    messages: list[ConversationItem]
    system: str | None = None
    tools: list[Tool] | None = None
    json_schema: JsonSchema | None = None
    thinking: ThinkingConfig | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
```

With the unified `complete()` method, `Request` always carries the full
parameter set. No more inconsistent metadata (#22).

### 5.2 Fix guardrail stream checking — accumulate content

Replace per-chunk checking with accumulated checking:

```python
def _wrap_stream(self, stream: Iterator[StreamEvent]) -> Iterator[StreamEvent]:
    accumulated = []
    for event in stream:
        if event.text:
            accumulated.append(event.text)
            full_text = "".join(accumulated)
            self._check_text(full_text, stage="output")
        yield event
```

This catches patterns split across chunks (`"pass"` + `"word"` →
`"password"` detected).

### 5.3 `ConversationMemory` — make the mutability intentional and documented

`ConversationMemory` is Memory with `access_control=append_only` in
first-principles terms. It *should* be mutable. But mark it clearly:

```python
@dataclass(slots=True)  # Mutable by design — this is append-only Memory
class ConversationMemory:
    ...
```

No code change needed, just document that Memory types are intentionally not
frozen. They are the one primitive that holds state.

---

## Phase 6: Provider Parity (Transform Consistency)

**Audit issues addressed: #23, #25, #27, #32, #33**

All providers are implementations of the same Transform. Their observable
behavior should be consistent.

### 6.1 Gemini: defensive tool argument parsing

Add the same try/catch pattern used by all other providers:

```python
try:
    arguments = fc.get("args", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
except (json.JSONDecodeError, TypeError):
    arguments = {"_raw": fc.get("args", "")}
```

### 6.2 Gemini: don't silently drop unknown kwargs — warn

```python
known_keys = {"temperature", "max_output_tokens", "top_p", ...}
unknown = set(kwargs) - known_keys
if unknown:
    logger.warning("Gemini provider ignoring unknown kwargs: %s", unknown)
```

### 6.3 Normalize streaming `Usage` shape

All providers should emit `Usage` with all fields populated. If a provider
doesn't report `input_tokens` in streaming (Anthropic), estimate from the
request or leave as 0 but document it.

Add to `Usage`:

```python
@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    estimated: bool = False  # True if any field was estimated/unavailable
```

### 6.4 Thinking blocks: document provider support, warn on no-ops

Don't silently ignore `thinking=ThinkingConfig(...)` on providers that don't
support it. Log a warning:

```python
if thinking and not self._supports_thinking:
    logger.warning(
        "%s does not support thinking/reasoning — parameter ignored",
        self._provider_name,
    )
```

### 6.5 Normalize unsupported content error behavior

All providers raise `UnsupportedContentError` on unsupported `ContentPart`
types. No silent skipping.

---

## Phase 7: Tiered Exports (Identity Hierarchy)

**Audit issues addressed: #20, #15**

The top-level `__init__.py` exports ~100 symbols with no hierarchy. Users
can't tell which 5 symbols they need. Apply Identity's "schema"
dimension — organize by what users need at each level of sophistication.

### 7.1 Tier the public API

**Tier 1 — Core** (top-level `__init__.py`, ~15 symbols):
```python
# The essentials
Client, Response, Message, Tool, ToolCall, ToolResult, Usage
ToolRegistry, tool
ReActAgent, AgentResult, AgentConfig, AgentStep
ConversationItem
```

**Tier 2 — Extended** (subpackage imports, explicit paths):
```python
from ai_arch_toolkit.llm import (
    AsyncClient, FallbackClient, BatchClient, AsyncBatchClient,
    StreamEvent, ThinkingConfig, ThinkingBlock, JsonSchema,
    ServerTool, RetryConfig,
    ImagePart, AudioPart, DocumentPart, TextPart,
)
from ai_arch_toolkit.llm.middleware import (
    Middleware, Request, CostTracker, ResponseCache,
    GuardrailMiddleware, TracingMiddleware,
)
from ai_arch_toolkit.llm.tokens import (
    estimate_text_tokens, estimate_conversation_tokens, ...
)
from ai_arch_toolkit.llm.cost import (
    CostSnapshot, CostPreview, ModelPricing, estimate_usage_cost, ...
)
from ai_arch_toolkit.agents import (
    PlanExecuteAgent, ReflexionAgent, LATSAgent,
    TreeOfThoughtsAgent, SelfDiscoveryAgent, ...
)
```

Users who `from ai_arch_toolkit import *` get a focused, usable set. Power
users import from subpackages. Internal implementation details (correction
factors, cache backends, etc.) live only in subpackage exports.

---

## Phase 8: Batch as Transform Extension

**Audit issues addressed: #16**

`BatchClient` should share middleware and provider resolution with `Client`.
A batch is a Parallel composition of Transforms — it should inherit the same
Identity and Transform pipeline.

### 8.1 Make `BatchClient` a method on `Client`

```python
class Client:
    def batch(self, requests: list[BatchRequest]) -> BatchJob:
        """Submit a batch of completions (Parallel Transform)."""
        # Uses the same provider, middleware, retry config
        ...
```

Or at minimum, `BatchClient` accepts a `Client` instance:

```python
batch = BatchClient(client)  # inherits middleware, provider, config
job = batch.submit(requests)
```

---

## Phase 9: FallbackClient Streaming Contract

**Audit issues addressed: #26**

Partial stream emission is a fundamental constraint — you can't un-yield
data. This is correct behavior but needs to be part of the Transform's
Identity (documented contract):

```python
class FallbackClient:
    """Tries clients in order until one succeeds.

    Streaming contract: fallback only occurs before the first chunk is
    emitted. Once streaming has begun, mid-stream failures propagate
    to the caller. This is inherent to streaming Transforms — content
    already delivered cannot be retracted.
    """
```

No code change — just make the boundary explicit in docs and docstrings.

---

## Phase 10: Tool Decorator Honesty

**Audit issues addressed: #35**

Silent error suppression violates Transform honesty. If the `@tool` decorator
can't resolve type hints, it should warn, not silently degrade:

```python
try:
    hints = get_type_hints(fn)
except Exception as exc:
    logger.warning(
        "Could not resolve type hints for @tool '%s': %s. "
        "All parameters will default to string type.",
        fn.__name__,
        exc,
    )
    hints = {}
```

---

## Migration Strategy

### Breaking changes (major version bump)

1. `chat()` → `complete()` (keep deprecated alias for 1 release)
2. `stream()` / `stream_events()` → `complete(stream=True)` + `stream_text()`
3. `async_run()` → `arun()`
4. `async_execute()` → `aexecute()`
5. `AsyncClient` merged into `Client` (keep deprecated alias)
6. Agent kwargs → typed config dataclasses
7. Top-level exports reduced to Tier 1

### Non-breaking changes (can ship incrementally)

1. `stop_reason` normalization (additive mapping)
2. Gemini tool call ID generation
3. Gemini defensive parsing
4. Guardrail accumulation fix
5. `on_event` propagation to inner agents
6. Dead cancellation code removal
7. `ConversationItem` export
8. Content-type error normalization
9. Streaming `Usage` normalization
10. `@tool` decorator warning
11. Logging for unsupported kwargs/features
12. `Request.operation` as Literal type

### Recommended order

1. **Non-breaking fixes first** — ship as patch/minor releases
2. **Unified `complete()` + async convention** — ship together as major release
3. **Agent refactoring** — can ship in same or next major
4. **Export tiering** — ship last (most visible to users)

---

## Summary: First Principles Mapping

| Primitive | Current Violation | Refactored State |
|-----------|-------------------|------------------|
| **Content** | `stream()` drops metadata; Usage shape varies | All delivery modes return full Content; Usage normalized |
| **Transform** | 3 methods for 1 operation; middleware partially covers | 1 method with delivery mode flag; middleware covers all |
| **Identity** | Inconsistent names; `stop_reason` varies; hidden kwargs; `Any` types | Consistent `a`-prefix async; canonical StopReason; typed params; protocols |
| **Memory** | ConversationMemory mutability undocumented; inner agents lose on_event | Mutability documented as Memory semantics; events propagate |
| **Boundary** | Guardrails check chunks not accumulated text; content errors inconsistent | Accumulated checking; uniform error semantics; honest contracts |
| **Composition** | Agent streaming type lies; batch disconnected from client | Honest return types; batch as Parallel composition of same Transform |
