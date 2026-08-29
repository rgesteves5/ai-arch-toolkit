# LLM Facade

## Reusable system prompts

The LLM core accepts model-visible strings and does not depend on toolkit prompt objects:

```python
from ai_arch_toolkit import LLM, load_prompt

template = load_prompt("prompts/reviewer.prompt.yaml")
rendered = template.render(language="Python")

llm = LLM("gpt-4.1-nano")
response = llm.complete_sync("Review this change.", system=rendered.text)
```

Prompt `layout` controls the input text. `output_schema`/`json_mode` control the model
response; these are independent settings.

The `LLM` class is the single interface to all providers. The model prefix auto-routes to the right adapter:

```python
from ai_arch_toolkit import LLM

llm = LLM("claude-sonnet-5")  # → Anthropic
llm = LLM("gpt-4o")                    # → OpenAI
llm = LLM("gemini-2.0-flash")          # → Gemini
llm = LLM("grok-2")                    # → xAI
```

See [Model Compatibility](model-compatibility.md) for per-provider support and [Framework Overview](framework-overview.md) for routing/`base_url` details (local OpenAI-compatible servers, forcing an adapter, etc.).

---

## Core methods

```python
# Simple completion
response = await llm.complete("What is 2+2?")
response = await llm.complete(messages, system="You are helpful.")

# With tools
response = await llm.complete(messages, tools=my_tool_group)

# Structured output (Pydantic model or OutputSchema)
response = await llm.complete(messages, output_schema=MyModel)
response.parsed  # → MyModel instance

# Extended thinking (Anthropic)
response = await llm.complete(messages, thinking=True, thinking_budget=5000)
response.thinking  # → tuple of ThinkingBlock

# JSON mode
response = await llm.complete(messages, json_mode=True)

# Streaming (text chunks)
async for chunk in llm.stream(messages):
    print(chunk, end="")

# Streaming (structured events)
async for event in llm.stream_events(messages, tools=tools):
    match event.kind:
        case "text": print(event.text, end="")
        case "thinking": print(f"[thinking] {event.thinking.text}")
        case "tool_call": print(f"[tool] {event.tool_call.name}")

# Sync versions
response = llm.complete_sync("Hello")
for chunk in llm.stream_sync("Hello"):
    print(chunk, end="")
for event in llm.stream_events_sync("Hello"):   # sync rich events (SyncRichStreamResponse)
    ...
```

The task/messages argument accepts `Content` — a string or a multimodal list. See [Content & Messages](content.md).

---

## Response

Every LLM call returns a `Response`:

```python
response.text           # answer text
response.tool_calls     # tuple of ToolCall(id, name, input)
response.thinking       # tuple of ThinkingBlock (extended thinking)
response.parsed         # structured output (if output_schema used)
response.usage          # Usage(input_tokens, output_tokens, cache_write_tokens, cache_read_tokens)
response.cost           # exact provider cost when reported, otherwise estimated USD
response.provider_cost  # exact provider-reported USD, or None
response.stop_reason    # "end_turn", "tool_use", "max_tokens", etc.
response.model          # actual model used
response.citations      # tuple of Citation (web search results)
response.attempts       # tuple of Attempt (retry/fallback history)
response.has_tool_calls # bool shorthand
response.to_message()   # convert to assistant message dict
```

The four `Usage` counters are disjoint. `input_tokens` contains non-cached input only;
add `cache_read_tokens` and `cache_write_tokens` to obtain total input. This keeps cache
reads/writes from being charged again at the regular input rate. `output_tokens` includes
billable reasoning/thinking tokens when a provider reports them separately.

---

## Fallback chains

```python
llm = LLM(
    "claude-opus-5",
    fallback=["claude-sonnet-5", "gpt-4o"],
    fallback_on=(APIError, TimeoutError),  # default
)

# If Opus fails → tries Sonnet → tries GPT-4o
# response.attempts records what happened at each step
```

A string fallback routes by its own model name (a recognizable model fails over to its own provider; a bare tag inherits the parent's connection).

---

## Retry

```python
from ai_arch_toolkit import RetryConfig

llm = LLM(
    "claude-sonnet-5",
    retry=RetryConfig(
        max_retries=3,
        base_delay=1.0,      # exponential backoff
        max_delay=60.0,
        retry_on_status=(429, 500, 502, 503, 504),
    ),
)
```

The adapters disable retry loops built into the provider SDKs. `RetryConfig`
is therefore the single retry owner: every attempt is metered and appears in
`Response.attempts`. `max_retries=N` means at most `N + 1` physical attempts
for that `LLM`. Retries are opt-in; omitting `retry=` performs one attempt.
Fallbacks supplied as `LLM` objects use their own retry configuration, so pass
configured instances when fallback models should retry too.

For streaming, provider I/O starts when iteration begins. A retry or fallback is
safe only before the first chunk/event becomes visible to the caller; after that
boundary an error is surfaced without replay, avoiding duplicated or spliced
output. Budget admission and the first call reservation still happen when
`stream()` / `stream_events()` creates the stream object.

Fully consuming a stream closes its provider iterator automatically. If the
consumer may stop early, use the async context manager (or call `await
stream.aclose()`) so provider resources are released immediately and the partial
response is recorded as abandoned:

```python
async with llm.stream(messages) as stream:
    async for chunk in stream:
        if enough(chunk):
            break
```

---

## Limiting concurrent inference

To cap how many `complete()` calls hit the model at once across a whole run
(protecting a local GPU or staying under a provider's concurrency limit), wrap
the run in `inference_limit(n)`:

```python
from ai_arch_toolkit import inference_limit

with inference_limit(2):            # ≤ 2 concurrent inferences, across all nested agents
    result = agent.run_sync(task)
```

It is a global, run-scoped, opt-in cap (default: unlimited). See
[Concurrency & Throttling](concurrency.md) for the full model and how it differs
from `Flow(max_parallelism=...)`. Streaming calls are deliberately not throttled
because their lifetime spans caller-controlled yields.

---

## Token counting

Provider-accurate counts (may call the provider's token-counting endpoint):

```python
token_count = await llm.count_tokens(messages, system="...", tools=tools)
# or sync:
token_count = llm.count_tokens_sync(messages)
```

For fast, offline estimates with no network call, use the local heuristics:

```python
from ai_arch_toolkit import (
    count_tokens_local, count_tokens_local_batch, chars_to_tokens, tokens_to_chars,
)

count_tokens_local("some text", model="gpt-4o")        # estimated tokens
count_tokens_local_batch(["a", "b", "c"])              # summed estimate
chars_to_tokens(4000)                                  # rough char→token
tokens_to_chars(1000)                                  # rough token→char
```

These are approximations (character-ratio based, with an optional `correction` factor) — use them for pre-flight budget checks, not billing.

---

## Batch API

For high-volume, non-interactive workloads, submit many requests as a single batch (cheaper and higher-throughput on providers that support it — **Anthropic and OpenAI**; other providers raise `NotImplementedError`).

```python
import dataclasses
from ai_arch_toolkit import LLM, BatchRequest, user

llm = LLM("claude-sonnet-5")

requests = [
    BatchRequest(messages=[user("Summarize the French Revolution.")], custom_id="job-1"),
    BatchRequest(
        messages=[user("Summarize the Industrial Revolution.")],
        custom_id="job-2",
        kwargs={"max_tokens": 1024},
    ),
]

# Submit (takes plain dicts — convert BatchRequest with dataclasses.asdict)
batch_id = llm.batch_submit_sync([dataclasses.asdict(r) for r in requests])

# Poll
status = llm.batch_status_sync(batch_id)   # provider status string, e.g. "in_progress" / "ended"

# Retrieve when finished — list[BatchResult]
for res in llm.batch_results_sync(batch_id):
    if res.response is not None:
        print(res.custom_id, "→", res.response.text)
    else:
        print(res.custom_id, "ERROR:", res.error)
```

- **`BatchRequest`** — `messages`, plus optional `system`, `tools`, `custom_id`, `kwargs` (e.g. `max_tokens`). `custom_id` ties a request to its result.
- **`BatchResult`** — `custom_id`, `response` (a `Response`, or `None` on failure), `error`.
- Async equivalents: `batch_submit()`, `batch_status()`, `batch_results()`.

---

## Structured output

Force the LLM to return data matching a schema:

```python
from pydantic import BaseModel
from ai_arch_toolkit import LLM, OutputSchema

class WeatherReport(BaseModel):
    city: str
    temperature: float
    conditions: str

# With a Pydantic model
response = await llm.complete("Weather in Paris", output_schema=WeatherReport)
report = response.parsed  # → WeatherReport(city="Paris", temperature=22.0, ...)

# With OutputSchema (manual JSON Schema)
schema = OutputSchema(
    name="weather",
    schema={"type": "object", "properties": {"city": {"type": "string"}}},
)
response = await llm.complete("Weather in Paris", output_schema=schema)
```

Anthropic uses native structured output (`output_config`); other providers use their JSON-schema response formats. `json_mode=True` is the looser "valid JSON, no schema" alternative.

---

## Extended thinking

Anthropic models can reason through a problem before answering:

```python
response = await llm.complete(
    "Solve this step by step: what is 127 * 389?",
    thinking=True,
    thinking_budget=5000,  # max thinking tokens
)

for block in response.thinking:
    print(f"[Thinking] {block.text}")

print(f"Answer: {response.text}")
```

---

See also: [Tools](tools.md) · [Middleware](middleware.md) · [Pricing & Cost Tracking](pricing.md) · [Flow Architecture](flow-architecture.md) for using an `LLM` inside agent flows.
