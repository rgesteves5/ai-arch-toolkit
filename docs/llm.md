# LLM Facade

The `LLM` class is the single interface to all providers. The model prefix auto-routes to the right adapter:

```python
from ai_arch_toolkit import LLM

llm = LLM("claude-sonnet-4-20250514")  # → Anthropic
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
async for chunk in await llm.stream(messages):
    print(chunk, end="")

# Streaming (structured events)
async for event in await llm.stream_events(messages, tools=tools):
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
response.cost           # estimated USD (from pricing registry)
response.stop_reason    # "end_turn", "tool_use", "max_tokens", etc.
response.model          # actual model used
response.citations      # tuple of Citation (web search results)
response.attempts       # tuple of Attempt (retry/fallback history)
response.has_tool_calls # bool shorthand
response.to_message()   # convert to assistant message dict
```

---

## Fallback chains

```python
llm = LLM(
    "claude-opus-4-0-20250514",
    fallback=["claude-sonnet-4-20250514", "gpt-4o"],
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
    "claude-sonnet-4-20250514",
    retry=RetryConfig(
        max_retries=3,
        base_delay=1.0,      # exponential backoff
        max_delay=60.0,
        retry_on_status=(429, 500, 502, 503, 504),
    ),
)
```

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

llm = LLM("claude-sonnet-4-20250514")

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
