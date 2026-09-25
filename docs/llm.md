# LLM Facade

## Reusable system prompts

The LLM core accepts model-visible strings and does not depend on toolkit prompt objects:

```python
from ai_arch_toolkit import LLM, load_prompt

template = load_prompt("prompts/reviewer.prompt.yaml")
rendered = template.render(language="Python")

llm = LLM("gpt-4.1-mini")
response = llm.complete_sync("Review this change.", system=rendered.text)
```

Prompt `layout` controls the input text. `output_schema`/`json_mode` control the model
response; these are independent settings.

The `LLM` class is the single interface to all providers. The model prefix auto-routes to the right adapter:

```python
from ai_arch_toolkit import LLM

llm = LLM("claude-sonnet-5")  # → Anthropic
llm = LLM("gpt-4o")                    # → OpenAI
llm = LLM("gemini-3.7-flash")          # → Gemini
llm = LLM("grok-4.3")                  # → xAI
llm = LLM("muse-spark-1.3")            # → Meta
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

# Thinking (each model's rules: see "Extended thinking" below)
response = await llm.complete(messages, thinking=True, thinking_effort="high")
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

`system=` never replaces the `system()` messages in `messages`, or the other way round: Anthropic, Gemini and xAI receive one system prompt with `system=` first and the `system()` messages after it, separated by a blank line, while the OpenAI adapter (and OpenAI-compatible servers) sends `system=` as a leading system message and keeps each `system()` message at its position — batch requests included. The Meta adapter behaves like the OpenAI one, with `system=` sent as the Responses API's `instructions`.

A system message's content is text: a string, or a list of strings and `cache()` parts, joined by a blank line. The Anthropic adapter keeps a `cache()` part's cache marker, sending the system prompt as text blocks, so a long system prompt can be cached; the other adapters send its text. An image or a document in a system message raises `TypeError` — send it in a user message.

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

## Provider errors

All normalized failures inherit `ProviderError` and carry a typed `delivery` disposition:
`"not_sent"`, `"unbilled"`, or `"indeterminate"`. Retry decisions use the error type and HTTP status;
billing uses delivery. The base constructor requires `delivery=`.

| Error | Existing handler remains valid | Detail |
|---|---|---|
| `RequestError` | `ValueError` | Refused before anything was sent (`not_sent`): common validation, a model rule of the adapter, or the SDK's own checks |
| `UnpricedModelError` | `RequestError` | A metered call to a model without a price ([Pricing](pricing.md#unpriced-models-under-a-meter)) |
| `APIError` | `APIError` | Retains `status_code` and `body`; `unbilled` or `indeterminate` by provider (below) |
| `RateLimitError` | `APIError` | Retains `retry_after`; `unbilled` on every provider |
| `TransportError` | `ConnectionError` | No usable HTTP response; `not_sent` when connecting failed, else `indeterminate` |
| `ProviderTimeout` | `TimeoutError` | Provider I/O timeout; `not_sent` when connecting timed out, else `indeterminate` |
| `ResponseError` | `ProviderError` | An unusable successful response or stream, or a failure reported without a status (`indeterminate`) |

The default fallback family is `(ProviderError,)`. `RequestError` remains terminal: retrying the
same invalid request is not a recovery. Caller cancellation propagates. After the first stream
item is visible, both retry and fallback are disabled. `complete` delivers only its final result.
See [failed-call cost accounting](safety.md#failed-llm-calls) for the cost table and bounded uncertainty.

Each adapter builds its request before the call is admitted, so a request it refuses (an option
the model does not take, for example) raises `RequestError` without opening a metered operation.
No SDK, HTTP-library, or gRPC exception leaves an adapter: each maps to the types above, and the
adapter knows whether the request reached its transport. `delivery` follows each provider's
documented billing:

| Provider | Error response | Transport failure after sending |
|---|---|---|
| Anthropic | `unbilled` ("failed requests aren't charged") | `indeterminate` (a client timeout is billed) |
| Gemini | `unbilled` for 400 and 500, else `indeterminate` | `indeterminate` |
| OpenAI, xAI, Meta | `indeterminate` (billing of errors is not documented) | `indeterminate` |

A 429 is `unbilled` everywhere, and an error that arrives inside a stream is `indeterminate`.
When a failed response reports its usage (Meta's `response.failed`), the error carries it as
`ProviderError.usage`; `Response.attempts` records it and a meter settles the failure at that
usage's price.

---

## Fallback chains

```python
from ai_arch_toolkit import LLM, ProviderError

llm = LLM(
    "claude-opus-5",
    fallback=["claude-sonnet-5", "gpt-4o"],
    fallback_on=(ProviderError,),  # default; includes HTTP and transport errors
)

# If Opus fails → tries Sonnet → tries GPT-4o
# response.attempts records what happened at each step
```

A string fallback routes by its own model name (a recognizable model fails over to its own provider; a bare tag inherits the parent's connection). `Response.attempts` includes every physical attempt, including failed intermediate fallback models and their retries. A candidate reachable through more than one chain is tried once. Async middleware surrounds the whole chain. `AdmissionDenied` is terminal even with `fallback_on=(Exception,)`.

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
        retry_on_status=(429, 500, 502, 503, 504, 529),
    ),
)
```

The adapters disable retry loops built into the provider SDKs. `RetryConfig`
is therefore the single retry owner: every attempt is metered and appears in
`Response.attempts`. `max_retries=N` means at most `N + 1` physical attempts
for that `LLM`. Retries are opt-in; omitting `retry=` performs one attempt.
Besides the statuses in `retry_on_status` and rate limits, a request that got no
HTTP response — a refused or dropped connection, a timeout — is retried: the
adapters raise `TransportError` or `ProviderTimeout`, which also trigger fallbacks and remain
catchable as `ConnectionError` or `TimeoutError`.
Fallbacks supplied as `LLM` objects use their own retry configuration, so pass
configured instances when fallback models should retry too.

For streaming, provider I/O starts when iteration begins. A retry or fallback is
safe only before the first chunk/event becomes visible to the caller; after that
boundary an error is surfaced without replay, avoiding duplicated or spliced
output. Budget admission and the call reservation still happen when
`stream()` / `stream_events()` creates the stream object; the call counts once
the first provider attempt starts. If async middleware changes the request's
metering facts (tools, `max_tokens`, injected content), the reservation is
replaced before that attempt, so admission and pricing see the request as sent;
a budget that denies the rewritten request raises `AdmissionDenied` without
calling the provider. A stream that is never iterated, or that middleware
rejects before the provider is called, releases its reservation and records no
attempt.

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

Models can reason through a problem before answering:

```python
response = await llm.complete(
    "Solve this step by step: what is 127 * 389?",
    thinking=True,
    thinking_effort="high",
)

for block in response.thinking:
    print(f"[Thinking] {block.text}")

print(f"Answer: {response.text}")
```

Three options drive it: `thinking=True` asks for thinking (and, where the provider hides it, for
a summary to show in `Response.thinking`), `thinking_effort` sets how hard the model thinks, and
`thinking_budget` caps thinking tokens on the models that take a budget. Each adapter applies
them by the model's documented rules and raises `RequestError` before sending when the model
does not take a value; a model the adapter does not know gets the rules of the provider's
current generation.

| Provider and models | `thinking=True` | `thinking_effort` | `thinking_budget` |
|---|---|---|---|
| Anthropic: Claude 5 family, Mythos Preview, Opus 4.8, 4.7, 4.6, Sonnet 4.6 | adaptive thinking, summarized | `low` to `max` (no `xhigh` on Mythos Preview and the 4.6 models), applies alone | ignored, with a warning |
| Anthropic: Opus, Sonnet, Haiku 4.5 and older | a budget of 10,000 tokens | a budget (2,048, 5,000 or 10,000), turns thinking on | at least 1,024; turns thinking on |
| OpenAI | sends `reasoning_effort` (`high` unless an effort is given); `RequestError` on a model that does not reason | only with `thinking=True`, checked per model | ignored, with a warning |
| Gemini 3 | thought summaries, and level `high` when no effort is given | the model's thinking levels, applies alone | ignored, with a warning |
| Gemini 2.5 | thought summaries, and a budget of 10,000 when no effort is given | a budget (2,048, 5,000 or 10,000) | within the model's documented range |
| xAI | nothing more; `RequestError` on a model that does not reason | `reasoning_effort` where the model documents one, applies alone | ignored, with a warning |
| Meta (Muse Spark) | reasoning summaries | `minimal` to `xhigh`, and `max` on standard `muse-spark-1.3`; applies alone | ignored, with a warning |

From GPT-5.4 on, OpenAI's Chat Completions takes tool calls only at the `"none"` effort: with
tools, `thinking=True` raises `RequestError` unless `thinking_effort="none"` (reasoning with tools
needs the Responses API). GPT-6 Sol and Luna reason at `medium` when no effort is sent, so a tool
call that asks for no thinking is sent at `"none"`; GPT-6 Astra takes no `"none"` and calls no
tools here.

On the Anthropic models that take a budget, the budget is added to `max_tokens`, so the answer
keeps its room; elsewhere reasoning tokens count toward `max_tokens`, so keep that budget
generous. Meta's raw reasoning stays encrypted: append `response.to_message()` to the
conversation and the next request replays it, so a tool loop keeps its chain of thought (the
Anthropic and Gemini adapters replay their providers' thinking signatures the same way). See
[Model Compatibility](model-compatibility.md) for each provider's models and limits.

---

See also: [Tools](tools.md) · [Middleware](middleware.md) · [Pricing & Cost Tracking](pricing.md) · [Flow Architecture](flow-architecture.md) for using an `LLM` inside agent flows.
