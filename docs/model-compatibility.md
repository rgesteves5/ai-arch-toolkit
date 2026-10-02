# Model Compatibility

This page lists the 35 model IDs tracked by the live probe inventory
(`scripts/model_probe_models.toml`) and what framework features were verified against them.

The baseline below comes from the manual live probe runner:

```bash
set -a; source .env; set +a
uv run python scripts/probe_models.py --suite full --timeout-seconds 120 --max-retries 1
```

Latest recorded full run, over the 28 models the inventory held then:

- Date: 2026-04-28
- Models: 28
- Probe scenarios: 147
- Passed: 144
- Failed: 3
- Report artifact: `scripts/output/model-probes/20260428T015941Z.md`

Latest OpenAI run, over the inventory's 13 OpenAI models, after OpenAI's own host moved to the
Responses API:

- Date: 2026-10-02
- Models: 13
- Probe scenarios: 77
- Passed: 77
- Failed: 0
- Report artifact: `scripts/output/model-probes/20261002T041947Z.md`

Of the seven models added since 2026-04-28, `muse-spark-1.3` (2026-09-13), `gpt-6-sol`,
`gpt-6-luna` (2026-09-25), `gpt-6-astra`, and `gpt-6.1-sol` (2026-10-02) passed runs of their
own. `grok-4.7` and `claude-opus-5-5` have no live result: the 2026-09-25 run stopped at their
accounts' credit limits.

Generated probe artifacts are local diagnostic files and are ignored by git.

The tables record each model's latest live run. The text above each table gives the adapter's
current rules, which follow each provider's documentation per model (OpenAI's efforts were also
measured live): an option a model does not take raises `RequestError` before anything is sent,
and a model the adapter does not know gets the rules of the provider's current generation.
`tests/integration/test_provider_hardening_live.py` holds the cheap owner-run checks of those
rules (`pytest -m live_api -k <provider>`).

## Status Legend

| Status | Meaning |
|---|---|
| Pass | Live probe passed through the framework. |
| Fail | Live probe reached the provider but the behavior did not satisfy the scenario. |
| Unsupported | Provider rejected the model or capability for the current API path. |
| Not probed | The scenario is not enabled for that model in the inventory. |
| Auto | The model/provider reasons automatically; no explicit framework thinking knob is sent. |

## Probe Scenarios

| Scenario | What It Verifies |
|---|---|
| Plain | `LLM.complete()` returns the requested text. |
| Tools | Client-side tool call round trip through `ToolGroup` and `run_tools()`. |
| Structured | `OutputSchema` native structured output produces `response.parsed`. |
| JSON mode | `json_mode=True` returns parseable JSON text. |
| Stream | `LLM.stream()` yields chunks and final response text. |
| Thinking | `thinking=True` is accepted and returns a coherent final response. Thinking blocks are observational unless a model config requires them. |

## OpenAI

The host decides the API. OpenAI's own host (no `base_url`, or a `base_url` on `api.openai.com`)
is driven through the Responses API, the only OpenAI API that takes tool calls while GPT-5.4 and
later models reason
([migration guide](https://developers.openai.com/api/docs/guides/migrate-to-responses)). Any
other `base_url` is an OpenAI-compatible server: it gets Chat Completions, with `max_tokens` and
none of the rules below.

On OpenAI's host:

- Requests are stateless (`store: false`). Each response carries its reasoning encrypted, and
  appending `response.to_message()` to the conversation replays it on the next call, to the same
  model family only ("Persisted reasoning can be reused only within the same model family",
  [reasoning guide](https://developers.openai.com/api/docs/guides/reasoning)). The adapter takes
  a generation as a family: `gpt-6` (Astra, Sol, Luna), `gpt-6.1`, `gpt-5.6`, and so on.
- `max_tokens` is sent as `max_output_tokens`. `thinking=True` sends `reasoning.effort`
  (`thinking_effort`, else `"high"`), checked against the model's efforts (below), with
  `summary: "auto"`: the summaries become `Response.thinking` (there is none at `"none"`).
  Without `thinking=True` no effort is sent, and the model runs at its default effort.
- Tool calls work at any effort. Function tools are sent with `strict: false`: left out, OpenAI
  rewrites the schema into strict mode and makes its optional parameters required (live,
  2026-10-02). `web_search()` without config runs as the hosted `web_search` tool (billed per
  call, so metering treats its cost as unknown); `code_execution()`, or a server tool with a
  config, raises `RequestError`.
- Structured output goes in `text.format`: `output_schema` (a strict schema is normalized to
  OpenAI's strict subset) or `json_mode`. `stop`, `seed`, `frequency_penalty`,
  `presence_penalty`, and a raw `response_format` raise `RequestError` before sending, since the
  Responses API has no place for them (live, `stop` and `seed` got a 400 and the two penalties a
  500 after about 90 seconds).
- `logprobs=True` asks for the output text's token logprobs, which `Response.logprobs` holds (a
  tuple of the SDK's `Logprob`).
- `count_tokens` uses `POST /v1/responses/input_tokens`. Batches are submitted to
  `/v1/responses` ([batch guide](https://developers.openai.com/api/docs/guides/batch)); a batch
  submitted to Chat Completions before still reads.
- A failure reported inside a response or a stream gets status 429 for `rate_limit_exceeded` and
  500 for `server_error` ([error codes](https://developers.openai.com/api/docs/guides/error-codes));
  any other code raises `ResponseError`.
- The models OpenAI shuts down by 2026-10-23 (`gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo`,
  `gpt-4.1-nano`, `o1`, `o1-pro`, `o3-mini`, `o4-mini`) have no price or rules of their own here
  ([deprecations](https://developers.openai.com/api/docs/deprecations)).

### Efforts per model

Each reasoning model's efforts, and the effort a request without one runs at, were measured live
on 2026-10-02: one request per effort and model, reading back the effort the response reports.
Some models take less than their pages list (GPT-5.5 refuses `max`). A model that reasons, at the
effort sent or by default, takes no `temperature` other than 1, so the adapter drops it (the `LLM`
always sends one); a GPT-6 model that reasons takes no sampling parameter or logprobs at all
("When reasoning effort is not `none`, remove `temperature`, `top_p`, and `top_logprobs`",
[latest-model guide](https://developers.openai.com/api/docs/guides/latest-model)).

| Models | Efforts | With no effort sent | Sampling while it reasons |
|---|---|---|---|
| `gpt-6-astra`, `gpt-6.1-sol` | `low`, `medium`, `high`, `xhigh`, `max` | `medium` (always reasons) | dropped, logprobs too |
| `gpt-6-sol`, `gpt-6-luna` | `none`, `low`, `medium`, `high`, `xhigh`, `max` | `medium` | dropped, logprobs too |
| `gpt-5.6`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna` | `none`, `low`, `medium`, `high`, `xhigh`, `max` | `medium` | `temperature` only at 1 |
| `gpt-5.5` | `none`, `low`, `medium`, `high`, `xhigh` | `medium` | `temperature` only at 1 |
| `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5.3-codex`, `gpt-5.2` | `none`, `low`, `medium`, `high`, `xhigh` | `none` | `temperature` only at 1 |
| `gpt-5.4-pro`, `gpt-5.2-pro` | `medium`, `high`, `xhigh` | `medium` (always reasons) | `temperature` only at 1 |
| `gpt-5.5-pro` | `medium`, `high`, `xhigh` | `high` (always reasons) | `temperature` only at 1 |
| `gpt-5-pro` | `high` | `high` (always reasons) | `temperature` only at 1 |
| `gpt-5.1` | `none`, `low`, `medium`, `high` | `none` | `temperature` only at 1 |
| `gpt-5`, `gpt-5-mini`, `gpt-5-nano` | `minimal`, `low`, `medium`, `high` | `medium` | `temperature` only at 1 |
| `o3` | `low`, `medium`, `high` | `medium` | `temperature` only at 1 |
| `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini` | none (`thinking` or an effort raises `RequestError`) | does not reason | — |
| A model not listed | every effort the SDK names | `medium` | `temperature` only at 1 |

`thinking_effort` applies on its own: to keep a model that reasons by default from reasoning,
pass `thinking_effort="none"` where it takes `"none"`.

### GPT-6 Astra (added 2026-09-04)

`gpt-6-astra` is registered with standard, cached, batch, long-context, and fast pricing. It
takes no `none` (nor `minimal`) effort, so it always reasons, at `medium` unless sent another
effort, and takes no sampling or logprob parameters. It calls tools at any effort; every probe
passed live on 2026-10-02.
See the [official Astra guide](https://developers.openai.com/api/docs/guides/latest-model)
and [model pricing](https://developers.openai.com/api/docs/models/gpt-6-astra).

### GPT-6.1 Sol (added 2026-10-01)

`gpt-6.1-sol` is registered with standard, cached, batch, long-context (above 272K input
tokens), and fast pricing: GPT-6 Sol's rates, with cached input at 5% of input instead of 10%.
Unlike `gpt-6-sol`, it takes no `none` (nor `minimal`) effort, so it follows Astra's rules: it
always reasons, at `medium` unless sent another effort, takes no sampling or logprob parameters,
and calls tools at any effort. Every probe passed live on 2026-10-02. See the
[model page](https://developers.openai.com/api/docs/models/gpt-6.1-sol).

### GPT-6 Sol and Luna (added 2026-09-25)

The GPT-6 family is Astra, Sol, and Luna; Terra is a GPT-5.6 tier (`gpt-5.6-terra`). `gpt-6-sol`
and `gpt-6-luna` are registered with standard, cached, batch, long-context (above 272K input
tokens), and fast pricing. They take efforts `none` to `max` (no `minimal`) and reason at
`medium` when no effort is sent. They take sampling parameters (`temperature`, `top_p`,
logprobs) only at `none`, so a request without an effort, which runs at `medium`, loses them:
pass `thinking_effort="none"` to sample. Tool calls go at any effort.

Both passed every probe live on 2026-09-25 through Chat Completions (report `20260925T013106Z`),
and again on 2026-10-02 through the Responses API. See the model pages for
[Sol](https://developers.openai.com/api/docs/models/gpt-6-sol) and
[Luna](https://developers.openai.com/api/docs/models/gpt-6-luna).

### Recorded live baseline

Every OpenAI model in the inventory passed every scenario on 2026-10-02, through the Responses
API (`scripts/probe_models.py --suite full`, report `20261002T041947Z`: 77 of 77), Astra and
GPT-6.1 Sol with tools for the first time. The OpenAI live checks
(`pytest -m live_api tests/integration -k openai`: system prompt merging, a plain Pydantic
`output_schema`, a parallel tool-call replay, complete and streamed, thinking on and off, and a
4xx under a cost cap) passed 6 of 6, in three runs.

| Model | Plain | Tools | Structured | JSON Mode | Stream | Thinking | Notes |
|---|---|---|---|---|---|---|---|
| `gpt-6-astra` | Pass | Pass | Pass | Pass | Pass | Pass | Uses `thinking_effort="low"` in probes. |
| `gpt-6.1-sol` | Pass | Pass | Pass | Pass | Pass | Pass | Uses `thinking_effort="low"` in probes. |
| `gpt-6-sol` | Pass | Pass | Pass | Pass | Pass | Pass | Uses `thinking_effort="low"` in probes. |
| `gpt-6-luna` | Pass | Pass | Pass | Pass | Pass | Pass | Uses `thinking_effort="low"` in probes. |
| `gpt-5.5` | Pass | Pass | Pass | Pass | Pass | Pass | Uses `thinking_effort="low"` in probes. |
| `gpt-5.4` | Pass | Pass | Pass | Pass | Pass | Pass | Uses `thinking_effort="low"` in probes. |
| `gpt-5.4-mini` | Pass | Pass | Pass | Pass | Pass | Pass | Uses `thinking_effort="low"` in probes. |
| `gpt-5.4-nano` | Pass | Pass | Pass | Pass | Pass | Pass | Uses `thinking_effort="low"` in probes. |
| `gpt-4.1` | Pass | Pass | Pass | Pass | Pass | Not probed | Does not reason: no thinking scenario. |
| `gpt-5-mini` | Pass | Pass | Pass | Pass | Pass | Pass | Needs a larger output budget than newer GPT-5 IDs in probes. |
| `gpt-5-nano` | Pass | Pass | Pass | Pass | Pass | Pass | Needs a larger output budget than newer GPT-5 IDs in probes. |
| `gpt-5` | Pass | Pass | Pass | Pass | Pass | Pass | Needs a larger output budget than newer GPT-5 IDs in probes. |
| `o3` | Pass | Pass | Pass | Pass | Pass | Pass | Exact `o3` routes to OpenAI. |

## xAI

Grok reasoning models reason on their own, so `thinking=True` sends nothing more (it raises
`RequestError` on a model that does not reason, such as `grok-4.20-non-reasoning`).
`thinking_effort` applies without it and is sent as `reasoning_effort` where the model documents
one: `low` to `xhigh` on `grok-4.7`, `grok-4.6`, `grok-4.5` (which serves `xhigh` as `high`), and
newer models; `none` to `xhigh` on `grok-4.3` and the reasoning ids retired on 2026-05-15 that xAI
now serves with it (`grok-4-1-fast-reasoning`, for example). `grok-4.20-reasoning` and
`grok-build-0.1` take no effort (`RequestError`), nor do the non-reasoning models. The reasoning
models refuse `stop`, `presence_penalty`, and `frequency_penalty`, so those raise `RequestError`
too. `tool_choice` takes a tool's name. A server tool raises `RequestError` (the adapter does not
send them), and images and documents are dropped with a warning.

`grok-4.20-multi-agent` takes no client-side tools (`RequestError`) and no `max_tokens` (the
adapter leaves it out); `thinking_effort` picks the number of agents (4 for `low` and `medium`,
16 for `high` and `xhigh`; `thinking=True` alone picks 4), and `agent_count=` sets it directly.
The probes ran it with `agent_count=4`, on `plain` and `stream` only.

`grok-4.7` (added 2026-09-25) costs $2 input and $6 output per million tokens, and $4 and $12 for
the whole request once the prompt reaches 200k tokens; it has no Batch API. It follows the
current generation's rules above. Grok 4.7 Fast is not on the API. Its probe is in the inventory
but has **no live result**: the 2026-09-25 run stopped at the account's credit limit
([model page](https://docs.x.ai/developers/models/grok-4.7)).

| Model | Plain | Tools | Structured | JSON Mode | Stream | Thinking | Notes |
|---|---|---|---|---|---|---|---|
| `grok-4.20-reasoning` | Pass | Pass | Pass | Pass | Pass | Auto | `thinking_effort` raises `RequestError`. |
| `grok-4.20-non-reasoning` | Pass | Pass | Pass | Pass | Pass | Not probed | Standard non-reasoning configuration. |
| `grok-4.20-multi-agent` | Pass | Not probed | Not probed | Not probed | Pass | `agent_count=4` | Custom client tools are not enabled for this model. |
| `grok-4-1-fast-reasoning` | Pass | Pass | Pass | Pass | Pass | Auto | Retired 2026-05-15; served by `grok-4.3`, which takes `thinking_effort`. |
| `grok-4-1-fast-non-reasoning` | Pass | Pass | Pass | Pass | Pass | Not probed | Retired 2026-05-15; served by `grok-4.3` at effort `none`, and takes no `thinking_effort` here. |

## Gemini

Gemini probes use the current Gemini generate-content provider path. Live API-only models
need separate provider support.

> **Known issue (2026-09): Gemini and tool calls — fixed in code, not confirmed live.** The
> "Tools" column covers a single tool call. The adapter used to send the results of *parallel*
> tool calls back as separate `user` turns without the call `id`, and a history replayed after
> `stream()` / `stream_events()` could drop function calls and thought signatures, so multi-tool
> agent loops could fail with HTTP 400. It now sends a turn's results in one `user` content, with
> each call's `id` when Gemini gave one, and replays the model's turn as Gemini sent it. No live
> run has confirmed it
> (`pytest tests/integration/test_provider_hardening_live.py -m live_api -k gemini`), so prefer
> another provider for multi-tool agents.

Thinking follows each model's documented controls. On Gemini 3, `thinking_effort` is the
thinking level and applies without `thinking=True`: `low`, `medium`, and `high` on the 3.8 and
3.7 Flash, 3.1 Pro, and newer models; also `minimal` on the 3.6 and 3.5 Flash, 3.5 and 3.1
Flash-Lite, and 3 Flash; `low` and `high` on 3 Pro. `thinking=True` asks for thought summaries
and, without an effort, thinks at `high`; a `thinking_budget` is ignored with a warning. On Gemini
2.5, the effort becomes a thinking budget (2,048, 5,000, or 10,000 tokens), and a
`thinking_budget` must be in the model's documented range: 128 to 32,768 on 2.5 Pro, 0 to 24,576
on 2.5 Flash, and 0 or 512 to 24,576 on 2.5 Flash-Lite (Gemini's dynamic `-1` raises
`RequestError`: the `LLM` takes no negative budget). The `web_search` and `code_execution`
server tools are sent as `google_search` and `code_execution`; a server tool with a config raises
`RequestError`. The SDK always sends through `httpx`, so it never re-sends a request on its own.

| Model | Plain | Tools | Structured | JSON Mode | Stream | Thinking | Notes |
|---|---|---|---|---|---|---|---|
| `gemini-3.1-pro-preview` | Pass | Pass | Pass | Pass | Pass | Pass | Some calls can be slow; one plain probe took about 127 seconds. |
| `gemini-3.1-flash-lite-preview` | Pass | Pass | Pass | Pass | Pass | Pass | Marked transient-tolerant in the inventory. |
| `gemini-3.1-flash-live-preview` | Unsupported | Not probed | Not probed | Not probed | Unsupported | Not probed | Rejected for `generateContent`; likely needs Gemini Live API support. |
| `gemini-3-flash-preview` | Pass | Pass | Pass | Pass | Pass | Pass | Requires a larger output budget than the smoke default. |
| `gemini-2.5-pro` | Pass | Pass | Pass | Pass | Pass | Pass | Thinking probe uses explicit `thinking_budget`. |
| `gemini-2.5-flash` | Pass | Pass | Pass | Pass | Pass | Pass | Thinking probe uses explicit `thinking_budget`. |
| `gemini-2.5-flash-lite` | Pass | Pass | Pass | Pass | Pass | Pass | Requires `thinking_budget >= 512`. |

## Anthropic

### Claude Opus 5.5 (added 2026-09-25)

`claude-opus-5-5` is registered with standard, cached, batch, and fast pricing ($4 input and $20
output per million tokens; cache reads cost $0.20, 5% of input). It always thinks: `thinking=False`
sends nothing and the model thinks anyway, and its thinking counts toward `max_tokens`, so leave
room for it. Its effort defaults to `medium`, one level below Opus 5: pass `thinking_effort`
(`low` to `max`) to choose another. It takes no sampling parameters and refuses a forced
`tool_choice` with `RequestError`. Its probes are in the inventory but have **no live result**:
the 2026-09-25 run stopped when the account ran out of credit.
See the [migration guide](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide).

### Recorded live baseline

Anthropic models use top-level `system`, `input_schema` for tools, and native
`output_config` for structured output (`structured_output_mode="prompt"` asks for the JSON in the
system prompt instead). `max_tokens` is required (the `LLM` always sends one).

Thinking follows each model's documented mode. The models that think adaptively (the Claude 5
family, Mythos Preview, Opus 4.8, 4.7, and 4.6, Sonnet 4.6) get
`{"type": "adaptive", "display": "summarized"}` on `thinking=True`, and `thinking=False` sends
nothing, so a model that thinks by default keeps doing so; `thinking_effort` goes in
`output_config.effort` (`low` to `max`; no `xhigh` on Mythos Preview and the 4.6 models) and
applies without `thinking=True`. The older models (Opus, Sonnet, and Haiku 4.5, and the Claude 4
models before them) take a budget: an effort or a `thinking_budget` (at least 1,024 tokens) turns
thinking on, and the budget is added to `max_tokens`. The newer models take no sampling
parameters: their `temperature` is dropped (the `LLM` always sends one) and `top_p` or `top_k`
raise `RequestError`; the 4.6 and older models accept them. `claude-opus-5-5`,
`claude-fable-5-1`, and `claude-mythos-5-1` refuse a forced `tool_choice` (`"required"` or a name)
with `RequestError`.
The `web_search` and `code_execution` server tools are sent as `web_search_20250305` and
`code_execution_20250825`. A turn's tool results go back in one `user` message, and an assistant
turn is replayed as Claude sent it, thinking signatures included.

`claude-opus-4-7` rejects `temperature`; the provider drops temperature for this model.
Its tool probe still fails because the model repeatedly returned a malformed tool call
with only `{"a": 2}` where the schema required both `a` and `b`.

| Model | Plain | Tools | Structured | JSON Mode | Stream | Thinking | Notes |
|---|---|---|---|---|---|---|---|
| `claude-opus-4-7` | Pass | Fail | Pass | Pass | Pass | Not probed | Tool call omitted required argument `b`. |
| `claude-sonnet-4-6` | Pass | Pass | Pass | Pass | Pass | Not probed | Native structured output passed. |
| `claude-haiku-4-5` | Pass | Pass | Pass | Pass | Pass | Not probed | Native structured output passed. |
| `claude-opus-4-6` | Pass | Pass | Pass | Pass | Pass | Not probed | Native structured output passed. |
| `claude-sonnet-4-5` | Pass | Pass | Pass | Pass | Pass | Not probed | Native structured output passed. |
| `claude-opus-4-5` | Pass | Pass | Pass | Pass | Pass | Not probed | Native structured output passed. |
| `claude-sonnet-4-0` | Pass | Pass | Not probed | Pass | Pass | Not probed | Retired: its snapshot `claude-sonnet-4-20250514` answers 404 since 2026-06-15, and it has no price here (`UnpricedModelError` in a `Flow` or `Agent` run). Native structured output was disabled; JSON mode passed. |

## Meta

`muse-spark-*` models run on the Meta Model API (`https://api.meta.ai/v1`, key from
`MODEL_API_KEY`). Meta ships no SDK, so the adapter drives the `openai` SDK's Responses
API — install the `meta` extra. It is the surface Meta recommends for agents: each response
carries the model's reasoning encrypted, and appending `response.to_message()` to the
conversation replays it on the next call, so tool loops keep their chain of thought.
Requests are stateless (`store: false`); nothing is kept on Meta's side.

What differs from other providers:

- Muse Spark always reasons. `thinking_effort` (`"minimal"`, `"low"`, `"medium"`, `"high"`,
  `"xhigh"`, and `"max"` on standard `muse-spark-1.3`) applies without `thinking=True`;
  `"none"` raises `RequestError`, since Meta answers it with a 400. `thinking=True` asks for
  reasoning summaries, which Meta does not produce on every call.
- Reasoning tokens count toward `max_tokens`. A budget that is too small ends the call with
  `stop_reason == "max_output_tokens"` and little or no text.
- `tool_choice` accepts only `"auto"`. `"none"` sends the request without tools; forcing a
  tool (`"required"` or a name) raises `RequestError` (a `ValueError`) before any call.
- `output_schema` is sent non-strict: Meta constrains the output to the schema either way,
  while strict mode would reject a plain Pydantic schema. Recursive schemas are rejected.
- `stop` is not supported, and `logprobs=True` raises `RequestError` (Meta answers it with a
  400). Meta tunes the model for `temperature=1.0`; the `LLM` default is `0.0`, so pass
  `temperature=1.0` unless you need otherwise.
- Built-in `web_search` is supported (billed by Meta per query, so metering treats the cost
  as unknown); `code_execution`, or a server tool with a config, raises `RequestError`. There is
  no batch API.
- A failure Meta reports inside a response or a stream gets the HTTP status its error table gives
  the code; a code outside the table, or none, raises `ResponseError`. A failed response's usage
  is kept on the error (`ProviderError.usage`) and settled by the meter.
- Text from several assistant messages in one response (for example a note before a web
  search and the answer after it) is joined with a blank line.
- The contributor tiers (`muse-spark-1.3-contributor`, `muse-spark-1.2-contributor`) are much
  cheaper but let Meta train on your prompts and completions.
- Meta's backend often answers `503 service_overloaded`; configure `RetryConfig` for
  unattended runs.

Recorded live on 2026-09-13 with `scripts/probe_models.py` (`20260913T040525Z`; the stream
scenario hit a `503 service_overloaded` and passed on rerun, `20260913T040824Z`) and
`pytest -m live_api tests/integration/test_meta_live.py` (6 passed: a priced, stateless
completion, tool-loop reasoning replay, a ReAct agent, a streamed tool turn replayed into a
streamed answer, structured output with JSON mode, `count_tokens`).

| Model | Plain | Tools | Structured | JSON Mode | Stream | Thinking | Notes |
|---|---|---|---|---|---|---|---|
| `muse-spark-1.3` | Pass | Pass | Pass | Pass | Pass | Pass | Probes use `tool_choice="auto"`, `thinking_effort="low"`, `temperature=1.0`; no summary was returned in the thinking probe. |

`muse-spark-1.2`, `muse-spark-1.1`, and the contributor tiers share the adapter and have
pricing entries of their own but were not probed.

## Current Gaps

- `gemini-3.1-flash-live-preview` needs the Gemini Live API, which no adapter drives, so it is
  not supported.
- `claude-opus-4-7` returned incomplete tool arguments in repeated live probes, although the
  adapter sends the expected schema.
- The prepared live checks (`tests/integration/test_provider_hardening_live.py`) of the
  per-model thinking rules above have run live only for OpenAI (2026-10-02).
- OpenAI's reasoning guide says reasoning summaries may require a verified organization; what
  `thinking=True` gets without one has not been seen. (A batch on `/v1/responses`, read back at
  the batch rates, and an assistant turn rebuilt from its fields both passed live on
  2026-10-02.)
- The results above are only as current as the last probe run: SDK upgrades, provider API
  changes, and inventory changes call for a new full run.
