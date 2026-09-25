# Model Compatibility

This page lists the model IDs currently tracked by the live probe inventory and what
framework features were verified against them.

The baseline below comes from the manual live probe runner:

```bash
set -a; source .env; set +a
uv run python scripts/probe_models.py --suite full --timeout-seconds 120 --max-retries 1
```

Latest recorded full run:

- Date: 2026-04-28
- Models: 28
- Probe scenarios: 147
- Passed: 144
- Failed: 3
- Report artifact: `scripts/output/model-probes/20260428T015941Z.md`

Generated probe artifacts are local diagnostic files and are ignored by git.

The tables record that run. The text above each table gives the adapter's current rules, which
follow each provider's documentation per model: an option a model does not take raises
`RequestError` before anything is sent, and a model the adapter does not know gets the rules of the
provider's current generation. `tests/integration/test_provider_hardening_live.py` holds the
cheap owner-run checks of those rules (`pytest -m live_api -k <provider>`).

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

### GPT-6 Astra (added 2026-09-04)

`gpt-6-astra` is registered with standard, cached, batch, long-context, and fast
pricing. The Chat Completions adapter translates `max_tokens`, removes unsupported
sampling/logprob parameters, and accepts reasoning efforts `low`, `medium`, `high`,
`xhigh`, and `max`. Text, structured output, and streaming probes are configured but
have **not been run live** for this model.

Tool calling requires Responses and is rejected with an explanatory error by this
Chat Completions adapter. Astra therefore cannot run tool-using agents here yet.
See the [official Astra guide](https://developers.openai.com/api/docs/guides/latest-model)
and [model pricing](https://developers.openai.com/api/docs/models/gpt-6-astra).

### GPT-6 Sol and Luna (added 2026-09-25)

The GPT-6 family is Astra, Sol, and Luna; Terra is a GPT-5.6 tier (`gpt-5.6-terra`). `gpt-6-sol`
and `gpt-6-luna` are registered with standard, cached, batch, long-context (above 272K input
tokens), and fast pricing. They take efforts `none` to `max` (no `minimal`) and reason at
`medium` when no effort is sent. Chat Completions takes their tool calls and sampling parameters
(`temperature`, `top_p`, logprobs) only at `none`, so the adapter:

- sends a tool call that asks for no thinking at `reasoning_effort="none"`, which keeps the
  sampling parameters;
- raises `RequestError` for tools with `thinking=True` at another effort (reasoning with tools
  needs the Responses API);
- drops the sampling parameters whenever the model reasons.

Both passed every probe live on 2026-09-25 (`scripts/probe_models.py --suite full`, report
`20260925T013106Z`). See the model pages for
[Sol](https://developers.openai.com/api/docs/models/gpt-6-sol) and
[Luna](https://developers.openai.com/api/docs/models/gpt-6-luna).

| Model | Plain | Tools | Structured | JSON Mode | Stream | Thinking | Notes |
|---|---|---|---|---|---|---|---|
| `gpt-6-sol` | Pass | Pass | Pass | Pass | Pass | Pass | Tool calls sent at `none`; `thinking_effort="low"` in probes. |
| `gpt-6-luna` | Pass | Pass | Pass | Pass | Pass | Pass | Tool calls sent at `none`; `thinking_effort="low"` in probes. |

### Recorded live baseline

On `api.openai.com` every model receives `max_completion_tokens` (the provider translates
`max_tokens`, which OpenAI deprecates and the o-series refuses); an OpenAI-compatible server
(`base_url=`) receives `max_tokens` and no OpenAI model rule. `thinking=True` sends
`reasoning_effort` (`thinking_effort`, else `"high"`), checked against the model's efforts, and
drops a `temperature` other than 1 unless the effort is `"none"`. The models that do not reason
(`gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini`) raise `RequestError` on `thinking=True`.
The models OpenAI shuts down by 2026-10-23 (`gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo`,
`gpt-4.1-nano`, `o1`, `o1-pro`, `o3-mini`, `o4-mini`) have no price or rules here any more
([deprecations](https://developers.openai.com/api/docs/deprecations)). Chat Completions takes
function tools only: a server tool raises `RequestError`. From GPT-5.4 on it takes tool calls
only at the `none` effort
([migration guide](https://developers.openai.com/api/docs/guides/migrate-to-responses)): tools with
`thinking=True` raise `RequestError` unless `thinking_effort="none"`, while the earlier reasoning
models (`gpt-5` to `gpt-5.3` and the o-series) take them at any effort.

| Model | Plain | Tools | Structured | JSON Mode | Stream | Thinking | Notes |
|---|---|---|---|---|---|---|---|
| `gpt-5.5` | Pass | Pass | Pass | Pass | Pass | Pass | Uses `thinking_effort="low"` in probes. |
| `gpt-5.4` | Pass | Pass | Pass | Pass | Pass | Pass | Uses `thinking_effort="low"` in probes. |
| `gpt-5.4-mini` | Pass | Pass | Pass | Pass | Pass | Pass | Uses `thinking_effort="low"` in probes. |
| `gpt-5.4-nano` | Pass | Pass | Pass | Pass | Pass | Pass | Uses `thinking_effort="low"` in probes. |
| `gpt-4.1` | Pass | Pass | Pass | Pass | Pass | Not probed | No thinking scenario enabled. |
| `gpt-5-mini` | Pass | Pass | Pass | Pass | Pass | Pass | Needs a larger output budget than newer GPT-5 IDs in probes. |
| `gpt-5-nano` | Pass | Pass | Pass | Pass | Pass | Pass | Needs a larger output budget than newer GPT-5 IDs in probes. |
| `gpt-5` | Pass | Pass | Pass | Pass | Pass | Pass | Needs a larger output budget than newer GPT-5 IDs in probes. |
| `o3` | Pass | Pass | Pass | Pass | Pass | Pass | Exact `o3` routes to OpenAI and uses `max_completion_tokens`. |

## xAI

Grok reasoning models reason on their own, so `thinking=True` sends nothing more (it raises
`RequestError` on a model that does not reason, `grok-4.20-non-reasoning`). `thinking_effort`
applies without it and is sent as `reasoning_effort` where the model documents one: `low` to
`xhigh` on `grok-4.7`, `grok-4.6`, `grok-4.5` (which serves `xhigh` as `high`), and newer models;
`none` to `xhigh` on `grok-4.3` and the ids retired on 2026-05-15 that xAI now serves with it
(`grok-4-1-fast-reasoning`, for example). `grok-4.20-reasoning` and `grok-build-0.1` take no
effort (`RequestError`). The reasoning models refuse `stop`, `presence_penalty`, and
`frequency_penalty`, so those raise `RequestError` too. `tool_choice` takes a tool's name. A
server tool raises `RequestError` (the adapter does not send them yet), and images and documents
are dropped with a warning.

`grok-4.20-multi-agent` takes no client-side tools (`RequestError`) and no `max_tokens` (the
adapter leaves it out); `thinking_effort` picks the number of agents (4 for `low` and `medium`,
16 for `high` and `xhigh`), and `agent_count=` sets it directly. The probes ran it with
`agent_count=4`, on `plain` and `stream` only.

`grok-4.7` (added 2026-09-25) costs $2 input and $6 output per million tokens, and $4 and $12 for
the whole request once the prompt reaches 200k tokens; it has no Batch API. It follows the
current generation's rules above. Grok 4.7 Fast is not on the API. Its probe is in the inventory
but has **not been run live** ([model page](https://docs.x.ai/developers/models/grok-4.7)).

| Model | Plain | Tools | Structured | JSON Mode | Stream | Thinking | Notes |
|---|---|---|---|---|---|---|---|
| `grok-4.20-reasoning` | Pass | Pass | Pass | Pass | Pass | Auto | `thinking_effort` raises `RequestError`. |
| `grok-4.20-non-reasoning` | Pass | Pass | Pass | Pass | Pass | Not probed | Standard non-reasoning configuration. |
| `grok-4.20-multi-agent` | Pass | Not probed | Not probed | Not probed | Pass | `agent_count=4` | Custom client tools are not enabled for this model. |
| `grok-4-1-fast-reasoning` | Pass | Pass | Pass | Pass | Pass | Auto | Retired 2026-05-15; served by `grok-4.3`, which takes `thinking_effort`. |
| `grok-4-1-fast-non-reasoning` | Pass | Pass | Pass | Pass | Pass | Not probed | Standard non-reasoning configuration. |

## Gemini

Gemini probes use the current Gemini generate-content provider path. Live API-only models
need separate provider support.

> **Known issue (2026-09): Gemini and tool calls — fixed in code, not yet confirmed live.** The
> "Tools" column covers a single tool call. The adapter used to send the results of *parallel*
> tool calls back as separate `user` turns without the call `id`, and a history replayed after
> `stream()` / `stream_events()` could drop function calls and thought signatures, so multi-tool
> agent loops could fail with HTTP 400. It now sends a turn's results in one `user` content, with
> each call's `id` when Gemini gave one, and replays the model's turn as Gemini sent it. This
> note stays until a live run confirms it
> (`pytest tests/integration/test_provider_hardening_live.py -m live_api -k gemini`); until then,
> prefer another provider for multi-tool agents.

Thinking follows each model's documented controls. On Gemini 3, `thinking_effort` is the
thinking level and applies without `thinking=True`: `low`, `medium`, and `high` on the 3.8 and
3.7 Flash, 3.1 Pro, and newer models; also `minimal` on the 3.6 and 3.5 Flash, 3.5 and 3.1
Flash-Lite, and 3 Flash; `low` and `high` on 3 Pro. `thinking=True` asks for thought summaries
and, without an effort, thinks at `high`; a `thinking_budget` is ignored with a warning. On Gemini
2.5, the effort becomes a thinking budget (2,048, 5,000, or 10,000 tokens), and a
`thinking_budget` must be in the model's documented range: 128 to 32,768 on 2.5 Pro, 0 to 24,576
on 2.5 Flash, and 0 or 512 to 24,576 on 2.5 Flash-Lite (`-1` is dynamic everywhere). The
`web_search` and `code_execution` server tools are sent as `google_search` and `code_execution`;
a server tool with a config raises `RequestError`. The SDK always sends through `httpx`, so it
never re-sends a request on its own.

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
`tool_choice` with `RequestError`. Its probes are in the inventory but have **not been run live**.
See the [migration guide](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide).

### Recorded live baseline

Anthropic models use top-level `system`, `input_schema` for tools, and native
`output_config` for structured output when the model supports it. `max_tokens` is required (the
`LLM` always sends one).

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
| `claude-sonnet-4-0` | Pass | Pass | Not probed | Pass | Pass | Not probed | Native structured output is disabled; JSON mode passed. |

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
`pytest -m live_api tests/integration/test_meta_live.py` (6 passed: tool-loop reasoning replay,
a ReAct agent, a streamed tool turn replayed into a streamed answer, structured output, JSON
mode, `count_tokens`).

| Model | Plain | Tools | Structured | JSON Mode | Stream | Thinking | Notes |
|---|---|---|---|---|---|---|---|
| `muse-spark-1.3` | Pass | Pass | Pass | Pass | Pass | Pass | Probes use `tool_choice="auto"`, `thinking_effort="low"`, `temperature=1.0`; no summary was returned in the thinking probe. |

`muse-spark-1.2`, `muse-spark-1.1`, and the contributor tiers share the adapter and
pricing entries but were not probed.

## Current Gaps

- Add a Gemini Live API provider path before advertising `gemini-3.1-flash-live-preview`
  as generally supported.
- Revisit `claude-opus-4-7` tool calling. The current provider sends the expected schema,
  but the model returned incomplete arguments in repeated live probes.
- Run the prepared live checks (`tests/integration/test_provider_hardening_live.py`) to confirm
  the per-model thinking rules above against each provider; they have not run live yet.
- Keep model support current by rerunning the full probe matrix after SDK upgrades,
  provider API changes, or inventory changes.
