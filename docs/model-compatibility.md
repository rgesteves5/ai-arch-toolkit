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

### Recorded live baseline

OpenAI GPT-5-style and o-series models use `max_completion_tokens` instead of
`max_tokens`. The provider translates `max_tokens` for these models. For GPT-5-style
reasoning calls, the provider drops non-default `temperature` unless the caller explicitly
uses `temperature=1` or `reasoning_effort="none"`.

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

Grok reasoning variants reason automatically. The framework does not send
`thinking`, `thinking_effort`, or `reasoning_effort` to normal Grok reasoning models.

`grok-4.20-multi-agent` is configured with `agent_count=4`. It is intentionally limited
to `plain` and `stream` probes because xAI documents multi-agent mode as incompatible
with client-side custom tools and `max_tokens`.

| Model | Plain | Tools | Structured | JSON Mode | Stream | Thinking | Notes |
|---|---|---|---|---|---|---|---|
| `grok-4.20-reasoning` | Pass | Pass | Pass | Pass | Pass | Auto | Explicit thinking parameters are ignored. |
| `grok-4.20-non-reasoning` | Pass | Pass | Pass | Pass | Pass | Not probed | Standard non-reasoning configuration. |
| `grok-4.20-multi-agent` | Pass | Not probed | Not probed | Not probed | Pass | `agent_count=4` | Custom client tools are not enabled for this model. |
| `grok-4-1-fast-reasoning` | Pass | Pass | Pass | Pass | Pass | Auto | Explicit thinking parameters are ignored. |
| `grok-4-1-fast-non-reasoning` | Pass | Pass | Pass | Pass | Pass | Not probed | Standard non-reasoning configuration. |

## Gemini

Gemini probes use the current Gemini generate-content provider path. Live API-only models
need separate provider support.

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

Anthropic models use top-level `system`, `input_schema` for tools, and native
`output_config` for structured output when the model supports it.

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

## Current Gaps

- Add a Gemini Live API provider path before advertising `gemini-3.1-flash-live-preview`
  as generally supported.
- Revisit `claude-opus-4-7` tool calling. The current provider sends the expected schema,
  but the model returned incomplete arguments in repeated live probes.
- Add explicit Anthropic thinking probes if Claude thinking support should be documented
  per model.
- Keep model support current by rerunning the full probe matrix after SDK upgrades,
  provider API changes, or inventory changes.
