# Model Probe Notes

Running `scripts/probe_models.py` against live providers should accumulate findings here.
Keep entries factual: model ID, scenario, observed error, and the local action taken.

## 2026-04-28

- `gemini-3.1-flash-lite-preview` passed `plain`, `tools_loop`, `structured`, `json_mode`,
  and `stream`, but the first `thinking` probe returned empty final text. Treat Gemini
  thinking as an observational probe unless `require_thinking = true`; do not require a
  deterministic text sentinel for this model by default.
- `gpt-5.4-mini` rejected `temperature = 0.0` during the `thinking` probe:
  `Unsupported value: 'temperature' does not support 0.0 with this model. Only the default
  (1) value is supported.` Remove explicit temperature from this model's default probe
  config so the provider uses its default.
- `grok-4.20-reasoning-latest` passed `plain`, `tools_loop`, `structured`, and `json_mode`,
  but rejected `reasoningEffort`: `Model ... does not support parameter reasoningEffort`.
  Do not configure `thinking_effort` for this model.
- `grok-4.20-reasoning-latest` returned a 403 safety rejection for the stream prompt with
  `Content violates usage guidelines` and `SAFETY_CHECK_TYPE_BIO`. This is not an auth
  failure; classify it as `content_policy`.
- `gpt-5.4-mini` still rejected the `thinking` probe after removing temperature from the
  TOML inventory because the `LLM` facade injects `temperature = 0.0` by default. For
  OpenAI GPT-5-style reasoning calls, drop non-default temperature unless
  `reasoning_effort = "none"` or the caller explicitly passes `temperature = 1`.
- `grok-4.20-reasoning-latest` stream succeeds with a neutral arithmetic prompt
  (`What is 2 + 3? Reply with only 5.`). Replace the previous `STREAM_OK` sentinel prompt
  for the stream scenario to avoid xAI's false-positive `SAFETY_CHECK_TYPE_BIO` rejection.
- xAI error payloads may include provider team/API-key identifiers. Redact those identifiers
  before writing probe reports.
- After the provider fixes, `20260428T013009Z` passed every Anthropic, OpenAI, and xAI
  scenario, including `gpt-5.4-mini / thinking` and `grok-4.20-reasoning-latest / stream`.
  Gemini returned transient 503 high-demand errors on five scenarios in that full run.
- The follow-up Gemini-only retry `20260428T013307Z` passed all five previously failing
  Gemini scenarios: `tools_loop`, `structured`, `json_mode`, `stream`, and `thinking`.
- Expanded the inventory to 28 configured models and 147 enabled model/scenario probes.
  xAI `grok-4.20-multi-agent` is intentionally limited to `plain` and `stream` with
  `agent_count = 4`; xAI documents multi-agent as unsupported for client-side tools and
  `max_tokens`.
- OpenAI exact `o3` failed provider routing until exact `o1`/`o3`/`o4` IDs were added to
  provider detection. Exact `o3` also needs `max_tokens` translated to
  `max_completion_tokens`, same as the prefixed o-series IDs.
- Older OpenAI `gpt-5`, `gpt-5-mini`, and `gpt-5-nano` returned empty or truncated content
  with `max_tokens = 64` in non-thinking probes. Increasing the probe budget to `1024`
  made `plain`, `tools_loop`, `structured`, `json_mode`, `stream`, and `thinking` pass.
- `gemini-3.1-pro-preview` and `gemini-3-flash-preview` had truncated/partial outputs with
  `max_tokens = 64`. Increasing those probe budgets to `512` made all configured scenarios
  pass. Some `gemini-3.1-pro-preview` calls are slow; one final full-run plain probe took
  about 127 seconds.
- `gemini-2.5-flash-lite / thinking` rejects `thinking_budget = 256`; the provider requires
  at least `512`. Updating the probe config to `512` made the scenario pass.
- `gemini-3.1-flash-live-preview` still fails `plain` and `stream` with 404
  `not supported for generateContent`. This model likely needs a separate Live API path
  rather than the current Gemini generate-content provider.
- `claude-opus-4-7` rejects `temperature`; dropping temperature for this model fixed
  `plain`, `structured`, `json_mode`, and `stream`. Its `tools_loop` probe still fails
  because Anthropic repeatedly returns a malformed tool call with only `{"a": 2}` while the
  schema requires both `a` and `b`.
- `claude-sonnet-4-0` does not support Anthropic native structured output
  (`output_config`), but it passes `json_mode`. The inventory disables `structured` for this
  model.
- Final expanded run `20260428T015941Z`: 144 passed, 3 failed. Remaining failures are
  `gemini-3.1-flash-live-preview / plain`, `gemini-3.1-flash-live-preview / stream`, and
  `claude-opus-4-7 / tools_loop`.

## 2026-09-13

- Added `muse-spark-1.3` (Meta Model API through the new `meta` provider, Responses API over
  the `openai` SDK). `tools_loop` needs `tool_choice = "auto"`: Meta rejects `"required"`,
  `"none"`, and named choices with 400 (`only "auto" is supported for tool_choice`). The
  inventory now takes a per-model `tool_choice`.
- Full run `20260913T040525Z`: `plain`, `tools_loop`, `structured`, `json_mode`, and
  `thinking` passed; `stream` failed with 503 `service_overloaded` after the probe's retries
  (max 5 s backoff). Rerun `20260913T040824Z` passed. Meta's backend returned 503 on a large
  share of calls that day; its docs send `Retry-After: 60` on overload.
- The `thinking` probe returned no reasoning summary: Muse Spark always reasons, but a summary
  is optional. Keep `require_thinking` off.
- As a precaution, the sanitizer also redacts the Meta key format (`LLM|<digits>|...`) in
  probe reports.


## 2026-10-02

- Front O probe, `scripts/probe_openai_responses.py` (report `openai-responses-20261002T022338Z`,
  $0.03): the Responses API against Chat Completions on `gpt-6-luna` and `gpt-6.1-sol`, 6
  repeats per scenario.
- Latency, p50: no penalty for Responses.
  - Luna at `none`: 1.11 s against 1.35 s on Chat Completions; first streamed token 0.50 s
    against 0.65 s.
  - `gpt-6.1-sol` at `low`: 2.49 s against 2.36 s; first streamed token 1.10 s against 1.27 s.
  - A two-turn tool loop on Luna at `none`: 1.02 s and 0.93 s per turn, against 0.91 s and
    0.87 s.
  - The same cached tokens on a repeated 3,000-token prefix (2,527 on average, on both).
- Responses counted fewer input tokens for the same tool: 69 against 151 on the loop's first turn.
- With `store: false`, reasoning items carry `encrypted_content` without `include`; the legacy
  `include: ["reasoning.encrypted_content"]` is still accepted.
- OpenAI accepts replays that Meta refuses: a reasoning item by id without its encrypted content,
  a reasoning item followed by a user message instead of its call, and Luna's reasoning replayed
  to `gpt-6.1-sol` and to `gpt-5.5`.
- A function tool without `strict` is rewritten into strict mode: the response echoes
  `strict: true`, with every property required and `additionalProperties: false`. With
  `strict: false` the schema stays as sent.
- `stop` and `seed` get a 400 `unknown_parameter`; `frequency_penalty` and `presence_penalty` get
  a 500 after about 90 s, three runs out of three.
- Sampling follows the Chat Completions rules: Luna takes `temperature` and `top_p` only at
  `none`, and `gpt-6.1-sol` refuses `temperature` on both endpoints. Logprobs at `none` come with
  `top_logprobs` plus `include: ["message.output_text.logprobs"]`.
- Chat Completions refuses the `max` effort for every GPT-6 model ("Supported values are: ...
  'high', and 'xhigh'"; Astra, Sol and Luna checked one request each), while Responses takes it
  for `gpt-6.1-sol`. The adapter now raises `RequestError` for `max` on the four.
- Chat Completions refuses tools with reasoning on `gpt-6.1-sol` (at its default effort and at
  `none`) and on Luna at `low`: "use /v1/responses or set reasoning_effort to 'none'".
- Responses messages carry `phase: "final_answer"`; function_call items carry no other fields.
