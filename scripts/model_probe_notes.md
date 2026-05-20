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
