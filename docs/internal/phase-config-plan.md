# Per-Phase Configuration — Restoration Plan (implementation contract)

> **Status:** implemented (all three stages) — kept as the design record.
> Scope: restore per-phase LLM/tools/prompt configuration to the recommended
> `Agent`/`ReasoningSpec` API and to agent manifests, without inventing a new
> config concept and without breaking the public flow-factory API.

---

## 1. Problem statement

Per-phase customization regressed across three commits:

- `c5f2099` introduced `PhaseConfig` (`{llm, tools}` per phase) across all 7
  multi-phase agents, plus per-phase `*_system` prompts, with ~20 tests.
- `fea2fb5` replaced legacy agents with Flow factories. Most capability
  survived as factory kwargs (`planner_llm`, `exec_tools`, `solver_system`, …);
  `PhaseConfig` and its tests were deleted.
- `cd68fb5` introduced `ReasoningSpec`/`Agent`; the builders forward only the
  global `llm`, `tools`, `system`, and `llm_kwargs`. The manifest schema
  followed the same single-model shape.

Current gap inventory (verified on `main`):

| # | Gap | Where |
|---|-----|-------|
| 1 | `llm_compiler` executor LLM override lost **even at factory level** — inner ReAct hardcodes the default `llm` | `flows/_llm_compiler.py:141` |
| 2 | `spec.llm_kwargs` silently dropped by 7 of 10 strategies (only `react`, `completion`, `generate_review` gen-phase forward it; the other factories don't even accept `llm_kwargs`) | `_builders.py` + 7 factories |
| 3 | Zero tests cover the surviving per-phase factory kwargs (`planner_llm`, `rollout_llm`, …) | `tests/` |
| 4 | `deps` is unvalidated — a typo'd key (`evalutor`) is silently ignored and the default evaluator runs | `_builders.py` |
| 5 | `plan_execute` planner lost tool coherence — the planner no longer sees the executor's tool list (`rewoo` kept it) | `flows/_plan_execute.py:62` vs `flows/_rewoo.py:74` |
| 6 | `generate_review` builder can't set `review_system`/`review_kwargs` (factory supports both) | `_builders.py:249` |
| 7 | `lats` `exploration_weight` factory param not exposed as a knob | `_builders.py:308` |
| 8 | Manifests cannot express per-phase prompts or models | `_manifest.py` |

## 2. Governing decisions (settled)

1. **No new "roles"/"phases" config object at the spec level.** The codebase
   already has the two buckets this needs, with the right semantics:
   - `spec.knobs` — serializable, strategy-specific options, validated by
     `FlowStrategy.allowed_knobs`/`knob_validators` → carries **per-phase
     prompts** (and later per-phase kwargs).
   - `BuildContext.deps` — runtime objects that cannot serialize → carries
     **per-phase `LLM`/`ToolGroup` instances** (generalizing the existing
     `generate_review` `deps["review_llm"]` precedent).
   `ReasoningSpec` itself does not change.
2. **Canonical phase vocabulary at every layer above the factories.** Phase
   names: `planner`, `executor`, `solver`, `reflector`, `reasoning`,
   `generator`, `evaluator`, `rollout`, `joiner`, `reviewer`. Derived keys:
   deps `<phase>_llm` / `<phase>_tools`, knobs `<phase>_system`. Builders
   translate canonical → factory kwarg (`executor_llm` → `exec_llm`) in the
   same place validation lives. **Factory kwargs are frozen public API** — no
   renames.
3. **Symmetric deps validation.** `FlowStrategy` gains `allowed_deps` +
   `dep_validators` (type checks: is `LLM`, is `ToolGroup`, is callable),
   mirroring `allowed_knobs`. `allowed_deps=None` (default) keeps deps
   unvalidated — preserves behavior for user-registered custom strategies.
   All built-ins declare explicit sets; `react`/`completion` declare the empty
   set, so stray deps fail loudly.
4. **Manifest: per-phase config lives under `strategy.phases`**, repeating the
   established top-level split — prompts flow into the spec, model configs are
   data the **application** resolves (the loader never constructs LLMs).
   Two-tier validation: the loader validates shape/paths/secrets
   (registry-agnostic, per its docstring); phase/knob names are validated
   registry-aware at build time and statically via a new CLI command.
5. **Audit comes from existing machinery.** The deterministic fingerprint
   canonicalizes the whole tree (inline phase config included); file-backed
   phase prompts join `referenced_fingerprints` (per-file sha256); dotted-path
   override governance covers `strategy.phases.*` by prefix with zero changes;
   `_assert_no_secrets` walks phase model configs for free.

## 3. Canonical vocabulary (per strategy)

| Strategy | Phase | deps keys | knobs | maps to factory kwargs |
|---|---|---|---|---|
| `react`, `completion` | — | ∅ (strict empty) | (existing knobs only) | — |
| `plan_execute` | planner | `planner_llm` | `planner_system` | `planner_llm`, `planner_system` |
| | executor | `executor_llm`, `executor_tools` | — | `exec_llm`, `exec_tools` |
| | solver | `solver_llm` | `solver_system` | `solver_llm`, `solver_system` |
| `rewoo` | planner | `planner_llm` | `planner_system` | same names |
| | solver | `solver_llm` | `solver_system` | same names |
| `reflexion` | executor | `executor_llm`, `executor_tools` | — | `exec_llm`, `exec_tools` |
| | reflector | `reflector_llm` | `reflector_system` | `reflect_llm`, `reflect_system` |
| `self_discovery` | reasoning | `reasoning_llm` | `select_system`, `adapt_system`, `plan_system` (plain knobs — see note) | `reasoning_llm`, `select_system`, `adapt_system`, `plan_system` |
| | solver | `solver_llm`, `solver_tools` | `solver_system` | `solver_llm`, `solver_tools`, `solve_system` |
| `tot` | generator | `generator_llm` | — | `gen_llm` |
| | evaluator | `evaluator_llm` | `evaluator_system` | `eval_llm`, `evaluator_system` |
| | solver | `solver_llm` | — | `solver_llm` |
| `lats` | rollout | `rollout_llm`, `rollout_tools` | — | `rollout_llm`, `rollout_tools` |
| | evaluator | `evaluator_llm` | `evaluator_system` | `eval_llm`, `evaluator_system` |
| | solver | `solver_llm` | — | `solver_llm` |
| | reflector | `reflector_llm` | `reflector_system` | `reflector_llm`, `reflect_system` |
| `llm_compiler` | planner | `planner_llm` | `planner_system` | same names |
| | executor | `executor_llm`, `executor_tools` | — | `exec_llm` (**new param**), `exec_tools` |
| | joiner | `joiner_llm` | `joiner_system` | same names |
| `generate_review` | generator | — (the global `llm`/`tools`/`system`/`llm_kwargs` *are* the generator) | — | `gen_llm`, `gen_tools`, `gen_system`, `gen_kwargs` |
| | reviewer | `reviewer_llm`, `reviewer_tools` | `reviewer_system`, `reviewer_kwargs` | `review_llm`, `review_tools`, `review_system`, `review_kwargs` |

Notes:
- **Evaluator callables stay in deps under their existing keys** (`reflexion` →
  `evaluator`, `lats` → `evaluator_fn`) — callables aren't phase LLMs and don't
  serialize. No churn.
- **`self_discovery` has no single `reasoning` system prompt** — its three
  sub-prompts remain plain knobs. `phases.reasoning.system` in a manifest is a
  validation error with a message pointing at the three knobs.
- `generate_review` legacy deps keys `review_llm`/`review_tools` remain
  accepted as documented aliases of the canonical `reviewer_*` keys (both in
  `allowed_deps`; canonical documented, legacy noted for compatibility).
- `lats` additionally gains the `exploration_weight` knob (gap 7).
- New `FlowStrategy.phases: frozenset[str]` metadata field enables CLI/docs
  introspection.

## 4. Stage 1 — capability in the recommended API

Everything below the manifest. Independently shippable.

**1a. Factory fixes** (`toolkit/agents/flows/`):
- `_llm_compiler.py`: add `exec_llm: LLM | None = None` param; use it for the
  inner ReAct (gap 1).
- All 7 multi-phase factories: accept `llm_kwargs: dict | None = None` and
  forward to every internal `*.complete()` call and inner `react_flow(...)`
  (gap 2). Global kwargs apply to every phase.
- Planner tool awareness via an explicit `{tools}` token (decided in post-review,
  replacing this plan's original "append like `rewoo`" approach — silent appends
  mutate declared prompts, breaking the "declared = effective" contract Stage 2
  establishes). `substitute_tools()` in `flows/_common.py` replaces the exact
  token with the phase's rendered catalog (`(none)` when empty, `str.replace`
  so JSON braces and `#E{n}` survive); a prompt without the token is never
  modified. Default planner prompts of `plan_execute`, `rewoo`, and
  `llm_compiler` carry the token (gap 5, extended to `llm_compiler`, which had
  no tool awareness; `rewoo`'s silent append removed).

**1b. Builder machinery** (`_builders.py`):
- `FlowStrategy`: add `phases: frozenset[str] = frozenset()`,
  `allowed_deps: frozenset[str] | None = None`,
  `dep_validators: Mapping[str, Callable[[Any], bool]]`. `build()` validates
  deps exactly like knobs: unknown keys and failed type checks raise
  `ValueError` naming the strategy (gap 4). `None` = legacy/custom, no checks.
- Type validators: `_is_llm`, `_is_tool_group`, `_is_callable`.
- Each `_build_*` reads canonical deps/knobs and translates to factory kwargs
  per the §3 table; forwards `spec.llm_kwargs` (gaps 2, 6, 7).

**1c. Tests** (restores the deleted `test_phase_config.py` coverage, gap 3):
- `tests/agents/test_phase_overrides.py` — builder-level, parametrized over
  (strategy × phase): the phase-specific `AsyncMock` LLM receives the call
  (and the default LLM doesn't); knob-overridden system prompt reaches the
  right `.complete()`; unknown dep key raises; wrong-typed dep raises;
  `react`/`completion` reject any dep. Follow `tests/agents/conftest.py`
  factories (`make_response`) and the repo's AsyncMock-LLM convention.
- `tests/agents/flows/` — factory-level: `exec_llm` routing in
  `llm_compiler_flow`; `llm_kwargs` reaches phase calls; `plan_execute`
  planner prompt contains tool descriptions.

**1d. Docs/changelog:** new "Per-phase configuration" section in
`docs/agents.md` with the §3 table; update the recognized-deps list;
`CHANGELOG.md` `[Unreleased]` Added (per-phase config) + Fixed (silent
`llm_kwargs` drop, `llm_compiler` executor LLM, planner tool coherence).

## 5. Stage 2 — declarative per-phase prompts in manifests + CLI validate

**2a. Schema** (`_manifest.py`): new `strategy.phases` mapping. Phase fields
this stage: `system` (inline string) XOR `system_file` (path). Loader
validates: known fields only, non-empty strings, XOR rule, path under
`allowed_roots`, file exists. Profiles/inheritance need no work (`phases`
lives under `strategy`, covered by the generic deep-merge and profile
validation).

**2b. Path-field mechanics:** generalize the static `_PATH_FIELDS` handling so
dynamic paths (`strategy.phases.*.system_file`) get resolution, root checks,
`referenced_fingerprints` entries, and portable-config rewriting. Changing a
phase prompt file changes the manifest fingerprint without touching the YAML.

**2c. Ambiguity rule:** declaring both `strategy.knobs.<phase>_system` and
`strategy.phases.<phase>.system`/`system_file` for the same phase is an
`AgentManifestError` at load (consistent with override-overlap rejection).
One declaration site per value.

**2d. Bridge:** `reasoning_spec()` folds `phases.*.system` — and the *content*
of `phases.*.system_file`, read at bridge time — into the canonical
`<phase>_system` knobs. Loader stays IO-free at load; file content is already
pinned by the fingerprint.

**2e. CLI** (`_cli.py`): `ai-arch agent validate <path> [--profile P]` —
loads/resolves the manifest, then registry-aware checks: strategy name
registered; declared phase names ⊆ `FlowStrategy.phases`; folded knobs pass
`allowed_knobs`/`knob_validators`. Non-zero exit with precise messages, for
CI. Add `ai-arch agent inspect <path>` (resolved config + fingerprint),
mirroring `prompt inspect` — cheap while in there.

**2f. Tests:** extend `tests/agents/test_manifest.py` (shape, XOR, roots,
fingerprint sensitivity to prompt-file edits, ambiguity rejection,
inheritance/profile merge of phases, override governance on
`strategy.phases.*` paths, `reasoning_spec()` folding); CLI tests alongside
the existing `prompt` command tests. Docs + CHANGELOG.

## 6. Stage 3 — declarative per-phase models + runtime resolution

**3a. Schema:** `strategy.phases.<name>.model` — object reusing the existing
`_MODEL_FIELDS` validation (provider, model, base_url, temperature,
max_tokens, structured_output_mode, profile). Secret scan already applies.
Same contract as the top-level `model` section: validated data, resolved by
the application.

**3b. Accessor:** `ResolvedAgentManifest.phase_models()` →
`dict[str, dict[str, Any]]` (thawed copies) of phases declaring a `model`.

**3c. Optional sugar** (toolkit-level, not loader): `agent_from_manifest(
manifest, *, llm, tools=None, llm_factory=None, deps=None, …)` — calls
`llm_factory(phase, model_config)` per `phase_models()` entry, merges results
into deps as `<phase>_llm` (explicit `deps` wins), builds
`reasoning_spec()` + `Agent` in one call. Signature finalized at
implementation; it is convenience, not architecture — the manual path
(`phase_models()` → own deps) stays first-class.

**3d. CLI:** `agent validate` also checks phase names for `model` entries.

**3e. Tests:** `phase_models()`, fingerprint sensitivity to phase-model edits,
secret rejection inside phase models, `agent_from_manifest` end-to-end with a
fake `llm_factory` (assert the right phase LLM receives the right call —
reuses Stage 1's test harness). Docs + CHANGELOG.

## 7. Deferred (explicitly out of scope)

- Per-phase `llm_kwargs` end-to-end (factory params + `<phase>_kwargs` knobs +
  `phases.*.llm_kwargs` manifest field) — only `generate_review` supports it
  today (`reviewer_kwargs`); generalize when demanded.
- `system_file` rendering through the prompts system (variables, includes,
  layouts) — `system_file` is verbatim text by contract; prompt-manifest
  suffixes are rejected at load, and template rendering stays app-side.
- Per-phase tools declaratively (`phases.<name>.tools.factory`, mirroring the
  top-level `tools` section).
- Per-field provenance ("which manifest in the chain set the planner model").
- Attaching the manifest fingerprint to metering run reports (execution audit
  trail).
- Renaming factory kwargs for cosmetic consistency (`gen_llm` →
  `generator_llm`) — frozen; canonical translation in builders covers it.

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Strict deps validation breaks existing callers passing extra deps keys | Only built-ins get strict sets; `allowed_deps=None` default preserves old behavior for custom strategies; CHANGELOG calls it out |
| Legacy `review_llm`/`review_tools` users | Both key sets accepted; canonical documented |
| `reasoning_spec()` now performs file IO (`system_file`) | Bridge-time only; content fingerprint-pinned; documented in the method docstring |
| `phases.reasoning.system` confusion in `self_discovery` | Dedicated error message naming the three sub-prompt knobs |
| Knob namespace collision (a future factory param named like a phase knob) | Canonical `<phase>_system` naming rule + per-strategy `allowed_knobs` keeps the namespace closed |

## 9. Definition of done (each stage)

- `uv run pytest` green; new tests per stage sections above.
- `uv run ruff check --fix` + `ruff format`; `uv run pyright src` clean.
- `CHANGELOG.md` `[Unreleased]` entries for user-visible changes.
- `docs/agents.md` (and manifest docs) updated in the same PR.
- Each stage ships alone: Stage 1 without 2/3 is already a net capability and
  bug-fix release; Stage 2 without 3 gives declarative prompts; Stage 3
  completes declarative models.
