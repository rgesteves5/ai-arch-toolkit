# Metering remediation plan — closing the gaps to "10/10"

**Goal.** Bring the `feat/metering-clean` implementation into full agreement with
the contract in [`metering-plan.md`](metering-plan.md). Source: an external LLM
code review (9 findings), independently re-verified against the source and the
plan's line references. This document records the *calibrated* verdict and the
concrete fix for each surviving gap.

**Verification summary.** 3 clean confirmations that fire on default/happy-path
usage (F1, F3, F9), 2 real-but-conditional (F2, F5), and 4 that need re-scoping
or a design decision (F4, F6, F7, F8). The base is solid and the focused suite is
green — this list is what stands between "implemented" and "contract-complete".

Severities: **P1** = wrong result on default usage (money not enforced/reported).
**P2** = wrong result only under an explicit non-default opt-in, or a misleading
public doc. **P3** = unbuilt-but-aspirational surface / cosmetic.

---

## Tier 1 — P1, must fix (default/happy-path correctness)

### F1 · Completion strategy swallows `AdmissionDenied`
- **Where.** `src/ai_arch_toolkit/toolkit/agents/_builders.py:156` — the
  `completion` strategy's `_complete` step wraps `llm.complete()` in a bare
  `except Exception: return Result(error=...)`.
- **Root cause.** `BudgetExceeded → AdmissionDenied → Exception`, so a budget
  denial is caught here and downgraded to a normal error `Result`. It never
  reaches the flow executor's denial handler, so no `budget_exceeded` decision is
  recorded. Violates the plan's remediation contract (`metering-plan.md:278-280`:
  the catch-`Exception`→`Result` sites "must re-raise `AdmissionDenied` /
  `NotMeteredOperationError`; ONLY the flow executor converts a denial"). The plan
  listed `_step_engine` and `_tools/_executor` but missed this fourth site.
- **Fix.** Add a terminal guard before the broad catch:
  ```python
  except (AdmissionDenied, NotMeteredOperationError):
      raise
  except Exception as exc:
      return Result(error=str(exc))
  ```
- **Also.** Grep every bespoke step fn in `toolkit/agents/` and `toolkit/flow/`
  for `except Exception` around an `llm.*`/tool call; `completion` is unlikely to
  be the only bespoke one. Add the same guard wherever found.
- **Tests.** Metering test: a `completion`-strategy `Agent` under an enforcing
  `BudgetPolicy(max_llm_calls=0)` must yield a `budget_exceeded` decision /
  `result.report.over_budget is True`, **not** a generic `Result(error=...)`.
- **Done when.** A denial in the single-shot completion agent surfaces as the
  terminal `budget_exceeded` decision, identical to the react path.

### F3 · Per-step `Policy.max_cost` ignores unknown/unpriced spend
- **Where.** `src/ai_arch_toolkit/core/_step_engine.py:118-124` checks
  `effective_cost = result.cost + _span_cost(span_id)`; `_span_cost()`
  (`:156-163`) returns only `snapshot.cost.to_float()` (known cost). A span whose
  spend is entirely unpriced (unpriced model / server tool) has `cost == 0` and
  `unknown_cost_count > 0`, so the step passes `Policy(max_cost=...)`.
- **Root cause.** The per-step path does not mirror the run-level fail-closed
  logic (`toolkit/budget/_controller.py:88-98`: deny when
  `snap.unknown_cost_count > 0` or `request.has_server_tools` under
  `unpriced="fail_closed"`). Violates the plan's "same mechanism as run-level,
  **unified**" (`metering-plan.md:318`).
- **Fix.** Add `_span_has_unknown_cost(span_id) -> bool` (reads
  `meter.for_span(span_id).unknown_cost_count`). In the `max_cost` block, when the
  span has unknown cost, fail closed — emit a decision and an error `Result`.
  - **Decided.** Fail closed by default (match run-level default), signalled with
    a **new, distinct `cost_unknown` `PolicyDecision` literal** — cleaner,
    unambiguous traces (a `cost_unknown` breach is not the same event as a
    `cost_exceeded` overspend). Add the literal to the `PolicyDecision` type,
    handle it in the step engine, and document it in `flow-architecture.md` /
    `safety.md` alongside the other decisions.
- **Tests.** A step with `Policy(max_cost=0.01)` calling an **unpriced** model
  (real `LLM` + fake provider, no pricing entry) must halt with the cost decision,
  not pass.
- **Done when.** Per-step `max_cost` and run-level `max_cost` treat unknown cost
  identically (both fail closed by default).

---

## Tier 2 — P2, real but conditional (fix; cheap)

### F2 · `complete()` / stream paths fall back after a denial under broad `fallback_on`
- **Where.** `src/ai_arch_toolkit/core/_llm.py` — `except self._fallback_on` at
  lines **461, 532, 644, 700, 803, 857**, with no `except AdmissionDenied` before
  them. Default `_fallback_on = PROVIDER_ERRORS` (`:209`) does **not** include
  `AdmissionDenied`, so the default path is safe; the "never retried" half already
  holds (`with_retry`/`_retry.py` only retries `RateLimitError`/`APIError`).
- **Root cause.** Under `fallback_on=(Exception,)` (or any tuple including a denial
  supertype) **with** fallbacks configured, a denial is caught and re-tried against
  fallback models. Violates the plan's absolute wording
  (`metering-plan.md:276-277`: "terminal — never retried, never fellback, **even
  under `fallback_on=(Exception,)`**").
- **Fix.** Insert a terminal guard immediately before each of the 6 sites:
  ```python
  except (AdmissionDenied, NotMeteredOperationError):
      raise
  except self._fallback_on as ...:
  ```
  This single change also hardens F1's class of bug across every LLM entry point.
- **Tests.** `LLM(..., fallback="…", fallback_on=(Exception,))` under an enforcing
  budget: the denial must propagate; the fallback model must **not** be called
  (assert on the fake provider's call count). Cover `complete`, `stream`,
  `stream_events` (+ `_sync`).

### F5 · `budget_scope(pricer=…)` can price the estimate and the settle differently
- **Where.** `src/ai_arch_toolkit/toolkit/budget/_scope.py:37` builds
  `BudgetController(policy, estimator=estimator or HeuristicEstimator())` — the
  passed `pricer` is **not** threaded into the default estimator, whose `pricer`
  defaults to `None` → falls back to the `pricing` singleton
  (`_estimator.py:45`). Settle uses `scope.pricer or pricing` (`_llm.py:351`).
- **Root cause.** Estimate (enforcement, under `reserve="strict"`) and settle
  (reporting) can use different pricing. Violates plan `:225-226` ("the SAME
  pricer used by the controller's estimate AND the charge-site settle"). Only
  reachable under `reserve="strict"` + custom `pricer=` + defaulted `estimator=`
  (default `reserve="none"` never estimates).
- **Fix.** One line: `estimator or HeuristicEstimator(pricer=pricer)`.
- **Tests.** `budget_scope(BudgetPolicy(reserve="strict", max_cost=…),
  pricer=custom)` — assert the reservation uses `custom`, not `pricing`.

---

## Tier 3 — re-scope / design decision (not a straight "bug fix")

### F4 · "Strict" reserve is not a proven worst-case ceiling
- **Verified.** Two real under-reservation sources; the review mislocated the
  third.
  - `_content_chars()` (`_llm.py:130`) omits `output_schema`/`output_config` from
    the char count, though the plan (`:116`) names it in `content_size_hint`. **Fix:**
    include the serialized output schema length.
  - `HeuristicEstimator.estimate` (`_estimator.py:42`) uses floor `//` — a
    worst-case ceiling should round **up**. **Fix:** `math.ceil` / `-(-h // c)`.
    (Magnitude < 1 token; correctness, not materiality.)
  - **Media (mis-attributed by the review).** Media is on a separate facts field
    `non_text_parts` (`_llm.py:400`), by design — *not* in `_content_chars`. The
    real gap: `estimate()` never **reads** `non_text_parts`, so images/documents
    contribute 0 to the input-token reservation. **Fix:** add a per-part token
    allowance in the estimator.
- **Severity.** P2. Only a strict-reserve concern; priced text models are largely
  fine. Do it, but it is not a happy-path breach.

### F6 · `RunConfig` missing `allow_unmetered_batch` / `retain_meter_events` / `sink_error_policy` / `Flow.run(config=…)`
- **Verified as fact, refuted as contract violation.** The core
  `_metering/_scope.py:54` `RunConfig` (`controller/sinks/redactor/pricer/clock`)
  was never spec'd to carry those. The plan scopes them to a **different,
  toolkit-level** `RunConfig` (`:246`), and `Flow.run(config=RunConfig(...))` is
  described as a **new/aspirational** signature (`:363,:368`). Absent fields
  default to no-op behavior (log/ignore, off).
- **Decided — DESCOPE.** No consumer needs the aspirational surface. Update
  `metering-plan.md` to mark the toolkit-level `RunConfig` and
  `Flow.run(config=RunConfig(...))` as *deferred / future API* and remove them
  from the "10/10" acceptance set. Zero code, zero runtime change. The core
  `RunConfig` (`controller/sinks/redactor/pricer/clock`) stays as-is.

### F7 · Batch: `unmetered_operations` report field never implemented
- **Verified.** Gating only `batch_submit` under enforcement (`_llm.py:954`) is
  **by-design** (status/results consume no metered budget). The genuine gap: the
  plan promises the report flags `unmetered_operations=["batch_submit"]` in
  metering-only mode (`:290-291, :427, :459`) and the string appears **nowhere**
  in `src/`; `BudgetReport` (`_report.py:32`) has no such field.
- **Fix.** (1) Accumulate unmetered-op names in the meter (a `set[str]` on the
  store/snapshot, appended when `batch_submit`/`batch_status`/`batch_results` run
  under a measure-only scope). (2) Add `unmetered_operations: tuple[str, ...]` to
  `BudgetReport` + `to_dict()` + `MeterSnapshot`. (3) Surface on
  `AgentResult.report`.
- **Severity.** P2. Observability gap, not a money-enforcement gap.

### F8 · `BudgetExceeded` lacks `.maximum` and `.to_dict()`
- **Verified.** `_exceptions.py:10` inherits `dimension/limit/current/attempted`
  from `AdmissionDenied` (`_admission.py:42-45`); `.limit` is present (as the cap
  **float**), `.maximum` and `.to_dict()` are absent. The plan requires all three
  (`:149, :381`). The review's "old API / backward-compat" framing is weak — this
  is a fresh branch with no shipped predecessor, so there is nothing to break.
- **Decided — DESCOPE.** No consumer depends on the old `.maximum`/`.to_dict()`
  shape (the sole consumer will adapt). Do **not** add legacy accessors. Update
  `metering-plan.md` to drop the "preserves `.limit`/`.maximum`/`.to_dict()`"
  requirement; `BudgetExceeded` keeps the clean `AdmissionDenied` surface
  (`dimension`/`limit`/`current`/`attempted`). If a serializable form is ever
  wanted, prefer `BudgetReport.to_dict()` (already exists) over reviving it here.

---

## Docs

### F9 · `safety.md` overstates the default `max_cost` as "hard"
- **Where.** `docs/safety.md:220` — "Enforcement is **hard, at the charge site**…
  the call never happens and nothing is charged." True only for **count/token
  caps** and `reserve="strict"`; the default soft `max_cost` (`reserve="none"`)
  can overshoot by the in-flight call(s) and denies at the *next* admit — as the
  same page concedes at `:217`.
- **Fix.** Scope the sentence: lead with "count and token caps are hard, pre-call;
  the default `max_cost` is soft (overshoot ≤ in-flight calls) unless
  `reserve='strict'`." Low-risk docs edit.

---

## Sequencing (suggested PRs)

1. **PR-1 (P1 guards).** F1 + F2 — the `except (AdmissionDenied,
   NotMeteredOperationError): raise` guards in `_builders.py` and the 6 `_llm.py`
   sites, plus the bespoke-step audit. One tight PR; highest value.
2. **PR-2 (P1 cost).** F3 — per-step `max_cost` fail-closed on unknown cost
   (+ the `PolicyDecision` decision on how to signal it).
3. **PR-3 (P2 cheap).** F5 pricer threading + F4 (`output_schema` in the hint,
   `ceil` division, media allowance).
4. **PR-4 (observability).** F7 `unmetered_operations` end-to-end.
5. **PR-5 (descope + plan sync).** F6 and F8 — no code; edit `metering-plan.md`
   to mark the toolkit `RunConfig`/`Flow.run(config=…)` deferred and drop the
   `BudgetExceeded.maximum/.to_dict()` requirement, so plan and code agree.
6. **Docs.** F9 — fold into PR-2 (same subsystem) or ship standalone.

## Acceptance ("10/10")

- [ ] F1, F2: no path converts `AdmissionDenied`/`NotMeteredOperationError` to a
      normal `Result` or a fallback; a regression test covers
      `fallback_on=(Exception,)` for complete + both stream paths.
- [ ] F3: per-step and run-level `max_cost` fail closed identically on unknown
      cost, signalled by a distinct `cost_unknown` `PolicyDecision`.
- [ ] F5: estimate and settle provably use the same pricer on the default path.
- [ ] F4: strict reserve is a real ceiling (output schema counted, round-up, media
      allowance).
- [ ] F7: `unmetered_operations` surfaces on `BudgetReport`/`AgentResult.report`.
- [ ] F6, F8: descoped — `metering-plan.md` updated so it no longer requires the
      toolkit `RunConfig`/`Flow.run(config=…)` or `BudgetExceeded.maximum/.to_dict()`.
- [ ] F9: docs no longer call the default `max_cost` unconditionally "hard".
- [ ] Full suite green; new regression tests for each P1/P2 above.
```
