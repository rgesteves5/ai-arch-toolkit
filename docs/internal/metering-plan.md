# Metering & Budget — Architecture Plan (implementation contract)

> **Status:** design + contract frozen, not yet implemented. Implement against this.
> Branch: `feat/metering-clean` (clean from `main`). Reference snapshot:
> `feat/agent-budget` @ `2af0fc7` — **never merge**.
> This revision integrates a 7-reviewer audit (5 internal persona agents + Codex + ChatGPT).

Replaces the escrow `Ledger` + per-strategy accounting + executor reconciliation with a
single source of truth: **a neutral meter in `core/`, a budget controller in `toolkit/`.**

---

## 1. Governing principles + the three modes

- **`core` measures; `toolkit` decides budget** (via an injected controller). The meter is
  the single source of truth; `trace`/`Result`/report are views. Agents don't accumulate.
- **Business budgets (user/team/org/monthly) live in the downstream app**, on top of
  `UsageEvent`/projection — resolved to a per-run `BudgetPolicy` at run start.

**Three modes (this is the answer to "use without budget?" — yes):**
1. **No `MeterScope`** (a bare `llm.complete()` outside a flow) → nothing measured, nothing blocked.
2. **`MeterScope` + `controller=None`** → measures usage/cost/calls, **no enforcement**. This is
   the **default for every Flow/Agent run** (so `result.usage`/`.cost`/report always populate).
3. **`MeterScope` + `BudgetAdmissionController`** → measures **and** enforces.

Invariant: `reported(scope) ≡ enforced(scope) ≡ the projection for that scope`, by construction —
one writer; read-models derived, never parallel sums. Counters/aggregates are the authority;
events are optional audit; `replay(events) == projection` holds only when events are retained.

---

## 2. Layer / ownership contract

| Layer | Responsibility |
|---|---|
| `core/_metering/` | Neutral mechanism: operation lifecycle, counters + per-span projections, injectable admission hook (no-op without a controller). Zero budget knowledge. |
| `core` (LLM/tools/pricing) | Charge sites: build `OperationFacts` → `open` → `mark_started` → execute → `settle`/`fail`, **per provider attempt**. Cost via `PricingRegistry`. |
| `toolkit/budget/` | Opinion: `BudgetPolicy`, reserve modes, fail-closed, `BudgetExceeded`, report, default pricer/estimator. Implements the core admission Protocol. |
| `toolkit/flow/` | **Always opens a `MeterScope` per run** (controller=None if no budget). Opens step spans, derives views, applies `Policy.max_cost` at the step span, **catches `AdmissionDenied` and records `policy_decision`** (preserves the cooperative non-raising flow contract). |
| `toolkit/agents/` | Compose flows. **No manual accounting.** |
| app | user/team/org/monthly budgets on top of `UsageEvent`/projection. |

**`core` never imports `toolkit`.** The controller (toolkit) is injected; **it runs OUTSIDE the
store lock** (see §4 `open`) and is typed by a core Protocol.

---

## 3. Module layout

```
core/_metering/
  _money.py     # Money (opaque pico-USD int) + Money.zero()
  _cost.py      # Cost (one class: kind known|estimated|unknown + factories)
  _events.py    # UsageEvent (status: settled|failed|incomplete|aborted), UsageSink (Protocol)
  _operation.py # OperationFacts, MeterOperation (lease lifecycle + state machine)
  _admission.py # AdmissionController (Protocol), AdmissionDecision, Reservation, MeterSnapshot,
                #   AdmissionDenied (NEUTRAL core exception), NotMeteredOperationError
  _store.py     # MeterStore (counters + per-span projections + optional events; injectable clock)
  _scope.py     # MeterScope (always has run_span), current_meter/bind_meter, open_span/current_span_id
  __init__.py
core/
  _llm.py        # complete/stream meter PER PROVIDER ATTEMPT; batch fail-closed under active scope
  _tools/_executor.py   # THE common metered+gated tool executor
  _pricing.py    # PricingRegistry; price(facts,usage)->Cost (mechanism; table overridable)
  _response.py   # Usage (REUSED); Response.cost(float, legacy) + cost_money(Money) + cost_known(property)
toolkit/budget/  # _policy, _controller(+estimator), _state(BudgetReport), _exceptions(BudgetExceeded)
toolkit/_runner.py          # run_tools — routes through the common metered+gated executor
toolkit/flow/_executor.py   # opens scope, RunConfig, step spans, max_cost, catches AdmissionDenied
toolkit/agents/flows/*.py   # zero accounting
```

Public surface: only `BudgetPolicy`, `BudgetReport`, `BudgetExceeded`, `Reserve`, `Unpriced`,
`RunConfig` (top-level) + the extension contracts (`AdmissionController`, `UsageSink`, `Cost`,
`Money`, `UsageEvent`, `AdmissionDenied`) under `ai_arch_toolkit.core`. The meter mechanism
(`MeterStore`/`MeterScope`/`MeterOperation`/`current_meter`/`bind_meter`) is `_`-internal.

---

## 4. Type contracts

```python
# _money.py — opaque int pico-USD; exact. from_usd(Decimal(str(x)), ROUND_HALF_EVEN); from_pico(rate*tokens).
@dataclass(frozen=True, slots=True)
class Money:                              # single field _pico: int -> free __eq__/__hash__/immutable
    @classmethod def zero(cls): ...
    @classmethod def from_usd(cls, x: float | Decimal): ...
    @classmethod def from_pico(cls, p: int): ...
    def to_float(self) -> float: ...      # display/compat only
    # __add__/__sub__/__mul__(int)/__lt__/__le__ exact ints.

# _cost.py — ONE class with a kind (NOT a union)
@dataclass(frozen=True, slots=True, kw_only=True)
class Cost:
    kind: Literal["known", "estimated", "unknown"]
    amount: Money | None = None          # None iff unknown
    reason: str | None = None            # set iff unknown
    @classmethod def known/estimated/unknown(...): ...
    @property def is_known(self) -> bool: ...
    @staticmethod def merged(*costs) -> Cost: ...   # composite of ONE op only; the projection never uses it.
# Pricing lives in _pricing.py: PricingRegistry.price(facts, usage) -> Cost (NOT Cost.from_response).
# Response.cost(float) / cost_known(property) / cost_money(Money|None) are COMPAT VIEWS of one Cost;
# cost stays a constructable float field (~20 call sites do `response.cost or 0.0`).

# _operation.py
@dataclass(frozen=True, slots=True, kw_only=True)
class OperationFacts:                    # CORE builds AFTER middleware `before`; pure FACTS, NO estimates
    kind: Literal["llm", "tool", "custom"]   # streaming = kind="llm", mode="stream"
    parent_span_id: str                  # current_span_id() (defaults to run_span_id if None)
    count: int = 1                       # 1 per llm/tool; custom -> 0 (never touches call caps)
    mode: Literal["complete", "stream"] | None = None
    model: str | None = None
    declared_max_output_tokens: int | None = None
    content_size_hint: int | None = None # text char count (FACT)
    non_text_parts: int = 0              # #images/docs (for the estimator's allowance)
    metadata: Mapping[str, Any] = field(default_factory=dict)   # LOW-CARDINALITY, NON-SENSITIVE only

class MeterOperation:                    # handle; delegates to MeterStore by op_id (handle is stateless)
    def mark_started(self): ...          # transfer base call reservation -> committed
    def settle(self, *, usage, cost): ...# add actuals; idempotent by op_id
    def fail(self): ...                  # STARTED-then-errored: count stays; no-op if already terminal
    def abort(self): ...                 # PENDING only: full release, no count

# State machine + accounting (states: PENDING, STARTED, SETTLED, FAILED, ABORTED, INCOMPLETE).
#   open:        store applies BASE call reservation (out_{llm|tool}_calls += count) + controller's
#                token/cost reservation (only if reserve != NONE).
#   mark_started:out_{llm|tool}_calls -= count; committed {llm|tool}_calls += count      (PENDING->STARTED)
#   settle:      release token/cost reservation; committed usage += actual;
#                committed cost += amount if Known else unknown_cost_count += 1           (STARTED->SETTLED)
#   fail:        release token/cost reservation; count stays; if Unknown cost -> unknown_cost_count += 1
#                (STARTED->FAILED). fail() on an already-terminal op is a NO-OP (cleanup must never raise).
#   abort:       release ALL incl. base count                                            (PENDING->ABORTED)
#   scope close: any still-STARTED op (e.g. un-drained stream) -> force fail + INCOMPLETE (count stays,
#                cost Unknown). EVERY transition updates the ancestor span aggregates (not just settle).
# Idempotency: settle(same op,same payload)=no-op; different payload=warn+keep-first (NOT raise);
#   abort after start=error. WHY count-on-start: bounds PHYSICAL attempts incl. retries/failed fallbacks.

# _admission.py
class AdmissionDenied(Exception):        # NEUTRAL core base, carries dimension/limit/current/attempted.
    ...                                  # toolkit BudgetExceeded(AdmissionDenied) preserves .limit/.maximum/.to_dict().
class NotMeteredOperationError(Exception): ...   # raised by batch/etc. inside an active enforcing scope.

@dataclass(frozen=True, slots=True, kw_only=True)
class Reservation:                       # the CONTROLLER's optional token/cost holds (NOT call counts)
    input_tokens: int = 0
    output_tokens: int = 0
    cost: Money = field(default_factory=Money.zero)
# The STORE always adds the base llm/tool call reservation from facts.count. Final applied =
# base_count_reservation(facts) + decision.reservation.

@dataclass(frozen=True, slots=True, kw_only=True)
class AdmissionDecision:
    admitted: bool
    reservation: Reservation = field(default_factory=Reservation)
    limits: ResourceLimits | None = None # the caps the store re-validates under the lock (TOCTOU)
    denial: AdmissionDenied | None = None

class AdmissionController(Protocol):
    def admit(self, snapshot: MeterSnapshot, facts: OperationFacts) -> AdmissionDecision: ...
# PURE: sync, no I/O, no await, no provider count_tokens. The injected estimator is sync too.
# Hard caps: the controller decides using snapshot.{committed+outstanding} + facts.count.

@dataclass(frozen=True, slots=True, kw_only=True)
class MeterSnapshot:                      # immutable; what snapshot()/for_span()/totals() return
    llm_calls: int; tool_calls: int                       # committed = STARTED physical attempts
    input_tokens: int; output_tokens: int                 # settled actuals
    cache_read_tokens: int; cache_write_tokens: int
    cost: Money                                           # SUM of KNOWN settled costs
    unknown_cost_count: int                               # settled/failed ops with Unknown cost
    out_llm_calls: int; out_tool_calls: int               # outstanding, PER DIMENSION
    out_input_tokens: int; out_output_tokens: int
    out_cost: Money
    elapsed_s: float                                      # store owns the (monotonic) clock
    @property def total_tokens(self) -> int: ...          # input+output+cache_read+cache_write
    @property def out_total_tokens(self) -> int: ...

# _store.py — ONE writer; threading.Lock (the stream finalizer settles from another OS thread).
class MeterStore:
    # open() — admit runs OUTSIDE the lock; only arithmetic + re-validate run under the short lock:
    def open(self, facts, controller) -> MeterOperation:
        snap = self.snapshot()                            # brief lock; immutable
        if controller is None:
            decision = AdmissionDecision(admitted=True)
        else:
            decision = controller.admit(snap, facts)      # PURE, NO LOCK (no foreign code under lock)
            if not decision.admitted: raise decision.denial
        applied = base_count_reservation(facts) + decision.reservation
        with self._lock:                                  # arithmetic-only critical section
            if decision.limits and exceeds(self._committed_plus_outstanding(), applied, decision.limits):
                raise make_denial(...)                    # TOCTOU recheck (snapshot may be stale)
            return self._apply_pending(facts, applied)    # assign op_id + operation span under parent_span_id
    def mark_started(op_id), settle(op_id,*,usage,cost), fail(op_id), abort(op_id)  # each: short lock
    def snapshot() -> MeterSnapshot ; def for_span(span_id,*,descendants=True) -> MeterSnapshot  # lock-guarded
    # settle/fail build an immutable UsageEvent UNDER lock; UsageSinks are called AFTER releasing the lock,
    #   each wrapped (sink_error_policy default = log/ignore). Span aggregates updated on EVERY transition.

# _scope.py
@dataclass(slots=True)
class MeterScope:                         # ContextVar-bound; ALWAYS creates a run_span at construction
    store: MeterStore                     # holds the clock
    controller: AdmissionController | None # None => measure-only
    def open(self, facts) -> MeterOperation: ...      # store.open(facts, self.controller)
    def open_span(self, scope_type, name): ...        # child span CM; binds current_span_id
def current_meter() -> MeterScope | None
def bind_meter(scope) -> Token            # nested flows REUSE the bound scope (idempotent); try/finally reset
def current_span_id() -> str | None       # OperationFacts.parent_span_id = this or scope.run_span_id

# toolkit/budget/_policy.py
@dataclass(frozen=True, slots=True, kw_only=True)
class BudgetPolicy:
    max_llm_calls / max_tool_calls: int | None = None
    max_input_tokens / max_output_tokens / max_total_tokens: int | None = None
    max_cost: float | Decimal | None = None    # compiled to Money once at run start
    max_wall_s: float | None = None            # alias max_wall_time -> deprecation warning
    reserve: Reserve = Reserve.NONE            # counts ALWAYS hard; cost/tokens soft by default
    unpriced: Unpriced = Unpriced.FAIL_CLOSED  # auto-relaxed to ALLOW for loopback/local models
    allow_unmetered_batch: bool = False
# Estimator/Pricer are typed seams: Estimator = Callable[[OperationFacts], Reservation] (sync);
#   Pricer.price(facts, usage) -> Cost. BudgetAdmissionController(policy, *, estimator=..., pricer=...).
```

---

## 4b. Lifecycle of one metered call

`MeterScope` (ContextVar) holds the `MeterStore` + injected `controller | None`. Flow/Agent runs
always bind a scope; bare `complete()` outside a flow has none (mode 1).

```python
# ONE provider attempt — core/_llm.py, INSIDE the retry/fallback callable.
meter = current_meter()
if meter is None:
    return await provider.complete(req)              # unmetered (mode 1)
facts = OperationFacts(kind="llm", mode="complete", parent_span_id=current_span_id(),
                       model=req.model, declared_max_output_tokens=req.max_tokens,
                       content_size_hint=text_chars(req), non_text_parts=count_media(req))  # AFTER middleware before
op = meter.open(facts)            # admit OUTSIDE lock; base+extra reservation applied under short lock | raise AdmissionDenied
op.mark_started()                 # COMMIT the call count (a failure still counts)
try:
    resp = await provider.complete(req)
except BaseException:             # incl. CancelledError/timeout -> NO reservation leak
    op.fail(); raise
op.settle(usage=resp.usage, cost=pricer.price(facts, resp.usage))   # settle BEFORE middleware `after`
return resp
```

- **Retry/fallback:** `with_retry`/fallback wrap THIS, so each attempt opens its own op ⇒ retries
  and failed fallbacks each count. `AdmissionDenied` is **terminal** — never retried, never fellback,
  even under `fallback_on=(Exception,)` (it must escape).
- **Middleware:** facts built after `before`; `settle` happens **before** `after` (a raising `after`
  does not turn a successful provider call into `fail`).
- **stream / stream_events (+ _sync):** capture `op`/`meter` at BUILD time; `mark_started` when the
  provider stream opens; `settle` in the finalizer closure; un-drained at scope close → `INCOMPLETE`.
- **tools:** governance gates FIRST (blocked/dry-run/approval-denied do NOT open an op, do NOT count);
  only an executed tool opens an op (`mark_started` before `fn`, `settle`/`fail` after). `run_tools`
  routes through this executor.
- **batch (`batch_*`):** **not metered in v1.** Inside an active enforcing scope it raises
  `NotMeteredOperationError` unless `allow_unmetered_batch=True` (no silent budget bypass).
- **server tools (`web_search`/`code_execution`):** run provider-side; counted only as part of the
  LLM call's cost/usage (if the provider reports it), NEVER toward `max_tool_calls`; if a cost cap is
  set and the server-tool cost is Unknown + `unpriced=FAIL_CLOSED`, fail-closed.
- **flow:** `bind_meter(MeterScope(store, controller_or_None))` at run start (default, even without
  budget); `open_span` per step; nested flows reuse the bound scope; executor reads `for_span(step)`
  for `Policy.max_cost` and fills `Result.cost`/`.usage`.

---

## 5. Cravadas (resolved) decisions — checklist

- [ ] **`open()`: store owns the BASE call reservation** from `facts.count` (even when `controller=None`); the controller decides using `snapshot.{committed+outstanding}+facts.count` and may ADD token/cost reservations. `mark_started` transfers the base count to committed. **(fixes the negative-outstanding bug.)**
- [ ] **`controller.admit` runs OUTSIDE the store lock** (pure); the lock wraps only arithmetic + the TOCTOU re-validate + apply. **Never run injected code (admit/estimator/pricer/sink) under the lock** (deadlock/loop-stall/priority-inversion).
- [ ] **`threading.Lock`, NOT `asyncio.Lock`** (stream finalizer settles from another OS thread); reads also lock-guarded, return immutable.
- [ ] **`UsageSink`s called OUTSIDE the lock**, each wrapped; `sink_error_policy` default = `log`/`ignore` (a slow/raising sink must not break the call or stall the meter).
- [ ] **`AdmissionDenied` is terminal** — never retried/fellback; carries dimension/limit/current/attempted; toolkit `BudgetExceeded` subclasses it.
- [ ] **Count commits on start; `fail` keeps count; `abort` PENDING-only; `fail` on a terminal op is a no-op.** Meter wraps EACH provider attempt. Cleanup catches `BaseException`.
- [ ] **EVERY transition (open/mark_started/settle/fail/abort/incomplete) updates ancestor span aggregates**, not just settle. Per-span incremental, O(depth).
- [ ] **`UsageEvent` carries op/span ids, kind, mode, model, usage, cost, cost_kind, status ∈ {settled,failed,incomplete,aborted}, timestamps** — so `replay(events)==projection` covers all states. **No raw prompts/tool-args/secrets/PII in events or `metadata`** by default.
- [ ] **`MeterScope` always has a `run_span_id`**; if `current_span_id()` is `None`, facts default to it.
- [ ] **Memory:** fold a settled/failed op's aggregate into its parent step-span and **drop the per-op span node** (keep only live op-ids for idempotency); `max_events` is a bounded ring buffer; default retains nothing. (Bounds O(ops) growth in LATS/cyclic.)
- [ ] **Per-dimension outstanding** (`out_llm_calls`/`out_tool_calls`/`out_input_tokens`/`out_output_tokens`/`out_cost`); caps checked against committed+outstanding.
- [ ] **Sync bridge:** `_run_sync` `copy_context()` (in-flow `*_sync` inherit the meter); `_stream_sync` does NOT (build-time capture). Sync inside an active scope is metered; outside, unmetered.
- [ ] **`Policy.max_cost` applied by the EXECUTOR at the step span** (reads `for_span(step)`); same mechanism as run-level, unified.
- [ ] **`max_wall_s`** from `snapshot.elapsed_s` (monotonic clock); checked at admit + step boundaries; does NOT interrupt in-flight calls (use `Policy.timeout`).
- [ ] **Reserve default = soft (`NONE`).** Counts always hard. With `NONE`, cost/token caps are post-hoc; **overshoot ≤ number of concurrently in-flight ops** (≤1 only sequential). `EXPECTED`/`WORST_CASE` opt-in (token worst-case needs `declared_max_output_tokens`).
- [ ] **Fail-closed at SETTLE too:** under `max_cost`+`FAIL_CLOSED`, a settled/failed Unknown cost is cap-relevant (charge a configured worst-case or breach), not merely a counter (closes the unpriced-spend-escapes-the-cost-cap hole). `unpriced` auto-relaxes to `ALLOW` for loopback/local models.
- [ ] **Token convention:** `max_input_tokens` = input+cache_read+cache_write; `max_total_tokens` = all four. `Response.tokens` stays `input+output` (do not redefine a public property); add `Usage.total` if needed.
- [ ] **`count_tokens` (provider I/O)** is never used by the sync estimator; it is unmetered/out-of-budget by itself.
- [ ] **Reasoning/thinking tokens** are included in `Usage.output_tokens` (or `Usage` gains `reasoning_tokens`) so reasoning models price correctly.
- [ ] **`Money` opaque pico-int, frozen+hashable;** `BudgetPolicy(max_cost=0.10)` at the API.
- [ ] **`Response.cost` (float) + `cost_money` (Money) + `cost_known` (property)** are compat views of one `Cost`; all public cost surfaces (`FlowResult.total_cost`, `AgentResult.cost`, `Result.cost`) stay `float`.

---

## 6. Coverage / Non-Coverage

| Surface | v1 |
|---|---|
| Flow/Agent runs, nested/parallel-DAG/cyclic, step spans | ✅ metered (scope per run; reuse for nested) |
| `LLM.complete`/`complete_sync` | ✅ (sync via `copy_context`) |
| `stream`/`stream_events` + `_sync` | ✅ (build-time capture; un-drained→INCOMPLETE) |
| 4 providers, structured output, fallbacks, retries, middleware | ✅ (per attempt; provider-agnostic `Usage`) |
| `ToolGroup` + `run_tools` | ✅ (governance gates first; only executed tools count) |
| **`batch_*`** | ⛔ unmetered → `NotMeteredOperationError` inside an enforcing scope unless `allow_unmetered_batch` |
| **server tools** (`web_search`/`code_execution`) | ⚠️ counted only as LLM cost (if reported); never toward `max_tool_calls`; fail-closed if Unknown under a cost cap |
| **`count_tokens`** | ⛔ out of budget (provider I/O); not used by the estimator |
| **direct provider SDK / manual facade bypass / `fn(**args)` direct** | ⛔ unmetered by design (can't prevent without a sandbox); documented boundary |
| **out-of-flow bare `complete()`** | ⛔ unmetered (no scope); documented — there is no `LLM(budget=...)` in v1 |

---

## 7. API / UX (metering default, budget opt-in)

```python
result = flow.run(input)                                   # measures usage/cost; no enforcement
result = flow.run(input, budget_policy=BudgetPolicy(max_cost=0.10))   # measures + enforces
flow = Flow(..., budget_policy=BudgetPolicy(max_llm_calls=5)); flow.run(input)
result = flow.run(input, config=RunConfig(budget_policy=..., retain_meter_events=True, usage_sinks=[...]))
```

- **Metering is default for Flow/Agent runs; budget is opt-in.** No `flow.with_metering()` as the
  primary path (a fluent `with_budget_policy` may exist as convenience, not the main API).
- Document loudly: call caps are **hard**; cost/token caps are **soft by default** (overshoot ≤
  in-flight count under parallelism — set `reserve=WORST_CASE` for a hard pre-call ceiling); retries
  and failed fallbacks each count; `batch`/direct-provider/out-of-flow are unmetered; local/unpriced
  models under a cost cap (the `BudgetExceeded` message names the cause + the `unpriced=ALLOW` fix).

---

## 8. Migration / breaking changes (0.1.0.dev0)

- `BudgetPolicy`/`BudgetExceeded` move to `toolkit/budget`; re-exported top-level. **`from ai_arch_toolkit.core import BudgetPolicy` is removed** (the core→toolkit re-export would violate the layering rule) — documented break.
- **`BudgetState` removed** → `BudgetReport` (mapping table in the migration note).
- `BudgetExceeded` re-parents to `AdmissionDenied` but **keeps `.limit`/`.maximum`/`.to_dict()`**.
- `max_wall_time` → `max_wall_s` with a deprecated alias (warns).
- Budget breach stays **non-raising at the flow level** (executor catches `AdmissionDenied`, returns `FlowResult` with `policy_decision="budget_exceeded"`) — current behavior preserved.
- CHANGELOG "Breaking" block + a "Migrating budgets" doc section.

---

## 9. Phased commit plan (strangler; green at each step)

1. `feat(core): neutral metering primitives` — all of `core/_metering/` **+ property-test oracle** with a **test-double `CapController`** (real controller is toolkit, step 4).
2. `feat(core): meter LLM operations` — `complete`/`stream` per attempt (build-time capture, `BaseException` cleanup, settle-before-`after`); **batch fail-closed**.
3. `feat(core): meter tool operations` — `ToolGroup` + `run_tools` via the common executor.
4. `feat(toolkit): run-level budget controller` — `toolkit/budget/` (+ typed estimator/pricer seams).
5. `feat(flow): scope-per-run + step spans` — executor always opens scope (controller=None w/o budget), catches `AdmissionDenied`, RunConfig, derives views. No reconciliation.
6. `feat(flow): Policy.max_cost from the step-span projection` (moves the check out of `_step_engine`).
7. `refactor(agents): derive usage/cost from metering` — delete the 9 accumulators.
8. `test+docs` — full test matrix; `docs/budgets.md` + the 4 budget-referencing docs; migration note.

---

## 10. Invariants / test matrix (oracle uses a test-double controller in step 1)

State/accounting: `committed ≤ cap` any schedule · per-dimension concurrent over-admit impossible ·
`outstanding == 0` after run incl. on `CancelledError`/timeout · `mark_started` transfers
reservation→committed (no double-count) · `fail` keeps count, `abort` PENDING-only, `fail`-terminal=no-op ·
double-settle same payload=no-op, different=warn-keep-first · started-then-failed keeps count ·
each retry/fallback its own op · `AdmissionDenied` never retried/fellback · `controller=None` base-reserves
counts (no negative outstanding) · `replay(events)==projection` over {settled,failed,incomplete,aborted}.

Surfaces: `complete`/`complete_sync` · `stream`/`stream_sync`/`stream_events`/`stream_events_sync` ·
parallel DAG + parallel tools cannot over-admit (suspending provider double + `threading.Barrier`;
cross-thread store test for the finalizer) · stream abandoned→INCOMPLETE (`__aexit__` path; GC path
out-of-scope) · cross-thread stream-finalizer settle lands (and a test that fails if `current_meter()`
is read at finalize) · `batch` under enforcing scope raises `NotMeteredOperationError` · server tools
don't increment `max_tool_calls` · dangerous/approval-denied/dry-run tools don't execute/count ·
nested flow reuses scope (no double-count; `Σ for_span(child)==for_span(parent)`) · `complete_sync`
inside flow inherits meter; outside unmetered · tool-that-calls-LLM is metered when context propagates ·
middleware `after` raising doesn't undo `settle`; facts built after `before` (×3 paths) · `UsageSink`
raise/slow doesn't break the call · `current_span_id None`→run span · metering-only mode (controller=None)
measures without blocking · fail-closed 2×2×2 (max_cost×{Known/Unknown}×{FAIL_CLOSED/ALLOW}) incl. at settle ·
monotonic-clock/backward-clock guard · `Money` round-trip + view-equality (float==to_float).

---

## 11. Keep / Delete / Move · 12. Footguns

**Keep:** ContextVar binding · reserve→settle for hard counts · 3 charge-site seams · typed `Cost` ·
`Usage` reused · cache-token subtraction · registry-derived strategy matrix · ambient (no signature
change) · zero business concepts. **Delete:** 9 accumulators · reconciliation machinery · public core
`Ledger` name · `BudgetPolicy`/`BudgetExceeded` in core · unwired duplication. **Move:** budget policy
→ toolkit · estimation heuristics → controller · pricing default table stays in core, overridable.

**Footguns to NOT reintroduce:** controller/estimator/pricer/sink under the lock · releasing a
started-then-failed op's count · metering the whole `complete()` instead of each attempt · `except
Exception` (CancelledError leaks) · generic `out_calls`/single `out_cost` · `AdmissionDenied` caught
by retry/fallback · silent batch bypass · sink-under-lock or sink-breaks-call · raw prompts/secrets in
events/metadata · `current_meter()` at stream-finalize · `kind="stream"` · `float` internal money ·
soft cap presented as hard · `Cost.from_response` (pricing belongs in `_pricing`) · negative outstanding
from a missing base reservation · `BudgetExceeded`/`Ledger` names in core · `asyncio.Lock`.

---

## 13. Decided product choices (override if you disagree)

1. **Breach is cooperative at the flow level** — charge site raises `AdmissionDenied`; the flow executor
   catches it and returns a `FlowResult` with `policy_decision="budget_exceeded"` (preserves today's
   non-raising contract). `AdmissionDenied` is terminal (no retry/fallback).
2. **Metering default for Flow/Agent; budget opt-in** via `run(budget_policy=)` / `Flow(budget_policy=)` /
   `RunConfig`. No `with_metering()` as the primary API.
3. **`reserve=NONE` (soft) default** — documented loudly; `EXPECTED`/`WORST_CASE` opt-in.
4. **`unpriced=FAIL_CLOSED` default but auto-`ALLOW` for loopback/local** models (so local-model users
   aren't blocked); cloud-unpriced under a cost cap fails closed with an actionable message.
5. **Batch fail-closed** inside an enforcing scope unless `allow_unmetered_batch=True`.

## 14. Deferred (with triggers)

Sub-caps per sub-agent (modelled now, API later) · OTel exporter (a `UsageSink`) · persistence/resumable
budgets · full batch metering (reserve N at submit, settle per item — needs a batch registry) · event
retention beyond the bounded buffer.
