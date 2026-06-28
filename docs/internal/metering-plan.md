# Metering & Budget — Architecture Plan (implementation contract)

> **Status:** design + contract frozen, not yet implemented. Implement against this.
> Branch: `feat/metering-clean` (clean from `main`). The previous experiment is on
> `feat/agent-budget` @ `2af0fc7` — **reference only, never merge**.

This rewrite replaces the escrow `Ledger` + per-strategy accounting + executor
reconciliation with a single source of truth: **a neutral meter in `core/`, a
budget controller in `toolkit/`.**

---

## 1. Governing principles

- **`core` measures; it does not decide budget.** Core has no concept of "budget".
- **`toolkit` decides budget**, via a controller injected into the meter.
- **The meter is the single source of truth.** `trace` / `Result` / report are **views**.
- **Agents do not accumulate cost/usage.** Strategies reason; the runtime measures.
- **Business budgets (user/team/org/monthly) live in the downstream app**, on top of
  the `UsageEvent` / projection. Never in the framework. The app resolves a monthly/DB
  budget into a per-run `BudgetPolicy` at run start.

Invariant: `reported(scope) ≡ enforced(scope) ≡ the projection for that scope`,
by construction — **one writer (`MeterStore`); read-models are derived, never parallel sums.**
The projection (counters/aggregates) is the authority; the event stream is optional audit.
`replay(events) == projection` holds **only when events are retained** (tests/debug); it is
not a runtime dependency.

---

## 2. Layer / ownership contract

| Layer | Responsibility |
|---|---|
| `core/_metering/` | Neutral mechanism: operation lifecycle, counters + per-span projections, injectable admission hook (no-op without a scope). Zero budget knowledge. |
| `core` (LLM/tools/pricing) | Charge sites: build `OperationFacts` → `open` → `mark_started` → execute → `settle`/`fail`, **per provider attempt**. Cost via `PricingRegistry`. |
| `toolkit/budget/` | Opinion: `BudgetPolicy`, reserve modes, fail-closed, `BudgetExceeded`, report, default pricer/**estimator**. Implements the core admission Protocol. |
| `toolkit/flow/` | Opens the run `MeterScope`, installs the controller, opens step spans, derives trace/Result/report from projections, applies `Policy.max_cost` at the step span. |
| `toolkit/agents/` | Compose flows. **No manual accounting.** |
| app | user/team/org/monthly budgets, on top of `UsageEvent`/projections. |

**`core` never imports `toolkit`** (verified: 0 refs). The admission *executes* at the
in-core charge site (it must, for hard caps) but is *supplied* by toolkit via an injected
hook typed by a core Protocol.

---

## 3. Module layout

```
core/_metering/
  _money.py        # Money (opaque; pico-USD int internally) + Money.zero()
  _cost.py         # Cost (one class: kind known|estimated|unknown + factories)
  _events.py       # UsageEvent, UsageSink (Protocol)
  _operation.py    # OperationFacts, MeterOperation (lease lifecycle + state machine)
  _admission.py    # AdmissionController (Protocol), AdmissionDecision, Reservation,
                   #   MeterSnapshot, AdmissionDenied (NEUTRAL core exception base)
  _store.py        # MeterStore (counters + per-span projections + optional events; clock)
  _scope.py        # MeterScope, current_meter/bind_meter, open_span/current_span_id
  __init__.py
core/
  _llm.py          # complete()/stream() — meter PER PROVIDER ATTEMPT (stream = kind="llm", mode="stream")
  _tools/_executor.py   # THE common metered+gated tool executor
  _pricing.py      # PricingRegistry, estimate_cost (mechanism; default table overridable)
  _response.py     # Usage (REUSED, not redefined), Response.cost/cost_money/cost_known
toolkit/budget/    # _policy, _controller (+estimator), _state (BudgetSnapshot), _report, _exceptions (BudgetExceeded)
toolkit/_runner.py # run_tools — STAYS in toolkit; MUST route through the common executor
toolkit/flow/_executor.py   # opens scope, installs controller, step spans, max_cost
toolkit/agents/flows/*.py   # zero accounting
```

Top-level re-export `from ai_arch_toolkit import BudgetPolicy` is OK; **ownership is `toolkit.budget`.**

---

## 4. Type contracts

```python
# _money.py — opaque; internal int pico-USD (1e-12 USD); exact arithmetic
class Money:
    @classmethod
    def zero(cls) -> Money: ...
    @classmethod
    def from_usd(cls, x: float | Decimal) -> Money: ...
    @classmethod
    def from_pico(cls, p: int) -> Money: ...          # rate_pico * tokens (the pricer path)
    def to_float(self) -> float: ...                  # display/compat only
    # __add__ / __sub__ / __mul__(int) / __lt__ / __le__ / __eq__ : exact ints. Never expose "pico".

# _cost.py — ONE class with a kind (NOT a 3-class union)
@dataclass(frozen=True, slots=True)
class Cost:
    kind: Literal["known", "estimated", "unknown"]
    amount: Money | None = None      # None iff unknown
    reason: str | None = None        # set iff unknown
    @classmethod
    def known(cls, m): ...
    @classmethod
    def estimated(cls, m): ...
    @classmethod
    def unknown(cls, reason): ...
    @property
    def is_known(self) -> bool: ...
    @staticmethod
    def merged(*costs) -> Cost: ...   # for ONE composite op only. The PROJECTION does NOT use this —
                                      # it keeps known-cost SUM + unknown count separately (see MeterSnapshot).
# Response.cost(float)/cost_known(bool)/cost_money(Money|None) are COMPAT VIEWS of one Cost.

# _operation.py
@dataclass(frozen=True, slots=True)
class OperationFacts:                 # CORE builds; pure FACTS; built AFTER middleware `before`
    kind: Literal["llm", "tool", "custom"]   # streaming = kind="llm", mode="stream" (never separate)
    parent_span_id: str               # the active span (run or step); the store assigns the op's own id
    count: int = 1                    # 1 per llm/tool call; custom -> 0 (does NOT touch call caps
                                      #   unless a custom controller opts in via metadata)
    mode: Literal["complete", "stream"] | None = None
    model: str | None = None
    declared_max_output_tokens: int | None = None     # the user's max_tokens (a FACT)
    content_size_hint: int | None = None              # text char count (a FACT)
    non_text_parts: int = 0                           # #images/docs — the estimator adds an allowance
    metadata: Mapping[str, Any] = field(default_factory=dict)
# NO estimate fields here. The controller's INJECTED, SYNC estimator turns facts -> estimate at admit.

class MeterOperation:                 # handle held by the charge site; delegates to MeterStore by op_id
    def mark_started(self) -> None: ...   # COMMIT the count
    def settle(self, *, usage, cost) -> None: ...   # add actuals; idempotent by op_id
    def fail(self) -> None: ...           # STARTED-then-errored: count stays
    def abort(self) -> None: ...          # PENDING only: full release, no count

# State machine + ACCOUNTING (write as tests). committed = real; out_* = reserved.
#   open(facts):   per reserve-mode -> out_{llm|tool}_calls += count;
#                  out_input/output_tokens, out_cost += worst-case  (only if reserve != NONE)
#   mark_started:  out_{llm|tool}_calls -= count;  committed {llm|tool}_calls += count   (PENDING->STARTED)
#   settle:        out_tokens/out_cost -= reserved;  committed usage += actual;
#                  committed cost += actual if Known else unknown_cost_count += 1         (STARTED->SETTLED)
#   fail:          out_tokens/out_cost -= reserved;  (count already committed)            (STARTED->FAILED)
#   abort:         out_{llm|tool}_calls -= count; out_tokens/out_cost -= reserved          (PENDING->ABORTED)
#   stream STARTED un-drained at scope close -> behave like fail + mark INCOMPLETE (count stays, cost Unknown)
# WHY count-on-start: a started-then-failed op that released its count would let a retry/fallback loop make
#   N real provider calls under a finite max_llm_calls. => caps bound PHYSICAL attempts (incl. retries and
#   failed fallbacks). Idempotency: settle(same op, same payload)=no-op; different payload=error;
#   settle/abort after fail=error; abort after start=error.

# _admission.py
class AdmissionDenied(Exception): ...  # NEUTRAL core base. toolkit's BudgetExceeded(AdmissionDenied) subclasses it.
                                       # core NEVER imports BudgetExceeded.

@dataclass(frozen=True, slots=True)
class Reservation:                     # per-dimension holds
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: Money = field(default_factory=Money.zero)

@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    admitted: bool
    reservation: Reservation = field(default_factory=Reservation)
    denial: AdmissionDenied | None = None   # set iff not admitted; the store raises it from open()

class AdmissionController(Protocol):
    def admit(self, snapshot: MeterSnapshot, facts: OperationFacts) -> AdmissionDecision: ...
# PURE: sync, no I/O, no await. The injected estimator is ALSO sync (no provider count_tokens call).

@dataclass(frozen=True, slots=True)
class MeterSnapshot:                   # what the controller sees; what snapshot()/for_span() return
    # committed (real):
    llm_calls: int; tool_calls: int                 # = STARTED physical attempts (committed at mark_started)
    input_tokens: int; output_tokens: int
    cache_read_tokens: int; cache_write_tokens: int  # settled actuals
    cost: Money                                      # SUM of KNOWN settled costs
    unknown_cost_count: int                          # settled ops with Unknown cost (for fail-closed + reports)
    # outstanding (reserved, not yet started/settled) — PER DIMENSION (prevents concurrent over-admit):
    out_llm_calls: int; out_tool_calls: int
    out_input_tokens: int; out_output_tokens: int
    out_cost: Money
    elapsed_s: float
# The controller compares caps against committed + outstanding (NOT committed alone).

# _store.py
class MeterStore:
    # Constructed with an INJECTABLE clock (so snapshot() computes elapsed_s; the store owns time).
    # ONE writer. self._lock: threading.Lock. counters O(1) + per-span incremental aggregates + optional events.
    def open(self, facts, controller) -> MeterOperation: ...
        # with self._lock: snap=snapshot(); d = controller.admit(snap, facts) if controller else ALLOW
        #   if not d.admitted: raise d.denial; else apply d.reservation -> PENDING op (assign op_id;
        #   operation span under facts.parent_span_id)
    def mark_started(self, op_id) -> None: ...
    def settle(self, op_id, *, usage, cost) -> None: ...   # under lock: mutate counters + walk parent chain;
                                                           #   build immutable UsageEvent. CALL SINKS OUTSIDE THE LOCK.
    def fail(self, op_id) -> None: ...
    def abort(self, op_id) -> None: ...
    def snapshot(self) -> MeterSnapshot: ...               # lock-guarded; returns immutable
    def for_span(self, span_id, *, descendants=True) -> MeterSnapshot: ...  # O(1) via incremental aggs; lock-guarded
# Span model: each span has span_id, parent_span_id, scope_type ∈ {run, step, operation}.
#   settle walks the parent chain updating each ancestor aggregate (O(depth)).

# _scope.py
@dataclass(slots=True)
class MeterScope:                      # bound to a ContextVar; what current_meter() returns
    store: MeterStore                  # holds the clock
    controller: AdmissionController | None    # injected by toolkit/flow; None => no enforcement
    def open(self, facts) -> MeterOperation: ...        # -> store.open(facts, self.controller)
    def open_span(self, scope_type, name) -> _SpanCtx: ...  # pushes a child span; binds current_span_id
def current_meter() -> MeterScope | None: ...
def bind_meter(scope) -> Token: ...    # nested flows REUSE the bound scope (no second meter); idempotent
def current_span_id() -> str | None: ...           # OperationFacts.parent_span_id = this

# toolkit/budget/_policy.py
@dataclass(frozen=True)
class BudgetPolicy:
    max_llm_calls / max_tool_calls: int | None
    max_input_tokens / max_output_tokens / max_total_tokens: int | None
    max_cost: float | Decimal | None   # user-facing; compiled to Money once at run start
    max_wall_s: float | None
    reserve: Reserve = Reserve.NONE    # cost/tokens knob; counts ALWAYS hard. default soft (post-hoc).
    unpriced: Unpriced = Unpriced.FAIL_CLOSED

# toolkit/budget/_controller.py — implements AdmissionController; holds policy + injected estimator + pricer
#   admit: estimate tokens (sync estimator) -> price -> apply policy (counts hard; cost/tokens per reserve;
#          wall-time via snapshot.elapsed_s; fail-closed if max_cost set & estimate Unknown & unpriced=FAIL_CLOSED)
#          -> ALLOW + Reservation | DENY + BudgetExceeded(AdmissionDenied)
# toolkit/budget/_state.py BudgetSnapshot = a DERIVED view of MeterSnapshot + caps (for the report); not a 2nd source.
```

---

## 4b. Lifecycle of one metered call

`MeterScope` (ContextVar) holds the `MeterStore` (state + lock + per-span aggregates + clock) and the
injected `AdmissionController | None`. Charge sites talk only to `current_meter()`. No scope ⇒ no-op.

```python
# ONE provider attempt — core/_llm.py, INSIDE the retry/fallback callable.
# (facts built AFTER middleware `before`, since middleware can alter messages/tools/kwargs.)
async def _metered_attempt(provider, req, *, kind, mode):
    meter = current_meter()
    if meter is None:
        return await provider.complete(req)              # unmetered (no active scope)
    facts = OperationFacts(kind=kind, mode=mode, parent_span_id=current_span_id(),
                           model=req.model, declared_max_output_tokens=req.max_tokens,
                           content_size_hint=text_chars(req), non_text_parts=count_media(req))
    op = meter.open(facts)            # lock: snapshot -> admit -> reserve | raise AdmissionDenied
    op.mark_started()                 # COMMIT the call count (a failure still counts)
    try:
        resp = await provider.complete(req)
    except BaseException:             # incl. CancelledError / Policy.timeout -> NO reservation leak
        op.fail(); raise
    op.settle(usage=resp.usage, cost=Cost.from_response(resp, pricer))   # idempotent
    return resp
```

- **Retry/fallback boundary:** `with_retry` / fallback wrap `_metered_attempt`, so **each attempt opens its
  own op** ⇒ retries and failed fallbacks each count. NEVER wrap the meter around the whole facade method.
- **stream():** same, BUT capture `op`/`meter` at **BUILD time** (the finalizer drains on another thread
  with the ContextVar unset). `mark_started` when the provider stream opens; `settle` in the finalizer
  closure; early/partial `__aexit__` with no final usage → `INCOMPLETE` (count stays, cost Unknown).
- **tools:** governance gates FIRST; only an executed tool opens an op (`mark_started` before `fn(...)`,
  `settle`/`fail` after). `run_tools` routes through THIS executor.
- **flow:** `bind_meter(MeterScope(store, controller))` at run start; `open_span` per step; nested flows
  **reuse the bound scope** and open child spans (no second source). Executor reads `for_span(step)` to
  apply `Policy.max_cost` and fill `Result.cost`/`.usage` (views).

---

## 5. Cravadas (resolved) decisions — checklist

- [ ] **`admit + reserve` atomic:** snapshot + `controller.admit` + apply, **all in one `with store._lock`** (no TOCTOU under parallel `gather`).
- [ ] **`threading.Lock`, NOT `asyncio.Lock`** (the stream finalizer settles from another OS thread). **Reads (`snapshot`/`for_span`) are also lock-guarded** and return immutable snapshots.
- [ ] **`UsageSink`s are called OUTSIDE the lock:** under lock mutate counters + build the immutable event; release; then call sinks. (Never run user sink code under the lock.)
- [ ] **Count commits on start; `abort` is PENDING-only.** `mark_started` transfers the call reservation → committed count; a started-then-failed op keeps its count (`fail()` releases only token/cost reservation). ⇒ caps bound **physical attempts** incl. retries/fallbacks (document it).
- [ ] **Meter wraps EACH provider attempt**, inside the retry/fallback callable — not the whole `complete()`.
- [ ] **Cleanup catches `BaseException`** (not just `Exception`) so `CancelledError`/timeout `op.fail()` before re-raise → no reservation leak.
- [ ] **`OperationFacts` built AFTER middleware `before`** (middleware can change messages/tools/kwargs).
- [ ] **`admit()` and the injected estimator are sync + pure** — no `await`, no I/O, no provider `count_tokens`. Monthly/DB budgets resolve to a per-run `BudgetPolicy` at run start.
- [ ] **Per-dimension outstanding** (`out_llm_calls`/`out_tool_calls`/`out_input_tokens`/`out_output_tokens`/`out_cost`); caps checked against **committed + outstanding**. (Generic `out_calls` reopens concurrent overspend.)
- [ ] **Sync bridge:** `_run_sync` `copy_context()` so in-flow `*_sync` inherit the meter; `_stream_sync` does NOT (hence build-time capture). **Rule:** sync inside an active scope is metered; outside, unmetered unless the caller binds a meter.
- [ ] **Streaming is `kind="llm"`/`mode="stream"`** — never a separate kind (so `max_llm_calls` counts it).
- [ ] **`run_tools()` routes through the common metered+gated executor** (today bypasses metering AND governance — a security hole).
- [ ] **`Policy.max_cost` applied by the EXECUTOR at the step span** (reads `for_span(step)`), not the step engine. Same mechanism as `BudgetPolicy.max_cost` at the run span — unify, don't deprecate.
- [ ] **`max_wall_s`:** the store (owning the clock) sets `snapshot.elapsed_s`; the controller checks it on each `admit`; the executor checks before/after each step. Does NOT interrupt in-flight calls (use `Policy.timeout`).
- [ ] **Reserve default = soft (`NONE`).** Counts always hard regardless. With `NONE`, cost/token caps are post-hoc; **overshoot ≤ number of concurrently in-flight ops** (≤1 only when sequential). `EXPECTED`/`WORST_CASE` opt-in for hard pre-call cost/token caps (token worst-case needs `declared_max_output_tokens`).
- [ ] **Token-cap convention:** `max_input_tokens` caps `input + cache_read + cache_write` (full context); `max_output_tokens` caps `output`; `max_total_tokens` all four. Cache rate differences are a *cost* concern. (Reconcile any `Response` total-token helper that excludes cache.)
- [ ] **Projection keeps cost certainty:** committed `cost: Money` (sum of KNOWN) + `unknown_cost_count` — do NOT collapse via `Cost.merged()` (reports + fail-closed need both).
- [ ] **`AdmissionDecision.denial` is a NEUTRAL `AdmissionDenied`** (core); toolkit fills a `BudgetExceeded(AdmissionDenied)`. Core never imports `BudgetExceeded`.
- [ ] **Events optional; counters/projections authoritative.** Default: counters retained, events NOT retained, streamed to sinks if present, `retain_events`/`max_events` opt-in.
- [ ] **`Money` opaque, pico-int internal;** `BudgetPolicy(max_cost=0.10)` at the API. **`max_cost: float | Decimal | None`.**
- [ ] **Per-span projections via incremental aggregates** (walk `parent_span_id` on settle), O(depth) write / O(1) read.
- [ ] **`Response.cost`/`cost_money`/`cost_known`** are compat views of one `Cost`.
- [ ] **`custom` kind → `count=0`:** does not affect `max_llm_calls`/`max_tool_calls`; contributes to cost/tokens only via its settled usage/cost.

---

## 6. Phased commit plan (strangler; suite green at each step)

1. `feat(core): neutral metering primitives` — all of `core/_metering/` **+ property-test oracle** using a **test-double `AdmissionController`** (e.g. `CapController(max_llm_calls=1)`), since the real one is toolkit (step 4).
2. `feat(core): meter LLM operations` — `complete()`/`stream()` charge sites, **per provider attempt** (build-time finalizer capture; `BaseException` cleanup).
3. `feat(core): meter tool operations` — `ToolGroup` **and `run_tools`** via the common metered+gated executor.
4. `feat(toolkit): run-level budget controller` — `toolkit/budget/` (policy, controller + sync estimator, BudgetExceeded, report, snapshot).
5. `feat(flow): bind meter scope + step spans` — executor opens `MeterScope`, installs controller, step spans, derives views. No reconciliation.
6. `feat(flow): apply Policy.max_cost from the step-span projection` — move the check out of `_step_engine` into the executor (same commit removes `_step_engine.py:89`).
7. `refactor(agents): derive usage/cost from flow metering` — delete the 9 accumulators.
8. `test+docs` — invariants/property tests; rewrite `docs/budgets.md`; `examples/38_budget.py`.

---

## 7. Invariants / property tests (oracle; step-1 uses a test-double controller)

- `committed ≤ cap` for every hard cap, any schedule.
- **Per-dimension:** concurrent ops cannot over-admit `max_llm_calls` vs `max_tool_calls` vs token caps (checked against committed + outstanding).
- `outstanding == 0` after the run (no leak) — including on `CancelledError`/timeout.
- `reported == projection`; `replay(events) == projection` only when events retained.
- Hard count caps exact under parallel `asyncio.gather` (incl. streams — `kind="llm"`).
- **Started-then-failed keeps its count; each retry/fallback is its own op** (no unbounded attempts under a finite call cap).
- Operation idempotency: double-settle same payload = no-op; different payload = error.
- `mark_started` transfers reservation→committed (no double-count of in-flight calls).
- Unpriced under a cost cap → fail-closed (unless `policy.unpriced == ALLOW`); `unknown_cost_count` surfaces it.
- Stream abandoned → `INCOMPLETE`, never silently dropped.
- `STRATEGIES` matrix (`usage == projection` per strategy × {no-breach, breach}) — **derive the list from the registry**.

---

## 8. Keep / Delete / Move

**Keep:** ContextVar run-scope binding · reserve→settle for hard counts · the 3 charge-site
seams · typed `Cost` · `Usage` (reused) · provider cache-token subtraction · the invariant
matrix (registry-derived) · ambient (no signature change) · zero business concepts.

**Delete:** per-strategy accumulators (9 flows) · reconciliation
(`_unrecorded_spend`/`_append_budget_reconciliation`/`_budget_owns_ledger`/`budget_reconcile`) ·
public core `Ledger` name (→ `Meter`) · `BudgetPolicy`/`BudgetExceeded` in core · unwired-vs-live duplication.

**Move:** `BudgetPolicy`/`BudgetExceeded`/enforcement → `toolkit/budget/` · estimation heuristics →
toolkit/controller · pricing default table stays in core **but fully overridable**.

---

## 9. Footguns to NOT reintroduce

- Unmetered tool surface. **Must-fix now:** `run_tools` (security). **Document-as-unmetered (not v1):** direct provider calls, batch, manual facade bypass, out-of-flow sync.
- Releasing the call count on a started-then-failed op (retries/fallbacks overspend the cap).
- Metering the whole `complete()` instead of each attempt (retries collapse to one call).
- `except Exception` instead of `except BaseException` (CancelledError leaks the reservation).
- Generic `out_calls`/single `out_cost` (concurrent caps overspend).
- Calling `UsageSink`s under the lock.
- Estimates as fields on the core-built `OperationFacts`; an estimator that does I/O.
- `Ledger`/`BudgetExceeded` names in core; `asyncio.Lock`; reading `current_meter()` at stream-finalize.
- `kind="stream"` separate from `"llm"`. `float` for internal money. Soft cap presented as hard.

---

## 10. Deferred (designed-for, with triggers)

- **Sub-caps per sub-agent:** span-scoped admission is modelled now; expose the public API when asked.
- **OTel exporter:** a `UsageSink`; design the seam, defer impl (event stays OTel-agnostic; map `gen_ai.*` in the toolkit adapter; decide cache-in-input vs the target semconv).
- **Persistence / resumable budgets:** event log with `op_id` is serializable; defer impl.
- **Event retention:** counters authoritative; events sink-and-forget; `retain_events`/`max_events` opt-in.
