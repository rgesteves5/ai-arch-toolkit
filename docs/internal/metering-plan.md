# Metering & Budget — Architecture Plan (implementation contract)

> **Status:** design frozen, not yet implemented. This file is the contract we
> implement against. Branch: `feat/metering-clean` (clean from `main`).
> The previous experiment is preserved as a reference snapshot on
> `feat/agent-budget` @ `2af0fc7` — **reference only, never merge** (it puts
> budget policy in `core/` and carries fix-on-fix bugs).

This rewrite replaces the escrow `Ledger` + per-strategy accounting + executor
reconciliation with a single source of truth: **a neutral meter in `core/`, a
budget controller in `toolkit/`.**

---

## 1. Governing principles

- **`core` measures; it does not decide budget.** Core has no concept of "budget".
- **`toolkit` decides budget**, via a controller injected into the meter.
- **The meter is the single source of truth.** `trace` / `Result` / report are
  **views (projections)** of it.
- **Agents do not accumulate cost/usage.** Strategies reason; the runtime measures.
- **Business budgets (user / team / org / monthly) live in the downstream app**,
  built on top of the `UsageEvent` / projection the framework exposes. They are
  **never** in the framework. The app resolves a monthly/DB budget into a
  per-run `BudgetPolicy` at run start.

Invariant: `reported(scope) ≡ enforced(scope) ≡ the projection for that scope`,
by construction — **one writer (`MeterStore`), many derived read-models, never
parallel sums.** The projection (counters/aggregates) is the authority; the
optional event stream is audit only. `replay(events) == projection` is a *test*
invariant, not a runtime dependency (events may be dropped in production).

---

## 2. Layer / ownership contract

| Layer | Responsibility |
|---|---|
| `core/_metering/` | Neutral mechanism: record facts, run the operation lifecycle, hold counters + projections, expose an **injectable admission hook** (no-op without a scope). Zero budget knowledge. |
| `core` (LLM / tools / pricing) | Charge sites: build `OperationFacts` → `open` → `mark_started` → execute → `settle`/`fail`. Cost computation via `PricingRegistry`. |
| `toolkit/budget/` | The opinion: `BudgetPolicy`, hard/soft, reservation modes, fail-closed, `BudgetExceeded`, the report, default pricer/estimator. Implements the core admission Protocol. |
| `toolkit/flow/` | Opens the run `MeterScope`, installs the `BudgetAdmissionController`, opens step spans, derives `trace`/`Result`/report from meter projections. Applies `Policy.max_cost` at the step span. |
| `toolkit/agents/` | Compose flows. **No manual `total_cost`/`total_usage`.** |
| app (downstream) | user/team/org/monthly budgets, on top of `UsageEvent`/projections. |

Rule the layering enforces: **`core` never imports `toolkit`** (verified: 0 refs).
The admission decision *executes* at the in-core charge site (it must, for hard
caps) but is *supplied* by toolkit via an injected hook typed by a core Protocol.

---

## 3. Module layout

```
core/_metering/
  _money.py        # Money (opaque; pico-USD int internally)
  _cost.py         # Cost (one class: kind known|estimated|unknown + factories)
  _events.py       # UsageEvent, UsageSink (Protocol)
  _operation.py    # OperationFacts, MeterOperation (lease lifecycle)
  _scope.py        # MeterScope, current_meter(), bind_meter()
  _store.py        # MeterStore (counters + per-span projections + optional event buffer)
  _admission.py    # AdmissionController (Protocol), MeterSnapshot, AdmissionDecision, Reservation
  __init__.py

core/                # charge sites + cost (already provider-agnostic)
  _llm.py          # complete()/stream() open operations (stream = kind="llm", mode="stream")
  _tools/_executor.py   # THE common metered+gated tool executor
  _pricing.py      # PricingRegistry, estimate_cost (mechanism; default table overridable)
  _response.py     # Usage, Response.cost (float, legacy), cost_money (Money), cost_known

toolkit/budget/
  _policy.py       # BudgetPolicy, reservation mode enum
  _controller.py   # BudgetAdmissionController (implements core Protocol) + the estimator
  _state.py        # BudgetSnapshot
  _report.py       # BudgetReport
  _exceptions.py   # BudgetExceeded
  __init__.py

toolkit/_runner.py          # run_tools — STAYS in toolkit (convenience), but MUST route
                            #   through core's common metered+gated tool executor
toolkit/flow/_executor.py   # opens scope, installs controller, step spans, max_cost
toolkit/agents/flows/*.py   # zero accounting
```

Top-level re-export `from ai_arch_toolkit import BudgetPolicy` is OK; **ownership
is `toolkit.budget`, not `core`.**

---

## 4. Type contracts (responsibilities + invariants — not full impl)

```python
# core/_metering/_money.py
class Money:                                # opaque; internal int pico-USD (1e-12 USD). Exact.
    @classmethod
    def from_usd(cls, x: float | Decimal) -> Money: ...
    @classmethod
    def from_pico(cls, p: int) -> Money: ...   # rate_pico * tokens -> Money (the pricer path)
    def to_float(self) -> float: ...           # display/compat only
    # __add__ / __sub__ / __mul__(int) / __lt__ / __le__ : exact integer arithmetic.
    # users never see "pico". Public API accepts/returns float|Decimal.

# core/_metering/_cost.py   — ONE class with a kind + factories (NOT a 3-class union)
@dataclass(frozen=True, slots=True)
class Cost:
    kind: Literal["known", "estimated", "unknown"]
    amount: Money | None = None              # None iff kind == "unknown"
    reason: str | None = None                # set iff kind == "unknown"
    @classmethod
    def known(cls, m: Money) -> Cost: ...
    @classmethod
    def estimated(cls, m: Money) -> Cost: ...
    @classmethod
    def unknown(cls, reason: str) -> Cost: ...
    @property
    def is_known(self) -> bool: ...
    @staticmethod
    def merged(*costs: Cost) -> Cost: ...     # any unknown -> unknown; else any estimated -> estimated
# Response.cost (float) / cost_known (bool) / cost_money (Money|None) are COMPAT VIEWS of one Cost.

# core/_metering/_operation.py
@dataclass(frozen=True, slots=True)
class OperationFacts:        # built by CORE; pure FACTS only — NO estimates, NO heuristics
    kind: Literal["llm", "tool", "custom"]   # streaming is kind="llm" (mode="stream"), never separate
    span_id: str
    count: int = 1                           # 1 per llm/tool call -> max_*_calls ; custom -> 0
    mode: Literal["complete", "stream"] | None = None
    model: str | None = None
    declared_max_output_tokens: int | None = None   # the user's max_tokens (a FACT)
    content_size_hint: int | None = None            # cheap char/byte count (a FACT, not a token estimate)
    metadata: Mapping[str, Any] = field(default_factory=dict)
# The token/cost ESTIMATE is NOT a field here. The controller's injected estimator turns these
# facts into an estimate at admit time (estimation heuristic stays in toolkit; see §4b).

class MeterOperation:        # the lease handle the charge site holds; delegates to MeterStore under lock
    def mark_started(self) -> None: ...   # COMMITS the call/tool count (reached the provider/tool)
    def settle(self, *, usage: Usage, cost: Cost) -> None: ...  # adds usage/cost; idempotent by op_id
    def fail(self) -> None: ...           # STARTED but errored: count STAYS, outstanding released
    def abort(self) -> None: ...          # PENDING only (never started): FULL release, no count

# MeterOperation state machine (write as tests):
#   PENDING --abort-->        released, NO count            (never reached provider)
#   PENDING --mark_started--> STARTED, count COMMITTED      (reached provider/tool)
#   STARTED --settle-->       SETTLED  (count + actual usage/cost)
#   STARTED --fail-->         FAILED   (count STAYS; usage 0, cost Unknown; outstanding released)
#   stream STARTED, un-drained at scope close -> INCOMPLETE (count stays; cost Estimated/Unknown)
# WHY count-on-start (Codex): if a started-then-failed op released its count, a retry/fallback loop
#   could make N real provider calls under a finite max_llm_calls. Counting attempts that reach the
#   provider closes that hole. => max_llm_calls / max_tool_calls bound PHYSICAL attempts (including
#   retries and failed fallbacks), not logical calls. This is deliberate; document it.
# Idempotency: settle(op, same payload) twice = no-op; different payload = error;
#              settle/abort after fail = error; abort after start = error.

# core/_metering/_admission.py
@dataclass(frozen=True, slots=True)
class Reservation:                           # amounts the store HOLDS until settle/fail/abort
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: Money | None = None                # worst-case hold (only if reserve != NONE)

@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    admitted: bool
    reservation: Reservation | None = None
    breach: BudgetExceeded | None = None     # set iff not admitted; the store raises it on open()

class AdmissionController(Protocol):
    def admit(self, snapshot: MeterSnapshot, facts: OperationFacts) -> AdmissionDecision: ...
# PURE: sync, no I/O, no DB, no await. Decides; the MeterStore applies the reservation under the lock.

@dataclass(frozen=True, slots=True)
class MeterSnapshot:          # what the controller sees; what snapshot()/for_span()/totals() return
    # committed (settled):
    llm_calls: int; tool_calls: int
    input_tokens: int; output_tokens: int; cache_read_tokens: int; cache_write_tokens: int
    cost: Money
    # outstanding (reserved, not yet settled):
    out_calls: int; out_cost: Money
    elapsed_s: float          # so the controller can enforce max_wall_s

# core/_metering/_store.py
class MeterStore:
    # ONE writer. counters O(1) (authoritative) + per-span incremental aggregates
    # + optional bounded event buffer + sinks. self._lock: threading.Lock.
    # --- WRITE (each acquires the lock) ---
    def open(self, facts: OperationFacts, controller: AdmissionController | None) -> MeterOperation: ...
        # under lock: snap=self.snapshot(); d = controller.admit(snap, facts) if controller else ALLOW
        #             if not d.admitted: raise d.breach  else apply d.reservation -> PENDING op
    def mark_started(self, op_id) -> None: ...   # commit the count for this op
    def settle(self, op_id, *, usage, cost) -> None: ...  # release reservation; add committed; update span aggs + emit event
    def fail(self, op_id) -> None: ...           # release reservation; count already committed
    def abort(self, op_id) -> None: ...          # release reservation; valid only if PENDING
    # --- READ (consistent snapshots) ---
    def snapshot(self) -> MeterSnapshot: ...
    def for_span(self, span_id, *, descendants=True) -> MeterSnapshot: ...   # O(1) via incremental aggs
# Span model: each span has span_id, parent_span_id, scope_type ∈ {run, step, operation}.
#   tree: run_span -> step_span -> {llm_op, tool_op}. settle walks the parent chain (O(depth)).

# core/_metering/_scope.py
@dataclass(slots=True)
class MeterScope:            # bound to a ContextVar; the object current_meter() returns
    store: MeterStore
    controller: AdmissionController | None     # injected by toolkit/flow; None => no enforcement
    started_at: float                          # via an injectable clock (testable wall-time)
    def open(self, facts: OperationFacts) -> MeterOperation: ...   # -> store.open(facts, self.controller)
def current_meter() -> MeterScope | None: ...  # ambient hook; None => no-op (unmetered)
def bind_meter(scope) -> Token: ...            # toolkit/flow binds at run start; reused by nested flows

# toolkit/budget/_policy.py
@dataclass(frozen=True)
class BudgetPolicy:
    max_llm_calls / max_tool_calls: int | None
    max_input_tokens / max_output_tokens / max_total_tokens: int | None
    max_cost: float | Decimal | None  # user-facing; compiled to Money once at run start
    max_wall_s: float | None
    reserve: Reserve = Reserve.NONE   # cost/tokens knob; counts are ALWAYS hard. default soft (post-hoc).
    unpriced: Unpriced = Unpriced.FAIL_CLOSED

# toolkit/budget/_controller.py
class BudgetAdmissionController:     # implements AdmissionController; holds policy + injected estimator + pricer
    def admit(self, snapshot, facts) -> AdmissionDecision: ...
        # 1. estimate input tokens from facts via the INJECTED estimator (heuristic lives here)
        # 2. price -> estimated cost via the pricer
        # 3. apply policy: counts hard; cost/tokens per reserve knob; wall-time via snapshot.elapsed_s
        # 4. return ALLOW+Reservation, or DENY+BudgetExceeded
```

---

## 4b. Lifecycle of one metered call (the object graph + the flow)

Object graph: `MeterScope` (bound to a ContextVar) holds the `MeterStore` (state +
lock + per-span aggregates + optional events), the injected `AdmissionController | None`,
and the clock. **Charge sites talk only to `current_meter()`.** No scope ⇒ no-op (unmetered).

```python
# complete() — core/_llm.py
meter = current_meter()
if meter is None:
    return await provider.complete(...)                    # unmetered (no active scope)
facts = OperationFacts(kind="llm", mode="complete", span_id=current_span(),
                       model=model, declared_max_output_tokens=max_tokens,
                       content_size_hint=cheap_len(messages))
op = meter.open(facts)            # store.lock: snapshot -> controller.admit(snap, facts)
                                  #   -> apply reservation  |  raise BudgetExceeded
op.mark_started()                 # COMMIT the call count (a failure still counts — see §4)
try:
    resp = await provider.complete(...)
except Exception:
    op.fail(); raise              # count stays; outstanding released
op.settle(usage=resp.usage, cost=Cost.from_response(resp, pricer))   # idempotent by op_id
```

- **stream()** — same, BUT capture `op`/`meter` at **BUILD time** (the finalizer runs on the
  drain thread where the ContextVar is unset). `op.mark_started()` when the provider stream
  opens; `op.settle(...)` in the finalizer closure; un-drained at scope close → `INCOMPLETE`.
- **tools** — `core/_tools/_executor.py` runs governance gates FIRST; only an actually-executed
  tool opens an op (`mark_started` before `fn(...)`, `settle`/`fail` after). `run_tools` routes
  through THIS executor (no direct `fn(...)`).
- **flow** — `toolkit/flow/_executor.py` does `bind_meter(MeterScope(store, controller, clock))`
  at run start, opens a step span per step, reads `for_span(step)` to apply `Policy.max_cost`
  and to fill `Result.cost`/`Result.usage` (views).

---

## 5. Cravadas (resolved) decisions — checklist for implementation

- [ ] **`admit + reserve` is atomic.** `with store.lock: snap = self.snapshot(); d = controller.admit(snap, facts); apply(d)` — **all in ONE critical section** (no TOCTOU; two parallel `gather` calls must not admit against the same remaining budget).
- [ ] **`threading.Lock`, NOT `asyncio.Lock`.** The stream finalizer settles from a **different OS thread** (sync drain). Reaching for `asyncio.Lock` because the system is async is the trap.
- [ ] **Count commits on start; `abort` is PENDING-only.** `mark_started` (reaching the provider/tool) commits the call/tool count. A started-then-failed op keeps its count — `fail()` releases only the outstanding reservation. `abort()` is valid **only before start**. ⇒ `max_llm_calls`/`max_tool_calls` bound **physical attempts** (incl. retries and failed fallbacks); document this so it isn't surprising.
- [ ] **Sync bridge:** `_run_sync` must `copy_context()` so `complete_sync`/`*_sync` called **inside a flow** inherit the meter (a real bug in the old design — sync calls escaped metering). `_stream_sync` does NOT copy context — which is exactly why the stream op is captured at build-time. **Rule:** sync calls inside an active `MeterScope` are metered; outside a scope they are unmetered unless the caller explicitly binds a meter.
- [ ] **`admit()` is sync + pure** — no `await`, no I/O, no DB. Monthly/DB budgets are resolved to a per-run `BudgetPolicy` at run start, not queried mid-flight.
- [ ] **Streaming is `kind="llm"`, `mode="stream"` — not a separate kind.** So `max_llm_calls` always counts a stream. The stream **op is captured at BUILD time**, carried in the finalizer closure. Never re-read `current_meter()` at finalize. Settle at drain; un-drained → `incomplete`, never silently lost.
- [ ] **`run_tools()` routes through the common metered + gated executor.** It STAYS in `toolkit/_runner.py`, but today it calls `fn(**input)` directly, bypassing metering **and** governance (DangerousToolGate/ApprovalGate). A **security** hole, not just budget. No tool path may bypass the common executor.
- [ ] **`Policy.max_cost` applied by the EXECUTOR at the step span** — not the step engine. Executor opens step span → step runs → ops attach → executor reads `for_span(step)` → applies `max_cost` → fills `Result.cost` as view/compat. `Policy.max_cost` (step) and `BudgetPolicy.max_cost` (run) are the SAME mechanism at different spans — **do not deprecate the step cap; unify it.**
- [ ] **`max_wall_s` enforcement:** `MeterScope` holds `started_at` (injectable clock); the controller checks `snapshot.elapsed_s` on each `admit`; the executor checks before/after each step. It does **not** interrupt an in-flight call — for that use `Policy.timeout`.
- [ ] **Reserve default = soft.** `reserve` is a cost/tokens knob; **counts are always hard** regardless. Default `NONE` (post-hoc soft) to avoid over-blocking; `EXPECTED`/`WORST_CASE` opt-in. Token worst-case reservation needs `declared_max_output_tokens`. **With `NONE`, cost/token caps are post-hoc and may overshoot by one in-flight op; hard pre-call cost/token caps require `EXPECTED`/`WORST_CASE`.**
- [ ] **Token-cap convention:** `Usage` = `input_tokens` (non-cached) · `cache_read_tokens` · `cache_write_tokens` · `output_tokens` (disjoint after provider normalization). `max_input_tokens` caps `input + cache_read + cache_write` (the **full context sent**) · `max_output_tokens` caps `output` · `max_total_tokens` caps all four. Cache **rate** differences are a *cost* concern (pricing), not the token cap.
- [ ] **Events optional; counters/projections authoritative.** Default: counters retained, events NOT retained, streamed to `UsageSink`s if present, `retain_events=True` opt-in, `max_events` cap. No O(n) memory leak on long runs.
- [ ] **`Money` opaque, pico-int internal.** `BudgetPolicy(max_cost=0.10)` at the API; `Money` internally. Never expose "pico".
- [ ] **Per-span projections via incremental aggregates** — maintained on `settle` (walk `parent_span_id` chain), O(depth) write / O(1) read. No fold-on-read.
- [ ] **estimate/facts boundary:** the core charge site builds `OperationFacts` (pure facts); the **estimator is injected into the controller** (toolkit) and produces the estimate at admit time. **Core never imports or calls toolkit code** — the heuristic and reserve-mode opinion never re-enter core.
- [ ] **`Response.cost` (float) + `cost_money` (Money) + `cost_known`** are compat views of one `Cost`; later `cost` becomes a property of `cost_money`.

---

## 6. Phased commit plan (strangler; suite green at each step)

1. `feat(core): neutral metering primitives` — `core/_metering/` (Money, Cost, OperationFacts, MeterOperation, MeterScope, MeterStore, AdmissionController/Decision/Snapshot/Reservation, UsageEvent/Sink, current_meter/bind_meter) **+ property-test oracle**. No `BudgetPolicy` — the oracle enforces via a **test-double controller** (a trivial cap), since the real one is toolkit (step 4).
2. `feat(core): meter LLM operations` — `complete()` + `stream()` charge sites (build-time finalizer capture; stream = `kind="llm"`, `mode="stream"`; `mark_started`/`fail`/`settle`).
3. `feat(core): meter tool operations` — `ToolGroup` **and `run_tools`** through the common metered+gated executor.
4. `feat(toolkit): run-level budget controller` — `toolkit/budget/` (BudgetPolicy, BudgetAdmissionController + estimator, BudgetExceeded, BudgetReport, BudgetSnapshot).
5. `feat(flow): bind meter scope + step spans` — executor opens `MeterScope`, installs controller, opens step spans, derives trace/result/report from projections. **No reconciliation.**
6. `feat(flow): apply Policy.max_cost from the step-span projection` — move the check out of `_step_engine` into the executor (step-scope budget; **not** deprecated). Removes the old `_step_engine.py:89` `Result.cost` check in the SAME commit.
7. `refactor(agents): derive usage/cost from flow metering` — delete manual accumulators from the 9 flows.
8. `test+docs` — invariants/property tests; rewrite `docs/budgets.md` + budget sections; `examples/38_budget.py`.

---

## 7. Invariants / property tests (the oracle — uses a test-double controller in step 1)

- `committed ≤ cap` for every hard cap, under any schedule.
- `outstanding == 0` after the run (no reservation leak).
- `reported == projection == replay(events)`.
- Hard count caps exact under parallel `asyncio.gather` (incl. streams — `kind="llm"`).
- **Started-then-failed op keeps its count** — no unbounded retries/failed fallbacks under a finite call cap.
- No reservation leak on fallback / abort / abandon.
- Operation idempotency: double-settle same payload = no-op; different payload = error.
- Unpriced under a cost cap → fail-closed (unless `policy.unpriced == ALLOW`).
- Stream abandoned → `incomplete`, never silently dropped.
- `STRATEGIES` matrix (`usage == projection` per strategy × {no-breach, breach}) — **derive the strategy list from the registry** so a new strategy can't escape coverage.

---

## 8. Keep / Delete / Move

**Keep:** ContextVar run-scope binding · reserve→settle escrow for hard counts ·
the 3 charge-site seams · typed `Cost(known/estimated/unknown)` · provider cache-token
subtraction · the invariant test matrix (registry-derived) · ambient (no signature change) ·
**zero business concepts**.

**Delete:** per-strategy accumulators (9 flows) · reconciliation
(`_unrecorded_spend`/`_append_budget_reconciliation`/`_budget_owns_ledger`/`budget_reconcile`) ·
public core `Ledger` name (→ `Meter`) · `BudgetPolicy`/`BudgetExceeded` in core ·
the unwired-vs-live duplication.

**Move:** `BudgetPolicy`/`BudgetExceeded`/enforcement → `toolkit/budget/` ·
estimation heuristics → toolkit/controller · pricing default table stays in core
**but fully overridable**.

---

## 9. Footguns to NOT reintroduce

- Unmetered tool surface. **Must-fix now:** `run_tools` (security: bypasses metering
  AND governance). **Document-as-unmetered (do NOT chase in v1):** direct provider
  calls, batch APIs not yet integrated, manual LLM-facade bypass, out-of-flow sync.
- Releasing the call count on a started-then-failed op (lets retries/fallbacks overspend the call cap).
- Per-strategy accounting; reconciliation as a crutch.
- `Ledger` name leaking budget semantics into core (use `Meter`).
- `asyncio.Lock` instead of `threading.Lock`.
- Reading `current_meter()` at stream-finalize time (use build-time capture).
- `kind="stream"` separate from `"llm"` (lets a stream escape `max_llm_calls`).
- Estimates as fields on the core-built `OperationFacts` (puts the heuristic in core).
- `float` for internal money (use `Money`/pico-int).
- Soft cost cap presented as hard (document the ≤1-call overshoot).

---

## 10. Deferred (designed-for, with triggers)

- **Sub-caps per sub-agent:** model the span-scoped admission now (needed to unify
  per-step + run anyway); expose public API only when asked.
- **OTel exporter:** a `UsageSink`; design the seam, defer the impl. (Keep the event
  OTel-agnostic; map `gen_ai.*` in the toolkit adapter — and decide the
  `input_tokens`-includes-cache convention against the target semconv version.)
- **Persistence / resumable budgets (multi-run):** event log with `op_id` is
  serializable; defer the impl.
- **Event retention:** counters authoritative; events sink-and-forget;
  `retain_events` / `max_events` opt-in.
