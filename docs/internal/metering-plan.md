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

Invariant: `reported(scope) ≡ enforced(scope) ≡ Σ committed charges in scope`,
by construction — one writer, many derived read-models, never parallel sums.

---

## 2. Layer / ownership contract

| Layer | Responsibility |
|---|---|
| `core/_metering/` | Neutral mechanism: record facts, run the operation lifecycle, hold counters + projections, expose an **injectable admission hook** (no-op without a scope). Zero budget knowledge. |
| `core` (LLM / tools / pricing) | Charge sites: open operation → atomic admit/reserve → execute → settle/abort. Cost computation via `PricingRegistry`. |
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
  _cost.py         # Cost = Known | Estimated | Unknown
  _events.py       # UsageEvent, UsageSink (Protocol)
  _operation.py    # OperationIntent, MeterOperation (lease lifecycle)
  _scope.py        # MeterScope, current_meter(), bind_meter()
  _store.py        # MeterStore (counters + per-span projections + optional event buffer)
  _admission.py    # AdmissionController (Protocol), MeterSnapshot, AdmissionDecision
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
class Money:
    # opaque; internal representation is int pico-USD (1e-12 USD). Exact +/-/compare.
    @classmethod
    def from_usd(cls, x: float | Decimal) -> Money: ...
    def to_float(self) -> float: ...          # display/compat only
    # users never see "pico". API accepts/returns float|Decimal.

# core/_metering/_cost.py
type Cost = Known | Estimated | Unknown        # sum type; THE single money model
# Known(Money) | Estimated(Money) | Unknown(reason). merged() degrades certainty.
# Response.cost (float) + Response.cost_known (bool) are COMPAT views derived from this;
# Response.cost_money is the Money from a Cost.Known. There is ONE model (Cost), not two.

# core/_metering/_operation.py
@dataclass(frozen=True)
class OperationIntent:                          # built by the charge site BEFORE the call
    kind: Literal["llm", "tool", "custom"]      # streaming is kind="llm" (mode="stream") —
                                                #   NEVER a separate kind, so max_llm_calls
                                                #   can never miss a stream
    span_id: str
    count: int = 1                              # 1 per call/tool -> max_*_calls is hard
    mode: Literal["complete", "stream"] | None = None
    model: str | None = None
    # FACTS + a NEUTRAL pre-estimate produced by an INJECTED estimator (the heuristic
    # — chars/tok, image allowance — lives in toolkit, NOT baked in core):
    estimated_input_tokens: int | None = None
    declared_max_output_tokens: int | None = None
    estimated_cost: Cost = Unknown("not estimated")
    metadata: Mapping[str, Any] = field(default_factory=dict)
# The controller reads these raw numbers to enforce BOTH token caps and cost, and to
# choose the reserve mode (worst/expected/none).

class MeterOperation:         # the lease; RAII
    def settle(self, *, usage: Usage, cost: Cost) -> None: ...  # idempotent by op_id
    def abort(self) -> None: ...        # fallback/cancel; releases reservation
    def mark_started(self) -> None: ... # stream: count committed when the stream opens
    # on scope close: un-drained stream op -> incomplete charge; non-stream open op -> leak error

# MeterOperation state machine + idempotency rules (write these as tests):
#   PENDING -> STARTED -> SETTLED | ABORTED ;  stream un-drained at scope close -> INCOMPLETE
#   double settle, same op_id, SAME payload      -> idempotent no-op
#   double settle, same op_id, DIFFERENT payload -> error (silent-drift guard)
#   abort after settle  -> no-op
#   settle after abort  -> error

# core/_metering/_admission.py
class AdmissionController(Protocol):
    def admit(self, snapshot: MeterSnapshot, intent: OperationIntent) -> AdmissionDecision: ...
# PURE: sync, no I/O, no DB, no await. Returns allow+reservation or deny.
# The MeterStore APPLIES the reservation under the lock. Controller decides; meter applies.

# core/_metering/_store.py
class MeterStore:
    # counters O(1) (authoritative for enforcement) + per-span incremental aggregates
    # + optional bounded event buffer + sinks. threading.Lock.
    def snapshot(self) -> MeterSnapshot: ...
    def for_span(self, span_id: str, *, descendants=True) -> MeterSnapshot: ...   # O(1)
    def totals(self) -> MeterSnapshot: ...
# Span model (required for the O(depth) ancestor walk to be well-defined):
#   each span has span_id, parent_span_id, scope_type ∈ {run, step, operation}.
#   tree: run_span -> step_span -> {llm_operation, tool_operation}.
#   on settle, walk parent chain updating each ancestor's aggregate.

# toolkit/budget/_policy.py
@dataclass(frozen=True)
class BudgetPolicy:
    max_llm_calls / max_tool_calls: int | None
    max_input_tokens / max_output_tokens / max_total_tokens: int | None
    max_cost: float | None          # user-facing dollars; -> Money at compile
    max_wall_s: float | None
    reserve: Reserve = WORST_CASE   # WORST_CASE | EXPECTED | NONE; counts always hard
    unpriced: Unpriced = FAIL_CLOSED

# toolkit/budget/_controller.py
class BudgetAdmissionController:     # implements AdmissionController
    def admit(self, snapshot, intent) -> AdmissionDecision: ...   # applies the policy + reserve mode
```

---

## 5. Cravadas (resolved) decisions — checklist for implementation

- [ ] **`admit + reserve` is atomic.** `with store.lock: snapshot = store.snapshot(); d = controller.admit(snapshot, intent); store.apply(d)` — **all three in ONE critical section** (no TOCTOU; two parallel `gather` calls must not admit against the same remaining budget).
- [ ] **`threading.Lock`, NOT `asyncio.Lock`.** The stream finalizer settles from a **different OS thread** (sync drain). Reaching for `asyncio.Lock` because the system is async is the trap.
- [ ] **`admit()` is sync + pure** — no `await`, no I/O, no DB. Monthly/DB budgets are resolved to a per-run `BudgetPolicy` at run start, not queried mid-flight.
- [ ] **Streaming is `kind="llm"`, `mode="stream"` — not a separate kind.** So `max_llm_calls` always counts a stream. The stream **op is captured at BUILD time**, carried in the finalizer closure. Never re-read `current_meter()` at finalize (the drain thread has the ContextVar unset). Settle at drain; un-drained → `incomplete`, never silently lost.
- [ ] **`run_tools()` routes through the common metered + gated executor.** It STAYS in `toolkit/_runner.py` (convenience), but today it calls `fn(**input)` directly, bypassing metering **and** governance (DangerousToolGate/ApprovalGate). This is a **security** hole, not just budget. No tool path may bypass the common executor.
- [ ] **`Policy.max_cost` applied by the EXECUTOR at the step span** — not the step engine. Executor opens step span → step runs → ops attach → executor reads `for_span(step)` projection → applies `max_cost` → fills `Result.cost` as view/compat. Step engine stays free of budget. `Policy.max_cost` (step) and `BudgetPolicy.max_cost` (run) are the SAME mechanism at different spans — **do not deprecate the step cap; unify it.**
- [ ] **`max_wall_s` enforcement:** `MeterScope` holds `started_at`; the controller checks it on each `admit`; the executor checks it before/after each step. It does **not** interrupt an in-flight call — for that use `Policy.timeout`. (Set the expectation in docs.)
- [ ] **Token-cap convention (decide now; override if you disagree):**
  `Usage` = `input_tokens` (non-cached) · `cache_read_tokens` · `cache_write_tokens` · `output_tokens`.
  `max_input_tokens` caps `input_tokens + cache_read_tokens` (all prompt tokens) ·
  `max_output_tokens` caps `output_tokens` ·
  `max_total_tokens` caps the sum of all four.
  Cache **rate** differences (read cheaper, write dearer) are a *cost* concern handled by `Cost`/pricing — **not** the token cap.
- [ ] **Events optional; counters/projections authoritative.** Default: counters retained, events NOT retained, events streamed to `UsageSink`s if present, `retain_events=True` opt-in for debug, `max_events` cap. No O(n) memory leak on long runs.
- [ ] **`Money` opaque, pico-int internal.** `BudgetPolicy(max_cost=0.10)` at the API; `Money` internally. Never expose "pico".
- [ ] **Per-span projections via incremental aggregates** — maintained on `settle` (walk `parent_span_id` chain), O(depth) write / O(1) read. No fold-on-read over the whole log.
- [ ] **estimate/intent boundary:** the intent carries FACTS + a pre-estimate from an **injected estimator** (toolkit). The controller decides the reserve mode (worst/expected/none). Estimation heuristic and reserve-mode opinion stay out of core.
- [ ] **`Response.cost` (float, legacy) + `cost_money` (Money) + `cost_known`** are compat views of one `Cost`; later `cost` becomes a property of `cost_money`.

---

## 6. Phased commit plan (strangler; suite green at each step)

1. `feat(core): neutral metering primitives` — `core/_metering/` (Money, Cost, OperationIntent, MeterOperation, MeterScope, MeterStore, AdmissionController Protocol, UsageEvent/Sink, current_meter/bind_meter) **+ property-test oracle**. No `BudgetPolicy`.
2. `feat(core): meter LLM operations` — `complete()` + `stream()` charge sites (incl. build-time finalizer capture; stream = `kind="llm"`, `mode="stream"`).
3. `feat(core): meter tool operations` — `ToolGroup` **and `run_tools`** through the common metered+gated executor.
4. `feat(toolkit): run-level budget controller` — `toolkit/budget/` (BudgetPolicy, BudgetAdmissionController + estimator, BudgetExceeded, BudgetReport, BudgetSnapshot).
5. `feat(flow): bind budget scope during execution` — executor opens `MeterScope`, installs controller, opens step spans, derives trace/result/report from projections. **No reconciliation.**
6. `refactor(agents): derive usage/cost from flow metering` — delete manual accumulators from the 9 flows.
7. `refactor(flow): reimplement Policy.max_cost as step-span budget` (executor-applied; not deprecated).
8. `test+docs` — invariants/property tests; rewrite `docs/budgets.md` + budget sections; `examples/38_budget.py`.

---

## 7. Invariants / property tests (the oracle)

- `committed ≤ cap` for every hard cap, under any schedule.
- `outstanding == 0` after the run (no reservation leak).
- `reported == Σ committed == replay(log)`.
- Hard count caps exact under parallel `asyncio.gather` (incl. streams — `kind="llm"`).
- No reservation leak on fallback / abort / abandon.
- Operation idempotency: double-settle same payload = no-op; different payload = error.
- Unpriced under a cost cap → fail-closed (unless `allow_unpriced`).
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
  Document the boundary honestly in `docs/budgets.md`.
- Per-strategy accounting; reconciliation as a crutch.
- `Ledger` name leaking budget semantics into core (use `Meter`).
- `asyncio.Lock` instead of `threading.Lock`.
- Reading `current_meter()` at stream-finalize time (use build-time capture).
- `kind="stream"` separate from `"llm"` (lets a stream escape `max_llm_calls`).
- `float` for internal money (use `Money`/pico-int).
- Soft cost cap presented as hard (document the ≤1-call overshoot).

---

## 10. Deferred (designed-for, with triggers)

- **Sub-caps per sub-agent:** model the span-scoped admission now (it's needed to
  unify per-step + run anyway); expose public API only when asked.
- **OTel exporter:** a `UsageSink`; design the seam, defer the impl. (Keep the
  event OTel-agnostic; map `gen_ai.*` in the toolkit adapter — and decide the
  `input_tokens`-includes-cache convention against the target semconv version.)
- **Persistence / resumable budgets (multi-run):** event log with `op_id` is
  serializable; defer the impl.
- **Event retention:** counters authoritative; events sink-and-forget;
  `retain_events` / `max_events` opt-in.
