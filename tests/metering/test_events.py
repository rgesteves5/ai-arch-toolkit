"""UsageEvent emission: status/cost per terminal kind, redaction, ordering, outside-lock emit."""

from __future__ import annotations

from ai_arch_toolkit.core._metering._cost import Cost
from ai_arch_toolkit.core._metering._events import UsageEvent
from ai_arch_toolkit.core._metering._money import Money
from ai_arch_toolkit.core._metering._operation import OperationRequest
from ai_arch_toolkit.core._metering._store import MeterStore
from ai_arch_toolkit.core._redaction import REDACTED
from ai_arch_toolkit.core._response import Usage


class Recorder:
    def __init__(self) -> None:
        self.events: list[UsageEvent] = []

    def emit(self, event: UsageEvent) -> None:
        self.events.append(event)


def llm(**kw) -> OperationRequest:
    return OperationRequest(kind="llm", parent_span_id="run", **kw)


def with_sink() -> tuple[MeterStore, Recorder]:
    rec = Recorder()
    return MeterStore(sinks=[rec]), rec


def test_no_sink_builds_no_events():
    # measure-only path stays zero-overhead: seq never advances without a sink.
    store = MeterStore()
    op = store.open(llm(), None)
    op.mark_started()
    op.settle(usage=Usage(input_tokens=5), cost=Cost.known(Money.from_usd(0.01)))
    assert store._seq == 0  # white-box: no event was minted


def test_settle_emits_a_settled_event():
    store, rec = with_sink()
    op = store.open(llm(model="claude-x", provider="anthropic", mode="complete"), None)
    op.mark_started()
    op.settle(usage=Usage(input_tokens=10, output_tokens=4), cost=Cost.known(Money.from_usd(0.02)))
    (ev,) = rec.events
    assert ev.status == "settled" and ev.kind == "llm" and ev.span_id == "run"
    assert ev.model == "claude-x" and ev.provider == "anthropic" and ev.mode == "complete"
    assert ev.usage.input_tokens == 10 and ev.cost == Cost.known(Money.from_usd(0.02))


def test_failed_llm_event_carries_unknown_cost():
    store, rec = with_sink()
    op = store.open(llm(), None)
    op.mark_started()
    op.fail()
    (ev,) = rec.events
    assert ev.status == "failed" and ev.cost.kind == "unknown"


def test_failed_tool_event_is_free():
    store, rec = with_sink()
    op = store.open(OperationRequest(kind="tool", parent_span_id="run"), None)
    op.mark_started()
    op.fail()
    (ev,) = rec.events
    assert ev.status == "failed" and ev.cost == Cost.known(Money.zero())


def test_abort_emits_aborted_event():
    store, rec = with_sink()
    store.open(llm(), None).abort()
    (ev,) = rec.events
    assert ev.status == "aborted" and ev.cost == Cost.known(Money.zero())


def test_close_emits_incomplete_for_a_started_op():
    store, rec = with_sink()
    op = store.open(llm(), None)
    op.mark_started()
    store.close()
    (ev,) = rec.events
    assert ev.status == "incomplete" and ev.cost.kind == "unknown"


def test_metadata_is_redacted_before_emit():
    store, rec = with_sink()
    meta = {"step": "plan", "api_key": "sk-super-secret-123"}
    op = store.open(llm(metadata=meta), None)
    op.mark_started()
    op.settle(usage=Usage(), cost=Cost.known(Money.zero()))
    (ev,) = rec.events
    assert ev.metadata["api_key"] == REDACTED  # sensitive key scrubbed
    assert ev.metadata["step"] == "plan"  # benign key kept
    assert meta["api_key"] == "sk-super-secret-123"  # original mapping untouched


def test_event_seq_is_monotonic():
    store, rec = with_sink()
    for _ in range(3):
        op = store.open(llm(), None)
        op.mark_started()
        op.settle(usage=Usage(), cost=Cost.known(Money.zero()))
    assert [e.seq for e in rec.events] == [1, 2, 3]


def test_a_raising_sink_never_breaks_the_run():
    class Boom:
        def emit(self, event: UsageEvent) -> None:
            raise RuntimeError("sink down")

    store = MeterStore(sinks=[Boom()])
    op = store.open(llm(), None)
    op.mark_started()
    op.settle(usage=Usage(input_tokens=7), cost=Cost.known(Money.from_usd(0.01)))
    assert store.snapshot().input_tokens == 7  # accounting intact despite the sink blowing up


def test_dispatch_runs_outside_the_lock():
    # A sink that reads the meter on emit would deadlock if dispatch held the (non-reentrant) lock.
    seen = {}

    class Reentrant:
        def __init__(self, store: MeterStore) -> None:
            self.store = store

        def emit(self, event: UsageEvent) -> None:
            seen["snap"] = self.store.snapshot()  # re-acquires the lock

    store = MeterStore()
    store._sinks = (Reentrant(store),)  # white-box: the sink needs the store ref
    op = store.open(llm(), None)
    op.mark_started()
    op.settle(usage=Usage(input_tokens=3), cost=Cost.known(Money.zero()))
    assert seen["snap"].input_tokens == 3  # emitted, read back, no deadlock
