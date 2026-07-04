"""LLM.complete charge site: metered per attempt, enforced, failed-attempt kept, sync + pricer."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._metering._admission import (
    AdmissionDecision,
    AdmissionDenied,
    MeterSnapshot,
    NotMeteredOperationError,
    ResourceLimits,
)
from ai_arch_toolkit.core._metering._cost import Cost
from ai_arch_toolkit.core._metering._money import Money
from ai_arch_toolkit.core._metering._operation import OperationRequest
from ai_arch_toolkit.core._metering._scope import MeterScope, RunConfig
from ai_arch_toolkit.core._providers._base import StreamState
from ai_arch_toolkit.core._response import Response, StreamEvent, Usage

MODEL = "claude-sonnet-4-6"  # priced in _default_pricing.toml


class FakeProvider:
    """Stands in for a real provider — returns a canned Response or raises."""

    def __init__(
        self, *, response: Response | None = None, error: Exception | None = None
    ) -> None:
        self._response = response
        self._error = error
        self.calls = 0

    async def complete(self, messages, *, system=None, tools=None, **kwargs) -> Response:
        self.calls += 1
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response

    async def batch_submit(self, requests) -> str:
        self.calls += 1
        return "batch-123"


class CapController:
    def __init__(self, **limits: int) -> None:
        self._limits = ResourceLimits(**limits)

    def admit(self, snapshot: MeterSnapshot, request: OperationRequest) -> AdmissionDecision:
        return AdmissionDecision.allow(limits=self._limits)


def make_llm(provider: FakeProvider) -> LLM:
    llm = LLM(MODEL, api_key="test")
    llm._provider = provider  # type: ignore[assignment]  # inject the fake
    return llm


def resp(**usage: int) -> Response:
    return Response(text="ok", usage=Usage(**usage), model=MODEL)


async def test_complete_without_a_scope_is_unchanged():
    llm = make_llm(FakeProvider(response=resp(input_tokens=10)))
    out = await llm.complete("hi")  # no MeterScope bound -> charge site is inert
    assert out.text == "ok"


async def test_complete_meters_one_llm_call_with_cost():
    llm = make_llm(FakeProvider(response=resp(input_tokens=1000, output_tokens=500)))
    with MeterScope() as scope:
        await llm.complete("hi")
    snap = scope.snapshot()
    assert snap.llm_calls == 1 and snap.out_llm_calls == 0
    assert snap.input_tokens == 1000 and snap.output_tokens == 500
    assert snap.cost.pico > 0 and snap.unknown_cost_count == 0


async def test_enforcing_scope_denies_over_the_call_cap():
    prov = FakeProvider(response=resp(input_tokens=10))
    llm = make_llm(prov)
    with (
        MeterScope(RunConfig(controller=CapController(max_llm_calls=0))) as scope,
        pytest.raises(AdmissionDenied),
    ):
        await llm.complete("hi")
    assert scope.snapshot().llm_calls == 0
    assert prov.calls == 0  # denied before the provider was ever touched


async def test_failed_attempt_keeps_the_count_as_unknown_cost():
    prov = FakeProvider(error=ValueError("boom"))  # non-retryable, not a PROVIDER_ERROR
    llm = make_llm(prov)
    with MeterScope() as scope, pytest.raises(ValueError, match="boom"):
        await llm.complete("hi")
    snap = scope.snapshot()
    assert snap.llm_calls == 1 and snap.unknown_cost_count == 1
    assert snap.out_llm_calls == 0 and snap.out_cost == Money.zero()


async def test_runconfig_pricer_overrides_the_default():
    class FixedPricer:
        def price(self, request: OperationRequest, usage: Usage) -> Cost:
            return Cost.known(Money.from_usd(0.42))

    llm = make_llm(FakeProvider(response=resp(input_tokens=100)))
    with MeterScope(RunConfig(pricer=FixedPricer())) as scope:
        await llm.complete("hi")
    assert scope.snapshot().cost == Money.from_usd(0.42)


def test_complete_sync_is_metered_too():
    # Plain sync test: complete_sync -> _run_sync (no running loop -> same-thread asyncio.run),
    # so the scope bound in this thread is visible to the coroutine.
    llm = make_llm(FakeProvider(response=resp(input_tokens=20, output_tokens=5)))
    with MeterScope() as scope:
        llm.complete_sync("hi")
    snap = scope.snapshot()
    assert snap.llm_calls == 1 and snap.input_tokens == 20 and snap.output_tokens == 5


# ── stream / stream_events charge sites ──────────────────────────────────────


class FakeStreamProvider:
    """Yields canned chunks; fills StreamState.usage for the finalizer to settle from."""

    def __init__(self, *, chunks=(), usage: Usage | None = None, error: Exception | None = None):
        self._chunks = list(chunks)
        self._usage = usage or Usage()
        self._error = error

    def stream(self, messages, *, system=None, tools=None, **kwargs):
        if self._error is not None:
            raise self._error
        state = StreamState()
        state.usage = self._usage
        chunks = self._chunks

        async def _aiter():
            for c in chunks:
                yield c

        return _aiter(), state

    def stream_events(self, messages, *, system=None, tools=None, **kwargs):
        if self._error is not None:
            raise self._error
        state = StreamState()
        state.usage = self._usage
        chunks = self._chunks

        async def _aiter():
            for c in chunks:
                yield StreamEvent(kind="text", text=c)

        return _aiter(), state


def make_stream_llm(provider: FakeStreamProvider) -> LLM:
    llm = LLM(MODEL, api_key="test")
    llm._provider = provider  # type: ignore[assignment]
    return llm


async def _drain(stream) -> None:
    async for _ in stream:
        pass


async def test_stream_starts_on_build_and_settles_on_drain():
    prov = FakeStreamProvider(chunks=["a", "b"], usage=Usage(input_tokens=30, output_tokens=10))
    llm = make_stream_llm(prov)
    with MeterScope() as scope:
        stream = llm.stream("hi")
        assert scope.snapshot().llm_calls == 1  # opened + started at build time
        assert scope.snapshot().input_tokens == 0  # usage not known until drained
        await _drain(stream)
        snap = scope.snapshot()
    assert snap.llm_calls == 1 and snap.input_tokens == 30 and snap.output_tokens == 10
    assert snap.cost.pico > 0 and snap.unknown_cost_count == 0


async def test_abandoned_stream_is_incomplete_at_scope_close():
    prov = FakeStreamProvider(chunks=["a"], usage=Usage(input_tokens=5))
    llm = make_stream_llm(prov)
    with MeterScope() as scope:
        llm.stream("hi")  # never drained -> op stays STARTED
    snap = scope.snapshot()
    assert snap.llm_calls == 1 and snap.unknown_cost_count == 1  # incomplete llm -> Unknown
    assert snap.input_tokens == 0  # never settled with usage


async def test_stream_enforce_denies_before_the_provider():
    prov = FakeStreamProvider(chunks=["a"], usage=Usage(input_tokens=5))
    llm = make_stream_llm(prov)
    with (
        MeterScope(RunConfig(controller=CapController(max_llm_calls=0))) as scope,
        pytest.raises(AdmissionDenied),
    ):
        llm.stream("hi")
    assert scope.snapshot().llm_calls == 0


async def test_stream_provider_failure_is_a_failed_attempt():
    prov = FakeStreamProvider(error=ConnectionError("down"))  # a PROVIDER_ERROR, no fallbacks
    llm = make_stream_llm(prov)
    with MeterScope() as scope, pytest.raises(ConnectionError):
        llm.stream("hi")
    snap = scope.snapshot()
    assert snap.llm_calls == 1 and snap.unknown_cost_count == 1


async def test_stream_events_is_metered_on_drain():
    prov = FakeStreamProvider(chunks=["x"], usage=Usage(input_tokens=12, output_tokens=3))
    llm = make_stream_llm(prov)
    with MeterScope() as scope:
        await _drain(llm.stream_events("hi"))
    snap = scope.snapshot()
    assert snap.llm_calls == 1 and snap.input_tokens == 12 and snap.output_tokens == 3


# ── batch fail-closed under an enforcing scope (F3) ──────────────────────────


async def test_batch_submit_blocked_under_an_enforcing_scope():
    prov = FakeProvider()
    llm = make_llm(prov)
    with (
        MeterScope(RunConfig(controller=CapController(max_llm_calls=10))),
        pytest.raises(NotMeteredOperationError),
    ):
        await llm.batch_submit([{"messages": "hi"}])
    assert prov.calls == 0  # rejected before the provider was touched


async def test_batch_submit_allowed_in_measure_only():
    prov = FakeProvider()
    llm = make_llm(prov)
    with MeterScope():  # controller=None -> measure-only, batch simply not metered
        assert await llm.batch_submit([{"messages": "hi"}]) == "batch-123"


async def test_batch_submit_allowed_without_a_scope():
    llm = make_llm(FakeProvider())
    assert await llm.batch_submit([{"messages": "hi"}]) == "batch-123"


def test_batch_submit_sync_is_also_blocked_under_enforcement():
    prov = FakeProvider()
    llm = make_llm(prov)
    with (
        MeterScope(RunConfig(controller=CapController(max_llm_calls=10))),
        pytest.raises(NotMeteredOperationError),
    ):
        llm.batch_submit_sync([{"messages": "hi"}])
    assert prov.calls == 0


async def test_baseexception_fails_the_op_promptly_not_leaked():
    # A cancelled/interrupted attempt (BaseException, not Exception) must fail the op right away,
    # not leak it as STARTED until scope close. Assert INSIDE the scope to distinguish the two.
    class Boom(BaseException):
        pass

    llm = make_llm(FakeProvider(error=Boom()))
    with MeterScope() as scope:
        with pytest.raises(Boom):
            await llm.complete("hi")
        snap = scope.snapshot()  # before close(): op is already failed, not merely started
        assert snap.llm_calls == 1 and snap.unknown_cost_count == 1
        assert snap.out_llm_calls == 0 and snap.out_cost == Money.zero()


# ── request facts: content_size_hint + has_server_tools (review #7) ──────────


async def test_strict_reserve_denies_an_oversized_prompt():
    # content_size_hint is now populated, so strict reserve actually holds input tokens up front
    # (it was always 0 before, admitting prompts that should be denied).
    from ai_arch_toolkit.toolkit.budget import BudgetController, BudgetExceeded, BudgetPolicy

    llm = make_llm(FakeProvider(response=resp(input_tokens=10)))
    policy = BudgetPolicy(reserve="strict", max_input_tokens=10)
    with (
        MeterScope(RunConfig(controller=BudgetController(policy))),
        pytest.raises(BudgetExceeded),
    ):
        await llm.complete("x" * 4000)  # ~4000 chars -> ~1000 estimated input tokens > 10


async def test_server_tool_call_is_costed_unknown():
    from ai_arch_toolkit.core._server_tools import web_search

    llm = make_llm(FakeProvider(response=resp(input_tokens=100, output_tokens=50)))
    with MeterScope() as scope:
        await llm.complete("hi", tools=[web_search()])
    # has_server_tools -> the pricer returns Cost.unknown (surcharge isn't in the token counts),
    # so it is counted as unknown, not silently token-priced.
    assert scope.snapshot().unknown_cost_count == 1
    assert scope.snapshot().cost == Money.zero()


def test_content_hint_skipped_unless_a_strict_reserve_wants_it():
    # Perf (review): _content_chars stringifies the whole request (every message + base64 image);
    # only a strict-reserve estimator reads content_size_hint, so measure-only and soft-budget runs
    # must skip computing it.
    from ai_arch_toolkit.toolkit.budget import BudgetController, BudgetPolicy

    llm = LLM(MODEL, api_key="test")
    msgs = [{"role": "user", "content": "hello world"}]

    def hint_under(config: RunConfig):
        with MeterScope(config):
            req = llm._meter_request("complete", {}, normalized=msgs, system=None, wire_tools=None)
        assert req is not None
        return req.content_size_hint

    assert hint_under(RunConfig()) is None  # measure-only
    soft = RunConfig(controller=BudgetController(BudgetPolicy(max_cost=1.0)))
    assert hint_under(soft) is None  # soft budget (reserve="none")
    strict = RunConfig(controller=BudgetController(BudgetPolicy(max_cost=1.0, reserve="strict")))
    assert (hint_under(strict) or 0) > 0  # strict reserve computes it
