"""LLM.complete charge site: metered per attempt, enforced, failed-attempt kept, sync + pricer."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._attempts import request_facts
from ai_arch_toolkit.core._exceptions import APIError, TransportError
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
from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._pricing import ModelPricing, pricing
from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._retry import RetryConfig
from tests.fake_provider import MODEL, FakeProvider, Reply, fake_llm  # MODEL is priced


class CapController:
    def __init__(self, **limits: int) -> None:
        self._limits = ResourceLimits(**limits)

    def admit(self, snapshot: MeterSnapshot, request: OperationRequest) -> AdmissionDecision:
        return AdmissionDecision.allow(limits=self._limits)


def resp(**usage: int) -> Response:
    return Response(text="ok", usage=Usage(**usage), model=MODEL)


def streamed(*chunks: str, provider_cost: float | None = None, **usage: int) -> Reply:
    """A streamed answer: ``chunks`` as text events, then the response with ``usage``."""
    response = Response(
        text="".join(chunks), usage=Usage(**usage), provider_cost=provider_cost, model=MODEL
    )
    return Reply(response=response, chunks=chunks)


async def test_complete_without_a_scope_is_unchanged():
    llm, _ = fake_llm(resp(input_tokens=10))
    out = await llm.complete("hi")  # no MeterScope bound -> charge site is inert
    assert out.text == "ok"


async def test_complete_meters_one_llm_call_with_cost():
    llm, _ = fake_llm(resp(input_tokens=1000, output_tokens=500))
    with MeterScope() as scope:
        await llm.complete("hi")
    snap = scope.snapshot()
    assert snap.llm_calls == 1 and snap.out_llm_calls == 0
    assert snap.input_tokens == 1000 and snap.output_tokens == 500
    assert snap.cost.pico > 0 and snap.unknown_cost_count == 0


async def test_default_pricer_prefers_exact_provider_cost():
    response = Response(
        text="ok",
        usage=Usage(input_tokens=100, output_tokens=50),
        provider_cost=0.123456,
        model=MODEL,
    )
    llm, _ = fake_llm(response)
    with MeterScope() as scope:
        await llm.complete("hi")
    assert scope.snapshot().cost == Money.from_usd(0.123456)


async def test_enforcing_scope_denies_over_the_call_cap():
    llm, prov = fake_llm(resp(input_tokens=10))
    with (
        MeterScope(RunConfig(controller=CapController(max_llm_calls=0))) as scope,
        pytest.raises(AdmissionDenied),
    ):
        await llm.complete("hi")
    assert scope.snapshot().llm_calls == 0
    assert prov.calls == 0  # denied before the provider was ever touched


async def test_failed_attempt_keeps_the_count_as_unknown_cost():
    llm, _ = fake_llm(ValueError("boom"))  # non-retryable, not a PROVIDER_ERROR
    with MeterScope() as scope, pytest.raises(ValueError, match="boom"):
        await llm.complete("hi")
    snap = scope.snapshot()
    assert snap.llm_calls == 1 and snap.unknown_cost_count == 1
    assert snap.out_llm_calls == 0 and snap.out_cost == Money.zero()


async def test_runconfig_pricer_overrides_the_default():
    class FixedPricer:
        def price(self, request: OperationRequest, usage: Usage) -> Cost:
            return Cost.known(Money.from_usd(0.42))

    response = Response(
        text="ok",
        usage=Usage(input_tokens=100),
        provider_cost=0.123456,
        model=MODEL,
    )
    llm, _ = fake_llm(response)
    with MeterScope(RunConfig(pricer=FixedPricer())) as scope:
        await llm.complete("hi")
    assert scope.snapshot().cost == Money.from_usd(0.42)


def test_complete_sync_is_metered_too():
    # Plain sync test: complete_sync -> _run_sync (no running loop -> same-thread asyncio.run),
    # so the scope bound in this thread is visible to the coroutine.
    llm, _ = fake_llm(resp(input_tokens=20, output_tokens=5))
    with MeterScope() as scope:
        llm.complete_sync("hi")
    snap = scope.snapshot()
    assert snap.llm_calls == 1 and snap.input_tokens == 20 and snap.output_tokens == 5


# ── stream / stream_events charge sites ──────────────────────────────────────


async def _drain(stream) -> None:
    async for _ in stream:
        pass


async def test_stream_reserves_on_build_starts_on_iteration_and_settles_on_drain():
    llm, _ = fake_llm(streamed("a", "b", input_tokens=30, output_tokens=10))
    with MeterScope() as scope:
        stream = llm.stream("hi")
        built = scope.snapshot()
        assert built.out_llm_calls == 1 and built.llm_calls == 0  # admitted + reserved at build
        await stream.__anext__()
        assert scope.snapshot().llm_calls == 1  # started with the first provider attempt
        assert scope.snapshot().input_tokens == 0  # usage not known until drained
        await _drain(stream)
        snap = scope.snapshot()
    assert snap.llm_calls == 1 and snap.input_tokens == 30 and snap.output_tokens == 10
    assert snap.cost.pico > 0 and snap.unknown_cost_count == 0


async def test_stream_prefers_exact_provider_cost():
    llm, _ = fake_llm(streamed("ok", input_tokens=30, output_tokens=10, provider_cost=0.234567))
    with MeterScope() as scope:
        stream = llm.stream("hi")
        await _drain(stream)
    assert stream.response is not None
    assert stream.response.provider_cost == 0.234567
    assert stream.response.cost == 0.234567
    assert scope.snapshot().cost == Money.from_usd(0.234567)


async def test_never_iterated_stream_releases_its_reservation_at_scope_close():
    llm, _ = fake_llm(streamed("a", input_tokens=5))
    with MeterScope() as scope:
        llm.stream("hi")  # never iterated -> the op stays PENDING, the provider is never called
    snap = scope.snapshot()
    assert snap.llm_calls == 0 and snap.out_llm_calls == 0 and snap.unknown_cost_count == 0


async def test_started_but_undrained_stream_is_incomplete_at_scope_close():
    llm, _ = fake_llm(streamed("a", "b", input_tokens=5))
    with MeterScope() as scope:
        stream = llm.stream("hi")
        await stream.__anext__()  # started, then left undrained and unclosed
        del stream
    snap = scope.snapshot()
    assert snap.llm_calls == 1 and snap.unknown_cost_count == 1  # count kept, cost unknown
    assert snap.input_tokens == 0  # never settled with usage


async def test_stream_enforce_denies_before_the_provider():
    llm, _ = fake_llm(streamed("a", input_tokens=5))
    with (
        MeterScope(RunConfig(controller=CapController(max_llm_calls=0))) as scope,
        pytest.raises(AdmissionDenied),
    ):
        llm.stream("hi")
    assert scope.snapshot().llm_calls == 0


async def test_stream_provider_failure_is_a_failed_attempt():
    llm, _ = fake_llm(TransportError("down"))  # a PROVIDER_ERROR, no fallbacks
    with MeterScope() as scope, pytest.raises(ConnectionError):
        await _drain(llm.stream("hi"))
    snap = scope.snapshot()
    assert snap.llm_calls == 1 and snap.unknown_cost_count == 1


async def test_stream_retry_meters_every_physical_attempt(monkeypatch):
    async def _no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("ai_arch_toolkit.core._retry.asyncio.sleep", _no_sleep)
    # The first stream fails before its first event; the retry answers.
    llm, prov = fake_llm(
        APIError(500, "temporary"), streamed("ok", input_tokens=20, output_tokens=4)
    )
    llm._retry = RetryConfig(max_retries=1, base_delay=0.01)

    with MeterScope() as scope:
        stream = llm.stream("hi")
        await _drain(stream)

    snap = scope.snapshot()
    assert prov.calls == 2
    assert snap.llm_calls == 2
    assert snap.unknown_cost_count == 1
    assert snap.input_tokens == 20 and snap.output_tokens == 4
    assert stream.response is not None
    assert [attempt.status for attempt in stream.response.attempts] == ["failed", "ok"]


async def test_stream_retry_admission_denial_is_terminal(monkeypatch):
    async def _no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("ai_arch_toolkit.core._retry.asyncio.sleep", _no_sleep)
    llm, prov = fake_llm(APIError(500, "temporary"))  # every stream fails before its first event
    llm._retry = RetryConfig(max_retries=1, base_delay=0.01)

    with (
        MeterScope(RunConfig(controller=CapController(max_llm_calls=1))) as scope,
        pytest.raises(AdmissionDenied),
    ):
        await _drain(llm.stream("hi"))

    assert prov.calls == 1
    assert scope.snapshot().llm_calls == 1
    assert scope.snapshot().unknown_cost_count == 1


async def test_stream_events_is_metered_on_drain():
    llm, _ = fake_llm(streamed("x", input_tokens=12, output_tokens=3))
    with MeterScope() as scope:
        await _drain(llm.stream_events("hi"))
    snap = scope.snapshot()
    assert snap.llm_calls == 1 and snap.input_tokens == 12 and snap.output_tokens == 3


# ── provider attribution on metering events (F10) ────────────────────────────


async def test_complete_event_names_the_provider():
    llm, _ = fake_llm(resp(input_tokens=10))
    with MeterScope(RunConfig(retain_meter_events=True)) as scope:
        await llm.complete("hi")
    assert [(e.model, e.provider) for e in scope.events()] == [(MODEL, "anthropic")]


async def test_stream_event_names_the_provider():
    llm, _ = fake_llm(streamed("a", input_tokens=3))
    with MeterScope(RunConfig(retain_meter_events=True)) as scope:
        await _drain(llm.stream("hi"))
    assert [(e.status, e.provider) for e in scope.events()] == [("settled", "anthropic")]


async def test_openai_compatible_server_is_attributed_to_openai():
    llm = LLM("gemma4:e4b", base_url="http://localhost:11434/v1")
    llm._provider = FakeProvider(resp(input_tokens=1), model="gemma4:e4b")
    pricing.register("gemma4:e4b", ModelPricing())  # a local model's price is an explicit zero
    try:
        with MeterScope(RunConfig(retain_meter_events=True)) as scope:
            await llm.complete("hi")
    finally:
        pricing.unregister("gemma4:e4b")
    assert [e.provider for e in scope.events()] == ["openai"]


async def test_cross_provider_fallback_attempt_names_its_own_provider():
    fallback = LLM("gpt-4o", api_key="test")
    fallback._provider = FakeProvider(resp(input_tokens=1), model="gpt-4o")
    llm = LLM(MODEL, api_key="test", fallback=fallback)
    llm._provider = FakeProvider(APIError(500, "down"), model=MODEL)
    with MeterScope(RunConfig(retain_meter_events=True)) as scope:
        await llm.complete("hi")
    assert [(e.status, e.model, e.provider) for e in scope.events()] == [
        ("failed", MODEL, "anthropic"),
        ("settled", "gpt-4o", "openai"),
    ]


# ── batch fail-closed under an enforcing scope (F3) ──────────────────────────


async def test_batch_submit_blocked_under_an_enforcing_scope():
    llm, prov = fake_llm()
    with (
        MeterScope(RunConfig(controller=CapController(max_llm_calls=10))),
        pytest.raises(NotMeteredOperationError),
    ):
        await llm.batch_submit([{"messages": "hi"}])
    assert prov.batches == []  # rejected before the provider was touched


async def test_batch_submit_allowed_in_measure_only():
    llm, _ = fake_llm()
    with MeterScope():  # controller=None -> measure-only, batch simply not metered
        assert await llm.batch_submit([{"messages": "hi"}]) == "batch-1"


async def test_batch_submit_allowed_without_a_scope():
    llm, _ = fake_llm()
    assert await llm.batch_submit([{"messages": "hi"}]) == "batch-1"


def test_batch_submit_sync_is_also_blocked_under_enforcement():
    llm, prov = fake_llm()
    with (
        MeterScope(RunConfig(controller=CapController(max_llm_calls=10))),
        pytest.raises(NotMeteredOperationError),
    ):
        llm.batch_submit_sync([{"messages": "hi"}])
    assert prov.batches == []


async def test_baseexception_fails_the_op_promptly_not_leaked():
    # A cancelled/interrupted attempt (BaseException, not Exception) must fail the op right away,
    # not leak it as STARTED until scope close. Assert INSIDE the scope to distinguish the two.
    class Boom(BaseException):
        pass

    llm, _ = fake_llm(Boom())
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

    llm, _ = fake_llm(resp(input_tokens=10))
    policy = BudgetPolicy(reserve="strict", max_input_tokens=10)
    with (
        MeterScope(RunConfig(controller=BudgetController(policy))),
        pytest.raises(BudgetExceeded),
    ):
        await llm.complete("x" * 4000)  # ~4000 chars -> ~1000 estimated input tokens > 10


async def test_server_tool_call_is_costed_unknown():
    from ai_arch_toolkit.core._server_tools import web_search

    llm, _ = fake_llm(resp(input_tokens=100, output_tokens=50))
    with MeterScope() as scope:
        await llm.complete("hi", tools=[web_search()])
    # has_server_tools -> the pricer returns Cost.unknown (surcharge isn't in the token counts),
    # so it is counted as unknown, not silently token-priced.
    assert scope.snapshot().unknown_cost_count == 1
    assert scope.snapshot().cost == Money.zero()


def test_content_hint_skipped_unless_a_strict_reserve_wants_it():
    # Perf (review): _request_size stringifies the whole request (every message + base64 image);
    # only a strict-reserve estimator reads content_size_hint, so measure-only and soft-budget runs
    # must skip computing it.
    from ai_arch_toolkit.toolkit.budget import BudgetController, BudgetPolicy

    llm = LLM(MODEL, api_key="test")
    msgs = [{"role": "user", "content": "hello world"}]

    def hint_under(config: RunConfig):
        with MeterScope(config) as scope:
            request = Request(messages=msgs, system=None, tools=None, model=MODEL)
            req = request_facts(llm, request, "complete", scope)
        assert req is not None
        return req.content_size_hint

    assert hint_under(RunConfig()) is None  # measure-only
    soft = RunConfig(controller=BudgetController(BudgetPolicy(max_cost=1.0)))
    assert hint_under(soft) is None  # soft budget (reserve="none")
    strict = RunConfig(controller=BudgetController(BudgetPolicy(max_cost=1.0, reserve="strict")))
    assert (hint_under(strict) or 0) > 0  # strict reserve computes it


async def test_meter_request_tolerates_a_non_callable_wants_request_size():
    # A custom controller exposing wants_request_size as a NON-callable attribute must not crash
    # the charge site — it falls back to computing the hint (safe default).
    class _WeirdController:
        wants_request_size = True  # an attribute, not a method

        def admit(self, snapshot, request) -> AdmissionDecision:
            return AdmissionDecision.allow()

    llm, _ = fake_llm(resp(input_tokens=1))
    msgs = [{"role": "user", "content": "hi"}]
    with MeterScope(RunConfig(controller=_WeirdController())) as scope:
        request = Request(messages=msgs, system=None, tools=None, model=MODEL)
        req = request_facts(llm, request, "complete", scope)
    assert req is not None and req.content_size_hint is not None  # no crash; hint computed
