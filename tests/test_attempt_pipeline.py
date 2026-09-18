"""Regressions for the shared physical-attempt lifecycle."""

from __future__ import annotations

import asyncio

import pytest

from ai_arch_toolkit.core import MeterScope, RunConfig, inference_limit
from ai_arch_toolkit.core._attempts import StreamAbandoned
from ai_arch_toolkit.core._exceptions import APIError
from tests.test_failure_matrix import ScriptedProvider, configured, invoke


@pytest.mark.parametrize("path", ("complete", "stream", "stream_events"))
async def test_cancelled_inference_queue_never_starts_or_prices_an_attempt(path: str) -> None:
    holder = ScriptedProvider("cancel")
    queued = ScriptedProvider(None)
    with MeterScope(RunConfig(retain_meter_events=True)) as scope, inference_limit(1):
        first = asyncio.create_task(configured(holder, "none").complete("hi"))
        await holder.entered.wait()
        second = asyncio.create_task(invoke(configured(queued, "none"), path))
        await asyncio.sleep(0)
        assert scope.snapshot().out_llm_calls == 1
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        assert queued.calls == 0
        assert scope.snapshot().llm_calls == 1
        assert scope.snapshot().unknown_cost_count == 0
        holder.release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
    assert [event.status for event in scope.events()] == ["aborted", "failed"]


@pytest.mark.parametrize("path", ("stream", "stream_events"))
async def test_stream_abandoned_in_the_inference_queue_ends_typed_and_unstarted(path: str) -> None:
    # SyncStreamResponse.close() abandons through the lifecycle handle from the consumer thread
    # before it cancels the worker, so the worker can still win the slot afterwards.
    holder = ScriptedProvider("cancel")
    queued = ScriptedProvider(None)
    with MeterScope(RunConfig(retain_meter_events=True)) as scope, inference_limit(1):
        first = asyncio.create_task(configured(holder, "none").complete("hi"))
        await holder.entered.wait()
        llm = configured(queued, "none")
        stream = llm.stream("hi") if path == "stream" else llm.stream_events("hi")
        consumer = asyncio.create_task(anext(aiter(stream)))
        await asyncio.sleep(0)
        stream._abandon()
        holder.release.set()
        with pytest.raises(StreamAbandoned):
            await consumer
        with pytest.raises(asyncio.CancelledError):
            await first
    assert queued.calls == 0
    assert scope.snapshot().llm_calls == 1  # only the holder started
    assert [event.status for event in scope.events()] == ["aborted", "failed"]


@pytest.mark.parametrize("path", ("complete", "stream", "stream_events"))
async def test_all_intermediate_fallback_attempts_are_retained(path: str) -> None:
    last = configured(ScriptedProvider(None), "none")
    middle = configured(ScriptedProvider("5xx"), "none", last)
    primary = configured(ScriptedProvider("5xx"), "none", middle)
    if path == "complete":
        response = await primary.complete("hi")
    else:
        stream = primary.stream("hi") if path == "stream" else primary.stream_events("hi")
        async for _ in stream:
            pass
        response = stream.response
    assert response is not None
    assert [attempt.status for attempt in response.attempts] == ["failed", "failed", "ok"]
    assert [attempt.status_code for attempt in response.attempts] == [503, 503, None]


async def test_429_preserves_count_but_does_not_poison_an_enforcing_scope() -> None:
    from tests.test_failure_matrix import meter_context

    provider = ScriptedProvider("429")
    with meter_context("soft") as scope:
        with pytest.raises(APIError):
            await invoke(configured(provider, "none"), "complete")
        assert scope.snapshot().llm_calls == 1
        assert scope.snapshot().unknown_cost_count == 0
        assert scope.snapshot().uncertain_cost_count == 0
        assert await invoke(configured(provider, "none"), "complete") == "ok"


async def test_empty_completion_keeps_provider_usage_and_metadata() -> None:
    from ai_arch_toolkit.core import Response, Usage

    class EmptyProvider(ScriptedProvider):
        async def complete(self, messages, *, system=None, tools=None, **kwargs):
            await self.send()
            return Response(
                usage=Usage(input_tokens=4, output_tokens=2),
                provider_cost=0.0123,
                stop_reason="length",
                response_id="empty-turn",
            )

    with MeterScope() as scope:
        response = await configured(EmptyProvider(None), "none").complete("hi")
    assert response.text == ""
    assert response.usage == Usage(input_tokens=4, output_tokens=2)
    assert response.response_id == "empty-turn"
    assert response.stop_reason == "length"
    assert scope.snapshot().cost.to_float() == 0.0123


@pytest.mark.parametrize("method", ("stream_sync", "stream_events_sync"))
def test_sync_abandonment_reports_only_consumed_text_after_worker_finishes(method: str) -> None:
    import threading

    from ai_arch_toolkit.core import Money
    from ai_arch_toolkit.core._providers._base import StreamState
    from tests.test_failure_matrix import USAGE, meter_context

    finished = threading.Event()

    class FastProvider(ScriptedProvider):
        def stream(self, messages, *, system=None, tools=None, **kwargs):
            state = StreamState()

            async def chunks():
                self.calls += 1
                state.usage = USAGE
                try:
                    yield "first"
                    yield "last"
                finally:
                    finished.set()

            return chunks(), state

    llm = configured(FastProvider(None), "none")
    with meter_context("strict") as scope:
        stream = getattr(llm, method)("hi")
        iterator = iter(stream)
        first = next(iterator)
        assert first == "first" if method == "stream_sync" else first.text == "first"
        assert finished.wait(timeout=1.0)
        stream.close()
        assert stream.response is not None
        assert stream.response.text == "first"
        assert stream.response.attempts[0].error_type == "StreamAbandoned"
        assert scope.snapshot().cost == Money.zero()
        assert scope.snapshot().uncertain_cost_count == 1
        assert not scope.has_live_ops(scope.run_span_id)
