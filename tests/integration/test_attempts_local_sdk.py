"""Official SDK errors through the R01 pipeline, using only a loopback fake server."""

from __future__ import annotations

import json

import pytest

from ai_arch_toolkit.core import (
    LLM,
    APIError,
    MeterScope,
    ProviderTimeout,
    RateLimitError,
    RunConfig,
)
from ai_arch_toolkit.toolkit.budget import BudgetController, BudgetPolicy
from tests.integration import fakeserver

pytestmark = pytest.mark.integration


def client(vendor: str, port: int, *, timeout: float = 2.0) -> LLM:
    """Disable SDK retries so each observed request is one physical pipeline attempt."""
    base_url = f"http://127.0.0.1:{port}"
    model = "claude-sonnet-4-6" if vendor == "anthropic" else "gpt-4o"
    llm = LLM(model, api_key="local-test", base_url=base_url, max_tokens=32, timeout=timeout)
    llm._provider._client = llm._provider._client.with_options(max_retries=0)
    return llm


async def invoke(llm: LLM, path: str) -> None:
    if path == "complete":
        await llm.complete("hi")
        return
    stream = llm.stream("hi") if path == "stream" else llm.stream_events("hi")
    async with stream:
        async for _ in stream:
            pass


@pytest.mark.parametrize("vendor", ("anthropic", "openai"))
@pytest.mark.parametrize("path", ("complete", "stream", "stream_events"))
@pytest.mark.parametrize("status", (429, 503))
async def test_local_sdk_error_delivery_and_bounded_meter(vendor, path, status):
    server, port, stats = await fakeserver.start(
        "status",
        status=status,
        body={"type": "error", "error": {"type": "rate_limit_error", "message": "busy"}},
    )
    controller = BudgetController(BudgetPolicy(max_cost=5.0))
    error_type = RateLimitError if status == 429 else APIError
    async with server, client(vendor, port) as llm:
        with MeterScope(RunConfig(controller=controller, retain_meter_events=True)) as scope:
            with pytest.raises(error_type) as raised:
                await invoke(llm, path)
            assert raised.value.delivery == ("unbilled" if status == 429 else "indeterminate")
            snap = scope.snapshot()
            assert snap.llm_calls == 1
            assert snap.cost.to_float() == 0.0
            assert snap.unknown_cost_count == 0
            assert snap.uncertain_cost_count == int(status == 503)
            assert (
                snap.uncertain_cost.to_float() > 0
                if status == 503
                else snap.uncertain_cost.pico == 0
            )
            assert scope.events()[0].delivery == raised.value.delivery
            assert not scope.has_live_ops(scope.run_span_id)
    assert stats.requests == 1
    body = json.loads(stats.bodies[0])
    assert body["model"] == llm._model
    assert body["temperature"] == 0.0  # the LLM default reaches the wire through the real SDK


@pytest.mark.parametrize("vendor", ("anthropic", "openai"))
async def test_local_sdk_timeout_retains_strict_reservation(vendor):
    server, port, stats = await fakeserver.start("hang", seconds=0.3)
    controller = BudgetController(BudgetPolicy(max_cost=5.0, reserve="strict"))
    async with server, client(vendor, port, timeout=0.1) as llm:
        with MeterScope(RunConfig(controller=controller)) as scope:
            with pytest.raises(ProviderTimeout) as raised:
                await llm.complete("hi")
            assert raised.value.delivery == "indeterminate"
            assert scope.snapshot().uncertain_cost.to_float() > 0
            assert scope.snapshot().unknown_cost_count == 0
            assert scope.snapshot().llm_calls == 1
            assert not scope.has_live_ops(scope.run_span_id)
    assert stats.requests == 1
