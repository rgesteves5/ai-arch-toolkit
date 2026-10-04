"""Every failure has a ceiling, metered with or without a budget (D49, the ai-network's G-29)."""

from __future__ import annotations

import pytest
from tests.fake_provider import MODEL, fake_llm

from ai_arch_toolkit.core import (
    MeterScope,
    Money,
    OperationRequest,
    Policy,
    Response,
    Result,
    Step,
    Usage,
    web_search,
)
from ai_arch_toolkit.core._exceptions import APIError
from ai_arch_toolkit.core._metering._worst_case import (
    CHARS_PER_TOKEN,
    IMAGE_OUTPUT_TOKEN_ALLOWANCE,
    MEDIA_TOKEN_ALLOWANCE,
    worst_case,
)
from ai_arch_toolkit.core._pricing import ModelPricing, PricingRegistry, pricing
from ai_arch_toolkit.core._retry import RetryConfig
from ai_arch_toolkit.core._state import StateSnapshot
from ai_arch_toolkit.core._step_engine import execute_step


def _registry() -> PricingRegistry:
    registry = PricingRegistry()
    registry.register("m", ModelPricing(input=1.0, output=2.0, image_output=30.0, per_image=0.5))
    return registry


def _llm(**facts: object) -> OperationRequest:
    return OperationRequest(kind="llm", parent_span_id="run", model="m", **facts)


def test_the_worst_case_holds_every_input_output_and_image_the_request_may_use() -> None:
    request = _llm(
        content_size_hint=401,
        non_text_parts=2,
        declared_max_output_tokens=100,
        declared_images=1,
    )
    found = worst_case(request, _registry())
    assert found is not None
    inputs = -(-401 // CHARS_PER_TOKEN) + 2 * MEDIA_TOKEN_ALLOWANCE
    assert (found.input_tokens, found.output_tokens) == (
        inputs,
        100 + IMAGE_OUTPUT_TOKEN_ALLOWANCE,
    )
    expected = (inputs * 1.0 + 100 * 2.0 + IMAGE_OUTPUT_TOKEN_ALLOWANCE * 30.0) / 1e6 + 0.5
    assert found.cost == Money.from_usd(expected)


def test_what_cannot_be_priced_has_no_worst_case() -> None:
    assert worst_case(_llm(declared_max_output_tokens=10), PricingRegistry()) is None

    class Raising:
        def price(self, request, usage):
            raise RuntimeError("no price")

    assert worst_case(_llm(), Raising()) is None


async def test_a_measured_failure_is_bounded_by_its_worst_case() -> None:
    llm, _ = fake_llm(APIError(503, "busy"))
    with MeterScope() as scope, pytest.raises(APIError):
        await llm.complete("hi")
    snap = scope.snapshot()
    assert (snap.unknown_cost_count, snap.uncertain_cost_count) == (0, 1)
    assert snap.cost == Money.zero()
    # The bound is the worst case of the request at the table's prices: its few input tokens
    # plus the LLM's max_tokens (4096) of output.
    output_only = pricing.estimate_cost(MODEL, output_tokens=4096)
    with_input = pricing.estimate_cost(MODEL, input_tokens=100, output_tokens=4096)
    assert output_only is not None and with_input is not None
    assert Money.from_usd(output_only) < snap.uncertain_cost < Money.from_usd(with_input)


async def test_a_failure_with_a_hosted_tool_stays_unknown() -> None:
    # A provider-hosted tool's charge is not in the token counts: no ceiling can be priced.
    llm, _ = fake_llm(APIError(503, "busy"))
    with MeterScope() as scope, pytest.raises(APIError):
        await llm.complete("hi", tools=[web_search()])
    snap = scope.snapshot()
    assert (snap.unknown_cost_count, snap.uncertain_cost_count) == (1, 0)


async def test_a_step_cap_passes_after_a_retry_recovered_a_failed_attempt() -> None:
    # The finding of 2026-09-18: without a budget, the failed attempt was unbounded, and the
    # step's max_cost failed closed although the retry succeeded.
    answer = Response(text="ok", usage=Usage(input_tokens=10, output_tokens=5), model=MODEL)
    llm, provider = fake_llm(
        APIError(503, "busy"), answer, retry=RetryConfig(max_retries=1, base_delay=0.001)
    )

    async def call(snapshot: StateSnapshot) -> Result:
        response = await llm.complete("hi")
        return Result(value=response.text)

    step = Step(name="s", fn=call, policy=Policy(max_cost=1.0))
    with MeterScope() as scope:
        result, trace = await execute_step(step, StateSnapshot())
    assert provider.calls == 2
    assert not result.is_error and result.value == "ok"
    assert "cost_exceeded" not in trace.policy_decisions
    assert scope.snapshot().uncertain_cost_count == 1
