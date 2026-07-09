"""PricingRegistry.price: the default Pricer bridging token usage -> a typed Cost."""

from __future__ import annotations

from ai_arch_toolkit.core._metering._cost import Cost
from ai_arch_toolkit.core._metering._money import Money
from ai_arch_toolkit.core._metering._operation import OperationRequest
from ai_arch_toolkit.core._pricing import PricingRegistry
from ai_arch_toolkit.core._response import Usage

PRICED_MODEL = "claude-sonnet-4-6"  # ships in _default_pricing.toml


def llm(**kw) -> OperationRequest:
    return OperationRequest(kind="llm", parent_span_id="run", **kw)


def test_price_known_model_wraps_estimate_cost_exactly():
    reg = PricingRegistry()
    usage = Usage(input_tokens=1000, output_tokens=500)
    usd = reg.estimate_cost(PRICED_MODEL, input_tokens=1000, output_tokens=500)
    assert usd is not None
    cost = reg.price(llm(model=PRICED_MODEL), usage)
    assert cost == Cost.known(Money.from_usd(usd))


def test_price_unknown_model_is_unknown_not_zero():
    reg = PricingRegistry()
    cost = reg.price(llm(model="totally-made-up-model-xyz"), Usage(input_tokens=10))
    assert cost.kind == "unknown" and "totally-made-up-model-xyz" in (cost.reason or "")


def test_price_without_a_model_is_unknown():
    reg = PricingRegistry()
    cost = reg.price(llm(), Usage(input_tokens=10))
    assert cost.kind == "unknown"


def test_server_tools_poison_the_price_even_for_a_known_model():
    reg = PricingRegistry()
    cost = reg.price(llm(model=PRICED_MODEL, has_server_tools=True), Usage(input_tokens=10))
    assert cost.kind == "unknown"


def test_non_llm_operation_is_free():
    reg = PricingRegistry()
    cost = reg.price(OperationRequest(kind="tool", parent_span_id="run"), Usage())
    assert cost == Cost.known(Money.zero())


def test_registry_is_usable_as_a_pricer_via_runconfig():
    from ai_arch_toolkit.core._metering._scope import MeterScope, RunConfig

    reg = PricingRegistry()
    with MeterScope(RunConfig(pricer=reg)) as scope:
        assert scope.pricer is reg
        priced = scope.pricer.price(llm(model=PRICED_MODEL), Usage(input_tokens=100))
    assert priced.is_known
