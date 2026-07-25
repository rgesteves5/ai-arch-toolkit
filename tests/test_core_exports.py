"""Tests for ai_arch_toolkit.core exports."""

from __future__ import annotations

import ai_arch_toolkit.core as core


def test_core_exports_tool_helpers() -> None:
    assert hasattr(core, "tool")
    assert hasattr(core, "infer_schema")
    assert hasattr(core, "prepare_tools")


def test_core_exports_pricing() -> None:
    assert hasattr(core, "pricing")
    custom = core.PricingRegistry()
    custom.register("custom-model", core.ModelPricing(input=1.0, output=2.0))
    assert custom.has("custom-model-v1")
    cost = core.pricing.estimate_cost("unknown-model", input_tokens=123)
    assert cost is None
