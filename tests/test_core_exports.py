"""Tests for ai_arch_toolkit.core exports."""

from __future__ import annotations

import ai_arch_toolkit.core as core


def test_core_exports_tool_helpers() -> None:
    assert hasattr(core, "tool")
    assert hasattr(core, "infer_schema")
    assert hasattr(core, "prepare_tools")


def test_core_exports_pricing() -> None:
    assert hasattr(core, "pricing")
    cost = core.pricing.estimate_cost("unknown-model", input_tokens=123)
    assert cost is None
