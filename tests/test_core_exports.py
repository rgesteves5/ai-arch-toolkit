"""Tests for ai_arch_toolkit.core exports."""

from __future__ import annotations

from types import ModuleType

import pytest

import ai_arch_toolkit
import ai_arch_toolkit.core as core
import ai_arch_toolkit.core._tools as core_tools


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


# Everything needed to write a custom gate for ToolGroup(gates=[...]) without reaching into `_…`.
_GATE_SURFACE = [
    "ExecutionContext",
    "GateBlock",
    "GateDryRun",
    "GateModify",
    "GateResult",
    "ToolGate",
]


@pytest.mark.parametrize("module", [core_tools, core, ai_arch_toolkit], ids=lambda m: m.__name__)
@pytest.mark.parametrize("name", _GATE_SURFACE)
def test_gate_surface_is_exported(module: ModuleType, name: str) -> None:
    assert hasattr(module, name), f"{name} is not importable from {module.__name__}"
    assert name in module.__all__, f"{name} is missing from {module.__name__}.__all__"


def test_custom_gate_built_from_public_imports_blocks_a_call() -> None:
    from ai_arch_toolkit import (
        ExecutionContext,
        GateBlock,
        GateResult,
        ToolCall,
        ToolGate,
        ToolGroup,
        tool,
    )

    @tool(capability="write")
    def save_note(text: str) -> str:
        """Save a note."""
        return f"saved {text}"

    @tool
    def read_notes() -> str:
        """Read all notes."""
        return "no notes"

    class ReadOnlyGate:
        def check_sync(self, ctx: ExecutionContext) -> GateResult | None:
            if ctx.definition.policy.capability == "write":
                return GateBlock(error_type="dangerous_tool_blocked", message="Read-only mode.")
            return None

        async def check(self, ctx: ExecutionContext) -> GateResult | None:
            return self.check_sync(ctx)

    gate = ReadOnlyGate()
    assert isinstance(gate, ToolGate)

    group = ToolGroup(save_note, read_notes, gates=[gate])
    blocked = group.execute(ToolCall(id="t1", name="save_note", input={"text": "hi"}))
    assert blocked.ok is False
    assert blocked.error is not None
    assert blocked.error.type == "dangerous_tool_blocked"
    assert blocked.error.message == "Read-only mode."

    allowed = group.execute(ToolCall(id="t2", name="read_notes", input={}))
    assert allowed.ok is True
    assert allowed.value == "no notes"
