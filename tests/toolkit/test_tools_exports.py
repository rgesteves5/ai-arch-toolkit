"""Tests for toolkit.tools safe and dangerous export surfaces."""

from __future__ import annotations

import pytest

import ai_arch_toolkit.toolkit.tools as safe_tools
from ai_arch_toolkit.toolkit.tools import dangerous

DANGEROUS_TOOL_NAMES = frozenset(
    {
        "http_get",
        "list_directory",
        "python_repl",
        "read_file",
        "run_command",
        "scrape_text",
        "search_files",
    }
)

# name -> (capability, risk_level)
DANGEROUS_TOOL_RISK = {
    "http_get": ("network", "high"),
    "list_directory": ("filesystem", "high"),
    "python_repl": ("python", "high"),
    "read_file": ("filesystem", "high"),
    "run_command": ("shell", "critical"),
    "scrape_text": ("network", "high"),
    "search_files": ("filesystem", "high"),
}


def test_default_tools_do_not_export_dangerous_tools() -> None:
    assert DANGEROUS_TOOL_NAMES.isdisjoint(safe_tools.__all__)
    for name in DANGEROUS_TOOL_NAMES:
        assert not hasattr(safe_tools, name)


def test_dangerous_tools_are_explicit_opt_in_exports() -> None:
    assert set(dangerous.__all__) == DANGEROUS_TOOL_NAMES
    for name in DANGEROUS_TOOL_NAMES:
        assert hasattr(dangerous, name)


@pytest.mark.parametrize("name", sorted(DANGEROUS_TOOL_NAMES))
def test_dangerous_tools_declare_risk_and_require_approval(name: str) -> None:
    capability, risk_level = DANGEROUS_TOOL_RISK[name]
    policy = getattr(dangerous, name).__tool_definition__.policy
    assert policy.capability == capability
    assert policy.risk_level == risk_level
    assert policy.requires_approval is True
    assert policy.approval_reason


def test_safe_tools_remain_in_default_exports() -> None:
    for name in ("datetime_now", "math_eval", "get_weather", "wikipedia_search"):
        assert name in safe_tools.__all__
        assert hasattr(safe_tools, name)
