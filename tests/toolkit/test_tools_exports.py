"""Tests for toolkit.tools safe and dangerous export surfaces."""

from __future__ import annotations

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


def test_default_tools_do_not_export_dangerous_tools() -> None:
    assert DANGEROUS_TOOL_NAMES.isdisjoint(safe_tools.__all__)
    for name in DANGEROUS_TOOL_NAMES:
        assert not hasattr(safe_tools, name)


def test_dangerous_tools_are_explicit_opt_in_exports() -> None:
    assert set(dangerous.__all__) == DANGEROUS_TOOL_NAMES
    for name in DANGEROUS_TOOL_NAMES:
        assert hasattr(dangerous, name)


def test_safe_tools_remain_in_default_exports() -> None:
    for name in ("datetime_now", "math_eval", "get_weather", "wikipedia_search"):
        assert name in safe_tools.__all__
        assert hasattr(safe_tools, name)
