"""Thinking systems catalog — loadable reasoning strategy registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ai_arch_toolkit.core import tool


def load_thinking_systems(path: str | Path | None = None) -> dict[str, dict[str, str]]:
    """Load thinking systems from a YAML file.

    Args:
        path: Path to YAML file. Defaults to thinking_systems.yaml in this package.

    Returns:
        Dict mapping system name to {summary, strategy, example}.
    """
    if path is None:
        path = Path(__file__).parent / "thinking_systems.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def make_thinking_system_tool(catalog: dict[str, dict[str, str]]) -> Any:
    """Create a thinking_system tool backed by the given catalog.

    Args:
        catalog: Dict mapping system name to {summary, strategy, example}.
    """

    @tool
    def thinking_system(ts_names: list[str] | None = None) -> str:
        """Browse reasoning strategies for the current task.

        Call with no arguments to see all available thinking systems.
        Call with specific names to get detailed strategy and worked example.

        Args:
            ts_names: Optional list of thinking system names to study in detail.
        """
        if not ts_names:
            lines = ["Available thinking systems:"]
            for name, entry in catalog.items():
                lines.append(f"- {name}: {entry['summary']}")
            lines.append("")
            lines.append("Call again with ts_names to get strategy + example.")
            return "\n".join(lines)

        lines = []
        for name in ts_names:
            entry = catalog.get(name)
            if entry is None:
                lines.append(f"## {name}\nUnknown system. Available: {list(catalog)}")
            else:
                lines.append(f"## {name}")
                lines.append(f"**Summary:** {entry['summary']}")
                lines.append(f"\n**Strategy:**\n{entry['strategy'].strip()}")
                lines.append(f"\n**Example:**\n{entry['example'].strip()}")
            lines.append("")
        return "\n".join(lines)

    return thinking_system
