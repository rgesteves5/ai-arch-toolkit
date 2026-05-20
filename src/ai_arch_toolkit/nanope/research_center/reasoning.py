"""Reasoning strategies catalog — loadable thinking system tool for research agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ai_arch_toolkit.core import tool

_DEFAULT_YAML = Path(__file__).parent / "thinking_systems.yaml"


def load_reasoning_systems(path: str | Path | None = None) -> dict[str, dict[str, str]]:
    """Load reasoning systems from a YAML file.

    Args:
        path: Path to YAML file. Defaults to thinking_systems.yaml in this package.
    """
    if path is None:
        path = _DEFAULT_YAML
    with open(path) as f:
        return yaml.safe_load(f)


def make_reasoning_tool(catalog: dict[str, dict[str, str]] | None = None) -> Any:
    """Create a reasoning strategy tool backed by a catalog.

    Args:
        catalog: Dict mapping system name to {summary, strategy, example}.
                 Defaults to the bundled thinking_systems.yaml.
    """
    if catalog is None:
        catalog = load_reasoning_systems()

    @tool
    def reasoning_strategy(strategy_names: list[str] | None = None) -> str:
        """Browse reasoning strategies to improve your thinking on the current task.

        Call with no arguments to see all available strategies.
        Call with specific names to get detailed strategy and worked example.

        Args:
            strategy_names: Optional list of strategy names to study in detail.
        """
        if not strategy_names:
            lines = ["Available reasoning strategies:"]
            for name, entry in catalog.items():
                lines.append(f"- {name}: {entry['summary']}")
            lines.append("")
            lines.append("Call again with strategy_names to get full strategy + example.")
            return "\n".join(lines)

        lines = []
        for name in strategy_names:
            entry = catalog.get(name)
            if entry is None:
                lines.append(f"## {name}\nUnknown strategy. Available: {list(catalog)}")
            else:
                lines.append(f"## {name}")
                lines.append(f"**Summary:** {entry['summary']}")
                lines.append(f"\n**Strategy:**\n{entry['strategy'].strip()}")
                lines.append(f"\n**Example:**\n{entry['example'].strip()}")
            lines.append("")
        return "\n".join(lines)

    return reasoning_strategy
