"""Shared helpers for flow factories."""

from __future__ import annotations

from ai_arch_toolkit.core._tools._group import ToolGroup

__all__ = ["TOOLS_PLACEHOLDER", "substitute_tools"]

TOOLS_PLACEHOLDER = "{tools}"


def substitute_tools(prompt: str, tools: ToolGroup) -> str:
    """Replace a literal ``{tools}`` token with the rendered tool catalog.

    This is the only substitution flow factories perform on prompts. A prompt
    without the token passes through byte-identical — the framework never
    appends tool text a caller did not declare. An empty group renders
    ``(none)``. Exact-token replacement (not ``str.format``) keeps other braces
    in prompts, such as JSON examples or ``#E{n}`` syntax, untouched.
    """
    if TOOLS_PLACEHOLDER not in prompt:
        return prompt
    definitions = tools.definitions if hasattr(tools, "definitions") else []
    rendered = "\n".join(
        f"- {d['name']}: {d.get('description', 'No description')}" for d in definitions
    )
    return prompt.replace(TOOLS_PLACEHOLDER, rendered or "(none)")
