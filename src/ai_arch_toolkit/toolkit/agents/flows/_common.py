"""Shared helpers for flow factories."""

from __future__ import annotations

from typing import TypedDict

from ai_arch_toolkit.core._policy import Policy
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.core._trace import TraceCapture
from ai_arch_toolkit.toolkit.budget import BudgetPolicy

__all__ = ["TOOLS_PLACEHOLDER", "FlowOptions", "nested", "substitute_tools"]

TOOLS_PLACEHOLDER = "{tools}"


class FlowOptions(TypedDict, total=False):
    """The options a flow factory hands, unchanged, to the ``Flow`` it builds.

    Each factory takes them as keyword arguments; one left out keeps the ``Flow`` default.

    Attributes:
        timeout: Wall-clock limit, in seconds, for a whole run (``None``, the default: none).
        trace_capture: What each step's trace records: key names (``"keys"``, the default), deep
            copies (``"full"``), or neither (``"none"``).
        policy: Default policy for each step of the flow.
        budget_policy: Cumulative budget for a run of the flow. ``run(budget_policy=...)``
            overrides it, and a flow run inside another runs under the outer one's.
    """

    timeout: float | None
    trace_capture: TraceCapture
    policy: Policy | None
    budget_policy: BudgetPolicy | None


def nested(options: FlowOptions) -> FlowOptions:
    """What a flow run inside a step of this one takes from its options: the trace capture.

    The inner run's steps are recorded as that step's children, so they record what this flow's
    steps record. The rest stays with this flow: its deadline and budget already cover the inner
    run (the two share the meter scope), and its policy governs its own steps.
    """
    return {"trace_capture": options["trace_capture"]} if "trace_capture" in options else {}


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
    rendered = "\n".join(
        f"- {d['name']}: {d.get('description', 'No description')}" for d in tools.definitions
    )
    return prompt.replace(TOOLS_PLACEHOLDER, rendered or "(none)")
