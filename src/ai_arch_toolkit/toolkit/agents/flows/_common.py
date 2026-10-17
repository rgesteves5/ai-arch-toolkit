"""Shared helpers for flow factories."""

from __future__ import annotations

import inspect
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, TypedDict

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._policy import Policy
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.core._trace import TraceCapture
from ai_arch_toolkit.toolkit._shape import Free
from ai_arch_toolkit.toolkit.budget import BudgetPolicy

__all__ = [
    "LLM_KWARGS",
    "RESERVED_LLM_KWARGS",
    "TOOLS_PLACEHOLDER",
    "FlowOptions",
    "check_llm_kwargs",
    "nested",
    "parse_score",
    "substitute_tools",
]

TOOLS_PLACEHOLDER = "{tools}"

# The conversation, ``llm.complete``'s one positional parameter, named as its signature names it.
_CONVERSATION = next(name for name in inspect.signature(LLM.complete).parameters if name != "self")

# The arguments every strategy gives ``llm.complete`` itself, each with where its value comes from.
# In ``llm_kwargs`` one would clash with the strategy's own, or, for ``tools`` in a call that sends
# none (``completion``, a planner, a solver), reach the provider past the tools the agent declares.
# The rule holds for every strategy, a registered one too: the spec has its own ``system``, and the
# agent its own tools.
RESERVED_LLM_KWARGS: Mapping[str, str] = MappingProxyType(
    {
        _CONVERSATION: "the strategy builds the messages from the task",
        "system": "the strategy sends its own system prompt; set system, or the phase's own",
        "tools": "the strategy sends the tools it was given, and only to the calls that run them",
    }
)

LLM_KWARGS = Free(reserved=RESERVED_LLM_KWARGS)
"""The shape of a strategy's extra ``llm.complete`` arguments: any, but ``RESERVED_LLM_KWARGS``."""

_NUMBER_RE = re.compile(r"\d*\.?\d+")


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


def check_llm_kwargs(
    kwargs: Mapping[str, Any] | None, where: str = "llm_kwargs"
) -> dict[str, Any]:
    """Return a copy of ``kwargs`` (empty for ``None``) once it has the ``LLM_KWARGS`` shape.

    Raises:
        ShapeError: ``kwargs`` is not a mapping, or holds a reserved key; the message names it
            and says why. It is a ``ValueError``.
    """
    given = {} if kwargs is None else kwargs
    LLM_KWARGS.check(given, where)
    return dict(given)


def nested(options: FlowOptions) -> FlowOptions:
    """What a flow run inside a step of this one takes from its options: the trace capture.

    The inner run's steps are recorded as that step's children, so they record what this flow's
    steps record. The rest stays with this flow: its deadline and budget already cover the inner
    run (the two share the meter scope), and its policy governs its own steps.
    """
    return {"trace_capture": options["trace_capture"]} if "trace_capture" in options else {}


def parse_score(text: str) -> float:
    """Read an evaluator's score from its reply: the last number in [0, 1], or 0.5 without one.

    The last one, because a reply may first repeat the prompt's ``Score (0.0-1.0):`` or name a
    step, as in ``Step 2 looks strong: 0.9``.
    """
    scores = [score for score in map(float, _NUMBER_RE.findall(text)) if score <= 1.0]
    return scores[-1] if scores else 0.5


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
