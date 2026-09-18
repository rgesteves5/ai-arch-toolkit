"""Compile a ReasoningSpec into a runnable Flow and read its output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ai_arch_toolkit.core._content import Content
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._response import Response
from ai_arch_toolkit.core._state import State
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents._builders import BuildContext, get_strategy
from ai_arch_toolkit.toolkit.agents._spec import ReasoningSpec
from ai_arch_toolkit.toolkit.agents.flows._keys import ANSWER, RESPONSE
from ai_arch_toolkit.toolkit.flow._flow import Flow, FlowResult

__all__ = ["build_flow", "extract_text", "initial_state", "read_answer"]


def build_flow(
    spec: ReasoningSpec,
    llm: LLM,
    tools: ToolGroup,
    *,
    deps: Mapping[str, Any] | None = None,
) -> Flow:
    """Compile a spec into a runnable Flow.

    The Flow is task-independent, so it can be built once and run on many tasks.
    """
    builder = get_strategy(spec.strategy)
    if spec.output_schema is not None and not builder.supports_output_schema:
        raise ValueError(f"strategy {spec.strategy!r} does not support output_schema")
    return builder.build(BuildContext(spec=spec, llm=llm, tools=tools, deps=deps or {}))


def initial_state(spec: ReasoningSpec, task: Content) -> dict[str, Any]:
    """Build the per-task initial operational state for a spec's strategy."""
    return get_strategy(spec.strategy).init_state(task)


def extract_text(state: State, result: FlowResult) -> str:
    """The run's answer: the text its flow left under ``"answer"``.

    A flow that leaves no answer (one built by hand, or a run that ended before its strategy
    answered) answers with its last step's value, as a flow nested in another one does.
    """
    return read_answer(state, result)[0]


def read_answer(state: State, result: FlowResult) -> tuple[str, Response | None]:
    """The run's answer and the ``Response`` behind it, both from one source (``extract_text``)."""
    answer = state.get(ANSWER)
    if answer is not None:
        response = state.get(RESPONSE)
        return str(answer), response if isinstance(response, Response) else None
    final = result.final_result
    value = final.value if final is not None else None
    if isinstance(value, Response):
        return value.text, value
    return ("" if value is None else str(value)), None
