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
from ai_arch_toolkit.toolkit.flow._flow import Flow, FlowResult

__all__ = ["build_flow", "extract_text", "initial_state"]


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
    """Pull a single answer string out of a finished flow run."""
    answer = state.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer
    response = state.get("response") or state.get("last_response")
    if isinstance(response, Response) and response.text:
        return response.text
    last_answer = state.get("last_answer")
    if isinstance(last_answer, str) and last_answer.strip():
        return last_answer
    final = result.final_result
    if final is not None and final.value is not None:
        if isinstance(final.value, Response):
            return final.value.text
        return str(final.value)
    return ""
