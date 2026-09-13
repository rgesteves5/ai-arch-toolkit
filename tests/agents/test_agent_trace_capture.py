"""ReasoningSpec.trace_capture reaches the compiled flow, from code and from a manifest."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents import (
    Agent,
    AgentManifestError,
    ReasoningSpec,
    build_flow,
    load_agent_manifest,
)

_MODEL = "claude-sonnet-4-6"

_STRATEGIES = (
    "react",
    "completion",
    "plan_execute",
    "rewoo",
    "reflexion",
    "generate_review",
    "self_discovery",
    "llm_compiler",
    "tot",
    "lats",
)


class _Provider:
    async def complete(self, messages, *, system=None, tools=None, **kwargs) -> Response:
        return Response(text="answer", usage=Usage(input_tokens=10, output_tokens=5), model=_MODEL)


def _llm() -> LLM:
    llm = LLM(_MODEL, api_key="test")
    llm._provider = _Provider()  # type: ignore[assignment]
    return llm


@pytest.mark.parametrize("strategy", _STRATEGIES)
def test_trace_capture_reaches_the_compiled_flow(strategy: str) -> None:
    flow = build_flow(ReasoningSpec(strategy=strategy, trace_capture="none"), _llm(), ToolGroup())

    assert flow.trace_capture == "none"


def test_the_spec_defaults_to_keys() -> None:
    assert ReasoningSpec().trace_capture == "keys"
    assert ReasoningSpec.from_mapping({"trace_capture": "full"}).trace_capture == "full"


async def test_a_full_capture_agent_records_the_state_each_step_read() -> None:
    agent = Agent(ReasoningSpec(strategy="react", trace_capture="full"), _llm())

    result = await agent.run("hi")

    first = result.flow_result.trace.steps[0]
    assert first.input_state["operational"]["messages"]


def _manifest(tmp_path: Path, trace_capture: str) -> Path:
    path = tmp_path / "traced.agent.yaml"
    path.write_text(
        dedent(
            f"""
            version: 1
            id: traced.agent
            strategy:
              name: react
              trace_capture: {trace_capture}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_a_manifest_sets_trace_capture(tmp_path: Path) -> None:
    manifest = load_agent_manifest(_manifest(tmp_path, "full"), allowed_roots=(tmp_path,))

    assert manifest.reasoning_spec().trace_capture == "full"


def test_a_manifest_rejects_an_unknown_trace_capture(tmp_path: Path) -> None:
    with pytest.raises(AgentManifestError, match=r"strategy\.trace_capture"):
        load_agent_manifest(_manifest(tmp_path, "everything"), allowed_roots=(tmp_path,))
